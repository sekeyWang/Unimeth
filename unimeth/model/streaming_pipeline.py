"""Bounded thread pipeline used by BAM-primary streaming inference."""

from __future__ import annotations

from dataclasses import dataclass, fields
from queue import Queue
from threading import Event, Thread
from typing import Any, Callable, Iterable, Protocol


class BatchProcessor(Protocol):
    """Worker-local processor created and closed inside its worker thread."""

    def process(self, batch: list[Any]) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class WorkerSummary:
    """Final state returned by one worker after its processor is closed."""

    worker_id: int
    stats: Any


@dataclass(frozen=True)
class RecordPatchBundle:
    """All inference patches generated from one accepted BAM record."""

    output_record_key: int | str
    signal_read_id: str
    patches: tuple[dict, ...]


@dataclass
class BamFeatureWorkerStats:
    """Feature-stage counters accumulated by one or more worker threads."""

    lookup_batches: int = 0
    signal_missing_records: int = 0
    hard_clipped_reconciled_records: int = 0
    signal_sequence_mismatch_records: int = 0
    feature_empty_records: int = 0
    feature_records: int = 0

    def add(self, other: Any) -> None:
        for stat_field in fields(self):
            name = stat_field.name
            setattr(self, name, getattr(self, name) + int(getattr(other, name, 0)))


class BamFeatureBatchProcessor:
    """Worker-local signal lookup and feature/patch extraction state."""

    def __init__(
        self,
        worker_id: int,
        routing_plan,
        args,
        signal_lookup_factory=None,
        extractor_factory=None,
        feature_stream_factory=None,
        dataset_factory=None,
    ):
        if signal_lookup_factory is None:
            from unimeth.ioutils.reader import SignalBatchLookup

            signal_lookup_factory = SignalBatchLookup
        if extractor_factory is None:
            from unimeth.data.extract import SignalFeatureExtractor

            extractor_factory = SignalFeatureExtractor
        if feature_stream_factory is None:
            from unimeth.ioutils.reader import BamSignalFeatureStream

            feature_stream_factory = BamSignalFeatureStream
        if dataset_factory is None:
            from unimeth.data.pipeline import get_datasets

            dataset_factory = get_datasets

        self.worker_id = worker_id
        self.args = args
        self.feature_stream_factory = feature_stream_factory
        self.dataset_factory = dataset_factory
        self.stats = BamFeatureWorkerStats()
        router = routing_plan.open_router()
        self.signal_lookup = signal_lookup_factory(
            router,
            max_open_files=getattr(args, "signal_max_open_files", 4),
        )
        self.extractor = extractor_factory(
            args,
            apply_alignment_filters=False,
        )

    def process(self, bam_batch: list[Any]) -> tuple[RecordPatchBundle, ...]:
        feature_stream = self.feature_stream_factory(
            bam_batch,
            self.signal_lookup,
            self.extractor,
            batch_size=len(bam_batch),
        )
        bundles = []
        for feature in feature_stream:
            bundles.append(
                RecordPatchBundle(
                    output_record_key=feature["output_record_key"],
                    signal_read_id=feature["signal_read_id"],
                    patches=tuple(self.dataset_factory(feature, self.args)),
                )
            )
        self.stats.add(feature_stream.stats)
        return tuple(bundles)

    def close(self) -> None:
        self.signal_lookup.close()


def merge_feature_worker_stats(
    summaries: Iterable[WorkerSummary],
) -> BamFeatureWorkerStats:
    """Combine feature counters returned by completed worker threads."""
    combined = BamFeatureWorkerStats()
    for summary in summaries:
        combined.add(summary.stats)
    return combined


@dataclass(frozen=True)
class _BatchResult:
    value: Any


@dataclass(frozen=True)
class _WorkerDone:
    summary: WorkerSummary


_TASK_STOP = object()


class ThreadedBatchPipeline:
    """Feed source batches to worker-local processors through bounded queues."""

    def __init__(
        self,
        source: Iterable[Any],
        worker_factory: Callable[[int], BatchProcessor],
        num_workers: int,
        batch_size: int,
        queue_size: int,
    ):
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")

        self.source = source
        self.worker_factory = worker_factory
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.task_queue: Queue = Queue(maxsize=queue_size)
        self.result_queue: Queue = Queue(maxsize=queue_size)
        self.stop_event = Event()
        self.worker_summaries: list[WorkerSummary] = []
        self._threads: list[Thread] = []
        self._started = False

    def _produce(self) -> None:
        batch = []
        for item in self.source:
            if self.stop_event.is_set():
                break
            batch.append(item)
            if len(batch) >= self.batch_size:
                self.task_queue.put(batch)
                batch = []
        if batch and not self.stop_event.is_set():
            self.task_queue.put(batch)
        for _ in range(self.num_workers):
            self.task_queue.put(_TASK_STOP)

    def _work(self, worker_id: int) -> None:
        processor = self.worker_factory(worker_id)
        try:
            while not self.stop_event.is_set():
                batch = self.task_queue.get()
                if batch is _TASK_STOP:
                    break
                self.result_queue.put(_BatchResult(processor.process(batch)))
        finally:
            processor.close()
            self.result_queue.put(
                _WorkerDone(
                    WorkerSummary(
                        worker_id=worker_id,
                        stats=getattr(processor, "stats", None),
                    )
                )
            )

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        producer = Thread(
            target=self._produce,
            name="unimeth-bam-producer",
            daemon=True,
        )
        self._threads.append(producer)
        for worker_id in range(self.num_workers):
            self._threads.append(
                Thread(
                    target=self._work,
                    args=(worker_id,),
                    name=f"unimeth-feature-{worker_id}",
                    daemon=True,
                )
            )
        for thread in self._threads:
            thread.start()

    def __iter__(self):
        self.start()
        workers_done = 0
        try:
            while workers_done < self.num_workers:
                message = self.result_queue.get()
                if isinstance(message, _BatchResult):
                    yield message.value
                elif isinstance(message, _WorkerDone):
                    self.worker_summaries.append(message.summary)
                    workers_done += 1
        finally:
            self.close()

    def close(self) -> None:
        self.stop_event.set()
        for thread in self._threads:
            thread.join()
