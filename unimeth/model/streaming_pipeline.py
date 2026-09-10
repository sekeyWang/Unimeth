"""Bounded thread pipeline used by BAM-primary streaming inference."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, fields
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
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
        self.signal_lookup = None
        try:
            self.signal_lookup = signal_lookup_factory(
                router,
                max_open_files=getattr(args, "signal_max_open_files", 4),
            )
            self.extractor = extractor_factory(
                args,
                apply_alignment_filters=False,
            )
        except BaseException:
            if self.signal_lookup is not None:
                self.signal_lookup.close()
            else:
                close_router = getattr(router, "close", None)
                if close_router is not None:
                    close_router()
            raise

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


@dataclass(frozen=True)
class _PipelineFailure:
    stage: str
    worker_id: int | None
    error: BaseException
    traceback_text: str


class PipelineExecutionError(RuntimeError):
    """An exception raised by the BAM producer or a feature worker."""

    def __init__(self, failure: _PipelineFailure):
        location = failure.stage
        if failure.worker_id is not None:
            location += f" {failure.worker_id}"
        super().__init__(f"Streaming {location} failed: {failure.error}")
        self.stage = failure.stage
        self.worker_id = failure.worker_id
        self.original_exception = failure.error
        self.worker_traceback = failure.traceback_text


class PipelineShutdownError(RuntimeError):
    """The streaming pipeline did not stop all of its threads in time."""


_TASK_STOP = object()
_QUEUE_POLL_SECONDS = 0.1


class ThreadedBatchPipeline:
    """Feed source batches to worker-local processors through bounded queues."""

    def __init__(
        self,
        source: Iterable[Any],
        worker_factory: Callable[[int], BatchProcessor],
        num_workers: int,
        batch_size: int,
        queue_size: int,
        shutdown_timeout: float = 30.0,
    ):
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        if shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be > 0")

        self.source = source
        self.worker_factory = worker_factory
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shutdown_timeout = shutdown_timeout
        self.task_queue: Queue = Queue(maxsize=queue_size)
        self.result_queue: Queue = Queue(maxsize=queue_size)
        self.stop_event = Event()
        self.worker_summaries: list[WorkerSummary] = []
        self._threads: list[Thread] = []
        self._started = False
        self._failure: _PipelineFailure | None = None
        self._failure_lock = Lock()
        self._close_lock = Lock()
        self.alive_thread_names: tuple[str, ...] = ()

    def _record_failure(
        self,
        stage: str,
        error: BaseException,
        worker_id: int | None = None,
    ) -> None:
        failure = _PipelineFailure(
            stage=stage,
            worker_id=worker_id,
            error=error,
            traceback_text=traceback.format_exc(),
        )
        with self._failure_lock:
            if self._failure is None:
                self._failure = failure
        self.stop_event.set()

    def _put_while_running(self, queue: Queue, item: Any) -> bool:
        while not self.stop_event.is_set():
            try:
                queue.put(item, timeout=_QUEUE_POLL_SECONDS)
                return True
            except Full:
                continue
        return False

    def _get_task(self):
        while not self.stop_event.is_set():
            try:
                return self.task_queue.get(timeout=_QUEUE_POLL_SECONDS)
            except Empty:
                continue
        return _TASK_STOP

    def _produce(self) -> None:
        try:
            batch = []
            for item in self.source:
                if self.stop_event.is_set():
                    break
                batch.append(item)
                if len(batch) >= self.batch_size:
                    if not self._put_while_running(self.task_queue, batch):
                        return
                    batch = []
            if batch and not self._put_while_running(self.task_queue, batch):
                return
            for _ in range(self.num_workers):
                if not self._put_while_running(self.task_queue, _TASK_STOP):
                    return
        except BaseException as error:
            self._record_failure("BAM producer", error)

    def _work(self, worker_id: int) -> None:
        processor = None
        worker_failed = False
        try:
            processor = self.worker_factory(worker_id)
            while not self.stop_event.is_set():
                batch = self._get_task()
                if batch is _TASK_STOP:
                    break
                result = processor.process(batch)
                if not self._put_while_running(
                    self.result_queue,
                    _BatchResult(result),
                ):
                    return
        except BaseException as error:
            worker_failed = True
            self._record_failure("feature worker", error, worker_id)
        finally:
            if processor is not None:
                try:
                    processor.close()
                except BaseException as error:
                    worker_failed = True
                    self._record_failure(
                        "feature worker close",
                        error,
                        worker_id,
                    )
            if not worker_failed and not self.stop_event.is_set():
                self._put_while_running(
                    self.result_queue,
                    _WorkerDone(
                        WorkerSummary(
                            worker_id=worker_id,
                            stats=getattr(processor, "stats", None),
                        )
                    ),
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
        pipeline_failed = False
        try:
            while workers_done < self.num_workers:
                if self._failure is not None:
                    pipeline_failed = True
                    raise PipelineExecutionError(self._failure) from self._failure.error
                try:
                    message = self.result_queue.get(
                        timeout=_QUEUE_POLL_SECONDS
                    )
                except Empty:
                    continue
                if isinstance(message, _BatchResult):
                    yield message.value
                elif isinstance(message, _WorkerDone):
                    self.worker_summaries.append(message.summary)
                    workers_done += 1
        finally:
            self.close(raise_on_timeout=not pipeline_failed)

    def close(self, raise_on_timeout: bool = True) -> None:
        self.stop_event.set()
        with self._close_lock:
            deadline = time.monotonic() + self.shutdown_timeout
            for thread in self._threads:
                remaining = max(0.0, deadline - time.monotonic())
                thread.join(timeout=remaining)
            self.alive_thread_names = tuple(
                thread.name for thread in self._threads if thread.is_alive()
            )
        if self.alive_thread_names and raise_on_timeout:
            names = ", ".join(self.alive_thread_names)
            raise PipelineShutdownError(
                f"Streaming pipeline threads did not stop within "
                f"{self.shutdown_timeout:.1f}s: {names}"
            )
