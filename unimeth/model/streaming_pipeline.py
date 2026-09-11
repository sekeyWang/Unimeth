"""Producer and worker utilities for BAM-primary streaming inference."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, fields
from queue import Empty, Full
from threading import Event, Thread
from typing import Any, Callable, Iterable


_QUEUE_POLL_SECONDS = 0.1


@dataclass(frozen=True)
class RecordPatchBundle:
    """All inference patches generated from one accepted BAM record."""

    output_record_key: int | str
    signal_read_id: str
    patches: Iterable[dict]


@dataclass
class BamFeatureWorkerStats:
    """Feature-stage counters accumulated by a worker process."""

    lookup_batches: int = 0
    signal_missing_records: int = 0
    signal_source_hint_missing_records: int = 0
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

    def process(self, bam_batch: list[Any]) -> Iterable[RecordPatchBundle]:
        """Yield each record bundle without materializing the full lookup batch."""
        feature_stream = self.feature_stream_factory(
            bam_batch,
            self.signal_lookup,
            self.extractor,
            batch_size=len(bam_batch),
        )
        try:
            for feature in feature_stream:
                yield RecordPatchBundle(
                    output_record_key=feature["output_record_key"],
                    signal_read_id=feature["signal_read_id"],
                    patches=self.dataset_factory(feature, self.args),
                )
        finally:
            self.stats.add(feature_stream.stats)

    def close(self) -> None:
        self.signal_lookup.close()


class PipelineShutdownError(RuntimeError):
    """A streaming background component did not stop in time."""


@dataclass(frozen=True)
class RecordStreamEnd:
    """One producer end marker consumed by one DataLoader worker."""


@dataclass(frozen=True)
class RecordStreamFailure:
    """A producer exception forwarded through the shared record queue."""

    error_type: str
    message: str
    traceback_text: str


class RecordProducerError(RuntimeError):
    """Raised in a DataLoader worker when sequential BAM production fails."""

    def __init__(self, failure: RecordStreamFailure):
        super().__init__(
            f"BAM producer failed with {failure.error_type}: {failure.message}"
        )
        self.failure = failure


def iter_record_batches(record_queue, stop_event=None):
    """Yield producer batches until this consumer receives its end marker."""
    while stop_event is None or not stop_event.is_set():
        try:
            message = record_queue.get(timeout=_QUEUE_POLL_SECONDS)
        except Empty:
            continue
        if isinstance(message, RecordStreamEnd):
            return
        if isinstance(message, RecordStreamFailure):
            raise RecordProducerError(message)
        yield message


class RecordBatchProducer:
    """Serialize one sequential source into bounded batches for worker processes."""

    def __init__(
        self,
        source_factory: Callable[[], Iterable[Any]],
        serializer: Callable[[Any], Any],
        record_queue,
        num_consumers: int,
        batch_size: int,
        shutdown_timeout: float = 30.0,
    ):
        if num_consumers < 1:
            raise ValueError("num_consumers must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be > 0")

        self.source_factory = source_factory
        self.serializer = serializer
        self.record_queue = record_queue
        self.num_consumers = num_consumers
        self.batch_size = batch_size
        self.shutdown_timeout = shutdown_timeout
        self.stop_event = Event()
        self.source_stats = None
        self.failure: RecordStreamFailure | None = None
        self._source = None
        self._thread: Thread | None = None

    @property
    def progress_stats(self):
        """Return live source counters while production is running."""
        source = self._source
        if source is not None:
            return getattr(source, "stats", self.source_stats)
        return self.source_stats

    def _put(self, item: Any) -> bool:
        while not self.stop_event.is_set():
            try:
                self.record_queue.put(item, timeout=_QUEUE_POLL_SECONDS)
                return True
            except Full:
                continue
        return False

    def _put_terminal_messages(self, message: Any) -> None:
        for _ in range(self.num_consumers):
            if not self._put(message):
                return

    def _run(self) -> None:
        source = None
        source_iterator = None
        try:
            source = self.source_factory()
            self._source = source
            source_iterator = iter(source)
            batch = []
            for item in source_iterator:
                if self.stop_event.is_set():
                    return
                batch.append(self.serializer(item))
                if len(batch) >= self.batch_size:
                    if not self._put(tuple(batch)):
                        return
                    batch = []
            if batch and not self._put(tuple(batch)):
                return
            self._put_terminal_messages(RecordStreamEnd())
        except BaseException as error:
            failure = RecordStreamFailure(
                error_type=type(error).__name__,
                message=str(error),
                traceback_text=traceback.format_exc(),
            )
            self.failure = failure
            self._put_terminal_messages(failure)
        finally:
            if source is not None:
                self.source_stats = getattr(source, "stats", None)
            close_iterator = getattr(source_iterator, "close", None)
            if close_iterator is not None:
                close_iterator()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(
            target=self._run,
            name="unimeth-bam-producer",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self._thread is None:
            return
        self._thread.join(timeout=self.shutdown_timeout)
        if self._thread.is_alive():
            raise PipelineShutdownError(
                "BAM producer did not stop within "
                f"{self.shutdown_timeout:.1f}s"
            )
