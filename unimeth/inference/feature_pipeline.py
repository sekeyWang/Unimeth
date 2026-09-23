"""Feature-worker utilities for BAM-primary streaming inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from queue import Empty
from typing import Any, Iterable


_QUEUE_POLL_SECONDS = 0.1
_NORMALIZATION_CHECK_READS = 20
_NORMALIZATION_MEAN_LIMIT = 5.0
_NORMALIZATION_STD_MIN = 0.1
_NORMALIZATION_STD_MAX = 10.0
_NORMALIZATION_MAX_PATCHES_PER_READ = 32
_NORMALIZATION_MAX_SIGNAL_VALUES_PER_READ = 65_536


@dataclass(frozen=True)
class NormalizationCheckResult:
    """Worker-local summary of normalized model-input signals."""

    worker_id: int
    read_count: int
    median_mean: float
    median_std: float
    has_non_finite_stats: bool = False

    @property
    def is_suspicious(self) -> bool:
        return (
            self.has_non_finite_stats
            or not math.isfinite(self.median_mean)
            or not math.isfinite(self.median_std)
            or abs(self.median_mean) > _NORMALIZATION_MEAN_LIMIT
            or self.median_std < _NORMALIZATION_STD_MIN
            or self.median_std > _NORMALIZATION_STD_MAX
        )


class NormalizationCheckSampler:
    """Inspect a bounded sample of the first model-ready reads in one worker."""

    def __init__(
        self,
        worker_id: int,
        read_target: int = _NORMALIZATION_CHECK_READS,
        max_patches_per_read: int = _NORMALIZATION_MAX_PATCHES_PER_READ,
        max_signal_values_per_read: int = _NORMALIZATION_MAX_SIGNAL_VALUES_PER_READ,
    ):
        self.worker_id = worker_id
        self.read_target = read_target
        self.max_patches_per_read = max_patches_per_read
        self.max_signal_values_per_read = max_signal_values_per_read
        self._read_means: list[float] = []
        self._read_stds: list[float] = []
        self._has_non_finite_stats = False
        self._reported = False

    def _read_stats(self, patches: Iterable[dict]) -> tuple[float, float] | None:
        import numpy as np

        sampled_values = []
        sampled_count = 0
        for patch_index, patch in enumerate(patches):
            if patch_index >= self.max_patches_per_read:
                break
            remaining = self.max_signal_values_per_read - sampled_count
            if remaining <= 0:
                break
            try:
                values = np.asarray(
                    patch.get("signals", ()),
                    dtype=np.float64,
                ).reshape(-1)
            except Exception:
                continue
            if values.size == 0:
                continue
            selected = values[:remaining]
            sampled_values.append(selected)
            sampled_count += int(selected.size)

        if not sampled_values:
            return None
        values = (
            sampled_values[0]
            if len(sampled_values) == 1
            else np.concatenate(sampled_values)
        )
        with np.errstate(all="ignore"):
            return float(np.mean(values)), float(np.std(values))

    def observe(
        self,
        patches: Iterable[dict],
    ) -> NormalizationCheckResult | None:
        if self._reported:
            return None
        try:
            stats = self._read_stats(patches)
        except Exception:
            return None
        if stats is None:
            return None

        read_mean, read_std = stats
        self._read_means.append(read_mean)
        self._read_stds.append(read_std)
        if not math.isfinite(read_mean) or not math.isfinite(read_std):
            self._has_non_finite_stats = True
        if len(self._read_means) < self.read_target:
            return None

        import numpy as np

        self._reported = True
        return NormalizationCheckResult(
            worker_id=self.worker_id,
            read_count=len(self._read_means),
            median_mean=float(np.median(self._read_means)),
            median_std=float(np.median(self._read_stds)),
            has_non_finite_stats=self._has_non_finite_stats,
        )


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
            setattr(
                self,
                name,
                getattr(self, name) + int(getattr(other, name, 0)),
            )


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
            from unimeth.ioutils.reader.signal_lookup import SignalBatchLookup

            signal_lookup_factory = SignalBatchLookup
        if extractor_factory is None:
            from unimeth.data.extract import SignalFeatureExtractor

            extractor_factory = SignalFeatureExtractor
        if feature_stream_factory is None:
            from unimeth.ioutils.reader.bam_signal_stream import (
                BamSignalFeatureStream,
            )

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
        """Yield each record bundle without materializing the lookup batch."""
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


@dataclass(frozen=True)
class RecordStreamEnd:
    """One BAM reader end marker consumed by one feature worker."""


def iter_record_batches(record_queue, stop_event=None):
    """Yield reader batches until this consumer receives its end marker."""
    while stop_event is None or not stop_event.is_set():
        try:
            message = record_queue.get(timeout=_QUEUE_POLL_SECONDS)
        except Empty:
            continue
        if isinstance(message, RecordStreamEnd):
            return
        yield message
