"""BAM-primary streaming dataset for single-process inference."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import IterableDataset, get_worker_info

from unimeth.data.extract import SignalFeatureExtractor
from unimeth.data.pipeline import get_datasets
from unimeth.ioutils.reader import (
    BamSignalFeatureStream,
    BamStreamReader,
    SignalBatchLookup,
)
from unimeth.ioutils.reader.raw_signal import POD5_SUFFIXES, collect_signal_paths
from unimeth.model.datasets import Binning
from unimeth.model.streaming_pipeline import (
    BamFeatureBatchProcessor,
    ThreadedBatchPipeline,
    merge_feature_worker_stats,
)


@dataclass
class BamStreamingDatasetStats:
    """Counters produced after feature extraction and dataset patching."""

    records_without_patches: int = 0
    yielded_patches: int = 0


class BamStreamingDataset(IterableDataset):
    """Stream one BAM sequentially and fetch matching signals in small batches."""

    n_shards = 1

    def __init__(self, pod5_dir, bam_dir, args):
        signal_suffixes = getattr(args, "signal_suffixes", POD5_SUFFIXES)
        signal_label = getattr(args, "signal_label", "POD5")
        self.signal_paths = collect_signal_paths(
            pod5_dir,
            suffixes=signal_suffixes,
            label=signal_label,
        )
        self.bam_path = bam_dir
        self.args = args
        self.routing_plan = getattr(args, "signal_routing_plan", None)
        if self.routing_plan is None:
            raise ValueError(
                "BamStreamingDataset requires a pre-built signal_routing_plan"
            )

        self.stats = BamStreamingDatasetStats()
        self.bam_stats = None
        self.feature_stats = None
        self._active_pipeline = None

    @staticmethod
    def _is_marker(item) -> bool:
        return isinstance(item, dict) and item.get("__reads_complete__", False)

    def _yield_record_patches(self, patches, binning):
        if not patches:
            self.stats.records_without_patches += 1
            return
        for patch in patches:
            for item in binning.get_data(patch):
                if not self._is_marker(item):
                    self.stats.yielded_patches += 1
                yield item

    def close(self) -> None:
        """Stop and join an active threaded feature pipeline."""
        pipeline = getattr(self, "_active_pipeline", None)
        if pipeline is not None:
            pipeline.close()

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError(
                "BAM-primary streaming currently requires DataLoader num_workers=0"
            )

        self.stats = BamStreamingDatasetStats()
        resolved_bam_mode = getattr(
            self.args,
            "resolved_bam_mode",
            getattr(self.args, "bam_mode", "auto"),
        )
        bam_reader = BamStreamReader(
            self.bam_path,
            bam_mode=resolved_bam_mode,
            mapq=getattr(self.args, "mapq_thres", 1),
            identity=getattr(self.args, "identity_thres", 0.0),
            include_supplementary=getattr(
                self.args, "include_supplementary", False
            ),
            skip_unmapped=getattr(self.args, "skip_unmapped", True),
            chromosome_filter=getattr(self.args, "chr", "|"),
            threads=getattr(self.args, "bam_threads", 1),
        )
        binning = Binning(self.args)
        feature_workers = int(getattr(self.args, "num_workers", 2) or 0)
        if feature_workers < 0:
            raise ValueError("num_workers must be >= 0")

        if feature_workers == 0:
            extractor = SignalFeatureExtractor(
                self.args,
                # BamStreamReader already applies every aligned-mode filter before
                # signal lookup; do not repeat those checks after signal I/O.
                apply_alignment_filters=False,
            )
            router = self.routing_plan.open_router()
            with SignalBatchLookup(
                router,
                max_open_files=getattr(self.args, "signal_max_open_files", 4),
            ) as signal_lookup:
                feature_stream = BamSignalFeatureStream(
                    bam_reader,
                    signal_lookup,
                    extractor,
                    batch_size=getattr(
                        self.args, "signal_lookup_batch_size", 256
                    ),
                )
                for feature in feature_stream:
                    yield from self._yield_record_patches(
                        tuple(get_datasets(feature, self.args)),
                        binning,
                    )
                self.feature_stats = feature_stream.stats
        else:
            pipeline = ThreadedBatchPipeline(
                source=bam_reader,
                worker_factory=lambda worker_id: BamFeatureBatchProcessor(
                    worker_id,
                    self.routing_plan,
                    self.args,
                ),
                num_workers=feature_workers,
                batch_size=getattr(
                    self.args, "signal_lookup_batch_size", 256
                ),
                queue_size=max(2, feature_workers * 2),
            )
            self._active_pipeline = pipeline
            try:
                for bundle_batch in pipeline:
                    for bundle in bundle_batch:
                        yield from self._yield_record_patches(
                            bundle.patches,
                            binning,
                        )
                self.feature_stats = merge_feature_worker_stats(
                    pipeline.worker_summaries
                )
            finally:
                try:
                    pipeline.close()
                finally:
                    if self._active_pipeline is pipeline:
                        self._active_pipeline = None

        self.bam_stats = bam_reader.stats
        for item in binning.flush():
            if not self._is_marker(item):
                self.stats.yielded_patches += 1
            yield item
        yield {"__reads_complete__": True}
