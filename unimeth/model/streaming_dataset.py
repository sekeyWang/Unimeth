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

    @staticmethod
    def _is_marker(item) -> bool:
        return isinstance(item, dict) and item.get("__reads_complete__", False)

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
            no_supplementary=getattr(self.args, "no_supplementary", False),
            skip_unmapped=getattr(self.args, "skip_unmapped", True),
            chromosome_filter=getattr(self.args, "chr", "|"),
            threads=getattr(self.args, "bam_threads", 1),
        )
        extractor = SignalFeatureExtractor(
            self.args,
            # BamStreamReader already applies every aligned-mode filter before
            # signal lookup; do not repeat those checks after signal I/O.
            apply_alignment_filters=False,
        )
        binning = Binning(self.args)

        router = self.routing_plan.open_router()
        with SignalBatchLookup(
            router,
            max_open_files=getattr(self.args, "signal_max_open_files", 4),
        ) as signal_lookup:
            feature_stream = BamSignalFeatureStream(
                bam_reader,
                signal_lookup,
                extractor,
                batch_size=getattr(self.args, "signal_lookup_batch_size", 256),
            )
            for feature in feature_stream:
                record_has_patches = False
                for patch in get_datasets(feature, self.args):
                    record_has_patches = True
                    for item in binning.get_data(patch):
                        if not self._is_marker(item):
                            self.stats.yielded_patches += 1
                        yield item
                if not record_has_patches:
                    self.stats.records_without_patches += 1

            self.feature_stats = feature_stream.stats

        self.bam_stats = bam_reader.stats
        for item in binning.flush():
            if not self._is_marker(item):
                self.stats.yielded_patches += 1
            yield item
        yield {"__reads_complete__": True}
