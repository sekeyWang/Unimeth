"""BAM-primary streaming dataset with process-based CPU preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import IterableDataset, get_worker_info

from unimeth.data.extract import SignalFeatureExtractor
from unimeth.data.pipeline import get_datasets
from unimeth.ioutils.reader import (
    BamSignalFeatureStream,
    BamStreamReader,
    SerializedBamStreamItem,
    SignalBatchLookup,
)
from unimeth.ioutils.reader.raw_signal import POD5_SUFFIXES, collect_signal_paths
from unimeth.model.datasets import Binning
from unimeth.model.streaming_pipeline import (
    BamFeatureBatchProcessor,
    RecordBatchProducer,
    iter_record_batches,
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
        import pysam

        with pysam.AlignmentFile(
            self.bam_path,
            "rb",
            check_sq=False,
        ) as bam_file:
            self._bam_header_dict = bam_file.header.to_dict()
        self.routing_plan = getattr(args, "signal_routing_plan", None)
        if self.routing_plan is None:
            raise ValueError(
                "BamStreamingDataset requires a pre-built signal_routing_plan"
            )

        self.stats = BamStreamingDatasetStats()
        self.bam_stats = None
        self.feature_stats = None
        self._record_queue = None
        self._worker_stop_event = None
        self._parallel_workers = 0
        self._producer = None

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

    def _make_bam_reader(self):
        resolved_bam_mode = getattr(
            self.args,
            "resolved_bam_mode",
            getattr(self.args, "bam_mode", "auto"),
        )
        return BamStreamReader(
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

    def configure_parallel_workers(self, num_workers: int) -> None:
        """Create the shared bounded queue before DataLoader workers start."""
        if num_workers < 1:
            return
        if self._record_queue is not None:
            if self._parallel_workers != num_workers:
                raise RuntimeError("streaming worker count is already configured")
            return

        import multiprocessing

        self._parallel_workers = num_workers
        context = multiprocessing.get_context()
        self._record_queue = context.Queue(
            maxsize=max(2, num_workers * 2)
        )
        self._worker_stop_event = context.Event()

    def start_producer(self) -> None:
        """Start the sole sequential BAM reader after workers have spawned."""
        if self._parallel_workers == 0:
            return
        if self._record_queue is None:
            raise RuntimeError("parallel streaming workers are not configured")
        if self._producer is not None:
            self._producer.start()
            return

        self._producer = RecordBatchProducer(
            source_factory=self._make_bam_reader,
            serializer=SerializedBamStreamItem.from_item,
            record_queue=self._record_queue,
            num_consumers=self._parallel_workers,
            batch_size=getattr(self.args, "signal_lookup_batch_size", 8),
        )
        self._producer.start()

    def close(self) -> None:
        """Stop and join the main-process sequential BAM producer."""
        stop_event = getattr(self, "_worker_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        producer = getattr(self, "_producer", None)
        if producer is not None:
            producer.close()
            self.bam_stats = producer.source_stats

    def _iter_direct(self, bam_reader, binning):
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
                    self.args, "signal_lookup_batch_size", 8
                ),
            )
            for feature in feature_stream:
                yield from self._yield_record_patches(
                    tuple(get_datasets(feature, self.args)),
                    binning,
                )
            self.feature_stats = feature_stream.stats
        self.bam_stats = bam_reader.stats

    def _iter_parallel(self, worker_id: int, binning):
        if self._record_queue is None:
            raise RuntimeError("parallel streaming record queue is not configured")

        import pysam

        bam_header = pysam.AlignmentHeader.from_dict(self._bam_header_dict)
        processor = BamFeatureBatchProcessor(
            worker_id,
            self.routing_plan,
            self.args,
        )
        try:
            for serialized_batch in iter_record_batches(
                self._record_queue,
                self._worker_stop_event,
            ):
                bam_batch = tuple(
                    item.restore(bam_header)
                    for item in serialized_batch
                )
                for bundle in processor.process(bam_batch):
                    yield from self._yield_record_patches(
                        bundle.patches,
                        binning,
                    )
        finally:
            processor.close()

    def __iter__(self):
        self.stats = BamStreamingDatasetStats()
        worker_info = get_worker_info()
        binning = Binning(self.args)
        # Completeness is tracked by output_record_key and patch indices in the
        # BAM writer, so process-local bins do not need global read-ID flush sets.
        binning.reads_per_flush = None

        if worker_info is None:
            yield from self._iter_direct(self._make_bam_reader(), binning)
        else:
            yield from self._iter_parallel(worker_info.id, binning)

        for item in binning.flush():
            if not self._is_marker(item):
                self.stats.yielded_patches += 1
            yield item
        yield {"__reads_complete__": True}
