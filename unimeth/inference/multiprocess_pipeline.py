"""Single-host multiprocessing pipeline for BAM-primary inference.

The main process only coordinates child processes and renders progress.  Data
flows through bounded queues owned by one BAM reader, a CPU feature pool, one
global binning/batching process, one model process per visible GPU, and one
output writer.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from unimeth.inference.progress import RecordCompletionTracker, format_compact_count


logger = logging.getLogger(__name__)
_QUEUE_POLL_SECONDS = 0.1
_PROCESS_JOIN_SECONDS = 30.0


@dataclass(frozen=True)
class PipelineLayout:
    """Resolved process counts for one inference invocation."""

    gpu_count: int
    model_workers: int
    feature_workers: int
    bam_read_threads: int
    bam_write_threads: int
    bam_finalize_threads: int
    total_processes: int
    is_auto: bool


def resolve_pipeline_layout(
    configured_feature_workers: int | None,
    gpu_count: int,
) -> PipelineLayout:
    """Resolve global worker counts without creating processes or CUDA state."""
    if gpu_count < 0:
        raise ValueError("gpu_count must be >= 0")
    if configured_feature_workers is not None and configured_feature_workers < 1:
        raise ValueError("num_workers must be >= 1")

    model_workers = max(1, gpu_count)
    is_auto = configured_feature_workers is None
    feature_workers = (
        2 * model_workers
        if is_auto
        else configured_feature_workers
    )
    # BAM input and output run concurrently, so split the configured CPU
    # budget between them.  A value of one is pysam's single-threaded mode.
    bam_read_threads = max(1, feature_workers // 2)
    bam_write_threads = max(1, feature_workers - bam_read_threads)
    # Main coordinator + reader + batcher + writer + both worker pools.
    total_processes = 4 + model_workers + feature_workers
    return PipelineLayout(
        gpu_count=gpu_count,
        model_workers=model_workers,
        feature_workers=feature_workers,
        bam_read_threads=bam_read_threads,
        bam_write_threads=bam_write_threads,
        bam_finalize_threads=feature_workers,
        total_processes=total_processes,
        is_auto=is_auto,
    )


def should_sort_and_index_bam(args) -> bool:
    """Return whether the resolved streaming BAM mode requires coordinates."""
    bam_mode = getattr(args, "resolved_bam_mode", None)
    if bam_mode is None:
        bam_mode = getattr(args, "bam_mode", None)
    if bam_mode not in ("aligned", "unaligned"):
        raise RuntimeError("BAM mode must be resolved before starting inference")
    return bam_mode == "aligned"


@dataclass(frozen=True)
class PipelineFailure:
    """An exception raised by one pipeline component."""

    stage: str
    worker_id: int | None
    error_type: str
    message: str
    traceback_text: str


@dataclass(frozen=True)
class ModelBatch:
    """One globally binned and collated CPU batch."""

    batch_id: int
    batch: dict[str, Any]


@dataclass(frozen=True)
class PredictionBatch:
    """CPU prediction values and metadata sent to the sole writer."""

    batch_id: int
    preds: Any
    methy: Any
    read_ids: list[str]
    chrs: list[str] | None
    strands: list[str] | None
    ref_pos: list[list[int]] | None
    read_pos: list[list[int]]
    labels: list[list[int]] | None
    patch_pos: list[list[int]]
    patch_idx: list[int]
    total_patches: list[int]
    output_record_keys: list[int | str]

    @property
    def site_count(self) -> int:
        return int(len(self.preds))


@dataclass(frozen=True)
class FeatureRecordBundle:
    """All inference patches produced for one accepted BAM record."""

    output_record_key: int | str
    patches: tuple[dict, ...]


@dataclass(frozen=True)
class FeatureStreamEnd:
    worker_id: int


@dataclass(frozen=True)
class ModelStreamEnd:
    pass


@dataclass(frozen=True)
class PredictionStreamEnd:
    worker_id: int


@dataclass(frozen=True)
class ReaderProgress:
    stats: dict[str, int]


@dataclass(frozen=True)
class ReaderDone:
    stats: dict[str, int]


@dataclass(frozen=True)
class FeatureWorkerDone:
    worker_id: int
    stats: dict[str, int]
    records_without_patches: int
    yielded_patches: int


@dataclass(frozen=True)
class BatcherDone:
    batches: int
    patches: int


@dataclass(frozen=True)
class GPUWorkerReady:
    worker_id: int
    device: str


@dataclass(frozen=True)
class GPUWorkerDone:
    worker_id: int
    batches: int
    sites: int
    model_seconds: float


@dataclass(frozen=True)
class WriterProgress:
    records: int
    batches: int
    sites: int


@dataclass(frozen=True)
class WriterFinalizing:
    records: int
    batches: int
    sites: int
    incomplete_records: int
    bam_path: str | None
    tsv_path: str | None


@dataclass(frozen=True)
class WriterDone:
    records: int
    batches: int
    sites: int
    incomplete_records: int
    bam_path: str | None
    tsv_path: str | None


class PipelineCancelled(RuntimeError):
    """Internal signal used to unwind a child after another child fails."""


class MultiprocessInferenceError(RuntimeError):
    """Raised in the main process when a pipeline component fails."""

    def __init__(self, failure: PipelineFailure):
        worker = "" if failure.worker_id is None else f" {failure.worker_id}"
        super().__init__(
            f"{failure.stage}{worker} failed with {failure.error_type}: "
            f"{failure.message}\n{failure.traceback_text}"
        )
        self.failure = failure


_MODEL_TENSOR_NAMES = (
    "signals",
    "encoder_mask",
    "decoder_input_ids",
    "signal_pos",
)
_BAM_UNUSED_BATCH_FIELDS = (
    "chr",
    "strand",
    "ref_pos",
    "labels",
)


def prepare_collated_batch_for_transfer(
    batch: dict[str, Any],
    output_format: str,
) -> dict[str, Any]:
    """Share model tensors and discard metadata unused by the output format."""
    if output_format not in ("bam", "tsv", "both"):
        raise ValueError(f"unsupported output format: {output_format}")

    transferred = dict(batch)
    transferred.pop("signal_read_id", None)
    if output_format == "bam":
        for name in _BAM_UNUSED_BATCH_FIELDS:
            transferred.pop(name, None)
    for name in _MODEL_TENSOR_NAMES:
        transferred[name].share_memory_()
    return transferred


def _put_bounded(target_queue, item, stop_event) -> None:
    while not stop_event.is_set():
        try:
            target_queue.put(item, timeout=_QUEUE_POLL_SECONDS)
            return
        except queue.Full:
            continue
    raise PipelineCancelled("pipeline cancellation requested")


def _get_bounded(source_queue, stop_event):
    while not stop_event.is_set():
        try:
            return source_queue.get(timeout=_QUEUE_POLL_SECONDS)
        except queue.Empty:
            continue
    raise PipelineCancelled("pipeline cancellation requested")


def _report_failure(
    status_queue,
    stop_event,
    stage: str,
    worker_id: int | None,
    error: BaseException,
) -> None:
    failure = PipelineFailure(
        stage=stage,
        worker_id=worker_id,
        error_type=type(error).__name__,
        message=str(error),
        traceback_text=traceback.format_exc(),
    )
    try:
        status_queue.put(failure)
    finally:
        stop_event.set()


def _guarded_worker(
    stage: str,
    worker_id: int | None,
    status_queue,
    stop_event,
    target,
    *args,
) -> None:
    try:
        target(*args)
    except PipelineCancelled:
        return
    except BaseException as error:
        _report_failure(status_queue, stop_event, stage, worker_id, error)


def _reader_worker_impl(
    args,
    record_queue,
    status_queue,
    stop_event,
    consumers: int,
    bam_threads: int,
) -> None:
    from unimeth.ioutils.reader.bam_stream import (
        BamStreamReader,
        SerializedBamStreamItem,
    )
    from unimeth.inference.feature_pipeline import RecordStreamEnd

    reader = BamStreamReader(
        args.bam_dir,
        bam_mode=getattr(args, "resolved_bam_mode", getattr(args, "bam_mode", "auto")),
        mapq=getattr(args, "mapq_thres", 1),
        identity=getattr(args, "identity_thres", 0.0),
        include_supplementary=getattr(args, "include_supplementary", False),
        skip_unmapped=getattr(args, "skip_unmapped", True),
        chromosome_filter=getattr(args, "chr", "|"),
        threads=bam_threads,
        limit=getattr(args, "limit", None),
    )
    lookup_size = max(1, int(getattr(args, "signal_lookup_batch_size", 8)))
    batch = []
    last_progress = time.monotonic()
    for item in reader:
        batch.append(SerializedBamStreamItem.from_item(item))
        if len(batch) >= lookup_size:
            _put_bounded(record_queue, tuple(batch), stop_event)
            now = time.monotonic()
            if now - last_progress >= 0.5:
                status_queue.put(ReaderProgress(asdict(reader.stats)))
                last_progress = now
            batch = []
    if batch:
        _put_bounded(record_queue, tuple(batch), stop_event)
        status_queue.put(ReaderProgress(asdict(reader.stats)))
    for _ in range(consumers):
        _put_bounded(record_queue, RecordStreamEnd(), stop_event)
    status_queue.put(ReaderDone(asdict(reader.stats)))


def _feature_worker_impl(
    worker_id: int,
    args,
    record_queue,
    feature_queue,
    status_queue,
    stop_event,
) -> None:
    import pysam

    from unimeth.inference.feature_pipeline import (
        BamFeatureBatchProcessor,
        iter_record_batches,
    )

    with pysam.AlignmentFile(args.bam_dir, "rb", check_sq=False) as bam_file:
        bam_header = bam_file.header
    processor = BamFeatureBatchProcessor(
        worker_id,
        args.signal_routing_plan,
        args,
    )
    records_without_patches = 0
    yielded_patches = 0
    try:
        for serialized_batch in iter_record_batches(record_queue, stop_event):
            bam_batch = tuple(item.restore(bam_header) for item in serialized_batch)
            for bundle in processor.process(bam_batch):
                patches = tuple(bundle.patches)
                if patches:
                    yielded_patches += len(patches)
                else:
                    records_without_patches += 1
                _put_bounded(
                    feature_queue,
                    FeatureRecordBundle(bundle.output_record_key, patches),
                    stop_event,
                )
    finally:
        processor.close()
    _put_bounded(feature_queue, FeatureStreamEnd(worker_id), stop_event)
    status_queue.put(
        FeatureWorkerDone(
            worker_id=worker_id,
            stats=asdict(processor.stats),
            records_without_patches=records_without_patches,
            yielded_patches=yielded_patches,
        )
    )


def _batcher_worker_impl(
    args,
    output_format: str,
    feature_queue,
    model_queue,
    status_queue,
    stop_event,
    feature_workers: int,
    model_workers: int,
    consumer_release_event,
) -> None:
    from unimeth.config import get_total_stride
    from unimeth.inference.batching import InferenceBinning, collate_inference

    binning = InferenceBinning(args)
    pending = []
    batch_id = 0
    patch_count = 0
    ended = 0
    batch_size = max(1, int(args.batch_size))
    total_stride = get_total_stride(getattr(args, "model_type", "default"))

    def emit_pending() -> None:
        nonlocal pending, batch_id
        collated = collate_inference(
            pending,
            total_stride=total_stride,
        )
        transferred = prepare_collated_batch_for_transfer(
            collated,
            output_format,
        )
        _put_bounded(
            model_queue,
            ModelBatch(batch_id, transferred),
            stop_event,
        )
        batch_id += 1
        pending = []

    def emit(sample: dict) -> None:
        nonlocal pending, patch_count
        pending.append(sample)
        patch_count += 1
        if len(pending) >= batch_size:
            emit_pending()

    while ended < feature_workers:
        message = _get_bounded(feature_queue, stop_event)
        if isinstance(message, FeatureStreamEnd):
            ended += 1
            continue
        if not isinstance(message, FeatureRecordBundle):
            raise TypeError(f"unexpected feature message: {type(message).__name__}")
        for patch in message.patches:
            for sample in binning.get_data(patch):
                if not (isinstance(sample, dict) and sample.get("__reads_complete__")):
                    emit(sample)

    for sample in binning.flush():
        emit(sample)
    if pending:
        emit_pending()
    for _ in range(model_workers):
        _put_bounded(model_queue, ModelStreamEnd(), stop_event)
    status_queue.put(BatcherDone(batches=batch_id, patches=patch_count))
    # PyTorch shared-memory senders must outlive every process using their tensors.
    while not consumer_release_event.wait(_QUEUE_POLL_SECONDS):
        if stop_event.is_set():
            return


def _extract_predictions(
    decoder_input_ids,
    logits,
    patch_pos,
    unmethylated_idx: int,
    methylated_idx: int,
):
    """Extract site probabilities without depending on an engine instance."""
    import torch

    device = logits.device
    site_counts = [len(positions) for positions in patch_pos]
    total_sites = sum(site_counts)
    if total_sites == 0:
        return (
            torch.tensor([], device=device),
            torch.tensor([], device=device, dtype=torch.long),
        )

    batch_indices = torch.arange(
        logits.shape[0],
        device=device,
    ).repeat_interleave(torch.as_tensor(site_counts, device=device))
    pos_indices = torch.cat(
        [
            torch.as_tensor(positions, device=device)
            for positions in patch_pos
            if len(positions) > 0
        ]
    )
    selected_logits = logits[batch_indices, pos_indices]
    relevant_logits = selected_logits[:, [unmethylated_idx, methylated_idx]]
    preds = torch.softmax(relevant_logits, dim=-1)[:, 1]
    methy = decoder_input_ids[batch_indices, pos_indices]
    return preds, methy


def _gpu_worker_impl(
    worker_id: int,
    device_index: int | None,
    args,
    model_queue,
    prediction_queue,
    status_queue,
    stop_event,
) -> None:
    import torch

    from unimeth.config import tokenizer
    from unimeth.model.loader import load_model

    if device_index is None:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)

    model = load_model(
        config=getattr(args, "model_type", "default"),
        model_path=getattr(args, "model_dir", None),
        mode="inference",
        device=device,
    )
    model.eval()
    status_queue.put(GPUWorkerReady(worker_id, str(device)))

    batches = 0
    sites = 0
    model_seconds = 0.0
    profile_timing = os.environ.get("UNIMETH_LOG_LEVEL", "INFO").upper() == "DEBUG"

    while True:
        message = _get_bounded(model_queue, stop_event)
        if isinstance(message, ModelStreamEnd):
            break
        if not isinstance(message, ModelBatch):
            raise TypeError(f"unexpected model message: {type(message).__name__}")

        batch = message.batch
        if device.type == "cuda":
            for name in _MODEL_TENSOR_NAMES:
                batch[name] = batch[name].pin_memory().to(device, non_blocking=True)
            autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            autocast = contextlib.nullcontext()

        started = time.perf_counter()
        with torch.inference_mode(), autocast:
            logits = model(
                signals=batch["signals"],
                encoder_mask=batch["encoder_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                signal_pos=batch["signal_pos"],
            )
            preds, methy = _extract_predictions(
                batch["decoder_input_ids"],
                logits,
                batch["patch_pos"],
                tokenizer["-"],
                tokenizer["+"],
            )
        if profile_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        model_seconds += time.perf_counter() - started

        preds_np = preds.float().cpu().numpy()
        methy_np = methy.long().cpu().numpy()
        result = PredictionBatch(
            batch_id=message.batch_id,
            preds=preds_np,
            methy=methy_np,
            read_ids=batch["read_id"],
            chrs=batch.get("chr"),
            strands=batch.get("strand"),
            ref_pos=batch.get("ref_pos"),
            read_pos=batch["read_pos"],
            labels=batch.get("labels"),
            patch_pos=batch["patch_pos"],
            patch_idx=batch["patch_idx"],
            total_patches=batch["total_patches"],
            output_record_keys=batch["output_record_key"],
        )
        _put_bounded(prediction_queue, result, stop_event)
        batches += 1
        sites += result.site_count

    _put_bounded(prediction_queue, PredictionStreamEnd(worker_id), stop_event)
    status_queue.put(
        GPUWorkerDone(
            worker_id=worker_id,
            batches=batches,
            sites=sites,
            model_seconds=model_seconds,
        )
    )


def _writer_worker_impl(
    args,
    output_format: str,
    prediction_queue,
    status_queue,
    stop_event,
    model_workers: int,
    bam_write_threads: int,
    bam_finalize_threads: int,
) -> None:
    import torch

    from unimeth.ioutils.reader.bam_stream import BamOffsetReader
    from unimeth.ioutils.writer.bam_aggregation import AggregationBAMWriter
    from unimeth.ioutils.writer.bam_finalize import (
        finalize_single_bam,
        normalize_bam_path,
    )
    from unimeth.ioutils.writer.tsv import DirectTSVWriter

    bam_final = bam_work = None
    tsv_final = tsv_work = None
    if output_format in ("bam", "both"):
        bam_final = str(normalize_bam_path(args.bam_out_dir or args.out_dir))
        bam_work = bam_final
    if output_format in ("tsv", "both"):
        tsv_final = str(args.tsv_out_dir or args.out_dir)
        if getattr(args, "gzip", False) and not tsv_final.endswith(".gz"):
            tsv_final = f"{tsv_final}.gz"
        tsv_work = tsv_final

    stack = contextlib.ExitStack()
    bam_writer = tsv_writer = None
    failed = True
    try:
        if bam_work is not None:
            Path(bam_work).parent.mkdir(parents=True, exist_ok=True)
            bam_writer = stack.enter_context(
                AggregationBAMWriter(
                    output_path=bam_work,
                    template_bam_path=args.bam_dir,
                    record_reader=BamOffsetReader(
                        args.bam_dir,
                        threads=1,
                    ),
                    keep_mv=getattr(args, "keep_mv", False),
                    threads=bam_write_threads,
                )
            )
        if tsv_work is not None:
            Path(tsv_work).parent.mkdir(parents=True, exist_ok=True)
            tsv_writer = stack.enter_context(
                DirectTSVWriter(
                    tsv_work,
                    gzip_output=tsv_final.endswith(".gz"),
                )
            )

        ended = 0
        batches = 0
        sites = 0
        record_tracker = RecordCompletionTracker() if bam_writer is None else None
        records = 0
        while ended < model_workers:
            message = _get_bounded(prediction_queue, stop_event)
            if isinstance(message, PredictionStreamEnd):
                ended += 1
                continue
            if not isinstance(message, PredictionBatch):
                raise TypeError(
                    f"unexpected prediction message: {type(message).__name__}"
                )

            preds = torch.from_numpy(message.preds)
            methy = torch.from_numpy(message.methy)
            if tsv_writer is not None:
                if any(
                    value is None
                    for value in (
                        message.chrs,
                        message.strands,
                        message.ref_pos,
                        message.labels,
                    )
                ):
                    raise RuntimeError(
                        "TSV prediction metadata is missing from a model batch"
                    )
                tsv_writer.write_batch(
                    preds=preds,
                    methy=methy,
                    read_ids=message.read_ids,
                    chrs=message.chrs,
                    strands=message.strands,
                    ref_pos=message.ref_pos,
                    read_pos=message.read_pos,
                    labels=message.labels,
                    patch_pos=message.patch_pos,
                )
            if bam_writer is not None:
                bam_writer.write_batch(
                    preds=preds,
                    methy=methy,
                    read_ids=message.read_ids,
                    chrs=message.chrs,
                    strands=message.strands,
                    ref_pos=message.ref_pos,
                    read_pos=message.read_pos,
                    patch_pos=message.patch_pos,
                    patch_idx=message.patch_idx,
                    total_patches=message.total_patches,
                    output_record_keys=message.output_record_keys,
                )
                records = int(bam_writer.stats["records_written"])
            else:
                records += record_tracker.update_batch(
                    message.output_record_keys,
                    message.patch_idx,
                    message.total_patches,
                )
            batches += 1
            sites += message.site_count
            status_queue.put(WriterProgress(records, batches, sites))

        if stop_event.is_set():
            raise PipelineCancelled("pipeline cancellation requested")
        failed = False
    finally:
        if failed:
            stack.__exit__(PipelineCancelled, PipelineCancelled(), None)
        else:
            stack.close()

    incomplete = (
        int(bam_writer.stats["reads_flushed_incomplete"])
        if bam_writer is not None
        else 0
    )
    if bam_writer is not None:
        records = int(bam_writer.stats["records_written"])
    status_queue.put(
        WriterFinalizing(
            records=records,
            batches=batches,
            sites=sites,
            incomplete_records=incomplete,
            bam_path=bam_final,
            tsv_path=tsv_final,
        )
    )
    if bam_work is not None:
        finalize_single_bam(
            bam_final,
            bam_work,
            threads=bam_finalize_threads,
            sort_and_index=should_sort_and_index_bam(args),
            index_threads=bam_finalize_threads,
        )
    if tsv_work is not None:
        if Path(tsv_work).resolve() != Path(tsv_final).resolve():
            os.replace(tsv_work, tsv_final)
    status_queue.put(
        WriterDone(
            records=records,
            batches=batches,
            sites=sites,
            incomplete_records=incomplete,
            bam_path=bam_final,
            tsv_path=tsv_final,
        )
    )


def _reader_worker(
    args,
    record_queue,
    status_queue,
    stop_event,
    consumers: int,
    bam_threads: int,
) -> None:
    _guarded_worker(
        "BAM reader",
        None,
        status_queue,
        stop_event,
        _reader_worker_impl,
        args,
        record_queue,
        status_queue,
        stop_event,
        consumers,
        bam_threads,
    )


def _feature_worker(worker_id, args, record_queue, feature_queue, status_queue, stop_event) -> None:
    _guarded_worker(
        "feature worker",
        worker_id,
        status_queue,
        stop_event,
        _feature_worker_impl,
        worker_id,
        args,
        record_queue,
        feature_queue,
        status_queue,
        stop_event,
    )


def _batcher_worker(
    args,
    output_format: str,
    feature_queue,
    model_queue,
    status_queue,
    stop_event,
    feature_workers,
    model_workers,
    consumer_release_event,
) -> None:
    _guarded_worker(
        "batcher",
        None,
        status_queue,
        stop_event,
        _batcher_worker_impl,
        args,
        output_format,
        feature_queue,
        model_queue,
        status_queue,
        stop_event,
        feature_workers,
        model_workers,
        consumer_release_event,
    )


def _gpu_worker(worker_id, device_index, args, model_queue, prediction_queue, status_queue, stop_event) -> None:
    _guarded_worker(
        "GPU worker" if device_index is not None else "model worker",
        worker_id,
        status_queue,
        stop_event,
        _gpu_worker_impl,
        worker_id,
        device_index,
        args,
        model_queue,
        prediction_queue,
        status_queue,
        stop_event,
    )


def _writer_worker(
    args,
    output_format,
    prediction_queue,
    status_queue,
    stop_event,
    model_workers,
    bam_write_threads,
    bam_finalize_threads,
) -> None:
    _guarded_worker(
        "writer",
        None,
        status_queue,
        stop_event,
        _writer_worker_impl,
        args,
        output_format,
        prediction_queue,
        status_queue,
        stop_event,
        model_workers,
        bam_write_threads,
        bam_finalize_threads,
    )


class MultiprocessInferencePipeline:
    """Coordinate the single-reader, multi-model, single-writer pipeline."""

    def __init__(self, args, output_format: str):
        if output_format not in ("bam", "tsv", "both"):
            raise ValueError(f"unsupported output format: {output_format}")
        self.args = args
        self.output_format = output_format
        if output_format in ("bam", "both"):
            from unimeth.ioutils.writer.bam_finalize import normalize_bam_path

            bam_path = Path(
                normalize_bam_path(args.bam_out_dir or args.out_dir)
            ).resolve()
            if bam_path == Path(args.bam_dir).resolve():
                raise ValueError("BAM output path must differ from the input BAM path")
        if output_format == "both":
            tsv_path = Path(args.tsv_out_dir or args.out_dir).resolve()
            if getattr(args, "gzip", False) and tsv_path.suffix != ".gz":
                tsv_path = Path(f"{tsv_path}.gz")
            if bam_path == tsv_path:
                raise ValueError(
                    "BAM and TSV outputs must use different paths when "
                    "--output_format both is selected"
                )

    @staticmethod
    def _visible_devices() -> tuple[int | None, ...]:
        import torch

        count = torch.cuda.device_count()
        return tuple(range(count)) if count else (None,)

    @staticmethod
    def _external_world_size() -> int:
        try:
            return int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            return 1

    def validate_launch(self) -> None:
        """Reject external multi-process launchers before shared setup begins."""
        if self._external_world_size() > 1:
            raise RuntimeError(
                "Do not use accelerate launch for inference. Run `unimeth infer` "
                "once; UniMeth will use all GPUs visible in CUDA_VISIBLE_DEVICES."
            )

    def _output_artifacts_to_clear(self) -> tuple[str, ...]:
        paths = []
        if self.output_format in ("bam", "both"):
            from unimeth.ioutils.writer.bam_finalize import (
                bam_sorting_path,
                normalize_bam_path,
            )

            final = str(normalize_bam_path(self.args.bam_out_dir or self.args.out_dir))
            paths.append(final)
            paths.append(str(bam_sorting_path(final)))
        if self.output_format in ("tsv", "both"):
            final = str(self.args.tsv_out_dir or self.args.out_dir)
            if getattr(self.args, "gzip", False) and not final.endswith(".gz"):
                final = f"{final}.gz"
            paths.append(final)
        return tuple(paths)

    def _cleanup_output_artifacts(self) -> None:
        for value in self._output_artifacts_to_clear():
            path = Path(value)
            if path.exists() and path.is_file():
                path.unlink()
        if self.output_format in ("bam", "both"):
            from unimeth.ioutils.writer.bam_finalize import normalize_bam_path

            final = normalize_bam_path(self.args.bam_out_dir or self.args.out_dir)
            index_paths = {
                Path(f"{final}.bai"),
                Path(f"{final}.csi"),
                final.with_suffix(".bai"),
                final.with_suffix(".csi"),
            }
            for index_path in index_paths:
                if index_path.exists() and index_path.is_file():
                    index_path.unlink()

    @staticmethod
    def _stop_processes(processes, stop_event) -> None:
        stop_event.set()
        deadline = time.monotonic() + _PROCESS_JOIN_SECONDS
        for process in processes:
            if process.pid is None:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=5.0)

    @staticmethod
    def _raise_on_dead_child(processes) -> None:
        for process in processes:
            if process.exitcode not in (None, 0):
                raise RuntimeError(
                    f"pipeline process {process.name!r} exited with code "
                    f"{process.exitcode} without reporting an error"
                )

    @staticmethod
    def _apply_status_message(
        message,
        reader_stats,
        feature_done,
        gpu_done,
    ):
        """Collect terminal diagnostics that may arrive just after writer completion."""
        batcher_done = None
        if isinstance(message, (ReaderProgress, ReaderDone)):
            reader_stats.clear()
            reader_stats.update(message.stats)
        elif isinstance(message, FeatureWorkerDone):
            feature_done.append(message)
        elif isinstance(message, BatcherDone):
            batcher_done = message
        elif isinstance(message, GPUWorkerDone):
            gpu_done.append(message)
        return batcher_done

    def run(self) -> WriterDone:
        self.validate_launch()

        import torch.multiprocessing as multiprocessing

        devices = self._visible_devices()
        gpu_count = len(devices) if devices[0] is not None else 0
        raw_workers = getattr(self.args, "num_workers", None)
        configured_workers = (
            None
            if getattr(self.args, "num_workers_auto", raw_workers is None)
            else int(raw_workers)
        )
        layout = resolve_pipeline_layout(configured_workers, gpu_count)
        model_workers = layout.model_workers
        feature_workers = layout.feature_workers
        bam_read_threads = layout.bam_read_threads
        bam_write_threads = layout.bam_write_threads
        bam_finalize_threads = layout.bam_finalize_threads
        device_label = (
            f"{layout.gpu_count} "
            f'{"GPU" if layout.gpu_count == 1 else "GPUs"}'
            if layout.gpu_count
            else "CPU"
        )
        logger.info(
            "Inference pipeline: %s, %s feature worker(s) (%s), "
            "BAM threads read=%s/write=%s/finalize=%s, "
            "%s processes total (including main)",
            device_label,
            feature_workers,
            "auto" if layout.is_auto else "configured",
            bam_read_threads,
            bam_write_threads,
            bam_finalize_threads,
            layout.total_processes,
        )

        context = multiprocessing.get_context("spawn")
        stop_event = context.Event()
        # Released by the coordinator only after every model consumer has exited.
        batch_consumers_done = context.Event()
        status_queue = context.Queue()
        record_queue = context.Queue(maxsize=max(2, feature_workers * 2))
        feature_queue = context.Queue(maxsize=max(4, feature_workers * 2))
        model_queue = context.Queue(maxsize=max(2, model_workers * 2))
        prediction_queue = context.Queue(maxsize=max(2, model_workers * 2))

        writer = context.Process(
            target=_writer_worker,
            name="unimeth-writer",
            args=(
                self.args,
                self.output_format,
                prediction_queue,
                status_queue,
                stop_event,
                model_workers,
                bam_write_threads,
                bam_finalize_threads,
            ),
        )
        gpu_processes = [
            context.Process(
                target=_gpu_worker,
                name=f"unimeth-model-{worker_id}",
                args=(
                    worker_id,
                    device_index,
                    self.args,
                    model_queue,
                    prediction_queue,
                    status_queue,
                    stop_event,
                ),
            )
            for worker_id, device_index in enumerate(devices)
        ]
        batcher = context.Process(
            target=_batcher_worker,
            name="unimeth-batcher",
            args=(
                self.args,
                self.output_format,
                feature_queue,
                model_queue,
                status_queue,
                stop_event,
                feature_workers,
                model_workers,
                batch_consumers_done,
            ),
        )
        feature_processes = [
            context.Process(
                target=_feature_worker,
                name=f"unimeth-feature-{worker_id}",
                args=(
                    worker_id,
                    self.args,
                    record_queue,
                    feature_queue,
                    status_queue,
                    stop_event,
                ),
            )
            for worker_id in range(feature_workers)
        ]
        reader = context.Process(
            target=_reader_worker,
            name="unimeth-bam-reader",
            args=(
                self.args,
                record_queue,
                status_queue,
                stop_event,
                feature_workers,
                bam_read_threads,
            ),
        )
        processes = [writer, *gpu_processes, batcher, *feature_processes, reader]

        ready = set()
        try:
            self._cleanup_output_artifacts()
            writer.start()
            for process in gpu_processes:
                process.start()

            while len(ready) < model_workers:
                try:
                    message = status_queue.get(timeout=_QUEUE_POLL_SECONDS)
                except queue.Empty:
                    self._raise_on_dead_child([writer, *gpu_processes])
                    if writer.exitcode == 0:
                        raise RuntimeError(
                            "writer exited before model workers became ready"
                        )
                    for worker_id, process in enumerate(gpu_processes):
                        if process.exitcode == 0 and worker_id not in ready:
                            raise RuntimeError(
                                f"model worker {worker_id} exited before reporting ready"
                            )
                    continue
                if isinstance(message, PipelineFailure):
                    raise MultiprocessInferenceError(message)
                if isinstance(message, GPUWorkerReady):
                    ready.add(message.worker_id)

            batcher.start()
            for process in feature_processes:
                process.start()
            reader.start()

            description = {
                "bam": "Inference (BAM)",
                "tsv": "Inference",
                "both": "Inference (TSV+BAM)",
            }[self.output_format]
            progress = tqdm(desc=description, unit=" record", dynamic_ncols=True)
            started = time.perf_counter()
            reader_stats = {"total_records": 0, "yielded_records": 0}
            writer_done = None
            feature_done = []
            gpu_done = []
            batcher_done = None
            inference_elapsed = None
            inference_logged = False
            try:
                while writer_done is None:
                    try:
                        message = status_queue.get(timeout=_QUEUE_POLL_SECONDS)
                    except queue.Empty:
                        self._raise_on_dead_child(processes)
                        if writer.exitcode == 0:
                            raise RuntimeError(
                                "writer exited without reporting completion"
                            )
                        continue
                    if isinstance(message, PipelineFailure):
                        raise MultiprocessInferenceError(message)
                    collected_batcher = self._apply_status_message(
                        message,
                        reader_stats,
                        feature_done,
                        gpu_done,
                    )
                    if collected_batcher is not None:
                        batcher_done = collected_batcher
                    if isinstance(message, WriterProgress):
                        progress.update(max(0, message.records - int(progress.n)))
                        elapsed = max(time.perf_counter() - started, 1e-9)
                        progress.set_postfix_str(
                            ", ".join(
                                (
                                    f"{reader_stats['total_records']:,} scan",
                                    f"{reader_stats['yielded_records']:,} pass",
                                    f"{format_compact_count(message.sites)} sites",
                                    f"{message.sites / elapsed:,.0f} sites/s",
                                )
                            )
                        )
                    elif isinstance(message, WriterFinalizing):
                        progress.update(max(0, message.records - int(progress.n)))
                        progress.close()
                        inference_elapsed = time.perf_counter() - started
                        logger.info(
                            "Inference complete: %s records, %s batches, "
                            "%s sites, %.1fs",
                            message.records,
                            message.batches,
                            message.sites,
                            inference_elapsed,
                        )
                        inference_logged = True
                        if message.bam_path is not None:
                            logger.info("Finalizing BAM output...")
                    elif isinstance(message, WriterDone):
                        writer_done = message
            finally:
                progress.close()

            gpu_join_deadline = time.monotonic() + _PROCESS_JOIN_SECONDS
            for process in gpu_processes:
                process.join(
                    timeout=max(0.0, gpu_join_deadline - time.monotonic())
                )
            self._raise_on_dead_child(gpu_processes)
            alive_gpu_workers = [
                process.name for process in gpu_processes if process.is_alive()
            ]
            if alive_gpu_workers:
                raise RuntimeError(
                    "model processes did not exit: "
                    + ", ".join(alive_gpu_workers)
                )
            batch_consumers_done.set()
            for process in processes:
                process.join(timeout=_PROCESS_JOIN_SECONDS)
            self._raise_on_dead_child(processes)
            alive = [process.name for process in processes if process.is_alive()]
            if alive:
                raise RuntimeError(
                    "pipeline processes did not exit: " + ", ".join(alive)
                )

            while True:
                try:
                    message = status_queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(message, PipelineFailure):
                    raise MultiprocessInferenceError(message)
                collected_batcher = self._apply_status_message(
                    message,
                    reader_stats,
                    feature_done,
                    gpu_done,
                )
                if collected_batcher is not None:
                    batcher_done = collected_batcher

            if not inference_logged:
                inference_elapsed = time.perf_counter() - started
                logger.info(
                    "Inference complete: %s records, %s batches, %s sites, %.1fs",
                    writer_done.records,
                    writer_done.batches,
                    writer_done.sites,
                    inference_elapsed,
                )
            logger.debug(
                "BAM records: total=%s, passed=%s",
                reader_stats.get("total_records", 0),
                reader_stats.get("yielded_records", 0),
            )
            logger.debug(
                "BAM filters: unmapped=%s, secondary=%s, duplicate=%s, "
                "supplementary=%s, mapq=%s, identity=%s, chromosome=%s",
                reader_stats.get("filtered_unmapped", 0),
                reader_stats.get("filtered_secondary", 0),
                reader_stats.get("filtered_duplicate", 0),
                reader_stats.get("filtered_supplementary", 0),
                reader_stats.get("filtered_mapq", 0),
                reader_stats.get("filtered_identity", 0),
                reader_stats.get("filtered_chromosome", 0),
            )
            if batcher_done is not None:
                logger.debug(
                    "Global batching: patches=%s, batches=%s",
                    batcher_done.patches,
                    batcher_done.batches,
                )
            if gpu_done:
                logger.debug(
                    "Model workers: %s",
                    ", ".join(
                        f"gpu{item.worker_id}={item.batches} batches/{item.model_seconds:.2f}s"
                        for item in sorted(gpu_done, key=lambda value: value.worker_id)
                    ),
                )
            if feature_done:
                feature_totals = {
                    name: sum(item.stats.get(name, 0) for item in feature_done)
                    for name in (
                        "lookup_batches",
                        "signal_missing_records",
                        "signal_source_hint_missing_records",
                        "hard_clipped_reconciled_records",
                        "signal_sequence_mismatch_records",
                        "feature_empty_records",
                        "feature_records",
                    )
                }
                logger.debug(
                    "Streaming features: signal_missing=%s, "
                    "signal_source_hint_missing=%s, hard_clipped_reconciled=%s, "
                    "signal_sequence_mismatch=%s, feature_empty=%s, "
                    "no_patches=%s, feature_records=%s, yielded_patches=%s, "
                    "lookup_batches=%s",
                    feature_totals["signal_missing_records"],
                    feature_totals["signal_source_hint_missing_records"],
                    feature_totals["hard_clipped_reconciled_records"],
                    feature_totals["signal_sequence_mismatch_records"],
                    feature_totals["feature_empty_records"],
                    sum(item.records_without_patches for item in feature_done),
                    feature_totals["feature_records"],
                    sum(item.yielded_patches for item in feature_done),
                    feature_totals["lookup_batches"],
                )
            if writer_done.incomplete_records:
                logger.warning(
                    "%s incomplete record(s) were written with partial MM/ML tags",
                    f"{writer_done.incomplete_records:,}",
                )
            if writer_done.bam_path is not None:
                logger.info("Final BAM: %s", writer_done.bam_path)
            if writer_done.tsv_path is not None:
                logger.info("Final TSV: %s", writer_done.tsv_path)
            return writer_done
        except BaseException:
            self._stop_processes(processes, stop_event)
            raise
        finally:
            for value in (
                record_queue,
                feature_queue,
                model_queue,
                prediction_queue,
                status_queue,
            ):
                try:
                    value.close()
                    value.join_thread()
                except Exception:
                    pass
