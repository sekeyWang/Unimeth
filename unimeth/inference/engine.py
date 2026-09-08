"""
Inference engine for UniMeth.
"""
import time
import os
import warnings

# Suppress transformers warnings in all processes
warnings.filterwarnings('ignore', message='.*past_key_values.*')
warnings.filterwarnings('ignore', message='.*EncoderDecoderCache.*')
warnings.filterwarnings('ignore', message='.*ipex flag.*')
warnings.filterwarnings('ignore', message='.*kernel version.*')

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from tqdm import tqdm

from unimeth.config import tokenizer, get_total_stride
from unimeth.model.datasets import collate_fn
from unimeth.model.loader import load_model
from unimeth.utils import local_print
from unimeth.ioutils.reader.bam import (
    BamReader,
    bam_index_needs_rebuild,
    cleanup_bam_index,
    default_bam_index_file,
    resolve_bam_index_file,
)
from unimeth.ioutils.writer.bam_finalize import (
    bam_has_references,
    bam_part_glob,
    bam_part_path,
    finalize_part_bams,
    normalize_bam_path,
    select_latest_bam_records,
)
from unimeth.inference.coordination import (
    CompletionCoordinator,
)
from unimeth.inference.resume import (
    GracefulStopRequested,
    GracefulStopper,
    ReadCompletionTracker,
    ResumeCheckpoint,
)


class InferenceEngine:
    """Unified inference engine supporting TSV/BAM output formats."""
    
    def __init__(self, args, dataset_class):
        self.args = args
        self.dataset_class = dataset_class
        self.accelerator = Accelerator(dataloader_config=DataLoaderConfiguration(dispatch_batches=False))
        self.model = None
        self.dataset = None
        self.dataloader = None
        self._methylated_idx = tokenizer['+']
        self._unmethylated_idx = tokenizer['-']
        self._bam_index_file = None
        self._bam_index_is_temporary = False
        self._bam_index_created = False
    
    def load_model(self):
        """Load and prepare model for inference."""
        device = self.accelerator.device
        self.model = load_model(
            config=getattr(self.args, 'model_type', 'default'),
            model_path=getattr(self.args, 'model_dir', None),
            mode='inference',
            device=device,

        )
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
    
    def setup_dataloader(self):
        """Setup dataset and dataloader for inference."""
        self.dataset = self.dataset_class(
            pod5_dir=self.args.signal_dir,
            bam_dir=self.args.bam_dir,
            args=self.args
        )
        
        num_workers = getattr(self.args, 'num_workers', 2)
        import functools
        total_stride = get_total_stride(getattr(self.args, 'model_type', 'default'))
        collate_fn_with_stride = functools.partial(collate_fn, 'inference', total_stride=total_stride)
        
        # Do NOT call accelerator.prepare(dataloader) for IterableDataset:
        # Accelerate's IterableDatasetShard shards at the patch level, which would
        # scatter patches from the same read across ranks. We shard at read level
        # inside Pod5BamDataset.__iter__ instead, so each rank gets complete reads.
        self.dataloader = DataLoader(
            self.dataset,
            collate_fn=collate_fn_with_stride,
            batch_size=self.args.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False
        )

    def _get_bam_index_cache_dir(self, output_format: str) -> str:
        """Use the selected output file directory for temporary fallback index cache."""
        if output_format in ('bam', 'both'):
            output_path = getattr(self.args, 'bam_out_dir', None) or getattr(self.args, 'out_dir', None)
        else:
            output_path = getattr(self.args, 'tsv_out_dir', None)

        if output_path is None:
            output_path = getattr(self.args, 'out_dir', None)
        if output_path is None:
            return os.getcwd()
        return os.path.dirname(os.path.abspath(output_path)) or os.getcwd()

    def _get_resume_output_path(self, output_format: str):
        """Use the selected final output path as the resume sidecar base."""
        if output_format in ('bam', 'both'):
            output_path = getattr(self.args, 'bam_out_dir', None) or getattr(self.args, 'out_dir', None)
            if output_path:
                return normalize_bam_path(output_path)

        output_path = getattr(self.args, 'tsv_out_dir', None) or getattr(self.args, 'out_dir', None)
        if output_path:
            return output_path

        raise ValueError("--resume requires an output path")

    def _wait_for_bam_index(self, bam_path: str, index_file: str):
        """Wait for the main process to finish writing the BAM index."""
        while bam_index_needs_rebuild(bam_path, index_file):
            time.sleep(1.0)

    def _prepare_bam_index(self, output_format: str):
        """Build or wait for the BAM read-id index before DataLoader workers start."""
        cache_dir = self._get_bam_index_cache_dir(output_format)
        self.args.bam_index_cache_dir = cache_dir

        preferred_index = default_bam_index_file(self.args.bam_dir)
        require_writable = bam_index_needs_rebuild(self.args.bam_dir, preferred_index)
        index_file, _ = resolve_bam_index_file(
            self.args.bam_dir,
            cache_dir=cache_dir,
            require_writable=require_writable,
        )

        if self.accelerator.is_main_process:
            index_threads = min(max(1, int(getattr(self.args, 'num_workers', 1) or 1)), 4)
            bam_reader = BamReader(
                self.args.bam_dir,
                force_rebuild_index=False,
                index_cache_dir=cache_dir,
                threads=index_threads,
            )
            self._bam_index_file = bam_reader.bam_index_file
            self._bam_index_is_temporary = bam_reader.bam_index_is_temporary
            self._bam_index_created = bam_reader.bam_index_created
            try:
                bam_reader.bam_file.close()
            except Exception:
                pass
        else:
            self._wait_for_bam_index(self.args.bam_dir, index_file)

    def _cleanup_bam_index(self):
        """Remove the temporary fallback BAM index created by this inference run."""
        if not self.accelerator.is_main_process:
            return
        cleanup_bam_index(
            self._bam_index_file,
            is_temporary=self._bam_index_is_temporary,
            created_by_this_run=self._bam_index_created,
        )
        self._bam_index_created = False
    
    def _extract_predictions(self, decoder_input_ids, logits, patch_pos):
        """Extract methylation predictions from unimeth.model outputs (fully vectorized)."""
        device = logits.device
        batch_size = logits.shape[0]
        
        site_counts = [len(p) for p in patch_pos]
        total_sites = sum(site_counts)
        if total_sites == 0:
            return torch.tensor([], device=device), torch.tensor([], device=device, dtype=torch.long)
        
        # Build flat indices on CPU then move to GPU (avoids large padded tensor)
        batch_indices = torch.arange(batch_size).repeat_interleave(torch.tensor(site_counts)).to(device)
        
        pos_tensors = [torch.as_tensor(p, device=device) for p in patch_pos if len(p) > 0]
        pos_indices = torch.cat(pos_tensors)
        
        # Gather logits and decoder ids in one shot
        selected_logits = logits[batch_indices, pos_indices]
        
        # Extract methylated/unmethylated logits and compute softmax
        relevant_logits = selected_logits[:, [self._unmethylated_idx, self._methylated_idx]]
        probs = torch.softmax(relevant_logits, dim=-1)
        all_preds = probs[:, 1]
        
        # Gather methylation types
        all_methy = decoder_input_ids.to(device)[batch_indices, pos_indices]
        
        return all_preds, all_methy
    
    def run(self, output_format: str = 'bam'):
        try:
            return self._run_impl(output_format=output_format)
        finally:
            self._cleanup_bam_index()

    def _run_impl(self, output_format: str = 'bam'):
        """
        Run inference with specified output format.

        Args:
            output_format: 'tsv', 'bam', or 'both' (dual output for verification)
        """
        # Disable reading progress by default for clean output
        if not getattr(self.args, 'show_reading_progress', False):
            os.environ['UNIMETH_DISABLE_READING_PROGRESS'] = '1'

        is_main = self.accelerator.is_main_process
        rank = self.accelerator.process_index
        completion_coordinator = CompletionCoordinator.start(
            output_path=self._get_resume_output_path(output_format),
            rank=rank,
            num_processes=self.accelerator.num_processes,
            is_main_process=is_main,
            startup_barrier=self.accelerator.wait_for_everyone,
        )
        resume_checkpoint = resume_tracker = None
        resume_part_suffix = None

        if getattr(self.args, 'resume', False):
            resume_checkpoint = ResumeCheckpoint(self._get_resume_output_path(output_format), rank)
            if output_format == 'tsv':
                resume_tracker = ReadCompletionTracker(resume_checkpoint)
            resume_part_suffix = resume_checkpoint.part_suffix
            self.args.resume_completed_read_ids = resume_checkpoint.completed_read_ids
            if is_main:
                skipped = len(resume_checkpoint.completed_read_ids)
                local_print(f"Resume enabled: skipping {skipped:,} completed read(s)")
        else:
            self.args.resume_completed_read_ids = None

        self._prepare_bam_index(output_format)
        self.setup_dataloader()
        self.load_model()

        # Initialize writer(s) based on format
        tsv_writer = bam_writer = None

        if output_format in ('tsv', 'both'):
            from unimeth.ioutils.writer.tsv import TSVWriter
            tsv_path = self.args.tsv_out_dir if self.args.tsv_out_dir else self.args.out_dir
            tsv_writer = TSVWriter(
                output_path=tsv_path,
                num_processes=self.accelerator.num_processes,
                process_index=rank,
                max_queue_size=50,
                gzip_output=getattr(self.args, 'gzip', False),
                part_suffix=resume_part_suffix,
                completed_read_ids=(
                    resume_checkpoint.completed_read_ids if resume_checkpoint is not None else None
                ),
                sync_writes=resume_checkpoint is not None,
            )

        if output_format in ('bam', 'both'):
            from unimeth.ioutils.writer.bam_aggregation import AggregationBAMWriter
            bam_path = normalize_bam_path(self.args.bam_out_dir if self.args.bam_out_dir else self.args.out_dir)
            rank_bam_path = bam_part_path(bam_path, rank, part_suffix=resume_part_suffix)
            bam_reader = BamReader(
                self.args.bam_dir,
                force_rebuild_index=False,
                index_cache_dir=getattr(self.args, 'bam_index_cache_dir', None),
                allow_index_build=False,
            )
            bam_writer = AggregationBAMWriter(
                output_path=str(rank_bam_path),
                template_bam_path=self.args.bam_dir,
                bam_reader=bam_reader,
                keep_mv=getattr(self.args, 'keep_mv', False),
            )

        def record_completed_bam_reads():
            """Persist only read IDs confirmed complete by the BAM writer."""
            if resume_checkpoint is None or bam_writer is None:
                return
            for completed_read_id in bam_writer.pop_completed_read_ids():
                resume_checkpoint.record_read(completed_read_id)

        pbar_desc = {'tsv': 'Inference', 'bam': 'Inference (BAM)', 'both': 'Inference (TSV+BAM)'}.get(output_format, 'Inference')

        total_batches = total_samples = 0
        pbar = tqdm(desc=pbar_desc, unit="batch", disable=not is_main, dynamic_ncols=True)

        # Timing
        times = {'model': [], 'extract': [], 'write': [], 'other': []}
        t3 = inference_start = time.perf_counter()
        Preload = Warmup = 0

        # Context manager helper: open all active writers
        import contextlib
        writers = [w for w in (tsv_writer, bam_writer) if w is not None]

        graceful_stop_requested = False
        with GracefulStopper(enabled=resume_checkpoint is not None) as graceful_stopper:
            try:
                with contextlib.ExitStack() as stack:
                    if resume_checkpoint is not None:
                        stack.callback(resume_checkpoint.close)
                    for w in writers:
                        stack.enter_context(w)

                    try:
                        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            batch_iter = enumerate(self.dataloader)
                            while True:
                                graceful_stopper.raise_if_requested()
                                try:
                                    batch_idx, batch = next(batch_iter)
                                except StopIteration:
                                    break

                                if self.args.limit is not None and batch_idx >= self.args.limit:
                                    break

                                total_batches += 1

                                has_reads_complete_marker = (
                                    isinstance(batch, dict)
                                    and batch.get('__reads_complete__', False)
                                )
                                has_batch_data = isinstance(batch, dict) and 'signals' in batch

                                # Handle pure reads_complete marker before model inference.
                                # If the marker is co-batched with data, process the data first.
                                if has_reads_complete_marker and not has_batch_data:
                                    if bam_writer is not None:
                                        bam_writer.on_reads_complete()
                                    record_completed_bam_reads()
                                    graceful_stopper.raise_if_requested()
                                    continue

                                # Model forward — manually move tensors to device (dataloader not prepared)
                                times['other'].append(time.perf_counter() - t3)
                                t0 = time.perf_counter()
                                logits = self.model(
                                    signals=batch['signals'].to(self.accelerator.device),
                                    encoder_mask=batch['encoder_mask'].to(self.accelerator.device),
                                    decoder_input_ids=batch['decoder_input_ids'].to(self.accelerator.device),
                                    signal_pos=batch['signal_pos'].to(self.accelerator.device),
                                )
                                torch.cuda.synchronize()
                                times['model'].append(time.perf_counter() - t0)

                                # Extract predictions
                                t1 = time.perf_counter()
                                preds, methy = self._extract_predictions(
                                    batch['decoder_input_ids'],
                                    logits,
                                    batch['patch_pos']
                                )
                                torch.cuda.synchronize()
                                times['extract'].append(time.perf_counter() - t1)

                                # Write (same preds/methy to all active writers)
                                t2 = time.perf_counter()
                                samples_written = 0
                                if tsv_writer is not None:
                                    samples_written = tsv_writer.write_batch(
                                        preds=preds, methy=methy,
                                        read_ids=batch['read_id'], chrs=batch['chr'],
                                        strands=batch['strand'], ref_pos=batch['ref_pos'],
                                        read_pos=batch['read_pos'], labels=batch['labels'],
                                        patch_pos=batch['patch_pos']
                                    )
                                if bam_writer is not None:
                                    n = bam_writer.write_batch(
                                        preds=preds, methy=methy,
                                        read_ids=batch['read_id'], chrs=batch['chr'],
                                        strands=batch['strand'], ref_pos=batch['ref_pos'],
                                        read_pos=batch['read_pos'], patch_pos=batch['patch_pos'],
                                        patch_idx=batch['patch_idx'], total_patches=batch['total_patches'],
                                    )
                                    if tsv_writer is None:
                                        samples_written = n
                                times['write'].append(time.perf_counter() - t2)
                                total_samples += samples_written

                                # Flush BAM buffer when marker was co-batched with data
                                if has_reads_complete_marker and bam_writer is not None:
                                    bam_writer.on_reads_complete()

                                record_completed_bam_reads()
                                if resume_tracker is not None:
                                    resume_tracker.update_batch(batch)

                                graceful_stopper.raise_if_requested()

                                if batch_idx == 0:
                                    Preload = t0 - inference_start
                                    Warmup = time.perf_counter() - t0

                                pbar.update(1)
                                elapsed = time.perf_counter() - inference_start
                                pbar.set_postfix_str(f'{total_samples:,} samples, {total_samples/elapsed:,.0f}/s')
                                t3 = time.perf_counter()
                    finally:
                        pbar.close()
                        if bam_writer is not None:
                            bam_writer.on_reads_complete()
                        record_completed_bam_reads()
            except GracefulStopRequested as stop:
                graceful_stop_requested = True
                if is_main:
                    local_print(f"\n{stop}; keeping resume files for the next run")

        if graceful_stop_requested:
            return

        # Finalize
        # Every rank has closed its writer. This filesystem protocol permits an
        # early rank to wait for a slow tail without the NCCL watchdog timeout.
        incomplete_read_count = (
            bam_writer.stats["reads_flushed_incomplete"]
            if bam_writer is not None
            else 0
        )
        completion_coordinator.mark_rank_complete(
            incomplete_read_count=incomplete_read_count,
        )

        if not is_main:
            completion_coordinator.wait_for_finalization()
            completion_coordinator.acknowledge_finalization()
            return

        local_print("Waiting for all ranks to finish writing their output parts...")
        completion_coordinator.wait_for_all()
        incomplete_read_count = completion_coordinator.incomplete_read_count()
        inference_time = time.perf_counter() - inference_start
        local_print(f"\nInference complete: {total_batches} batches, {total_samples} samples, {inference_time:.1f}s")

        if total_batches > 0:
            times['preload'] = [0, Preload]
            times['warmup'] = [0, Warmup]
            local_print(f"\n{'='*60}")
            local_print("Per-batch timing breakdown:")
            cover = 0
            for name, vals in times.items():
                vals = vals[1:]
                if len(vals) == 0:
                    continue
                avg_ms = sum(vals) / len(vals) * 1000
                total_pct = sum(vals) / inference_time * 100
                local_print(f"  {name:10s}: {avg_ms:10.2f} ms/batch ({sum(vals):7.2f}/{inference_time:7.2f}={total_pct:5.1f}% total)")
                cover += sum(vals)
            local_print(f"  {'Cover':10s}: {cover:10.2f}/{inference_time:7.2f}={(100*cover/inference_time):5.1f}% total")
            local_print(f"{'='*60}")

        has_incomplete_reads = incomplete_read_count > 0
        finalize_ok = True
        try:
            if has_incomplete_reads:
                local_print(
                    f"Warning: {incomplete_read_count:,} incomplete read(s) were written with "
                    "partial MM/ML tags. Finalizing all latest records"
                )
            if bam_writer is not None:
                import glob

                bam_path = normalize_bam_path(self.args.bam_out_dir or self.args.out_dir)
                part_files = sorted(glob.glob(bam_part_glob(bam_path)))
                if part_files:
                    selected_part_files = part_files
                    if resume_checkpoint is not None:
                        selected_part_files = select_latest_bam_records(part_files)
                    if not selected_part_files:
                        finalize_ok = False
                        local_print("Warning: No BAM records were available to finalize")
                    else:
                        local_print(f"Merging {len(selected_part_files)} rank BAM(s)...")
                        try:
                            sort_and_index = bam_has_references(self.args.bam_dir)
                            finalize_part_bams(
                                str(bam_path),
                                selected_part_files,
                                sort_and_index=sort_and_index,
                            )
                            if resume_checkpoint is not None:
                                for part_file in part_files:
                                    if os.path.exists(part_file):
                                        os.remove(part_file)
                            if has_incomplete_reads:
                                local_print(
                                    f"Final BAM: {bam_path} "
                                    f"(including {incomplete_read_count:,} incomplete read(s) with "
                                    "partial MM/ML tags)"
                                )
                            else:
                                local_print(f"Final BAM: {bam_path}")
                        except Exception as e:
                            finalize_ok = False
                            local_print(f"Warning: Failed to finalize BAM files: {e}")

            if tsv_writer is not None:
                if not finalize_ok:
                    local_print(
                        "Skipping TSV merge because BAM finalization was not safe; "
                        "resume files were kept"
                    )
                else:
                    if resume_checkpoint is not None:
                        tsv_writer.completed_read_ids = resume_checkpoint.refresh_completed_read_ids()
                    tsv_writer.merge_outputs(is_main_process=True)

            if resume_checkpoint is not None and finalize_ok:
                resume_checkpoint.cleanup()
        except Exception:
            finalize_ok = False
            raise
        finally:
            completion_coordinator.mark_finalized(finalize_ok)

        if finalize_ok:
            completion_coordinator.wait_for_finalization_acknowledgements()
            completion_coordinator.cleanup()
    
    # Backward compatibility alias
    def run_bam(self):
        """Backward compatible alias for run('bam')."""
        return self.run(output_format='bam')
