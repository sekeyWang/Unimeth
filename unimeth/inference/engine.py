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
from unimeth.ioutils.reader.bam import BamReader
# from unimeth.ioutils.writer.bam_aggregation import AggregationBAMWriter
from unimeth.ioutils.writer.bam_finalize import finalize_part_bams


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
    
    def setup_dataloader(self):
        """Setup dataset and dataloader for inference."""
        self.dataset = self.dataset_class(
            pod5_dir=self.args.pod5_dir,
            bam_dir=self.args.bam_dir,
            args=self.args
        )
        
        num_workers = getattr(self.args, 'num_workers', 8)
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
    
    def run(self, output_format: str = 'tsv'):
        """
        Run inference with specified output format.

        Args:
            output_format: 'tsv', 'bam', or 'both' (dual output for verification)
        """
        # Disable reading progress by default for clean output
        if not getattr(self.args, 'show_reading_progress', False):
            os.environ['UNIMETH_DISABLE_READING_PROGRESS'] = '1'

        self.setup_dataloader()
        self.load_model()

        is_main = self.accelerator.is_main_process
        rank = self.accelerator.process_index

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
            )

        if output_format in ('bam', 'both'):
            from unimeth.ioutils.writer.bam_aggregation import AggregationBAMWriter
            bam_path = self.args.bam_out_dir if self.args.bam_out_dir else self.args.out_dir
            bam_part_path = bam_path.replace('.bam', f'.part_{rank}.bam')
            # Only rank 0 rebuilds the index to avoid race condition when all ranks
            # write to the same cache file simultaneously.
            if is_main:
                BamReader(self.args.bam_dir, force_rebuild_index=True)
            self.accelerator.wait_for_everyone()
            bam_reader = BamReader(self.args.bam_dir, force_rebuild_index=False)
            bam_writer = AggregationBAMWriter(
                output_path=bam_part_path,
                template_bam_path=self.args.bam_dir,
                bam_reader=bam_reader,
            )

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

        with contextlib.ExitStack() as stack:
            for w in writers:
                stack.enter_context(w)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for batch_idx, batch in enumerate(self.dataloader):
                    if self.args.limit is not None and batch_idx >= self.args.limit:
                        break

                    total_batches += 1

                    # Handle reads_complete marker before model inference
                    if isinstance(batch, dict) and batch.get('__reads_complete__', False):
                        if bam_writer is not None:
                            bam_writer.on_reads_complete()
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
                    if batch.get('__reads_complete__', False) and bam_writer is not None:
                        bam_writer.on_reads_complete()

                    if batch_idx == 0:
                        Preload = t0 - inference_start
                        Warmup = time.perf_counter() - t0

                    pbar.update(1)
                    elapsed = time.perf_counter() - inference_start
                    pbar.set_postfix_str(f'{total_samples:,} samples, {total_samples/elapsed:,.0f}/s')
                    t3 = time.perf_counter()

                pbar.close()

        # Finalize
        self.accelerator.wait_for_everyone()

        if is_main:
            inference_time = time.perf_counter() - inference_start
            local_print(f"\nInference complete: {total_batches} batches, {total_samples} samples, {inference_time:.1f}s")

            # Print timing
            if total_batches > 0:
                times['preload'] = [0, Preload]
                times['warmup'] = [0, Warmup]
                local_print(f"\n{'='*60}")
                local_print(f"Per-batch timing breakdown:")
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

        # Format-specific finalization
        if tsv_writer is not None:
            tsv_writer.merge_outputs(is_main_process=is_main)

        if bam_writer is not None:
            if is_main:
                import glob

                bam_path = self.args.bam_out_dir or self.args.out_dir
                part_files = sorted(glob.glob(bam_path.replace('.bam', '.part_*.bam')))
                if part_files:
                    local_print(f"Merging {len(part_files)} part BAM(s)...")
                    try:
                        finalize_part_bams(bam_path, part_files)
                        local_print(f"Final BAM: {bam_path}")
                    except Exception as e:
                        local_print(f"Warning: Failed to finalize BAM files: {e}")
    
    # Backward compatibility alias
    def run_bam(self):
        """Backward compatible alias for run('bam')."""
        return self.run(output_format='bam')
