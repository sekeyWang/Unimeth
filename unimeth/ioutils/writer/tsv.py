"""
TSV writer using background thread for asynchronous I/O.
"""
import gzip
import shutil
import threading
import queue
from pathlib import Path
from typing import List, Dict, Any
import torch

from unimeth.config import VOCAB as vocab
from unimeth.inference.resume import is_completed_tsv_line
from unimeth.utils.common import make_output_path, merge_rank_files


class TSVWriter:
    """
    TSV writer that offloads file I/O to a background thread for better performance.
    
    This significantly reduces the time spent in write_batch() by:
    1. Queuing writes instead of waiting for disk I/O
    2. Background thread handles all file operations
    3. Main thread only does fast queue.put()
    """
    
    def __init__(self, output_path: str, num_processes: int, process_index: int,
                 max_queue_size: int = 100, gzip_output: bool = False,
                 part_suffix: str | None = None,
                 completed_read_ids: set[str] | None = None,
                 sync_writes: bool = False):
        """
        Initialize async TSV writer.
        
        Args:
            output_path: Final output file path
            num_processes: Total number of processes (ranks)
            process_index: Current process index
            max_queue_size: Maximum write queue size (default: 100)
            gzip_output: Whether to gzip-compress the final TSV output
            part_suffix: Optional suffix for resume part files
            completed_read_ids: If set, merge only rows from completed reads
            sync_writes: Write batches synchronously instead of using a thread
        """
        output_path = make_output_path(output_path)
        self.gzip_output = gzip_output or output_path.suffix == '.gz'
        self.output_path = self._gzip_path(output_path) if self.gzip_output else output_path
        self.rank_output_base = self._plain_tsv_path(self.output_path) if self.gzip_output else self.output_path
        self.num_processes = num_processes
        self.process_index = process_index
        self.part_suffix = part_suffix
        self.completed_read_ids = completed_read_ids
        self.sync_writes = sync_writes
        self._file = None
        
        # Each rank gets its own output file
        self.rank_output = self.rank_output_for(process_index)
        
        # Threading setup
        self.write_queue = queue.Queue(maxsize=max_queue_size)
        self.writer_thread = None
        self._stop_event = threading.Event()
        self._total_written = 0

    @staticmethod
    def _gzip_path(output_path: Path) -> Path:
        """Return output_path with a .gz suffix, without duplicating it."""
        if output_path.suffix == '.gz':
            return output_path
        return Path(f"{output_path}.gz")

    @staticmethod
    def _plain_tsv_path(output_path: Path) -> Path:
        """Return the uncompressed TSV path used for rank temp files."""
        if output_path.suffix == '.gz':
            return output_path.with_suffix('')
        return output_path

    def rank_output_for(self, rank: int, part_suffix: str | None = None) -> Path:
        """Return rank-specific temporary output path."""
        selected_suffix = self.part_suffix if part_suffix is None else part_suffix
        suffix = f"_{selected_suffix}" if selected_suffix else ""
        return self.rank_output_base.parent / \
            f"{self.rank_output_base.stem}_rank{rank}{suffix}{self.rank_output_base.suffix}"

    def rank_outputs_for_merge(self, rank: int) -> List[Path]:
        """Return all temporary output files for a rank in merge order."""
        if not self.part_suffix and self.completed_read_ids is None:
            rank_output = self.rank_output_for(rank)
            return [rank_output] if rank_output.exists() else []

        outputs = []
        base_output = self.rank_output_for(rank, part_suffix="")
        if base_output.exists():
            outputs.append(base_output)

        pattern = f"{self.rank_output_base.stem}_rank{rank}_*{self.rank_output_base.suffix}"
        for path in sorted(self.rank_output_base.parent.glob(pattern)):
            if path not in outputs:
                outputs.append(path)
        return outputs

    def _writer_loop(self):
        """Background thread: consume queue, format and write to file."""
        with open(self.rank_output, 'w') as f:
            while not self._stop_event.is_set() or not self.write_queue.empty():
                try:
                    # Get batch from queue (timeout to check stop_event)
                    batch = self.write_queue.get(timeout=0.1)
                    if batch is None:  # Sentinel value
                        break
                    
                    # Format to string (in background thread)
                    batch_str = self._format_batch(**batch)
                    
                    # Write to file
                    f.write(batch_str)
                    f.flush()
                    self._total_written += batch_str.count('\n')
                    self.write_queue.task_done()
                    
                except queue.Empty:
                    continue
    
    def open(self):
        """Start the background writer thread."""
        if self.sync_writes:
            self._file = open(self.rank_output, 'w')
            return

        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        
    def close(self):
        """Stop the background thread and wait for completion."""
        if self.sync_writes:
            if self._file is not None:
                self._file.close()
                self._file = None
            return

        # Signal thread to stop
        self._stop_event.set()
        # Wait for queue to empty
        self.write_queue.join()
        # Wait for thread to finish
        if self.writer_thread:
            self.writer_thread.join(timeout=5.0)
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def merge_outputs(self, is_main_process: bool = True) -> None:
        """Merge all rank-specific output files into one."""
        if not is_main_process:
            return

        if self.gzip_output:
            self._merge_outputs_gzip(remove_temp=True)
            return

        if self.part_suffix or self.completed_read_ids is not None:
            self._merge_outputs_plain(remove_temp=True)
            return

        merge_rank_files(
            output_path=self.output_path,
            num_processes=self.num_processes,
            remove_temp=True,
            verbose=True
        )

    def _copy_rank_output_text(self, infile_path: Path, outfile, remove_temp: bool) -> None:
        """Copy one rank TSV file, optionally filtering to completed reads."""
        if self.completed_read_ids is None:
            with open(infile_path, 'r', encoding='utf-8') as infile:
                shutil.copyfileobj(infile, outfile)
        else:
            with open(infile_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    if is_completed_tsv_line(line, self.completed_read_ids):
                        outfile.write(line)

        if remove_temp:
            infile_path.unlink()

    def _merge_outputs_plain(self, remove_temp: bool = True) -> None:
        """Merge rank-specific TSV files into a single plain-text output."""
        with open(self.output_path, 'w', encoding='utf-8') as outfile:
            for rank in range(self.num_processes):
                for rank_file in self.rank_outputs_for_merge(rank):
                    self._copy_rank_output_text(rank_file, outfile, remove_temp)

    def _merge_outputs_gzip(self, remove_temp: bool = True) -> None:
        """Merge rank-specific TSV files into a single gzip-compressed output."""
        mode = 'wt' if self.completed_read_ids is not None else 'wb'
        with gzip.open(self.output_path, mode) as outfile:
            for rank in range(self.num_processes):
                for rank_file in self.rank_outputs_for_merge(rank):
                    if self.completed_read_ids is not None:
                        self._copy_rank_output_text(rank_file, outfile, remove_temp)
                        continue
                    with open(rank_file, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
                    if remove_temp:
                        rank_file.unlink()
    
    def _format_batch(self, preds: torch.Tensor, methy: torch.Tensor,
                     read_ids: List[str], chrs: List[str], strands: List[str],
                     ref_pos: List[List[int]], read_pos: List[List[int]],
                     labels: List[List[int]], patch_pos: List[List[int]]) -> str:
        """Format a batch to TSV string (fast string builder)."""
        lines = []
        idx = 0
        
        # Convert tensors to lists once
        preds_list = preds.tolist()
        methy_list = methy.tolist()
        
        for i in range(len(read_ids)):
            patch_pos_i = patch_pos[i]
            for j in range(len(patch_pos_i)):
                prob1 = preds_list[idx]
                prob0 = 1 - prob1
                probbool = 1 if prob1 > 0.5 else 0
                
                # Use f-string for faster formatting
                line = f"{chrs[i]}\t{ref_pos[i][j]}\t{strands[i]}\t{labels[i][j]}\t{read_ids[i]}\t{read_pos[i][j]}\t{vocab[methy_list[idx]]}\t{prob0:.6f}\t{prob1:.6f}\t{probbool}\t.\n"
                lines.append(line)
                idx += 1
        
        return ''.join(lines)
    
    def write_batch(self, **kwargs) -> int:
        """
        Queue a batch for async writing.
        
        Returns immediately (non-blocking).
        Returns number of samples queued.
        """
        # Move tensors to CPU (fast), but don't format yet
        # Formatting will be done in background thread
        cpu_batch = {
            'preds': kwargs['preds'].cpu(),
            'methy': kwargs['methy'].cpu(),
            'read_ids': kwargs['read_ids'],
            'chrs': kwargs['chrs'],
            'strands': kwargs['strands'],
            'ref_pos': kwargs['ref_pos'],
            'read_pos': kwargs['read_pos'],
            'labels': kwargs['labels'],
            'patch_pos': kwargs['patch_pos'],
        }

        if self.sync_writes:
            batch_str = self._format_batch(**cpu_batch)
            self._file.write(batch_str)
            self._file.flush()
            self._total_written += batch_str.count('\n')
            return sum(len(p) for p in cpu_batch['patch_pos'])
        
        # Queue for background processing and writing
        try:
            self.write_queue.put(cpu_batch, block=True, timeout=1.0)
        except queue.Full:
            # Queue full, format and write synchronously as fallback
            batch_str = self._format_batch(**cpu_batch)
            with open(self.rank_output, 'a') as f:
                f.write(batch_str)
        
        # Estimate samples
        return sum(len(p) for p in cpu_batch['patch_pos'])
