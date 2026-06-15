"""
Aggregation BAM writer for inference results.

Writes methylation predictions to BAM format with MM/ML tags.
Features:
- Per-read aggregation: collects all patches for a read before writing
- Bitmap tracking: uses patch_idx/total_patches to detect completeness
- Periodic flush: flushes oldest reads when buffer is full
- Independent per-rank operation: no inter-process communication needed
"""
import os
import time
import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


import pysam
import numpy as np
import torch

from unimeth.config import tokenizer, methy_types
from unimeth.ioutils.reader.bam import BamReader
from unimeth.utils.bam_tags import write_mm_ml_tags

logger = logging.getLogger(__name__)


@dataclass
class PatchPrediction:
    """Single patch prediction data."""
    prob: float  # Methylation probability (0-1)
    methy_type: str  # Token like '[CpG]', '[CHG]', etc.
    read_pos: int
    ref_pos: int
    chr: str
    strand: str
    patch_idx: int  # Index within read


@dataclass
class ReadBuffer:
    """Buffer for aggregating patches of a single read."""
    read_id: str
    expected: int  # Total patches expected
    received: Dict[int, List[PatchPrediction]] = field(default_factory=lambda: defaultdict(list))
    create_time: float = field(default_factory=time.time)
    
    @property
    def is_complete(self) -> bool:
        return len(self.received) == self.expected
    
    def add_patch(self, patch_idx: int, pred: PatchPrediction):
        """Add predictions for a patch. A patch may contain multiple sites."""
        self.received[patch_idx].append(pred)


class AggregationBAMWriter:
    """
    BAM writer with per-read aggregation support.
    
    Each rank maintains its own buffer. Reads are aggregated until all patches
    are received, then written to BAM with MM/ML tags.
    
    This writer relies on the Dataset to signal when reads are complete via
    __reads_complete__ markers, rather than using internal buffer limits.
    
    Args:
        output_path: Path to output BAM file (should include rank info, e.g., "output.part_0.bam")
        template_bam_path: Path to template BAM for header copying
        bam_reader: BamReader instance for fetching original reads
    """
    
    def __init__(
        self,
        output_path: str,
        template_bam_path: str,
        bam_reader: BamReader,
    ):
        self.output_path = output_path
        self.bam_reader = bam_reader
        
        # Initialize output BAM with template header
        template = pysam.AlignmentFile(template_bam_path, "rb")
        self.output_bam = pysam.AlignmentFile(output_path, "wb", header=template.header)
        template.close()
        
        # Buffer state: OrderedDict maintains insertion order
        self.buffer: Dict[str, ReadBuffer] = OrderedDict()
        
        # Statistics
        self.stats = {
            'reads_completed': 0,
            'reads_flushed_incomplete': 0,
            'patches_received': 0,
            'patches_dropped': 0,
        }
        
        # Token to methylation type mapping
        self.token_to_type = {tokenizer[t]: t for t in methy_types}
    
    def write_batch(
        self,
        preds: torch.Tensor,
        methy: torch.Tensor,
        read_ids: List[str],
        chrs: List[str],
        strands: List[str],
        ref_pos: List[List[int]],
        read_pos: List[List[int]],
        patch_pos: List[List[int]],
        patch_idx: List[int],
        total_patches: List[int],
    ) -> int:
        """
        Process a batch of predictions.
        
        Args:
            preds: Tensor of shape [N_sites] with methylation probabilities
            methy: Tensor of shape [N_sites] with methylation type token IDs
            read_ids, chrs, strands: Per-read metadata
            ref_pos, read_pos, patch_pos: Per-read lists of positions
            patch_idx, total_patches: Per-read patch indices
            
        Returns:
            Number of sites processed
        """
        # Move to CPU for processing
        preds_np = preds.cpu().numpy()
        methy_np = methy.cpu().numpy()
        
        # Flatten the per-read structure
        idx = 0
        sites_processed = 0
        
        for i, read_id in enumerate(read_ids):
            num_sites = len(patch_pos[i])

            # Initialize buffer for new read
            if read_id not in self.buffer:
                self.buffer[read_id] = ReadBuffer(read_id, total_patches[i])

            buf = self.buffer[read_id]
            p_idx = patch_idx[i] if isinstance(patch_idx[i], int) else patch_idx[i][0]

            if num_sites == 0:
                # Empty patch: mark as received so it counts toward completeness
                buf.received[p_idx]  # defaultdict auto-creates empty list
                if buf.is_complete:
                    self._flush_complete_read(read_id)
                continue

            # Add all sites from this patch
            for j in range(num_sites):
                pred_data = PatchPrediction(
                    prob=float(preds_np[idx]),
                    methy_type=self.token_to_type.get(int(methy_np[idx]), '[CpG]'),
                    read_pos=int(read_pos[i][j]),
                    ref_pos=int(ref_pos[i][j]),
                    chr=chrs[i],
                    strand=strands[i],
                    patch_idx=p_idx,
                )
                buf.add_patch(p_idx, pred_data)
                idx += 1
                sites_processed += 1

            # Check if read is complete
            if buf.is_complete:
                self._flush_complete_read(read_id)
        
        return sites_processed
    
    def on_reads_complete(self):
        """
        Called when Dataset signals that a batch of reads is complete.
        Only flushes reads that have received all expected patches; incomplete
        reads stay in the buffer to receive their remaining patches from the
        next bin flush before the next on_reads_complete call.
        """
        if not self.buffer:
            return

        for read_id in list(self.buffer.keys()):
            if self.buffer[read_id].is_complete:
                self._flush_complete_read(read_id)
    
    def _flush_complete_read(self, read_id: str):
        """Flush a complete read to BAM."""
        if read_id not in self.buffer:
            return
        
        buf = self.buffer.pop(read_id)

        # Generate MM/ML and write
        self._write_read_to_bam(buf)
        self.stats['reads_completed'] += 1
    
    def _extract_positions_scores(
        self,
        preds: list,
        base_char: str,
        fwd_seq: str,
        bam_read
    ) -> tuple:
        """
        Extract base-type indices and scores for predictions of a specific base type.

        The MM tag encodes skip counts between modified bases, where each skip is the
        number of unmodified same-type bases. So positions must be the 0-based index
        of each modified base within the list of all same-type bases in the read.

        Args:
            preds: List of PatchPrediction objects
            base_char: Base character ('C' or 'A')
            fwd_seq: Forward sequence of the read
            bam_read: pysam AlignedSegment

        Returns:
            Tuple of (base_indices, scores): base_indices[i] is the position of the
            i-th modified base within all occurrences of base_char in fwd_seq.
        """
        # Find all forward-sequence positions of base_char
        all_fwd_positions = [i for i, c in enumerate(fwd_seq) if c == base_char]
        if not all_fwd_positions:
            return [], []

        pred_by_read_pos = {p.read_pos: p for p in preds if p.read_pos is not None}

        base_indices = []
        scores = []

        for base_idx, read_pos in enumerate(all_fwd_positions):
            if read_pos in pred_by_read_pos:
                base_indices.append(base_idx)
                scores.append(pred_by_read_pos[read_pos].prob * 100)

        return base_indices, scores

    def _write_read_to_bam(self, buf: ReadBuffer):
        """Write a read's aggregated predictions to BAM."""
        # Get original BAM read
        bam_reads = self.bam_reader.get_read_by_id(buf.read_id)
        if not bam_reads:
            logger.warning(f"Read {buf.read_id} not found in BAM")
            return
        
        bam_read = bam_reads[0]  # Take primary alignment
        
        # Get forward sequence (original orientation)
        fwd_seq = bam_read.get_forward_sequence()
        if fwd_seq is None:
            return
        fwd_seq = fwd_seq.upper()
        
        # MM/ML tags are read-coordinate, so unaligned reads can still be written.
        # if bam_read.reference_name is None:
        #     return

        # Collect all predictions, grouped by canonical base type (C vs A).
        c_preds: list = []   # CpG + CHG + CHH
        a_preds: list = []   # m6A
        for patch_idx in sorted(buf.received.keys()):
            for pred in buf.received[patch_idx]:
                if pred.methy_type == '[m6A]':
                    a_preds.append(pred)
                else:
                    c_preds.append(pred)

        # Generate and write MM/ML tags for each base type
        written = False
        
        # Process C modifications (CpG, CHG, CHH)
        if c_preds:
            c_positions, c_scores = self._extract_positions_scores(
                c_preds, 'C', fwd_seq, bam_read
            )
            if c_positions and write_mm_ml_tags(bam_read, c_positions, c_scores, 'C+m,'):
                written = True
        
        # Process A modifications (m6A)
        if a_preds:
            a_positions, a_scores = self._extract_positions_scores(
                a_preds, 'A', fwd_seq, bam_read
            )
            if a_positions and write_mm_ml_tags(bam_read, a_positions, a_scores, 'A+a,'):
                written = True
        
        if written:
            self.output_bam.write(bam_read)
    
    def close(self):
        """Close writer and flush remaining reads."""
        logger.info(f"Closing BAM writer, flushing {len(self.buffer)} remaining reads")
        
        # Flush all remaining reads
        for read_id in list(self.buffer.keys()):
            buf = self.buffer[read_id]
            if not buf.is_complete:
                logger.warning(
                    f"At close, read {read_id} incomplete: "
                    f"{len(buf.received)}/{buf.expected}"
                )
                self.stats['reads_flushed_incomplete'] += 1
            self._write_read_to_bam(buf)
        
        self.buffer.clear()
        self.output_bam.close()
        
        # Log statistics
        logger.info(
            f"BAM writer stats: completed={self.stats['reads_completed']}, "
            f"incomplete={self.stats['reads_flushed_incomplete']}"
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
