"""
BAM writer for calibration annotation.

Writes pre-computed labels (read-level or site-level) to BAM with MM/ML tags.
Used in the calibration pipeline (annotator, bed_to_bam).

Label formats:
- Read-level: {read_id: [(chr, pos, score), ...]}  where pos is a reference position
- Site-level: {(chr, pos): score}
"""
import os
import multiprocessing
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
import pysam

from unimeth.utils.bam_tags import write_mm_ml_tags
from unimeth.utils.common import get_cpu_count

ReadLabels = Dict[str, List[Tuple[str, int, float]]]
SiteLabels = Dict[Tuple[str, int], float]


class LabelBAMWriter:
    """
    Multi-process BAM writer that annotates reads with pre-computed labels.

    Supports both read-level and site-level label formats.
    """

    def __init__(self, bam_dir: str, out_folder: str, mapq_thres: int = 10):
        self.bam_dir = bam_dir
        self.out_folder = out_folder
        os.makedirs(out_folder, exist_ok=True)
        self.mapq_thres = mapq_thres
        self.num_process = get_cpu_count()

    def _should_process_read(self, bam_read) -> bool:
        if bam_read is None:
            return False
        if bam_read.is_supplementary or bam_read.is_secondary:
            return False
        if bam_read.get_forward_sequence() is None:
            return False
        if bam_read.mapping_quality < self.mapq_thres:
            return False
        return True

    def _is_read_level_labels(self, labels: Union[ReadLabels, SiteLabels]) -> bool:
        if not labels:
            return False
        first_key = next(iter(labels.keys()))
        return isinstance(first_key, str)

    def _ref_pos_to_base_index(self, bam_read, ref_positions: List[int], base_char: str) -> Tuple[List[int], List[int]]:
        """
        Convert reference positions to base-type indices within the forward read sequence.

        MM/ML tags require skip counts between modified bases, which are computed as
        the index of each modified base within all same-type bases in the read.

        Returns (base_indices, matched_order) where base_indices[i] is the 0-based
        index of the i-th matched ref_pos within all base_char occurrences in fwd_seq.
        """
        fwd_seq = bam_read.get_forward_sequence()
        if fwd_seq is None:
            return [], []
        fwd_seq = fwd_seq.upper()

        # Map: query_pos → index within all base_char occurrences in fwd_seq
        all_fwd_positions = [i for i, c in enumerate(fwd_seq) if c == base_char]
        fwd_pos_to_base_idx = {pos: idx for idx, pos in enumerate(all_fwd_positions)}

        is_rev = bam_read.is_reverse
        seq_len = len(bam_read.query_sequence)

        # ref_pos → query_pos mapping
        ref_to_query = {r: q for q, r in bam_read.get_aligned_pairs(matches_only=True) if r is not None}

        base_indices = []
        order = []  # original index in ref_positions for score alignment

        for orig_idx, ref_pos in enumerate(ref_positions):
            if ref_pos not in ref_to_query:
                continue
            query_pos = ref_to_query[ref_pos]
            # Convert query_pos to fwd_pos
            fwd_pos = (seq_len - 1 - query_pos) if is_rev else query_pos
            if fwd_pos in fwd_pos_to_base_idx:
                base_indices.append(fwd_pos_to_base_idx[fwd_pos])
                order.append(orig_idx)

        return base_indices, order

    def write_to_bam(self, pid: int, labels: Union[ReadLabels, SiteLabels]):
        """Write annotated reads to BAM file (worker process)."""
        out_path = os.path.join(self.out_folder, f'{pid}.bam')
        is_read_level = self._is_read_level_labels(labels)

        with pysam.AlignmentFile(self.bam_dir, "rb") as in_bam, \
             pysam.AlignmentFile(out_path, "wb", header=in_bam.header, add_sam_header=False) as out_bam:

            for idx, bam_read in tqdm(enumerate(in_bam), disable=(pid != 0)):
                if idx % self.num_process != pid:
                    continue
                if not self._should_process_read(bam_read):
                    continue

                if is_read_level:
                    read_id = bam_read.query_name
                    if read_id not in labels:
                        continue
                    ref_positions = [pos for _, pos, _ in labels[read_id]]
                    raw_scores = [score for _, _, score in labels[read_id]]
                else:
                    chr_name = bam_read.reference_name
                    if chr_name is None:
                        continue
                    ref_positions = []
                    raw_scores = []
                    for _, ref_pos in bam_read.get_aligned_pairs(matches_only=True):
                        if ref_pos is not None and (chr_name, ref_pos) in labels:
                            ref_positions.append(ref_pos)
                            raw_scores.append(labels[(chr_name, ref_pos)])

                if not ref_positions:
                    continue

                # All label modifications are C-type (CpG/CHG/CHH) in calibration
                base_indices, order = self._ref_pos_to_base_index(bam_read, ref_positions, 'C')
                if not base_indices:
                    continue

                scores = [raw_scores[i] for i in order]
                if write_mm_ml_tags(bam_read, base_indices, scores):
                    out_bam.write(bam_read)

    def write_multi_process(self, labels: Union[ReadLabels, SiteLabels]):
        """Write BAM files using multiple processes."""
        args_list = [(pid, labels) for pid in range(self.num_process)]
        processes = []
        for args in args_list:
            p = multiprocessing.Process(target=self.write_to_bam, args=args)
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
