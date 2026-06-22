"""
BAM file reader with indexing support.

Provides efficient random access to BAM records and methylation prediction loading.
"""
import os
import pickle
from typing import Dict, List

import pysam

from unimeth.utils import local_print
from unimeth.utils.bam_tags import get_mod_config, get_target_positions
from unimeth.data.coords import get_ref_pos


class BamReader:
    """
    Indexed BAM file reader for random access by read ID.
    
    Automatically builds and caches an index mapping read IDs to file positions
    for efficient retrieval.
    
    Attributes:
        bam_file: Open pysam AlignmentFile handle
        bam_index: Dictionary mapping read_id -> list of file positions
    """
    
    def __init__(self, bam_path: str, force_rebuild_index: bool = False):
        """
        Initialize indexed BAM reader.
        
        Args:
            bam_path: Path to BAM file
            force_rebuild_index: If True, rebuild index even if cached index exists
        """
        self.bam_file = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
        
        filename = bam_path.split('/')[-1]
        os.makedirs('data_cache/bam_index', exist_ok=True)
        bam_index_file = f'data_cache/bam_index/{filename}.index'
        
        if force_rebuild_index or not os.path.exists(bam_index_file):
            self.bam_index = self._build_bam_index(bam_index_file)
        else:
            try:
                with open(bam_index_file, 'rb') as fr:
                    self.bam_index = pickle.load(fr)
            except Exception:
                self.bam_index = self._build_bam_index(bam_index_file)
    

    def _build_bam_index(self, bam_index_file: str) -> dict:
        local_print('Building bam index...')
        self.bam_file.reset()
        bam_index = {}
        read_ptr = self.bam_file.tell()
        
        for bam_read in self.bam_file:
            if bam_read.is_supplementary or bam_read.is_secondary:
                read_ptr = self.bam_file.tell()
                continue
            
            if bam_read.has_tag('pi'):
                # Index by pi (parent read ID) for SignalReader lookup
                pi_id = bam_read.get_tag("pi")
                if pi_id not in bam_index:
                    bam_index[pi_id] = []
                bam_index[pi_id].append(read_ptr)
                # Also index by query_name so BAM writer can find split reads
                qname = bam_read.query_name
                if qname not in bam_index:
                    bam_index[qname] = []
                bam_index[qname].append(read_ptr)
            else:
                read_id = bam_read.query_name
                if read_id not in bam_index:
                    bam_index[read_id] = []
                bam_index[read_id].append(read_ptr)
            read_ptr = self.bam_file.tell()
        
        with open(bam_index_file, 'wb') as fw:
            pickle.dump(bam_index, fw)
        
        local_print(f'Bam index built and saved in {bam_index_file}. Size: {len(bam_index)}')
        return bam_index
    
    def get_read_by_id(self, read_id: str) -> list:
        """
        Get BAM records by read ID.
        
        Args:
            read_id: Read ID to look up
            
        Returns:
            List of pysam AlignedSegment objects
        """
        bam_reads = []
        if read_id not in self.bam_index:
            return bam_reads
        
        read_ptrs = self.bam_index[read_id]
        for read_ptr in read_ptrs:
            self.bam_file.seek(read_ptr)
            bam_read = next(self.bam_file)
            bam_reads.append(bam_read)
        
        return bam_reads
    
    def load_site_predictions(
        self,
        mod_type: str = 'm'
    ) -> Dict[str, List[float]]:
        """
        Load site-level methylation predictions from BAM MM/ML tags.

        Args:
            mod_type: Modification type ('m' for CpG/CHG/CHH, 'a' for m6A)

        Returns:
            Dictionary mapping chr_pos to list of predictions
        """
        from tqdm import tqdm
        from unimeth.utils.bam_tags import get_modifications

        site: Dict[str, List[float]] = {}
        base, mod_key = get_mod_config(mod_type)

        # Sequential scan is much faster than indexed random access
        self.bam_file.reset()
        seen_reads: set = set()
        for bam_read in tqdm(self.bam_file, desc='Reading BAM'):
            if bam_read.is_supplementary or bam_read.is_secondary:
                continue
            if bam_read.query_name in seen_reads:
                continue
            seen_reads.add(bam_read.query_name)
            if bam_read.get_forward_sequence() is None:
                continue
            if not bam_read.has_tag('ML') or len(bam_read.get_tag('ML')) == 0:
                continue

            seq = bam_read.get_forward_sequence().upper()
            pred_pos = get_target_positions(seq, base)

            aligned_pairs = bam_read.get_aligned_pairs()
            if len(aligned_pairs) == 0:
                continue

            if bam_read.is_reverse:
                aligned_pairs = aligned_pairs[::-1]

            ref_pos_list = get_ref_pos(seq, pred_pos, aligned_pairs, bam_read.is_reverse)
            pos_to_ref = {pred_pos[i]: ref_pos_list[i] for i in range(len(pred_pos))}

            chrom = bam_read.reference_name
            mod = get_modifications(bam_read, mod_key)

            for pos, label in mod.items():
                pred_value = label / 255
                ref_pos = pos_to_ref.get(pos, -1)
                if ref_pos == -1:
                    continue

                name = f'{chrom}_{ref_pos}'
                if name not in site:
                    site[name] = []
                site[name].append(pred_value)

        return site

