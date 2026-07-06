"""
Feature extraction from nanopore signal and BAM alignment data.

Provides classes and functions for extracting methylation sites and 
raw signal features from POD5 and BAM files.
"""
import numpy as np

from .coords import parse_chromosome_filter, complement_seq, get_ref_pos, align_to_ref
from .sites import find_methylation_sites


class SignalFeatureExtractor:
    """
    Extractor for raw nanopore signal features.
    
    Extracts signal events and methylation labels from POD5 and BAM data.
    """
    
    def __init__(self, args):
        """
        Initialize raw feature extractor.
        
        Args:
            args: Arguments with mapq_thres, pore_type, chr, and methylation flags
        """
        self.mapq_thres = getattr(args, 'mapq_thres', 10)
        self.align_ref = (getattr(args, 'pore_type', 'R10.4.1') == 'R9.4.1')
        self.chr_mode, self.chr_list = parse_chromosome_filter(getattr(args, 'chr', '|'))
        
        # Store methylation detection flags
        self.detect_cpg = getattr(args, 'cpg', 0)
        self.detect_chg = getattr(args, 'chg', 0)
        self.detect_chh = getattr(args, 'chh', 0)
        self.detect_m6a = getattr(args, 'm6A', 0)
        
        # Determine modification type
        if self.detect_m6a == 1:
            self.detect_mod = ('A', 0, 'a')
        elif self.detect_cpg or self.detect_chg or self.detect_chh:
            self.detect_mod = ('C', 0, 'm')
        else:
            self.detect_mod = None
    
    def get_signal(self, signal, move):
        """
        Extract signal events from raw signal using move table.
        
        Args:
            signal: Raw signal array
            move: Tuple of (stride, start, move_table)
            
        Returns:
            List of signal segments per base
        """
        stride, start, move_table = move
        move_index = np.where(move_table)[0]
        if len(move_index) == 0:
            return []
        
        boundaries = move_index * stride + start
        boundaries = np.append(boundaries, len(signal))
        
        # Vectorized split using numpy (avoids Python loop overhead)
        event = [signal[boundaries[i]:boundaries[i+1]] for i in range(len(move_index))]
        return event
    
    def get_move(self, bam_read):
        """
        Extract move table from BAM read tags.
        
        Args:
            bam_read: pysam AlignedSegment
            
        Returns:
            Tuple of (stride, start, move_table)
        """
        start = bam_read.get_tag('ts')
        if bam_read.has_tag('sp'):
            start += bam_read.get_tag('sp')
        
        mv = bam_read.get_tag('mv')
        stride = mv[0]
        move_table = np.fromiter(mv[1:], dtype=np.int8)
        
        return stride, start, move_table
    
    def get_feature(self, bam_read, pod5_read):
        """
        Extract features from a BAM read and POD5 read pair.
        
        Args:
            bam_read: pysam AlignedSegment
            pod5_read: POD5 read object
            
        Returns:
            Dictionary with extracted features or None if filtered out
        """
        is_unmapped = getattr(bam_read, 'is_unmapped', False) or bam_read.reference_name is None

        # Filter by mapping quality and chromosome only for aligned reads.
        mapq = bam_read.mapping_quality
        if not is_unmapped and mapq < self.mapq_thres:
            return None

        chrom = bam_read.reference_name if not is_unmapped else '*'
        if not is_unmapped:
            if self.chr_mode == 'exclude' and chrom in self.chr_list:
                return None
            elif self.chr_mode == 'include' and chrom not in self.chr_list:
                return None
        
        # Get sequence and signal
        seq = bam_read.get_forward_sequence().upper()
        signal = pod5_read.signal

        # Calibration parameters
        shift_dacs_to_pa = pod5_read.calibration.offset
        scale_dacs_to_pa = pod5_read.calibration.scale
        shift_pa_to_norm = bam_read.get_tag("sm")
        scale_pa_to_norm = bam_read.get_tag("sd")

        # Extract signal events
        move = self.get_move(bam_read)
        signal_event = self.get_signal(signal, move)

        from unimeth.utils.bam_tags import get_modifications
        mod_dict = get_modifications(bam_read, self.detect_mod) if self.detect_mod else {}

        if is_unmapped:
            pred_pos = find_methylation_sites(
                seq, self.detect_cpg, self.detect_chg, self.detect_chh, self.detect_m6a
            )
            ref_pos = [-1] * len(pred_pos)
        else:
            # Get alignment and modification info
            aligned_pairs = bam_read.get_aligned_pairs()

            # Reverse alignment if on reverse strand
            if bam_read.is_reverse:
                aligned_pairs = aligned_pairs[::-1]

            # Align to reference for R9.4.1
            if self.align_ref:
                result = align_to_ref(bam_read, seq, signal_event, aligned_pairs, mod_dict)
                if result is None:
                    return None
                seq, signal_event, aligned_pairs, mod_dict = result

            # Find methylation positions
            pred_pos = find_methylation_sites(
                seq, self.detect_cpg, self.detect_chg, self.detect_chh, self.detect_m6a
            )
            ref_pos = get_ref_pos(seq, pred_pos, aligned_pairs, bam_read.is_reverse)
        
        # Get labels
        bis_label = []
        for pos in pred_pos:
            if pos in mod_dict:
                bis_label.append(int(round(mod_dict[pos] / 255 * 100)))
            else:
                bis_label.append(-1)
        
        return {
            'read_id': bam_read.query_name,
            'chr': chrom,
            'reference_start': bam_read.reference_start,
            'bases': list(seq),
            'signal_event': signal_event,
            'pred_pos': pred_pos,
            'bis_label': bis_label,
            'mapQ': mapq,
            'shift_dacs_to_pa': shift_dacs_to_pa,
            'scale_dacs_to_pa': scale_dacs_to_pa,
            'shift_pa_to_norm': shift_pa_to_norm,
            'scale_pa_to_norm': scale_pa_to_norm,
            'ref_pos': ref_pos,
            'strand': '-' if bam_read.is_reverse else '+',
        }
