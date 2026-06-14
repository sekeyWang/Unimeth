"""
Coordinate transformation and chromosome filtering utilities.

Provides functions for sequence complementation, reverse complementation,
coordinate transformations, and chromosome filtering.
"""


# DNA base pairing
BASE_PAIRS = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
}


def complement_base(letter: str) -> str:
    """Get the complement of a single DNA base."""
    return BASE_PAIRS.get(letter, "N")


def complement_seq(base_seq: str) -> str:
    """
    Get the reverse complement of a DNA sequence.
    
    Args:
        base_seq: Input DNA sequence
        
    Returns:
        Reverse complement sequence
    """
    rbase_seq = base_seq[::-1]
    try:
        comseq = "".join([complement_base(x) for x in rbase_seq])
    except Exception:
        print("something wrong in the dna/rna sequence.")
        comseq = ""
    return comseq


def get_ref_pos(seq, pred_pos, aligned_pairs, is_reverse):
    """
    Map read positions to reference positions.
    
    Given read sequence, aligned_pairs and strand information,
    output reference positions for predicted sites.
    
    Args:
        seq: Read sequence
        pred_pos: Positions of interest in the read (0-indexed)
        aligned_pairs: List of (query_pos, ref_pos) tuples from BAM
        is_reverse: Whether read is on reverse strand
        
    Returns:
        List of reference positions corresponding to pred_pos
    """
    ref_pos = []
    if len(pred_pos) == 0 or len(aligned_pairs) == 0:
        return [-1] * len(pred_pos)
    
    idx = 0
    for ps, pr in aligned_pairs:
        if ps is None:
            continue
        
        # Adjust position for reverse strand
        ps = len(seq) - ps - 1 if is_reverse else ps
        
        if ps == pred_pos[idx]:
            ref_pos.append(pr if pr is not None else -1)
            idx += 1
        elif ps > pred_pos[idx]:
            ref_pos.append(-1)
            idx += 1
        
        if idx == len(pred_pos):
            break
    
    assert len(ref_pos) == len(pred_pos), f'{len(ref_pos)} and {len(pred_pos)}'
    return ref_pos


def align_to_ref(bam_read, seq, signal_event, aligned_pairs, mod_dict):
    """
    Align read sequence and signal to reference coordinates.
    
    For R9.4.1 chemistry, this maps the read to reference positions
    to handle potential signal drift.
    
    Args:
        bam_read: pysam AlignedSegment
        seq: Read sequence
        signal_event: List of signal events per base
        aligned_pairs: List of (query_pos, ref_pos) tuples
        mod_dict: Dictionary of modified base positions
        
    Returns:
        Tuple of (ref_seq, ref_signal_event, ref_aligned_pairs, ref_mod_dict)
        or None if reference sequence unavailable
    """
    import numpy as np
    
    try:
        ref_seq = bam_read.get_reference_sequence().upper()
    except:
        return None
    
    if bam_read.is_reverse:
        ref_seq = complement_seq(ref_seq)
    
    ref_start = bam_read.reference_start
    first_index = next((i for i in range(len(aligned_pairs)) if aligned_pairs[i][1] is not None), len(aligned_pairs))
    last_index = next((i for i in range(len(aligned_pairs)-1, 0, -1) if aligned_pairs[i][1] is not None), len(aligned_pairs))
    
    new_ref_seq, new_ref_signal, new_ref_pos = [], [], []
    new_mod_dict = {}
    
    for i, (ps, pr) in enumerate(aligned_pairs):
        if ps is not None and bam_read.is_reverse:
            ps = len(seq) - ps - 1
        
        signal = [] if ps is None or signal_event is None else signal_event[ps]
        
        if i < first_index or i > last_index:
            if ps is not None:
                if ps in mod_dict:
                    new_mod_dict[len(new_ref_seq)] = mod_dict[ps]
                new_ref_seq.append(seq[ps])
                new_ref_signal.append(signal)
                new_ref_pos.append(pr)
        else:
            if pr is not None:
                if bam_read.is_reverse:
                    ref_base = ref_seq[len(ref_seq) - (pr - ref_start) - 1]
                else:
                    ref_base = ref_seq[pr - ref_start]
                
                if ps in mod_dict:
                    new_mod_dict[len(new_ref_seq)] = mod_dict[ps]
                
                new_ref_seq.append(ref_base)
                new_ref_signal.append(signal)
                new_ref_pos.append(pr)
            elif len(new_ref_signal) > 0:
                new_ref_signal[-1] = np.append(new_ref_signal[-1], signal)
    
    seq = ''.join(new_ref_seq)
    signal_event = new_ref_signal
    
    if bam_read.is_reverse:
        aligned_pairs = [(len(new_ref_seq) - 1 - i, new_ref_pos[i]) for i in range(len(new_ref_seq))]
    else:
        aligned_pairs = [(i, new_ref_pos[i]) for i in range(len(new_ref_seq))]
    
    return seq, signal_event, aligned_pairs, new_mod_dict


def parse_chromosome_filter(chr_str: str):
    """
    Parse chromosome filter string into mode and list.
    
    Format:
        - "|" or "": Use all chromosomes
        - "Chr1,Chr2|": Exclude Chr1 and Chr2
        - "|Chr1,Chr2": Include only Chr1 and Chr2
    
    Args:
        chr_str: Filter string with format "exclude_list|include_list"
        
    Returns:
        Tuple of (chr_mode, chr_list) where:
        - chr_mode: 'exclude' or 'include'
        - chr_list: Set of chromosome names
    """
    if '|' not in chr_str:
        print('error, default as using all chr')
        return 'exclude', set()
    
    parts = chr_str.split('|')
    
    if parts[0] != '' and parts[1] != '':
        print('error, default as using all chr')
        return 'exclude', set()
    elif parts[0] == '' and parts[1] == '':
        chr_mode, chr_list = 'exclude', set()
    elif parts[0] != '':
        chr_mode, chr_list = 'exclude', set(parts[0].split(','))
    else:
        chr_mode, chr_list = 'include', set(parts[1].split(','))
    
    return chr_mode, chr_list
