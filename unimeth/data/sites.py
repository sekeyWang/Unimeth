"""
Methylation site detection and handling.

Provides functions for finding methylation sites (CpG, CHG, CHH, m6A)
and determining their types.
"""


def find_methylation_sites(seq: str, detect_cpg: int, detect_chg: int,
                           detect_chh: int, detect_m6a: int) -> list:
    """
    Find positions of methylation sites in a sequence.
    
    Identifies CpG, CHG, CHH, and m6A sites based on configuration flags.
    
    Args:
        seq: DNA sequence
        detect_cpg: Enable CpG detection (1=yes)
        detect_chg: Enable CHG detection (1=yes)
        detect_chh: Enable CHH detection (1=yes)
        detect_m6a: Enable m6A detection (1=yes)
        
    Returns:
        List of positions (0-indexed) where methylation can occur
    """
    pred_pos = []
    for i in range(len(seq)):
        if seq[i] == 'C':
            if i + 1 < len(seq) and seq[i+1] == 'G' and detect_cpg:
                pred_pos.append(i)
            elif i + 2 < len(seq) and seq[i+2] == 'G' and detect_chg:
                pred_pos.append(i)
            elif i + 2 < len(seq) and detect_chh:
                pred_pos.append(i)
        elif seq[i] == 'A':
            if detect_m6a:
                pred_pos.append(i)
    return pred_pos


def get_methy_type(seq: str, pos: int, detect_cpg: int, detect_chg: int,
                   detect_chh: int, detect_m6a: int) -> str | None:
    """
    Get methylation type at a position.
    
    Args:
        seq: DNA sequence
        pos: Position in sequence
        detect_cpg: Enable CpG detection
        detect_chg: Enable CHG detection
        detect_chh: Enable CHH detection
        detect_m6a: Enable m6A detection
        
    Returns:
        Methylation type token ('[CpG]', '[CHG]', '[CHH]', '[m6A]') or None
    """
    if seq[pos] == 'C':
        if pos + 1 < len(seq) and seq[pos+1] == 'G':
            return '[CpG]' if detect_cpg else None
        elif pos + 2 < len(seq) and seq[pos+2] == 'G':
            return '[CHG]' if detect_chg else None
        elif pos + 2 < len(seq):
            return '[CHH]' if detect_chh else None
    elif seq[pos] == 'A':
        return '[m6A]' if detect_m6a else None
    return None
