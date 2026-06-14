"""
BED file reader for bisulfite methylation data.

Provides BEDReader for loading WGBS bisulfite labels.
"""
import os
from typing import Dict, Optional

from unimeth.data.coords import parse_chromosome_filter


class BEDReader:
    """
    BED file reader for bisulfite methylation labels.
    
    Reads WGBS data in BED format (11 columns):
    chrom, start, end, name, score, strand, thickStart, thickEnd,
    itemRgb, coverage, percentage
    """
    
    def __init__(self, file_path: str):
        """
        Initialize BED reader.
        
        Args:
            file_path: Path to BED file
        """
        self.file_path = file_path
    
    def load_labels(
        self,
        coverage_thres: int = 5,
        chr_str: str = '|'
    ) -> Optional[Dict[str, float]]:
        """
        Load bisulfite labels from BED file.
        
        Args:
            coverage_thres: Minimum coverage threshold
            chr_str: Chromosome filter string (e.g., "|" for all, "|Chr1,Chr2" for include)
            
        Returns:
            Dictionary mapping chr_pos to methylation score (0-100), or None if file not found
        """
        if self.file_path is None or not os.path.exists(self.file_path):
            print('No data')
            return None
        
        chr_mode, chr_list = parse_chromosome_filter(chr_str)
        if len(chr_list) > 0:
            print(f"Chromosome filter: {chr_mode} {chr_list}")
        
        cnt_exclude, cnt_not_cover = 0, 0
        bisulfite: Dict[str, float] = {}
        
        print(f'Loading bisulfite labels from {self.file_path}...')
        
        with open(self.file_path, 'r') as fr:
            for line in fr:
                if not line or line.startswith('#') or line.strip() == '':
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 11:
                    continue
                
                chrom = fields[0]
                
                # Apply chromosome filter
                if chr_mode == 'exclude' and chrom in chr_list:
                    cnt_exclude += 1
                    continue
                elif chr_mode == 'include' and chrom not in chr_list:
                    cnt_exclude += 1
                    continue
                
                start = int(fields[1])
                name = f'{chrom}_{start}'
                coverage = int(fields[9])
                
                if coverage < coverage_thres:
                    cnt_not_cover += 1
                    continue
                
                score = float(fields[10])
                if name in bisulfite:
                    print(f'Warning: Duplicate entry for {name}')
                    continue
                
                bisulfite[name] = score
        
        print(f'Read {len(bisulfite)} rows. Excluded {cnt_exclude} rows. '
              f'{cnt_not_cover} rows below coverage threshold.')
        return bisulfite
