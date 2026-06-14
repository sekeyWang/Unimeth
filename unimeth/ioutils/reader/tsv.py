"""
TSV prediction result file reader.

Provides TSVReader for reading model prediction results.
"""
import os
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple


class PredictionRecord(NamedTuple):
    """Single prediction record from TSV file."""
    chrom: str
    ref_pos: int
    strand: str
    label: int
    read_id: str
    read_pos: int
    methy_type: str
    prob_unmethylated: float
    prob_methylated: float
    prediction: float


class TSVReader:
    """
    TSV prediction result file reader.
    
    Reads model prediction results in TSV format (11 columns).
    """
    
    def __init__(self, file_path: str):
        """
        Initialize TSV reader.
        
        Args:
            file_path: Path to TSV file
        """
        self.file_path = file_path
        self.file_handle = None
    
    def open(self):
        """Open the file for reading."""
        if self.file_path and os.path.exists(self.file_path):
            self.file_handle = open(self.file_path, 'r')
        return self
    
    def close(self):
        """Close the file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def __enter__(self):
        return self.open()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def _make_site_key(self, chrom: str, ref_pos: int) -> str:
        """Create site key from chrom and position."""
        return f'{chrom}_{ref_pos}'
    
    def _iter_with_limit(self, max_lines: Optional[int] = None):
        """Iterate records with optional line limit."""
        for i, record in enumerate(self.iter_records()):
            if max_lines is not None and i >= max_lines:
                break
            yield record
    
    def iter_records(self) -> Iterator[PredictionRecord]:
        """
        Iterate over all valid prediction records.
        
        Yields:
            PredictionRecord objects
        """
        def _parse_line(line: str) -> Optional[PredictionRecord]:
            line = line.strip('\n')
            if not line or line.startswith('#') or line.strip() == '':
                return None
            
            fields = line.split('\t')
            if len(fields) < 10:
                return None
            
            try:
                return PredictionRecord(
                    chrom=fields[0],
                    ref_pos=int(fields[1]),
                    strand=fields[2],
                    label=int(fields[3]),
                    read_id=fields[4],
                    read_pos=int(fields[5]) if fields[5].isdigit() else -1,
                    methy_type=fields[6],
                    prob_unmethylated=float(fields[7]),
                    prob_methylated=float(fields[8]),
                    prediction=float(fields[9])
                )
            except (ValueError, IndexError):
                return None
        
        if self.file_handle is None:
            with open(self.file_path, 'r') as f:
                for line in f:
                    record = _parse_line(line)
                    if record is not None:
                        yield record
        else:
            for line in self.file_handle:
                record = _parse_line(line)
                if record is not None:
                    yield record
    
    def load_site_results(
        self,
        max_lines: int = 200_000_000
    ) -> Tuple[Dict[str, int], Dict[str, List]]:
        """
        Load site-level labels and predictions.
        
        Returns:
            Tuple of (site_label_dict, site_result_dict)
            - site_label: {chr_pos: bisulfite_score} (0 or 100)
            - site_result: {chr_pos: [[read_id, pred], ...]}
        """
        site_label: Dict[str, int] = {}
        site_result: Dict[str, List] = {}
        
        for record in self._iter_with_limit(max_lines):
            if record.label == -1:
                continue
            
            name = self._make_site_key(record.chrom, record.ref_pos)
            
            # Validate consistent labels
            if name in site_label:
                assert site_label[name] == record.label, \
                    f'Label mismatch for {name}: {site_label[name]} vs {record.label}'
            
            site_label[name] = record.label
            
            if name not in site_result:
                site_result[name] = []
            site_result[name].append([record.read_id, record.prediction])
        
        return site_label, site_result
    
    def load_read_results(
        self,
        max_lines: int = 200_000_000
    ) -> Dict[str, Dict[str, float]]:
        """
        Load read-level predictions.
        
        Returns:
            Dictionary: {read_id: {chr_pos: pred}}
        """
        read_result: Dict[str, Dict[str, float]] = {}
        
        for record in self._iter_with_limit(max_lines):
            name = self._make_site_key(record.chrom, record.ref_pos)
            
            if record.read_id not in read_result:
                read_result[record.read_id] = {}
            read_result[record.read_id][name] = record.prediction
        
        return read_result
    
    def load_site_predictions(
        self,
        skip_invalid_pos: bool = True,
        methy_type: str = None
    ) -> Dict[str, List[float]]:
        """
        Load site-level predictions aggregated by genomic position.

        Returns:
            Dictionary mapping chr_pos to list of prob_methylated values
        """
        site: Dict[str, List[float]] = {}

        for record in self.iter_records():
            if skip_invalid_pos and record.ref_pos == -1:
                continue
            if methy_type is not None and record.methy_type != methy_type:
                continue

            name = self._make_site_key(record.chrom, record.ref_pos)
            if name not in site:
                site[name] = []
            site[name].append(record.prob_methylated)

        return site
