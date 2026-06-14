"""
Signal reader for extracting features from POD5 and BAM files.

Provides multi-worker capable readers for processing nanopore data.
"""
import os
from torch.utils.data import get_worker_info
from tqdm import tqdm
import pod5 as p5

from unimeth.utils import local_print

# Lazy import PartialState to avoid import-time side effects
def _is_local_main_process():
    """Check if current process is the local main process."""
    try:
        from accelerate.state import PartialState
        return PartialState().is_local_main_process
    except Exception:
        return True  # Default to True if accelerate not available
from unimeth.data.extract import SignalFeatureExtractor

from .bam import BamReader


class SignalReader:
    """
    Reader for extracting signal features from POD5 and BAM files.
    
    Supports multi-worker data loading with proper worker ID handling.
    """
    
    def __init__(self, pod5_file: p5.DatasetReader, bam_file: BamReader, args):
        """
        Initialize signal reader.
        
        Args:
            pod5_file: Open POD5 dataset reader
            bam_file: Indexed BAM file reader
            args: Configuration arguments
        """
        self.pod5_file = pod5_file
        self.bam_file = bam_file
        self.extractor = SignalFeatureExtractor(args)
    
    def get_features(self, subset_name: str, read_ids: list):
        """
        Generator that yields features for a subset of read IDs.
        
        Handles multi-worker distribution where each worker processes
        a subset of reads based on worker ID.
        
        Args:
            subset_name: Name of this subset (for progress bar)
            read_ids: List of read IDs to process
            
        Yields:
            Feature dictionaries from SignalFeatureExtractor.get_feature()
        """
        # Get worker info for multi-worker distribution
        worker_info = get_worker_info()
        if worker_info is not None:
            pid, num_workers = worker_info.id, worker_info.num_workers
        else:
            pid, num_workers = 0, 1
        
        # Print subset name from main worker (unless progress is disabled)
        if pid == 0 and os.environ.get('UNIMETH_DISABLE_READING_PROGRESS', '0') != '1':
            local_print(subset_name)
        
        # Process reads assigned to this worker
        # Check if reading progress should be disabled (for clean output)
        disable_progress = (pid != 0 or not _is_local_main_process() or 
                          os.environ.get('UNIMETH_DISABLE_READING_PROGRESS', '0') == '1')
        
        iterator = enumerate(tqdm(
            read_ids,
            desc=subset_name,
            disable=disable_progress
        ))
        
        for i, read_id in iterator:
            if i % num_workers != pid:
                continue
            
            # Get POD5 and BAM data
            try:
                pod5_read = self.pod5_file.get_read(read_id)
            except Exception:
                continue
            
            bam_reads = self.bam_file.get_read_by_id(read_id)
            
            # Extract features for each BAM alignment
            for bam_read in bam_reads:
                if bam_read is None:
                    continue
                
                feature = self.extractor.get_feature(bam_read, pod5_read)
                if feature is None:
                    continue
                
                yield feature
