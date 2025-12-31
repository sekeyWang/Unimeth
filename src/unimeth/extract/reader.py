'''
    Given pod5 file and bam file, extract read features
'''

import pod5 as p5
from util import local_print, state
from torch.utils.data import get_worker_info
from extract.feature import Extractor_raw
from extract.bam import Read_indexed_bam
from tqdm import tqdm

class Reader_raw:
    def __init__(self, pod5_dir, bam_file:Read_indexed_bam, args):
        self.subset_name = pod5_dir.split('/')[-1]
        self.pod5_file = p5.DatasetReader(pod5_dir, recursive=True, index=True)
        self.bam_file = bam_file
        self.read_ids = self.get_read_ids()
        self.extractor = Extractor_raw(args)

    def get_read_ids(self):
        read_ids = []
        for x in self.pod5_file.read_ids:
            read_ids.append(x)
        return read_ids
    
    def get_features(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            pid, num_workers = worker_info.id, worker_info.num_workers
        else:
            pid, num_workers = 0, 1
        if pid == 0:
            local_print(self.subset_name)
        for i, read_id in enumerate(tqdm(self.read_ids, desc=self.subset_name, disable=(pid != 0 or not state.is_local_main_process))):
            if i % num_workers != pid:
                continue    
            pod5_read = self.pod5_file.get_read(read_id)
            bam_reads = self.bam_file.get_read_by_id(read_id)
            for bam_read in bam_reads:
                if bam_read is None:
                    continue
                feature = self.extractor.get_feature(bam_read, pod5_read)
                if feature is None:
                    continue
                yield feature
