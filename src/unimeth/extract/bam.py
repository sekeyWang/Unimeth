import os
import pysam
import pickle
from util import local_print

class Read_indexed_bam:
    def __init__(self, bam_dir, force_write = False):
        self.bam_file = pysam.AlignmentFile(bam_dir, "rb", check_sq=False)
        filename = bam_dir.split('/')[-1]
        os.makedirs('data_cache/bam_index', exist_ok = True)
        bam_index_file = f'data_cache/bam_index/{filename}.index'
        if force_write or not os.path.exists(bam_index_file):
            self.bam_index = self._build_bam_index(bam_index_file)
        else:
            self.bam_index = self._load_index(bam_index_file)
    
    def _load_index(self, bam_index_file):
        with open(bam_index_file, 'rb') as fr:
            bam_index = pickle.load(fr)
#        print(f'Load index from {bam_index_file}')
        return bam_index

    def _build_bam_index(self, bam_index_file):
        local_print('Building bam index...')
        bam_index = {}
        read_ptr = self.bam_file.tell()
        for bam_read in self.bam_file:
            if bam_read.is_supplementary or bam_read.is_secondary:
                read_ptr = self.bam_file.tell()
                continue
            if bam_read.has_tag('pi'):
                read_id = bam_read.get_tag("pi")
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
    
    def get_read_by_id(self, read_id):
        bam_reads = []
        if read_id not in self.bam_index:
#            print(read_id, "Not in bam_index")
            return bam_reads
        read_ptrs = self.bam_index[read_id]    
        for read_ptr in read_ptrs:
            self.bam_file.seek(read_ptr)
            bam_read = next(self.bam_file)
            bam_reads.append(bam_read)
        return bam_reads