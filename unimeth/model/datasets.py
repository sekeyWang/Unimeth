import os
import torch
from torch.utils.data import IterableDataset, Dataset
from unimeth.ioutils.reader import SignalReader as Reader_raw
from unimeth.data.pipeline import get_datasets
from unimeth.ioutils.reader import BamReader
from unimeth.utils import local_print
from itertools import zip_longest
import pod5 as p5

class MultiFileDataset(IterableDataset):
    def __init__(self, generators):
        self.generators = generators
    def __iter__(self):
        for binned_data in zip_longest(*self.generators, fillvalue=None):
            for data in binned_data:
                yield data

def get_read_ids(pod5_file):
    read_ids = []
    for x in pod5_file.read_ids:
        read_ids.append(x)
    return read_ids


class Pod5BamDataset(IterableDataset):
    n_shards = 1
    def __init__(self, pod5_dir, bam_dir, args):
        if os.path.isdir(pod5_dir):
            subsets = os.listdir(pod5_dir)
            self.pod5_dirs = [os.path.join(pod5_dir, x) for x in subsets]
        else:
            self.pod5_dirs = [pod5_dir]
        self.bam_dir = bam_dir
        self.args = args
        self.args = args  # Store args for get_datasets
        self.binning = Binning(args)
        self.read_ids = None
        BamReader(self.bam_dir, force_rebuild_index=True)
    
    def set_read_ids(self, read_ids):
        self.read_ids = read_ids

    def __iter__(self):
        # Shard reads at rank level so all patches of a read go to the same GPU.
        # This must be done here (not via Accelerate's IterableDatasetShard) because
        # binning scatters patches across batches — item-level sharding would split
        # the same read's patches across ranks.
        try:
            from accelerate.state import PartialState
            state = PartialState()
            rank, num_ranks = state.process_index, state.num_processes
        except Exception:
            rank, num_ranks = 0, 1

        # SignalReader further shards reads by worker (i % num_workers == pid).
        # reads_per_flush must exceed the per-worker read count, otherwise mid-stream
        # _flush_all events create mixed-length transition batches → uncertain predictions.
        from torch.utils.data import get_worker_info as _get_worker_info
        worker_info = _get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1

        bam_file = BamReader(self.bam_dir, force_rebuild_index=False)

        # Two-pass: collect all rank-level read_ids first so we can compute the
        # per-worker count and set reads_per_flush accordingly.
        pod5_entries = []
        total_rank_reads = 0
        for pod5_dir in self.pod5_dirs:
            pod5_file = p5.DatasetReader(pod5_dir, recursive=True, index=True)
            rids = get_read_ids(pod5_file) if self.read_ids is None else self.read_ids
            rids = rids[rank::num_ranks]
            pod5_entries.append((pod5_dir, pod5_file, rids))
            total_rank_reads += len(rids)

        # Raise reads_per_flush above per-worker count to eliminate mid-stream flushes.
        per_worker_reads = (total_rank_reads + num_workers - 1) // num_workers
        if per_worker_reads >= self.binning.reads_per_flush:
            self.binning.reads_per_flush = per_worker_reads + 1

        for pod5_dir, pod5_file, read_ids in pod5_entries:
            subset_name = pod5_dir.split('/')[-1]
            reader = Reader_raw(pod5_file, bam_file=bam_file, args=self.args)
            for feature in reader.get_features(subset_name, read_ids):
                for dataset in get_datasets(feature, self.args):
                    for binned_data in self.binning.get_data(dataset):
                        yield binned_data
        yield from self.binning.flush()
        yield {'__reads_complete__': True}

class ValidationDataset(Dataset):
    def __init__(self, pod5_dirs, bam_dirs, args):
        if len(pod5_dirs) != len(bam_dirs):
            print('Need same number of pod5 files and bam files')
        self.datas = []
        total_max = 5000
        per_max = total_max // len(pod5_dirs)
        for i in range(len(pod5_dirs)):
            dataset = Pod5BamDataset(pod5_dirs[i], bam_dirs[i], args)    
            for num, x in enumerate(dataset):
                self.datas.append(x)
                if num > per_max:
                    break
        local_print(f'Validation data: {len(self.datas)}')

    def __getitem__(self, idx):
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)


def collate_fn(mode, datas, total_stride=4):
    # Separate flush markers from real data samples (marker may land at any batch position)
    has_flush = any(isinstance(d, dict) and '__reads_complete__' in d for d in datas)
    datas = [d for d in datas if not (isinstance(d, dict) and '__reads_complete__' in d)]

    if not datas:
        return {'__reads_complete__': True}

    signals = [torch.tensor(sample["signals"], dtype=torch.float) for sample in datas]    
    padded_signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True, padding_value=0)

    decoder_inputs = [torch.tensor(sample["decoder_input_ids"], dtype=torch.long) for sample in datas]
    padded_decoder_inputs = torch.nn.utils.rnn.pad_sequence(decoder_inputs, batch_first=True, padding_value=-100)

    signal_pos = [torch.tensor(sample["signal_pos"], dtype=torch.int) for sample in datas]
    padded_signal_pos = torch.nn.utils.rnn.pad_sequence(signal_pos, batch_first=True, padding_value=256)

    encoder_masks = [torch.ones(((len(sample["signals"]) - 1) // total_stride) + 1, dtype=torch.long) for sample in datas]
    padded_encoder_masks = torch.nn.utils.rnn.pad_sequence(encoder_masks, batch_first=True, padding_value=0)

    if mode == 'inference':
        ret = {
            'signals': padded_signals,
            'encoder_mask': padded_encoder_masks,
            'decoder_input_ids': padded_decoder_inputs,
            'signal_pos': padded_signal_pos,

            'patch_pos': [sample['patch_pos'] for sample in datas],
            'read_pos': [sample['read_pos'] for sample in datas],
            'ref_pos': [sample['ref_pos'] for sample in datas],
            'labels': [sample['labels'] for sample in datas],
            'read_id': [sample['read_id'] for sample in datas],
            'chr': [sample['chr'] for sample in datas],
            'strand': [sample['strand'] for sample in datas],
            'patch_idx': [sample['patch_idx'] for sample in datas],
            'total_patches': [sample['total_patches'] for sample in datas],
        }
    elif mode == 'finetune':
        padded_label = torch.ones_like(padded_decoder_inputs) * -100
        padded_weight = torch.ones_like(padded_decoder_inputs)
        for i, sample in enumerate(datas):
            patch_pos, label, weight = sample['patch_pos'], sample['labels'], sample['weight']
            for j in range(len(patch_pos)):
                pos, label_pos, weight_pos = patch_pos[j], label[j], weight[j]
                padded_label[i, pos] = label_pos
                padded_weight[i, pos] = weight_pos
        ret = {
            'signals': padded_signals,
            'encoder_mask': padded_encoder_masks,
            'decoder_input_ids': padded_decoder_inputs,
            'signal_pos': padded_signal_pos,
            'labels': padded_label,
            'weights': padded_weight,
        }
    elif mode == 'pretrain':
        ret = {
            'signals': padded_signals,
            'encoder_mask': padded_encoder_masks,
            'labels': padded_decoder_inputs,
            'signal_pos': padded_signal_pos,
        }
    ret['__reads_complete__'] = has_flush
    return ret

import functools
collate_fn_pretrain = functools.partial(collate_fn, 'pretrain')
collate_fn_finetune = functools.partial(collate_fn, 'finetune')
collate_fn_inference = functools.partial(collate_fn, 'inference')


class Binning:

    def __init__(self, args):
        self.use_binning = bool(getattr(args, 'use_binning', 1))
        self.num_bins = args.num_bins
        self.bin_size, self.max_bin_length = args.bin_size, args.max_bin_length
        self.bins = [[] for _ in range(self.num_bins)]
        # Read-level flush: flush all bins every N reads to ensure BAM writer can safely write
        self.reads_per_flush = getattr(args, 'reads_per_flush', 1000)
        self.read_count = 0
        self.seen_read_ids = set()

    def get_data(self, dataset):
        l = len(dataset['signals'])
        if l < 50 or l >= self.bin_size * self.num_bins:
            return
        if not self.use_binning:
            yield dataset
            # Track unique reads
            read_id = dataset.get('read_id')
            if read_id not in self.seen_read_ids:
                self.seen_read_ids.add(read_id)
                self.read_count += 1
                if self.read_count >= self.reads_per_flush:
                    yield {'__reads_complete__': True}
                    self.read_count = 0
                    self.seen_read_ids.clear()
            return
        bin_id = l // self.bin_size
        self.bins[bin_id].append(dataset)
        # Track unique reads for read-level flush
        read_id = dataset.get('read_id')
        if read_id not in self.seen_read_ids:
            self.seen_read_ids.add(read_id)
            self.read_count += 1
            if self.read_count >= self.reads_per_flush:
                # Flush all bins and signal completion
                yield from self._flush_all()
                yield {'__reads_complete__': True}
                self.read_count = 0
                self.seen_read_ids.clear()
        # Normal bin flush based on size
        if len(self.bins[bin_id]) >= self.max_bin_length:
            for x in self.bins[bin_id]:
                yield x
            self.bins[bin_id] = []

    def _flush_all(self):
        """Flush all bins sorted by signal length to minimise padding in transition batches."""
        all_patches = []
        for bin in self.bins:
            all_patches.extend(bin)
            bin.clear()
        all_patches.sort(key=lambda x: len(x['signals']))
        yield from all_patches

    def flush(self):
        """Final flush at end of iteration."""
        if not self.use_binning:
            return
        yield from self._flush_all()
