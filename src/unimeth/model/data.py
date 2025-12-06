import os
import torch
from torch.utils.data import IterableDataset, Dataset
from unimeth.extract.reader import Reader_raw
from unimeth.extract.processors import ReadProcessor
from unimeth.extract.bam import Read_indexed_bam
from unimeth.util import local_print
from itertools import zip_longest

class MultiDataset(IterableDataset):
    def __init__(self, generators):
        self.generators = generators
    def __iter__(self):
        for binned_data in zip_longest(*self.generators, fillvalue=None):
            for data in binned_data:
                yield data
    
class RawDataset(IterableDataset):
    def __init__(self, pod5_dir, bam_dir, args, negative_thres=0, positive_thres=100):
        if os.path.isdir(pod5_dir):
            subsets = os.listdir(pod5_dir)
            self.pod5_dirs = [os.path.join(pod5_dir, x) for x in subsets]
        else:
            self.pod5_dirs = [pod5_dir]
        self.bam_dir = bam_dir
        self.args = args
        self.read_processor = ReadProcessor(self.args, negative_thres, positive_thres)
        self.binning = Binning(args)

    def __iter__(self):
        bam_file = Read_indexed_bam(self.bam_dir, force_write=True)
        for pod5_dir in self.pod5_dirs:
            reader = Reader_raw(pod5_dir=pod5_dir, bam_file=bam_file, args=self.args)
            for feature in reader.get_features():
                for dataset in self.read_processor.get_datasets(feature):
                    for binned_data in self.binning.get_data(dataset):
                        yield binned_data
        yield from self.binning.flush()

class RawValDataset(Dataset):
    def __init__(self, pod5_dirs, bam_dirs, args):
        if len(pod5_dirs) != len(bam_dirs):
            print('Need same number of pod5 files and bam files')
        self.datas = []
        total_max = 5000
        per_max = total_max // len(pod5_dirs)
        for i in range(len(pod5_dirs)):
            dataset = RawDataset(pod5_dirs[i], bam_dirs[i], args)    
            for num, x in enumerate(dataset):
                self.datas.append(x)
                if num > per_max:
                    break
        local_print(f'Validation data: {len(self.datas)}')

    def __getitem__(self, idx):
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)


def collate_fn(mode, datas):
    signals = [torch.tensor(sample["signals"], dtype=torch.float) for sample in datas]    
    padded_signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True, padding_value=0)

    decoder_inputs = [torch.tensor(sample["decoder_input_ids"], dtype=torch.long) for sample in datas]
    padded_decoder_inputs = torch.nn.utils.rnn.pad_sequence(decoder_inputs, batch_first=True, padding_value=-100)

    signal_pos = [torch.tensor(sample["signal_pos"], dtype=torch.int) for sample in datas]
    padded_signal_pos = torch.nn.utils.rnn.pad_sequence(signal_pos, batch_first=True, padding_value=256)

    encoder_masks = [torch.ones(((len(sample["signals"]) - 1) // 4) + 1, dtype=torch.long) for sample in datas]
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
    return ret

import functools
collate_fn_pretrain = functools.partial(collate_fn, 'pretrain')
collate_fn_finetune = functools.partial(collate_fn, 'finetune')
collate_fn_inference = functools.partial(collate_fn, 'inference')


class Binning:

    def __init__(self, args):
        self.num_bins = args.num_bins
        self.bin_size, self.max_bin_length = args.bin_size, args.max_bin_length
        self.bins = [[] for _ in range(self.num_bins)]

    def get_data(self, dataset):
        l = len(dataset['signals'])
        if l < 50 or l >= self.bin_size * self.num_bins:
            return
        bin_id = l // self.bin_size
        self.bins[bin_id].append(dataset)
        if len(self.bins[bin_id]) >= self.max_bin_length:
            for x in self.bins[bin_id]:
                yield x
            self.bins[bin_id] = []

    def flush(self):
        for bin in self.bins:
            for x in bin:
                yield x
