import numpy as np
from config import methy_types, tokenizer

class Patcher:
    def __init__(self, args):
        self.overlap, self.chunk_size = args.overlap, args.chunk_size
        self.mode = args.mode
        self.use_prune = args.use_prune

    def prune(self, chunk_bases, ignore_left, ignore_right):
        first, last = len(chunk_bases), 0 
        for j in range(ignore_left, len(chunk_bases)-(0 if ignore_right == 0 else self.overlap)):
            if chunk_bases[j] in ['-', '+', '[MASK]']:
                first = min(first, j)
                last = max(last, j)
        if last == 0:
            return 0, 0
        first = max(first - self.overlap, 0)
        last = min(last + self.overlap + 1, len(chunk_bases))
        return first, last

    def get_patch_generator(self, mode, bases, signal_event, shift, scale, meta_data=[]):
        seq_len = len(bases)
        overlap = self.overlap
        chunk_size = self.chunk_size
        gap = chunk_size - overlap * 2
        meta_idx = 0
        for i in range(0, seq_len-overlap*2, gap):
            # sequence processing
            left, right = i, i + chunk_size
            chunk_bases = bases[left: right]
            chunk_signal_event = signal_event[left : right]
            chunk_bases = np.array([tokenizer[atcg] for atcg in chunk_bases], dtype=np.int8)
            ignore_left = overlap if i > 0 else 0
            ignore_right = overlap + 1 if i + gap < seq_len-overlap*2 else 0

            methys = np.array([tokenizer[atcg] for atcg in methy_types])
            result = np.where(np.isin(chunk_bases, methys))[0]

            meta_input = {
                'patch_pos': [],
                'read_pos': [],
                'ref_pos': [],
                'labels': [],
            }
            first, last = len(chunk_bases), 0 
            for patch_pos in result:
                if patch_pos < ignore_left or patch_pos > len(chunk_bases) - ignore_right:
                    continue
                first = min(first, patch_pos)
                last = max(last, patch_pos)
                meta = meta_data[meta_idx]
                meta_input['patch_pos'].append(patch_pos)
                if 'read_pos' in meta:
                    meta_input['read_pos'].append(meta['read_pos']) 
                if 'ref_pos' in meta:
                    meta_input['ref_pos'].append(meta['ref_pos'])
                if 'labels' in meta:
                    meta_input['labels'].append(meta['labels'])
                meta_idx += 1
            if self.use_prune and last != 0:
                left = max(first - self.overlap, 0)
                right = min(last + self.overlap + 1, len(chunk_bases))
                chunk_bases = chunk_bases[left:right]
                chunk_signal_event = chunk_signal_event[left:right]
                meta_input['patch_pos'] = np.array(meta_input['patch_pos']) - left

            # get signal pos
            signal_pos = [[i] * len(sublist) for i, sublist in enumerate(chunk_signal_event)]
            signal_pos = sum(signal_pos, [])
            signal_pos = np.array(signal_pos[::4], dtype=np.int32)
            assert len(chunk_bases) == len(chunk_signal_event)

            chunk_bases = np.append(chunk_bases, tokenizer['[END]'])
            chunk_signal = np.hstack(chunk_signal_event)
            chunk_signal = chunk_signal = (chunk_signal - shift) / scale
            if mode == 'pretrain':
                data_dict = {
                    'signals': chunk_signal,
                    'signal_pos': signal_pos,
                    'decoder_input_ids': chunk_bases,
                }
            else: 
                data_dict = {
                    'signals': chunk_signal,
                    'signal_pos': signal_pos,
                    'decoder_input_ids': chunk_bases,
                    'patch_pos': meta_input['patch_pos'],
                    'labels': meta_input['labels'],
                    'read_pos': meta_input['read_pos'],
                    'ref_pos': meta_input['ref_pos'],
                }
            yield data_dict
        assert meta_idx == len(meta_data), print(meta_idx, len(meta_data))