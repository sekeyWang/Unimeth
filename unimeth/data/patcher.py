"""
Sequence patcher for model input.

Provides functions for chunking sequences into model-input patches.
"""
import numpy as np
from unimeth.config import methy_types, tokenizer


def patch_sequence(mode, bases, signal_event, shift, scale, meta_data,
                   chunk_size: int, overlap: int, total_stride: int = 4):
    """
    Generate patches from a sequence for model input.
    
    Args:
        mode: Model mode ('pretrain', 'finetune', or 'inference')
        bases: List of base tokens
        signal_event: List of signal events per base
        shift: Signal normalization shift
        scale: Signal normalization scale
        meta_data: List of metadata dicts with ref_pos, labels, read_pos
        chunk_size: Size of each chunk/patch
        overlap: Overlap between consecutive chunks
        
    Yields:
        Dictionary with patch data
    """
    seq_len = len(bases)
    gap = chunk_size - overlap * 2
    meta_idx = 0
    
    for i in range(0, seq_len - overlap * 2, gap):
        # Sequence processing
        left, right = i, i + chunk_size
        chunk_bases = bases[left: right]
        chunk_signal_event = signal_event[left : right]
        chunk_bases = np.array([tokenizer[atcg] for atcg in chunk_bases], dtype=np.int8)
        ignore_left = overlap if i > 0 else 0
        ignore_right = overlap + 1 if i + gap < seq_len - overlap * 2 else 0

        methys = np.array([tokenizer[atcg] for atcg in methy_types])
        result = np.where(np.isin(chunk_bases, methys))[0]

        meta_input = {
            'patch_pos': [],
            'read_pos': [],
            'ref_pos': [],
            'labels': [],
        }
        
        for patch_pos in result:
            if patch_pos < ignore_left or patch_pos > len(chunk_bases) - ignore_right:
                continue
            
            meta = meta_data[meta_idx]
            meta_input['patch_pos'].append(patch_pos)
            if 'read_pos' in meta:
                meta_input['read_pos'].append(meta['read_pos'])
            if 'ref_pos' in meta:
                meta_input['ref_pos'].append(meta['ref_pos'])
            if 'labels' in meta:
                meta_input['labels'].append(meta['labels'])
            meta_idx += 1

        # Get signal pos - use total_stride for downsampling
        # Vectorized construction: repeat indices according to event lengths
        event_lengths = np.fromiter((len(sublist) for sublist in chunk_signal_event), dtype=np.int32, count=len(chunk_signal_event))
        signal_pos = np.repeat(np.arange(len(chunk_signal_event), dtype=np.int32), event_lengths)
        signal_pos = signal_pos[::total_stride]
        
        assert len(chunk_bases) == len(chunk_signal_event)

        chunk_bases = np.append(chunk_bases, tokenizer['[END]'])
        chunk_signal = np.hstack(chunk_signal_event)
        chunk_signal = (chunk_signal - shift) / scale
        
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
    
    assert meta_idx == len(meta_data), f"{meta_idx} vs {len(meta_data)}"
