"""
Read processing pipeline.

Provides functions for processing reads into model input format.
"""
import numpy as np

from unimeth.config import tokenizer
from .patcher import patch_sequence
from .sites import get_methy_type


def weight_bis(num):
    """Calculate weight for bisulfite label based on distance from 50."""
    return 10 * ((num - 50) / 50) ** 2


def get_norm_params(feature, frequency, dorado_version):
    """
    Get signal normalization parameters.
    
    Args:
        feature: Feature dictionary with calibration values
        frequency: Sampling frequency ('4khz' or '5khz')
        dorado_version: Dorado basecaller version
        
    Returns:
        Tuple of (shift, scale)
    """
    shift_dacs_to_pa, scale_dacs_to_pa = feature['shift_dacs_to_pa'], feature['scale_dacs_to_pa']
    shift_pa_to_norm, scale_pa_to_norm = feature['shift_pa_to_norm'], feature['scale_pa_to_norm']
    
    if frequency == '5khz':
        if dorado_version <= 0.71:
            shift = 1 - shift_pa_to_norm
            scale = 1 / scale_pa_to_norm
        else:
            shift = shift_pa_to_norm
            scale = scale_pa_to_norm
    else:
        shift = (shift_pa_to_norm / scale_dacs_to_pa) - shift_dacs_to_pa
        scale = scale_pa_to_norm / scale_dacs_to_pa
    
    return shift, scale


def add_tokens(bases, signal_event, pred_pos, ref_pos=None, bis_label=None,
               detect_cpg=0, detect_chg=0, detect_chh=0, detect_m6a=0):
    """
    Add methylation type tokens to sequence.
    
    Args:
        bases: List of bases
        signal_event: List of signal events
        pred_pos: Positions of methylation sites
        ref_pos: Reference positions (optional)
        bis_label: Bisulfite labels (optional)
        detect_cpg/chg/chh/m6a: Detection flags
        
    Returns:
        Tuple of (data_dict, meta_data)
    """
    meta_data = []
    for i in reversed(range(len(pred_pos))):
        bis_l = -1 if bis_label is None else bis_label[i]
        pos_r = None if ref_pos is None else ref_pos[i]
        pos_p = pred_pos[i]
        
        methy_type = get_methy_type(bases, pos_p, detect_cpg, detect_chg, detect_chh, detect_m6a)
        
        if methy_type is not None:
            bases.insert(pos_p+1, methy_type)
            signal_event.insert(pos_p+1, [])
            meta_data.append({
                'ref_pos': pos_r,
                'labels': bis_l,
                'read_pos': pos_p,
            })
    
    meta_data = meta_data[::-1]
    data_dict = {
        'bases': bases,
        'signal_event': signal_event,
    }
    return data_dict, meta_data


from unimeth.config import get_total_stride as _compute_total_stride


def _get_total_stride(args):
    """Get total stride from args based on model_type."""
    model_type = getattr(args, 'model_type', 'default')
    return _compute_total_stride(model_type)


def get_inference_datasets(feature, args):
    """
    Generate inference datasets from a feature.
    
    Args:
        feature: Feature dictionary
        args: Arguments with overlap, chunk_size, etc.
        
    Yields:
        Inference data dictionaries
    """
    new_feature, meta_data = add_tokens(
        bases=feature['bases'],
        signal_event=feature['signal_event'],
        pred_pos=feature['pred_pos'],
        ref_pos=feature['ref_pos'],
        bis_label=feature['bis_label'],
        detect_cpg=args.cpg,
        detect_chg=args.chg,
        detect_chh=args.chh,
        detect_m6a=args.m6A
    )
    
    shift, scale = get_norm_params(feature, args.frequency, args.dorado_version)
    total_stride = _get_total_stride(args)
    
    # Apply same filters as Binning.get_data() so total_patches matches what Binning will yield
    bin_max = getattr(args, 'bin_size', 500) * getattr(args, 'num_bins', 15)
    patches = [
        patch for patch in patch_sequence(
            mode=args.mode,
            bases=new_feature['bases'],
            signal_event=new_feature['signal_event'],
            shift=shift,
            scale=scale,
            meta_data=meta_data,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            total_stride=total_stride
        )
        if len(patch['patch_pos']) > 0
        and len(patch['signals']) >= 50
        and len(patch['signals']) < bin_max
    ]
    total_patches = len(patches)

    for patch_idx, patch in enumerate(patches):
        inference_data = {
            'read_id': feature['read_id'],
            'chr': feature['chr'],
            'strand': feature['strand'],
            'signals': patch['signals'],
            'signal_pos': patch['signal_pos'],
            'decoder_input_ids': patch['decoder_input_ids'],
            'patch_pos': patch['patch_pos'],
            'read_pos': patch['read_pos'],
            'ref_pos': patch['ref_pos'],
            'labels': patch['labels'],
            'patch_idx': patch_idx,
            'total_patches': total_patches,
        }
        yield inference_data


def get_finetune_datasets(feature, args):
    """
    Generate finetune datasets from a feature.
    
    Args:
        feature: Feature dictionary
        args: Arguments with overlap, chunk_size, thresholds, etc.
        
    Yields:
        Training data dictionaries
    """
    read_label = feature['bis_label']
    new_feature, meta_data = add_tokens(
        bases=feature['bases'],
        signal_event=feature['signal_event'],
        pred_pos=feature['pred_pos'],
        bis_label=read_label,
        detect_cpg=args.cpg,
        detect_chg=args.chg,
        detect_chh=args.chh,
        detect_m6a=args.m6A
    )
    
    shift, scale = get_norm_params(feature, args.frequency, args.dorado_version)
    total_stride = _get_total_stride(args)
    
    for patch in patch_sequence(
        mode=args.mode,
        bases=new_feature['bases'],
        signal_event=new_feature['signal_event'],
        shift=shift,
        scale=scale,
        meta_data=meta_data,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        total_stride=total_stride
    ):
        label_tokens, weight = [], []
        for x in patch['labels']:
            if x == -1:
                label_tokens.append(tokenizer['[MASK]'])
                weight.append(0)
            elif x <= args.negative_thres:
                label_tokens.append(tokenizer['-'])
                weight.append(weight_bis(x))
            elif x >= args.positive_thres:
                label_tokens.append(tokenizer['+'])
                weight.append(weight_bis(x))
            else:
                label_tokens.append(tokenizer['[MASK]'])
                weight.append(0)
        
        if (np.array(label_tokens) == tokenizer['[MASK]']).all():
            continue
        
        train_data = {
            'signals': patch['signals'],
            'signal_pos': patch['signal_pos'],
            'decoder_input_ids': patch['decoder_input_ids'],
            'patch_pos': patch['patch_pos'],
            'labels': label_tokens,
            'weight': weight,
        }
        yield train_data


def get_pretrain_datasets(feature, args):
    """
    Generate pretrain datasets from a feature.
    
    Args:
        feature: Feature dictionary
        args: Arguments with overlap, chunk_size, etc.
        
    Yields:
        Training data dictionaries
    """
    shift, scale = get_norm_params(feature, args.frequency, args.dorado_version)
    total_stride = _get_total_stride(args)
    
    for patch in patch_sequence(
        mode=args.mode,
        bases=list(feature['bases']),
        signal_event=feature['signal_event'],
        shift=shift,
        scale=scale,
        meta_data=[],
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        total_stride=total_stride
    ):
        train_data = {
            'signals': patch['signals'],
            'signal_pos': patch['signal_pos'],
            'decoder_input_ids': patch['decoder_input_ids']
        }
        yield train_data


def get_datasets(feature, args):
    """
    Generate datasets from a feature based on mode.
    
    Args:
        feature: Feature dictionary
        args: Arguments with mode
        
    Yields:
        Data dictionaries
    """
    if args.mode == 'pretrain':
        yield from get_pretrain_datasets(feature, args)
    elif args.mode in ('finetune', 'calibration'):
        yield from get_finetune_datasets(feature, args)
    elif args.mode == 'inference':
        yield from get_inference_datasets(feature, args)
