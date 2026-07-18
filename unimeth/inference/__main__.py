"""
Inference script for UniMeth.

Supports TSV and BAM output formats.
Each GPU processes data independently (no inter-rank synchronization)
for maximum throughput.

Example usage:
    accelerate launch -m unimeth.inference \
        --pod5_dir <path_to_pod5> \
        --bam_dir <path_to_bam> \
        --model_dir <finetuned_model_path> \
        --out_dir results/predictions.txt \
        --cpg 1 --chg 1 --chh 1 \
        --batch_size 256
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*past_key_values.*')
warnings.filterwarnings('ignore', message='.*ipex flag.*')
warnings.filterwarnings('ignore', message='.*kernel version.*')
warnings.filterwarnings('ignore', message='.*EncoderDecoderCache.*')
warnings.filterwarnings('ignore', message='.*deprecated.*')

import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.models.bart.modeling_bart').setLevel(logging.ERROR)

from unimeth.config import create_argument_parser, merge_with_default_config, defaultconfig
from unimeth.config.model_config import ModelConfig
from unimeth.utils import local_print


def get_model_info(args):
    """Get model architecture info from unimeth.model config."""
    cfg = ModelConfig.from_name(getattr(args, 'model_type', 'default'))
    return cfg.d_model, cfg.num_layers, cfg.total_stride


def format_inference_args(args):
    """Format inference arguments for readable output."""
    d_model, num_layers, cnn_stride = get_model_info(args)

    sections = {
        'Input/Output': [
            ('Signal', args.signal_dir),
            ('Signal Format', args.signal_format),
            ('BAM', args.bam_dir),
            ('Model', args.model_dir),
            ('Output', args.out_dir),
            ('Format', args.output_format),
        ],
        'Model Config': [
            ('Type', args.model_type),
            ('d_model', d_model),
            ('Layers', num_layers),
            ('CNN Downsample', f'{cnn_stride}x'),

        ],
        'Platform': [
            ('Pore Type', args.pore_type),
            ('Frequency', args.frequency),
            ('Dorado Ver', args.dorado_version),
        ],
        'Methylation': [
            ('CpG', 'yes' if args.cpg else 'no'),
            ('CHG', 'yes' if args.chg else 'no'),
            ('CHH', 'yes' if args.chh else 'no'),
            ('m6A', 'yes' if args.m6A else 'no'),
        ],
        'Processing': [
            ('Batch Size', args.batch_size),
            ('Workers', args.num_workers),
            *([('Bins', args.num_bins), ('Max Bin Length', args.max_bin_length)] if args.use_binning else []),
            ('Use Binning', 'yes' if args.use_binning else 'no'),
        ],
    }

    lines = ['', '=' * 50, 'Inference Configuration', '=' * 50]
    for section, items in sections.items():
        lines.append(f"\n[{section}]")
        for key, value in items:
            lines.append(f"  {key:15s}: {value}")
    lines.append('\n' + '=' * 50)
    return '\n'.join(lines)


def normalize_signal_input(args, parser):
    """Validate POD5/SLOW5 options and expose one internal signal path."""
    pod5_dir = getattr(args, 'pod5_dir', None)
    slow5_dir = getattr(args, 'slow5_dir', None)

    if pod5_dir and slow5_dir:
        parser.error('Use only one raw signal input option: --pod5/--pod5_dir or --slow5/--slow5_dir')
    if slow5_dir:
        args.signal_dir = slow5_dir
        args.signal_format = 'SLOW5/BLOW5'
    else:
        args.signal_dir = pod5_dir
        args.signal_format = 'POD5'

    # Keep the legacy internal field populated until dataset names are cleaned up.
    args.pod5_dir = args.signal_dir
    return args


def main():
    parser = create_argument_parser('inference')
    if parser.prog.endswith('__main__.py'):
        parser.prog = 'python -m unimeth.inference'
    args = parser.parse_args()

    args = merge_with_default_config(args, defaultconfig)
    args.mode = 'inference'
    args = normalize_signal_input(args, parser)

    local_print(format_inference_args(args))

    from unimeth.model.datasets import Pod5BamDataset
    from unimeth.inference.engine import InferenceEngine

    engine = InferenceEngine(args, Pod5BamDataset)
    engine.run(output_format=args.output_format)


if __name__ == '__main__':
    main()
