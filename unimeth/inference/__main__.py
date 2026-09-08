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

POD5_SUFFIXES = ('.pod5',)
SLOW5_SUFFIXES = ('.slow5', '.blow5')


def get_model_info(args):
    """Get model architecture info from unimeth.model config."""
    cfg = ModelConfig.from_name(getattr(args, 'model_type', 'default'))
    return cfg.d_model, cfg.num_layers, cfg.total_stride


def format_inference_args(args):
    """Format inference arguments for readable output."""
    d_model, num_layers, cnn_stride = get_model_info(args)

    output_items = [('Format', args.output_format)]
    if args.output_format in ('bam', 'both'):
        output_items.extend([
            ('BAM Output', args.bam_out_dir or args.out_dir),
            ('Keep mv', 'yes' if args.keep_mv else 'no'),
        ])
    if args.output_format in ('tsv', 'both'):
        tsv_output = args.tsv_out_dir or args.out_dir
        gzip_tsv = args.gzip or str(tsv_output).lower().endswith('.gz')
        output_items.extend([
            ('TSV Output', tsv_output),
            ('Gzip TSV', 'yes' if gzip_tsv else 'no'),
        ])
    output_items.append(('Resume', 'yes' if args.resume else 'no'))

    sections = {
        'Input': [
            ('Signal', args.signal_dir),
            ('Signal Format', args.signal_format),
            ('BAM', args.bam_dir),
            ('Model', args.model_dir),
        ],
        'Output': output_items,
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
            ('Dorado Source', getattr(args, 'dorado_version_source', 'configured')),
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
            ('Use Binning', 'yes' if args.use_binning else 'no'),
            *([('Max Bin Length', args.max_bin_length)] if args.use_binning else []),
        ],
        'BAM Filtering (aligned mode only)': [
            ('Mode', args.bam_mode),
            ('Chromosomes', args.chr),
            ('MAPQ >=', args.mapq_thres),
            ('Identity >=', args.identity_thres),
            ('Skip Unmapped', 'yes' if args.skip_unmapped else 'no'),
            ('Supplementary', 'skip' if args.no_supplementary else 'keep'),
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
        args.signal_suffixes = SLOW5_SUFFIXES
    else:
        args.signal_dir = pod5_dir
        args.signal_format = 'POD5'
        args.signal_suffixes = POD5_SUFFIXES
    args.signal_label = args.signal_format

    # Keep the legacy internal field populated until dataset names are cleaned up.
    args.pod5_dir = args.signal_dir
    return args


def resolve_dorado_version(args, parser, detector=None):
    """Resolve the inference Dorado version from CLI, BAM header, or fallback."""
    if getattr(args, 'dorado_version', None) is not None:
        args.dorado_version_source = 'command line'
        return args

    fallback_version = str(defaultconfig['dorado_version'])
    bam_path = getattr(args, 'bam_dir', None)
    if not bam_path:
        args.dorado_version = fallback_version
        args.dorado_version_source = 'default (no BAM input)'
        return args

    if detector is None:
        from unimeth.ioutils.reader.bam import detect_dorado_version_from_bam
        detector = detect_dorado_version_from_bam

    try:
        detected_version = detector(bam_path)
    except (OSError, ValueError) as exc:
        parser.error(
            f"Could not auto-detect Dorado version from BAM header: {exc}. "
            "Pass --dorado_version major.minor.patch to override auto-detection."
        )

    if detected_version is None:
        args.dorado_version = fallback_version
        args.dorado_version_source = 'default (not found in BAM header)'
    else:
        args.dorado_version = detected_version
        args.dorado_version_source = 'BAM header (auto-detected)'
    return args


def main():
    parser = create_argument_parser('inference')
    if parser.prog.endswith('__main__.py'):
        parser.prog = 'python -m unimeth.inference'
    args = parser.parse_args()

    args = resolve_dorado_version(args, parser)
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
