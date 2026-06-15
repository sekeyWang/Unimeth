"""
Unified argument parsing configuration for all modes.
"""
import argparse
import datetime

from unimeth import __version__



def create_argument_parser(mode: str) -> argparse.ArgumentParser:
    """
    Create an argument parser for the specified mode.
    
    Args:
        mode: One of 'pretrain', 'finetune', 'calibration', 'inference'
    
    Returns:
        Configured ArgumentParser instance
    """
    # Mode-specific descriptions
    descriptions = {
        'inference': 'UniMeth Inference - Predict DNA methylation from nanopore data',
        'pretrain': 'UniMeth Pretraining - Self-supervised training on unlabeled data',
        'finetune': 'UniMeth Fine-tuning - Supervised training with WGBS labels',
        'calibration': 'UniMeth Calibration - Calibrate model on target dataset',
    }
    
    # Mode-specific epilog (examples)
    epilogs = {
        'inference': '''
Examples:
  # TSV output with the default model
  unimeth-infer \\
      --pod5 data.pod5 --bam data.bam \\
      --model model.pt --out results.tsv \\
      --cpg 1 --chg 1 --chh 1 \\
      --pore_type R10.4.1 --frequency 5khz --dorado_version 0.71

  # Multi-GPU TSV output
  accelerate launch --num_processes 8 -m unimeth.inference \\
      --pod5 data.pod5 --bam data.bam \\
      --model model.pt --out results.tsv \\
      --cpg 1 --chg 1 --chh 1 \\
      --pore_type R10.4.1 --frequency 5khz --dorado_version 0.71

  # BAM output with the distilled model
  unimeth-infer \\
      --pod5 data.pod5 --bam data.bam \\
      --model distilled_model.pt --out results.bam --output_format bam \\
      --model_type distilled \\
      --cpg 1 --batch_size 512 \\
      --pore_type R10.4.1 --frequency 5khz --dorado_version 0.71

Model Types:
  default   - 100M params, d_model=384, 12 layers, 4x CNN downsample
  distilled -  62M params, d_model=256,  6 layers, 8x CNN downsample (faster)
        ''',
    }
    
    parser = argparse.ArgumentParser(
        description=descriptions.get(mode, f'UniMeth {mode} mode'),
        epilog=epilogs.get(mode),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Common data arguments
    parser.add_argument('--bam_dir', '--bam', dest='bam_dir', type=str,
                        help='Path to BAM file (must contain mv/ts tags)')
    parser.add_argument('--pod5_dir', '--pod5', dest='pod5_dir', type=str,
                        help='Path to POD5 file or directory containing raw signal files')
    parser.add_argument('--chr', type=str, default='|', 
                        help='Chromosome filter: "|"=all, "|Chr1,Chr2"=include only, "Chr1,Chr2|"=exclude')
    parser.add_argument('--max_bin_length', type=int, default=512,
                        help='Max sequences per bin before yielding (default: 512). Smaller values reduce GPU idle wait at the cost of more padding.')
    parser.add_argument('--use_binning', type=int, default=1,
                        help='Enable length-based binning for DataLoader (1=yes, 0=no). Disabling reduces latency but may increase padding.')
    
    # Common model arguments
    parser.add_argument('--model_dir', '--model', dest='model_dir', type=str,
                        help='Path to model checkpoint (.pt, .bin, or pytorch_model.bin)')
    parser.add_argument('--batch_size', type=int, 
                        help='Batch size per GPU (default: 512 after config merge)')
    parser.add_argument('--pore_type', type=str, choices=['R9.4.1', 'R10.4.1'],
                        help='Nanopore chemistry type')
    parser.add_argument('--frequency', type=str, choices=['4khz', '5khz'],
                        help='Sampling frequency of the data')
    parser.add_argument('--dorado_version', type=float, 
                        help='Dorado basecaller version (e.g., 0.71, affects signal norm)')
    
    # Training-specific arguments
    if mode in ['pretrain', 'finetune', 'calibration']:
        parser.add_argument('--train_pod5_dir', '--train_pod5', dest='train_pod5_dir', type=str,
                            help='Path to training POD5 directory')
        parser.add_argument('--val_pod5_dir', '--val_pod5', dest='val_pod5_dir', type=str,
                            help='Path to validation POD5 directory')
        parser.add_argument('--max_steps', type=int, help='Maximum training steps')
        parser.add_argument('--val_num', type=int, help='Number of validation samples')
    
    # Methylation type flags (for finetune, calibration, inference)
    if mode in ['finetune', 'calibration', 'inference']:
        parser.add_argument('--cpg', type=int, default=0, help='Enable CpG detection (1=yes)')
        parser.add_argument('--chg', type=int, default=0, help='Enable CHG detection (1=yes)')
        parser.add_argument('--chh', type=int, default=0, help='Enable CHH detection (1=yes)')
        parser.add_argument('--m6A', type=int, default=0, help='Enable m6A detection (1=yes)')
    
    # Fine-tuning specific
    if mode in ['finetune', 'calibration']:
        parser.add_argument('--plant', type=int, default=0, 
                           help='Use plant-specific unbalanced loss (1=yes)')
    
    # Inference-specific arguments
    if mode == 'inference':
        parser.add_argument('-v', '--version', action='version',
                           version=f'unimeth-infer {__version__}')
        parser.add_argument('--out_dir', '--out', dest='out_dir', type=str,
                           help='Output file path (.txt for TSV, .bam for BAM)')
        parser.add_argument('--limit', type=int, default=None, 
                           help='Process only first N batches (for quick testing)')
        parser.add_argument('--output_format', type=str, choices=['tsv', 'bam', 'both'],
                           default='tsv', help='Output format: tsv (default), bam, or both (dual output for verification)')
        parser.add_argument('--tsv_out_dir', '--tsv_out', dest='tsv_out_dir', type=str, default=None,
                           help='TSV output path when --output_format both (defaults to --out_dir)')
        parser.add_argument('--gzip', action='store_true',
                           help='Compress TSV output with gzip (.gz). Applies to --output_format tsv or both')
        parser.add_argument('--bam_out_dir', '--bam_out', dest='bam_out_dir', type=str, default=None,
                           help='BAM output path when --output_format both (defaults to --out_dir)')
        parser.add_argument('--model_type', type=str, choices=['default', 'distilled'],
                           default='default', 
                           help='Model architecture: default (100M params) or distilled (62M params, faster)')
        parser.add_argument('--num_workers', type=int, default=8,
                           help='Number of CPU workers per GPU for data loading (default: 8, total=8 x num_gpus)')
        parser.add_argument('--mapq', dest='mapq_thres', type=int, default=0,
                           help='Minimum BAM mapping quality for aligned reads (default: 0)')
        parser.add_argument('--show_reading_progress', action='store_true',
                           help='Show tqdm progress bar for data reading (default: disabled for clean output)')
        # BAM output: read-level flush control
        # The Dataset flushes bins every N reads and signals the BAM writer via __reads_complete__ markers.
        # This ensures all patches for a batch of reads are collected before writing to BAM.
        parser.add_argument('--reads_per_flush', type=int, default=1000,
                           help='Number of reads to accumulate before flushing to BAM (default: 1000)')
    
    # Run name with timestamp
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    parser.add_argument('--run_name', default=f'run_{formatted_time}',
                       help='Run name for logging and checkpoint naming')
    
    return parser


def merge_with_default_config(args: argparse.Namespace, default_config: dict) -> argparse.Namespace:
    """
    Merge parsed arguments with default configuration.
    
    Args:
        args: Parsed arguments
        default_config: Default configuration dictionary
    
    Returns:
        args with default values filled in
    """
    for key, value in default_config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    return args
