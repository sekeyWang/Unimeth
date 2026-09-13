"""
Unified training script for UniMeth.

This script provides a single entry point for pretraining, fine-tuning, and calibration.

Example usage:
    # Pretraining
    accelerate launch -m unimeth.training --mode pretrain \
        --bam_dir <path_to_bam> \
        --train_pod5_dir <path_to_pod5> \
        --val_pod5_dir <path_to_val_pod5> \
        --max_steps 100000 \
        --batch_size 512

    # Fine-tuning
    accelerate launch -m unimeth.training --mode finetune \
        --bam_dir <path_to_labeled_bam> \
        --train_pod5_dir <path_to_pod5> \
        --val_pod5_dir <path_to_val_pod5> \
        --model_dir <pretrained_model_path> \
        --cpg 1 --chg 1 --chh 1 \
        --max_steps 100000 \
        --batch_size 256

    # Calibration
    accelerate launch -m unimeth.training --mode calibration \
        --bam_dir <path_to_labeled_bam> \
        --train_pod5_dir <path_to_pod5> \
        --val_pod5_dir <path_to_val_pod5> \
        --model_dir <finetuned_model_path> \
        --max_steps 10000 \
        --batch_size 256
"""
import argparse
import datetime

from unimeth.config import merge_with_default_config, defaultconfig
from unimeth.training import PretrainTrainer, FinetuneTrainer, CalibrationTrainer
from unimeth.utils import local_print

TRAINER_MAP = {
    'pretrain': PretrainTrainer,
    'finetune': FinetuneTrainer,
    'calibration': CalibrationTrainer,
}

VALID_MODES = list(TRAINER_MAP.keys())


def create_training_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='UniMeth Training')

    parser.add_argument('--mode', type=str, required=True, choices=VALID_MODES,
                        help='Training mode: pretrain, finetune, or calibration')

    parser.add_argument('--bam_dir', type=str, help='Path to BAM file or directory')
    parser.add_argument('--train_pod5_dir', type=str, help='Path to training POD5 directory')
    parser.add_argument('--val_pod5_dir', type=str, help='Path to validation POD5 directory')
    parser.add_argument('--chr', type=str, default='|',
                        help='Chromosome filter (e.g., "|" for all, "|Chr1,Chr2" for include)')

    parser.add_argument('--model_dir', type=str, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--pore_type', type=str, help='Nanopore type (R9.4.1 or R10.4.1)')
    parser.add_argument('--frequency', type=str, help='Sampling frequency (4khz or 5khz)')
    parser.add_argument('--dorado_version', type=float, help='Dorado basecaller version')

    parser.add_argument('--max_steps', type=int, help='Maximum training steps')
    parser.add_argument('--val_num', type=int, help='Number of validation samples')

    parser.add_argument('--cpg', type=int, default=0, help='Enable CpG detection (1=yes)')
    parser.add_argument('--chg', type=int, default=0, help='Enable CHG detection (1=yes)')
    parser.add_argument('--chh', type=int, default=0, help='Enable CHH detection (1=yes)')
    parser.add_argument('--m6A', type=int, default=0, help='Enable m6A detection (1=yes)')

    parser.add_argument('--plant', type=int, default=0,
                        help='Use plant-specific unbalanced loss (1=yes)')

    parser.add_argument('--teacher_model_dir', type=str,
                        help='Path to teacher model checkpoint (for distill mode)')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Temperature for distillation (default: 4.0)')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Weight for hard label loss in distillation (default: 0.3)')
    parser.add_argument('--beta', type=float, default=0.7,
                        help='Weight for KL divergence loss in distillation (default: 0.7)')
    parser.add_argument('--student_hidden_size', type=int, default=256,
                        help='Student model hidden size (default: 256)')
    parser.add_argument('--student_num_layers', type=int, default=6,
                        help='Student model number of layers (default: 6)')
    parser.add_argument('--student_num_heads', type=int, default=8,
                        help='Student model number of attention heads (default: 8)')

    current_time = datetime.datetime.now()
    parser.add_argument('--run_name', default=f'run_{current_time.strftime("%Y%m%d_%H%M%S")}',
                        help='Run name for logging and checkpoint naming')

    return parser


def main():
    parser = create_training_argument_parser()
    args = parser.parse_args()

    args = merge_with_default_config(args, defaultconfig)

    trainer_class = TRAINER_MAP.get(args.mode)
    if trainer_class is None:
        raise ValueError(f"Unknown training mode: {args.mode}. Supported modes: {VALID_MODES}")

    local_print(args)

    trainer = trainer_class(args)
    trainer.train()


if __name__ == '__main__':
    main()
