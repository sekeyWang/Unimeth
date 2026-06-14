"""
Fine-tuning trainer for UniMeth.
"""
import functools
import torch
from unimeth.model.loader import load_model
from unimeth.model.datasets import Pod5BamDataset, ValidationDataset, MultiFileDataset, collate_fn
from unimeth.eval import compute_metrics
from unimeth.config import tokenizer
from .trainer_base import BaseTrainer


class FinetuneTrainer(BaseTrainer):
    """Trainer for supervised fine-tuning with WGBS labels."""
    
    def __init__(self, args):
        super().__init__(args, mode='finetune')
        self.data_collator = functools.partial(collate_fn, 'finetune')
        self.compute_metrics_fn = self._create_metrics_fn()
    
    def _create_metrics_fn(self):
        """Create metrics function with tokenizer."""
        def metrics_fn(pred):
            return compute_metrics(pred, tokenizer)
        return metrics_fn
    
    def create_model(self):
        """Create model for fine-tuning."""
        # config can be: None (default), config name (str), or path to JSON
        config = getattr(self.args, 'config', None)
        
        self.model = load_model(
            config=config,
            model_path=self.args.model_dir,
            mode='finetune',
            plant=getattr(self.args, 'plant', False)
        )
        
        # Set main input name for Trainer
        setattr(self.model, 'main_input_name', 'decoder_input_ids')
    
    def create_datasets(self):
        """Create fine-tuning datasets."""
        bam_dirs, train_pod5_dirs, val_pod5_dirs = self._parse_data_dirs()
        
        # Set thresholds for fine-tuning
        self.args.negative_thres = 20
        self.args.positive_thres = 80
        
        # Create training datasets (possibly multiple)
        datasets = []
        for i in range(len(bam_dirs)):
            dataset = Pod5BamDataset(
                train_pod5_dirs[i],
                bam_dirs[i],
                self.args
            )
            datasets.append(dataset)
        
        # Combine datasets
        if len(datasets) == 1:
            self.train_dataset = datasets[0]
        else:
            self.train_dataset = MultiFileDataset(datasets)
        
        # Create validation dataset
        self.eval_dataset = ValidationDataset(
            pod5_dirs=val_pod5_dirs,
            bam_dirs=bam_dirs,
            args=self.args
        )


class CalibrationTrainer(FinetuneTrainer):
    """Trainer for calibration on target dataset."""
    
    def __init__(self, args):
        # Initialize with finetune mode but use 'calibration' for paths
        super().__init__(args)
        self.mode = 'calibration'
        # Recreate training args with correct mode
        self.training_args = self._create_training_args()
