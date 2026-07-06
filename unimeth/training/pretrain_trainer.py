"""
Pretraining trainer for UniMeth.
"""
import functools
import torch
from unimeth.model.loader import load_model
from unimeth.model.datasets import Pod5BamDataset, ValidationDataset, MultiFileDataset, collate_fn
from unimeth.config import ModelConfig
from .trainer_base import BaseTrainer


class PretrainTrainer(BaseTrainer):
    """Trainer for self-supervised pretraining."""

    def __init__(self, args):
        super().__init__(args, mode='pretrain')
        self.data_collator = functools.partial(collate_fn, 'pretrain')
        self.compute_metrics_fn = None

    def create_model(self):
        """Create model for pretraining."""
        # config can be: None (default), config name (str), or path to JSON
        config = getattr(self.args, 'config', None)

        self.model = load_model(
            config=config,
            model_path=getattr(self.args, 'model_dir', None),
            mode='pretrain',
        )
        if getattr(self.args, 'model_dir', None) is not None:
            print(f"Loaded pretrained model from {self.args.model_dir}")

    def create_datasets(self):
        """Create pretraining datasets."""
        bam_dirs, train_pod5_dirs, val_pod5_dirs = self._parse_data_dirs()

        # Create training datasets (possibly multiple bam/pod5 pairs)
        datasets = []
        for i in range(len(bam_dirs)):
            dataset = Pod5BamDataset(
                train_pod5_dirs[i],
                bam_dirs[i],
                self.args
            )
            datasets.append(dataset)

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
