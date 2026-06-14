"""
Base trainer class for UniMeth training.
"""
import os
from abc import ABC, abstractmethod
from transformers import TrainingArguments, Trainer
from transformers.trainer_pt_utils import AcceleratorConfig

from unimeth.utils import local_print, MetricsCallback



class BaseTrainer(ABC):
    """
    Base class for pretraining and fine-tuning trainers.
    """
    
    def __init__(self, args, mode: str):
        """
        Initialize base trainer.
        
        Args:
            args: Configuration arguments
            mode: Training mode ('pretrain', 'finetune', or 'calibration')
        """
        self.args = args
        self.mode = mode
        self.training_args = self._create_training_args()
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.data_collator = None
        self.compute_metrics_fn = None
    
    def _create_accelerator_config(self):
        """Create AcceleratorConfig for distributed training."""
        return AcceleratorConfig(
            dispatch_batches=True,
            split_batches=True
        )
    
    def _create_training_args(self):
        """Create TrainingArguments with common settings."""
        accelerator_config = self._create_accelerator_config()
        
        return TrainingArguments(
            # Training args
            learning_rate=self._get_learning_rate(),
            warmup_ratio=0.05 if self.mode == 'pretrain' else 0,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            max_steps=self.args.max_steps,
            weight_decay=0,
            ddp_find_unused_parameters=False,
            accelerator_config=accelerator_config,
            remove_unused_columns=False,
            
            # Dataloader
            bf16=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=2,
            dataloader_drop_last=True,
            
            # Save model
            output_dir=f"models/{self.mode}/{self.args.run_name}",
            overwrite_output_dir=True,
            save_only_model=False,
            save_safetensors=False,
            save_strategy="steps",
            save_steps=self._get_save_steps(),
            
            # Logging
            logging_strategy="steps",
            logging_steps=500,
            
            # Evaluation
            eval_strategy="steps",
            eval_steps=self._get_eval_steps(),
            include_for_metrics=["inputs"] if self.mode != 'pretrain' else False,
            
            # Wandb
            report_to=self._get_report_to(),
            run_name=self.args.run_name
        )
    
    def _get_learning_rate(self):
        """Get learning rate for the mode."""
        rates = {
            'pretrain': 1e-4,
            'finetune': 1e-5,
            'calibration': 1e-5
        }
        return rates.get(self.mode, 1e-5)
    
    def _get_save_steps(self):
        """Get save steps for the mode."""
        steps = {
            'pretrain': 20000,
            'finetune': 5000,
            'calibration': 5000
        }
        return steps.get(self.mode, 5000)
    
    def _get_eval_steps(self):
        """Get evaluation steps for the mode."""
        steps = {
            'pretrain': 10000,
            'finetune': 10000,
            'calibration': 2000
        }
        return steps.get(self.mode, 10000)
    
    def _get_report_to(self):
        """Get report destination for the mode."""
        if self.mode == 'pretrain':
            return "wandb"
        return "none"
    
    def _parse_data_dirs(self):
        """Parse comma-separated data directories."""
        def parse_dirs(dir_str):
            if dir_str is None:
                return []
            if ',' in dir_str:
                return dir_str.split(',')
            return [dir_str]
        
        bam_dirs = parse_dirs(getattr(self.args, 'bam_dir', None))
        train_pod5_dirs = parse_dirs(getattr(self.args, 'train_pod5_dir', None))
        val_pod5_dirs = parse_dirs(getattr(self.args, 'val_pod5_dir', None))
        
        if bam_dirs and train_pod5_dirs and len(bam_dirs) != len(train_pod5_dirs):
            raise ValueError('Need same number of bam files and pod5 directories')
        
        return bam_dirs, train_pod5_dirs, val_pod5_dirs
    
    @abstractmethod
    def create_model(self):
        """Create and load model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def create_datasets(self):
        """Create training and evaluation datasets. Must be implemented by subclasses."""
        pass
    
    def create_trainer(self):
        """Create HuggingFace Trainer instance."""
        # Set wandb project
        os.environ["WANDB_PROJECT"] = self.mode
        
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics_fn,
            callbacks=[MetricsCallback],
        )
        return trainer
    
    def train(self):
        """Run training."""
        self.create_model()
        self.create_datasets()
        
        trainer = self.create_trainer()
        
        # Run initial evaluation for non-pretrain modes
        if self.mode != 'pretrain':
            local_print(trainer.evaluate())
        
        # Train
        trainer.train()
