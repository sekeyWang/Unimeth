"""
Training callbacks for UniMeth.
"""
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers import TrainingArguments


class MetricsCallback(TrainerCallback):
    """Custom callback for model saving and logging."""
    
    def on_save(self, args: TrainingArguments, state: TrainerState, 
                control: TrainerControl, **kwargs):
        """Print message when model is saved."""
        if state.is_local_process_zero:
            print(f'Save model on step {state.global_step}')
        return super().on_save(args, state, control, **kwargs)

    def on_log(self, args: TrainingArguments, state: TrainerState, 
               control: TrainerControl, logs=None, **kwargs):
        """Add step number to logs."""
        if logs is not None:
            logs['steps'] = state.global_step
        return super().on_log(args, state, control, logs=logs, **kwargs)


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback based on evaluation metric."""
    
    def __init__(self, patience: int = 5, metric_name: str = 'eval_loss'):
        self.patience = patience
        self.metric_name = metric_name
        self.best_score = None
        self.counter = 0
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        """Check if training should stop early."""
        if state.is_local_process_zero:
            metrics = kwargs.get('metrics', {})
            if self.metric_name in metrics:
                score = metrics[self.metric_name]
                if self.best_score is None or score < self.best_score:
                    self.best_score = score
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        control.should_training_stop = True
                        print(f"Early stopping triggered at step {state.global_step}")
        return control
