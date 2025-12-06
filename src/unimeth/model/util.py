from transformers import TrainingArguments, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
class Mycallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_local_process_zero:
            print(f'Save model on step {state.global_step}')
        return super().on_save(args, state, control, **kwargs)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        logs['steps'] = state.global_step
        return super().on_log(args, state, control, **kwargs)