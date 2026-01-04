import subprocess
import os
import unimeth
import argparse

def start():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config-file', type=str, default=None)
    parser.add_argument('--num-processes', type=int, default=None)
    parser.add_argument('--multi-gpu', action='store_true')    
    parser.add_argument('--mixed-precision', type=str, default=None)
    accelerate_args, training_args = parser.parse_known_args()
    unimeth_path = os.path.join(
        os.path.dirname(unimeth.__file__),
        'unimeth.py'
    )
    cmd = ['accelerate', 'launch']
    if accelerate_args.config_file:
        cmd.extend(['--config_file', accelerate_args.config_file])
    if accelerate_args.num_processes:
        cmd.extend(['--num_processes', str(accelerate_args.num_processes)])
    if accelerate_args.multi_gpu:
        cmd.append('--multi_gpu')
    if accelerate_args.mixed_precision:
        cmd.extend(['--mixed_precision', accelerate_args.mixed_precision])
    
    cmd.extend([unimeth_path])
    cmd.extend(training_args)
    subprocess.run(cmd)

if __name__ == "__main__":
    start()
