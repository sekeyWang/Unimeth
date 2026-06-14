"""
Common utility functions for UniMeth.
"""
import shutil
import multiprocessing
from pathlib import Path

# Lazy import accelerate to avoid import-time side effects
def _get_partial_state():
    """Get or create PartialState (lazy initialization)."""
    from accelerate.state import PartialState
    return PartialState()


def local_print(message, flush: bool = False):
    """
    Print message only on the local main process in distributed training.
    
    This is a common pattern in distributed training to avoid duplicate
    output from multiple processes. For non-distributed environments,
    it behaves like normal print.
    
    Args:
        message: Message to print (will be converted to string)
        flush: Whether to flush the output buffer immediately
        
    Example:
        >>> local_print("Training started")
        >>> local_print(f"Epoch {epoch}, Loss: {loss:.4f}")
    """
    try:
        state = _get_partial_state()
        if state.is_local_main_process:
            print(str(message), flush=flush)
    except Exception:
        # Fallback: if accelerate is not available or fails, print anyway
        print(str(message), flush=flush)


def token2seq(tokens, vocab):
    """
    Convert token indices to sequence string.
    
    Args:
        tokens: List of token indices
        vocab: Vocabulary list
    
    Returns:
        Sequence string
    """
    seq = [vocab[x] for x in tokens if x != -100]
    seq = ''.join(seq)
    return seq


def get_cpu_count() -> int:
    """Get the number of CPUs, with fallback."""
    return multiprocessing.cpu_count()


def make_output_path(output_path: str, create_parent: bool = True) -> Path:
    """
    Create and return a Path object, optionally creating parent directories.
    
    Args:
        output_path: Output file/directory path string
        create_parent: Whether to create parent directories
        
    Returns:
        Path object
    """
    path = Path(output_path)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def merge_rank_files(
    output_path: Path,
    num_processes: int,
    remove_temp: bool = True,
    verbose: bool = True
) -> None:
    """
    Merge all rank-specific output files into one.
    
    Args:
        output_path: Final output file path
        num_processes: Total number of processes (ranks)
        remove_temp: Whether to remove temporary rank files after merging
        verbose: Whether to print progress messages
    """
    if verbose:
        local_print(f"Merging output files...")
    
    with open(output_path, 'wb') as outfile:
        for rank in range(num_processes):
            rank_file = output_path.parent / f"{output_path.stem}_rank{rank}{output_path.suffix}"
            if rank_file.exists():
                with open(rank_file, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
                if remove_temp:
                    rank_file.unlink()
    
    if verbose:
        local_print(f"Merged: {output_path}")
