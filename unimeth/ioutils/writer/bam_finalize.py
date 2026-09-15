"""
Finalize BAM part files produced during inference.
"""
import os
from pathlib import Path


def normalize_bam_path(bam_path: str | Path) -> Path:
    """Return a BAM output path with a .bam suffix."""
    path = Path(bam_path)
    if path.suffix == '.bam':
        return path
    return Path(f"{path}.bam")


def bam_part_path(bam_path: str | Path, rank: int, part_suffix: str | None = None) -> Path:
    """Return the rank-specific BAM part path for a final BAM path."""
    path = normalize_bam_path(bam_path)
    suffix = f"_{part_suffix}" if part_suffix else ""
    return path.with_name(f"{path.stem}_rank{rank}{suffix}{path.suffix}")


def bam_part_glob(bam_path: str | Path) -> str:
    """Return the glob pattern for rank-specific BAM part paths."""
    path = normalize_bam_path(bam_path)
    return str(path.with_name(f"{path.stem}_rank*{path.suffix}"))


def merged_unsorted_path(bam_path: str | Path) -> Path:
    """Return the temporary unsorted BAM path for a final BAM path."""
    path = normalize_bam_path(bam_path)
    return path.with_name(f"{path.stem}.merged_unsorted{path.suffix}")


def bam_sorting_path(bam_path: str | Path) -> Path:
    """Return the visible temporary path used while sorting a streamed BAM."""
    path = normalize_bam_path(bam_path)
    return path.with_name(f"{path.stem}.unimeth-sorting{path.suffix}")


def finalize_single_bam(
    bam_path: str,
    work_path: str,
    threads: int = 8,
    sort_and_index: bool = True,
    index_threads: int | None = None,
) -> None:
    """Finalize the sole writer's BAM stream without merging rank parts."""
    import pysam

    final_path = normalize_bam_path(bam_path)
    threads = max(1, int(threads or 1))
    if sort_and_index:
        work_path = Path(work_path)
        if work_path.resolve() == final_path.resolve():
            sorting_path = bam_sorting_path(final_path)
            if sorting_path.exists():
                sorting_path.unlink()
            pysam.sort(
                "-@",
                str(threads),
                "-o",
                str(sorting_path),
                str(work_path),
            )
            os.replace(sorting_path, final_path)
        else:
            pysam.sort("-@", str(threads), "-o", str(final_path), str(work_path))
            os.remove(work_path)
        if index_threads is None:
            pysam.index(str(final_path))
        else:
            index_threads = max(1, min(threads, int(index_threads or 1)))
            pysam.index("-@", str(index_threads), str(final_path))
        return
    if Path(work_path).resolve() != final_path.resolve():
        os.replace(work_path, final_path)


def finalize_part_bams(
    bam_path: str,
    part_files: list[str],
    threads: int = 8,
    sort_and_index: bool = True,
) -> None:
    """Merge, sort, and index BAM part files using pysam."""
    if not part_files:
        return

    import pysam

    final_path = normalize_bam_path(bam_path)
    threads = max(1, int(threads or 1))

    if len(part_files) == 1:
        finalize_single_bam(
            str(final_path),
            part_files[0],
            threads=threads,
            sort_and_index=sort_and_index,
        )
    else:
        if sort_and_index:
            merged_unsorted = merged_unsorted_path(final_path)
            pysam.merge("-@", str(threads), "-f", str(merged_unsorted), *part_files)
            pysam.sort("-@", str(threads), "-o", str(final_path), str(merged_unsorted))
            os.remove(merged_unsorted)
        else:
            pysam.merge("-@", str(threads), "-f", str(final_path), *part_files)
        for part_file in part_files:
            os.remove(part_file)

    if sort_and_index and len(part_files) > 1:
        pysam.index(str(final_path))
