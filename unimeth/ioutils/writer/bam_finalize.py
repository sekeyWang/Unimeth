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


def bam_part_path(bam_path: str | Path, rank: int) -> Path:
    """Return the rank-specific BAM part path for a final BAM path."""
    path = normalize_bam_path(bam_path)
    return path.with_name(f"{path.stem}.part_{rank}{path.suffix}")


def bam_part_glob(bam_path: str | Path) -> str:
    """Return the glob pattern for rank-specific BAM part paths."""
    path = normalize_bam_path(bam_path)
    return str(path.with_name(f"{path.stem}.part_*{path.suffix}"))


def merged_unsorted_path(bam_path: str | Path) -> Path:
    """Return the temporary unsorted BAM path for a final BAM path."""
    path = normalize_bam_path(bam_path)
    return path.with_name(f"{path.stem}.merged_unsorted{path.suffix}")


def bam_has_references(bam_path: str) -> bool:
    """Return whether a BAM header contains reference sequences."""
    import pysam

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam_file:
        return bam_file.nreferences > 0


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

    if len(part_files) == 1:
        os.rename(part_files[0], final_path)
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

    if sort_and_index:
        pysam.index(str(final_path))
