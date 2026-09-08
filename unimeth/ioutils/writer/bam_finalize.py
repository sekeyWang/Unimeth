"""
Finalize BAM part files produced during inference.
"""
import os
import re
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


def _resume_part_sort_key(part_file: str | Path) -> tuple[int, str]:
    """Sort base parts before numbered resume attempts."""
    path = Path(part_file)
    match = re.search(r"_resume(\d+)$", path.stem)
    attempt = int(match.group(1)) if match else -1
    return attempt, path.name


def selected_bam_part_path(part_file: str | Path) -> Path:
    """Return a temporary part path used for resume-safe finalization."""
    path = Path(part_file)
    return path.with_name(f".{path.stem}.completed{path.suffix}")


def _select_bam_records(
    part_files: list[str],
    completed_read_ids: set[str] | None = None,
) -> list[str]:
    """Keep one latest BAM record per read across attempts."""
    import pysam

    selected_paths = []
    selected_read_ids = set()
    for part_file in sorted(part_files, key=_resume_part_sort_key, reverse=True):
        selected_path = selected_bam_part_path(part_file)
        wrote_record = False
        with pysam.AlignmentFile(part_file, "rb", check_sq=False) as input_bam:
            with pysam.AlignmentFile(str(selected_path), "wb", template=input_bam) as output_bam:
                for bam_read in input_bam:
                    read_id = bam_read.query_name
                    if (
                        read_id in selected_read_ids
                        or (
                            completed_read_ids is not None
                            and read_id not in completed_read_ids
                        )
                    ):
                        continue
                    output_bam.write(bam_read)
                    selected_read_ids.add(read_id)
                    wrote_record = True
        if wrote_record:
            selected_paths.append(str(selected_path))
        elif selected_path.exists():
            selected_path.unlink()
    return selected_paths


def select_completed_bam_records(
    part_files: list[str],
    completed_read_ids: set[str],
) -> list[str]:
    """Keep one latest BAM record for each checkpointed read across attempts."""
    return _select_bam_records(part_files, completed_read_ids)


def select_latest_bam_records(part_files: list[str]) -> list[str]:
    """Keep one latest BAM record for every emitted read across attempts."""
    return _select_bam_records(part_files)


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
