"""
Finalize BAM part files produced during inference.
"""
import os


def finalize_part_bams(bam_path: str, part_files: list[str], threads: int = 8) -> None:
    """Merge, sort, and index BAM part files using pysam."""
    if not part_files:
        return

    import pysam

    if len(part_files) == 1:
        os.rename(part_files[0], bam_path)
    else:
        merged_unsorted = bam_path.replace('.bam', '.merged_unsorted.bam')
        pysam.merge("-@", str(threads), "-f", merged_unsorted, *part_files)
        pysam.sort("-@", str(threads), "-o", bam_path, merged_unsorted)
        os.remove(merged_unsorted)
        for part_file in part_files:
            os.remove(part_file)

    pysam.index(bam_path)
