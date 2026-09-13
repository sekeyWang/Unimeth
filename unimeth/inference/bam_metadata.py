"""Lightweight BAM header metadata used only by inference startup."""

from __future__ import annotations

import re

from packaging.version import Version


_DORADO_VERSION_PREFIX = re.compile(r"^[vV]?(\d+\.\d+\.\d+)")


def extract_dorado_version_from_header(header: dict) -> str | None:
    """Extract one normalized Dorado basecaller version from a BAM header."""
    versions = set()
    for program in header.get("PG", []):
        program_id = str(program.get("ID", "")).lower()
        program_name = str(program.get("PN", "")).lower()
        if (
            program_name != "dorado"
            or not re.fullmatch(r"basecaller(?:_\d+)?", program_id)
        ):
            continue

        match = _DORADO_VERSION_PREFIX.match(str(program.get("VN", "")))
        if match:
            versions.add(str(Version(match.group(1))))

    if len(versions) > 1:
        version_list = ", ".join(sorted(versions, key=Version))
        raise ValueError(
            "multiple Dorado basecaller versions found in BAM header: "
            f"{version_list}"
        )
    return next(iter(versions), None)


def detect_dorado_version_from_bam(bam_path: str) -> str | None:
    """Read only a BAM header and return its Dorado basecaller version."""
    import pysam

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam_file:
        return extract_dorado_version_from_header(bam_file.header.to_dict())
