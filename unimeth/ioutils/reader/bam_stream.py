"""Sequential BAM record reading for streaming inference."""

from dataclasses import dataclass
from typing import Iterator, Literal

import pysam


BamMode = Literal["auto", "aligned", "unaligned"]
ResolvedBamMode = Literal["aligned", "unaligned"]
_CIGAR_OPERATION_COUNT = 10
_CIGAR_CLIP_OPERATIONS = frozenset((4, 5))


@dataclass(frozen=True)
class BamStreamItem:
    """One accepted BAM record and the two IDs used downstream."""

    output_record_key: int
    signal_read_id: str
    bam_record: pysam.AlignedSegment


@dataclass
class BamStreamStats:
    """Mutually exclusive BAM filtering counters for one sequential pass."""

    total_records: int = 0
    filtered_unmapped: int = 0
    filtered_secondary: int = 0
    filtered_duplicate: int = 0
    filtered_supplementary: int = 0
    filtered_mapq: int = 0
    filtered_identity: int = 0
    filtered_chromosome: int = 0
    yielded_records: int = 0

    @property
    def filtered_records(self) -> int:
        return (
            self.filtered_unmapped
            + self.filtered_secondary
            + self.filtered_duplicate
            + self.filtered_supplementary
            + self.filtered_mapq
            + self.filtered_identity
            + self.filtered_chromosome
        )


def _parse_chromosome_filter(value: str) -> tuple[str, frozenset[str]]:
    if value.count("|") != 1:
        return "exclude", frozenset()
    excluded, included = value.split("|", 1)
    if excluded and included:
        return "exclude", frozenset()
    if included:
        return "include", frozenset(included.split(","))
    if excluded:
        return "exclude", frozenset(excluded.split(","))
    return "exclude", frozenset()


def get_signal_read_id(bam_record: pysam.AlignedSegment) -> str:
    """Return the parent signal ID when present, otherwise the BAM query name."""
    if bam_record.has_tag("pi"):
        parent_id = bam_record.get_tag("pi")
        if parent_id is not None and str(parent_id):
            return str(parent_id)

    query_name = bam_record.query_name
    if query_name is None or not str(query_name):
        raise ValueError("BAM record has neither a non-empty pi tag nor a query name")
    return str(query_name)


def resolve_bam_mode(
    requested_mode: BamMode,
    bam_file: pysam.AlignmentFile,
) -> tuple[ResolvedBamMode, bool]:
    """Resolve ``auto`` from header references without scanning BAM records."""
    if requested_mode not in ("auto", "aligned", "unaligned"):
        raise ValueError(f"unsupported BAM mode: {requested_mode}")
    if requested_mode != "auto":
        return requested_mode, False

    return ("aligned" if bam_file.nreferences > 0 else "unaligned"), True


def resolve_bam_mode_from_path(
    bam_path: str,
    requested_mode: BamMode = "auto",
    threads: int = 1,
) -> tuple[ResolvedBamMode, bool]:
    """Resolve a BAM mode from its header without scanning any records."""
    with pysam.AlignmentFile(
        bam_path,
        "rb",
        check_sq=False,
        threads=max(1, int(threads or 1)),
    ) as bam_file:
        return resolve_bam_mode(requested_mode, bam_file)


def calculate_alignment_identity(bam_record: pysam.AlignedSegment) -> float:
    """Calculate CIGAR identity using the same definition as ccsmeth."""
    cigar_stats, _ = bam_record.get_cigar_stats()
    try:
        aligned_length = sum(
            cigar_stats[index]
            for index in range(_CIGAR_OPERATION_COUNT)
            if index not in _CIGAR_CLIP_OPERATIONS
        )
        matched_length = cigar_stats[0] + cigar_stats[7]
    except IndexError:
        return 0.0

    if aligned_length <= 0:
        return 0.0
    return matched_length / float(aligned_length)


class BamStreamReader:
    """Stream BAM records in file order without building a read-ID index."""

    def __init__(
        self,
        bam_path: str,
        bam_mode: BamMode = "auto",
        mapq: int = 1,
        identity: float = 0.0,
        no_supplementary: bool = False,
        skip_unmapped: bool = True,
        chromosome_filter: str = "|",
        threads: int = 1,
    ):
        if bam_mode not in ("auto", "aligned", "unaligned"):
            raise ValueError(f"unsupported BAM mode: {bam_mode}")
        if mapq < 0:
            raise ValueError("mapq must be >= 0")
        if not 0.0 <= identity <= 1.0:
            raise ValueError("identity must be between 0.0 and 1.0")

        self.bam_path = bam_path
        self.requested_bam_mode = bam_mode
        self.mapq = mapq
        self.identity = identity
        self.no_supplementary = no_supplementary
        self.skip_unmapped = skip_unmapped
        self.chromosome_mode, self.chromosomes = _parse_chromosome_filter(
            chromosome_filter
        )
        self.threads = max(1, int(threads or 1))

        self.bam_mode: ResolvedBamMode | None = None
        self.mode_was_auto_detected = False
        self.stats = BamStreamStats()

    def _filter_reason(self, bam_record: pysam.AlignedSegment) -> str | None:
        if self.bam_mode != "aligned":
            return None

        if bam_record.is_unmapped and self.skip_unmapped:
            return "filtered_unmapped"
        if bam_record.is_secondary:
            return "filtered_secondary"
        if bam_record.is_duplicate:
            return "filtered_duplicate"
        if bam_record.is_supplementary and self.no_supplementary:
            return "filtered_supplementary"

        # MAPQ and identity are undefined for unmapped records. When the user
        # explicitly keeps them, only the flag filters above apply.
        if bam_record.is_unmapped:
            return None
        if bam_record.mapping_quality < self.mapq:
            return "filtered_mapq"
        if self.identity > 0.0 and calculate_alignment_identity(bam_record) < self.identity:
            return "filtered_identity"
        reference_name = bam_record.reference_name
        if self.chromosome_mode == "include" and reference_name not in self.chromosomes:
            return "filtered_chromosome"
        if self.chromosome_mode == "exclude" and reference_name in self.chromosomes:
            return "filtered_chromosome"
        return None

    def __iter__(self) -> Iterator[BamStreamItem]:
        self.stats = BamStreamStats()
        with pysam.AlignmentFile(
            self.bam_path,
            "rb",
            check_sq=False,
            threads=self.threads,
        ) as bam_file:
            self.bam_mode, self.mode_was_auto_detected = resolve_bam_mode(
                self.requested_bam_mode,
                bam_file,
            )

            while True:
                record_offset = bam_file.tell()
                try:
                    bam_record = next(bam_file)
                except StopIteration:
                    break

                self.stats.total_records += 1
                filter_reason = self._filter_reason(bam_record)
                if filter_reason is not None:
                    setattr(
                        self.stats,
                        filter_reason,
                        getattr(self.stats, filter_reason) + 1,
                    )
                    continue

                item = BamStreamItem(
                    output_record_key=int(record_offset),
                    signal_read_id=get_signal_read_id(bam_record),
                    bam_record=bam_record,
                )
                self.stats.yielded_records += 1
                yield item


class BamOffsetReader:
    """Read individual BAM records by their stable virtual file offsets."""

    def __init__(self, bam_path: str, threads: int = 1):
        self.bam_path = bam_path
        self._bam_file = pysam.AlignmentFile(
            bam_path,
            "rb",
            check_sq=False,
            threads=max(1, int(threads or 1)),
        )

    def get_record(self, output_record_key: int) -> pysam.AlignedSegment:
        offset = int(output_record_key)
        if offset < 0:
            raise ValueError("output_record_key must be a non-negative BAM offset")
        if self._bam_file is None:
            raise RuntimeError("BAM offset reader is closed")
        self._bam_file.seek(offset)
        try:
            return next(self._bam_file)
        except StopIteration as exc:
            raise KeyError(f"No BAM record at virtual offset {offset}") from exc

    def close(self) -> None:
        if self._bam_file is not None:
            self._bam_file.close()
            self._bam_file = None

    def __enter__(self) -> "BamOffsetReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
