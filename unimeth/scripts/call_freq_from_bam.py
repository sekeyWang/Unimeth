"""Count site methylation frequencies from explicit binary modBAM predictions.

The count and ML probability conventions follow ccsmeth's count mode with
effective coverage (no_amb_cov). Uncalled bases never contribute to coverage.
Only the forward molecule's C+m and A+a predictions are supported.
"""
import argparse
import atexit
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import re
import tempfile

from unimeth.config.modification_names import (
    DEFAULT_FREQUENCY_TYPES, FREQUENCY_TYPES, INTERNAL_FREQUENCY_TYPES,
    PUBLIC_FREQUENCY_TYPES, normalize_frequency_type,
)

MOD_TYPES = FREQUENCY_TYPES
TSV_HEADER = "chrom\tposition\tstrand\tmod_type\tcoverage\tmethylated\tunmethylated\tfrequency\n"
_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_H_BASES = frozenset("ACT")


def _normalize_output_format(value):
    return "bed" if value == "bedmethyl" else value


def _output_label(key, combine_cpg=False):
    kind, hap = key
    label = PUBLIC_FREQUENCY_TYPES[kind]
    if combine_cpg and kind == "CpG":
        label += ".combined"
    return label + (f".hp{hap}" if hap else "")


def _haplotype(read, tag):
    try:
        hap = int(read.get_tag(tag))
    except (KeyError, ValueError, TypeError, OverflowError):
        return 0
    return hap if hap in (1, 2) else 0


@dataclass(frozen=True)
class CountOptions:
    mod_types: tuple[str, ...]
    prob_cf: float
    refsites_only: bool
    mapq: int
    no_hap: bool = False
    hap_tag: str = "HP"
    combine_cpg: bool = False


def _probability(ml):
    # ccsmeth calls ML=128 methylated; preserve its probability quantization
    # and confidence-boundary rounding for differential validation.
    return round(ml / 256 + 0.000001, 6) if ml > 0 else 0.0


def _context(sequence, offset, position, strand, base):
    """Classify the reference on the prediction's strand; H excludes N."""
    index = position - offset
    canonical = base if strand == "+" else base.translate(_COMPLEMENT)
    if not 0 <= index < len(sequence) or sequence[index] != canonical:
        return None
    if base == "A":
        return "m6A"
    if strand == "+":
        motif = sequence[index:index + 3]
    else:
        motif = sequence[max(0, index - 2):index + 1].translate(_COMPLEMENT)[::-1]
    if len(motif) >= 2 and motif[:2] == "CG":
        return "CpG"
    if len(motif) == 3 and motif[1] in _H_BASES:
        if motif[2] == "G":
            return "CHG"
        if motif[2] in _H_BASES:
            return "CHH"
    return None


def _reference_base_matches(sequence, offset, position, strand, base):
    index = position - offset
    canonical = base if strand == "+" else base.translate(_COMPLEMENT)
    return 0 <= index < len(sequence) and sequence[index] == canonical


def _explicit_calls(read, options):
    """Decode scored predictions; fail on stale or competing mod tags."""
    if read.has_tag("MN") and read.get_tag("MN") != read.query_length:
        raise ValueError(f"Read {read.query_name!r}: MN does not match SEQ length")
    if not (read.has_tag("MM") or read.has_tag("Mm")):
        return {}
    # MM without ML is legal SAM, but has no probabilities to count.
    ml_tag = "ML" if read.has_tag("ML") else "Ml"
    if not read.has_tag(ml_tag):
        return {}
    modifications = read.modified_bases_forward
    if modifications is None:
        raise ValueError(f"Read {read.query_name!r}: invalid MM/ML tags")
    if sum(len(values) for values in modifications.values()) != len(read.get_tag(ml_tag)):
        raise ValueError(f"Read {read.query_name!r}: MM/ML lengths differ")
    targets = set()
    if any(kind != "m6A" for kind in options.mod_types):
        targets.add("C")
    if "m6A" in options.mod_types:
        targets.add("A")
    result = {}
    for (base, strand, code), calls in modifications.items():
        if base not in targets:
            continue
        expected = "m" if base == "C" else "a"
        if strand != 0 or code != expected:
            raise ValueError(
                f"Read {read.query_name!r}: only binary C+m / A+a modifications "
                "are supported; competing or opposite-strand modifications cannot be counted"
            )
        positions = set()
        for position, ml in calls:
            if position in positions:
                raise ValueError(f"Read {read.query_name!r}: duplicate modification position")
            positions.add(position)
            if ml < 0:  # Unknown probability, per pysam's modified_bases API.
                continue
            result[(base, position)] = ml
    return result


def _count_region(bam, reference, region, options):
    """Aggregate integer counts for one disjoint interval of output anchors."""
    chrom, start, end = region
    chrom_length = bam.get_reference_length(chrom)
    offset = max(0, start - 2)
    sequence = reference.fetch(chrom, offset, min(chrom_length, end + 2)).upper()
    # A negative CpG call at end belongs to the C anchor at end-1. Reads
    # starting on that G must be fetched even if they do not cover the C.
    fetch_end = min(chrom_length, end + int(options.combine_cpg))
    counts = {}
    for read in bam.fetch(chrom, start, fetch_end):
        if (read.is_unmapped or read.is_secondary or read.is_supplementary
                or read.is_duplicate or read.is_qcfail or read.mapping_quality < options.mapq):
            continue
        if not read.query_sequence:
            continue
        calls = _explicit_calls(read, options)
        if not calls:
            continue
        hap = _haplotype(read, options.hap_tag) if not options.no_hap else 0
        groups = (0, hap) if hap else (0,)
        # This array is in BAM SEQ orientation. modified_bases_forward uses
        # original molecule orientation, so reverse reads need one inversion.
        reference_positions = read.get_reference_positions(full_length=True)
        length = read.query_length
        strand = "-" if read.is_reverse else "+"
        for (base, forward_position), ml in calls.items():
            query_position = length - 1 - forward_position if read.is_reverse else forward_position
            if not 0 <= query_position < len(reference_positions):
                raise ValueError(f"Read {read.query_name!r}: modification outside SEQ")
            position = reference_positions[query_position]
            if position is None or not start <= position < fetch_end:
                continue
            matches_reference = _reference_base_matches(sequence, offset, position, strand, base)
            if options.refsites_only and not matches_reference:
                continue
            kinds = []
            if base == "C" and "5mC" in options.mod_types:
                kinds.append("5mC")
            if base == "A" and "m6A" in options.mod_types:
                kinds.append("m6A")
            # Context-specific selection always refers to the reference motif.
            # All-C selection includes read C calls with unknown/mismatched
            # reference contexts unless refsites_only was requested.
            if base == "C":
                kind = _context(sequence, offset, position, strand, base)
                if kind in options.mod_types:
                    kinds.append(kind)
            if not kinds:
                continue
            probability = _probability(ml)
            if abs(probability - (1 - probability)) < options.prob_cf:
                continue
            for kind in kinds:
                output_position, output_strand = position, strand
                if options.combine_cpg and kind == "CpG":
                    output_position -= int(read.is_reverse)
                    output_strand = "+"
                # Ownership follows this output's coordinate. Other types in
                # the right overlap retain their own coordinates and region.
                if not start <= output_position < end:
                    continue
                for group in groups:
                    key = kind, output_position, output_strand, group
                    values = counts.setdefault(key, [0, 0])
                    values[0] += 1
                    values[1] += int(probability > 0.5)
    return counts


_worker_bam = None
_worker_reference = None
_worker_options = None


def _init_worker(input_bam, reference, options):
    import pysam

    global _worker_bam, _worker_reference, _worker_options
    _worker_bam = pysam.AlignmentFile(input_bam, "rb")
    _worker_reference = pysam.FastaFile(reference)
    _worker_options = options
    atexit.register(_worker_bam.close)
    atexit.register(_worker_reference.close)


def _worker_count(region):
    return _count_region(_worker_bam, _worker_reference, region, _worker_options)


def _regions(contigs, lengths, chunk_len):
    for chrom in contigs:
        for start in range(0, lengths[chrom], chunk_len):
            yield chrom, start, min(lengths[chrom], start + chunk_len)


def _bounded_results(executor, regions, workers):
    """Keep at most 2*workers region dictionaries and futures in flight."""
    pending = deque()
    iterator = iter(regions)
    for _ in range(workers * 2):
        region = next(iterator, None)
        if region is None:
            break
        pending.append((region, executor.submit(_worker_count, region)))
    while pending:
        region, future = pending.popleft()
        yield region, future.result()
        region = next(iterator, None)
        if region is not None:
            pending.append((region, executor.submit(_worker_count, region)))


def _write_counts(handles, region, counts, min_cov, output_format):
    chrom = region[0]
    written = set()
    for (kind, position, strand, hap), (coverage, methylated) in sorted(
        counts.items(), key=lambda item: (item[0][1], item[0][2], item[0][0], item[0][3])
    ):
        if coverage < min_cov:
            continue
        fraction = methylated / coverage
        if output_format == "bed":
            # ccsmeth's 11-column bedMethyl with one-base intervals. Both
            # coverage columns agree. Combined CpG uses the positive C anchor.
            fields = (chrom, position, position + 1, ".", coverage, strand,
                      position, position + 1, "0,0,0", coverage,
                      int(round(fraction * 100 + 0.001)))
        else:
            fields = (chrom, position, strand, PUBLIC_FREQUENCY_TYPES[kind], coverage, methylated,
                      coverage - methylated, f"{fraction:.6f}")
        key = kind, hap
        handles[key].write("\t".join(map(str, fields)) + "\n")
        written.add(key)
    return written


def call_frequency(input_bam, reference, output_prefix, *, mod_types=DEFAULT_FREQUENCY_TYPES,
                   prob_cf=0.0, refsites_only=False, mapq=1, min_cov=1, threads=4,
                   chunk_len=500_000, contigs=None, sort=False,
                   output_format="bed", compress=False, no_hap=False, hap_tag="HP",
                   combine_cpg=False):
    """Write type and populated hap files; return {type[.hpN]: output_path}.

    Input BAM and FASTA must already be indexed. Default order follows BAM SQ;
    sort (also implied by compress) orders contigs lexicographically. Memory
    stores region counts and current read positions, never genome-wide scores.
    """
    import pysam

    if not mod_types:
        raise ValueError("Select at least one modification type")
    try:
        requested = {normalize_frequency_type(kind) for kind in mod_types}
    except argparse.ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc
    selected = tuple(INTERNAL_FREQUENCY_TYPES[kind] for kind in MOD_TYPES if kind in requested)
    if combine_cpg and "CpG" not in selected:
        raise ValueError("combine_cpg requires 5mCpG in mod_types")
    if not 0 <= prob_cf <= 1:
        raise ValueError("prob_cf must be between 0 and 1")
    if not 0 <= mapq <= 255:
        raise ValueError("mapq must be between 0 and 255")
    if min_cov < 1 or threads < 1 or chunk_len < 1:
        raise ValueError("min_cov, threads, and chunk_len must be positive")
    if not isinstance(hap_tag, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9]", hap_tag) is None:
        raise ValueError("hap_tag must be a two-character SAM tag")
    output_format = _normalize_output_format(output_format)
    if output_format not in ("bed", "tsv"):
        raise ValueError("output_format must be bed or tsv")
    input_bam, reference = str(input_bam), str(reference)
    if not Path(input_bam).is_file() or not Path(reference).is_file():
        raise ValueError("Input BAM and reference FASTA must exist")
    if not Path(reference + ".fai").is_file():
        raise ValueError("Reference FASTA index (.fai) is required; run samtools faidx first")
    with ExitStack() as stack:
        bam = stack.enter_context(pysam.AlignmentFile(input_bam, "rb"))
        if not bam.has_index():
            raise ValueError("Coordinate-sorted BAM index is required; run samtools index first")
        if bam.header.to_dict().get("HD", {}).get("SO") in ("unsorted", "queryname"):
            raise ValueError("Input BAM must be coordinate sorted")
        fa = stack.enter_context(pysam.FastaFile(reference))
        lengths = dict(zip(bam.references, bam.lengths))
        names = list(dict.fromkeys(contigs)) if contigs is not None else list(bam.references)
        if not names:
            raise ValueError("No reference contigs selected")
        for name in names:
            if name not in lengths or name not in fa.references:
                raise ValueError(f"Contig {name!r} is missing from BAM or reference")
            if lengths[name] != fa.get_reference_length(name):
                raise ValueError(f"Contig {name!r}: BAM and reference lengths differ")
        if sort or compress:
            names.sort()
        prefix = Path(output_prefix).resolve()
        extension = "bed" if output_format == "bed" else "tsv"
        haps = (0,) if no_hap else (0, 1, 2)
        keys = [(kind, hap) for kind in selected for hap in haps]
        outputs = {key: str(prefix) + f".{_output_label(key, combine_cpg)}.{extension}" + (".gz" if compress else "")
                   for key in keys}
        for path in outputs.values():
            if Path(path).exists() or Path(path + ".csi").exists():
                raise FileExistsError(f"Output already exists: {path}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        options = CountOptions(selected, prob_cf, refsites_only, mapq, no_hap, hap_tag, combine_cpg)
        # Stage outputs until every worker succeeds, so malformed input never
        # leaves files that appear to be a successful genome-wide result.
        with tempfile.TemporaryDirectory(prefix=".call-freq-", dir=prefix.parent) as staging:
            staged = {key: Path(staging) / f"{_output_label(key, combine_cpg)}.{extension}" for key in keys}
            written = set()
            with ExitStack() as file_stack:
                handles = {key: file_stack.enter_context(path.open("w", encoding="utf-8", newline=""))
                           for key, path in staged.items()}
                if output_format == "tsv":
                    for handle in handles.values():
                        handle.write(TSV_HEADER)
                regions = _regions(names, lengths, chunk_len)
                if threads == 1:
                    for region in regions:
                        written.update(_write_counts(handles, region, _count_region(bam, fa, region, options),
                                                     min_cov, output_format))
                else:
                    # Spawn keeps the parent BAM/FASTA handles out of workers.
                    with ProcessPoolExecutor(
                        max_workers=threads, mp_context=multiprocessing.get_context("spawn"),
                        initializer=_init_worker, initargs=(input_bam, reference, options),
                    ) as executor:
                        for region, counts in _bounded_results(executor, regions, threads):
                            written.update(_write_counts(handles, region, counts, min_cov, output_format))
            # A TSV header is not hap data. Only groups with eligible site rows
            # are published; total type files retain the existing empty behavior.
            staged = {key: path for key, path in staged.items() if key[1] == 0 or key in written}
            outputs = {key: outputs[key] for key in staged}
            for key, path in list(staged.items()):
                if compress:
                    compressed = Path(str(path) + ".gz")
                    pysam.tabix_compress(str(path), str(compressed), force=True)
                    if output_format == "bed":
                        pysam.tabix_index(str(compressed), preset="bed", csi=True, force=True)
                    staged[key] = compressed
            published = []
            try:
                for key, path in staged.items():
                    # Staging is on the output filesystem. Linking publishes
                    # atomically and refuses any file created during counting.
                    os.link(path, outputs[key])
                    published.append(Path(outputs[key]))
                    if compress and output_format == "bed":
                        index = outputs[key] + ".csi"
                        os.link(str(path) + ".csi", index)
                        published.append(Path(index))
            except OSError as exc:
                # A publication failure must not strand results that block a
                # retry, or leave a compressed BED without its paired index.
                remaining = []
                for path in reversed(published):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        remaining.append(str(path))
                if remaining:
                    raise OSError("Output publication failed; could not remove: "
                                  + ", ".join(remaining)) from exc
                raise
    return {_output_label(key): path for key, path in outputs.items()}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_bam", required=True, help="Coordinate-sorted, indexed modBAM")
    parser.add_argument("--ref", required=True, help="Reference FASTA with .fai index")
    parser.add_argument("--output", required=True, help="Output prefix; files are PREFIX.TYPE.bed (or .tsv)")
    parser.add_argument("--mod_types", nargs="+", type=normalize_frequency_type,
                        choices=MOD_TYPES, metavar="TYPE", default=list(DEFAULT_FREQUENCY_TYPES),
                        help="Types to count: 5mC (all C), 5mCpG, 5mCHG, 5mCHH, 6mA")
    parser.add_argument("--prob_cf", type=float, default=0.0,
                        help="Minimum |P(modified)-P(unmodified)|; failing calls do not count as coverage")
    parser.add_argument("--mapq", type=int, default=1, help="Minimum mapping quality")
    parser.add_argument("--min_cov", type=int, default=1, help="Minimum effective site coverage")
    parser.add_argument("--refsites_only", action="store_true",
                        help="For 5mC/6mA, require the reference base to match C/A on the call's strand; "
                             "context-specific types always use reference motifs")
    parser.add_argument("--combine_cpg", action="store_true", help="Combine CpG strands")
    parser.add_argument("--hap_tag", default="HP", help="Haplotype tag")
    parser.add_argument("--no_hap", action="store_true", help="Disable haplotype outputs")
    parser.add_argument("--contigs", nargs="+", help="Restrict to these contigs (exact names)")
    parser.add_argument("--threads", type=int, default=4, help="Number of counting processes")
    parser.add_argument("--chunk_len", type=int, default=500_000, help="Reference bases per counting region")
    parser.add_argument("--output_format", type=_normalize_output_format,
                        choices=("bed", "tsv"), default="bed",
                        help="bed: bedMethyl; tsv: tab-separated text")
    parser.add_argument("--sort", action="store_true", help="Sort by chromosome name and position (default: BAM SQ order)")
    parser.add_argument("--gzip", action="store_true", help="Compress")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        outputs = call_frequency(
            args.input_bam, args.ref, args.output, mod_types=args.mod_types, prob_cf=args.prob_cf,
            refsites_only=args.refsites_only, mapq=args.mapq, min_cov=args.min_cov, threads=args.threads,
            chunk_len=args.chunk_len, contigs=args.contigs, sort=args.sort,
            output_format=args.output_format, compress=args.gzip,
            no_hap=args.no_hap, hap_tag=args.hap_tag, combine_cpg=args.combine_cpg,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for kind, path in outputs.items():
        print(f"{kind}\t{path}")


if __name__ == "__main__":
    main()
