"""User-facing modification names, with legacy names kept at the boundary.

Model tokens, stored training labels, and configuration field names do not
depend on these display names.
"""
import argparse


FREQUENCY_TYPES = ("5mC", "5mCpG", "5mCHG", "5mCHH", "6mA")
DEFAULT_FREQUENCY_TYPES = FREQUENCY_TYPES[1:]
INTERNAL_FREQUENCY_TYPES = {
    "5mC": "5mC", "5mCpG": "CpG", "5mCHG": "CHG", "5mCHH": "CHH", "6mA": "m6A",
}
PUBLIC_FREQUENCY_TYPES = {value: key for key, value in INTERNAL_FREQUENCY_TYPES.items()}
_FREQUENCY_ALIASES = {**{name: name for name in FREQUENCY_TYPES}, **PUBLIC_FREQUENCY_TYPES}
_BASE_ALIASES = {
    **{name: "5mC" for name in ("5mC", "5mCpG", "5mCHG", "5mCHH", "CpG", "CHG", "CHH", "m", "C+m")},
    **{name: "6mA" for name in ("6mA", "m6A", "a", "A+a")},
}


def normalize_frequency_type(value):
    try:
        return _FREQUENCY_ALIASES[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"Unknown modification {value!r}; choose from {', '.join(FREQUENCY_TYPES)}"
        ) from exc


def normalize_base_type(value):
    """Normalize chemical types for tools that do not filter sequence motifs."""
    try:
        return _BASE_ALIASES[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Unknown modification {value!r}; choose 5mC or 6mA") from exc


def add_methylation_flags(parser):
    """Expose consistent flags without changing the model's Namespace fields."""
    for public, legacy, field, description in (
        ("5mCpG", "cpg", "cpg", "5mC detection at CpG sites"),
        ("5mCHG", "chg", "chg", "5mC detection at CHG sites"),
        ("5mCHH", "chh", "chh", "5mC detection at CHH sites"),
        ("6mA", "m6A", "m6A", "6mA detection"),
    ):
        parser.add_argument(f"--{public}", dest=field, type=int, default=0,
                            help=f"Enable {description} (1=yes)")
        parser.add_argument(f"--{legacy}", dest=field, type=int, default=argparse.SUPPRESS,
                            help=argparse.SUPPRESS)
