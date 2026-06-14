"""
Standalone utility scripts for UniMeth.

These are command-line tools for data conversion, evaluation, and visualization.
They are not imported as modules - use the unified entry point or run directly.

Unified entry point:
    python -m scripts <command> [subcommand] [options]

Direct execution (still supported):
    python -m scripts.calibration.bed_to_bam --help
    python -m scripts.m6a.visualize --help
    python -m scripts.evaluate --help

Categories:
- calibration: Tools for bisulfite calibration data preparation
- m6a: Tools for m6A methylation analysis
- evaluate: General prediction evaluation
"""
