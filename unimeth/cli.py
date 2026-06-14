"""Command-line entry point for Unimeth."""

import sys


def start():
    """Main CLI entry point dispatching to subcommands."""
    from unimeth.scripts.__main__ import main
    main()
