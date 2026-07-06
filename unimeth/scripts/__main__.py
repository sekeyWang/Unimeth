"""
Unified entry point for UniMeth utility scripts.

Usage:
    unimeth <command> [options]
    
Commands:
    infer               Run methylation inference
    calibration         Calibration data preparation tools
        bed_to_bam      Convert BED bisulfite labels to BAM
        annotator       Create read-level calibration labels
    
    m6a                 m6A analysis tools
        bam_to_bam      Extract MM/ML tags for m6A training
        predictions_to_bam  Convert TSV predictions to BAM
        compare         Compare predictions with baseline
        visualize       Visualize m6A and nucleosome patterns
    
    evaluate            Evaluate predictions against bisulfite

Examples:
    unimeth infer --help
    unimeth calibration bed_to_bam --help
    unimeth m6a visualize --pred_dir <bam> --dorado_dir <bam>
    unimeth evaluate --tsv_dir <predictions.txt> --CpG_bed_dir <labels.bed>
"""
import sys

from unimeth import __version__


# Command registry
DIRECT_COMMANDS = {
    'infer': {
        'description': 'Run methylation inference',
        'module': 'unimeth.inference.__main__',
        'prog': 'unimeth infer',
    },
    'evaluate': {
        'description': 'Evaluate predictions against bisulfite ground truth',
        'module': 'unimeth.scripts.evaluate',
        'prog': 'unimeth evaluate',
    },
}

GROUP_COMMANDS = {
    'calibration': {
        'description': 'Calibration data preparation tools',
        'subcommands': {
            'bed_to_bam': 'unimeth.scripts.calibration.bed_to_bam',
            'annotator': 'unimeth.scripts.calibration.annotator',
        }
    },
    'm6a': {
        'description': 'm6A analysis tools',
        'subcommands': {
            'bam_to_bam': 'unimeth.scripts.m6a.bam_to_bam',
            'predictions_to_bam': 'unimeth.scripts.m6a.predictions_to_bam',
            'compare': 'unimeth.scripts.m6a.compare',
            'visualize': 'unimeth.scripts.m6a.visualize',
        }
    },
}

COMMANDS = {**DIRECT_COMMANDS, **GROUP_COMMANDS}


def print_help():
    """Print help message."""
    print(__doc__)
    print(f"Version: {__version__}")
    print("\nAvailable commands:\n")
    for cmd, info in DIRECT_COMMANDS.items():
        print(f"  {cmd:15} {info['description']}")
    for cmd, info in GROUP_COMMANDS.items():
        print(f"  {cmd:15} {info['description']}")
        for sub, module in info['subcommands'].items():
            print(f"    {sub:15} ({module})")
    print()


def main():
    """Main entry point."""
    if len(sys.argv) >= 2 and sys.argv[1] in ('-v', '--version'):
        print(f"unimeth {__version__}")
        sys.exit(0)

    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help', 'help'):
        print_help()
        sys.exit(0)
    
    cmd = sys.argv[1]

    if cmd in DIRECT_COMMANDS:
        command = DIRECT_COMMANDS[cmd]
        module = __import__(command['module'], fromlist=['main'])
        sys.argv = [command['prog']] + sys.argv[2:]
        module.main()
        return
    
    # Category subcommands: calibration, m6a
    if cmd in GROUP_COMMANDS:
        if len(sys.argv) < 3:
            print(f"Error: {cmd} requires a subcommand")
            print(f"\nAvailable subcommands for '{cmd}':")
            for sub in GROUP_COMMANDS[cmd]['subcommands']:
                print(f"  {sub}")
            print()
            sys.exit(1)
        
        subcmd = sys.argv[2]
        subcommands = GROUP_COMMANDS[cmd]['subcommands']
        
        if subcmd in ('-h', '--help', 'help'):
            print(f"\nUsage: unimeth {cmd} <subcommand> [options]\n")
            print(f"Available subcommands for '{cmd}':")
            for sub in subcommands:
                print(f"  {sub}")
            print()
            sys.exit(0)
        
        if subcmd not in subcommands:
            print(f"Error: Unknown subcommand '{subcmd}' for {cmd}")
            print(f"\nAvailable: {', '.join(subcommands.keys())}")
            sys.exit(1)
        
        # Import and run the module's main function
        module_path = subcommands[subcmd]
        module = __import__(module_path, fromlist=['main'])
        sys.argv = [f"unimeth {cmd} {subcmd}"] + sys.argv[3:]
        module.main()
        return
    
    print(f"Error: Unknown command '{cmd}'")
    print_help()
    sys.exit(1)


if __name__ == '__main__':
    main()
