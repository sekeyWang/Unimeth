"""
Unified entry point for UniMeth utility scripts.

Usage:
    python -m scripts <command> [options]
    
Commands:
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
    python -m scripts calibration bed_to_bam --help
    python -m scripts m6a visualize --pred_dir <bam> --dorado_dir <bam>
    python -m scripts evaluate --tsv_dir <predictions.txt> --CpG_bed_dir <labels.bed>
"""
import sys
import argparse


# Command registry
COMMANDS = {
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


def print_help():
    """Print help message."""
    print(__doc__)
    print("\nAvailable commands:\n")
    for cmd, info in COMMANDS.items():
        print(f"  {cmd:15} {info['description']}")
        for sub, module in info['subcommands'].items():
            print(f"    {sub:15} ({module})")
    print("\n  evaluate        Evaluate predictions against bisulfite ground truth")
    print()


def main():
    """Main entry point."""
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help', 'help'):
        print_help()
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    # Direct command: evaluate
    if cmd == 'evaluate':
        from unimeth.scripts.evaluate import main as evaluate_main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        evaluate_main()
        return
    
    # Category subcommands: calibration, m6a
    if cmd in COMMANDS:
        if len(sys.argv) < 3:
            print(f"Error: {cmd} requires a subcommand")
            print(f"\nAvailable subcommands for '{cmd}':")
            for sub in COMMANDS[cmd]['subcommands']:
                print(f"  {sub}")
            print()
            sys.exit(1)
        
        subcmd = sys.argv[2]
        subcommands = COMMANDS[cmd]['subcommands']
        
        if subcmd in ('-h', '--help', 'help'):
            print(f"\nUsage: python -m scripts {cmd} <subcommand> [options]\n")
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
        sys.argv = [sys.argv[0]] + sys.argv[3:]
        module.main()
        return
    
    print(f"Error: Unknown command '{cmd}'")
    print_help()
    sys.exit(1)


if __name__ == '__main__':
    main()
