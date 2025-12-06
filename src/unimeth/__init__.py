from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__version__ = '0.0.1'

def start():
    parser = ArgumentParser(
        'unimeth', 
        description='Unimeth is a unified deep learning framework for accurate and efficient detection of DNA methylation (5mC, 6mA) from Oxford Nanopore sequencing data. Built on a transformer-based architecture, Unimeth supports multiple sequencing chemistries (R9.4.1, R10.4.1 4kHz/5kHz), handles both plant and mammalian genomes, and achieves state-of-the-art performance across diverse genomic contexts.',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-v', '--version', 
        action='version', 
        version=f'unimeth {__version__}'
    )
    
    input_group = parser.add_argument_group('Input/Output options')
    input_group.add_argument(
        '--pod5_dir', type=str, required=True,
        help='Directory containing POD5 files (raw signal data)'
    )
    input_group.add_argument(
        '--bam_dir', type=str, required=True,
        help='Directory containing BAM files (aligned sequencing data)'
    )
    input_group.add_argument(
        '--model_dir', type=str, required=True,
        help='Path to the trained model file (.pt)'
    )
    input_group.add_argument(
        '--out_dir', type=str, required=True,
        help='Output file path for methylation results'
    )
    
    chr_group = parser.add_argument_group('Chromosome filtering options')
    chr_group.add_argument(
        '--chr', type=str, default='|',
        help=
        '''
        Chromosome selection in format "exclude_chr|include_chr". 
        Examples:
        - "|" : process all chromosomes (default)
        - "chrM,chrY|" : exclude chrM and chrY
        - "|chr1,chr2,chr3" : only include chr1, chr2, chr3
        - "chrM|chr1,chr2" : invalid, will use all chromosomes
        '''
    )
    
    control_group = parser.add_argument_group('Runtime control options')
    control_group.add_argument(
        '--limit', type=int, default=None,
        help='Limit number of batches to process (for testing)'
    )
    control_group.add_argument(
        '--batch_size', type=int, default=512,
        help='Batch size for inference'
    )
    
    meth_group = parser.add_argument_group('Methylation type options')
    meth_group.add_argument(
        '--cpg', type=int, default=0, choices=[0, 1],
        help='Detect CpG methylation sites (0: disable, 1: enable)'
    )
    meth_group.add_argument(
        '--chg', type=int, default=0, choices=[0, 1],
        help='Detect CHG methylation sites (0: disable, 1: enable)'
    )
    meth_group.add_argument(
        '--chh', type=int, default=0, choices=[0, 1],
        help='Detect CHH methylation sites (0: disable, 1: enable)'
    )
    meth_group.add_argument(
        '--m6A', type=int, default=0, choices=[0, 1],
        help='Detect m6A methylation sites (0: disable, 1: enable)'
    )
    
    # 技术参数
    tech_group = parser.add_argument_group('Technical options')
    tech_group.add_argument(
        '--pore_type', type=str, required=True, choices=['R10.4.1', 'R9.4.1'],
        help='Nanopore pore type'
    )
    tech_group.add_argument(
        '--frequency', type=str, required=True, choices=['4khz', '5khz'],
        help='Sequencing frequency (e.g., "4000" for 4kHz)'
    )
    tech_group.add_argument(
        '--dorado_version', type=float, required=True, choices=[0.71, 0.81],
        help='Dorado basecaller version used for data generation'
    )
    args = parser.parse_args()
    
    from unimeth.inference import main
    main(args)