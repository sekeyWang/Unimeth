Changelog
=========

v0.2.2
------
- Add SLOW5/BLOW5 input support for inference
- Refine BAM index caching
- Fix incomplete MM/ML tags caused by skipped co-batched completion markers

v0.2.1
------
- support unaligned BAM input
- add `unimeth infer` subcommand

v0.2.0
------
- Add BAM output support (``--output_format bam/tsv/both``)
- Switch attention implementation to PyTorch SDPA (built-in, no extra dependency)
- Add minimum version constraints to all dependencies
- Refactor BAM site prediction to sequential scan for better performance

v0.0.3
------
- Support multi-GPUs

v0.0.2
------
- Catch up PyPI release

v0.0.1
------
- Initialize project
