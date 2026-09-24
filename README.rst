Changelog
=========

v0.3.2
------
- Fix signal normalization for recent Dorado versions
- Add a warning for abnormal normalized signals

v0.3.1
------
- Improve inference data prefetching and throughput
- Reduce progress reporting overhead

v0.3.0
------
- Switch inference to BAM-primary streaming
- Add BAM modes and alignment filtering
- Add multi-process, multi-GPU inference with a single writer
- Add multi-file POD5/SLOW5 signal routing
- Improve modBAM tags, Dorado detection, logging, and progress

v0.2.5
------
- Improve multi-GPU BAM resume coordination
- Finalize incomplete reads and clean temporary files
- Update the POD5 requirement

v0.2.4
------
- Add ``--resume`` for BAM/modBAM inference
- Reduce long startup wait before inference

v0.2.3
------
- Fix BAM index timeout issue
- Refine dependency constraints

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
