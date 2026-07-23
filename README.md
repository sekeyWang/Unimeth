# Unimeth: A Unified Transformer Framework for DNA Methylation Detection from Nanopore Reads

[![License](https://img.shields.io/badge/license-BSD--3--Clause--Clear-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.64898/2025.12.05.692231-brightgreen)](https://doi.org/10.64898/2025.12.05.692231)

[![PyPI-version](https://img.shields.io/pypi/v/unimeth)](https://pypi.org/project/unimeth/)
[![PyPI-Downloads](https://static.pepy.tech/badge/unimeth)](https://pepy.tech/project/unimeth/)
[![Conda Version](https://img.shields.io/conda/vn/bioconda/unimeth.svg)](https://anaconda.org/bioconda/unimeth)
[![Conda Downloads](https://img.shields.io/conda/dn/bioconda/unimeth.svg)](https://anaconda.org/bioconda/unimeth)

<!-- Workflow figure temporarily hidden while it is being updated: ![description](https://raw.githubusercontent.com/sekeyWang/Unimeth/main/images/workflow.jpg) -->
**Unimeth** is a unified deep learning framework for detecting DNA methylation (5mC, 6mA) from Oxford Nanopore reads. Built on a transformer-based architecture, Unimeth supports multiple sequencing chemistries (R9.4.1, R10.4.1 4kHz/5kHz) and methylation calling across plant, mammalian, and bacterial genomes.

---

## 🧬 Features

- **Unified Detection**: Supports DNA 5mC (CpG, CHG, CHH) and 6mA detection.
- **Multi-Chemistry Support**: Compatible with R9.4.1, R10.4.1 4kHz, and R10.4.1 5kHz chemistries.
- **Easy-to-Use**: Standard input/output formats (POD5/SLOW5/BLOW5 and BAM, BED).

---

## 📦 Installation

### Prerequisites

- Python 3.12+
- [Dorado](https://github.com/nanoporetech/dorado) for basecalling

### Option 1. Install with GPU support

```bash
conda create -n unimeth -c conda-forge -c bioconda --strict-channel-priority unimeth pytorch-gpu cuda-version=12.4
conda activate unimeth
```

Unimeth is available from Bioconda. The command above installs Unimeth with a GPU-enabled PyTorch build from conda-forge. Adjust `cuda-version` if your system requires a different CUDA runtime.

If you have cloned this repository, you can use the provided environment file instead:

```bash
conda env create -f envs/environment-gpu.yml
conda activate unimeth
```

### Option 2. Install via pip

```bash
conda create -n unimeth python=3.12
conda activate unimeth

pip install unimeth
```

For SLOW5/BLOW5 input, install `pyslow5` separately with `pip install pyslow5`, or install the optional pip extra with `pip install "unimeth[slow5]"`.

### Option 3. Install from Source

```bash
git clone https://github.com/sekeyWang/Unimeth.git
cd Unimeth

conda create -n unimeth python=3.12
conda activate unimeth

pip install -e .
```

Use `unimeth --help` to list utility subcommands, `unimeth --version` to print the installed version.

---


## 🚀 Quick Start

### 1. Download model checkpoints and sample data

- **Model**: Download `unimeth_r10.4.1_5kHz_5mC.pt` from [Google Drive](https://drive.google.com/drive/folders/1f8bWVFmbPxL6WqukOUi_BufCEvpOHaxR) to the `checkpoints` folder
- **Sample Data**: Download the demo dataset using one of the following methods:

```bash
mkdir demo
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1Gu7hgOQbHSUULG1MXjdE_qJ3na-6AdLi -O demo/
```
The demo dataset includes:

- `demo.bam` - aligned reads
- `subset_18.pod5` - raw signal data

### 2. Basecalling and Alignment

Use `dorado` to basecall and align the nanopore reads (there is already a `demo.bam` file in the demo folder, this step is optional):

```bash
dorado basecaller --device cuda:all --recursive --emit-moves \
--reference /path/to/reference.fasta \
/path/to/dorado/models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0 \
/path/to/subset_18.pod5 > demo.bam
```


### 3. Methylation Calling with Unimeth

Run Unimeth to detect methylation. Use `unimeth infer` for single-process inference. For multi-GPU inference, launch the same module with `accelerate launch -m unimeth.inference`.

```bash
# modBAM output (default)
unimeth infer \
--pod5 demo/subset_18.pod5 \
--bam demo/demo.bam \
--model checkpoints/unimeth_r10.4.1_5kHz_5mC.pt \
--out results/arab.bam \
--cpg 1 \
--batch_size 256 \
--pore_type R10.4.1 \
--frequency 5khz
```

```bash
# TSV output
unimeth infer \
--pod5 demo/subset_18.pod5 \
--bam demo/demo.bam \
--model checkpoints/unimeth_r10.4.1_5kHz_5mC.pt \
--out results/arab.tsv \
--output_format tsv \
--cpg 1 \
--chg 1 \
--chh 1 \
--batch_size 256 \
--pore_type R10.4.1 \
--frequency 5khz
```

The examples set `--batch_size 256` for conservative demo memory usage; if omitted, the current code default is `512`. Use `--output_format both --tsv_out results/arab.tsv --bam_out results/arab.bam` to generate TSV and modBAM simultaneously.
Use `--slow5 reads.slow5` or `--slow5 reads.blow5` instead of `--pod5` for SLOW5/BLOW5 input.

#### Output

Unimeth outputs read-level methylation calls in **TSV** or **modBAM** format. A sample TSV output is as follows:


| Chromosome | Ref pos | Strand | Label | Read id | Read pos | Methylation type | Prob-negative | Prob-positive | Pred(0/1) | . |
|--------|-----------|----|-------|-----------------------------------|-------|-------------------|---------------|---------------|-------|------|
| Chr2 | 15338477 | - | -1 | 28752a76-7007-40d7-8ede-f2939fe2ab26 | 0 | [CpG] | 0.985000 | 0.014000 | 0 | . |
| Chr2 | 15338471 | - | -1 | 28752a76-7007-40d7-8ede-f2939fe2ab26 | 6 | [CpG] | 0.990000 | 0.009000 | 0 | . |
| Chr2 | 15338465 | - | -1 | 28752a76-7007-40d7-8ede-f2939fe2ab26 | 12 | [CHG] | 0.998000 | 0.001000 | 0 | . |
| Chr2 | 15338462 | - | -1 | 28752a76-7007-40d7-8ede-f2939fe2ab26 | 15 | [CHH] | 0.998000 | 0.001000 | 0 | . |
| Chr2 | 15338457 | - | -1 | 28752a76-7007-40d7-8ede-f2939fe2ab26 | 20 | [CHH] | 0.999000 | 0.000000 | 0 | . |
---

The TSV file can be further processed to generate site-level methylation frequencies using the provided `scripts/call_modification_frequency.py` script. It can also be converted to modBAM format using `scripts/generate_5mC_modbam_file.py` (5mC only).


## 🧪 Models

We provide pre-trained models for:

- **Plant 5mC** (R10.4.1 5kHz, R9.4.1)
- **Human 5mCpG** (R10.4.1 5kHz/4kHz, R9.4.1)
- **6mA Detection** (R10.4.1)

Download models from the [Google Drive](https://drive.google.com/drive/folders/1f8bWVFmbPxL6WqukOUi_BufCEvpOHaxR) page.

---

<!--
## 📊 Performance Highlights
Benchmark figure temporarily hidden while results are being updated:
![description](https://raw.githubusercontent.com/sekeyWang/Unimeth/main/images/plant_result.jpg)
- Outperforms DeepPlant, Dorado, Rockfish, and DeepMod2 in cross-species benchmarks.
- Superior accuracy in repetitive regions (centromeres, transposons).
- Lower false positive rates in CHH and 6mA contexts.
- Robust to batch effects and unseen species.

For detailed benchmarks, see the [manuscript](https://doi.org/10.64898/2025.12.05.692231).

---
-->

## 📁 Input/Output Formats

| Input Format | Description |
|--------------|-------------|
| POD5         | Raw nanopore signals |
| SLOW5/BLOW5  | Raw nanopore signals (`--slow5`; inference only) |
| BAM          | Basecalled reads, aligned or unaligned |


| Output Format | Description |
|---------------|-------------|
| modBAM        | BAM with MM/ML methylation tags (`--output_format bam`, default) |
| tsv           | Per-read methylation calls (`--output_format tsv`) |
| both          | TSV and modBAM simultaneously (`--output_format both`; use `--tsv_out`/`--tsv_out_dir` and `--bam_out`/`--bam_out_dir` for separate paths) |
| bedmethyl     | Site-level methylation frequencies (post-processing) |
---

## 📚 Citation

If you use Unimeth in your research, please cite:

> Wang S, Xiao Y, Sheng T, et al. Unimeth: A unified transformer framework for accurate DNA methylation detection from nanopore reads[J]. *bioRxiv*, 2025: 2025.12.05.692231.

---

## 📄 License

This project is licensed under the BSD 3-Clause Clear License. See [LICENSE](LICENSE) for details.

---

## 📬 Contact
- GitHub Issues: [https://github.com/sekeyWang/Unimeth/issues](https://github.com/sekeyWang/Unimeth/issues)
