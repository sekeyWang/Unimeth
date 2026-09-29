# Unimeth: A Unified Transformer Framework for DNA Methylation Detection from Nanopore Reads

[![License](https://img.shields.io/badge/license-BSD--3--Clause--Clear-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.64898/2025.12.05.692231-brightgreen)](https://doi.org/10.64898/2025.12.05.692231)

[![PyPI-version](https://img.shields.io/pypi/v/unimeth)](https://pypi.org/project/unimeth/)
[![PyPI-Downloads](https://static.pepy.tech/badge/unimeth)](https://pepy.tech/project/unimeth/)
[![Conda Version](https://img.shields.io/conda/vn/bioconda/unimeth.svg)](https://anaconda.org/bioconda/unimeth)
[![Conda Downloads](https://img.shields.io/conda/dn/bioconda/unimeth.svg)](https://anaconda.org/bioconda/unimeth)

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

### Option 1. Install from Source

```bash
git clone https://github.com/sekeyWang/Unimeth.git
cd Unimeth

conda create -n unimeth python=3.12
conda activate unimeth

pip install .
```

For SLOW5/BLOW5 input, install `pyslow5` separately with `pip install pyslow5`, or install the optional pip extra with `pip install ".[slow5]"`.

### Option 2. Install with Conda and pip

```bash
git clone https://github.com/sekeyWang/Unimeth.git
cd Unimeth

conda create -y -n unimeth \
  -c conda-forge -c bioconda \
  --override-channels --strict-channel-priority \
  python=3.12 pip \
  "pytorch=2.5.1=cuda126*" \
  accelerate transformers numpy tqdm pysam scikit-learn scipy packaging

conda activate unimeth

python -m pip install --only-binary=:all: \
  "pod5==0.3.44" \
  "lib-pod5==0.3.44" \
  "pyarrow>=22,<23"

python -m pip install --no-deps .
```

### Option 3. Install via pip

```bash
conda create -n unimeth python=3.12
conda activate unimeth

pip install unimeth
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

Run Unimeth to detect methylation. A single `unimeth infer` invocation automatically uses all visible GPUs; restrict them with `CUDA_VISIBLE_DEVICES` when needed.

```bash
# modBAM output (default)
unimeth infer \
--pod5 demo/subset_18.pod5 \
--bam demo/demo.bam \
--model checkpoints/unimeth_r10.4.1_5kHz_5mC.pt \
--out results/arab.bam \
--5mCpG 1 \
--5mCHG 1 \
--5mCHH 1 \
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
--5mCpG 1 \
--5mCHG 1 \
--5mCHH 1 \
--batch_size 256 \
--pore_type R10.4.1 \
--frequency 5khz
```

Notes:
- The default inference batch size is `256`; reduce `--batch_size` on GPUs with less available memory.
- To generate TSV and modBAM together, use `--output_format both --tsv_out results/arab.tsv --bam_out results/arab.bam`.
- For SLOW5/BLOW5 input, use `--slow5 reads.slow5` or `--slow5 reads.blow5` instead of `--pod5`.
- Public modification names use `5mC` and `6mA`; context-specific names are `5mCpG`, `5mCHG`, and `5mCHH`. Inference, training, and calibration annotation accept `--5mCpG 1`, `--5mCHG 1`, `--5mCHH 1`, and `--6mA 1`. Legacy `--cpg`, `--chg`, `--chh`, and `--m6A` remain accepted. Model tokens, checkpoint fields, training labels, and existing inference TSV labels retain their original names for compatibility.

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

### 4. Methylation frequency calling

Calculate site methylation frequencies from the modBAM produced above. The BAM must be coordinate-sorted and indexed, with a matching indexed reference FASTA.

```bash
# Count the default modification types:
unimeth call_freq --input_bam results/arab.bam --ref reference.fa \
    --output results/freq

# Count CpG only and combine its two strands:
unimeth call_freq --input_bam results/arab.bam --ref reference.fa \
    --output results/cpg_freq --mod_types 5mCpG --combine_cpg

# Count all C sites with predictions:
unimeth call_freq --input_bam results/arab.bam --ref reference.fa \
    --output results/all_c --mod_types 5mC
```

Notes:
- The default output is a bedMethyl file for each modification type, such as `results/freq.5mCpG.bed`.
- Add `--combine_cpg` to combine CpG strands.
- Reads with `HP=1` or `HP=2` also produce haplotype files when eligible sites exist. Use `--no_hap` for total output only.


## 🧪 Models

We provide pre-trained models for:

- **Plant 5mC** (R10.4.1 5kHz, R9.4.1)
- **Human 5mCpG** (R10.4.1 5kHz/4kHz, R9.4.1)
- **6mA Detection** (R10.4.1)

Download models from the [Google Drive](https://drive.google.com/drive/folders/1f8bWVFmbPxL6WqukOUi_BufCEvpOHaxR) page.

> **Dorado model compatibility:** The current R10.4.1 5 kHz models are primarily optimized and validated for reads basecalled with Dorado `dna_r10.4.1_e8.2_400bps_sup@v5.0.0`. Other Dorado basecalling models may work, but prediction accuracy and calibration can differ.

---


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
| bed           | Site-level methylation frequencies in bedMethyl format (`unimeth call_freq --output_format bed`, default) |
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

---

## TODO
- [ ] After the official POD5 Conda packages are fixed, update UniMeth's Conda package, dependencies, and installation instructions.
- [ ] Create a new `envs/environment-gpu.yml` after the Conda installation path is working.
- [x] Module for methylation frequency calculation.
- [ ] Make bam sorting and indexing optional.
- [ ] Evaluate and improve compatibility with additional Dorado basecalling models.
- [ ] Prevent normalization warnings from disrupting tqdm progress output.
