# Shared presentation order leaks through subject-independent evaluation

Code and manuscript for:

> Pinzhen He. *Shared presentation order leaks through subject-independent evaluation of
> fNIRS workload decoders: a multi-dataset audit with a null-window probe.* Manuscript
> (`paper/manuscript/main.pdf`).

When every participant in a dataset receives the same block or condition sequence,
leave-one-subject-out (LOSO) evaluation does not remove the information that sequence
carries: position in the session, carry-over from the preceding block and slow drift
are aligned with the labels in the same way for every participant. This repository
contains a model-agnostic **null-window probe** (decode the upcoming condition from a
window that ends before the block and before any cue), its controls, and the analyses
of three public fNIRS datasets and three published decoders reported in the paper.

## Main results (see the manuscript for details)

| | Result |
|---|---|
| Shin 2018 n-back, fNIRS, causal filter | null window +6.3 points above chance (p = 0.003, n = 26) |
| Tufts fNIRS2MW, causal baseline | null window +6.9 points (p = 0.019, n = 68) |
| Chen 2023 Stroop | condition decoding falls to chance on temporally reversed block pairs (p = 0.011, n = 18) |
| Shin word generation (per-subject order) | null window at chance (negative control) |
| EEG (Shin) | no effect |
| BenchNIRS (published) | ~80 % of its above-chance margin available 12 s before onset |
| Tufts DeepConvNet (official) | output tracks session position about as strongly as load |
| Con4m (NeurIPS 2024) | unaffected |

## Layout

```
src/eegfnirs/          data readers (Shin 2018 BrainVision/NIRx, Tufts raw) and helpers
scripts/20_ ... 32_    analyses, one script per analysis (docstrings describe each)
scripts/91_, 92_       per-subject tables and manuscript figures
docs/                  analysis notes (Chinese), including dataset quirks found (docs/DATA.md)
paper/manuscript/      LaTeX source (elsarticle), references, figures, compiled PDF
```

| Script | Analysis |
|---|---|
| `20_tufts_g1_probe.py` | null-window probe on Tufts (raw blocks 14-16), permutation null |
| `21_tufts_baselines.py` | offset-removal robustness (causal baseline, recording mean, none) |
| `22_shin_nullwindow.py` | Shin probe with causal / zero-phase / low-pass-only / no filter |
| `23_tufts_injection.py` | drift-injection positive control |
| `24_tufts_transfer.py` | task <-> rest transfer |
| `25_tufts_official.py` | official data: decoder output vs session position and preceding block |
| `26_shin_wg_control.py` | negative control (word generation, per-subject order) |
| `27_mentalfnirs_position.py` | PhysioNet mental-fnirs (uninformative, n = 12) |
| `28_chen_stroop.py` | forward vs reversed block pairs |
| `29_tufts_deepconvnet.py` | official Tufts DeepConvNet, generic protocol |
| `30_`, `31_benchnirs_shift*.py` | BenchNIRS, published pipeline with markers shifted before onset |
| `32_con4m_order.py` | Con4m with fixed vs per-subject shuffled block concatenation |

## Data (not included; all public)

| Dataset | Source | Expected location |
|---|---|---|
| Shin et al. 2018 (EEG+NIRS, n-back, word generation) | https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/ (`scripts/01_download_data.py`) | `data/raw/shin2018/` |
| Tufts fNIRS2MW | https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html | `data/raw/fNIRS2MW/` |
| Chen et al. 2023 Stroop | https://doi.org/10.6084/m9.figshare.c.6623431 | `data/raw/chen2023_stroop/` |
| PhysioNet mental-fnirs | https://physionet.org/content/mental-fnirs/1.0/ | `data/raw/mental-workload-*-1.0/` |

Note: the Tufts raw release contains the first twelve blocks for only 9 of the 68
eligible participants; the analyses use the final four blocks, which exist for all.

## Environment

```bash
conda env create -f environment.yml
conda activate eeg-fnirs
pip install -e .
```

BenchNIRS (`pip install benchnirs==1.7.1`) and Con4m
(https://github.com/MrNobodyCali/Con4m, with `transformers==4.38.2`, `h5py`, `psutil`)
were run in a separate environment; see the docstrings of scripts 30-32.

The manuscript compiles with `tectonic paper/manuscript/main.tex` (or any LaTeX
distribution with `elsarticle`).

## Third-party material

`scripts/data_tufts_partition_plan.txt` is the official subject split from
https://github.com/tufts-ml/fNIRS-mental-workload-classifiers (MIT licence).
The DeepConvNet architecture in `scripts/29_tufts_deepconvnet.py` follows the same
repository.

## Licence

MIT (see `LICENSE`).
