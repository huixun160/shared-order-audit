# Shared presentation order leaks through subject-independent evaluation

Code and manuscript for:

> Pinzhen He. *Shared presentation order leaks through subject-independent evaluation of
> fNIRS decoders: identifiability, a validated two-step diagnostic, and a multi-dataset
> audit.* Manuscript (`paper/manuscript/main.pdf`).

When every participant in a dataset receives the same block or condition sequence,
leave-one-subject-out (LOSO) evaluation does not remove the information that sequence
carries: position in the session, carry-over from the preceding block and slow drift
are aligned with the labels in the same way for every participant. This repository
contains

- a **two-step diagnostic**: step 1, the null-window probe (decode the upcoming condition
  from a window that ends before the block and before any cue); step 2, rest-to-task
  transfer (train on those windows, test on task windows), which indicates how much of
  task accuracy is obtainable without task data;
- design-only accuracy ceilings and an identifiability analysis (what a sequence allows
  before any data are recorded);
- conditional decoding to separate carry-over from position/drift;
- a ground-truth fNIRS simulation (`src/eegfnirs/sim/`) used to validate all of the above;
- within-subject permutation tests that re-run the whole LOSO pipeline;
- the analyses of three public fNIRS datasets, six decoders and three published methods.

## Main results (see the manuscript for details)

Excess over chance in percentage points; p from 500-permutation tests (only p < 0.01 is
treated as primary evidence).

| | Result |
|---|---|
| Design only | Latin squares rotated across subjects: position at chance, but the previous condition predicts the current one with 80 % accuracy (4 classes) |
| Simulation | step 1 alone does not predict task-decoding inflation; step 2 does (r = 0.65) |
| Shin 2018 n-back, causal filter | step 1 +6.7 (p = 0.002), step 2 +5.6 (p = 0.002) |
| Shin 2018 n-back, source | previous-block carry-over (conditional decoding, matches the simulated carry-over signature) |
| Tufts fNIRS2MW, final blocks | 42-100 % of task accuracy obtainable from pre-cue windows across six decoders |
| Six decoders, all settings | rest-to-task transfer tracks task accuracy (r = 0.73) |
| Shin word generation (per-subject order) | step 1 at chance (negative control) |
| Chen 2023 Stroop | condition decoding falls to chance on temporally reversed block pairs |
| BenchNIRS (published) | same pipeline 12 s before onset: 37.9 % vs 39.2 % (permutation p = 0.008) |
| Tufts DeepConvNet / Con4m | position effect without mean inflation (balanced positions) / unaffected |

## Layout

```
src/eegfnirs/          data readers (Shin 2018 BrainVision/NIRx, Tufts raw) and helpers
src/eegfnirs/sim/      ground-truth fNIRS simulator and probe utilities
scripts/20_ ... 32_    analyses of the original submission, one script per analysis
scripts/40_ ... 48_    simulation, decomposition, permutation tests, six-decoder benchmark
scripts/91_ ... 93_    per-subject tables and manuscript figures
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
| `40_sim_experiments.py` | simulation: calibration, sensitivity, identifiability, design-only ceilings |
| `41_shin_decompose.py` | conditional decoding in Shin (previous load / position held fixed) |
| `42_sim_validate_decompose.py` | conditional decoding validated on simulated single-source data |
| `43_sim_inflation.py` | simulation: step 1 / step 2 vs task-decoding inflation |
| `44_shin_null_task.py` | Shin null and task window features |
| `45_permutation_real.py` | permutation tests for Shin, Tufts and the WG control |
| `46_benchmark_windows.py`, `47_benchmark_models.py` | six decoders x three analyses, LOSO |
| `48_benchnirs_perm.py` | BenchNIRS permutation test (run in the BenchNIRS environment) |
| `93_revision_figures.py` | Figures 2 and 4 (simulation, two-step diagnostic) |

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
