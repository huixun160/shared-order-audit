# Datasets

Everything below for `shin2018` was verified against the actual files (VP001 headers and the
official `Dataset description_MATLAB.pdf`), not copied from a paper.

---

## `shin2018` - primary (attention / working memory)

**Shin, J. et al., "Simultaneous Acquisition of EEG and NIRS during Cognitive Tasks for an Open
Access Dataset", *Scientific Data*.**
Host: <https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/> - DOI `10.14279/depositonce-5830`

### Acquisition

| | |
|---|---|
| Subjects | 26 healthy |
| EEG | BrainAmp, **1000 Hz**, 30 channels (28 EEG + HEOG + VEOG), 10-5 system |
| EEG reference / ground | TP9 / TP10 |
| fNIRS | NIRScout, **10.4 Hz**, 16 sources + 16 detectors -> **36 channels**, SD distance 30 mm |
| fNIRS coverage | 16 (pre)frontal, 4+4 motor (C3/C4), 4+4 parietal (P3/P4), 4 occipital (POz) |
| Synchronisation | Triggers sent to both amplifiers simultaneously over a parallel port |

The parallel-port trigger matters: EEG and fNIRS event times are hardware-aligned to under 20 ms once
the per-run offset is removed, so any multi-second lag between the modalities is **physiological, not
instrumental**.

That licenses *interpreting* a lag, but it does not make one measurable here. The n-back paradigm is a
block design - 42 s of task, 20 trials at a 2.2 s ISI - so the haemodynamic response is a slow
build-up across the whole block. Reading fNIRS "later" and reading it "lag-corrected" are then
confounded, and a lag sweep measures response accumulation rather than latency. See
`results/decoding/lag_sweep.csv`, where accuracy rises monotonically to the edge of the grid instead
of peaking at the canonical 5-7 s. Resolving a haemodynamic latency needs event-related timing or the
block transitions, not the plateau.

EEG channel order in file: `Fp1 AFF5h AFz F1 FC5 FC1 T7 C3 Cz CP5 CP1 P7 P3 Pz POz O1 Fp2 AFF6h F2
FC2 FC6 C4 T8 CP2 CP6 P4 P8 O2 HEOG VEOG` (BrainVision names them `FP1`/`AFF5`/`AFF6`).

### Tasks and runs

Each subject has 9 EEG runs and 9 matching NIRS session folders:

| Task | Runs | Classes |
|---|---|---|
| n-back | `nback1/2/3` | each run holds **9 blocks**: 3 x 0-back, 3 x 2-back, 3 x 3-back, order counterbalanced |
| DSR (labelled `gonogo*` in the raw files) | `gonogo1/2/3` | Go vs No-go |
| word generation | `word1/2/3` | verbal fluency vs baseline |

> Note the naming trap: the paper calls dataset B "discrimination/selection response (DSR)"; the
> raw filenames say `gonogo`, and the MATLAB release calls the same thing `cnt_dsr`. Same task.

### Trigger codes (BrainVision `Stimulus` markers, `S<code>`)

**n-back**

The official PDF's code table is mis-columned when the text is extracted, so the mapping below was
**recovered empirically from VP001** and is re-checked for every subject at load time.

| Code | Meaning | | Code | Meaning |
|---|---|---|---|---|
| 16 | 0-back stimulus | | 96 | 3-back non-target |
| 48 | 2-back target | | 112 | **0-back block onset** |
| 64 | 2-back non-target | | 128 | **2-back block onset** |
| 80 | 3-back target | | 144 | **3-back block onset** |

Recovered by partitioning VP001's `nback1` markers by the preceding block-onset code. The partition
is perfectly clean, which is what makes the assignment safe:

```
S112 blocks -> S16 x60                    (0-back)
S128 blocks -> S48 x18 + S64 x42          (2-back: target / non-target)
S144 blocks -> S80 x18 + S96 x42          (3-back: target / non-target)
```

Two corrections to the published table fall out of this:

* **0-back carries no target/non-target split.** All 60 0-back trials use `S16`; the `S32` the PDF
  implies ("0-back non-target") never appears. So "target vs non-target" is a **2- and 3-back only**
  label, while **workload level (0/2/3-back) is defined for every trial**. Our primary task is
  therefore workload level, which is also the more interesting cognitive-state target.
* Each `nback` run contains all three levels (9 blocks, 20 trials each), not one level per run.

**DSR / Go-NoGo:** 16 = Go, 32 = No-go, 48 = block onset
**Word generation:** 16 = verbal fluency, 32 = baseline

> `data/shin2018.py` re-derives this partition for every run of every subject and raises if a block
> contains a trial code outside its expected set, or if `S32` ever appears in an `S112` block. The
> mapping is checked, not trusted: if it is wrong for some subject, the run aborts rather than
> silently mislabelling.

### Aligning the two modalities

The NIRx `.evt` files carry **block-level triggers only** (9 for an n-back run, 6 for Go-NoGo, 21 for
word generation); trial-level events exist only in the EEG `.vmrk`. So the alignment is: match blocks
via the onset triggers present in both streams, then place EEG trial events on the shared timeline.

Run identity is not inferred from clocks. The NIRx `.inf` file records it outright:

```
[Subject Demographics]
Name="vp001"
Additional Notes="gonogo1"
```

This matters, because the two devices' clocks disagree - VP001's `nback1` EEG segment is stamped
12:38 while the matching NIRS header says 11:26. Pairing on timestamps would have quietly produced
wrong pairs. For VP001 the resulting map is `001-003 -> gonogo1-3`, `004-006 -> word1-3`,
`007-009 -> nback1-3`, but it is read per subject, never assumed.

### Distribution formats

| Format | Size | Notes |
|---|---|---|
| Raw per-subject BrainVision + NIRx | ~270 MB + ~61 MB per subject (~8.6 GB total) | **What we use.** Full 1000 Hz EEG, raw wavelength fNIRS -> our own Beer-Lambert |
| Combined MATLAB (BBCI format) | 6.09 GB EEG + 798 MB NIRS | Pre-downsampled to 200 Hz / 10 Hz, HbO+HbR already computed. Kept as a **cross-check**, not the primary path |
| `behavior.zip` | 234 KB | Per-run behavioural summaries |

Using the raw path means our preprocessing is a documented part of the benchmark rather than an
undocumented inheritance from the authors' MATLAB conversion. The MATLAB path exists so we can show
our pipeline reproduces the released version.

---

## `shin2017` - secondary (cross-dataset transfer)

**Shin, J. et al., "Open Access Dataset for EEG+NIRS Single-Trial Classification", *IEEE TNSRE*
25(10):1735-1745, 2017.** Host: <https://doc.ml.tu-berlin.de/hBCI/>

29 subjects; left vs right hand motor imagery, and mental arithmetic vs rest. Different subjects,
different paradigms, same lab and near-identical montage - which makes it a fair test of whether a
representation trained on n-back transfers, rather than a test of whether it memorised one protocol.
Access requires agreeing to the licence on the page; the downloader will not click through that for
you.

---

## Citation

Cite all three when publishing with `shin2018`:

1. Shin, J. et al. *Simultaneous Acquisition of EEG and NIRS during Cognitive Tasks for an Open
   Access Dataset.* Scientific Data.
2. Shin, J. et al. *Open Access Dataset for EEG+NIRS Single-Trial Classification.* IEEE TNSRE 25(10), 2017.
3. Blankertz, B. et al. *The Berlin Brain-Computer Interface: Non-Medical Uses of BCI Technology.*
   Front. Neurosci. 4, 2010.

---

## Data-quality findings

Six issues found by working from the raw distribution rather than the authors' MATLAB
conversion. Each is stated with the evidence that established it, because each one is a
place where a naive pipeline produces wrong numbers **without failing**.

### 1. The published trigger table is mis-columned

See above. Recovered by partitioning markers on the preceding block-onset code, then
confirmed independently by the NIRS device's own codes (7/8/9).

### 2. The two devices start recording seconds apart, and the sign varies

Across 26 subjects x 9 runs the EEG->fNIRS offset spans roughly **-5 to +6 s**, changing
sign between runs of the same subject. Assuming the streams share `t = 0` therefore
misaligns them by an amount comparable to the haemodynamic response itself.

After correcting per run using the shared parallel-port triggers, the residual is
**under 20 ms worst-case**. So the correction is easy - but only if you know to make it.

### 3. MNE cannot read these NIRx files

`mne.io.read_raw_nirx` requires a `probeInfo.mat` that NIRStar 14.1 (2016) did not
write. We parse the format directly instead; see `eegfnirs/data/nirx.py`.

### 4. Two subjects have wrong or missing run labels in the NIRx `.inf`

| Subject | Directory | `.inf` says | Actually is | Evidence |
|---|---|---|---|---|
| VP016 | `2016-06-08_005` | `nback2` | **word2** | 21 events with codes 1/2/13 - the word-generation signature, not n-back's nine 7/8/9 |
| VP016 | `2016-06-08_002` | `nback2` | nback2 | code sequence `8 7 9 7 9 8 9 8 7` matches the EEG block sequence exactly |
| VP024 | `2016-06-16_011` | *(empty)* | **nback2** | same match against EEG `128 112 144 112 144 128 144 128 112` |

`nirs_session_map` resolves these from the trigger sequence. It never falls back to
ordering by timestamp - finding 2 is exactly why that would be unsafe.

### 4b. One EEG header points at a misspelled file name

`VP017/eeg/word2.vhdr` in the original archive references `wor2.eeg` / `wor2.vmrk`, while
the extracted files are `word2.*`. The interim copy's header was patched on 2026-10-06 to
point at `word2.*` (original kept as `word2.vhdr.orig`). It is the only mismatch among all
234 headers. n-back runs were never affected.

### 5. One corrupted NIRS trigger

**77 of 78** n-back runs agree between the two devices' independent block-condition
triggers. The exception:

```
VP015/nback3, block 1:   EEG says 3-back,  NIRS says 2-back
```

We treat the EEG label as correct and the NIRS trigger as corrupted, on evidence rather
than preference: the EEG block code is corroborated by the 20 trial codes inside that
block (which are 3-back codes), so the EEG side has two agreeing sources against the
NIRS side's one. The NIRS codes 8 and 9 differ by a single bit, as do the EEG codes 128
and 144, which is consistent with a single-bit trigger error.

This is only detectable **because** two independent trigger streams exist. It is not
visible in the MATLAB distribution, and no analysis using one modality's markers alone
could find it.

### 6. The n-back block order is identical for every subject

```
nback1   3 2 0 2 0 3 0 3 2
nback2   2 0 3 0 3 2 3 2 0
nback3   0 3 2 3 2 0 2 0 3
```

Verified across all 26 subjects: one order per run, no variation. Across the three runs
this is a proper Latin square, so block *position* alone is uninformative - but **within
a run the previous block's load determines the current one**. Any feature that encodes
position in the run therefore encodes load, without any cognitive information.

Measured by decoding load from windows lying entirely before block onset, where no
legitimate information can exist:

| Modality | Accuracy | Chance | Excess | *p* | Subjects above chance |
|---|---|---|---|---|---|
| EEG | 0.3500 | 0.3333 | **+0.0166** | **0.024** | 17 / 26 |
| fNIRS | 0.3405 | 0.3333 | +0.0071 | 0.26 | 14 / 26 |

> **2026-10-06 更正**：上表的窗口包含区组开始前 2 s 的指导语画面，并且使用零相位滤波；fNIRS 还用开始前 5 s 做了基线。
> 修正后（`scripts/22_shin_nullwindow.py`）EEG 不显著，fNIRS 在因果滤波下为 +6 ~ +7 点（p ≈ 0.003）。详见 `docs/G1_TUFTS_2026-10-06.md` §7.3。

The leak is small and confined to EEG, where slow drift carries time-in-run information;
fNIRS features are baseline-corrected per block, which removes it. Every block-level
accuracy on this dataset should be read as containing roughly this much inflation, and
comparisons between conditions remain valid because all arms carry the same confound.

Unlike findings 1-5 this is not a defect in the release - the design is a deliberate
counterbalance. It is a confound that analyses must control for, and we have not found a
paper using this dataset that mentions it.
