"""Shin 2018 n-back: the null-window probe, rebuilt to remove three known defects.

The original probe (``eval/timeresolved.leakage_test``, +1.7 points EEG) had:

1. **zero-phase filtering** of EEG (band-pass and band envelopes) and fNIRS, which can
   move post-onset information backwards in time (van Driel et al. 2021);
2. an **fNIRS baseline taken from the 5 s before onset**, overlapping the probe window;
3. windows ending at onset, which **include the 2 s instruction screen** that shows the
   coming n (Shin et al. 2018, Methods). Anything there may be anticipation, not leakage.

Here every window ends at least 2 s before the marked onset (i.e. before the
instruction), filters are either causal (``phase="minimum"``) or zero-phase for
comparison, and fNIRS is not baseline-corrected per block. fNIRS "mean" features are
relative to the run-wide channel mean (a label-free constant per run), and slopes are
reported separately because they need no baseline at all.

    python scripts/22_shin_nullwindow.py                 # causal vs zero-phase band-pass
    python scripts/22_shin_nullwindow.py lp_only none    # no high-pass at all
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eegfnirs.data import shin2018  # noqa: E402
from eegfnirs.data.preprocess import (BANDS, EEG_BAND, EEG_NOTCH_HZ, FNIRS_BAND,  # noqa: E402
                                      FNIRS_PPF, EEG_RENAME)
from eegfnirs.eval.canonical import LOAD_OF  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402

g1 = importlib.import_module("20_tufts_g1_probe")

WINDOW_S = 4.0
INSTRUCTION_S = 2.0
GAPS = (0.0, 2.0, 4.0)          # extra distance before the instruction
EEG_SFREQ = 200.0


def _eeg(raw, phase: str):
    raw = raw.copy().load_data()
    raw.rename_channels({k: v for k, v in EEG_RENAME.items() if k in raw.ch_names})
    raw.notch_filter(EEG_NOTCH_HZ, picks="all", phase=phase, verbose="ERROR")
    raw.filter(EEG_BAND[0], EEG_BAND[1], picks="all", phase=phase, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.resample(EEG_SFREQ, verbose="ERROR")
    picks = [i for i, t in enumerate(raw.get_channel_types()) if t == "eeg"]
    env = {b: raw.copy().pick(picks).filter(lo, hi, phase=phase, verbose="ERROR").get_data() ** 2
           for b, (lo, hi) in BANDS.items()}
    return raw.times, env


def _hb(raw, phase: str):
    """``phase`` is "minimum" / "zero" (0.01-0.2 Hz band-pass), or "lp_only" (causal
    0.2 Hz low-pass, no high-pass) / "none" (no filter at all). The last two rule out a
    high-pass dragging the previous block's response forward into the probe window."""
    import mne
    od = mne.preprocessing.nirs.optical_density(raw.copy().load_data(), verbose="ERROR")
    od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(od, verbose="ERROR")
    hb = mne.preprocessing.nirs.beer_lambert_law(od, ppf=FNIRS_PPF)
    if phase == "lp_only":
        hb.filter(None, FNIRS_BAND[1], phase="minimum", verbose="ERROR")
    elif phase != "none":
        hb.filter(FNIRS_BAND[0], FNIRS_BAND[1], phase=phase, verbose="ERROR")
    return hb.times, hb.get_data() * 1e6


def _slope(t, y):
    tt = t - t.mean()
    return (tt * (y - y.mean(-1, keepdims=True))).sum(-1) / (tt ** 2).sum()


def run_rows(subject_dir: Path, run: str, phase: str) -> list[dict]:
    task = shin2018.task_of_run(run)
    eeg_phase = phase if phase in ("minimum", "zero") else "minimum"
    eeg_t, env = _eeg(shin2018.load_eeg_run(subject_dir, run, preload=True), eeg_phase)
    hb_t, hb = _hb(shin2018.load_nirs_run(subject_dir, run), phase)
    shift = shin2018.align_run(subject_dir, run).offset
    markers = shin2018.parse_vmrk(subject_dir / "eeg" / (run + ".vmrk"))
    blocks = shin2018.segment_blocks(markers, task, context=subject_dir.name + "/" + run)
    hb_mean = hb.mean(1)

    rows = []
    for pos, b in enumerate(blocks, start=1):
        onset = (b.onset_sample - 1) / shin2018.EEG_SFREQ
        for gap in GAPS:
            t1 = onset - INSTRUCTION_S - gap
            t0 = t1 - WINDOW_S
            em = (eeg_t >= t0) & (eeg_t < t1)
            hm = (hb_t >= t0 + shift) & (hb_t < t1 + shift)
            if em.sum() < 10 or hm.sum() < 3:
                continue
            r = {"subject": subject_dir.name, "run": run, "position": pos,
                 "load": LOAD_OF[b.condition], "gap": gap, "phase": phase}
            for band, p in env.items():
                for i, v in enumerate(np.log10(p[:, em].mean(1) + 1e-30)):
                    r[f"eeg_{band}_{i}"] = v
            seg = hb[:, hm]
            for i, v in enumerate(seg.mean(1) - hb_mean):
                r[f"fnirsM_{i}"] = v
            for i, v in enumerate(_slope(hb_t[hm], seg)):
                r[f"fnirsS_{i}"] = v
            rows.append(r)
    return rows


def main() -> None:
    paths = project_paths()
    out = paths.results / "shin_nullwindow"
    out.mkdir(parents=True, exist_ok=True)
    phases = tuple(sys.argv[1:]) or ("minimum", "zero")
    feats = out / ("features.csv" if phases == ("minimum", "zero")
                   else "features_{}.csv".format("_".join(phases)))
    if feats.exists():
        df = pd.read_csv(feats)
    else:
        rows = []
        root = paths.interim / "shin2018"
        for sd in sorted(root.glob("VP*")):
            for run in shin2018.NBACK.runs:
                for phase in phases:
                    rows += run_rows(sd, run, phase)
            print("done", sd.name, flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(feats, index=False)

    sets = {
        "eeg": [c for c in df.columns if c.startswith("eeg_")],
        "fnirs_mean+slope": [c for c in df.columns if c.startswith("fnirs")],
        "fnirs_slope": [c for c in df.columns if c.startswith("fnirsS_")],
    }
    res = []
    for (phase, gap), ch in df.groupby(["phase", "gap"]):
        ch = ch.sort_values(["subject", "run", "position"])
        for name, cols in sets.items():
            acc = g1.loso(ch[cols].to_numpy(), ch["load"].to_numpy(),
                          ch["subject"].to_numpy(), tune=True)
            r = {"phase": phase, "gap_before_instruction": gap, "features": name,
                 **g1.summarise(acc)}
            res.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                  flush=True)
    pd.DataFrame(res).to_csv(out / feats.name.replace("features", "summary"), index=False)


if __name__ == "__main__":
    main()
