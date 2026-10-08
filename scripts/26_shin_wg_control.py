"""Negative control: the same null-window probe on Shin 2018 word generation (WG).

Same 26 subjects, montage, session and preprocessing as the n-back positive
(scripts/22_shin_nullwindow.py), but the WG / baseline trial order is randomised per
subject (24 distinct 3-run sequences across 26 subjects, read from the EEG markers).
WG trials: 2 s instruction, 10 s task, 13-15 s rest; windows end before the instruction.

The randomisation is not unconstrained: P(label differs from the previous trial) = 0.62
for everyone, and with ~25 s trials the previous trial's response is still present
before the next onset. So four numbers are reported:

1. ``current``      - decode the current label (what the n-back probe measured);
2. ``prev_oracle``  - accuracy of "current = not previous", the most the shared
                      randomisation rule alone can give;
3. ``previous``     - decode the *previous* trial's label (size of the carry-over);
4. ``conditional``  - decode the current label separately within prev = WG and
                      prev = baseline trials. With carry-over held fixed, any residual
                      accuracy would be leakage; this is the clean negative control.

    python scripts/26_shin_wg_control.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from eegfnirs.data import shin2018  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402

nw = importlib.import_module("22_shin_nullwindow")
g1 = importlib.import_module("20_tufts_g1_probe")
GAPS = (0.0, 2.0)


def run_rows(subject_dir: Path, run: str) -> list[dict]:
    eeg_t, env = nw._eeg(shin2018.load_eeg_run(subject_dir, run, preload=True), "minimum")
    hb_t, hb = nw._hb(shin2018.load_nirs_run(subject_dir, run), "minimum")
    shift = shin2018.align_run(subject_dir, run).offset
    blocks = shin2018.segment_blocks(shin2018.parse_vmrk(subject_dir / "eeg" / (run + ".vmrk")),
                                     shin2018.WORD, context=subject_dir.name)
    labels = [1 if b.trial_labels[0] == "verbal_fluency" else 0 for b in blocks]
    hb_mean = hb.mean(1)
    rows = []
    for i, b in enumerate(blocks):
        if i == 0:
            continue                                   # no previous trial
        onset = (b.onset_sample - 1) / shin2018.EEG_SFREQ
        for gap in GAPS:
            t1 = onset - nw.INSTRUCTION_S - gap
            t0 = t1 - nw.WINDOW_S
            em = (eeg_t >= t0) & (eeg_t < t1)
            hm = (hb_t >= t0 + shift) & (hb_t < t1 + shift)
            if em.sum() < 10 or hm.sum() < 3:
                continue
            r = {"subject": subject_dir.name, "run": run, "position": i + 1,
                 "label": labels[i], "prev": labels[i - 1], "gap": gap}
            for band, p in env.items():
                for c, v in enumerate(np.log10(p[:, em].mean(1) + 1e-30)):
                    r[f"eeg_{band}_{c}"] = v
            seg = hb[:, hm]
            for c, v in enumerate(seg.mean(1) - hb_mean):
                r[f"fnirsM_{c}"] = v
            for c, v in enumerate(nw._slope(hb_t[hm], seg)):
                r[f"fnirsS_{c}"] = v
            rows.append(r)
    return rows


def _balanced_conditional(d, cols):
    """Per-subject balanced accuracy for the current label, fitted separately within
    prev = 0 and prev = 1 trials and pooled. Plain accuracy would be wrong here: within
    a stratum the labels are ~62:38, so always guessing the majority scores 0.62."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pred = pd.Series(index=d.index, dtype=float)
    for p in (0, 1):
        sub = d[d.prev == p]
        for s in sub["subject"].unique():
            tr, te = sub[sub.subject != s], sub[sub.subject == s]
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
            m.fit(tr[cols].to_numpy(), tr["label"].to_numpy())
            pred[te.index] = m.predict(te[cols].to_numpy())
    out = {}
    for s, x in d.groupby("subject"):
        recalls = [(pred[x.index][x.label == c] == c).mean() for c in (0, 1) if (x.label == c).any()]
        out[s] = float(np.mean(recalls))
    return pd.Series(out)


def _acc(d, cols, target):
    return g1.loso(d[cols].to_numpy(), d[target].to_numpy(), d["subject"].to_numpy(), tune=True)


def main() -> None:
    paths = project_paths()
    out = paths.results / "shin_wg_control"
    out.mkdir(parents=True, exist_ok=True)
    feats = out / "features.csv"
    if feats.exists():
        df = pd.read_csv(feats)
    else:
        rows = []
        for sd in sorted((paths.interim / "shin2018").glob("VP*")):
            for run in shin2018.WORD.runs:
                rows += run_rows(sd, run)
            print("done", sd.name, flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(feats, index=False)

    sets = {"eeg": [c for c in df.columns if c.startswith("eeg_")],
            "fnirs_mean+slope": [c for c in df.columns if c.startswith("fnirs")]}
    g1.CHANCE = 0.5
    res = []
    for gap, ch in df.groupby("gap"):
        ch = ch.sort_values(["subject", "run", "position"]).reset_index(drop=True)
        oracle = (ch["label"] != ch["prev"]).groupby(ch["subject"]).mean()
        res.append({"gap": gap, "features": "-", "analysis": "prev_oracle", **g1.summarise(oracle)})
        for name, cols in sets.items():
            res.append({"gap": gap, "features": name, "analysis": "current",
                        **g1.summarise(_acc(ch, cols, "label"))})
            res.append({"gap": gap, "features": name, "analysis": "previous",
                        **g1.summarise(_acc(ch, cols, "prev"))})
            res.append({"gap": gap, "features": name, "analysis": "conditional_balanced",
                        **g1.summarise(_balanced_conditional(ch, cols))})
        for r in res[-7:]:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(res).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
