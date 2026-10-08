"""PhysioNet mental-fnirs (Mukli et al. 2021): is 0-back vs 2-back separable from position?

Every subject did four ~134 s blocks in one fixed order: 0-back, 1-back, 0-back, 2-back
(event codes 1/2/4/8 mark the *position*). 12 of the 14 subjects have fNIRS. With one
block per condition and the 2-back always last, load and time-in-session are aliased.
The ~10 s between blocks is undocumented, so no pre-onset window is used here; the
test needs none:

    position 1 vs position 3 are **both 0-back**. A cross-subject decoder that separates
    them as well as it separates 0-back from 2-back is reading time, not load.

Features: HbO/HbR from optical density (TDDR, modified Beer-Lambert, no filter or a
causal 0.2 Hz low-pass), mean and slope over 30 s windows at 10-40, 40-70, 70-100 and
100-130 s after onset, relative to the recording mean. LOSO logistic regression (C=0.1).

    python scripts/27_mentalfnirs_position.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from eegfnirs.data.nirx import read_nirx, to_mne_raw  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402

LOAD_AT = {1: "0a", 2: "1", 3: "0b", 4: "2"}          # position -> condition
WINDOWS = ((10, 40), (40, 70), (70, 100), (100, 130))
CONTRASTS = {
    "0b vs 2  (load, adjacent positions 3-4)": (3, 4),
    "0a vs 2  (load, positions 1-4)": (1, 4),
    "0a vs 1  (load, positions 1-2)": (1, 2),
    "0b vs 1  (load, positions 3-2)": (3, 2),
    "0a vs 0b (SAME load, positions 1-3)": (1, 3),
}


def subject_rows(d: Path, variant: str) -> list[dict]:
    import mne
    rec = read_nirx(d)
    raw = to_mne_raw(rec)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="ERROR")
    od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(od, verbose="ERROR")
    hb = mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)
    if variant == "lp_causal":
        hb.filter(None, 0.2, phase="minimum", verbose="ERROR")
    x, t = hb.get_data() * 1e6, hb.times
    base = x.mean(1)
    rows = []
    for pos, ev in enumerate(sorted(rec.events, key=lambda e: e.time), start=1):
        for w0, w1 in WINDOWS:
            m = (t >= ev.time + w0) & (t < ev.time + w1)
            seg, tt = x[:, m], t[m] - t[m].mean()
            slope = (tt * (seg - seg.mean(1, keepdims=True))).sum(1) / (tt ** 2).sum()
            rows.append({"subject": d.name, "position": pos, "cond": LOAD_AT[pos],
                         "window": f"{w0}-{w1}", "variant": variant,
                         **{f"f{i}": v for i, v in enumerate(np.r_[seg.mean(1) - base, slope])}})
    return rows


def loso(d: pd.DataFrame, feat: list[str], y: str) -> pd.Series:
    acc = {}
    for s in d["subject"].unique():
        tr, te = d[d.subject != s], d[d.subject == s]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        m.fit(tr[feat].to_numpy(), tr[y].to_numpy())
        acc[s] = float((m.predict(te[feat].to_numpy()) == te[y].to_numpy()).mean())
    return pd.Series(acc)


def main() -> None:
    paths = project_paths()
    out = paths.results / "mentalfnirs"
    out.mkdir(parents=True, exist_ok=True)
    root = next(paths.raw.glob("mental-workload*-1.0")) / "NIRS_data"
    rows = []
    for d in sorted(root.glob("SUBJ*")):
        for variant in ("none", "lp_causal"):
            rows += subject_rows(d, variant)
    df = pd.DataFrame(rows)
    df.to_csv(out / "features.csv", index=False)
    feat = [c for c in df.columns if c.startswith("f")]

    res = []
    for variant, ch in df.groupby("variant"):
        for name, (a, b) in CONTRASTS.items():
            sub = ch[ch.position.isin([a, b])].copy()
            sub["y"] = (sub.position == b).astype(int)
            acc = loso(sub, feat, "y")
            _, p = stats.wilcoxon(acc - 0.5, alternative="greater", zero_method="zsplit")
            res.append({"variant": variant, "contrast": name, "n": len(acc),
                        "accuracy": acc.mean(), "p": p,
                        "n_above": int((acc > 0.5).sum()), "n_below": int((acc < 0.5).sum())})
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[-1].items()},
                  flush=True)
    pd.DataFrame(res).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
