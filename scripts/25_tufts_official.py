"""Order reliance inside the *official* Tufts benchmark data.

Uses the released ``band_pass_filtered/whole_data`` (68 subjects x 16 task segments of
426 samples, 8 haemoglobin channels, zero-phase 0.001-0.2 Hz). Rest periods are not in
this release, so the null-window probe cannot be run here; instead:

1. **Benchmark replication**: 30 s windows (150 samples, stride 15) with the official
   LR features (per-channel mean, std, slope, intercept), generic LOSO, 0 vs 2-back.
2. **Within-block time course**: 5 s windows ending 5, 10, ..., 80 s after onset. The
   first window (0-5 s) precedes most of the haemodynamic response; decodability there
   that matches late windows points to information carried in from before the block.
3. **Accuracy by block position**: the first 0-back (position 1) is the only 0-back not
   preceded by a 3-back and a 40 s rest. If the generic decoder relies on that
   carry-over, it should do worse there than on 0-backs at positions 8, 11, 14.

    python scripts/25_tufts_official.py
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

from eegfnirs.data import tufts  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402

BLOCK = 426
FS = 5.2
C = 0.1


def official_features(seg: np.ndarray) -> np.ndarray:
    """Tufts LR features: per channel mean, std, slope, intercept."""
    t = np.arange(len(seg))
    tt = t - t.mean()
    slope = (tt[:, None] * (seg - seg.mean(0))).sum(0) / (tt ** 2).sum()
    intercept = seg.mean(0) - slope * t.mean()
    return np.r_[seg.mean(0), seg.std(0), slope, intercept]


def windows(raw: Path) -> pd.DataFrame:
    root = raw / "fNIRS2MW" / "experiment" / "fNIRS_data" / "band_pass_filtered" / "whole_data"
    rows = []
    for f in sorted(root.glob("sub_*.csv")):
        s = int(f.stem.split("_")[1])
        x = pd.read_csv(f).drop(columns="label").to_numpy(float)
        for b in range(16):
            blk = x[b * BLOCK:(b + 1) * BLOCK]
            n, pos = tufts.ORDER[b], b + 1
            for end in range(150, BLOCK + 1, 15):           # benchmark: 30 s windows
                rows.append({"subject": s, "n": n, "position": pos, "kind": "w30",
                             "end_s": end / FS, **dict(enumerate(official_features(blk[end - 150:end])))})
            for k in range(1, 17):                           # time course: 5 s windows
                end = min(int(round(k * 5 * FS)), BLOCK)
                rows.append({"subject": s, "n": n, "position": pos, "kind": "w5",
                             "end_s": k * 5.0, **dict(enumerate(official_features(blk[end - 26:end])))})
    df = pd.DataFrame(rows)
    df.columns = [f"f{c}" if isinstance(c, (int, np.integer)) else c for c in df.columns]
    return df


def loso_predict(d: pd.DataFrame, feat: list[str]) -> pd.Series:
    pred = pd.Series(index=d.index, dtype=float)
    for s in d["subject"].unique():
        tr, te = d["subject"] != s, d["subject"] == s
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        m.fit(d.loc[tr, feat].to_numpy(), d.loc[tr, "n"].to_numpy())
        pred[te] = m.predict(d.loc[te, feat].to_numpy())
    return pred


def per_subject_p(d: pd.DataFrame, correct: pd.Series, chance: float):
    acc = correct.groupby(d["subject"]).mean()
    _, p = stats.wilcoxon(acc - chance, alternative="greater", zero_method="zsplit")
    return acc.mean(), p


def main() -> None:
    paths = project_paths()
    out = paths.results / "tufts_official"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "windows.parquet"
    df = pd.read_parquet(cache) if cache.exists() else windows(paths.raw)
    df.to_parquet(cache)
    feat = [c for c in df.columns if c.startswith("f")]
    d02 = df[df["n"].isin([0, 2])]

    # 1. benchmark replication
    w30 = d02[d02.kind == "w30"].copy()
    w30["correct"] = loso_predict(w30, feat) == w30["n"]
    acc, p = per_subject_p(w30, w30["correct"], 0.5)
    last8 = w30[w30.position >= 9]
    acc8, _ = per_subject_p(last8, last8["correct"], 0.5)
    print(f"benchmark 0vs2 30s LOSO: all blocks {acc:.3f} (p={p:.2g}); blocks 9-16 {acc8:.3f}")

    by_pos = (w30.groupby(["n", "position"])["correct"].mean().rename("accuracy").reset_index())
    by_pos["prev_n"] = [tufts.ORDER[p - 2] if p > 1 else -1 for p in by_pos.position]
    print(by_pos.to_string(index=False))
    by_pos.to_csv(out / "accuracy_by_position.csv", index=False)

    # first 0-back vs later 0-backs, paired within subject
    z = w30[w30.n == 0].groupby(["subject", "position"])["correct"].mean().unstack()
    diff = z[1] - z[[8, 11, 14]].mean(1)
    _, p_pos = stats.wilcoxon(diff)
    print(f"0-back pos1 {z[1].mean():.3f} vs pos8/11/14 {z[[8, 11, 14]].mean(1).mean():.3f}, "
          f"paired Wilcoxon p={p_pos:.2g}")

    # 2. within-block time course
    rows = []
    for e, ch in d02[d02.kind == "w5"].groupby("end_s"):
        ch = ch.copy()
        corr = loso_predict(ch, feat) == ch["n"]
        a, pp = per_subject_p(ch, corr, 0.5)
        rows.append({"window_end_s": e, "accuracy": a, "p": pp})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "timecourse_0vs2.csv", index=False)
    w30.drop(columns=feat).to_csv(out / "w30_predictions.csv", index=False)


if __name__ == "__main__":
    main()
