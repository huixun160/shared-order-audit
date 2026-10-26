"""Real-data decomposition of the Shin 2018 n-back null-window leak (revision, R2).

Shin's three runs use different block orders (a Latin square across runs, identical for
every subject), so within this shared design the previous block's load and the block's
position in the run are not aliased with each other. Two complementary analyses:

1. Conditional decoding: decode the current load from the null window *within strata*
   that hold one design variable fixed.
     - same previous load   -> what remains cannot come from carry-over of the previous
                               block's load
     - same position in run -> what remains cannot come from position / drift
   Balanced accuracy, LOSO, per subject; chance = 1 / number of classes in the stratum.
2. Mean-level regression: null-window HbO / HbR ~ position + previous load + run + subject.

Uses the null-window features of scripts/22_shin_nullwindow.py (window ends 2 s before
the instruction).

    python scripts/41_shin_decompose.py
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

from eegfnirs.utils import project_paths  # noqa: E402

ORD = {"nback1": [3, 2, 0, 2, 0, 3, 0, 3, 2], "nback2": [2, 0, 3, 0, 3, 2, 3, 2, 0],
       "nback3": [0, 3, 2, 3, 2, 0, 2, 0, 3]}


def balanced_loso(d: pd.DataFrame, cols, strata: str) -> pd.Series:
    """Pooled-over-strata balanced accuracy per subject; a model per stratum."""
    pred = pd.Series(index=d.index, dtype=float)
    for _, g in d.groupby(strata):
        if g["load"].nunique() < 2:
            continue
        for s in g["subject"].unique():
            tr, te = g[g.subject != s], g[g.subject == s]
            if tr["load"].nunique() < 2 or te.empty:
                continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
            m.fit(tr[cols].to_numpy(), tr["load"].to_numpy())
            pred[te.index] = m.predict(te[cols].to_numpy())
    d = d.assign(pred=pred).dropna(subset=["pred"])
    out = {}
    for s, x in d.groupby("subject"):
        rec = [(x.pred[(x.load == c) & (x[strata] == k)] == c).mean()
               for k in x[strata].unique() for c in x.load[x[strata] == k].unique()]
        out[s] = float(np.mean(rec))
    return pd.Series(out)


def chance_of(d: pd.DataFrame, strata: str) -> float:
    return float(np.mean([1 / g["load"].nunique() for _, g in d.groupby(strata)
                          if g["load"].nunique() >= 2]))


def main() -> None:
    res = project_paths().results
    df = pd.concat([pd.read_csv(res / "shin_nullwindow" / "features.csv"),
                    pd.read_csv(res / "shin_nullwindow" / "features_lp_only_none.csv")])
    df = df[df.gap == 2.0].copy()
    df["prev"] = [ORD[r][p - 2] if p > 1 else -1 for r, p in zip(df.run, df.position)]
    df["one"] = 0
    rows = []
    for phase, ch in df.groupby("phase"):
        ch = ch[ch.prev >= 0].reset_index(drop=True)
        cols = [c for c in ch.columns if c.startswith("fnirs")]
        for strata, label in (("one", "unconditional"), ("prev", "same previous load"),
                              ("position", "same position in run")):
            acc = balanced_loso(ch, cols, strata)
            chance = chance_of(ch, strata) if strata != "one" else 1 / 3
            p = stats.wilcoxon(acc - chance, alternative="greater", zero_method="zsplit").pvalue
            rows.append(dict(filter=phase, analysis=label, chance=chance, accuracy=acc.mean(),
                             excess=acc.mean() - chance, p=p, n=len(acc)))
            print(rows[-1], flush=True)
    out = res / "shin_decompose"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "conditional_decoding.csv", index=False)


if __name__ == "__main__":
    main()
