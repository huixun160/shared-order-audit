"""Window features, LOSO decoding and design-variable decomposition for simulated data.

Mirrors the analyses applied to the real datasets (scripts 20-24): 4 s windows, per-channel
mean and slope, mean features relative to the recording mean, standardised logistic
regression, leave-one-subject-out, per-subject accuracy as the unit of analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WINDOW_S = 4.0


def window(rec, t_end: float, base: np.ndarray, window_s: float = WINDOW_S) -> np.ndarray:
    m = (rec.t >= t_end - window_s) & (rec.t < t_end)
    seg, tt = rec.x[m], rec.t[m] - rec.t[m].mean()
    slope = (tt[:, None] * (seg - seg.mean(0))).sum(0) / (tt ** 2).sum()
    return np.r_[seg.mean(0) - base, slope]


def features(recs, skip_first: bool = True) -> pd.DataFrame:
    """Null window = last 4 s of rest before the instruction; task = 20-24 s after onset."""
    rows = []
    for r in recs:
        base = r.x.mean(0)
        for b in r.blocks:
            if skip_first and b["position"] == 1:
                continue
            for kind, te in (("null", b["instr_start"]), ("task", b["onset"] + 24.0)):
                f = window(r, te, base)
                rows.append(dict(subject=r.subject, kind=kind, n=b["n"], position=b["position"],
                                 prev_n=b["prev_n"], prev_rest=b["prev_rest"],
                                 **{f"f{i}": v for i, v in enumerate(f)}))
    return pd.DataFrame(rows)


def _balanced(y, p) -> float:
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def loso(df: pd.DataFrame, train_kind: str = "null", test_kind: str | None = None,
         C: float = 0.1, labels: np.ndarray | None = None) -> pd.Series:
    """Per-subject *balanced* accuracy, so chance is 1/k whatever the class counts
    (dropping the first block makes classes unequal under a shared order).
    ``labels`` replaces df['n'] (used by the permutation test)."""
    test_kind = test_kind or train_kind
    feat = [c for c in df.columns if c.startswith("f")]
    y_all = df["n"].to_numpy() if labels is None else labels
    X_all = df[feat].to_numpy()
    subj, kind = df["subject"].to_numpy(), df["kind"].to_numpy()
    acc = {}
    for s in np.unique(subj):
        tr = (subj != s) & (kind == train_kind)
        te = (subj == s) & (kind == test_kind)
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
        m.fit(X_all[tr], y_all[tr])
        acc[s] = _balanced(y_all[te], m.predict(X_all[te]))
    return pd.Series(acc)


def permutation_p(df: pd.DataFrame, kind: str = "null", n_perm: int = 100,
                  seed: int = 0) -> tuple[float, float, np.ndarray]:
    """Group-mean balanced accuracy vs a null in which each subject's labels are permuted
    independently (i.e. what a per-subject random order would give). The whole LOSO
    pipeline is re-run for every permutation, so the dependence between folds is kept."""
    d = df[df.kind == kind].reset_index(drop=True)
    obs = loso(d, kind).mean()
    rng = np.random.default_rng(seed)
    y = d["n"].to_numpy()
    groups = [np.flatnonzero(d["subject"].to_numpy() == s) for s in d["subject"].unique()]
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = y.copy()
        for g in groups:
            yp[g] = rng.permutation(yp[g])
        null[i] = loso(d, kind, labels=yp).mean()
    return float(obs), float((1 + (null >= obs).sum()) / (1 + n_perm)), null


def summary(acc: pd.Series, chance: float) -> dict:
    p = stats.wilcoxon(acc - chance, alternative="greater", zero_method="zsplit").pvalue
    return dict(accuracy=float(acc.mean()), excess=float(acc.mean() - chance), p=float(p))


def design_regression(df: pd.DataFrame) -> dict:
    """Null-window signal level ~ position + previous load (+ previous rest), subject FE.

    Returns standardised coefficients and the design's condition number. In a shared
    order these regressors are collinear by construction; with per-subject random order
    they are not, and the coefficients identify the separate sources.
    """
    d = df[df.kind == "null"].copy()
    feat = [c for c in d.columns if c.startswith("f")]
    nch = len(feat) // 2
    d["y"] = d[feat[:nch]].mean(1)
    cols = ["position", "prev_n"] + (["prev_rest"] if d["prev_rest"].nunique() > 1 else [])
    X = d[cols].astype(float)
    X = X - X.groupby(d["subject"]).transform("mean")
    y = d["y"] - d.groupby("subject")["y"].transform("mean")
    Xs = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    beta, *_ = np.linalg.lstsq(Xs.to_numpy(), y.to_numpy(), rcond=None)
    cond = float(np.linalg.cond(Xs.to_numpy()))
    corr = float(np.corrcoef(X["position"], X["prev_n"])[0, 1])
    return dict(**{f"b_{c}": float(b) for c, b in zip(cols, beta)}, cond=cond, r_pos_prev=corr)
