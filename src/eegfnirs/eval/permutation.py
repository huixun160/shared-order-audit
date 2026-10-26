"""Leave-one-subject-out decoding with a calibrated permutation test.

Per-subject LOSO accuracies are not independent (every fold shares most of its training
data), so a Wilcoxon test of those accuracies against chance is anticonservative; in
simulation it rejected a true null in 8-12 % of runs at alpha = 0.05
(results/sim/E1_calibration.csv). Here the null distribution is built by re-running the
entire LOSO pipeline after permuting, within each subject, which label belongs to which
block. That keeps the fold dependence, the class counts per subject and the pairing of
windows taken from the same block (needed for train-on-null / test-on-task transfer),
and corresponds to what a per-subject random condition order would give.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _balanced(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def loso_balanced(X: np.ndarray, y: np.ndarray, subj: np.ndarray, train: np.ndarray,
                  test: np.ndarray, C: float = 0.1) -> np.ndarray:
    """Per-subject balanced accuracy; ``train`` / ``test`` are boolean row masks."""
    out = []
    for s in np.unique(subj):
        tr, te = train & (subj != s), test & (subj == s)
        if not te.any():
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        m.fit(X[tr], y[tr])
        out.append(_balanced(y[te], m.predict(X[te])))
    return np.asarray(out)


def permutation_test(df: pd.DataFrame, feat: list[str], label: str = "load",
                     subject: str = "subject", block: str = "block", kind: str = "kind",
                     train_kind: str = "null", test_kind: str | None = None,
                     n_perm: int = 500, seed: int = 12345, C: float = 0.1) -> dict:
    test_kind = test_kind or train_kind
    d = df.reset_index(drop=True)
    X = d[feat].to_numpy(float)
    y = d[label].to_numpy()
    subj = d[subject].to_numpy()
    train = (d[kind] == train_kind).to_numpy()
    test = (d[kind] == test_kind).to_numpy()
    acc = loso_balanced(X, y, subj, train, test, C)
    obs = float(acc.mean())

    blocks = d[[subject, block, label]].drop_duplicates([subject, block])
    rows_of = {k: np.flatnonzero((d[subject] == k[0]).to_numpy() & (d[block] == k[1]).to_numpy())
               for k in zip(blocks[subject], blocks[block])}
    by_subj = {s: g for s, g in blocks.groupby(subject)}
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = y.copy()
        for s, g in by_subj.items():
            perm = rng.permutation(g[label].to_numpy())
            for (b, lab) in zip(g[block], perm):
                yp[rows_of[(s, b)]] = lab
        null[i] = loso_balanced(X, yp, subj, train, test, C).mean()
    k = len(np.unique(y[test]))
    return dict(accuracy=obs, chance=1.0 / k, excess=obs - 1.0 / k,
                null_mean=float(null.mean()), null_95=float(np.quantile(null, 0.95)),
                p_perm=float((1 + (null >= obs).sum()) / (1 + n_perm)),
                n_subjects=len(acc), n_perm=n_perm)
