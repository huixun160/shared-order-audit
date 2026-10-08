"""Positive control for the null-window probe on Tufts: inject a known drift.

Starting from the real rest-end window features (G1, 68 subjects, positions 14-16),
each subject's three windows are **shuffled across positions**, which destroys whatever
order information the real data carry while keeping real noise and real between-subject
variability. Labels are then the shared-order labels of the positions (0, 1, 2), and a
drift ``k * D * step`` is added, where ``D`` is the observed group-mean change per
position in the real data. So:

* ``k = 0`` should give chance - the probe does not fire on scrambled data;
* the smallest ``k`` that is reliably detected is the probe's sensitivity;
* if ``k = 1`` reproduces the observed accuracy (0.50), the real leakage is fully
  accounted for by a drift common to all subjects.

    python scripts/23_tufts_injection.py
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

from eegfnirs.utils import project_paths  # noqa: E402

g1 = importlib.import_module("20_tufts_g1_probe")
KS = (0.0, 0.25, 0.5, 1.0, 2.0)
N_REP = 20
RNG = np.random.default_rng(1)


def main() -> None:
    out = project_paths().results / "tufts_g1"
    df = pd.read_csv(out / "features.csv")
    df = df[df.window == "rest_end_gap0"].sort_values(["subject", "position"])
    feat = [c for c in df.columns if c.startswith("f")]
    subjects = df["subject"].unique()
    X = df[feat].to_numpy().reshape(len(subjects), 3, len(feat))   # subject, position, feat
    D = (X[:, 2] - X[:, 0]).mean(0) / 2.0                            # mean change per step
    y = np.tile([0, 1, 2], len(subjects))
    g = np.repeat(subjects, 3)

    rows = []
    for k in KS:
        accs = []
        for _ in range(N_REP):
            Xs = np.stack([x[RNG.permutation(3)] for x in X])
            Xs = Xs + k * D[None, None, :] * np.arange(3)[None, :, None]
            accs.append(g1.loso(Xs.reshape(-1, len(feat)), y, g, tune=False))
        per_subj = pd.concat(accs, axis=1).mean(1)
        _, p = stats.wilcoxon(per_subj - 1 / 3, alternative="greater", zero_method="zsplit")
        rows.append({"k": k, "accuracy": float(np.mean([a.mean() for a in accs])),
                     "sd_over_reps": float(np.std([a.mean() for a in accs])),
                     "p_wilcoxon_rep_mean": p})
        print(rows[-1], flush=True)
    real = g1.loso(X.reshape(-1, len(feat)), y, g, tune=False).mean()
    rows.append({"k": "real data", "accuracy": real})
    print(rows[-1])
    pd.DataFrame(rows).to_csv(out / "injection.csv", index=False)


if __name__ == "__main__":
    main()
