"""Gate G1: is there pre-onset (null-window) decodability on Tufts fNIRS2MW?

Uses the raw ``fnirs_test.csv`` of every subject (blocks 13-16, n = 3, 0, 1, 2 - the
same for everyone). Blocks 14-16 (n = 0, 1, 2) each have a rest -> instruction -> task
sequence before them, which gives a 3-class problem with chance 1/3 over 68 subjects.
Block 13 is dropped: its pre-task segment is the start of the recording.

Windows (4 s, raw data, no filtering):

* ``rest_end_gapG`` - ends G s before the instruction screen appears. Nothing about the
  coming block has been shown yet, so anything decodable here is order leakage.
* ``pre_onset``     - ends at task onset, i.e. inside the instruction + beep period. The
  screen shows the coming n, so this mixes anticipation with leakage. Reported only
  to show why the window has to be anchored to the instruction, not the task.
* ``task``          - 20-24 s after onset, as a reference for real decodability.

Statistics are per subject (3 blocks each): one-sided Wilcoxon against 1/3, and a
permutation null that shuffles labels *independently within each subject*, i.e. the
null a design with per-subject random order would give.

    python scripts/20_tufts_g1_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from eegfnirs.data import tufts  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402

WINDOW_S = 4.0
CHANCE = 1.0 / 3.0
C_GRID = [0.01, 0.1, 1.0, 10.0]
N_PERM = 500
RNG = np.random.default_rng(0)


def anchors(b: pd.Series) -> dict[str, float]:
    a = {f"rest_end_gap{g}": b.instr_start - g for g in (0, 2, 4)}
    a["pre_onset"] = b.onset
    a["task"] = b.onset + 24.0
    return a


def build(raw: Path) -> pd.DataFrame:
    rows = []
    for s in tufts.subjects(raw):
        rec = tufts.read_recording(raw, s, "test")
        if rec is None:
            continue
        # recording-wide mean per channel: label-free per-subject baseline for the
        # "mean" features (raw intensity offsets differ by an order of magnitude)
        x = rec.x.copy()
        ac = [i for i, c in enumerate(rec.channels) if c.startswith("AC-")]
        x[:, ac] = np.log(np.clip(x[:, ac], 1e-6, None))
        base = np.r_[x.mean(0), np.zeros(x.shape[1])]
        for _, b in rec.blocks.iterrows():
            if not b.has_rest_before:
                continue
            for name, t_end in anchors(b).items():
                f = tufts.window_features(rec, t_end, WINDOW_S)
                if f is None:
                    continue
                rows.append({"subject": s, "n": b.n, "position": b.position,
                             "window": name, "prev_rest_s": b.prev_rest_s,
                             **{f"f{i}": v for i, v in enumerate(f - base)}})
    return pd.DataFrame(rows)


def loso(X, y, groups, *, tune: bool) -> pd.Series:
    acc = {}
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
        if tune:
            pipe = GridSearchCV(pipe, {"logisticregression__C": C_GRID},
                                cv=GroupKFold(5)).fit(X[tr], y[tr], groups=groups[tr])
        else:
            pipe.fit(X[tr], y[tr])
        acc[s] = float((pipe.predict(X[te]) == y[te]).mean())
    return pd.Series(acc)


def summarise(per_subj: pd.Series) -> dict:
    _, p = stats.wilcoxon(per_subj - CHANCE, alternative="greater", zero_method="zsplit")
    return {"n": len(per_subj), "accuracy": per_subj.mean(),
            "excess": per_subj.mean() - CHANCE, "p_wilcoxon": p,
            "n_above": int((per_subj > CHANCE).sum()),
            "n_below": int((per_subj < CHANCE).sum())}


def main() -> None:
    paths = project_paths()
    out = paths.results / "tufts_g1"
    out.mkdir(parents=True, exist_ok=True)

    df = build(paths.raw)
    df.to_csv(out / "features.csv", index=False)
    feat = [c for c in df.columns if c.startswith("f")]
    nch = len(feat) // 2
    sets = {"mean+slope": feat, "slope_only": feat[nch:]}

    rows, per = [], []
    for (win, chunk) in df.groupby("window"):
        chunk = chunk.sort_values(["subject", "position"])
        y, g = chunk["n"].to_numpy(), chunk["subject"].to_numpy()
        for fs_name, cols in sets.items():
            acc = loso(chunk[cols].to_numpy(), y, g, tune=True)
            rows.append({"window": win, "features": fs_name, **summarise(acc)})
            per.append(acc.rename("accuracy").to_frame().assign(window=win, features=fs_name))
            print(rows[-1], flush=True)

    # permutation null for the main window: labels shuffled within each subject
    main_win = df[df.window == "rest_end_gap0"].sort_values(["subject", "position"])
    X, y, g = main_win[feat].to_numpy(), main_win["n"].to_numpy(), main_win["subject"].to_numpy()
    observed = loso(X, y, g, tune=False).mean()
    null = []
    for _ in range(N_PERM):
        yp = y.copy()
        for s in np.unique(g):
            idx = np.flatnonzero(g == s)
            yp[idx] = RNG.permutation(yp[idx])
        null.append(loso(X, yp, g, tune=False).mean())
    null = np.array(null)
    p_perm = (1 + (null >= observed).sum()) / (1 + N_PERM)
    rows.append({"window": "rest_end_gap0", "features": "mean+slope (C=0.1, perm)",
                 "accuracy": observed, "excess": observed - CHANCE, "p_perm": p_perm,
                 "null_mean": null.mean(), "null_95": np.quantile(null, 0.95)})
    print(rows[-1])

    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.concat(per).to_csv(out / "per_subject.csv")
    rest = df[df.window == "rest_end_gap0"].groupby("n")["prev_rest_s"].describe()
    rest.to_csv(out / "rest_by_load.csv")
    print(rest)


if __name__ == "__main__":
    main()
