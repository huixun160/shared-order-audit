"""Simulation experiments for the revision (docs/PLAN_REVISION_2026-10.md, section 1).

E1 calibration     pure noise; false-positive rate of the null-window probe per design
E2 sensitivity     one temporal component at a time, per design; null and task accuracy
E3 identifiability design-variable regression recovers which component is present,
                   and when it cannot (aliased designs)
E4 ceilings        accuracy obtainable from design variables alone (no signal)

    python scripts/40_sim_experiments.py E1 E2 E3 E4
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eegfnirs.sim.generator import SimConfig, simulate, subject_order  # noqa: E402
from eegfnirs.sim.probe import design_regression, features, loso, permutation_p, summary  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402

OUT = project_paths().results / "sim"
DESIGNS = ("shared", "shared_rest", "latin", "random", "random_rule")
NOISE = dict(beta=0.0, beta0=0.0, kappa=0.0, mu_drift=0.0, phi=0.0)


def run(cfg: SimConfig, subset=None) -> dict:
    df = features(simulate(cfg))
    if subset is not None:
        df = df[df.position.isin(subset)]
    chance = 1.0 / df["n"].nunique()
    n = summary(loso(df, "null"), chance)
    t = summary(loso(df, "task"), chance)
    return dict(null_acc=n["accuracy"], null_p=n["p"], task_acc=t["accuracy"], task_p=t["p"],
                chance=chance, **design_regression(df))


def e1(n_seeds=200, n_perm=100):
    """Calibration under pure noise: Wilcoxon-vs-chance (fold-dependent, anticonservative)
    and the within-subject label-permutation test (re-runs the whole LOSO)."""
    rows = []
    for design, seed in itertools.product(("shared", "latin", "random"), range(n_seeds)):
        df = features(simulate(SimConfig(design=design, seed=seed, **NOISE)))
        acc = loso(df, "null")
        w = summary(acc, 0.25)
        obs, p_perm, _ = permutation_p(df, n_perm=n_perm, seed=10_000 + seed)
        rows.append(dict(design=design, seed=seed, null_acc=obs, p_wilcoxon=w["p"],
                         p_perm=p_perm))
        if seed % 20 == 0:
            print(design, seed, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E1_calibration.csv", index=False)
    print(df.groupby("design").agg(null_acc=("null_acc", "mean"),
                                   fpr_wilcoxon=("p_wilcoxon", lambda p: (p < 0.05).mean()),
                                   fpr_perm=("p_perm", lambda p: (p < 0.05).mean())))


GRID = {
    "task_only": [dict()],
    "carryover": [dict(kappa=k) for k in (0.25, 0.5, 1.0)],
    "drift": [dict(mu_drift=d) for d in (1.0, 2.0, 4.0)],
    "position": [dict(phi=p) for p in (0.05, 0.1, 0.2)],
}


def e2(n_seeds=10):
    rows = []
    for comp, settings in GRID.items():
        for kw in settings:
            for design, seed in itertools.product(("shared", "shared_rest", "latin", "random"),
                                                  range(n_seeds)):
                for sub_name, subset in (("all", None), ("last3", (14, 15, 16))):
                    r = run(SimConfig(design=design, seed=seed, **kw), subset)
                    rows.append(dict(component=comp, setting=str(kw), design=design, seed=seed,
                                     eval=sub_name, **r))
            print(comp, kw, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E2_sensitivity.csv", index=False)
    print(df.groupby(["component", "setting", "eval", "design"])[["null_acc", "task_acc"]]
          .mean().round(3).to_string())


def e3(n_seeds=20):
    cases = {"carryover": dict(beta=0, beta0=0, kappa=1.0),
             "drift": dict(beta=0, beta0=0, mu_drift=4.0, sd_drift=0.5),
             "position": dict(beta=0, beta0=0, phi=0.2)}
    rows = []
    for (case, kw), design, seed in itertools.product(cases.items(),
                                                      ("shared", "shared_rest", "random"),
                                                      range(n_seeds)):
        df = features(simulate(SimConfig(design=design, seed=seed, **kw)))
        for sub_name, subset in (("all", None), ("last3", (14, 15, 16))):
            d = df if subset is None else df[df.position.isin(subset)]
            rows.append(dict(truth=case, design=design, seed=seed, eval=sub_name,
                             **design_regression(d)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E3_identifiability.csv", index=False)
    print(df.groupby(["eval", "truth", "design"]).mean(numeric_only=True)
          .drop(columns="seed").round(2).to_string())


def e4():
    """Bayes-optimal accuracy from design variables alone, pooled over subjects."""
    rng = np.random.default_rng(0)
    rows = []
    for design in DESIGNS:
        cfg = SimConfig(design=design)
        recs = []
        for s in range(400):
            o = subject_order(cfg, s, rng)
            for pos in range(2, 17):
                recs.append(dict(position=pos, n=o[pos - 1], prev=o[pos - 2]))
        d = pd.DataFrame(recs)
        for var in ("position", "prev"):
            best = d.groupby(var)["n"].agg(lambda v: v.value_counts().iloc[0]).sum() / len(d)
            rows.append(dict(design=design, predictor=var, ceiling=best))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E4_ceilings.csv", index=False)
    print(df.pivot(index="design", columns="predictor", values="ceiling").round(3))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for e in sys.argv[1:] or ("E1", "E2", "E3", "E4"):
        {"E1": e1, "E2": e2, "E3": e3, "E4": e4}[e]()
