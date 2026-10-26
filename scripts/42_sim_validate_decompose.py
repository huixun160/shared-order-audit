"""Does the conditional-decoding decomposition (scripts/41) attribute the leak correctly?

Simulated Shin-like data (three runs of nine blocks, shared run orders, loads 0/2/3) with
exactly one temporal source switched on at a time:

    carryover   previous block's load leaves an extra recovery term     -> should vanish
                                                                           when the previous
                                                                           load is held fixed
    drift       group-mean drift over the session                       -> should survive
    position    step per block position within a run (fatigue-like)     -> should survive
                                                                           with previous load
                                                                           fixed, vanish with
                                                                           position fixed

    python scripts/42_sim_validate_decompose.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from eegfnirs.sim.generator import shin_config, simulate  # noqa: E402
from eegfnirs.sim.probe import features  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402

dec = importlib.import_module("41_shin_decompose")

CASES = {
    "none (pure noise)": dict(beta=0, beta0=0),
    "carryover": dict(beta=0, beta0=0, kappa=1.0),
    "drift": dict(beta=0, beta0=0, mu_drift=4.0, sd_drift=0.5),
    "position": dict(beta=0, beta0=0, phi=0.3),
    "task response only": dict(),
}


def sim_frame(kw, seed):
    recs = simulate(shin_config(n_subjects=26, seed=seed, **kw))
    df = features(recs, skip_first=False)
    meta = pd.DataFrame([dict(subject=r.subject, position=b["position"], run=b["run"],
                              pos_in_run=b["pos_in_run"]) for r in recs for b in r.blocks])
    df = df.merge(meta, on=["subject", "position"])
    df = df[(df.kind == "null") & (df.prev_n >= 0)].reset_index(drop=True)
    df = df.rename(columns={"n": "load", "prev_n": "prev"})
    df = df.drop(columns="position").rename(columns={"pos_in_run": "position"})
    return df.rename(columns={c: "fnirs_" + c for c in df.columns if c.startswith("f") and c[1:].isdigit()})


def main(n_seeds: int = 10) -> None:
    rows = []
    for case, kw in CASES.items():
        for seed in range(n_seeds):
            df = sim_frame(kw, seed)
            df["one"] = 0
            cols = [c for c in df.columns if c.startswith("fnirs_")]
            for strata, label in (("one", "unconditional"), ("prev", "same previous load"),
                                  ("position", "same position in run")):
                acc = dec.balanced_loso(df, cols, strata)
                chance = dec.chance_of(df, strata) if strata != "one" else 1 / 3
                rows.append(dict(case=case, seed=seed, analysis=label,
                                 excess=acc.mean() - chance))
        print(case, flush=True)
    d = pd.DataFrame(rows)
    out = project_paths().results / "sim"
    d.to_csv(out / "E5_decompose_validation.csv", index=False)
    s = d.groupby(["case", "analysis"])["excess"].agg(["mean", "sem"]).round(3)
    s["t_p"] = d.groupby(["case", "analysis"])["excess"].apply(
        lambda x: stats.ttest_1samp(x, 0).pvalue).round(4)
    print(s.unstack("analysis").to_string())


if __name__ == "__main__":
    main()
