"""When does order leakage inflate *task* decoding? (revision, R3/R4)

Task-window LOSO accuracy with a temporal component switched on, minus the same design and
seed with it switched off ("inflation"), as a function of the true task effect size.
Null-window probe excess and rest->task transfer (train on null windows, test on task
windows) are recorded alongside, to test whether the probe predicts the
inflation once the task signal is weak enough for the order signal to matter.

    python scripts/43_sim_inflation.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from eegfnirs.sim.generator import SimConfig, simulate  # noqa: E402
from eegfnirs.sim.probe import features, loso  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402

BETAS = (0.05, 0.15, 0.3, 1.0)
COMPS = {"off": {}, "carryover": dict(kappa=0.5), "drift": dict(mu_drift=2.0),
         "position": dict(phi=0.1)}


def main(n_seeds: int = 8) -> None:
    rows = []
    for beta, (comp, kw), design, seed in itertools.product(
            BETAS, COMPS.items(), ("shared", "random"), range(n_seeds)):
        df = features(simulate(SimConfig(design=design, seed=seed, beta=beta, beta0=beta / 2, **kw)))
        for ev, subset in (("all", None), ("last3", (14, 15, 16))):
            d = df if subset is None else df[df.position.isin(subset)]
            k = d["n"].nunique()
            rows.append(dict(beta=beta, component=comp, design=design, seed=seed, eval=ev,
                             null=loso(d, "null").mean() - 1 / k,
                             transfer=loso(d, "null", "task").mean() - 1 / k,
                             task=loso(d, "task").mean() - 1 / k))
        print(beta, comp, design, seed, flush=True) if seed == 0 else None
    d = pd.DataFrame(rows)
    off = d[d.component == "off"].set_index(["beta", "design", "seed", "eval"])["task"]
    d = d[d.component != "off"].copy()
    d["inflation"] = [r.task - off.loc[(r.beta, r.design, r.seed, r.eval)] for r in d.itertuples()]
    out = project_paths().results / "sim"
    d.to_csv(out / "E6_inflation.csv", index=False)
    s = d.groupby(["eval", "component", "beta", "design"])[["null", "transfer", "inflation"]].mean().round(3)
    print(s.unstack("design").to_string())
    for ev, g in d.groupby("eval"):
        g = g.groupby(["component", "beta", "design"])[["null", "transfer", "inflation"]].mean()
        print(ev, "probe vs inflation r=%.2f p=%.2g" % stats.pearsonr(g["null"], g["inflation"]))
        print(ev, "transfer vs inflation r=%.2f p=%.2g" % stats.pearsonr(g["transfer"], g["inflation"]))


if __name__ == "__main__":
    main()
