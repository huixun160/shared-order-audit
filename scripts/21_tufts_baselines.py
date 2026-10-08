"""G1 follow-up (a): does the Tufts null-window effect survive other subject-offset removals?

G1 (scripts/20_tufts_g1_probe.py) subtracted each subject's whole-recording channel
means. That constant cannot carry label information, but it is computed from data that
lie after the probe windows, and a short 15 s causal baseline left the effect
non-significant. This script settles which of the two readings is right.

Baselines for the "mean" features (slopes need none):

* ``whole_file``  - recording-wide mean (the G1 choice).
* ``causal_long`` - mean over everything up to the end of the file's first task block
  (~100 s). Every probe window lies after it, so it uses no future data.
* ``causal_15s``  - first 15 s of the file (the short baseline that failed).
* ``none``        - raw values.

Two cohorts:

* ``test68`` - the last 4 blocks of all 68 subjects (3 classes; blocks 14-16).
* ``full9``  - the 9 subjects with all 16 blocks (4 classes; 14 windows each, blocks 1
  and 13 excluded because nothing precedes them but the recording start).

    python scripts/21_tufts_baselines.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eegfnirs.data import tufts  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402

g1 = importlib.import_module("20_tufts_g1_probe")
WINDOW_S = 4.0


def _log_ac(rec: tufts.Recording) -> np.ndarray:
    x = rec.x.copy()
    ac = [i for i, c in enumerate(rec.channels) if c.startswith("AC-")]
    x[:, ac] = np.log(np.clip(x[:, ac], 1e-6, None))
    return x


def file_rows(rec: tufts.Recording, cohort: str) -> list[dict]:
    x = _log_ac(rec)
    first = rec.blocks.iloc[0]
    bases = {
        "whole_file": x.mean(0),
        "causal_long": x[rec.t <= first.offset].mean(0),
        "causal_15s": x[rec.t < rec.t[0] + 15.0].mean(0),
        "none": np.zeros(x.shape[1]),
    }
    rows = []
    for _, b in rec.blocks.iterrows():
        if not b.has_rest_before:
            continue
        for win, t_end in (("rest_end", b.instr_start), ("task", b.onset + 24.0)):
            f = tufts.window_features(rec, t_end, WINDOW_S)
            if f is None:
                continue
            nch = len(f) // 2
            for name, base in bases.items():
                rows.append({"cohort": cohort, "subject": rec.subject, "part": rec.part,
                             "position": b.position, "n": b.n, "window": win,
                             "baseline": name,
                             **{f"f{i}": v for i, v in enumerate(np.r_[f[:nch] - base, f[nch:]])}})
    return rows


def main() -> None:
    paths = project_paths()
    out = paths.results / "tufts_g1"
    rows = []
    for s in tufts.subjects(paths.raw):
        test = tufts.read_recording(paths.raw, s, "test")
        rows += file_rows(test, "test68")
        train = tufts.read_recording(paths.raw, s, "train")
        if train is not None:
            rows += file_rows(train, "full9") + file_rows(test, "full9")
    df = pd.DataFrame(rows)
    feat = [c for c in df.columns if c.startswith("f")]

    res = []
    for (cohort, win, base), ch in df.groupby(["cohort", "window", "baseline"]):
        ch = ch.sort_values(["subject", "part", "position"])
        acc = g1.loso(ch[feat].to_numpy(), ch["n"].to_numpy(), ch["subject"].to_numpy(),
                      tune=True)
        chance = 1.0 / ch["n"].nunique()
        g1.CHANCE = chance
        r = {"cohort": cohort, "window": win, "baseline": base, "chance": chance,
             **g1.summarise(acc)}
        res.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(res).to_csv(out / "baselines.csv", index=False)


if __name__ == "__main__":
    main()
