"""Recompute the main real-data results with the calibrated permutation test, and add the
second diagnostic step (rest -> task transfer) for Shin and Tufts.

Every row: LOSO balanced accuracy, chance, permutation p (labels permuted across blocks
within subject, whole pipeline re-run). C fixed at 0.1 (no inner tuning, so that every
permutation runs the identical pipeline).

    python scripts/45_permutation_real.py [n_perm]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from eegfnirs.eval.permutation import permutation_test  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402


def shin(res: Path, n_perm: int) -> list[dict]:
    df = pd.read_csv(res / "shin_null_task" / "features.csv", keep_default_na=False)  # "null" is a label
    feat = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    rows = []
    for phase, d in df.groupby("phase"):
        for tr, te, name in (("null", "null", "null-window probe"),
                             ("null", "task", "rest->task transfer"),
                             ("task", "task", "task decoding")):
            r = permutation_test(d, feat, train_kind=tr, test_kind=te, n_perm=n_perm)
            rows.append(dict(dataset="Shin 2018 n-back", setting=f"fNIRS, filter={phase}",
                             analysis=name, **r))
            print(rows[-1], flush=True)
    return rows


def tufts(res: Path, n_perm: int) -> list[dict]:
    df = pd.read_parquet(res / "tufts_transfer" / "features.parquet")
    df = df[df.variant == "raw"].copy()
    df["block"] = df["n"]                      # one block per load in blocks 14-16
    df["kind"] = df["kind"].replace({"rest": "null"})
    feat = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    rows = []
    for base, d in df.groupby("baseline"):
        for task, labels in (("3-class", (0, 1, 2)), ("0 vs 2", (0, 2))):
            dd = d[d["n"].isin(labels)]
            for tr, te, name in (("null", "null", "null-window probe"),
                                 ("null", "task", "rest->task transfer"),
                                 ("task", "task", "task decoding")):
                r = permutation_test(dd, feat, label="n", train_kind=tr, test_kind=te,
                                     n_perm=n_perm)
                rows.append(dict(dataset="Tufts fNIRS2MW (raw, blocks 14-16)",
                                 setting=f"{task}, baseline={base}", analysis=name, **r))
                print(rows[-1], flush=True)
    return rows


def wg(res: Path, n_perm: int) -> list[dict]:
    df = pd.read_csv(res / "shin_wg_control" / "features.csv")
    df = df[df.gap == 2.0].copy()
    df["block"] = df["run"] + "_" + df["position"].astype(str)
    df["kind"] = "null"
    feat = [c for c in df.columns if c.startswith("fnirs")]
    r = permutation_test(df, feat, label="label", n_perm=n_perm)
    print(r, flush=True)
    return [dict(dataset="Shin 2018 word generation (per-subject order)",
                 setting="fNIRS, causal", analysis="null-window probe", **r)]


def main() -> None:
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    res = project_paths().results
    rows = shin(res, n_perm) + tufts(res, n_perm) + wg(res, n_perm)
    out = res / "permutation_real"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
