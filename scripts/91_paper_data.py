"""Per-subject accuracies for the manuscript figures.

Several analyses saved only group summaries. This recomputes the per-subject values
with the same functions and settings (``20_tufts_g1_probe.loso`` with inner-CV C), from
the cached feature tables, and writes ``results/paper/*.csv``.

    python scripts/91_paper_data.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from eegfnirs.utils import project_paths  # noqa: E402

g1 = importlib.import_module("20_tufts_g1_probe")
b21 = importlib.import_module("21_tufts_baselines")
wg = importlib.import_module("26_shin_wg_control")


def _loso(df, cols, y):
    return g1.loso(df[cols].to_numpy(), df[y].to_numpy(), df["subject"].to_numpy(), tune=True)


def shin_probe(res: Path) -> pd.DataFrame:
    a = pd.read_csv(res / "shin_nullwindow" / "features.csv")
    b = pd.read_csv(res / "shin_nullwindow" / "features_lp_only_none.csv")
    df = pd.concat([a, b])
    df = df[df.gap == 2.0].sort_values(["subject", "run", "position"])
    out = []
    for phase, ch in df.groupby("phase"):
        sets = {"fNIRS": [c for c in ch.columns if c.startswith("fnirs")]}
        if phase in ("minimum", "zero"):
            sets["EEG"] = [c for c in ch.columns if c.startswith("eeg_")]
        for mod, cols in sets.items():
            acc = _loso(ch, cols, "load")
            out.append(acc.rename("accuracy").to_frame().assign(filter=phase, modality=mod))
    return pd.concat(out).rename_axis("subject").reset_index()


def tufts_probe(raw: Path) -> pd.DataFrame:
    rows = []
    for s in b21.tufts.subjects(raw):
        rows += b21.file_rows(b21.tufts.read_recording(raw, s, "test"), "test68")
    df = pd.DataFrame(rows)
    df = df[df.baseline.isin(["causal_long", "whole_file"])]
    feat = [c for c in df.columns if c.startswith("f")]
    out = []
    for (win, base), ch in df.groupby(["window", "baseline"]):
        ch = ch.sort_values(["subject", "position"])
        acc = _loso(ch, feat, "n")
        out.append(acc.rename("accuracy").to_frame().assign(window=win, baseline=base))
    return pd.concat(out).rename_axis("subject").reset_index()


def wg_control(res: Path) -> pd.DataFrame:
    df = pd.read_csv(res / "shin_wg_control" / "features.csv")
    df = df[df.gap == 2.0].sort_values(["subject", "run", "position"])
    cols = [c for c in df.columns if c.startswith("fnirs")]
    acc = _loso(df, cols, "label")
    oracle = (df["label"] != df["prev"]).groupby(df["subject"]).mean()
    return pd.DataFrame({"accuracy": acc, "rule_oracle": oracle}).rename_axis("subject").reset_index()


def main() -> None:
    p = project_paths()
    out = p.results / "paper"
    out.mkdir(parents=True, exist_ok=True)
    shin_probe(p.results).to_csv(out / "shin_probe_per_subject.csv", index=False)
    print("shin done", flush=True)
    tufts_probe(p.raw).to_csv(out / "tufts_probe_per_subject.csv", index=False)
    print("tufts done", flush=True)
    wg_control(p.results).to_csv(out / "wg_control_per_subject.csv", index=False)
    print("wg done", flush=True)


if __name__ == "__main__":
    main()
