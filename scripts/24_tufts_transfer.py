"""Does a Tufts workload decoder actually decode workload, or the shared block order?

Cross-transfer between task windows and rest-end windows (H5), on the raw test file
of all 68 subjects (blocks 14-16: n = 0, 1, 2; plus 0 vs 2 as in the official binary
benchmark). All four train -> test combinations, leave-one-subject-out:

    task -> task   the usual decoder
    task -> rest   does the task decoder already "predict" the coming block during the
                   rest before it, when nothing about that block has been shown yet?
    rest -> task   a decoder that never saw a task: how well does it score on tasks?
    rest -> rest   the null-window probe

If rest -> task approaches task -> task, the task benchmark score is largely available
from order alone; if task -> rest is above chance, the task decoder relies on it.

Preprocessing variants: ``raw`` (no filter) and ``official`` (3rd-order Butterworth
0.001-0.2 Hz applied zero-phase to the whole recording, as in the Tufts release).
Mean features are relative to the recording mean (``whole_file``) or to the data up to
the end of the file's first task block (``causal_long``, no future data).

    python scripts/24_tufts_transfer.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal, stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from eegfnirs.data import tufts  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402

WINDOW_S = 4.0
TASK_ENDS = np.arange(10.0, 81.0, 10.0)      # task windows end 10, 20, ..., 80 s after onset
C = 0.1


def _prep(rec: tufts.Recording, variant: str) -> np.ndarray:
    x = rec.x.copy()
    ac = [i for i, c in enumerate(rec.channels) if c.startswith("AC-")]
    x[:, ac] = np.log(np.clip(x[:, ac], 1e-6, None))
    if variant == "official":
        fs = 1.0 / np.median(np.diff(rec.t))
        b, a = signal.butter(3, [0.001, 0.2], btype="band", fs=fs)
        x = signal.filtfilt(b, a, x, axis=0)
    return x


def _feat(t, x, t_end):
    m = (t >= t_end - WINDOW_S) & (t < t_end)
    if m.sum() < 3:
        return None
    seg, tt = x[m], t[m] - t[m].mean()
    slope = (tt[:, None] * (seg - seg.mean(0))).sum(0) / (tt ** 2).sum()
    return seg.mean(0), slope


def build(raw: Path) -> pd.DataFrame:
    rows = []
    for s in tufts.subjects(raw):
        rec = tufts.read_recording(raw, s, "test")
        for variant in ("raw", "official"):
            x = _prep(rec, variant)
            first = rec.blocks.iloc[0]
            bases = {"whole_file": x.mean(0), "causal_long": x[rec.t <= first.offset].mean(0)}
            for _, b in rec.blocks.iterrows():
                if not b.has_rest_before:
                    continue
                wins = [("rest", 0.0, b.instr_start)] + [("task", e, b.onset + e) for e in TASK_ENDS]
                for kind, offset, t_end in wins:
                    f = _feat(rec.t, x, t_end)
                    if f is None:
                        continue
                    for bname, base in bases.items():
                        rows.append({"subject": s, "variant": variant, "baseline": bname,
                                     "n": b.n, "kind": kind, "offset": offset,
                                     **{f"f{i}": v for i, v in enumerate(np.r_[f[0] - base, f[1]])}})
    return pd.DataFrame(rows)


def transfer(df: pd.DataFrame, feat: list[str], train_kind: str, test_kind: str) -> pd.Series:
    acc = {}
    for s in df["subject"].unique():
        tr = df[(df.subject != s) & (df.kind == train_kind)]
        te = df[(df.subject == s) & (df.kind == test_kind)]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        m.fit(tr[feat].to_numpy(), tr["n"].to_numpy())
        acc[s] = float((m.predict(te[feat].to_numpy()) == te["n"].to_numpy()).mean())
    return pd.Series(acc)


def main() -> None:
    paths = project_paths()
    out = paths.results / "tufts_transfer"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "features.parquet"
    df = pd.read_parquet(cache) if cache.exists() else build(paths.raw)
    df.to_parquet(cache)
    feat = [c for c in df.columns if c.startswith("f")]

    res, per = [], []
    for (variant, base), ch in df.groupby(["variant", "baseline"]):
        for task_name, labels in (("3class", (0, 1, 2)), ("0vs2", (0, 2))):
            sub = ch[ch["n"].isin(labels)]
            chance = 1.0 / len(labels)
            for trk in ("task", "rest"):
                for tek in ("task", "rest"):
                    acc = transfer(sub, feat, trk, tek)
                    _, p = stats.wilcoxon(acc - chance, alternative="greater", zero_method="zsplit")
                    r = {"variant": variant, "baseline": base, "task": task_name,
                         "train": trk, "test": tek, "chance": chance,
                         "accuracy": acc.mean(), "excess": acc.mean() - chance, "p": p,
                         "n_above": int((acc > chance).sum()), "n_below": int((acc < chance).sum())}
                    res.append(r)
                    per.append(acc.rename("accuracy").to_frame().assign(**{k: r[k] for k in
                               ("variant", "baseline", "task", "train", "test")}))
                    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                          flush=True)
    pd.DataFrame(res).to_csv(out / "summary.csv", index=False)
    pd.concat(per).to_csv(out / "per_subject.csv")


if __name__ == "__main__":
    main()
