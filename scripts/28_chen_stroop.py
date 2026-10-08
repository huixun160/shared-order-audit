"""Chen et al. 2023 (Sci Data) EEG-fNIRS Stroop: condition vs time-in-session.

**Order, established from the data** (not the paper): every subject saw four 32 s blocks
in the order neutral, incongruent, neutral, incongruent (EEG event types 1-2-1-2;
matching the authors' own per-condition preprocessed fNIRS back onto the raw timeline
gives N_1 = position 1, I_1 = 2, N_2 = 3, I_2 = 4 for 17 of 19 checkable subjects).
The paper's text says "neutral, neutral, incongruent, incongruent", which the data
contradict. Excluded: S14 (2000 s recording, stray markers), S18 (first block start
missing), S10 (block-to-condition match inconsistent) -> 18 subjects.

Because conditions alternate, neutral/incongruent pairs come in two temporal kinds:

    forward  (1 vs 2, 3 vs 4): neutral earlier than incongruent
    reverse  (2 vs 3):         incongruent earlier than neutral

A condition decoder should be equally right on both. A decoder that has learned
"later = incongruent" is right on forward pairs and wrong on the reverse pair. A pure
time decoder (trained only on same-condition pairs, early vs late) makes this explicit.

fNIRS: optical density of all 40 raw columns (20 channels x 785/850 nm), unfiltered or
causal 0.2 Hz low-pass, mean and slope over 8 s windows inside each block, relative to
the recording mean. Pre-onset windows (break before each block; only a beep precedes a
block, no visual cue) are probed too.

    python scripts/28_chen_stroop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io as sio  # noqa: E402
from scipy import signal, stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from eegfnirs.utils import project_paths  # noqa: E402

FS = 100.0
EXCLUDE = (10, 14, 18)
COND_AT = {1: 0, 2: 1, 3: 0, 4: 1}                 # 0 = neutral, 1 = incongruent
BLOCK_PERIOD_S = 56.1
TASK_WINDOWS = ((5, 13), (13, 21), (21, 29), (29, 37))
PRE_WINDOW = (-7.0, -1.0)                          # inside the break; beep is at -0.5 s


def block_starts(marker: np.ndarray) -> list[float] | None:
    thr = (marker.max() + np.median(marker)) / 2
    t = (np.flatnonzero((marker[1:] > thr) & (marker[:-1] <= thr)) + 1) / FS
    for c in t:
        chain = [c + BLOCK_PERIOD_S * k for k in range(4)]
        if all(np.any(np.abs(t - x) < 0.8) for x in chain):
            return [float(t[np.argmin(np.abs(t - x))]) for x in chain]
    return None


def subject_rows(path: Path, variant: str) -> list[dict]:
    yy = sio.loadmat(path, squeeze_me=True)["yy"]
    starts = block_starts(np.asarray(yy[2], float))
    x = -np.log(np.clip(np.stack([np.asarray(yy[i], float) for i in range(3, 43)], 1), 1e-6, None))
    if variant == "lp_causal":
        b, a = signal.butter(3, 0.2, btype="low", fs=FS)
        x = signal.lfilter(b, a, x - x[:200].mean(0), axis=0)
    t = np.arange(len(x)) / FS
    base = x.mean(0)
    rows = []
    for pos, s0 in enumerate(starts, start=1):
        wins = [("task", w) for w in TASK_WINDOWS] + [("pre", PRE_WINDOW)]
        for kind, (w0, w1) in wins:
            m = (t >= s0 + w0) & (t < s0 + w1)
            seg, tt = x[m], t[m] - t[m].mean()
            slope = (tt[:, None] * (seg - seg.mean(0))).sum(0) / (tt ** 2).sum()
            rows.append({"subject": path.stem, "position": pos, "cond": COND_AT[pos],
                         "kind": kind, "window": f"{w0}-{w1}", "variant": variant,
                         **{f"f{i}": v for i, v in enumerate(np.r_[seg.mean(0) - base, slope])}})
    return rows


def _fit(tr, feat, y):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    return m.fit(tr[feat].to_numpy(), tr[y].to_numpy())


def pair_accuracy(d: pd.DataFrame, feat: list[str], train_y: str, pairs) -> pd.DataFrame:
    """LOSO. ``train_y`` = "cond" (condition decoder) or "late" (time decoder trained on
    same-condition pairs only). Accuracy is always scored against the condition."""
    out = []
    for s in d["subject"].unique():
        tr = d[d.subject != s]
        if train_y == "late":
            tr = tr[tr.position.isin([1, 3]) | tr.position.isin([2, 4])].copy()
            tr["late"] = (tr.position >= 3).astype(int)      # 1 vs 3 and 2 vs 4
        m = _fit(tr, feat, train_y)
        te = d[d.subject == s]
        for name, (a, b) in pairs.items():
            x = te[te.position.isin([a, b])]
            pred = m.predict(x[feat].to_numpy())
            if train_y == "late":                       # "late" -> the later block's condition
                later_cond = COND_AT[max(a, b)]
                pred = np.where(pred == 1, later_cond, 1 - later_cond)
            out.append({"subject": s, "pair": name, "accuracy": float((pred == x["cond"]).mean())})
    return pd.DataFrame(out)


PAIRS = {"forward 1v2": (1, 2), "forward 3v4": (3, 4), "reverse 2v3": (2, 3)}


def main() -> None:
    paths = project_paths()
    out = paths.results / "chen_stroop"
    out.mkdir(parents=True, exist_ok=True)
    root = paths.raw / "chen2023_stroop" / "Raw fNIRS Data In mat"
    rows = []
    for s in range(1, 22):
        if s in EXCLUDE:
            continue
        for variant in ("none", "lp_causal"):
            rows += subject_rows(root / f"S{s}.mat", variant)
    df = pd.DataFrame(rows)
    feat = [c for c in df.columns if c.startswith("f")]

    res = []
    for (variant, kind), ch in df.groupby(["variant", "kind"]):
        for decoder in ("cond", "late"):
            pa = pair_accuracy(ch, feat, decoder, PAIRS)
            for pair, g in pa.groupby("pair"):
                acc = g.set_index("subject")["accuracy"]
                _, p_hi = stats.wilcoxon(acc - 0.5, alternative="greater", zero_method="zsplit")
                _, p_lo = stats.wilcoxon(acc - 0.5, alternative="less", zero_method="zsplit")
                res.append({"variant": variant, "window": kind, "decoder": decoder, "pair": pair,
                            "n": len(acc), "accuracy": acc.mean(), "p_above": p_hi, "p_below": p_lo})
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[-1].items()},
                      flush=True)
    pd.DataFrame(res).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
