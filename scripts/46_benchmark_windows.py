"""Raw fNIRS window tensors for the expanded benchmark (revision, R3).

For every block: a 10 s null window ending at the instruction (Tufts) or 2 s before it
(Shin; the instruction is 2 s), and a 10 s task window 20-30 s after onset. Per recording,
each channel is centred and scaled by its whole-recording mean and sd (label-free,
the same constant for every window of that recording).

Outputs results/benchmark/{shin_none,shin_minimum,tufts_raw}.npz with
X_null, X_task (n_blocks, channels, time), y, subject, block.

    python scripts/46_benchmark_windows.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.signal import resample  # noqa: E402

from eegfnirs.data import shin2018, tufts  # noqa: E402
from eegfnirs.eval.canonical import LOAD_OF  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402

nw = importlib.import_module("22_shin_nullwindow")
WIN = 10.0
FS_OUT = 5.0                    # common rate, 50 samples per window
NT = int(WIN * FS_OUT)


def cut(t, x, t0):
    m = (t >= t0) & (t < t0 + WIN)
    seg = x[:, m]
    return resample(seg, NT, axis=1) if seg.shape[1] != NT else seg


def shin(phase: str):
    p = project_paths()
    Xn, Xt, y, subj, blk = [], [], [], [], []
    for sd in sorted((p.interim / "shin2018").glob("VP*")):
        for run in shin2018.NBACK.runs:
            t, x = nw._hb(shin2018.load_nirs_run(sd, run), phase)
            x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
            shift = shin2018.align_run(sd, run).offset
            blocks = shin2018.segment_blocks(shin2018.parse_vmrk(sd / "eeg" / (run + ".vmrk")),
                                             shin2018.NBACK, context=sd.name + "/" + run)
            for pos, b in enumerate(blocks, start=1):
                onset = (b.onset_sample - 1) / shin2018.EEG_SFREQ + shift
                Xn.append(cut(t, x, onset - nw.INSTRUCTION_S - WIN))
                Xt.append(cut(t, x, onset + 20.0))
                y.append(LOAD_OF[b.condition]); subj.append(sd.name); blk.append(f"{run}_{pos}")
        print(phase, sd.name, flush=True)
    return np.stack(Xn), np.stack(Xt), np.array(y), np.array(subj), np.array(blk)


def tufts_raw():
    p = project_paths()
    Xn, Xt, y, subj, blk = [], [], [], [], []
    for s in tufts.subjects(p.raw):
        rec = tufts.read_recording(p.raw, s, "test")
        x = rec.x.copy()
        ac = [i for i, c in enumerate(rec.channels) if c.startswith("AC-")]
        x[:, ac] = np.log(np.clip(x[:, ac], 1e-6, None))
        x = ((x - x.mean(0)) / x.std(0)).T
        for _, b in rec.blocks.iterrows():
            if not b.has_rest_before:
                continue
            Xn.append(cut(rec.t, x, b.instr_start - WIN))
            Xt.append(cut(rec.t, x, b.onset + 20.0))
            y.append(int(b.n)); subj.append(str(s)); blk.append(str(b.position))
    return np.stack(Xn), np.stack(Xt), np.array(y), np.array(subj), np.array(blk)


def main() -> None:
    out = project_paths().results / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("tufts_raw", tufts_raw), ("shin_none", lambda: shin("none")),
                     ("shin_minimum", lambda: shin("minimum"))):
        Xn, Xt, y, subj, blk = fn()
        np.savez_compressed(out / f"{name}.npz", X_null=Xn.astype(np.float32),
                            X_task=Xt.astype(np.float32), y=y, subject=subj, block=blk)
        print(name, Xn.shape, Xt.shape, np.bincount(y), flush=True)


if __name__ == "__main__":
    main()
