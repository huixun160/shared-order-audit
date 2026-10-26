"""Shin 2018 n-back fNIRS: null-window and task-window features from the same pipeline.

Needed for the second step of the diagnostic (rest -> task transfer: train on null
windows, test on task windows). fNIRS only; same preprocessing as
scripts/22_shin_nullwindow.py. Null window: 4 s ending 2 s before the instruction
(i.e. 4 s before the marked onset). Task window: 20-24 s after onset.

    python scripts/44_shin_null_task.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eegfnirs.data import shin2018  # noqa: E402
from eegfnirs.eval.canonical import LOAD_OF  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402

nw = importlib.import_module("22_shin_nullwindow")
WIN = 4.0
WINDOWS = {"null": (-nw.INSTRUCTION_S - 2.0 - WIN, -nw.INSTRUCTION_S - 2.0), "task": (20.0, 24.0)}


def run_rows(subject_dir: Path, run: str, phase: str) -> list[dict]:
    hb_t, hb = nw._hb(shin2018.load_nirs_run(subject_dir, run), phase)
    shift = shin2018.align_run(subject_dir, run).offset
    blocks = shin2018.segment_blocks(shin2018.parse_vmrk(subject_dir / "eeg" / (run + ".vmrk")),
                                     shin2018.NBACK, context=subject_dir.name + "/" + run)
    base = hb.mean(1)
    rows = []
    for pos, b in enumerate(blocks, start=1):
        onset = (b.onset_sample - 1) / shin2018.EEG_SFREQ
        for kind, (w0, w1) in WINDOWS.items():
            m = (hb_t >= onset + w0 + shift) & (hb_t < onset + w1 + shift)
            seg = hb[:, m]
            r = dict(subject=subject_dir.name, run=run, position=pos, block=f"{run}_{pos}",
                     load=LOAD_OF[b.condition], kind=kind, phase=phase)
            r.update({f"f{i}": v for i, v in enumerate(np.r_[seg.mean(1) - base,
                                                             nw._slope(hb_t[m], seg)])})
            rows.append(r)
    return rows


def main() -> None:
    p = project_paths()
    out = p.results / "shin_null_task"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sd in sorted((p.interim / "shin2018").glob("VP*")):
        for run in shin2018.NBACK.runs:
            for phase in ("minimum", "none"):
                rows += run_rows(sd, run, phase)
        print("done", sd.name, flush=True)
    pd.DataFrame(rows).to_csv(out / "features.csv", index=False)


if __name__ == "__main__":
    main()
