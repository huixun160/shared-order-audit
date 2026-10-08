"""Reproduce BenchNIRS (Benerradi et al. 2023) on Shin 2018 n-back, then move its epochs
before block onset and change nothing else.

BenchNIRS reports subject-independent 0/2/3-back accuracy of 38.9 % (LDA) and 39.3 %
(CNN) against a 33.3 % chance level, using its own loader and its published
"generalised" pipeline (examples/generalised.py): IIR band-pass 0.01-0.5 Hz, TDDR,
baseline -2..0 s, four ROI channels, features mean/std/slope over 0..9.9 s, nested
group cross-validation.

Here the only change is ``SHIFT_S``: every block marker is moved that many seconds
earlier before epoching. With a 12 s shift the feature window becomes -12..-2.1 s and
the baseline -14..-12 s relative to the true onset - entirely before the 2 s instruction
screen. If accuracy stays near the published figure, the published margin above chance
is available before the block starts, i.e. from the shared block order.

Runs in a separate venv with ``pip install benchnirs`` (1.7.1), not the project env:

    <scratchpad>/benv/Scripts/python scripts/30_benchnirs_shift.py 0 12
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import benchnirs.load as bl
from benchnirs.learn import machine_learn
from benchnirs.load import load_dataset
from benchnirs.process import extract_features, process_epochs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / "shin2018_matlab"
OUT = ROOT / "results" / "benchnirs_shift"
CLASSES = ["0-back", "2-back", "3-back"]

_orig_load = bl._DatasetShin2018NB.load


def _shifted_loader(shift_s: float):
    def load(self, subject, path=bl.PATH_SHIN_2018):
        data = _orig_load(self, subject, path)
        lab = data["labels"]
        idx = np.flatnonzero(lab[0])
        k = int(round(shift_s * data["sfreq"]))
        new = np.zeros_like(lab)
        new[0, idx - k] = lab[0, idx]
        data["labels"] = new
        return data
    return load


def run(shift_s: float) -> dict:
    bl._DatasetShin2018NB.load = _shifted_loader(shift_s)
    epochs = load_dataset("shin_2018_nb", str(DATA), bandpass=[0.01, 0.5],
                          baseline=(-2, 0), roi_sides=True, tddr=True)
    nirs, labels, groups = process_epochs(epochs[CLASSES], tmax=9.9)
    feats = extract_features(nirs, ["mean", "std", "slope"])
    out = {"shift_s": shift_s, "n_epochs": int(len(labels))}
    for model in ("lda", "svc", "knn"):
        folder = OUT / f"shift{int(shift_s)}_{model}"
        folder.mkdir(parents=True, exist_ok=True)
        acc, hps, _ = machine_learn(model, feats, labels, groups, output_folder=str(folder))
        out[model] = {"folds": [float(a) for a in acc], "mean": float(np.mean(acc)),
                      "sd": float(np.std(acc))}
        print(shift_s, model, round(float(np.mean(acc)), 4), [round(float(a), 3) for a in acc],
              flush=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shifts = [float(a) for a in sys.argv[1:]] or [0.0, 12.0]
    res = [run(s) for s in shifts]
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
