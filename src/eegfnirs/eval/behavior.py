"""Behavioural validation of the n-back task.

This should have been the first analysis in the project, not one of the last. It is the
most direct check that the labels mean what we think they mean: if accuracy does not
fall and reaction time does not rise with n-back load, then either the task did not work
or our condition labels are wrong, and every neural result downstream is suspect.

It also supplies a **third independent confirmation of the trigger mapping**. Each
behavioural summary carries an ``nback`` field giving the block-level load sequence for
that run. It is written by the stimulus program and has nothing to do with either
trigger stream, so agreement with the sequence we derive from the EEG markers closes the
question that started as a reverse-engineering exercise.

File layout (``behavior.zip``)::

    N-back/summary/VP0NN-EEG/summary{1,2,3}.mat   ->  one per run

Each contains, for 9 blocks x 20 trials:

===============  ==================================================
``nback``        (9,)      load of each block: 0, 2 or 3
``result``       (9, 20)   1 correct, 0 incorrect, -1 no response required
``reaction_time``(9, 20)   seconds; 2.0 is the timeout / no-response value
``response``     (9, 20)   button code, -1 for none
===============  ==================================================

The leading ``-1`` entries are the first *n* trials of an n-back block, where no
comparison is yet possible - they are excluded rather than scored as errors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eegfnirs.data import shin2018
from eegfnirs.utils.logging import get_logger

log = get_logger(__name__)

RT_TIMEOUT = 2.0
LOAD_TO_CONDITION = {0: "0-back", 2: "2-back", 3: "3-back"}


def behavior_root(interim_root: Path) -> Path:
    return interim_root / "shin2018" / "_behavior" / "N-back" / "summary"


def _unwrap(obj, _max_depth: int = 8):
    """Strip the 0-d object arrays and singleton lists pymatreader wraps structs in.

    Bounded on purpose. ``np.asarray(some_dict)`` yields a 0-d object array whose
    ``.item()`` returns the same dict, so an unguarded ``while`` here spins forever -
    which is exactly what it did the first time this ran.
    """
    for _ in range(_max_depth):
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, (list, tuple)) and len(obj) == 1:
            obj = obj[0]
            continue
        arr = np.asarray(obj, dtype=object) if not isinstance(obj, np.ndarray) else obj
        if arr.dtype == object and arr.shape == ():
            obj = arr.item()
            continue
        if arr.dtype == object and arr.size == 1:
            obj = arr.ravel()[0]
            continue
        return obj
    raise ValueError("could not unwrap MATLAB struct within " + str(_max_depth) + " levels")


def read_summary(path: Path) -> dict:
    from pymatreader import read_mat

    raw = read_mat(str(path))
    key = next(k for k in raw if not k.startswith("__"))
    return _unwrap(raw[key])


def run_behavior(path: Path, subject: str, run: str) -> pd.DataFrame:
    """One row per block: load, accuracy and mean correct reaction time."""
    s = read_summary(path)
    loads = np.asarray(s["nback"]).ravel()
    result = np.asarray(s["result"])
    rt = np.asarray(s["reaction_time"])

    rows = []
    for i, load in enumerate(loads):
        scored = result[i] >= 0                       # drop the un-scorable lead-in
        correct = (result[i] == 1) & scored
        timely = correct & (rt[i] < RT_TIMEOUT)
        rows.append({
            "subject": subject,
            "run": run,
            "block": i,
            "load": int(load),
            "condition": LOAD_TO_CONDITION[int(load)],
            "n_scored": int(scored.sum()),
            "accuracy": float(correct.sum() / scored.sum()) if scored.sum() else np.nan,
            "rt_mean": float(rt[i][timely].mean()) if timely.any() else np.nan,
            "n_timeout": int(((rt[i] >= RT_TIMEOUT) & scored).sum()),
        })
    return pd.DataFrame(rows)


def subject_behavior(interim_root: Path, subject: int) -> pd.DataFrame:
    vp = shin2018.subject_id(subject)
    folder = behavior_root(interim_root) / (vp + "-EEG")
    if not folder.is_dir():
        raise FileNotFoundError("no behavioural data at " + str(folder))
    frames = []
    for i, run in enumerate(shin2018.NBACK.runs, start=1):
        path = folder / ("summary{}.mat".format(i))
        if not path.exists():
            log.warning("%s: missing %s", vp, path.name)
            continue
        frames.append(run_behavior(path, vp, run))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------------------
# Independent verification of the trigger mapping
# --------------------------------------------------------------------------------------

def verify_block_sequence(interim_root: Path, subject: int) -> pd.DataFrame:
    """Compare the behavioural log's block sequence with the one derived from EEG markers.

    The behavioural ``nback`` field is written by the stimulus program; the EEG sequence
    comes from block-onset trigger codes whose meaning we reconstructed. Agreement is
    therefore independent evidence, not a restatement of the same assumption.
    """
    vp = shin2018.subject_id(subject)
    subject_dir = interim_root / "shin2018" / vp
    folder = behavior_root(interim_root) / (vp + "-EEG")

    rows = []
    for i, run in enumerate(shin2018.NBACK.runs, start=1):
        path = folder / ("summary{}.mat".format(i))
        if not path.exists():
            continue
        from_behavior = [LOAD_TO_CONDITION[int(v)]
                         for v in np.asarray(read_summary(path)["nback"]).ravel()]

        markers = shin2018.parse_vmrk(subject_dir / "eeg" / (run + ".vmrk"))
        blocks = shin2018.segment_blocks(markers, shin2018.NBACK,
                                         context=vp + "/" + run)
        from_eeg = [b.condition for b in blocks]

        rows.append({
            "subject": vp,
            "run": run,
            "agree": from_eeg == from_behavior,
            "n_blocks": len(from_eeg),
            "from_eeg": ",".join(from_eeg),
            "from_behavior": ",".join(from_behavior),
        })
    return pd.DataFrame(rows)
