"""Con4m (Chen et al., NeurIPS 2024) on Tufts: does it rely on the shared block order?

Con4m reads the official ``whole_data`` (16 task segments per subject, butted together
with rest removed), keeps 0- and 2-back, samples 40 s segments that may straddle a
block boundary, and labels them with a context- and transition-aware model. Under the
fixed Tufts order every subject's 0/2 sequence is the same (0,2,2,0,2,0,0,2).

Intervention: keep every block's content and re-concatenate the 16 blocks in a random
order drawn independently per subject. The joins are artificial in ``whole_data``
anyway, so task content is unchanged; only the shared order is removed.

This script builds the inputs and the segment databases and runs Con4m's own test
script (``b_test.py``). Training is done with Con4m's ``a_train.py`` (3.5 h per run on
CPU here); see docs/REPRO_CON4M_2026-10-08.md for the exact commands.

To separate an order effect from segment-sampling noise, the test databases are
regenerated with several sampling seeds (fixed order) and several independent shuffles
(shuffled order), always with the **same subject groups** as the trained model.

Requires a checkout of github.com/MrNobodyCali/Con4m (``CON4M_DIR``) and a Python with
its dependencies (``CON4M_PYTHON``; transformers==4.38.2, h5py, psutil):

    CON4M_DIR=... CON4M_PYTHON=... python scripts/32_con4m_order.py eval <ckpt_dir> <tag>
"""

from __future__ import annotations

import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CON4M = Path(os.environ.get("CON4M_DIR", ROOT.parent / "Con4m"))
PY = Path(os.environ.get("CON4M_PYTHON", sys.executable))
WORK = ROOT / "data" / "interim" / "con4m"
SRC = ROOT / "data" / "raw" / "fNIRS2MW" / "experiment" / "fNIRS_data" / "band_pass_filtered" / "whole_data"
OUT = ROOT / "results" / "con4m"
BLOCK = 426
# subject groups produced by the original database generation (np.random.seed(0))
GROUPS = [[48, 27, 32, 22, 31, 45, 30, 60, 55, 62, 28, 41, 67, 58, 52, 40, 4],
          [51, 26, 2, 66, 11, 3, 42, 10, 34, 61, 43, 54, 53, 7, 14, 33, 63],
          [29, 49, 35, 18, 0, 15, 5, 16, 20, 65, 8, 13, 25, 37, 17, 56, 46],
          [59, 39, 38, 1, 12, 57, 24, 6, 23, 36, 50, 21, 19, 9, 64, 47, 44]]


def build_inputs(dest: Path, shuffle_seed: int | None) -> None:
    """sub_1..sub_68.csv in Con4m's numbering; blocks re-concatenated if shuffle_seed."""
    mapping = json.loads((WORK / "mapping.json").read_text())["con4m_index_to_tufts_subject"]
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    dest.mkdir(parents=True, exist_ok=True)
    for i, s in sorted(mapping.items(), key=lambda kv: int(kv[0])):
        d = pd.read_csv(SRC / f"sub_{s}.csv")
        blocks = [d.iloc[b * BLOCK:(b + 1) * BLOCK] for b in range(16)]
        if rng is not None:
            blocks = [blocks[k] for k in rng.permutation(16)]
        pd.concat(blocks).to_csv(dest / f"sub_{i}.csv", index=False)


def build_db(load_dir: Path, save_dir: Path, sampling_seed: int) -> None:
    sys.path.insert(0, str(CON4M / "pipeline"))
    gen = runpy.run_path(str(CON4M / "pipeline" / "b_fNIRS_database_generate.py"))
    extract = gen["extract_database"]
    extract.__globals__["__random_split_subject__"] = lambda n, num_split=4: [list(g) for g in GROUPS]
    np.random.seed(sampling_seed)
    extract(root_path=str(load_dir / "Tufts_fNIRS_data"), save_dir=str(save_dir / "fNIRS_2"),
            data_name="fNIRS_2", seg_len=200, seg_num=6, num_class=2, noise_ratio=[0.0], num_level=5)


def run_test(db: Path, ckpt: Path) -> dict:
    cmd = [str(PY), "-u", "b_test.py", "--database_save_dir", str(db) + "/", "--data_name", "fNIRS_2",
           "--noise_ratio", "0", "--exp_id", "1", "--n_process_loader", "1", "--load_path", str(ckpt) + "/"]
    txt = subprocess.run(cmd, cwd=CON4M, capture_output=True, text=True).stdout
    return {k: float(re.search(rf"{k}: ([0-9.]+)", txt).group(1)) for k in ("Accuracy", "F1")}


def evaluate(ckpt: Path, tag: str, n: int = 5) -> None:
    rows = []
    for k in range(1, n + 1):
        for cond in ("fixed", "shuffled"):
            base = WORK / "eval" / f"{cond}_{k}"
            build_inputs(base / "Tufts_fNIRS_data", shuffle_seed=1000 + k if cond == "shuffled" else None)
            build_db(base, base / "db", sampling_seed=k)
            r = {"model": tag, "test_order": cond, "rep": k, **run_test(base / "db", ckpt)}
            rows.append(r)
            print(r, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"eval_{tag}.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate(Path(sys.argv[2]), sys.argv[3])
