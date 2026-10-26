"""Permutation p-values for the BenchNIRS shift experiment (scripts/30, 31).

Same features as scripts/31_benchnirs_shift_loso.py; LOSO LDA; per-subject labels
permuted across epochs, whole LOSO re-run (500 permutations). Run in the BenchNIRS venv.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("s31", Path(__file__).with_name("31_benchnirs_shift_loso.py"))
s31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s31)


def main(n_perm=500):
    res = {}
    for shift in (0.0, 12.0):
        x, y, g = s31.features(shift)
        obs = np.mean(list(s31.loso(x, y, g).values()))
        rng = np.random.default_rng(777)
        null = []
        for _ in range(n_perm):
            yp = y.copy()
            for s in np.unique(g):
                i = np.flatnonzero(g == s)
                yp[i] = rng.permutation(yp[i])
            null.append(np.mean(list(s31.loso(x, yp, g).values())))
        null = np.array(null)
        res[shift] = dict(acc=float(obs), null_mean=float(null.mean()),
                          p_perm=float((1 + (null >= obs).sum()) / (1 + n_perm)))
        print(shift, res[shift], flush=True)
    (s31.s30.OUT / "perm_lda.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
