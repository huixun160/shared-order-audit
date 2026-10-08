"""Per-subject statistics for the BenchNIRS shift experiment (scripts/30_benchnirs_shift.py).

The BenchNIRS pipeline reports 5 outer folds, too few for a test. This rebuilds the
identical features (same loader, filter, TDDR, baseline, ROI averaging, mean/std/slope
over 0..9.9 s) for each marker shift and scores BenchNIRS's LDA under leave-one-subject-
out, so each of the 26 subjects contributes one accuracy. Same venv as script 30:

    <scratchpad>/benv/Scripts/python scripts/31_benchnirs_shift_loso.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

spec = importlib.util.spec_from_file_location("s30", Path(__file__).with_name("30_benchnirs_shift.py"))
s30 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s30)


def features(shift_s):
    s30.bl._DatasetShin2018NB.load = s30._shifted_loader(shift_s)
    ep = s30.load_dataset("shin_2018_nb", str(s30.DATA), bandpass=[0.01, 0.5],
                          baseline=(-2, 0), roi_sides=True, tddr=True)
    nirs, labels, groups = s30.process_epochs(ep[s30.CLASSES], tmax=9.9)
    f = s30.extract_features(nirs, ["mean", "std", "slope"])
    return f.reshape(len(f), -1), np.asarray(labels), np.asarray(groups)


def loso(x, y, g):
    acc = {}
    for s in np.unique(g):
        m = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        m.fit(x[g != s], y[g != s])
        acc[int(s)] = float((m.predict(x[g == s]) == y[g == s]).mean())
    return acc


def main():
    res = {}
    for shift in (0.0, 12.0, 16.0):
        acc = loso(*features(shift))
        a = np.array(list(acc.values()))
        p = stats.wilcoxon(a - 1 / 3, alternative="greater").pvalue
        res[shift] = {"per_subject": acc, "mean": float(a.mean()), "p": float(p),
                      "n_above": int((a > 1 / 3).sum())}
        print(shift, round(a.mean(), 4), "p", round(p, 4), "above", res[shift]["n_above"], "/", len(a), flush=True)
    a0, a12 = (np.array(list(res[s]["per_subject"].values())) for s in (0.0, 12.0))
    print("paired post vs pre(12 s): p", stats.wilcoxon(a0, a12).pvalue)
    (s30.OUT / "loso_lda.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
