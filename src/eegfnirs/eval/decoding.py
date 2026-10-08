"""Cross-subject decoding of working-memory load.

Deliberately linear. With 26 subjects and 27 blocks each there is no honest way to claim
a deep model is warranted, and a simple, well-regularised baseline is what makes any
later gain from a learned model interpretable rather than decorative.

Three things this module produces:

1. **Subject-disjoint (LOSO) accuracy** for EEG alone, fNIRS alone and the two
   concatenated. No trial from a test subject is ever seen in training.
2. **Label-efficiency curves**: accuracy on a held-out subject as a function of the
   number of that subject's own labelled blocks used for calibration.
3. **A haemodynamic lag sweep.** fNIRS features are read at ``t + lag`` relative to the
   EEG window, and we sweep ``lag``. If decoding peaks near the canonical 5-7 s
   haemodynamic delay, that is direct evidence for modelling the lag - obtained from a
   linear model, before any contrastive machinery exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eegfnirs.data.preprocess import BANDS, PreprocessedRun
from eegfnirs.eval.canonical import CONDITIONS, LOAD_OF, PREFRONTAL_IDX
from eegfnirs.utils.logging import get_logger

log = get_logger(__name__)

WINDOW_S = 10.0
STEP_S = 5.0
TASK_WINDOW = (2.0, 42.0)      # within-block task period
DEFAULT_FNIRS_LAG = 5.0        # seconds; swept in lag_sweep()
LAG_GRID = (0.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0)


def _eeg_band_envelopes(eeg) -> dict[str, np.ndarray]:
    """Squared band-limited signal per channel. One filter per band, not one PSD per window."""
    picks = [i for i, t in enumerate(eeg.get_channel_types()) if t == "eeg"]
    out = {}
    for name, (lo, hi) in BANDS.items():
        filt = eeg.copy().pick(picks).filter(lo, hi, verbose="ERROR")
        out[name] = filt.get_data() ** 2
    return out


def window_features(
    pr: PreprocessedRun,
    *,
    window_s: float = WINDOW_S,
    step_s: float = STEP_S,
    fnirs_lags=LAG_GRID,
) -> pd.DataFrame:
    """Sliding-window features for every n-back block of one run.

    Windows come from inside task blocks only. They overlap, so windows from one block
    are strongly dependent -- which is why the evaluation is subject-disjoint and the
    calibration protocol samples whole blocks, never windows.

    fNIRS features are emitted at every lag in ``fnirs_lags`` (columns are suffixed
    ``@<lag>``), so the whole lag sweep costs one pass over the raw data instead of one
    pass per lag.
    """
    if pr.task != "nback":
        return pd.DataFrame()

    env = _eeg_band_envelopes(pr.eeg)
    eeg_times = pr.eeg.times
    hb_times = pr.hb.times
    hb_types = pr.hb.get_channel_types()
    picks_hbo = [i for i, t in enumerate(hb_types) if t == "hbo"]
    picks_hbr = [i for i, t in enumerate(hb_types) if t == "hbr"]
    sel_hbo = [picks_hbo[i] for i in PREFRONTAL_IDX if i < len(picks_hbo)]
    sel_hbr = [picks_hbr[i] for i in PREFRONTAL_IDX if i < len(picks_hbr)]
    hb_data = pr.hb.get_data()

    rows = []
    for block_i, (onset, condition) in enumerate(zip(pr.block_onsets_s, pr.block_conditions)):
        start = TASK_WINDOW[0]
        while start + window_s <= TASK_WINDOW[1]:
            t0, t1 = onset + start, onset + start + window_s
            m = (eeg_times >= t0) & (eeg_times < t1)
            if m.sum() < 10:
                start += step_s
                continue

            feat: dict[str, float] = {}
            for band, power in env.items():
                vals = np.log10(power[:, m].mean(axis=1) + 1e-30)
                for ch_i, v in enumerate(vals):
                    feat["eeg_{}_{}".format(band, ch_i)] = float(v)

            # fNIRS read at t + lag on the fNIRS clock, for every candidate lag.
            for lag in fnirs_lags:
                n0 = t0 + pr.hb_time_shift + lag
                n1 = t1 + pr.hb_time_shift + lag
                hm = (hb_times >= n0) & (hb_times < n1)
                ok = hm.sum() >= 2
                idx = np.nonzero(hm)[0]
                for tag, sel in (("hbo", sel_hbo), ("hbr", sel_hbr)):
                    if ok:
                        vals = hb_data[np.ix_(sel, idx)].mean(axis=1) * 1e6
                    else:
                        vals = np.full(len(sel), np.nan)
                    for ch_i, v in enumerate(vals):
                        feat["fnirs_{}_{}@{:g}".format(tag, ch_i, lag)] = float(v)

            feat.update({
                "subject": pr.subject,
                "run": pr.run,
                "block": pr.run + "_b{}".format(block_i),
                "condition": condition,
                "load": LOAD_OF[condition],
            })
            rows.append(feat)
            start += step_s

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

META = ("subject", "run", "block", "condition", "load")


def feature_columns(df: pd.DataFrame, modality: str,
                    lag: float = DEFAULT_FNIRS_LAG) -> list[str]:
    """Columns for a modality. fNIRS columns are selected at one lag."""
    eeg = [c for c in df.columns if c.startswith("eeg_")]
    suffix = "@{:g}".format(lag)
    fnirs = [c for c in df.columns if c.startswith("fnirs_") and c.endswith(suffix)]
    if modality == "eeg":
        return eeg
    if modality == "fnirs":
        return fnirs
    if modality == "both":
        return eeg + fnirs
    raise ValueError("unknown modality: " + modality)


def _make_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.1),
    )


# --------------------------------------------------------------------------------------
# Domain adaptation
# --------------------------------------------------------------------------------------
#
# The single largest known lever on this problem, and it was missing from the first
# baseline. Wu et al. (2021) report an unadapted SVM at 1.48x chance and optimal-transport
# alignment at 2.20x on fNIRS workload, so the gap between "no adaptation" and "some
# adaptation" is larger than most modelling choices.
#
# Both methods below use only the **unlabelled** features of the held-out subject. That
# is transductive but not leakage: no label of the test subject is touched. It is the
# standard setting in BCI, where an unlabelled recording from the new user is free and
# labels are not.

def _subject_zscore(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Centre and scale each subject's features independently.

    Removes per-person offset and gain, which is the crudest and often most effective
    form of subject alignment. It should have been the default from the start.
    """
    out = X.astype(float, copy=True)
    for s in np.unique(subjects):
        m = subjects == s
        block = out[m]
        out[m] = (block - block.mean(0)) / (block.std(0) + 1e-8)
    return out


def _coral(source: np.ndarray, target: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """CORAL: recolour ``source`` so its covariance matches ``target``'s.

    Closed form, no training, no hyperparameters. Whiten the source, then apply the
    target's colouring transform.
    """
    def _sqrtm_inv(cov, inverse):
        w, v = np.linalg.eigh(cov)
        w = np.clip(w, eps, None)
        p = w ** (-0.5 if inverse else 0.5)
        return (v * p) @ v.T

    s = source - source.mean(0)
    t_mean = target.mean(0)
    cs = np.cov(s, rowvar=False) + eps * np.eye(s.shape[1])
    ct = np.cov(target - t_mean, rowvar=False) + eps * np.eye(s.shape[1])
    return s @ _sqrtm_inv(cs, True) @ _sqrtm_inv(ct, False) + t_mean


def align(train_X, train_subjects, test_X, method: str):
    """Apply a domain-adaptation scheme. Returns ``(train_X, test_X)``."""
    if method == "none":
        return train_X, test_X
    if method == "subject_zscore":
        return (_subject_zscore(train_X, train_subjects),
                _subject_zscore(test_X, np.zeros(len(test_X))))
    if method == "coral":
        # Recolour the training pool onto the held-out subject's covariance.
        return _coral(train_X, test_X), test_X
    if method == "zscore_coral":
        tr = _subject_zscore(train_X, train_subjects)
        te = _subject_zscore(test_X, np.zeros(len(test_X)))
        return _coral(tr, te), te
    raise ValueError("unknown alignment: " + method)


ALIGNMENTS = ("none", "subject_zscore", "coral", "zscore_coral")


def loso(df: pd.DataFrame, modality: str, *, lag: float = DEFAULT_FNIRS_LAG,
         alignment: str = "none", seed: int = 0) -> pd.DataFrame:
    """Leave-one-subject-out accuracy. One row per held-out subject."""
    cols = feature_columns(df, modality, lag)
    data = df.dropna(subset=cols)
    subjects = sorted(data["subject"].unique())
    rows = []

    for s in subjects:
        train = data[data["subject"] != s]
        test = data[data["subject"] == s]
        if train.empty or test.empty:
            continue
        Xtr, Xte = align(train[cols].to_numpy(), train["subject"].to_numpy(),
                         test[cols].to_numpy(), alignment)
        model = _make_model()
        model.fit(Xtr, train["load"].to_numpy())
        acc = float((model.predict(Xte) == test["load"].to_numpy()).mean())
        rows.append({
            "subject": s, "modality": modality, "k_calibration": 0, "lag": lag,
            "alignment": alignment,
            "accuracy": acc, "n_test": int(len(test)), "seed": seed,
        })
    return pd.DataFrame(rows)


def label_efficiency(
    df: pd.DataFrame,
    modality: str,
    k_values=(0, 1, 2, 3, 6, 9),
    n_seeds: int = 5,
    lag: float = DEFAULT_FNIRS_LAG,
    alignment: str = "none",
) -> pd.DataFrame:
    """Accuracy vs number of calibration **blocks** taken from the held-out subject.

    Calibration samples whole blocks, never windows: windows within a block overlap, so
    sampling windows would leak the test block into training and inflate every number
    on the curve.
    """
    cols = feature_columns(df, modality, lag)
    data = df.dropna(subset=cols)
    subjects = sorted(data["subject"].unique())
    rows = []

    for s in subjects:
        others = data[data["subject"] != s]
        held = data[data["subject"] == s]
        if others.empty or held.empty:
            continue
        blocks = sorted(held["block"].unique())

        for k in k_values:
            if k > 0 and k * len(CONDITIONS) > len(blocks):
                continue
            for seed in range(n_seeds if k > 0 else 1):
                rng = np.random.default_rng(seed)
                if k == 0:
                    cal_blocks: list[str] = []
                else:
                    # k blocks per condition, so calibration stays class-balanced.
                    cal_blocks = []
                    for cond in CONDITIONS:
                        pool = sorted(held[held["condition"] == cond]["block"].unique())
                        if len(pool) < k:
                            cal_blocks = []
                            break
                        cal_blocks.extend(rng.choice(pool, size=k, replace=False).tolist())
                    if not cal_blocks:
                        continue

                cal = held[held["block"].isin(cal_blocks)]
                test = held[~held["block"].isin(cal_blocks)]
                if test.empty:
                    continue

                train = pd.concat([others, cal], ignore_index=True) if k else others
                # Align on the held-out subject's own distribution. The calibration
                # blocks belong to that subject, so they are aligned with the test data,
                # not with the training pool.
                Xtr, Xte = align(train[cols].to_numpy(), train["subject"].to_numpy(),
                                 test[cols].to_numpy(), alignment)
                model = _make_model()
                model.fit(Xtr, train["load"].to_numpy())
                pred = model.predict(Xte)
                rows.append({
                    "subject": s, "modality": modality, "k_calibration": k, "lag": lag,
                    "alignment": alignment,
                    "accuracy": float((pred == test["load"].to_numpy()).mean()),
                    "n_test": int(len(test)), "seed": seed,
                })
    return pd.DataFrame(rows)


def paired_modality_test(res: pd.DataFrame, a: str, b: str, k: int = 0) -> dict:
    """Wilcoxon across subjects comparing two modalities at one calibration level."""
    from scipy import stats

    sub = res[res["k_calibration"] == k]
    wide = (sub.groupby(["subject", "modality"], observed=True)["accuracy"]
               .mean().unstack())
    if a not in wide.columns or b not in wide.columns:
        return {"contrast": a + " vs " + b, "n": 0}
    wide = wide[[a, b]].dropna()
    if len(wide) < 5:
        return {"contrast": a + " vs " + b, "n": int(len(wide))}

    x, y = wide[a].to_numpy(), wide[b].to_numpy()
    stat, p = stats.wilcoxon(x, y)
    d = y - x
    n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
    return {
        "contrast": a + " vs " + b,
        "k_calibration": k,
        "n": int(len(wide)),
        "mean_" + a: float(x.mean()),
        "mean_" + b: float(y.mean()),
        "mean_diff": float(d.mean()),
        "p": float(p),
        "rank_biserial": (n_pos - n_neg) / max(n_pos + n_neg, 1),
        "n_subjects_better": n_pos,
    }
