"""Canonical working-memory effects.

Before any decoding claim is worth making, the pipeline has to reproduce what the
cognitive-neuroscience literature says should be there. If frontal-midline theta does
not rise with n-back load in our own preprocessing, nothing downstream can be trusted -
and a decoder can still hit high accuracy off an artefact, which is exactly how these
papers go wrong.

The three effects we test, and the direction each should take:

===========================  ===================================================
frontal-midline theta        **increases** with load (Gevins, Jensen & Tesche,
                             Onton et al.) - the classic WM-load signature
parietal alpha               **decreases** with load (desynchronisation)
prefrontal HbO               **increases** with load, HbR mirrors it downward
===========================  ===================================================

One caveat stated up front rather than discovered later: prefrontal activation is not
reliably monotonic in load. Once demand exceeds capacity, HbO can *fall* again, so a
3-back below 2-back is a documented inverted-U, not automatically a bug. We report the
full 0/2/3 profile instead of a single monotonicity test so the shape is visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eegfnirs.data import shin2018
from eegfnirs.data.preprocess import (
    BANDS,
    FRONTAL_MIDLINE,
    PARIETAL,
    PreprocessedRun,
    preprocess_run,
)
from eegfnirs.utils.logging import get_logger

log = get_logger(__name__)

# Windows relative to block onset. Blocks repeat every ~69.5 s: 42 s of task
# (20 trials, ISI 2.22 s) then ~27 s of rest.
EEG_TASK_WINDOW = (2.0, 42.0)      # skip the onset transient
FNIRS_BASELINE = (-5.0, 0.0)
FNIRS_RESPONSE = (5.0, 40.0)       # haemodynamic response lags the task by seconds

CONDITIONS = ("0-back", "2-back", "3-back")
LOAD_OF = {"0-back": 0, "2-back": 2, "3-back": 3}

# fNIRS channel labels in acquisition order, from the dataset description. Our reader
# orders channels by their S-D-Key column index, which is the same order.
NIRS_CHANNEL_LABELS = (
    "AF7", "AFF5", "AFp7", "AF5h", "AFp3", "AFF3h", "AF1", "AFFz", "AFpz",
    "AF2", "AFp4", "FCC3", "C3h", "C5h", "CCP3", "CPP3", "P3h", "P5h", "PPO3",
    "AFF4h", "AF6h", "AFF6", "AFp8", "AF8", "FCC4", "C6h", "C4h", "CCP4",
    "CPP4", "P6h", "P4h", "PPO4", "PPOz", "PO1", "PO2", "POOz",
)

#: Indices of the 16 (pre)frontal channels - every label beginning with AF.
#: This is an assumption about channel ordering; see docs/DATA.md.
PREFRONTAL_IDX = tuple(i for i, lab in enumerate(NIRS_CHANNEL_LABELS) if lab.startswith("AF"))


def _picks_present(raw, names) -> list[str]:
    return [n for n in names if n in raw.ch_names]


def _mean_band_power(raw, band, picks, tmin, tmax) -> float:
    """Log mean PSD over ``picks`` in ``band``. Log because power is heavily skewed."""
    lo = max(tmin, float(raw.times[0]))
    hi = min(tmax, float(raw.times[-1]))
    if hi - lo < 1.0 or not picks:
        return float("nan")
    seg = raw.copy().crop(tmin=lo, tmax=hi)
    spec = seg.compute_psd(fmin=band[0], fmax=band[1], picks=picks, verbose="ERROR")
    psd = spec.get_data()                      # (n_channels, n_freqs)
    return float(np.log10(psd.mean()))


def _hb_block_response(hb, idx, tmin_base, tmax_base, tmin_resp, tmax_resp, chroma) -> float:
    """Baseline-corrected mean concentration change for one block, in micromolar."""
    picks = [i for i, t in enumerate(hb.get_channel_types()) if t == chroma]
    if not picks:
        return float("nan")
    # Channel order is [all hbo..., all hbr...] only if MNE kept pairs together; it does
    # not, so select by position within this chromophore's own ordering.
    sel = [picks[i] for i in idx if i < len(picks)]
    data = hb.get_data(picks=sel)
    times = hb.times

    base_m = (times >= tmin_base) & (times < tmax_base)
    resp_m = (times >= tmin_resp) & (times < tmax_resp)
    if base_m.sum() < 2 or resp_m.sum() < 2:
        return float("nan")
    value = data[:, resp_m].mean() - data[:, base_m].mean()
    return float(value * 1e6)                  # molar -> micromolar


def nback_block_features(pr: PreprocessedRun) -> list[dict]:
    """One row per n-back block: EEG band powers and prefrontal haemodynamics."""
    if pr.task != "nback":
        return []

    fm = _picks_present(pr.eeg, FRONTAL_MIDLINE)
    par = _picks_present(pr.eeg, PARIETAL)
    rows: list[dict] = []

    for onset, condition in zip(pr.block_onsets_s, pr.block_conditions):
        e0, e1 = onset + EEG_TASK_WINDOW[0], onset + EEG_TASK_WINDOW[1]
        # fNIRS lives on its own clock; shift by the offset we measured from the shared
        # triggers rather than assuming the recordings start together.
        n_on = onset + pr.hb_time_shift

        row = {
            "subject": pr.subject,
            "run": pr.run,
            "condition": condition,
            "load": LOAD_OF[condition],
            "theta_fm": _mean_band_power(pr.eeg, BANDS["theta"], fm, e0, e1),
            "alpha_par": _mean_band_power(pr.eeg, BANDS["alpha"], par, e0, e1),
            "beta_fm": _mean_band_power(pr.eeg, BANDS["beta"], fm, e0, e1),
            "hbo_pfc": _hb_block_response(
                pr.hb, PREFRONTAL_IDX,
                n_on + FNIRS_BASELINE[0], n_on + FNIRS_BASELINE[1],
                n_on + FNIRS_RESPONSE[0], n_on + FNIRS_RESPONSE[1], "hbo"),
            "hbr_pfc": _hb_block_response(
                pr.hb, PREFRONTAL_IDX,
                n_on + FNIRS_BASELINE[0], n_on + FNIRS_BASELINE[1],
                n_on + FNIRS_RESPONSE[0], n_on + FNIRS_RESPONSE[1], "hbr"),
            "sync_residual_ms": pr.alignment.max_abs_residual * 1000.0,
            "device_offset_s": pr.alignment.offset,
        }
        rows.append(row)
    return rows


def hb_block_timecourse(
    pr: PreprocessedRun,
    condition: str,
    tmin: float = -5.0,
    tmax: float = 60.0,
    n_points: int = 260,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block-averaged prefrontal HbO/HbR time course for one condition.

    Returns ``(times, hbo, hbr)`` on a common grid relative to block onset, in
    micromolar, baseline-corrected on ``FNIRS_BASELINE``. This is the figure that makes
    a haemodynamic response look like a haemodynamic response - or exposes that it does
    not.
    """
    grid = np.linspace(tmin, tmax, n_points)
    types = pr.hb.get_channel_types()
    picks_hbo = [i for i, t in enumerate(types) if t == "hbo"]
    picks_hbr = [i for i, t in enumerate(types) if t == "hbr"]
    sel_hbo = [picks_hbo[i] for i in PREFRONTAL_IDX if i < len(picks_hbo)]
    sel_hbr = [picks_hbr[i] for i in PREFRONTAL_IDX if i < len(picks_hbr)]

    times = pr.hb.times
    hbo_all, hbr_all = [], []

    for onset, cond in zip(pr.block_onsets_s, pr.block_conditions):
        if cond != condition:
            continue
        t0 = onset + pr.hb_time_shift
        mask = (times >= t0 + tmin) & (times <= t0 + tmax)
        if mask.sum() < 10:
            continue
        rel = times[mask] - t0
        base = (rel >= FNIRS_BASELINE[0]) & (rel < FNIRS_BASELINE[1])
        if base.sum() < 2:
            continue
        for sel, store in ((sel_hbo, hbo_all), (sel_hbr, hbr_all)):
            trace = pr.hb.get_data(picks=sel)[:, mask].mean(axis=0)
            trace = (trace - trace[base].mean()) * 1e6
            store.append(np.interp(grid, rel, trace))

    if not hbo_all:
        nan = np.full_like(grid, np.nan)
        return grid, nan, nan
    return grid, np.mean(hbo_all, axis=0), np.mean(hbr_all, axis=0)


def subject_timecourses(subject_dir, runs=shin2018.NBACK.runs) -> pd.DataFrame:
    """Block-averaged haemodynamic time courses per condition for one subject."""
    from eegfnirs.data.preprocess import preprocess_run as _pr

    acc: dict[str, list] = {}
    grid = None
    for run in runs:
        pr = _pr(subject_dir, run)
        for cond in CONDITIONS:
            g, hbo, hbr = hb_block_timecourse(pr, cond)
            grid = g
            acc.setdefault(cond, []).append((hbo, hbr))

    rows = []
    for cond, pairs in acc.items():
        hbo = np.nanmean([p[0] for p in pairs], axis=0)
        hbr = np.nanmean([p[1] for p in pairs], axis=0)
        for t, o, r in zip(grid, hbo, hbr):
            rows.append({"subject": subject_dir.name, "condition": cond,
                         "time": float(t), "hbo": float(o), "hbr": float(r)})
    return pd.DataFrame(rows)


def subject_features(subject_dir, runs=shin2018.NBACK.runs) -> pd.DataFrame:
    """Feature rows for every n-back block of one subject."""
    rows: list[dict] = []
    for run in runs:
        pr = preprocess_run(subject_dir, run)
        rows.extend(nback_block_features(pr))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

def subject_condition_means(df: pd.DataFrame, measures) -> pd.DataFrame:
    """Collapse blocks to one value per subject x condition - the unit of analysis.

    Blocks within a subject are not independent, so every test below treats the
    **subject** as the sampling unit. This is the difference between a real effect and
    a p-value manufactured from 234 correlated blocks.
    """
    return (df.groupby(["subject", "condition"], observed=True)[list(measures)]
              .mean()
              .reset_index())


def paired_tests(sc: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Wilcoxon signed-rank across subjects for each condition pair, plus effect size."""
    from scipy import stats

    wide = sc.pivot(index="subject", columns="condition", values=measure)
    wide = wide.dropna()
    out = []
    pairs = [("0-back", "2-back"), ("0-back", "3-back"), ("2-back", "3-back")]
    for a, b in pairs:
        if a not in wide.columns or b not in wide.columns or len(wide) < 5:
            continue
        x, y = wide[a].to_numpy(), wide[b].to_numpy()
        d = y - x
        try:
            stat, p = stats.wilcoxon(x, y)
        except ValueError:
            stat, p = float("nan"), float("nan")
        # Matched-pairs rank-biserial correlation: interpretable, distribution-free.
        n_pos = int((d > 0).sum())
        n_neg = int((d < 0).sum())
        rbc = (n_pos - n_neg) / max(n_pos + n_neg, 1)
        out.append({
            "measure": measure,
            "contrast": a + " vs " + b,
            "n": int(len(wide)),
            "mean_diff": float(np.mean(d)),
            "median_diff": float(np.median(d)),
            "wilcoxon_W": float(stat),
            "p": float(p),
            "rank_biserial": float(rbc),
            "n_subjects_increasing": n_pos,
        })
    return pd.DataFrame(out)


def fdr_correct(df: pd.DataFrame, p_column: str = "p", alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg FDR across a family of tests.

    Five measures x three load contrasts is fifteen tests; reporting raw p-values across
    that family overstates the evidence. BH rather than Bonferroni because the tests are
    positively dependent (they share subjects and conditions), which is the situation BH
    is built for.
    """
    out = df.copy()
    p = out[p_column].to_numpy(dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)

    idx = np.nonzero(ok)[0]
    if idx.size:
        order = idx[np.argsort(p[idx])]
        m = order.size
        ranked = p[order] * m / np.arange(1, m + 1)
        # enforce monotonicity from the largest p downwards
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        q[order] = np.minimum(ranked, 1.0)

    out["q_fdr"] = q
    out["significant_fdr"] = q < alpha
    return out


def load_trend_test(sc: pd.DataFrame, measure: str) -> dict:
    """Friedman test across the three loads, plus a per-subject Spearman trend.

    Friedman answers 'do the three loads differ at all'; the Spearman summary answers
    'is it monotonic in load', which is the part that can legitimately fail for
    prefrontal haemodynamics (inverted-U).
    """
    from scipy import stats

    wide = sc.pivot(index="subject", columns="condition", values=measure).dropna()
    cols = [c for c in CONDITIONS if c in wide.columns]
    if len(cols) < 3 or len(wide) < 5:
        return {"measure": measure, "n": int(len(wide)), "friedman_p": float("nan")}

    stat, p = stats.friedmanchisquare(*[wide[c].to_numpy() for c in cols])
    loads = np.array([LOAD_OF[c] for c in cols], dtype=float)
    rhos = [stats.spearmanr(loads, wide.loc[s, cols].to_numpy().astype(float)).statistic
            for s in wide.index]
    rhos = np.array([r for r in rhos if np.isfinite(r)])

    return {
        "measure": measure,
        "n": int(len(wide)),
        "friedman_chi2": float(stat),
        "friedman_p": float(p),
        "median_spearman_rho": float(np.median(rhos)) if rhos.size else float("nan"),
        "frac_subjects_monotonic_up": float((rhos > 0).mean()) if rhos.size else float("nan"),
    }
