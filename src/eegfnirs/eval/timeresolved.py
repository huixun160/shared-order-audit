"""Time-resolved decoding: when does each modality become informative?

This replaces the lag sweep, which could not answer the question it was built for.

**Why the lag sweep failed.** Reading fNIRS at ``t + lag`` and sweeping ``lag`` inside a
42 s block confounds two different things: correcting a latency, and simply sampling a
larger accumulated response. Both predict "later is better", so the sweep rises
monotonically whatever the truth is, and its shape carries no information about latency.

**What this measures instead.** Slide a short window across the block and, at each
offset, ask how well load can be decoded from that window alone. The quantity of
interest is not the height of the curve but **when it leaves chance**:

* EEG should become discriminative almost immediately - electrophysiology tracks the
  task at millisecond scale.
* fNIRS cannot become discriminative until the haemodynamic response has risen, which
  the literature puts at roughly 5-7 s.

**The latency is the difference between those two onsets.** That difference is immune to
the accumulation confound, because accumulation affects how fast each curve climbs, not
where it starts. If the two onsets coincide, there is no measurable lag and the premise
of a lag-aware model is not supported on this paradigm. If fNIRS lags EEG by seconds,
it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eegfnirs.data.preprocess import BANDS, PreprocessedRun
from eegfnirs.eval.canonical import LOAD_OF, PREFRONTAL_IDX
from eegfnirs.utils.logging import get_logger

log = get_logger(__name__)

WINDOW_S = 4.0                       # short, so the estimate is about timing not extent
TAU_GRID = np.round(np.arange(-6.0, 46.0, 1.0), 1)   # window start, from block onset
CHANCE = 1.0 / 3.0


def block_timecourse_features(pr: PreprocessedRun, *, window_s: float = WINDOW_S,
                              taus=TAU_GRID) -> pd.DataFrame:
    """One row per (block, tau): EEG band powers and prefrontal haemoglobin."""
    if pr.task != "nback":
        return pd.DataFrame()

    picks_eeg = [i for i, t in enumerate(pr.eeg.get_channel_types()) if t == "eeg"]
    env = {}
    for name, (lo, hi) in BANDS.items():
        env[name] = pr.eeg.copy().pick(picks_eeg).filter(lo, hi, verbose="ERROR").get_data() ** 2
    eeg_times = pr.eeg.times

    hb_types = pr.hb.get_channel_types()
    picks_hbo = [i for i, t in enumerate(hb_types) if t == "hbo"]
    picks_hbr = [i for i, t in enumerate(hb_types) if t == "hbr"]
    sel_hbo = [picks_hbo[i] for i in PREFRONTAL_IDX if i < len(picks_hbo)]
    sel_hbr = [picks_hbr[i] for i in PREFRONTAL_IDX if i < len(picks_hbr)]
    hb_data = pr.hb.get_data()
    hb_times = pr.hb.times

    rows = []
    for block_i, (onset, condition) in enumerate(zip(pr.block_onsets_s, pr.block_conditions)):
        # fNIRS baseline from the pre-block rest, so a block response is a change and
        # not an absolute level -- absolute levels drift and would leak subject identity.
        nb0, nb1 = onset + pr.hb_time_shift - 5.0, onset + pr.hb_time_shift
        bmask = (hb_times >= nb0) & (hb_times < nb1)
        baseline = (hb_data[:, bmask].mean(axis=1) if bmask.sum() >= 2
                    else np.zeros(hb_data.shape[0]))

        for tau in taus:
            t0, t1 = onset + tau, onset + tau + window_s
            em = (eeg_times >= t0) & (eeg_times < t1)
            n0 = t0 + pr.hb_time_shift
            n1 = t1 + pr.hb_time_shift
            hm = (hb_times >= n0) & (hb_times < n1)
            if em.sum() < 10 or hm.sum() < 2:
                continue

            feat = {"subject": pr.subject, "run": pr.run,
                    "block": pr.run + "_b{}".format(block_i),
                    "condition": condition, "load": LOAD_OF[condition],
                    "tau": float(tau)}
            for band, power in env.items():
                for ch_i, v in enumerate(np.log10(power[:, em].mean(axis=1) + 1e-30)):
                    feat["eeg_{}_{}".format(band, ch_i)] = float(v)
            idx = np.nonzero(hm)[0]
            for tag, sel in (("hbo", sel_hbo), ("hbr", sel_hbr)):
                vals = (hb_data[np.ix_(sel, idx)].mean(axis=1) - baseline[sel]) * 1e6
                for ch_i, v in enumerate(vals):
                    feat["fnirs_{}_{}".format(tag, ch_i)] = float(v)
            rows.append(feat)

    return pd.DataFrame(rows)


def _model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))


def decode_over_time(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """LOSO accuracy at each tau. One row per (tau, held-out subject)."""
    prefix = "eeg_" if modality == "eeg" else "fnirs_"
    cols = [c for c in df.columns if c.startswith(prefix)]
    subjects = sorted(df["subject"].unique())
    rows = []

    for tau, chunk in df.groupby("tau", observed=True):
        chunk = chunk.dropna(subset=cols)
        for s in subjects:
            train = chunk[chunk["subject"] != s]
            test = chunk[chunk["subject"] == s]
            if len(train) < 30 or test.empty:
                continue
            m = _model()
            m.fit(train[cols].to_numpy(), train["load"].to_numpy())
            acc = float((m.predict(test[cols].to_numpy()) == test["load"].to_numpy()).mean())
            rows.append({"tau": float(tau), "subject": s,
                         "modality": modality, "accuracy": acc})
    return pd.DataFrame(rows)


def leakage_test(res: pd.DataFrame, window_s: float = WINDOW_S) -> dict:
    """Decoding accuracy from windows that end at or before block onset.

    Any accuracy here is leakage, because no information about a block's condition can
    legitimately exist before it starts. In this dataset the leak has a known cause: the
    block order is **identical for all 26 subjects**, so within a run the previous
    block's load determines the current one, and slow drift that encodes position in the
    run therefore encodes load.

    Note ``tau`` is the window *start*: a window is entirely pre-onset only when
    ``tau <= -window_s``. Ignoring that was what first made this look catastrophic.
    """
    from scipy import stats

    pre = res[res["tau"] <= -window_s]
    if pre.empty:
        return {"n": 0}
    per_subject = pre.groupby("subject", observed=True)["accuracy"].mean()
    if len(per_subject) < 5:
        return {"n": int(len(per_subject))}
    _, p = stats.wilcoxon(per_subject - CHANCE, alternative="greater")
    return {
        "modality": res["modality"].iloc[0],
        "n": int(len(per_subject)),
        "mean_accuracy": float(per_subject.mean()),
        "excess_over_chance": float(per_subject.mean() - CHANCE),
        "p": float(p),
        "n_above_chance": int((per_subject > CHANCE).sum()),
    }


def onset_latency(res: pd.DataFrame, *, alpha: float = 0.05, n_consecutive: int = 3,
                  window_s: float = WINDOW_S) -> dict:
    """When each modality becomes informative, and how fast it rises.

    Time is reported as the window **centre**, not its start. A 4 s window beginning at
    ``tau = -2`` already contains 2 s of the block, so start-referenced onsets are
    biased earlier by half the window and will happily report an onset before the block
    exists.

    ``half_rise_centre`` - the first centre at which accuracy reaches half of that
    modality's own peak excess over chance - is the more robust summary. It does not
    depend on a significance threshold, and comparing it between modalities isolates
    *timing* from *magnitude*, which is the whole point: response accumulation changes
    how high a curve climbs, not when it starts climbing.
    """
    from scipy import stats

    taus = sorted(res["tau"].unique())
    centres = [t + window_s / 2.0 for t in taus]
    pvals, means = [], []
    for tau in taus:
        acc = res[res["tau"] == tau]["accuracy"].to_numpy()
        means.append(float(acc.mean()))
        if len(acc) < 5:
            pvals.append(np.nan)
            continue
        try:
            _, p = stats.wilcoxon(acc - CHANCE, alternative="greater")
        except ValueError:
            p = np.nan
        pvals.append(float(p))

    # Onsets are only meaningful for windows that lie inside the block.
    sig = [(p == p) and p < alpha and c >= window_s / 2.0
           for p, c in zip(pvals, centres)]
    onset = None
    for i in range(len(sig) - n_consecutive + 1):
        if all(sig[i:i + n_consecutive]):
            onset = centres[i]
            break

    in_block = [i for i, c in enumerate(centres) if 0.0 <= c <= 42.0]
    half_rise = None
    peak_centre = peak_acc = None
    if in_block:
        vals = np.array([means[i] for i in in_block])
        peak_acc = float(vals.max())
        peak_centre = float(centres[in_block[int(np.argmax(vals))]])
        threshold = CHANCE + 0.5 * (peak_acc - CHANCE)
        for i in in_block:
            if means[i] >= threshold:
                half_rise = float(centres[i])
                break

    return {
        "modality": res["modality"].iloc[0] if len(res) else None,
        "onset_centre": onset,
        "half_rise_centre": half_rise,
        "peak_centre": peak_centre,
        "peak_accuracy": peak_acc,
        "centres": centres,
        "taus": taus,
        "mean_accuracy": means,
        "p_values": pvals,
    }
