"""Synthetic block-design fNIRS whose temporal components are known by construction.

The real datasets cannot separate the candidate sources of order leakage, because in a
single shared sequence they are aliased: block position, the preceding block's load,
the length of the rest that follows it, and slow drift all move together. Here every
source is a separate, switchable term, so the null-window probe and the design-variable
decomposition can be checked against ground truth.

Signal of subject s, channel c (HbO-like, arbitrary units)::

    x = a_s * w_sc * sum_b beta(load_b) * (HRF * boxcar_b)        task response (its tail
                                                                 is real carry-over)
      + kappa * w_sc * load_prev * exp(-(t - end_prev) / tau)    extra recovery term
      + (delta_s * w'_sc) * t / T                                slow drift
      + phi * w_sc * position                                    position / fatigue step
      + offsets + physiology (cardiac, respiration, Mayer) + AR(1) noise

``delta_s ~ N(mu_drift, sd_drift)``: only the group mean ``mu_drift`` is shared across
subjects, so only it can leak through leave-one-subject-out evaluation.

Designs (how condition order is assigned to subjects):

* ``shared``        one sequence for everyone (Tufts' 16 blocks)
* ``shared_rest``   as ``shared``, and rest after a block lasts 20/20/30/40 s by load
* ``latin``         rows of a 4x4 Latin square rotated across subjects (counterbalanced)
* ``random``        an independent random permutation per subject
* ``random_rule``   per-subject random, but with one rule for everyone (no repeats)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter
from scipy.stats import gamma

TUFTS_ORDER = (0, 1, 2, 3, 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2)
LATIN = ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2))


@dataclass
class SimConfig:
    n_subjects: int = 40
    n_channels: int = 8
    fs: float = 5.0
    task_s: float = 80.0
    instr_s: float = 15.0
    rest_s: float = 25.0                  # rest after every block unless design couples it
    rest_by_load: tuple = (20.0, 20.0, 30.0, 40.0)
    lead_s: float = 30.0                  # recording before the first instruction
    design: str = "shared"
    # task response
    beta: float = 1.0                     # response per unit load (load 0..3)
    beta0: float = 0.5                    # response common to all loads (task vs rest)
    gain_sd: float = 0.3                  # between-subject log-gain sd
    pattern_sd: float = 0.5               # subject deviation from the group channel pattern
    # extra temporal terms
    kappa: float = 0.0                    # extra post-block recovery, per unit load
    tau: float = 20.0
    mu_drift: float = 0.0                 # group-mean drift over the session
    sd_drift: float = 1.0                 # between-subject drift sd
    phi: float = 0.0                      # step per block position
    # noise
    noise_sd: float = 1.0
    ar: float = 0.95
    physio: float = 0.5
    offset_sd: float = 5.0
    seed: int = 0
    extra: dict = field(default_factory=dict)


def hrf(fs: float, length_s: float = 32.0) -> np.ndarray:
    t = np.arange(0, length_s, 1 / fs)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / h.sum()


SHIN_RUNS = ((3, 2, 0, 2, 0, 3, 0, 3, 2), (2, 0, 3, 0, 3, 2, 3, 2, 0), (0, 3, 2, 3, 2, 0, 2, 0, 3))


def shin_config(**kw) -> SimConfig:
    """Shin 2018 n-back timing: 2 s instruction, 40 s task, 20 s rest, loads 0/2/3,
    three runs of nine blocks (orders differ between runs, identical across subjects)."""
    base = dict(design="shin", task_s=40.0, instr_s=2.0, rest_s=20.0, lead_s=30.0)
    base.update(kw)
    return SimConfig(**base)


def subject_order(cfg: SimConfig, s: int, rng: np.random.Generator) -> list[int]:
    if cfg.design == "shin":
        return [x for run in SHIN_RUNS for x in run]
    if cfg.design == "shin_random":
        return [x for run in SHIN_RUNS for x in rng.permutation(list(run))]
    if cfg.design in ("shared", "shared_rest"):
        return list(TUFTS_ORDER)
    if cfg.design == "latin":
        r = s % 4
        return [x for row in LATIN[r:] + LATIN[:r] for x in row]
    if cfg.design == "random":
        return list(rng.permutation(list(TUFTS_ORDER)))
    if cfg.design == "random_rule":
        while True:
            o = list(rng.permutation(list(TUFTS_ORDER)))
            if all(a != b for a, b in zip(o, o[1:])):
                return o
    raise ValueError(cfg.design)


@dataclass
class Recording:
    subject: int
    t: np.ndarray
    x: np.ndarray                 # (time, channel)
    blocks: list[dict]            # n, position, onset, offset, instr_start, prev_n, prev_rest


def simulate(cfg: SimConfig) -> list[Recording]:
    rng = np.random.default_rng(cfg.seed)
    h = hrf(cfg.fs)
    group_w = rng.normal(1.0, 0.3, cfg.n_channels)
    group_wd = rng.normal(1.0, 0.3, cfg.n_channels)
    recs = []
    for s in range(cfg.n_subjects):
        order = subject_order(cfg, s, rng)
        run_len = 9 if cfg.design.startswith("shin") else len(order)
        blocks, t0 = [], cfg.lead_s
        prev_n, prev_rest = -1, np.nan
        for pos, n in enumerate(order, start=1):
            if pos > 1 and (pos - 1) % run_len == 0:      # run break: long gap, no carry-over
                t0 += 120.0
                prev_n, prev_rest = -1, np.nan
            onset = t0 + cfg.instr_s
            offset = onset + cfg.task_s
            rest = cfg.rest_by_load[n] if cfg.design == "shared_rest" else cfg.rest_s
            blocks.append(dict(n=n, position=pos, run=(pos - 1) // run_len,
                               pos_in_run=(pos - 1) % run_len + 1, instr_start=t0,
                               onset=onset, offset=offset, prev_n=prev_n, prev_rest=prev_rest))
            prev_n, prev_rest = n, rest
            t0 = offset + rest
        T = t0 + 10.0
        t = np.arange(0, T, 1 / cfg.fs)
        nt = len(t)

        w = group_w + rng.normal(0, cfg.pattern_sd, cfg.n_channels)
        wd = group_wd + rng.normal(0, cfg.pattern_sd, cfg.n_channels)
        gain = np.exp(rng.normal(0, cfg.gain_sd))

        drive = np.zeros(nt)
        recov = np.zeros(nt)
        step = np.zeros(nt)
        for b in blocks:
            m = (t >= b["onset"]) & (t < b["offset"])
            drive[m] += cfg.beta0 + cfg.beta * b["n"]
            after = t >= b["offset"]
            recov[after] += b["n"] * np.exp(-(t[after] - b["offset"]) / cfg.tau)
            step[t >= b["instr_start"]] = b["pos_in_run"]
        task = np.convolve(drive, h)[:nt]

        delta = rng.normal(cfg.mu_drift, cfg.sd_drift)
        x = (np.outer(gain * task + cfg.kappa * recov + cfg.phi * step, w)
             + np.outer(delta * t / T, wd))
        x += rng.normal(0, cfg.offset_sd, cfg.n_channels)
        ph = rng.uniform(0, 2 * np.pi, (3, cfg.n_channels))
        for k, f in enumerate((1.1, 0.25, 0.1)):
            x += cfg.physio * np.sin(2 * np.pi * f * t[:, None] + ph[k])
        e = rng.normal(0, cfg.noise_sd * np.sqrt(1 - cfg.ar ** 2), (nt, cfg.n_channels))
        x += lfilter([1.0], [1.0, -cfg.ar], e, axis=0)
        recs.append(Recording(s, t, x, blocks))
    return recs
