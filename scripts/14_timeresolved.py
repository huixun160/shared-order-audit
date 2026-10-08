"""Time-resolved decoding: measure the EEG->fNIRS latency without the accumulation confound.

This is the decisive test for whether a lag-aware model is worth building on this
paradigm. See src/eegfnirs/eval/timeresolved.py for why the earlier lag sweep could not
answer it.

    python scripts/14_timeresolved.py
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eegfnirs.data import shin2018  # noqa: E402
from eegfnirs.data.preprocess import preprocess_run  # noqa: E402
from eegfnirs.eval import timeresolved as tr  # noqa: E402
from eegfnirs.utils import get_logger, project_paths, seed_everything, setup_logging  # noqa: E402

log = get_logger("timeresolved")
COLOR = {"eeg": "#4C72B0", "fnirs": "#DD8452"}


def subject_features(subject_dir: Path) -> pd.DataFrame:
    frames = [tr.block_timecourse_features(preprocess_run(subject_dir, r))
              for r in shin2018.NBACK.runs]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_figure(curves: dict, onsets: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for mod in ("eeg", "fnirs"):
        info = onsets[mod]
        res = curves[mod]
        g = res.groupby("tau", observed=True)["accuracy"]
        m, se = g.mean(), g.sem()
        centres = m.index.to_numpy() + tr.WINDOW_S / 2.0
        hr = info["half_rise_centre"]
        ax.plot(centres, m.to_numpy(), color=COLOR[mod], lw=2,
                label="{} (half-rise {})".format(
                    mod.upper(), "{:g} s".format(hr) if hr is not None else "n/a"))
        ax.fill_between(centres, (m - se).to_numpy(), (m + se).to_numpy(),
                        color=COLOR[mod], alpha=0.18, lw=0)
        if hr is not None:
            ax.axvline(hr, color=COLOR[mod], ls=":", lw=1.6)

    ax.axhline(tr.CHANCE, color="0.4", ls="--", lw=1, label="chance")
    ax.axvspan(0, 42, color="0.93", zorder=0)
    ax.axvline(0, color="0.5", lw=0.9)
    ax.set_xlabel("window centre relative to block onset (s)")
    ax.set_ylabel("LOSO decoding accuracy")
    ax.set_title("When does each modality become informative about load?", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = project_paths().ensure()
    setup_logging(logfile=paths.runs / "timeresolved.log")
    seed_everything(0)

    root = paths.interim / "shin2018"
    available = sorted(int(d.name.replace("VP", "")) for d in root.iterdir()
                       if d.is_dir() and (d / "eeg").is_dir())
    if args.subjects:
        want = set()
        for part in args.subjects.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                want.update(range(int(lo), int(hi) + 1))
            elif part.strip():
                want.add(int(part))
        available = [s for s in available if s in want]

    cache_dir = paths.processed / "timeresolved"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames, failures = [], []
    for s in available:
        vp = shin2018.subject_id(s)
        cache = cache_dir / (vp + ".parquet")
        if cache.exists() and not args.force:
            frames.append(pd.read_parquet(cache))
            continue
        try:
            log.info("%s: time-resolved features", vp)
            df = subject_features(root / vp)
            if df.empty:
                raise RuntimeError("no features")
            df.to_parquet(cache, index=False)
            frames.append(df)
        except Exception as exc:
            log.error("%s FAILED: %s", vp, exc)
            log.debug("%s", traceback.format_exc())
            failures.append(vp + ": " + str(exc))

    if not frames:
        return 1
    data = pd.concat(frames, ignore_index=True)

    out_dir = paths.results / "timeresolved"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 74)
    print("TIME-RESOLVED DECODING   {} subjects, {} blocks, {:g}s windows".format(
        data["subject"].nunique(), len(data.groupby(["subject","block"]).size()), tr.WINDOW_S))
    print("chance = {:.3f}".format(tr.CHANCE))
    print("=" * 74)

    curves, onsets = {}, {}
    for mod in ("eeg", "fnirs"):
        log.info("decoding over time: %s", mod)
        res = tr.decode_over_time(data, mod)
        res.to_csv(out_dir / ("curve_" + mod + ".csv"), index=False)
        curves[mod] = res
        onsets[mod] = tr.onset_latency(res)

    print("\n-- leakage probe: windows ending at or before block onset --")
    print("   Any accuracy here is leakage. The block order is identical for all 26")
    print("   subjects, so within a run the previous block's load determines this one.")
    leaks = []
    for mod in ("eeg", "fnirs"):
        lk = tr.leakage_test(curves[mod])
        leaks.append(lk)
        if lk.get("n", 0) >= 5:
            print("   {:<6} {:.4f} (chance {:.4f}, excess {:+.4f})  p={:.4g}  "
                  "above in {}/{}".format(
                      mod, lk["mean_accuracy"], tr.CHANCE, lk["excess_over_chance"],
                      lk["p"], lk["n_above_chance"], lk["n"]))
    pd.DataFrame(leaks).to_csv(out_dir / "leakage.csv", index=False)

    print("\n{:<8} {:>12} {:>14} {:>12} {:>14}".format(
        "modality", "onset (s)", "half-rise (s)", "peak (s)", "peak accuracy"))
    for mod in ("eeg", "fnirs"):
        o = onsets[mod]

        def fmt(v):
            return "{:g}".format(v) if v is not None else "n/a"

        print("{:<8} {:>12} {:>14} {:>12} {:>14.4f}".format(
            mod, fmt(o["onset_centre"]), fmt(o["half_rise_centre"]),
            fmt(o["peak_centre"]), o["peak_accuracy"]))

    summary = pd.DataFrame([{k: v for k, v in o.items()
                             if k not in ("taus", "centres", "mean_accuracy", "p_values")}
                            for o in onsets.values()])
    summary.to_csv(out_dir / "onsets.csv", index=False)

    detail = pd.DataFrame({
        "tau": onsets["eeg"]["taus"],
        "centre": onsets["eeg"]["centres"],
        "eeg_mean": onsets["eeg"]["mean_accuracy"],
        "eeg_p": onsets["eeg"]["p_values"],
        "fnirs_mean": onsets["fnirs"]["mean_accuracy"],
        "fnirs_p": onsets["fnirs"]["p_values"],
    })
    detail.to_csv(out_dir / "curves.csv", index=False)

    eo, fo = onsets["eeg"]["half_rise_centre"], onsets["fnirs"]["half_rise_centre"]
    print("\n-- interpretation --")
    if eo is None or fo is None:
        print("  one modality never became reliably informative; the latency cannot be")
        print("  estimated from these data.")
    else:
        lag = fo - eo
        print("  EEG reaches half its peak discriminability at {:g} s, fNIRS at {:g} s.".format(eo, fo))
        print("  Timing difference: {:+g} s.".format(lag))
        if lag >= 3.0:
            print("  The two modalities carry decodable information on clearly different")
            print("  timescales: EEG is at half its peak immediately, fNIRS takes seconds.")
            print("  Reading fNIRS at the EEG's optimal moment is therefore leaving")
            print("  information on the table, which is the premise a multimodal model")
            print("  needs.")
            print()
            print("  What this is NOT: a clean estimate of haemodynamic latency. In a")
            print("  block design the fNIRS rise reflects the response accumulating over")
            print("  the whole block as well as its onset delay, so a half-rise longer")
            print("  than the canonical 5-7 s peak is expected and the two causes are not")
            print("  separable here. Separating them needs an event-related paradigm.")
        elif lag <= 1.0:
            print("  The two modalities become informative at essentially the same time.")
            print("  There is no measurable latency to model here.")
        else:
            print("  The latency is small and its interpretation is ambiguous at this")
            print("  temporal resolution.")

    make_figure(curves, onsets, paths.root / "paper" / "figures" / "fig6_timeresolved.png")

    if failures:
        print("\nFAILED ({}):".format(len(failures)))
        for f in failures:
            print("  " + f)
    print("\nwrote -> " + str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
