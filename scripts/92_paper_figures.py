"""Manuscript figures (vector PDF + PNG preview) for the shared-order audit.

Inputs are the result tables written by scripts 20-32 and 91. Light-mode print figures:
categorical slots in fixed order (blue, orange, aqua), an ordinal blue ramp for n-back
load, chance as a grey dashed line, text in neutral ink.

    python scripts/92_paper_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eegfnirs.utils import project_paths  # noqa: E402

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"           # categorical slots 1-3
LOAD = ["#b7d3f2", "#6fa6e6", "#2a78d6", "#103f7a"]     # ordinal ramp, 0..3-back
INK, INK2, GRID, CHANCE = "#0b0b0b", "#52514e", "#e6e5e0", "#8a8984"
MM = 1 / 25.4
W1, W2 = 90 * MM, 190 * MM
ORDER = (0, 1, 2, 3, 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})


def panel(ax, letter):
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def ci(x, n=4000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975)


def strip(ax, pos, values, color, jitter=0.12, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    ax.scatter(pos + rng.uniform(-jitter, jitter, len(v)), v, s=6, color="#c9c8c2",
               lw=0, zorder=1)
    m, lo, hi = ci(v, seed=seed)
    ax.plot([pos, pos], [lo, hi], color=color, lw=2.0, zorder=3, solid_capstyle="round")
    ax.scatter([pos], [m], s=34, color=color, edgecolor="white", lw=1.0, zorder=4)


def chance(ax, y, label=True):
    ax.axhline(y, color=CHANCE, ls=(0, (3, 2)), lw=0.8, zorder=0)
    if label:
        ax.text(ax.get_xlim()[1], y, " chance", color=CHANCE, fontsize=6, va="center")


def grid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------- figure 1
def fig1(out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(W2, 78 * MM),
                             gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 1.25})
    # Shin: one run, 40 s task + 20 s rest, 2 s instruction
    ax = axes[0]
    t = 0
    for n in (3, 2, 0, 2, 0, 3, 0, 3, 2):
        ax.add_patch(plt.Rectangle((t + 2, 0), 40, 1, color=LOAD[n], lw=0))
        ax.text(t + 22, 0.5, f"{n}", ha="center", va="center", fontsize=6.5,
                color="white" if n >= 2 else INK)
        ax.add_patch(plt.Rectangle((t, 0), 2, 1, color=INK2, lw=0))
        t += 62
    ax.add_patch(plt.Rectangle((62 * 3 - 8, -0.35), 4, 0.25, color=S2, lw=0))
    ax.text(62 * 3 - 10, -0.22, "null window (ends before the 2 s instruction) ",
            fontsize=6, color=S2, ha="right", va="center")
    ax.set_title("Shin et al. 2018 n-back (run 1 of 3; identical for all 26 subjects)",
                 loc="left", fontsize=7)
    # Tufts: 16 blocks, task ~83 s, rest 20/20/30/40 s after 0/1/2/3-back
    ax = axes[1]
    t = 0
    rest = {0: 20, 1: 20, 2: 30, 3: 40}
    for n in ORDER:
        ax.add_patch(plt.Rectangle((t + 15, 0), 83, 1, color=LOAD[n], lw=0))
        ax.text(t + 15 + 41, 0.5, f"{n}", ha="center", va="center", fontsize=6.5,
                color="white" if n >= 2 else INK)
        ax.add_patch(plt.Rectangle((t + 98 + 20, 0), rest[n], 1, color="#efeee9", lw=0))
        t += 15 + 83 + 20 + rest[n]
    ax.set_title("Tufts fNIRS2MW (16 blocks, identical for all 68 subjects; grey = rest, 20/20/30/40 s after 0/1/2/3-back)", loc="left", fontsize=7)
    # Chen: N I N I, 32 s blocks, ~24 s breaks
    ax = axes[2]
    t = 30
    for c in (0, 1, 0, 1):
        ax.add_patch(plt.Rectangle((t, 0), 32, 1, color=S1 if c == 0 else S2, lw=0))
        ax.text(t + 16, 0.5, "N" if c == 0 else "I", ha="center", va="center",
                fontsize=6.5, color="white")
        t += 56
    ax.set_title("Chen et al. 2023 Stroop (N = neutral, I = incongruent; identical for all subjects)", loc="left", fontsize=7)
    for ax, xmax in zip(axes, (62 * 9, 2400, 280)):
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.4, 1.05)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("time (s)", labelpad=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LOAD[i]) for i in range(4)]
    axes[0].legend(handles, ["0-back", "1-back", "2-back", "3-back"], ncol=4, frameon=False,
                   loc="upper right", bbox_to_anchor=(1.0, 1.9), handlelength=1.0)
    fig.savefig(out / "fig1_designs.pdf", bbox_inches="tight")
    fig.savefig(out / "fig1_designs.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- figure 2
def fig2(res: Path, out: Path) -> None:
    shin = pd.read_csv(res / "paper" / "shin_probe_per_subject.csv")
    tufts = pd.read_csv(res / "paper" / "tufts_probe_per_subject.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 62 * MM),
                               gridspec_kw={"width_ratios": [5, 4], "wspace": 0.38})
    cases = [("zero", "fNIRS", "fNIRS\nzero-phase\nBP", S2),
             ("minimum", "fNIRS", "fNIRS\ncausal\nBP", S1),
             ("lp_only", "fNIRS", "fNIRS\ncausal\nLP only", S1),
             ("none", "fNIRS", "fNIRS\nno\nfilter", S1),
             ("minimum", "EEG", "EEG\ncausal\nBP", S3)]
    for i, (f, m, lab, col) in enumerate(cases):
        v = shin[(shin["filter"] == f) & (shin.modality == m)].accuracy
        strip(a, i, v, col, seed=i)
    a.set_xticks(range(len(cases)), [c[2] for c in cases])
    a.set_xlim(-0.6, len(cases) - 0.4)
    a.set_ylim(0, 0.85)
    chance(a, 1 / 3)
    a.set_ylabel("LOSO accuracy (per subject)")
    a.set_title("Shin 2018 n-back (n = 26)", loc="left")
    grid(a)
    panel(a, "A")

    cases = [("rest_end", "causal_long", "null\ncausal\nbaseline", S1),
             ("rest_end", "whole_file", "null\nrecording\nmean", S1),
             ("task", "causal_long", "task\ncausal\nbaseline", INK2),
             ("task", "whole_file", "task\nrecording\nmean", INK2)]
    for i, (w, bse, lab, col) in enumerate(cases):
        v = tufts[(tufts.window == w) & (tufts.baseline == bse)].accuracy
        strip(b, i, v, col, jitter=0.18, seed=10 + i)
    b.set_xticks(range(len(cases)), [c[2] for c in cases])
    b.set_xlim(-0.6, len(cases) - 0.4)
    b.set_ylim(-0.03, 1.03)
    chance(b, 1 / 3)
    b.set_title("Tufts fNIRS2MW, raw (n = 68)", loc="left")
    grid(b)
    panel(b, "B")
    fig.savefig(out / "fig2_null_window.pdf", bbox_inches="tight")
    fig.savefig(out / "fig2_null_window.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- figure 3
def fig3(res: Path, out: Path) -> None:
    inj = pd.read_csv(res / "tufts_g1" / "injection.csv")
    real = float(inj[inj.k == "real data"].accuracy.iloc[0])
    inj = inj[inj.k != "real data"].astype({"k": float})
    shin = pd.read_csv(res / "paper" / "shin_probe_per_subject.csv")
    wg = pd.read_csv(res / "paper" / "wg_control_per_subject.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 58 * MM), gridspec_kw={"wspace": 0.35})

    a.errorbar(inj.k, inj.accuracy, yerr=inj.sd_over_reps, color=S1, marker="o", ms=4,
               capsize=0, lw=1.4, elinewidth=1.0)
    a.axhline(real, color=S2, lw=1.0)
    a.text(2.0, real + 0.012, "real data (0.480)", color=S2, fontsize=6, ha="right")
    a.set_xlim(-0.1, 2.1)
    a.set_ylim(0.28, 0.72)
    chance(a, 1 / 3)
    a.set_xlabel("injected drift (x observed group-mean drift)")
    a.set_ylabel("null-window LOSO accuracy")
    a.set_title("Positive control: injected drift (Tufts)", loc="left")
    grid(a)
    panel(a, "A")

    nb = shin[(shin["filter"] == "minimum") & (shin.modality == "fNIRS")].accuracy - 1 / 3
    strip(b, 0, nb, S1, seed=1)
    strip(b, 1, wg.accuracy - 0.5, INK2, seed=2)
    b.scatter([1.32], [wg.rule_oracle.mean() - 0.5], marker="_", s=120, color=S2, lw=1.6)
    b.text(1.4, wg.rule_oracle.mean() - 0.5, "randomisation rule\nalone (no data)",
           color=S2, fontsize=6, va="center")
    b.set_xticks([0, 1], ["n-back\nshared order", "word generation\nper-subject order"])
    b.set_xlim(-0.6, 2.1)
    b.axhline(0, color=CHANCE, ls=(0, (3, 2)), lw=0.8)
    b.set_ylabel("accuracy minus chance (per subject)")
    b.set_title("Negative control (Shin, fNIRS, n = 26)", loc="left")
    grid(b)
    panel(b, "B")
    fig.savefig(out / "fig3_controls.pdf", bbox_inches="tight")
    fig.savefig(out / "fig3_controls.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- figure 4
def _p0_by_position(p0: pd.DataFrame) -> pd.DataFrame:
    return p0.groupby(["subject", "position", "n"]).p0.mean().reset_index()


def fig4(res: Path, out: Path) -> None:
    tr = pd.read_csv(res / "tufts_transfer" / "summary.csv")
    lr = _p0_by_position(pd.read_csv(res / "tufts_official" / "p0_all_blocks.csv"))
    dcn = _p0_by_position(pd.read_csv(res / "tufts_deepconvnet" / "p0_all_blocks.csv"))
    fig = plt.figure(figsize=(W2, 64 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.4, 3, 3], wspace=0.42)
    a = fig.add_subplot(gs[0])
    m = tr[(tr.variant == "raw") & (tr.baseline == "whole_file") & (tr.task == "0vs2")]
    mat = np.array([[m[(m.train == r) & (m.test == c)].accuracy.iloc[0] for c in ("task", "rest")]
                    for r in ("task", "rest")])
    a.imshow(mat, cmap="Blues", vmin=0.5, vmax=0.8)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{mat[i, j] * 100:.1f}%", ha="center", va="center", fontsize=7,
                   color="white" if mat[i, j] > 0.66 else INK)
    a.set_xticks([0, 1], ["task", "rest\n(pre-instruction)"])
    a.set_yticks([0, 1], ["task", "rest"])
    a.set_xlabel("tested on")
    a.set_ylabel("trained on")
    a.set_title("0 vs 2-back, raw\n(chance 50 %)", loc="left")
    a.spines[:].set_visible(False)
    panel(a, "A")

    for ax, d, title, letter in ((fig.add_subplot(gs[1]), lr, "Logistic regression", "B"),
                                 (fig.add_subplot(gs[2]), dcn, "DeepConvNet", "C")):
        for n in range(4):
            g = d[d.n == n].groupby("position").p0.agg(["mean", "sem"])
            ax.errorbar(g.index, g["mean"], yerr=g["sem"], color=LOAD[n], marker="o", ms=3.5,
                        lw=1.2, elinewidth=0.8, capsize=0, label=f"{n}-back")
        ax.axhline(0.5, color=CHANCE, ls=(0, (3, 2)), lw=0.8)
        ax.set_xticks([1, 4, 8, 12, 16])
        ax.set_xlabel("block position in session")
        ax.set_ylabel("P(0-back) from 0-vs-2 decoder")
        ax.set_ylim(0.25, 0.8)
        ax.set_title(f"{title} (n = 68)", loc="left")
        grid(ax)
        panel(ax, letter)
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.2)
    fig.savefig(out / "fig4_benchmark.pdf", bbox_inches="tight")
    fig.savefig(out / "fig4_benchmark.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- figure 5
def fig5(res: Path, out: Path) -> None:
    d = pd.read_csv(res / "chen_stroop" / "pairs_lp_causal.csv")
    fig, ax = plt.subplots(figsize=(W1, 62 * MM))
    for _, r in d.iterrows():
        ax.plot([0, 1], [r.fwd, r.rev], color="#c9c8c2", lw=0.7, zorder=1)
    for x, col, c in ((0, "fwd", S1), (1, "rev", S2)):
        m, lo, hi = ci(d[col])
        ax.plot([x, x], [lo, hi], color=c, lw=2.0, zorder=3)
        ax.scatter([x], [m], s=34, color=c, edgecolor="white", lw=1, zorder=4)
    ax.set_xticks([0, 1], ["forward pairs\n(N before I)", "reverse pair\n(I before N)"])
    ax.set_xlim(-0.4, 1.6)
    ax.set_ylim(0, 1.02)
    chance(ax, 0.5)
    ax.set_ylabel("condition-decoding accuracy (per subject)")
    ax.set_title("Chen 2023 Stroop, task windows (n = 18)", loc="left")
    grid(ax)
    fig.savefig(out / "fig5_stroop.pdf", bbox_inches="tight")
    fig.savefig(out / "fig5_stroop.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- figure 6
def fig6(res: Path, out: Path) -> None:
    bn = json.loads((res / "benchnirs_shift" / "loso_lda.json").read_text())
    post = pd.Series(bn["0.0"]["per_subject"])
    pre = pd.Series(bn["12.0"]["per_subject"])
    con = pd.concat([pd.read_csv(res / "con4m" / "eval_fixedmodel.csv"),
                     pd.read_csv(res / "con4m" / "eval_shuffledmodel.csv")])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 60 * MM), gridspec_kw={"wspace": 0.35})
    for s in post.index:
        a.plot([0, 1], [post[s], pre[s]], color="#c9c8c2", lw=0.7, zorder=1)
    for x, v, c in ((0, post, S1), (1, pre, S2)):
        m, lo, hi = ci(v)
        a.plot([x, x], [lo, hi], color=c, lw=2.0, zorder=3)
        a.scatter([x], [m], s=34, color=c, edgecolor="white", lw=1, zorder=4)
    a.set_xticks([0, 1], ["published window\n0 to 10 s", "same pipeline,\n-12 to -2 s"])
    a.set_xlim(-0.4, 1.6)
    a.set_ylim(0.1, 0.65)
    chance(a, 1 / 3)
    a.set_ylabel("LDA accuracy (LOSO, per subject)")
    a.set_title("BenchNIRS on Shin n-back (n = 26)", loc="left")
    grid(a)
    panel(a, "A")

    combos = [("fixedmodel", "fixed"), ("fixedmodel", "shuffled"),
              ("shuffledmodel", "fixed"), ("shuffledmodel", "shuffled")]
    for i, (mdl, tst) in enumerate(combos):
        v = con[(con.model == mdl) & (con.test_order == tst)].F1
        b.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=10,
                  color="#c9c8c2", lw=0)
        b.scatter([i], [v.mean()], s=34, color=S1 if tst == "fixed" else S2,
                  edgecolor="white", lw=1, zorder=3)
    b.set_xticks(range(4), ["fixed\n-> fixed", "fixed\n-> shuffled",
                            "shuffled\n-> fixed", "shuffled\n-> shuffled"])
    b.set_xlabel("training order -> test order")
    b.set_ylim(0.6, 0.75)
    b.axhline(0.7128, color=INK2, lw=0.8, ls=":")
    b.text(3.45, 0.7128, "published\n71.3 %", fontsize=6, color=INK2, va="center")
    b.set_ylabel("segment F1")
    b.set_title("Con4m on Tufts, 5 test resamples each", loc="left")
    grid(b)
    panel(b, "B")
    fig.savefig(out / "fig6_reproductions.pdf", bbox_inches="tight")
    fig.savefig(out / "fig6_reproductions.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = project_paths()
    out = p.root / "paper" / "manuscript" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig1(out)
    fig2(p.results, out)
    fig3(p.results, out)
    fig4(p.results, out)
    fig5(p.results, out)
    fig6(p.results, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
