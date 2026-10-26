"""Figures added in the revision: simulation (fig7) and the two-step diagnostic (fig8).

    python scripts/93_revision_figures.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

f92 = importlib.import_module("92_paper_figures")
plt = f92.plt
S1, S2, S3, INK2, CHANCE, GRID, W2, MM = f92.S1, f92.S2, f92.S3, f92.INK2, f92.CHANCE, f92.GRID, f92.W2, f92.MM
from eegfnirs.utils import project_paths  # noqa: E402

GREY = "#b9b8b2"


def fig7(res: Path, out: Path) -> None:
    ceil = pd.read_csv(res / "sim" / "E4_ceilings.csv")
    val = pd.read_csv(res / "sim" / "E5_decompose_validation.csv")
    real = pd.read_csv(res / "shin_decompose" / "conditional_decoding.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 64 * MM),
                               gridspec_kw={"width_ratios": [4, 6], "wspace": 0.3})

    designs = [("shared", "shared"), ("latin", "Latin sq.\nacross\nsubjects"),
               ("random_rule", "random,\nshared\nrule"), ("random", "random\nper\nsubject")]
    x = np.arange(len(designs))
    for off, pred, col, lab in ((-0.18, "position", S1, "block position"),
                                (0.18, "prev", S2, "previous condition")):
        v = [ceil[(ceil.design == d) & (ceil.predictor == pred)].ceiling.iloc[0] * 100 for d, _ in designs]
        a.bar(x + off, v, 0.34, color=col, label=lab, zorder=2)
    a.axhline(25, color=CHANCE, ls=(0, (3, 2)), lw=0.8)
    a.set_xticks(x, [d[1] for d in designs])
    a.set_ylabel("accuracy from design alone (%)")
    a.set_ylim(0, 105)
    a.set_title("Design-only accuracy (4 classes)", loc="left")
    a.legend(frameon=False, loc="upper right")
    f92.grid(a)
    f92.panel(a, "A")

    cases = [("carryover", "sim:\ncarry-\nover"), ("drift", "sim:\ndrift"),
             ("position", "sim:\nposition"), ("none (pure noise)", "sim:\nnoise"),
             ("real_none", "Shin\nno filter"), ("real_minimum", "Shin\ncausal")]
    analyses = [("unconditional", "no control", INK2), ("same previous load", "previous held fixed", S2),
                ("same position in run", "position held fixed", S1)]
    xs = np.arange(len(cases))
    for j, (an, lab, col) in enumerate(analyses):
        means, errs = [], []
        for c, _ in cases:
            if c.startswith("real_"):
                r = real[(real["filter"] == c[5:]) & (real.analysis == an)]
                means.append(r.excess.iloc[0] * 100); errs.append(0)
            else:
                g = val[(val.case == c) & (val.analysis == an)].excess * 100
                means.append(g.mean()); errs.append(g.sem())
        b.bar(xs + (j - 1) * 0.26, means, 0.24, yerr=errs, color=col, label=lab, zorder=2,
              error_kw=dict(lw=0.7, capsize=0))
    b.axhline(0, color=CHANCE, lw=0.8)
    b.axvline(3.5, color=GRID, lw=1)
    b.set_xticks(xs, [c[1] for c in cases])
    b.set_ylabel("null-window accuracy above chance (pp)")
    b.set_title("Conditional decoding (Shin-like design)", loc="left")
    b.legend(frameon=False, loc="upper right")
    f92.grid(b)
    f92.panel(b, "B")
    fig.savefig(out / "fig7_simulation.pdf", bbox_inches="tight")
    fig.savefig(out / "fig7_simulation.png", bbox_inches="tight")
    plt.close(fig)


def fig8(res: Path, out: Path) -> None:
    e6 = pd.read_csv(res / "sim" / "E6_inflation.csv")
    bench = pd.read_csv(res / "benchmark" / "summary_wide.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W2, 62 * MM), gridspec_kw={"wspace": 0.42})
    g = e6[(e6.design == "shared")].groupby(["eval", "component", "beta"])[["null", "transfer", "inflation"]].mean().reset_index()
    cols = {"carryover": S2, "drift": S1, "position": S3}
    for ax, xcol, title, letter in ((a, "null", "step 1 (null probe)", "A"),
                                    (b, "transfer", "step 2 (rest-to-task)", "B")):
        for comp, col in cols.items():
            for ev, mk in (("all", "o"), ("last3", "^")):
                d = g[(g.component == comp) & (g["eval"] == ev)]
                ax.scatter(d[xcol] * 100, d.inflation * 100, s=22, color=col, marker=mk,
                           edgecolor="white", lw=0.5, zorder=3)
        ax.axhline(0, color=CHANCE, lw=0.8)
        ax.axvline(0, color=CHANCE, lw=0.8)
        ax.set_xlabel("null probe above chance (pp)" if xcol == "null" else "rest$\\to$task above chance (pp)")
        ax.set_ylabel("task-decoding inflation (pp)")
        ax.set_title(f"Simulation: {title}", loc="left", fontsize=7)
        f92.grid(ax)
        f92.panel(ax, letter)
    h = [plt.Line2D([], [], color=col, marker="o", ls="", label=k) for k, col in cols.items()]
    h += [plt.Line2D([], [], color=INK2, marker="^", ls="", label="unbalanced positions"),
          plt.Line2D([], [], color=INK2, marker="o", ls="", label="balanced positions")]
    fig.legend(handles=h, frameon=False, fontsize=6, loc="lower center", ncol=5, bbox_to_anchor=(0.36, -0.17))

    dcol = {"shin_none": (GREY, "Shin, no filter"), "shin_minimum": (INK2, "Shin, causal"),
            "tufts_raw": (S1, "Tufts, 3-class"), "tufts_raw_(0, 2)": (S2, "Tufts, 0 vs 2")}
    for ds, (col, lab) in dcol.items():
        d = bench[bench.dataset == ds]
        c.scatter(d.task * 100, d["rest->task"] * 100, s=24, color=col, label=lab,
                  edgecolor="white", lw=0.5, zorder=3)
    lim = (-8, 30)
    c.plot(lim, lim, color=CHANCE, ls=(0, (3, 2)), lw=0.8)
    c.axhline(0, color=CHANCE, lw=0.8)
    c.set_xlim(0, 30)
    c.set_ylim(*lim)
    c.set_xlabel("task decoding above chance (pp)")
    c.set_ylabel("rest$\\to$task above chance (pp)")
    c.set_title("Real data: six decoders", loc="left", fontsize=7)
    c.legend(frameon=False, fontsize=5.5, loc="upper left")
    f92.grid(c)
    f92.panel(c, "C")
    fig.savefig(out / "fig8_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(out / "fig8_diagnostic.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = project_paths()
    out = p.root / "paper" / "manuscript" / "figures"
    fig7(p.results, out)
    fig8(p.results, out)
    print("ok")


if __name__ == "__main__":
    main()
