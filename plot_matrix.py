"""
plot_matrix.py -- figures for the Phase-A "stronger models" matrix.

Reads results/stronger_models_matrix.csv (from run_matrix.py) and writes:
  results/stronger_models_R2.png  - R2 by model across the 3 splits (leaked)
  results/leak_effect.png         - how much the MinMax leak inflates R2

Robust to a partially-complete CSV (plots whatever rows exist).
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CSV = os.path.join(RESULTS, "stronger_models_matrix.csv")

SPLIT_LABEL = {"random": "Random", "scaffold": "Murcko scaffold",
               "scaffold_balanced": "Scaffold-balanced"}
SPLIT_COLOR = {"random": "#2563eb", "scaffold": "#e11d48",
               "scaffold_balanced": "#f59e0b"}


def fig_models(df):
    d = df[df["leak"] == True].copy()  # noqa: E712
    labels = [l for l in ["kNN", "RF", "SVR", "LightGBM", "MLP", "MLP-GROVER", "MLP-all"]
              if l in set(d["label"])]
    splits = [s for s in ["random", "scaffold", "scaffold_balanced"] if s in set(d["split"])]
    x = np.arange(len(labels))
    w = 0.8 / max(len(splits), 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for j, sp in enumerate(splits):
        sub = d[d["split"] == sp].set_index("label")
        means = [sub.loc[l, "R2_mean"] if l in sub.index else np.nan for l in labels]
        stds = [sub.loc[l, "R2_std"] if l in sub.index else 0 for l in labels]
        ax.bar(x + (j - (len(splits) - 1) / 2) * w, means, w, yerr=stds, capsize=3,
               label=SPLIT_LABEL[sp], color=SPLIT_COLOR[sp], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Test $R^2$ (mean $\\pm$ std, 10 seeds)")
    ax.set_title("Stronger-models comparison across data splits (leaked scaling)")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS, "stronger_models_R2.png")
    fig.savefig(out, dpi=150)
    print("saved ->", out)


def fig_leak(df):
    # Clean comparison only on the random split: on scaffold splits the no-leak
    # protocol makes flexible models extrapolate on out-of-range test features and
    # blow up numerically, which confounds the leak comparison (discussed in REPORT).
    d = df[df["split"] == "random"]
    piv = d.pivot_table(index="label", columns="leak", values="R2_mean")
    if True not in piv.columns or False not in piv.columns:
        print("leak_effect: need both leaked and no-leak random rows; skipping")
        return
    piv["delta"] = piv[True] - piv[False]
    labels = [l for l in ["kNN", "RF", "SVR", "LightGBM", "MLP"] if l in piv.index]
    vals = [piv.loc[l, "delta"] for l in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    colors = ["#c0392b" if v > 0 else "#2f855a" for v in vals]
    ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.0004 if v >= 0 else -0.0004), f"{v:+.4f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_ylabel("$R^2$ inflation from the leak\n(leaked $-$ honest), random split")
    ax.set_title("The MinMax data leak barely moves scores (random split)")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out = os.path.join(RESULTS, "leak_effect.png")
    fig.savefig(out, dpi=150)
    print("saved ->", out)


def main():
    df = pd.read_csv(CSV)
    print(f"{len(df)} rows in {CSV}")
    fig_models(df)
    fig_leak(df)


if __name__ == "__main__":
    main()
