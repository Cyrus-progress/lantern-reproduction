"""
plot_robustness.py -- Phase 2 robustness figure.

Reads the two multi-seed metric CSVs produced by
    python train.py --model all --split random   --seeds 10
    python train.py --model all --split scaffold --seeds 10
and draws a grouped bar chart of test R2 (mean +/- std over 10 seeds) for every
model, on the random split vs the Murcko scaffold split, side by side.

The point of the figure: show BOTH the collapse in mean R2 when moving to the
scaffold split AND the blow-up in run-to-run variance (error bars) that the
paper never reports.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

MODELS = ["knn", "rf", "svr", "mlp"]
LABELS = {"knn": "kNN", "rf": "RF", "svr": "SVR", "mlp": "MLP"}
# Paper's cited AGILE numbers (fixed baseline, not re-run).
AGILE = {"random": 0.2655, "scaffold": 0.0057}


def load(split):
    df = pd.read_csv(os.path.join(RESULTS, f"metrics_{split}_circular-expert_10seeds.csv"))
    df = df.set_index("model")
    return df


def main():
    rnd = load("random")
    scf = load("scaffold")

    x = np.arange(len(MODELS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))

    r_mean = [rnd.loc[m, "R2_mean"] for m in MODELS]
    r_std = [rnd.loc[m, "R2_std"] for m in MODELS]
    s_mean = [scf.loc[m, "R2_mean"] for m in MODELS]
    s_std = [scf.loc[m, "R2_std"] for m in MODELS]

    b1 = ax.bar(x - w / 2, r_mean, w, yerr=r_std, capsize=5,
                color="#4C72B0", label="Random split", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w / 2, s_mean, w, yerr=s_std, capsize=5,
                color="#C44E52", label="Murcko scaffold split", edgecolor="black", linewidth=0.6)

    # AGILE reference lines (cited, not re-run)
    ax.axhline(AGILE["random"], ls="--", lw=1.2, color="#4C72B0", alpha=0.8)
    ax.axhline(AGILE["scaffold"], ls="--", lw=1.2, color="#C44E52", alpha=0.8)
    ax.text(-0.45, AGILE["random"] + 0.012, "AGILE (random, cited)",
            ha="left", va="bottom", fontsize=8, color="#4C72B0")
    ax.text(-0.45, AGILE["scaffold"] + 0.012, "AGILE (scaffold, cited)",
            ha="left", va="bottom", fontsize=8, color="#C44E52")

    for bars, means, stds in [(b1, r_mean, r_std), (b2, s_mean, s_std)]:
        for rect, mn, sd in zip(bars, means, stds):
            ax.text(rect.get_x() + rect.get_width() / 2, mn + sd + 0.015,
                    f"{mn:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in MODELS])
    ax.set_ylabel("Test $R^2$  (mean $\\pm$ std over 10 seeds)")
    ax.set_title("LANTERN reproduction: model robustness across data splits\n"
                 "Random (in-distribution) vs Murcko scaffold (novel structures)")
    ax.set_ylim(-0.05, 0.92)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()

    out = os.path.join(RESULTS, "robustness_R2_random_vs_scaffold.png")
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")

    # also emit a tidy combined table
    rows = []
    for m in MODELS:
        rows.append({
            "model": LABELS[m],
            "random_R2": f"{rnd.loc[m,'R2_mean']:.4f} +/- {rnd.loc[m,'R2_std']:.4f}",
            "scaffold_R2": f"{scf.loc[m,'R2_mean']:.4f} +/- {scf.loc[m,'R2_std']:.4f}",
            "delta_R2": f"{rnd.loc[m,'R2_mean'] - scf.loc[m,'R2_mean']:+.4f}",
        })
    tbl = pd.DataFrame(rows)
    tbl_out = os.path.join(RESULTS, "robustness_table.csv")
    tbl.to_csv(tbl_out, index=False)
    print(tbl.to_string(index=False))
    print(f"saved -> {tbl_out}")


if __name__ == "__main__":
    main()
