"""
evaluate.py -- metrics and the reproduction scatter plot for LANTERN.

Metrics match the paper: R^2, RMSE, MAE, Pearson r, all computed in the
ORIGINAL target units (predictions are inverse-transformed before scoring).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Pearson_r": float(pearsonr(y_true, y_pred)[0]),
    }


def format_metrics(m: dict) -> str:
    return (f"R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  "
            f"MAE={m['MAE']:.4f}  r={m['Pearson_r']:.4f}")


def scatter_plot(y_true, y_pred, title, out_path, metrics=None):
    """Predicted-vs-true scatter, saved to out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(y_true, y_pred, s=22, alpha=0.6, edgecolor="none", color="#2b6cb0")
    ax.plot([lo, hi], [lo, hi], "--", color="0.4", lw=1.2, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True transfection efficiency")
    ax.set_ylabel("Predicted transfection efficiency")
    ax.set_title(title)
    if metrics:
        txt = (f"$R^2$ = {metrics['R2']:.4f}\nRMSE = {metrics['RMSE']:.4f}\n"
               f"MAE = {metrics['MAE']:.4f}\nPearson r = {metrics['Pearson_r']:.4f}")
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
