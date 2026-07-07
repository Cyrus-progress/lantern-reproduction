"""
run_matrix.py -- Phase A: the full "stronger models" comparison matrix.

Runs every model x split x (leaked / no-leak) at 10 seeds and writes a tidy
long-format table to results/stronger_models_matrix.csv (saved incrementally,
so partial progress survives interruption).

Configs include the new LightGBM model and two GROVER-representation variants
(MLP on the 4800-dim pretrained-GNN embeddings, alone and combined). GROVER is
only available for the dataset molecules, so those configs run leaked only.

Run:  python run_matrix.py            (in the background; ~30 min of MLP training)
"""
from __future__ import annotations

import os
import time

import pandas as pd

import train

SPLITS = ["random", "scaffold", "scaffold_balanced"]
SEEDS = 10
OUT = os.path.join(train.RESULTS_DIR, "stronger_models_matrix.csv")

# (label, model, feature_combo)
CONFIGS = [
    ("kNN",        "knn",  ("circular", "expert")),
    ("RF",         "rf",   ("circular", "expert")),
    ("SVR",        "svr",  ("circular", "expert")),
    ("LightGBM",   "lgbm", ("circular", "expert")),
    ("MLP",        "mlp",  ("circular", "expert")),
    ("MLP-GROVER", "mlp",  ("grover",)),
    ("MLP-all",    "mlp",  ("circular", "expert", "grover")),
]
CIRC_EXP = ("circular", "expert")


def main():
    os.makedirs(train.RESULTS_DIR, exist_ok=True)
    rows, t0 = [], time.time()
    for split in SPLITS:
        for leak in (True, False):
            # no-leak is only about quantifying the scaler leak on the core
            # circular+expert features -> skip the GROVER variants there.
            configs = CONFIGS if leak else [c for c in CONFIGS if c[2] == CIRC_EXP]
            for label, model, combo in configs:
                agg, _ = train.run_multi_seed(
                    model, split, SEEDS, feature_combo=combo, leak=leak)
                rec = {"split": split, "leak": leak, "label": label,
                       "model": model, "features": "+".join(combo)}
                for k, (m, s) in agg.items():
                    rec[f"{k}_mean"] = round(m, 4)
                    rec[f"{k}_std"] = round(s, 4)
                rows.append(rec)
                pd.DataFrame(rows).to_csv(OUT, index=False)  # incremental
                print(f"[{time.time()-t0:5.0f}s] {split:17s} leak={int(leak)} "
                      f"{label:11s} R2={rec['R2_mean']:.4f}+/-{rec['R2_std']:.4f}",
                      flush=True)
    print(f"\nDONE ({time.time()-t0:.0f}s) -> {OUT}")


if __name__ == "__main__":
    main()
