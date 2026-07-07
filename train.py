"""
train.py -- training / evaluation pipeline for the LANTERN reproduction.

Reproduces the repo's preprocessing exactly (pipeline/preprocess.py):
  * features concatenated in order [circular(2048), expert(210)] = 2258 dims
  * a single MinMaxScaler(feature_range=(-1, 1)) is fit on hstack([X_all, y])
    over ALL 1100 rows, INCLUDING the target column and the test rows
    (a mild leak that we replicate so numbers are comparable to the paper)
  * models train on scaled X and scaled y; predictions are inverse-transformed
    to original target units before metrics are computed
  * train/val/test membership comes from the repo's precomputed index files
    data/splits/AGILE/{random,Murcko_scaffold,scaffold_balanced}.npy

Examples:
  python train.py --model all --split random           # baselines + MLP, 1 seed
  python train.py --model mlp --split random --seed 0   # single MLP run + scatter
  python train.py --model all --split random --seeds 10 # 10-seed mean+/-std (Phase 2)
  python train.py --model knn --features circular       # ablate feature blocks
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# LightGBM + sklearn emit a benign feature-names warning when fed numpy arrays.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import featurize
from evaluate import format_metrics, regression_metrics, scatter_plot
from models import MODEL_NAMES, make_model

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
SPLIT_DIR = featurize.SPLIT_DIR

SPLIT_FILES = {
    "random": "random.npy",
    "scaffold": "Murcko_scaffold.npy",
    "scaffold_balanced": "scaffold_balanced.npy",
}

# Paper's reported numbers (Morgan + Expert), for the side-by-side table.
PAPER = {
    "random": {
        "mlp": {"R2": 0.8161, "RMSE": 1.4308, "MAE": 1.1003, "Pearson_r": 0.9053},
        "svr": {"R2": 0.7285, "RMSE": 1.7387, "MAE": 1.3739, "Pearson_r": 0.8702},
        "rf":  {"R2": 0.7169, "RMSE": 1.7753, "MAE": 1.3391, "Pearson_r": 0.8480},
        "knn": {"R2": 0.6148, "RMSE": 2.0710, "MAE": 1.5969, "Pearson_r": 0.7866},
        "agile": {"R2": 0.2655, "RMSE": 2.8600, "MAE": 2.3328, "Pearson_r": 0.5488},
    },
    "scaffold": {
        "mlp": {"R2": 0.4532, "RMSE": 2.0746, "MAE": 1.7726, "Pearson_r": 0.7344},
        "svr": {"R2": 0.3157, "RMSE": 2.3209, "MAE": 1.9835, "Pearson_r": 0.6425},
        "rf":  {"R2": 0.4747, "RMSE": 2.0334, "MAE": 1.5947, "Pearson_r": 0.6895},
        "knn": {"R2": 0.5919, "RMSE": 1.7923, "MAE": 1.4177, "Pearson_r": 0.7892},
        "agile": {"R2": 0.0057, "RMSE": 2.7976, "MAE": 2.3389, "Pearson_r": 0.4690},
    },
}


# --------------------------------------------------------------------------- #
# Data assembly + preprocessing
# --------------------------------------------------------------------------- #
def load_feature_matrix(feature_combo, source="computed"):
    """Return (X [N, D], y [N]) for the requested feature blocks."""
    if source == "computed":
        circ, exp, y = featurize.load_cached()
    elif source == "answerkey":
        import pickle
        smiles, y = featurize.load_smiles_and_target()
        with open(os.path.join(featurize.ANSWER_KEY_DIR, "circular.pkl"), "rb") as f:
            cd = pickle.load(f)
        with open(os.path.join(featurize.ANSWER_KEY_DIR, "expert.pkl"), "rb") as f:
            ed = pickle.load(f)
        circ = np.array([np.asarray(cd[s], float) for s in smiles])
        exp = np.array([np.asarray(ed[s], float) for s in smiles])
    else:
        raise ValueError(f"unknown source {source!r}")

    blocks = {"circular": circ, "expert": exp}
    if "grover" in feature_combo:
        smiles, _ = featurize.load_smiles_and_target()
        blocks["grover"] = featurize.load_grover(smiles)
    X = np.hstack([blocks[name] for name in feature_combo])
    return X, y


def preprocess(X, y, train_idx=None, leak=True):
    """MinMax(-1,1) scaling of [X | y].

    leak=True (default): fit the scaler over ALL rows, including the test rows and
        the target column -- this reproduces the paper's pipeline/preprocess.py and
        is kept so our numbers stay comparable to theirs.
    leak=False: fit the scaler on TRAIN rows only (the honest protocol). Requires
        train_idx. Everything is then transformed with that train-fit scaler.
    """
    y = np.asarray(y).reshape(-1, 1)
    data = np.hstack([X, y])
    fit_on = data if (leak or train_idx is None) else data[train_idx]
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(fit_on)
    scaled = scaler.transform(data)
    d = X.shape[1]
    return scaled[:, :d], scaled[:, d:], scaler


def inverse_target(y_scaled, scaler):
    """Invert MinMax scaling for the (last) target column only."""
    y_scaled = np.asarray(y_scaled).reshape(-1)
    return (y_scaled - scaler.min_[-1]) / scaler.scale_[-1]


def load_split(split):
    path = os.path.join(SPLIT_DIR, SPLIT_FILES[split])
    train_idx, val_idx, test_idx = np.load(path, allow_pickle=True)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# --------------------------------------------------------------------------- #
# One experiment
# --------------------------------------------------------------------------- #
def run_experiment(model_name, split, seed=None, feature_combo=("circular", "expert"),
                   source="computed", verbose=False, return_preds=False, leak=True):
    X, y = load_feature_matrix(feature_combo, source=source)
    tr, va, te = load_split(split)
    Xs, ys, scaler = preprocess(X, y, train_idx=tr, leak=leak)

    model = make_model(model_name, input_dim=Xs.shape[1], seed=seed, verbose=verbose)
    model.fit(Xs[tr], ys[tr], Xs[va], ys[va])

    pred_scaled = model.predict(Xs[te])
    y_pred = inverse_target(pred_scaled, scaler)
    y_true = inverse_target(ys[te], scaler)  # == original y[te]
    metrics = regression_metrics(y_true, y_pred)
    if return_preds:
        return metrics, y_true, y_pred
    return metrics


def run_multi_seed(model_name, split, seeds, **kw):
    rows = [run_experiment(model_name, split, seed=s, **kw) for s in range(seeds)]
    keys = ["R2", "RMSE", "MAE", "Pearson_r"]
    agg = {k: (float(np.mean([r[k] for r in rows])),
              float(np.std([r[k] for r in rows]))) for k in keys}
    return agg, rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all",
                    help="knn | rf | svr | mlp | all")
    ap.add_argument("--split", default="random",
                    choices=list(SPLIT_FILES), help="split scheme")
    ap.add_argument("--features", default="circular,expert",
                    help="comma list of blocks: circular,expert")
    ap.add_argument("--seed", type=int, default=0, help="single-run seed")
    ap.add_argument("--seeds", type=int, default=0,
                    help="if >0, run this many seeds and report mean+/-std")
    ap.add_argument("--source", default="computed",
                    choices=["computed", "answerkey"],
                    help="use our featurize.py output or the repo's pickles")
    ap.add_argument("--no-leak", action="store_true",
                    help="fit the scaler on train rows only (honest, uninflated)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    combo = tuple(s.strip() for s in args.features.split(","))
    models = MODEL_NAMES if args.model == "all" else [args.model.lower()]
    leak = not args.no_leak
    tag = (args.tag or f"{args.split}_{'-'.join(combo)}") + ("" if leak else "_noleak")

    print(f"\nSplit={args.split}  features={combo}  source={args.source}  "
          f"leak={leak}")
    print("=" * 78)

    records = []
    if args.seeds > 0:
        print(f"Multi-seed run: {args.seeds} seeds\n")
        for m in models:
            agg, _ = run_multi_seed(m, args.split, args.seeds, feature_combo=combo,
                                    source=args.source, leak=leak)
            print(f"{m.upper():4s}  R2={agg['R2'][0]:.4f}+/-{agg['R2'][1]:.4f}  "
                  f"RMSE={agg['RMSE'][0]:.4f}+/-{agg['RMSE'][1]:.4f}  "
                  f"MAE={agg['MAE'][0]:.4f}+/-{agg['MAE'][1]:.4f}  "
                  f"r={agg['Pearson_r'][0]:.4f}+/-{agg['Pearson_r'][1]:.4f}")
            rec = {"model": m, **{f"{k}_mean": v[0] for k, v in agg.items()},
                   **{f"{k}_std": v[1] for k, v in agg.items()}}
            records.append(rec)
        out = os.path.join(RESULTS_DIR, f"metrics_{tag}_{args.seeds}seeds.csv")
        pd.DataFrame(records).to_csv(out, index=False)
        print(f"\nsaved -> {out}")
        return

    # single-seed run for each model
    paper = PAPER.get(args.split, {})
    for m in models:
        make_plot = (not args.no_plot) and (m == "mlp" or len(models) == 1)
        metrics, y_true, y_pred = run_experiment(
            m, args.split, seed=args.seed, feature_combo=combo,
            source=args.source, return_preds=True, leak=leak)
        line = f"{m.upper():4s}  {format_metrics(metrics)}"
        if m in paper:
            line += f"   | paper R2={paper[m]['R2']:.4f}"
        print(line)
        rec = {"model": m, **metrics}
        if m in paper:
            rec["paper_R2"] = paper[m]["R2"]
        records.append(rec)
        if make_plot:
            p = scatter_plot(
                y_true, y_pred,
                title=f"LANTERN {m.upper()} ({args.split} split, {'+'.join(combo)})",
                out_path=os.path.join(RESULTS_DIR, f"scatter_{m}_{tag}.png"),
                metrics=metrics)
            print(f"      scatter -> {p}")

    out = os.path.join(RESULTS_DIR, f"metrics_{tag}.csv")
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
