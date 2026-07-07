"""
export_model.py -- freeze a deployable model for the demo backend (Phase C).

Trains the headline MLP on the circular+expert features of ALL 1,100 molecules
(the demo's "known" set) and exports everything the backend needs as plain numpy,
so the Hugging Face Space needs only RDKit + numpy (no torch, no sklearn):

  serve/artifacts/mlp_weights.npz  - the 7 Linear layers (W,b) as numpy
  serve/artifacts/scaler.npz       - MinMaxScaler min_/scale_ (2258 features + target)
  serve/artifacts/train.npz        - scaled feature matrix, true targets, NN threshold
  serve/artifacts/smiles.json      - the 1,100 known SMILES (for nearest-neighbour lookup)

The backend reproduces training featurization exactly by importing featurize.py.

Run (after the model matrix, to avoid CPU contention):  python export_model.py
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import featurize
from models import MLPRegressor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve", "artifacts")


def main(seed: int = 0):
    os.makedirs(OUT, exist_ok=True)
    smiles, y = featurize.load_smiles_and_target()
    circ, exp, _ = featurize.load_cached()
    X = np.hstack([circ, exp]).astype(np.float64)          # [1100, 2258]

    # scale [X | y] over all rows (deployment uses all known data as reference)
    data = np.hstack([X, y.reshape(-1, 1)])
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
    scaled = scaler.transform(data)
    Xs, ys = scaled[:, :X.shape[1]], scaled[:, X.shape[1]:]

    # train on all 1100 with a 90/10 split for best-epoch early stopping
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(Xs))
    cut = int(0.9 * len(Xs))
    tr, va = perm[:cut], perm[cut:]
    print(f"Training MLP on {len(tr)} (val {len(va)}) ...")
    mlp = MLPRegressor(input_dim=Xs.shape[1], seed=seed)
    mlp.fit(Xs[tr], ys[tr], Xs[va], ys[va])

    # extract the 7 Linear layers as numpy
    import torch.nn as nn
    layers = [m for m in mlp.model if isinstance(m, nn.Linear)]
    weights = {}
    for i, lin in enumerate(layers):
        weights[f"W{i}"] = lin.weight.detach().cpu().numpy().astype(np.float32)
        weights[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float32)
    np.savez(os.path.join(OUT, "mlp_weights.npz"), n_layers=len(layers), **weights)

    # scaler params (transform: X_scaled = X * scale_ + min_)
    np.savez(os.path.join(OUT, "scaler.npz"),
             min_=scaler.min_.astype(np.float64), scale_=scaler.scale_.astype(np.float64))

    # applicability-domain reference: nearest-neighbour distance distribution
    from sklearn.neighbors import NearestNeighbors
    nn5 = NearestNeighbors(n_neighbors=2).fit(Xs)
    d, _ = nn5.kneighbors(Xs)                 # [:,1] = distance to nearest *other* point
    nn_dist = d[:, 1]
    threshold = float(np.percentile(nn_dist, 95))   # "out of domain" above this
    np.savez(os.path.join(OUT, "train.npz"),
             X=Xs.astype(np.float32), y=y.astype(np.float32),
             nn_threshold=np.float32(threshold),
             nn_median=np.float32(np.median(nn_dist)))
    json.dump(list(smiles), open(os.path.join(OUT, "smiles.json"), "w"))

    # quick self-check: predicted vs true on a couple of rows
    def predict_scaled(xs_row):
        h = xs_row
        for i in range(len(layers)):
            h = h @ weights[f"W{i}"].T + weights[f"b{i}"]
            if i < len(layers) - 1:
                h = np.maximum(h, 0.0)          # ReLU
        return h
    ps = predict_scaled(Xs[:3])
    y_pred = (ps.ravel() - scaler.min_[-1]) / scaler.scale_[-1]
    print("sample true :", np.round(y[:3], 3))
    print("sample pred :", np.round(y_pred, 3))
    print(f"NN threshold (95th pct): {threshold:.3f}   artifacts -> {OUT}/")

    # make serve/ a self-contained Space bundle (featurize.py must ship with it)
    shutil.copy(featurize.__file__, os.path.join(os.path.dirname(OUT), "featurize.py"))
    print("copied featurize.py -> serve/")


if __name__ == "__main__":
    main()
