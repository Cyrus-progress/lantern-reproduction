"""
LANTERN demo backend (Phase C) -- exact transfection-efficiency prediction.

A tiny FastAPI service that predicts a pasted SMILES using the exported MLP.
Featurization reuses the project's featurize.py, so predictions are byte-identical
to training. Inference is pure numpy (no torch/sklearn needed on the server).

Returns: predicted score, an applicability/reliability badge (distance to the
nearest known lipid), the k nearest known lipids, and an RDKit 2D structure SVG.

Local run:  uvicorn app:app --port 7860     (from serve/)
On Hugging Face Spaces (Docker SDK) it listens on $PORT (7860).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# featurize.py lives one level up in the repo; on the Space it's copied alongside.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import featurize  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

# ---- load frozen artifacts once ----
_w = np.load(os.path.join(ART, "mlp_weights.npz"))
N_LAYERS = int(_w["n_layers"])
WEIGHTS = [(_w[f"W{i}"], _w[f"b{i}"]) for i in range(N_LAYERS)]
_s = np.load(os.path.join(ART, "scaler.npz"))
SC_MIN, SC_SCALE = _s["min_"], _s["scale_"]          # transform: Xs = X*scale + min
_t = np.load(os.path.join(ART, "train.npz"))
X_TRAIN, Y_TRAIN = _t["X"], _t["y"]
NN_THRESHOLD, NN_MEDIAN = float(_t["nn_threshold"]), float(_t["nn_median"])
SMILES = json.load(open(os.path.join(ART, "smiles.json")))
NFEAT = X_TRAIN.shape[1]


def mlp_forward(xs: np.ndarray) -> float:
    h = xs
    for i, (W, b) in enumerate(WEIGHTS):
        h = h @ W.T + b
        if i < N_LAYERS - 1:
            h = np.maximum(h, 0.0)                    # ReLU
    return float(h.ravel()[0])


def predict(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "RDKit could not parse that SMILES."}
    raw = featurize.featurize_combined([smiles])[0]                 # [2258], unscaled
    xs = raw * SC_SCALE[:NFEAT] + SC_MIN[:NFEAT]                    # MinMax transform
    pred_scaled = mlp_forward(xs)
    score = (pred_scaled - SC_MIN[-1]) / SC_SCALE[-1]              # inverse target

    # applicability domain: distance to nearest known lipid
    dists = np.linalg.norm(X_TRAIN - xs, axis=1)
    order = np.argsort(dists)[:5]
    nearest = float(dists[order[0]])
    if nearest <= NN_MEDIAN:
        reliability = "high"
    elif nearest <= NN_THRESHOLD:
        reliability = "moderate"
    else:
        reliability = "low"                            # out of the training domain

    neighbors = [{"smiles": SMILES[i], "score": round(float(Y_TRAIN[i]), 3),
                  "distance": round(float(dists[i]), 3)} for i in order]

    d = rdMolDraw2D.MolDraw2DSVG(320, 220)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    svg = d.GetDrawingText()

    return {"score": round(score, 3), "reliability": reliability,
            "nearest_distance": round(nearest, 3),
            "domain_threshold": round(NN_THRESHOLD, 3),
            "neighbors": neighbors, "svg": svg}


class Req(BaseModel):
    smiles: str


app = FastAPI(title="LANTERN transfection predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/")
def health():
    return {"status": "ok", "n_known_lipids": len(SMILES), "features": NFEAT,
            "note": "POST /predict {smiles} -> transfection-efficiency prediction"}


@app.post("/predict")
def predict_route(req: Req):
    return predict(req.smiles)
