"""
models.py -- model definitions for the LANTERN reproduction.

Baselines (scikit-learn), matching the paper's settings:
  * kNN : KNeighborsRegressor(n_neighbors=5, weights="uniform")
  * RF  : RandomForestRegressor(n_estimators=100, criterion="squared_error")
  * SVR : SVR(kernel="rbf")   (scikit-learn defaults)

MLP (PyTorch), matching the paper's Methods and the repo's FeedforwardRegressor:
  * 7 linear layers: in -> 200 -> 300 -> 500 -> 500 -> 300 -> 200 -> 1, ReLU
  * Adam, lr=2e-4, MSE loss, batch size 100, up to 100 epochs
  * "early stopping" = keep the weights from the lowest validation-loss epoch

Every model exposes the same interface:
    model.fit(X_train, y_train, X_val, y_val)
    model.predict(X) -> (N, 1) ndarray
so train.py can treat them uniformly.
"""
from __future__ import annotations

import copy

import numpy as np


# --------------------------------------------------------------------------- #
# scikit-learn baselines
# --------------------------------------------------------------------------- #
class SklearnRegressor:
    """Uniform wrapper around a scikit-learn regressor."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.estimator.fit(X_train, np.asarray(y_train).reshape(-1))
        return self

    def predict(self, X):
        return self.estimator.predict(X).reshape(-1, 1)


def make_knn():
    from sklearn.neighbors import KNeighborsRegressor
    return SklearnRegressor(KNeighborsRegressor(n_neighbors=5, weights="uniform"))


def make_rf(seed=None):
    from sklearn.ensemble import RandomForestRegressor
    return SklearnRegressor(
        RandomForestRegressor(n_estimators=100, criterion="squared_error",
                              random_state=seed)
    )


def make_svr():
    from sklearn.svm import SVR
    return SklearnRegressor(SVR(kernel="rbf"))


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
def _build_feedforward(input_dim: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 200), nn.ReLU(),
        nn.Linear(200, 300), nn.ReLU(),
        nn.Linear(300, 500), nn.ReLU(),
        nn.Linear(500, 500), nn.ReLU(),
        nn.Linear(500, 300), nn.ReLU(),
        nn.Linear(300, 200), nn.ReLU(),
        nn.Linear(200, 1),
    )


class MLPRegressor:
    """Feed-forward MLP with keep-best-validation-epoch model selection."""

    def __init__(self, input_dim, lr=2e-4, epochs=100, batch_size=100,
                 device="cpu", seed=None, verbose=False):
        import torch

        self.torch = torch
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.verbose = verbose
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.model = _build_feedforward(input_dim).to(device)
        self.history = {"train": [], "val": []}

    def _loader(self, X, y, shuffle):
        torch = self.torch
        from torch.utils.data import DataLoader, TensorDataset

        ds = TensorDataset(
            torch.tensor(np.asarray(X), dtype=torch.float32),
            torch.tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train, y_train, X_val, y_val):
        torch = self.torch
        import torch.nn as nn

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        train_loader = self._loader(X_train, y_train, shuffle=True)
        val_loader = self._loader(X_val, y_val, shuffle=False)

        best_val = None
        best_state = None
        for epoch in range(self.epochs):
            self.model.train()
            tot, n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * xb.size(0)
                n += xb.size(0)
            train_loss = tot / n

            self.model.eval()
            vtot, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    vtot += loss_fn(self.model(xb), yb).item() * xb.size(0)
                    vn += xb.size(0)
            val_loss = vtot / vn

            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
            if self.verbose:
                print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")

        self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
            return self.model(xb).cpu().numpy()


# --------------------------------------------------------------------------- #
def make_model(name: str, input_dim: int, seed=None, **kw):
    name = name.lower()
    if name == "knn":
        return make_knn()
    if name == "rf":
        return make_rf(seed=seed)
    if name == "svr":
        return make_svr()
    if name == "mlp":
        return MLPRegressor(input_dim=input_dim, seed=seed, **kw)
    raise ValueError(f"unknown model {name!r}")


MODEL_NAMES = ["knn", "rf", "svr", "mlp"]
