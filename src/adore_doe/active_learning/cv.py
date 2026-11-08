"""Cross-validated predictive R^2 (ADR-009) for GPR surrogates.

predictive R^2 = 1 - SSE/SST with SSE from POOLED held-out (out-of-fold) predictions and
SST = sum (y - mean(y))^2 over the same observations.  Negative values are KEPT.  If SST = 0 or
the sample is too small for CV the value is NaN with an explicit status (treated as "target not
met" by the stopping rule).  The training R^2 is never used here.

Leakage control: for every fold the GP is fitted -- INCLUDING its hyper-parameters -- from the
training part of that fold only, with data-independent starts (no warm start from a full-data fit,
which would let the held-out responses influence the optimiser).  `refit_hyperparameters=False`
(hyper-parameters fixed to the full-data fit) is available for speed but is leaky and is flagged.
Folds are reproducible from `seed`.  scheme: 'kfold' | 'repeated_kfold' | 'loo'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.seed import derive_seed, make_rng, resolve_seed
from ..exceptions import DesignError
from .gpr import GaussianProcess

SCHEMES = ("kfold", "repeated_kfold", "loo")


def make_folds(n: int, scheme: str = "kfold", n_splits: int = 5, repeats: int = 1, seed=None):
    """List (one entry per repeat) of lists of (train_idx, test_idx)."""
    if scheme not in SCHEMES:
        raise DesignError(f"scheme must be one of {SCHEMES}.")
    if n < 2:
        raise DesignError("cross-validation needs at least 2 observations.")
    if scheme == "loo":
        return [[(np.delete(np.arange(n), i), np.array([i])) for i in range(n)]]
    if n_splits < 2 or repeats < 1:
        raise DesignError("n_splits must be >= 2 and repeats >= 1.")
    k, root = min(int(n_splits), n), resolve_seed(seed)
    out = []
    for r in range(int(repeats) if scheme == "repeated_kfold" else 1):
        perm = make_rng(derive_seed(root, "folds", str(r))).permutation(n)
        chunks = np.array_split(perm, k)
        out.append([(np.concatenate([c for j, c in enumerate(chunks) if j != f]), np.sort(chunks[f]))
                    for f in range(k)])
    return out


def oof_predict(X, y, fit_predict, scheme="kfold", n_splits=5, repeats=1, seed=None) -> np.ndarray:
    """Out-of-fold predictions, shape (repeats, n).  fit_predict(Xtr, ytr, Xte, fold_seed) -> means."""
    X, y = np.atleast_2d(np.asarray(X, float)), np.asarray(y, float).ravel()
    root = resolve_seed(seed)
    folds = make_folds(len(y), scheme, n_splits, repeats, root)
    oof = np.full((len(folds), len(y)), np.nan)
    for r, rep in enumerate(folds):
        for f, (tr, te) in enumerate(rep):
            oof[r, te] = fit_predict(X[tr], y[tr], X[te], derive_seed(root, "cvfit", str(r), str(f)))
    return oof


@dataclass
class R2Result:
    value: float            # NaN when status != 'ok'
    status: str             # 'ok' | 'sst_zero' | 'insufficient_samples' | 'nonfinite_prediction'
    sse: float
    sst: float
    rmse: float
    mae: float
    n: int
    reason: str = ""

    @property
    def valid(self) -> bool:
        return self.status == "ok" and np.isfinite(self.value)


def predictive_r2(y, oof, min_samples: int = 3) -> R2Result:
    y = np.asarray(y, float).ravel()
    oof = np.atleast_2d(np.asarray(oof, float))
    n = len(y)
    nan = float("nan")
    if n < min_samples:
        return R2Result(nan, "insufficient_samples", nan, nan, nan, nan, n,
                        f"{n} observations < {min_samples}: too few for cross-validation")
    if not np.all(np.isfinite(oof)):
        return R2Result(nan, "nonfinite_prediction", nan, nan, nan, nan, n, "out-of-fold predictions are not finite")
    sst = float(np.sum((y - y.mean()) ** 2))
    err = y[None, :] - oof
    sse = float(np.mean(np.sum(err ** 2, axis=1)))
    rmse, mae = float(np.sqrt(sse / n)), float(np.mean(np.abs(err)))
    if np.ptp(y) <= 1e-12 * float(np.max(np.abs(y))):          # constant relative to its own magnitude
        return R2Result(nan, "sst_zero", sse, sst, rmse, mae, n, "the response is constant (SST = 0): R^2 undefined")
    return R2Result(1.0 - sse / sst, "ok", sse, sst, rmse, mae, n)


def gpr_cv_r2(X, y, *, scheme="kfold", n_splits=5, repeats=1, refit_hyperparameters=True,
              kernel="matern52", noise=None, n_restarts=2, gp_kwargs=None, seed=None):
    """Predictive R^2 of a GPR by cross-validation.  Returns (R2Result, oof, info)."""
    X, y = np.atleast_2d(np.asarray(X, float)), np.asarray(y, float).ravel()
    gp_kwargs = dict(gp_kwargs or {})
    root = resolve_seed(seed)
    fixed = None
    if not refit_hyperparameters:
        fixed = GaussianProcess(kernel, noise, n_restarts=n_restarts, seed=derive_seed(root, "full"),
                                **gp_kwargs).fit(X, y).theta

    def fp(Xtr, ytr, Xte, s):
        gp = GaussianProcess(kernel, noise, n_restarts=n_restarts, seed=s, **gp_kwargs)
        gp.fit(Xtr, ytr) if fixed is None else gp.fit(Xtr, ytr, fixed_theta=fixed)
        return gp.predict(Xte, return_std=False)

    if len(y) < 3:
        res = predictive_r2(y, np.zeros((1, len(y))))
        return res, None, {"scheme": scheme, "leaky": not refit_hyperparameters}
    oof = oof_predict(X, y, fp, scheme, n_splits, repeats, root)
    info = {"scheme": scheme, "n_splits": min(n_splits, len(y)) if scheme != "loo" else len(y),
            "repeats": oof.shape[0], "leaky": not refit_hyperparameters, "seed": root}
    return predictive_r2(y, oof), oof, info
