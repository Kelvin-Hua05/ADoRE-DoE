"""Out-of-fold (OOF) RSM predictions and residual targets (ADR-006).

For every fold the RSM -- INCLUDING its term selection -- is refitted from the training part of that fold only.
The residual target of run i is  e_i = y_i - mean_r yhat_r^(-fold(i))  (mean over the repeats of the repeated K-fold),
so e_i never depends on y_i through a model that saw y_i.  Folds whose model is not identifiable are repaired by
dropping the aliased terms (counted in `info`), never by a pseudo-inverse.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..exceptions import DesignError
from ..rsm import fit_arrays, select_terms, universe
from .folds import make_folds


@dataclass
class RSMFitter:
    """Recipe for the RSM refitted inside every fold: full `model`, optionally with term selection."""
    model: str = "quadratic"
    selection: dict | None = None
    alpha: float = 0.05
    on_rank_deficient: str = "drop"
    factor_names: list | None = None

    def fit(self, Z, y):
        Z = np.atleast_2d(np.asarray(Z, float))
        names = self.factor_names or [f"x{i + 1}" for i in range(Z.shape[1])]
        terms = (select_terms(Z, y, kind=self.model, **self.selection).terms if self.selection
                 else universe(self.model, Z.shape[1]))
        return fit_arrays(Z, y, terms, names, alpha=self.alpha, on_rank_deficient=self.on_rank_deficient)


@dataclass
class OOFResult:
    pred: np.ndarray
    residual: np.ndarray
    per_repeat: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residual ** 2)))


def oof_rsm(fitter: RSMFitter, Z, y, n_splits: int = 5, repeats: int = 3, seed=None) -> OOFResult:
    Z, y = np.atleast_2d(np.asarray(Z, float)), np.asarray(y, float).ravel()
    folds = make_folds(len(y), n_splits, repeats, seed)
    pr = np.full((len(folds), len(y)), np.nan)
    dropped_folds = 0
    for r, rep in enumerate(folds):
        for tr, te in rep:
            if len(tr) < 2:
                raise DesignError("a training fold has fewer than 2 runs.")
            res = fitter.fit(Z[tr], y[tr])
            dropped_folds += bool(getattr(res, "dropped_terms", []))
            pr[r, te] = res.predict(Z[te])
    pred = pr.mean(axis=0)
    return OOFResult(pred, y - pred, pr, {"n_splits": min(n_splits, len(y)), "repeats": len(folds),
                                          "n_folds_with_dropped_terms": int(dropped_folds)})
