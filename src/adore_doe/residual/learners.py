"""Residual learners (ADR-011): Gaussian process, small neural network (MLP), support-vector regression.
Random Forest is intentionally NOT included in this version (ADR-011: added later depending on research results).

All learners map coded factors -> residual, standardise the target internally, and take an explicit seed.
They need scikit-learn (`pip install 'adore-doe[ml]'`).  `none` predicts 0 (used as the fallback).
"""
from __future__ import annotations

import warnings

import numpy as np

from ..exceptions import DesignError


def _sk():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError("Residual learners need scikit-learn: pip install 'adore-doe[ml]'") from e


class _Base:
    name = "base"

    def __init__(self):
        self._zero = False

    def _prep(self, e):
        e = np.asarray(e, float).ravel()
        self.mu, self.sd = float(e.mean()), float(e.std())
        self._zero = self.sd < 1e-14 * max(1.0, abs(self.mu))
        return (e - self.mu) / (1.0 if self._zero else self.sd)

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if self._zero or self._model is None:
            return np.full(len(X), getattr(self, "mu", 0.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._model.predict(X) * self.sd + self.mu


class NoneLearner(_Base):
    name = "none"

    def fit(self, X, e, seed=None):
        self._model, self._zero, self.mu = None, True, 0.0
        return self

    def predict(self, X):
        return np.zeros(len(np.atleast_2d(X)))


class GPLearner(_Base):
    """sklearn GP: constant * Matern-5/2 (ARD) + white noise, marginal-likelihood hyper-parameters."""
    name = "gp"

    def __init__(self, n_restarts: int = 2):
        super().__init__()
        self.n_restarts = n_restarts

    def fit(self, X, e, seed=None):
        _sk()
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        X = np.atleast_2d(np.asarray(X, float))
        y = self._prep(e)
        k = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(X.shape[1]), (0.05, 20.0), nu=2.5)
             + WhiteKernel(1e-2, (1e-8, 1.0)))
        self._model = None
        if not self._zero:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=self.n_restarts,
                                                       random_state=seed).fit(X, y)
        return self


class MLPLearner(_Base):
    """One hidden tanh layer, L2-regularised, L-BFGS (appropriate for tiny samples)."""
    name = "mlp"

    def __init__(self, hidden=(8,), alpha: float = 1.0, max_iter: int = 2000):
        super().__init__()
        self.hidden, self.alpha, self.max_iter = tuple(hidden), alpha, max_iter

    def fit(self, X, e, seed=None):
        _sk()
        from sklearn.neural_network import MLPRegressor
        X = np.atleast_2d(np.asarray(X, float))
        y = self._prep(e)
        self._model = None
        if not self._zero:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = MLPRegressor(hidden_layer_sizes=self.hidden, activation="tanh", solver="lbfgs", alpha=self.alpha,
                                           max_iter=self.max_iter, random_state=seed).fit(X, y)
        return self


class SVRLearner(_Base):
    """RBF support-vector regression; C and epsilon chosen by an inner K-fold CV on a small grid."""
    name = "svr"

    def __init__(self, C=(0.1, 1.0, 10.0, 100.0), epsilon=(0.01, 0.1), inner_splits: int = 5):
        super().__init__()
        self.C, self.epsilon, self.inner_splits = C, epsilon, inner_splits

    def fit(self, X, e, seed=None):
        _sk()
        from sklearn.svm import SVR
        from .folds import make_folds
        X = np.atleast_2d(np.asarray(X, float))
        y = self._prep(e)
        self._model = None
        if self._zero:
            return self
        best, folds = None, make_folds(len(y), self.inner_splits, 1, seed)[0] if len(y) >= 4 else None
        for C in self.C:
            for eps in self.epsilon:
                if folds is None:
                    sc = 0.0
                else:
                    err = 0.0
                    for tr, te in folds:
                        m = SVR(kernel="rbf", C=C, epsilon=eps, gamma="scale").fit(X[tr], y[tr])
                        err += float(np.sum((y[te] - m.predict(X[te])) ** 2))
                    sc = err
                if best is None or sc < best[0]:
                    best = (sc, C, eps)
        self.C_, self.epsilon_ = best[1], best[2]
        self._model = SVR(kernel="rbf", C=self.C_, epsilon=self.epsilon_, gamma="scale").fit(X, y)
        return self


LEARNERS = {"gp": GPLearner, "mlp": MLPLearner, "svr": SVRLearner}


def make_learner(name: str):
    if name == "none":
        return NoneLearner()
    if name not in LEARNERS:
        extra = " (Random Forest is deferred, ADR-011)" if name in ("rf", "random_forest") else ""
        raise DesignError(f"unknown residual learner {name!r}; available: {sorted(LEARNERS)}{extra}.")
    return LEARNERS[name]()
