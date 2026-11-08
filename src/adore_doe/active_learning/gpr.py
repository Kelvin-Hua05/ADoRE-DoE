"""Gaussian-process surrogate (ADR-002): one GP per response, Matern-5/2 + ARD (RBF optional).

Model (coded inputs, standardised response ys = (y - mean)/std):
    f ~ GP(0, k),   k(x,x') = sf2 * m(r),   r = sqrt(sum_j ((x_j - x'_j)/ell_j)^2)
    matern52: m(r) = (1 + sqrt5 r + 5 r^2 / 3) exp(-sqrt5 r)          rbf: m(r) = exp(-r^2/2)
    ys = f + eps,  eps ~ N(0, noise_var)
Hyper-parameters theta = log[ell_1..ell_d, sf2, noise_var] (on the standardised scale) maximise the
log marginal likelihood by multi-start L-BFGS-B inside explicit BOUNDS; the start list contains an
optional warm start, a default start and `n_restarts` random starts (seeded).  If every start fails,
the default hyper-parameters are used, a warning is issued and `fit_info` records the fallback.

The latent posterior covariance depends only on the inputs and the hyper-parameters (not on y), which
is what makes rank-1 updates possible inside one acquisition round.
All returned means / variances are in the ORIGINAL units of y.  This is a GPR predictive variance,
NOT the classical I-optimality variance of a polynomial model (research_methodology §4.3).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg, optimize

from ..core.seed import make_rng
from ..exceptions import DesignError, InferenceWarning

_SQ5 = np.sqrt(5.0)
_BAD = 1e25


def kernel_matrix(A, B, ls, sf2, kind="matern52"):
    a, b = A / ls, B / ls
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T
    r = np.sqrt(np.maximum(d2, 0.0))
    if kind == "matern52":
        return sf2 * (1.0 + _SQ5 * r + 5.0 * r ** 2 / 3.0) * np.exp(-_SQ5 * r)
    if kind == "rbf":
        return sf2 * np.exp(-0.5 * r ** 2)
    raise DesignError("kernel must be 'matern52' or 'rbf'.")


def _chol(K):
    """Cholesky with escalating jitter; returns (L, jitter) or (None, None)."""
    n = K.shape[0]
    scale = max(float(np.mean(np.diag(K))), 1e-12)
    for j in (0.0, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            return linalg.cholesky(K + j * scale * np.eye(n), lower=True), j * scale
        except linalg.LinAlgError:
            continue
    return None, None


class GaussianProcess:
    def __init__(self, kernel: str = "matern52", noise: float | None = None,
                 ls_bounds=(0.05, 20.0), sf2_bounds=(0.01, 100.0), noise_bounds=(1e-6, 1.0),
                 n_restarts: int = 3, seed=None):
        if kernel not in ("matern52", "rbf"):
            raise DesignError("kernel must be 'matern52' or 'rbf'.")
        for nm, b in (("ls_bounds", ls_bounds), ("sf2_bounds", sf2_bounds), ("noise_bounds", noise_bounds)):
            if not (0 < b[0] < b[1]):
                raise DesignError(f"{nm} must satisfy 0 < lo < hi.")
        if noise is not None and noise <= 0:
            raise DesignError("noise (variance, original units) must be positive or None (learned).")
        self.kernel, self.noise, self.n_restarts, self.seed = kernel, noise, int(n_restarts), seed
        self.ls_bounds, self.sf2_bounds, self.noise_bounds = tuple(ls_bounds), tuple(sf2_bounds), tuple(noise_bounds)
        self._fitted = False

    # ------------------------------------------------------------ helpers
    def _bounds(self, d):
        nb = self.noise_bounds
        if self.noise is not None:                       # fixed noise: collapse the bound
            v = self.noise / self.y_std ** 2
            nb = (v, v)
        lo = [np.log(self.ls_bounds[0])] * d + [np.log(self.sf2_bounds[0]), np.log(nb[0])]
        hi = [np.log(self.ls_bounds[1])] * d + [np.log(self.sf2_bounds[1]), np.log(nb[1])]
        return list(zip(lo, hi))

    def _unpack(self, theta, d):
        return np.exp(theta[:d]), float(np.exp(theta[d])), float(np.exp(theta[d + 1]))

    def _nlml(self, theta, X, ys):
        n, d = X.shape
        ls, sf2, nv = self._unpack(theta, d)
        K = kernel_matrix(X, X, ls, sf2, self.kernel) + nv * np.eye(n)
        L, _ = _chol(K)
        if L is None:
            return _BAD
        a = linalg.cho_solve((L, True), ys)
        v = 0.5 * float(ys @ a) + float(np.sum(np.log(np.diag(L)))) + 0.5 * n * np.log(2 * np.pi)
        return v if np.isfinite(v) else _BAD

    def _default_theta(self, d):
        nv = (self.noise / self.y_std ** 2) if self.noise is not None else 1e-3
        return np.array([0.0] * d + [0.0, np.log(nv)])

    # ---------------------------------------------------------------- fit
    def fit(self, X, y, init_theta=None, fixed_theta=None):
        """Fit hyper-parameters (or use `fixed_theta`) and condition on (X, y).
        `init_theta` is a warm start; it must NOT be derived from held-out data (cross-validation
        therefore never passes it)."""
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float).ravel()
        n, d = X.shape
        if n != len(y) or n < 2:
            raise DesignError("need >= 2 observations and len(y) == len(X).")
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            raise DesignError("X and y must be finite.")
        self.X, self.d = X, d
        self.y_mean = float(y.mean())
        sd = float(y.std())
        # "constant" is judged RELATIVE to the magnitude of y (femto-farad responses are not constant)
        self.constant_response = bool(np.ptp(y) <= 1e-12 * float(np.max(np.abs(y))))
        self.y_std = 1.0 if self.constant_response else sd
        ys = (y - self.y_mean) / self.y_std
        bounds = self._bounds(d)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        info = {"n_starts": 0, "fallback": False, "fixed": False, "success": []}
        if fixed_theta is not None:
            theta = np.clip(np.asarray(fixed_theta, float), lo, hi)
            info["fixed"], nlml = True, self._nlml(theta, X, ys)
        else:
            rng = make_rng(self.seed)
            starts = []
            if init_theta is not None:
                starts.append(np.clip(np.asarray(init_theta, float), lo, hi))
            starts.append(np.clip(self._default_theta(d), lo, hi))
            for _ in range(self.n_restarts):
                starts.append(rng.uniform(lo, hi))
            best = None
            for s in starts:
                res = optimize.minimize(self._nlml, s, args=(X, ys), method="L-BFGS-B", bounds=bounds)
                info["success"].append(bool(res.success))
                if np.isfinite(res.fun) and res.fun < _BAD and (best is None or res.fun < best.fun):
                    best = res
            info["n_starts"] = len(starts)
            if best is None:
                warnings.warn("GPR hyper-parameter optimisation failed from every start; the default "
                              "hyper-parameters are used.", InferenceWarning, stacklevel=2)
                theta, nlml, info["fallback"] = np.clip(self._default_theta(d), lo, hi), float("nan"), True
            else:
                theta, nlml = best.x, float(best.fun)
        self.theta = np.asarray(theta, float)
        ls, sf2, nv = self._unpack(self.theta, d)
        self._ls, self._sf2, self._nv = ls, sf2, nv
        K = kernel_matrix(X, X, ls, sf2, self.kernel) + nv * np.eye(n)
        L, jit = _chol(K)
        if L is None:
            raise DesignError("covariance matrix is not positive definite even with jitter.")
        self._L, self.jitter = L, float(jit)
        self._alpha = linalg.cho_solve((L, True), ys)
        info["nlml"] = nlml
        info["jitter"] = self.jitter
        self.fit_info = info
        self._fitted = True
        return self

    # ----------------------------------------------------------- properties
    def _chk(self):
        if not self._fitted:
            raise DesignError("call fit() first.")

    @property
    def lengthscales(self):
        self._chk(); return self._ls.copy()

    @property
    def signal_var(self) -> float:
        self._chk(); return self._sf2 * self.y_std ** 2

    @property
    def noise_var(self) -> float:
        """Observation-noise variance in the ORIGINAL units of y."""
        self._chk(); return self._nv * self.y_std ** 2

    @property
    def data_variance(self) -> float:
        """Variance of the training responses (the normaliser of ADR-002); 0 if y is constant."""
        self._chk(); return 0.0 if self.constant_response else self.y_std ** 2

    def hyperparameters(self) -> dict:
        self._chk()
        return {"kernel": self.kernel, "lengthscales": self._ls.tolist(), "signal_var": self.signal_var,
                "noise_var": self.noise_var, "theta": self.theta.tolist()}

    # ------------------------------------------------------------ prediction
    def predict(self, Xs, return_std: bool = True, include_noise: bool = False):
        """Posterior mean and standard deviation (original units) at coded points Xs."""
        self._chk()
        Xs = np.atleast_2d(np.asarray(Xs, float))
        Ks = kernel_matrix(self.X, Xs, self._ls, self._sf2, self.kernel)
        mean = Ks.T @ self._alpha * self.y_std + self.y_mean
        if not return_std:
            return mean
        v = linalg.solve_triangular(self._L, Ks, lower=True)
        var = self._sf2 - np.sum(v ** 2, axis=0) + (self._nv if include_noise else 0.0)
        return mean, np.sqrt(np.maximum(var, 0.0) * self.y_std ** 2)

    def posterior_cov(self, Xa, Xb=None) -> np.ndarray:
        """Latent posterior covariance (original units); depends on inputs and hyper-parameters only."""
        self._chk()
        Xa = np.atleast_2d(np.asarray(Xa, float))
        Ka = kernel_matrix(self.X, Xa, self._ls, self._sf2, self.kernel)
        va = linalg.solve_triangular(self._L, Ka, lower=True)
        if Xb is None:
            return (kernel_matrix(Xa, Xa, self._ls, self._sf2, self.kernel) - va.T @ va) * self.y_std ** 2
        Xb = np.atleast_2d(np.asarray(Xb, float))
        Kb = kernel_matrix(self.X, Xb, self._ls, self._sf2, self.kernel)
        vb = linalg.solve_triangular(self._L, Kb, lower=True)
        return (kernel_matrix(Xa, Xb, self._ls, self._sf2, self.kernel) - va.T @ vb) * self.y_std ** 2
