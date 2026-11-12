"""OLS fit of a polynomial RSM model in coded factors, with the statistics of a standard RSM report.

    y = X beta + e,  beta_hat = argmin ||y - X beta||^2 (QR),  sigma^2 = RSS / (n - p)
    Cov(beta_hat) = sigma^2 (X'X)^-1,  t = b / SE,  CI = b +- t_{1-alpha/2, n-p} SE
Fit statistics: Std.Dev = sqrt(MSE); Mean = mean(y); C.V.% = 100 Std.Dev / Mean;
  R^2 = 1 - RSS/SST;  adj R^2 = 1 - (RSS/(n-p)) / (SST/(n-1));
  PRESS = sum (e_i / (1 - h_ii))^2  (exact leave-one-out);  predicted R^2 = 1 - PRESS/SST;
  Adeq. Precision = (max y_hat - min y_hat) / sqrt(p sigma^2 / n)  over the design points;
  VIF_j = diag(R^-1) of the correlation matrix R of the non-constant columns.
Information criteria (up to an additive constant common to all models of the same data):
  AIC = n ln(RSS/n) + 2p,  AICc = AIC + 2p(p+1)/(n-p-1),  BIC = n ln(RSS/n) + p ln n;
  `aic_sm` / `bic_sm` add the constant n(1 + ln 2pi), i.e. equal -2 loglik + 2p (+ p ln n form) as printed by statsmodels.
Not identifiable models raise RankDeficientError naming the aliased terms (no pseudo-inverse).
With n - p = 0 residual df, SE / t / p / CI and the predicted R^2 are NaN and a warning is issued.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import linalg, stats

from ..core.factor import Factor
from ..exceptions import InferenceWarning, RankDeficientError
from . import terms as T


def aliased_columns(X, names, tol: float = 1e-9):
    """Columns that are (numerically) linear combinations of the others, via pivoted QR."""
    _, R, piv = linalg.qr(X, pivoting=True, mode="economic")
    d = np.abs(np.diag(R))
    rank = int(np.sum(d > tol * max(X.shape) * (d[0] if d.size and d[0] > 0 else 1.0)))
    return [names[j] for j in sorted(piv[rank:])], rank


def info_criteria(rss: float, n: int, p: int) -> dict:
    rss = max(float(rss), 1e-300)
    base = n * np.log(rss / n)
    aic = base + 2 * p
    aicc = aic + 2 * p * (p + 1) / (n - p - 1) if n - p - 1 > 0 else float("inf")
    const = n * (1 + np.log(2 * np.pi))
    return {"aic": float(aic), "aicc": float(aicc), "bic": float(base + p * np.log(n)),
            "aic_sm": float(aic + const), "bic_sm": float(base + p * np.log(n) + const)}


class RSMResult:
    def __init__(self, Z, y, terms, factor_names, factors=None, run_order=None, alpha=0.05, response="y"):
        self.Z = np.atleast_2d(np.asarray(Z, float))
        self.y = np.asarray(y, float).ravel()
        self.n, self.k = self.Z.shape
        self.terms = [tuple(int(v) for v in t) for t in terms]
        self.factor_names = list(factor_names)
        self.factors = factors
        self.run_order = None if run_order is None else np.asarray(run_order)
        self.alpha, self.response = alpha, response
        self.selection = None
        self.term_names = [T.term_name(t, self.factor_names) for t in self.terms]
        self._fit()

    # ------------------------------------------------------------------ fit
    def _fit(self):
        X = T.expand(self.Z, self.terms)
        n, p = X.shape
        if n < p:
            raise RankDeficientError(f"{n} runs < {p} model terms ({', '.join(self.term_names)}): not identifiable.")
        bad, rank = aliased_columns(X, self.term_names)
        if rank < p:
            raise RankDeficientError(f"model is not identifiable: rank {rank} < {p} terms; aliased / "
                                     f"non-estimable term(s): {bad}.")
        Q, R = np.linalg.qr(X)
        self.X, self.p = X, p
        self.beta = linalg.solve_triangular(R, Q.T @ self.y)
        Rinv = linalg.solve_triangular(R, np.eye(p))
        self.XtXinv = Rinv @ Rinv.T
        self.fitted = X @ self.beta
        self.resid = self.y - self.fitted
        self.rss = float(self.resid @ self.resid)
        self.leverage = np.sum(Q ** 2, axis=1)
        self.df_resid = n - p
        ybar = float(self.y.mean())
        self.sst = float(np.sum((self.y - ybar) ** 2))
        self.mean_y = ybar
        self.ic = info_criteria(self.rss, n, p)
        if self.df_resid >= 1:
            self.sigma2 = self.rss / self.df_resid
            self.sigma = float(np.sqrt(self.sigma2))
            self.se = np.sqrt(np.diag(self.XtXinv) * self.sigma2)
            with np.errstate(divide="ignore", invalid="ignore"):
                self.t = self.beta / self.se
            self.pvalue = 2 * stats.t.sf(np.abs(self.t), self.df_resid)
            tc = stats.t.ppf(1 - self.alpha / 2, self.df_resid)
            self.ci = np.column_stack([self.beta - tc * self.se, self.beta + tc * self.se])
        else:
            warnings.warn("No residual degrees of freedom: SE, t, p, CI and predicted R^2 are not available.",
                          InferenceWarning, stacklevel=3)
            self.sigma2 = self.sigma = float("nan")
            self.se = self.t = self.pvalue = np.full(p, np.nan)
            self.ci = np.full((p, 2), np.nan)
        self.vif = self._vif()

    def _vif(self):
        v = np.full(self.p, np.nan)
        idx = [j for j, t in enumerate(self.terms) if sum(t) > 0]
        if len(idx) == 1:
            v[idx[0]] = 1.0
        elif len(idx) > 1:
            C = np.corrcoef(self.X[:, idx], rowvar=False)
            if np.all(np.isfinite(C)) and np.linalg.matrix_rank(C) == len(idx):
                v[idx] = np.diag(np.linalg.inv(C))
        return v

    # ----------------------------------------------------------- statistics
    @property
    def r2(self) -> float:
        return float(1 - self.rss / self.sst) if self.sst > 0 else float("nan")

    @property
    def r2_adj(self) -> float:
        if self.sst <= 0 or self.df_resid < 1 or self.n < 2:
            return float("nan")
        return float(1 - (self.rss / self.df_resid) / (self.sst / (self.n - 1)))

    @property
    def press(self) -> float:
        if np.any(self.leverage >= 1 - 1e-10):
            return float("inf")
        return float(np.sum((self.resid / (1 - self.leverage)) ** 2))

    @property
    def r2_pred(self) -> float:
        if self.sst <= 0 or not np.isfinite(self.press):
            return float("nan")
        return float(1 - self.press / self.sst)

    @property
    def adeq_precision(self) -> float:
        if not np.isfinite(self.sigma2) or self.sigma2 <= 0:
            return float("nan")
        return float((self.fitted.max() - self.fitted.min()) / np.sqrt(self.p * self.sigma2 / self.n))

    def fit_statistics(self) -> pd.Series:
        cv = 100 * self.sigma / self.mean_y if self.mean_y else float("nan")
        return pd.Series({"Std. Dev.": self.sigma, "Mean": self.mean_y, "C.V. %": cv, "R²": self.r2,
                          "Adjusted R²": self.r2_adj, "Predicted R²": self.r2_pred,
                          "Adeq Precision": self.adeq_precision, "PRESS": self.press,
                          "n": self.n, "p": self.p, "df residual": self.df_resid,
                          "AIC": self.ic["aic"], "AICc": self.ic["aicc"], "BIC": self.ic["bic"]})

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame({"Term": self.term_names, "Coefficient": self.beta, "df": 1, "StdErr": self.se,
                             f"CI_low": self.ci[:, 0], f"CI_high": self.ci[:, 1], "t": self.t,
                             "p": self.pvalue, "VIF": self.vif})

    # ----------------------------------------------------------- prediction
    def predict(self, Z, return_se: bool = False):
        F = T.expand(Z, self.terms)
        mean = F @ self.beta
        if not return_se:
            return mean
        se = np.sqrt(np.maximum(np.einsum("ij,ij->i", F @ self.XtXinv, F), 0.0)) * self.sigma
        return mean, se

    # ------------------------------------------------------------- equation
    def equation(self, coded: bool = True, digits: int = 6) -> str:
        """Explicit polynomial.  coded=True: in coded factors; False: in the natural (actual) units."""
        coef = {t: float(b) for t, b in zip(self.terms, self.beta)}
        names = self.factor_names
        if not coded:
            if not self.factors:
                raise ValueError("actual-unit equation needs the Factor definitions.")
            coef = natural_polynomial(coef, self.factors)
        items = sorted(coef.items(), key=lambda kv: T.order_key(kv[0]))
        out = []
        for t, b in items:
            name = T.term_name(t, names)
            s = f"{abs(b):.{digits}g}" + ("" if name == "Intercept" else f"*{name}")
            out.append(("- " if b < 0 else "+ ") + s)
        txt = " ".join(out)
        txt = txt[2:] if txt.startswith("+ ") else "-" + txt[2:] if txt.startswith("- ") else txt
        return f"{self.response} = {txt}"

    def anova(self):
        from .anova import anova_table
        return anova_table(self)

    def diagnostics(self):
        from ..diagnostics import compute_diagnostics
        return compute_diagnostics(self)

    def __repr__(self):
        return f"<RSMResult {self.response!r} n={self.n} p={self.p} R2={self.r2:.4f} R2pred={self.r2_pred:.4f}>"


def natural_polynomial(coef: dict, factors) -> dict:
    """Substitute z_j = (x_j - c_j)/h_j into a polynomial in coded z; returns {exponents: coef} in x."""
    k = len(factors)
    cs = [f.center for f in factors]
    hs = [f.half_range for f in factors]
    out: dict = {}

    def mul(a, b):
        r: dict = {}
        for ea, ca in a.items():
            for eb, cb in b.items():
                e = tuple(x + y for x, y in zip(ea, eb))
                r[e] = r.get(e, 0.0) + ca * cb
        return r

    one = {(0,) * k: 1.0}
    for t, b in coef.items():
        poly = dict(one)
        for j, pw in enumerate(t):
            lin = {tuple(1 if i == j else 0 for i in range(k)): 1.0 / hs[j], (0,) * k: -cs[j] / hs[j]}
            for _ in range(pw):
                poly = mul(poly, lin)
        for e, c in poly.items():
            out[e] = out.get(e, 0.0) + b * c
    return {e: c for e, c in out.items() if abs(c) > 0.0}


def fit_arrays(Z, y, terms, factor_names, factors=None, run_order=None, alpha=0.05, response="y",
               on_rank_deficient: str = "error") -> RSMResult:
    """Fit given terms.  on_rank_deficient='drop' removes aliased terms (recorded in `.dropped_terms`)."""
    terms = [tuple(t) for t in terms]
    dropped = []
    while True:
        try:
            res = RSMResult(Z, y, terms, factor_names, factors, run_order, alpha, response)
            res.dropped_terms = dropped
            return res
        except RankDeficientError:
            if on_rank_deficient != "drop":
                raise
            X = T.expand(Z, terms)
            names = [T.term_name(t, factor_names) for t in terms]
            bad, _ = aliased_columns(X, names)
            if not bad or len(terms) <= 1:
                raise
            drop = bad[-1]
            dropped.append(drop)
            terms = [t for t, nm in zip(terms, names) if nm != drop]
