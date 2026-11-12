"""Residual and influence diagnostics of an OLS model (formulas, h_ii = diag of the hat matrix):

  internally studentised   r_i = e_i / (sigma sqrt(1 - h_ii))
  externally studentised   t_i = r_i sqrt((n - p - 1) / (n - p - r_i^2))   (leave-one-out sigma)
  Cook's distance          D_i = r_i^2 h_ii / (p (1 - h_ii))
  DFFITS                   t_i sqrt(h_ii / (1 - h_ii))
Flags: outlier if the Bonferroni-adjusted two-sided p-value of t_i (df = n - p - 1) is < alpha;
high leverage if h_ii > 2p/n; influential if D_i > 1.  Normality: Shapiro-Wilk on t_i.  A rough check of
constant variance: Spearman correlation between |residual| and the fitted value.
Points with h_ii = 1 (or no residual df) give NaN with the reason recorded -- never a made-up number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DiagnosticsResult:
    table: pd.DataFrame
    normality: dict
    variance_trend: dict
    notes: list = field(default_factory=list)
    _res: object = field(default=None, repr=False)

    def summary(self) -> str:
        t = self.table
        lines = [f"n={len(t)}  outliers={int(t.Outlier.sum())}  high leverage={int(t.HighLeverage.sum())}  "
                 f"influential={int(t.Influential.sum())}",
                 f"Shapiro-Wilk on externally studentised residuals: W={self.normality['W']:.4f}, "
                 f"p={self.normality['p']:.4f}" if np.isfinite(self.normality["p"]) else
                 f"normality test not available: {self.normality['reason']}"]
        lines += [f"NOTE: {n}" for n in self.notes]
        return "\n".join(lines)

    # plots (matplotlib is an optional extra)
    def plot_normal(self, **kw):
        from . import plots
        return plots.normal_plot(self.table["ExtStud"].to_numpy(), **kw)

    def plot_residual_vs_predicted(self, **kw):
        from . import plots
        return plots.scatter(self.table["Fitted"], self.table["ExtStud"], "Predicted",
                             "Externally studentised residuals", hline=0.0, **kw)

    def plot_residual_vs_run(self, **kw):
        from . import plots
        if self._res is None or self._res.run_order is None:
            raise ValueError("run order is not available for this model.")
        return plots.scatter(self._res.run_order, self.table["ExtStud"], "Run order",
                             "Externally studentised residuals", hline=0.0, **kw)

    def plot_predicted_vs_actual(self, **kw):
        from . import plots
        return plots.scatter(self.table["Actual"], self.table["Fitted"], "Actual", "Predicted", identity=True, **kw)

    def plot_residual_vs_factor(self, factor: str, **kw):
        from . import plots
        r = self._res
        if factor not in r.factor_names:
            raise ValueError(f"unknown factor {factor!r}.")
        return plots.scatter(r.Z[:, r.factor_names.index(factor)], self.table["ExtStud"], f"{factor} (coded)",
                             "Externally studentised residuals", hline=0.0, **kw)


def compute_diagnostics(res, alpha: float = 0.05) -> DiagnosticsResult:
    n, p, e, h = res.n, res.p, res.resid, res.leverage
    notes = []
    nan = np.full(n, np.nan)
    if res.df_resid < 1:
        notes.append("no residual degrees of freedom: studentised residuals, Cook's D and DFFITS are not available.")
        r_int = r_ext = cook = dff = nan
    else:
        ok = h < 1 - 1e-10
        if not ok.all():
            notes.append(f"{int((~ok).sum())} point(s) have leverage 1: their studentised residuals are undefined.")
        with np.errstate(divide="ignore", invalid="ignore"):
            r_int = np.where(ok, e / (res.sigma * np.sqrt(1 - h)), np.nan)
            if n - p - 1 >= 1:
                r_ext = r_int * np.sqrt((n - p - 1) / (n - p - r_int ** 2))
            else:
                r_ext = nan
                notes.append("n - p - 1 < 1: externally studentised residuals are not available.")
            cook = np.where(ok, r_int ** 2 * h / (p * (1 - h)), np.nan)
            dff = np.where(ok, r_ext * np.sqrt(h / (1 - h)), np.nan)
    if np.all(np.isfinite(r_ext)):
        pv = np.minimum(2 * stats.t.sf(np.abs(r_ext), n - p - 1) * n, 1.0)
    else:
        pv = np.where(np.isfinite(r_ext), np.minimum(2 * stats.t.sf(np.abs(np.nan_to_num(r_ext)), max(n - p - 1, 1)) * n, 1.0), np.nan)
    tab = pd.DataFrame({
        "Run": res.run_order if res.run_order is not None else np.arange(1, n + 1),
        "Actual": res.y, "Fitted": res.fitted, "Residual": e, "Leverage": h, "IntStud": r_int, "ExtStud": r_ext,
        "CooksD": cook, "DFFITS": dff,
        "Outlier": np.where(np.isfinite(pv), pv < alpha, False),
        "HighLeverage": h > 2 * p / n, "Influential": np.where(np.isfinite(cook), cook > 1.0, False)})
    v = r_ext[np.isfinite(r_ext)]
    if len(v) >= 3 and np.ptp(v) > 0:
        W, pw = stats.shapiro(v)
        norm = {"W": float(W), "p": float(pw), "reason": ""}
    else:
        norm = {"W": float("nan"), "p": float("nan"), "reason": "needs >= 3 finite, non-constant residuals"}
    if n >= 4 and np.ptp(res.fitted) > 0 and np.ptp(np.abs(e)) > 0:
        rho, pr = stats.spearmanr(res.fitted, np.abs(e))
        var = {"rho": float(rho), "p": float(pr)}
    else:
        var = {"rho": float("nan"), "p": float("nan")}
    return DiagnosticsResult(tab, norm, var, notes, res)
