"""ANOVA with Type III (partial) sums of squares and a Lack-of-Fit test.

  SS_total = sum (y - ybar)^2,  SS_model = SS_total - RSS (df = p - 1),  MS = SS/df,  F = MS / MSE
  Term j (one df): SS_j = RSS(model without j) - RSS(full)  ("partial", the order-independent SS), F_j = SS_j / MSE,
  p_j = P(F_{1, n-p} > F_j)  (equals the t-test p-value of the coefficient).
  Lack of fit needs PURE ERROR from replicated runs (identical coded settings):
      SS_PE = sum_g sum_{i in g} (y_i - ybar_g)^2,   df_PE = n - G  (G = number of distinct settings),
      SS_LOF = RSS - SS_PE,  df_LOF = (n - p) - df_PE,  F_LOF = (SS_LOF/df_LOF) / (SS_PE/df_PE).
  Without replicates (e.g. a CCD with a single centre point, ADR-010) or with df_LOF = 0 the LOF test is
  reported as NOT AVAILABLE together with the reason -- it is never faked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import terms as T


def replicate_groups(Z, decimals: int = 9) -> np.ndarray:
    keys = [tuple(np.round(r, decimals) + 0.0) for r in np.atleast_2d(Z)]
    ids: dict = {}
    return np.array([ids.setdefault(k, len(ids)) for k in keys])


def pure_error(Z, y):
    g = replicate_groups(Z)
    y = np.asarray(y, float)
    ss = 0.0
    for gid in np.unique(g):
        v = y[g == gid]
        ss += float(np.sum((v - v.mean()) ** 2))
    return ss, int(len(y) - len(np.unique(g))), int(len(np.unique(g)))


def _rss_without(res, j):
    keep = [i for i in range(res.p) if i != j]
    X = res.X[:, keep]
    beta, *_ = np.linalg.lstsq(X, res.y, rcond=None)
    r = res.y - X @ beta
    return float(r @ r)


def anova_table(res) -> pd.DataFrame:
    n, p, dfe = res.n, res.p, res.df_resid
    mse = res.rss / dfe if dfe >= 1 else np.nan
    rows = []
    ss_model = res.sst - res.rss
    df_model = p - 1

    def add(src, ss, df, note=""):
        ms = ss / df if df and df > 0 else np.nan
        F = ms / mse if np.isfinite(mse) and mse > 0 and df and df > 0 else np.nan
        pv = float(stats.f.sf(F, df, dfe)) if np.isfinite(F) else np.nan
        rows.append({"Source": src, "SS": ss, "df": df, "MS": ms, "F": F, "p": pv, "note": note})

    add("Model", ss_model, df_model)
    for j, t in enumerate(res.terms):
        if sum(t) == 0:
            continue
        add(res.term_names[j], _rss_without(res, j) - res.rss, 1)
    rows.append({"Source": "Residual", "SS": res.rss, "df": dfe, "MS": mse, "F": np.nan, "p": np.nan,
                 "note": "" if dfe >= 1 else "no residual df: F / p not available"})
    sspe, dfpe, G = pure_error(res.Z, res.y)
    dflof = dfe - dfpe
    if dfpe < 1:
        rows.append({"Source": "Lack of Fit", "SS": np.nan, "df": np.nan, "MS": np.nan, "F": np.nan, "p": np.nan,
                     "note": "NOT AVAILABLE: no replicated runs, hence no pure error (e.g. one centre point only)"})
        rows.append({"Source": "Pure Error", "SS": np.nan, "df": 0, "MS": np.nan, "F": np.nan, "p": np.nan, "note": ""})
    elif dflof < 1:
        rows.append({"Source": "Lack of Fit", "SS": np.nan, "df": dflof, "MS": np.nan, "F": np.nan, "p": np.nan,
                     "note": "NOT AVAILABLE: the model already fits every distinct design point (df_LOF = 0)"})
        rows.append({"Source": "Pure Error", "SS": sspe, "df": dfpe, "MS": sspe / dfpe, "F": np.nan, "p": np.nan, "note": ""})
    else:
        ss_lof = res.rss - sspe
        mspe = sspe / dfpe
        F = (ss_lof / dflof) / mspe if mspe > 0 else np.inf
        pv = float(stats.f.sf(F, dflof, dfpe)) if np.isfinite(F) else 0.0
        rows.append({"Source": "Lack of Fit", "SS": ss_lof, "df": dflof, "MS": ss_lof / dflof, "F": F, "p": pv,
                     "note": "significant LOF" if pv < 0.05 else ""})
        rows.append({"Source": "Pure Error", "SS": sspe, "df": dfpe, "MS": mspe, "F": np.nan, "p": np.nan, "note": ""})
    rows.append({"Source": "Cor Total", "SS": res.sst, "df": n - 1, "MS": np.nan, "F": np.nan, "p": np.nan, "note": ""})
    return pd.DataFrame(rows)
