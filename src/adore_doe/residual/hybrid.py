"""Hybrid RSM + ML residual correction (research_methodology §8):

        y_final(x) = y_RSM(x) + lambda * e_ML(x),      e_i = y_i - y_RSM^(OOF)(x_i)

* the residual targets are OUT-OF-FOLD (oof.py): the RSM, with its term selection, is refitted inside every fold;
* learners (gp / mlp / svr) are compared by the cross-validated error of the residual prediction (K-fold over
  (x_i, e_i)); `none` (no correction, lambda = 0) is always a candidate: if no learner reduces the OOF residual RMSE
  by at least `min_gain` (relative; default 5 %) the model falls back to the plain RSM;
* lambda: 'auto' (least squares of e on the OOF learner predictions, clipped to [0, 1]) or a fixed value in [0, 1];
* the final RSM is fitted on all data, the chosen learner on the OOF residuals.

Remaining optimism (documented, not hidden): the residual targets e_j were produced by RSM fold-models that had seen
other runs' responses, so the inner comparison of learners/lambda is not a fully independent estimate.  Always judge
the hybrid on INDEPENDENT validation data (`evaluate`).  The independent validation data are never used in any fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.dataset import ExperimentDataset
from ..core.seed import SeedManager, derive_seed
from ..exceptions import DesignError
from .folds import make_folds
from .learners import make_learner
from .oof import OOFResult, RSMFitter, oof_rsm


def _metrics(y, p):
    e = np.asarray(y, float) - np.asarray(p, float)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "R2": float(1 - np.sum(e ** 2) / sst) if sst > 0 else float("nan")}


@dataclass
class HybridModel:
    base: object                      # RSMResult fitted on all training data
    learner: object                   # fitted residual learner (NoneLearner when no correction)
    learner_name: str
    lam: float
    report: pd.DataFrame
    oof: OOFResult
    config: dict = field(default_factory=dict)

    def predict(self, Z):
        Z = np.atleast_2d(np.asarray(Z, float))
        return self.base.predict(Z) + self.lam * self.learner.predict(Z)

    def predict_rsm(self, Z):
        return self.base.predict(Z)

    @property
    def corrected(self) -> bool:
        return self.learner_name != "none" and self.lam > 0

    def evaluate(self, Z_val, y_val) -> pd.DataFrame:
        """Metrics of the plain RSM and of the hybrid on INDEPENDENT validation data (rows: RSM, Hybrid)."""
        y_val = np.asarray(y_val, float).ravel()
        a, b = _metrics(y_val, self.predict_rsm(Z_val)), _metrics(y_val, self.predict(Z_val))
        t = pd.DataFrame([a, b], index=["RSM", "Hybrid"])
        t.loc["Hybrid - RSM"] = t.loc["Hybrid"] - t.loc["RSM"]
        t["learner"] = [None, self.learner_name, None]
        t["lambda"] = [None, self.lam, None]
        return t


def fit_hybrid(Z, y, factor_names=None, *, fitter: RSMFitter | None = None, learners=("gp", "mlp", "svr"),
               cv: dict | None = None, inner_splits: int = 5, lam="auto", min_gain: float = 0.05,
               seed=None) -> HybridModel:
    Z, y = np.atleast_2d(np.asarray(Z, float)), np.asarray(y, float).ravel()
    if not (lam == "auto" or (isinstance(lam, (int, float)) and 0.0 <= lam <= 1.0)):
        raise DesignError("lam must be 'auto' or a number in [0, 1].")
    if min_gain < 0:
        raise DesignError("min_gain must be >= 0.")
    cvc = {"n_splits": 5, "repeats": 3, **(cv or {})}
    sm = SeedManager(seed)
    fitter = fitter or RSMFitter()
    if factor_names is not None and fitter.factor_names is None:
        fitter = RSMFitter(fitter.model, fitter.selection, fitter.alpha, fitter.on_rank_deficient, list(factor_names))
    oof = oof_rsm(fitter, Z, y, cvc["n_splits"], cvc["repeats"], sm.child_seed("oof"))
    e = oof.residual
    base = fitter.fit(Z, y)
    rmse0 = float(np.sqrt(np.mean(e ** 2)))
    inner = make_folds(len(y), inner_splits, 1, sm.child_seed("inner"))[0]
    rows, cache = [], {}
    for name in learners:
        eh = np.full(len(y), np.nan)
        for tr, te in inner:
            m = make_learner(name).fit(Z[tr], e[tr], derive_seed(sm.root, f"fold-{name}"))
            eh[te] = m.predict(Z[te])
        rmse1 = float(np.sqrt(np.mean((e - eh) ** 2)))
        denom = float(eh @ eh)
        lam_opt = float(np.clip((e @ eh) / denom, 0.0, 1.0)) if denom > 0 else 0.0
        lam_use = lam_opt if lam == "auto" else float(lam)
        rmse_l = float(np.sqrt(np.mean((e - lam_use * eh) ** 2)))
        rows.append({"learner": name, "oof_rmse_lambda1": rmse1, "lambda": lam_use, "oof_rmse_at_lambda": rmse_l,
                     "gain": 1 - rmse_l / rmse0 if rmse0 > 0 else 0.0,
                     "gain_lambda1": 1 - rmse1 / rmse0 if rmse0 > 0 else 0.0})
        cache[name] = lam_use
    rep = pd.DataFrame(rows)
    chosen, lam_final = "none", 0.0
    if len(rep):
        b = rep.loc[rep["gain"].idxmax()]
        if b["gain"] >= min_gain and b["lambda"] > 0:
            chosen, lam_final = str(b["learner"]), float(b["lambda"])
    rep["chosen"] = rep["learner"] == chosen
    learner = make_learner(chosen).fit(Z, e, derive_seed(sm.root, f"final-{chosen}"))
    cfg = {"cv": cvc, "inner_splits": inner_splits, "lam": lam, "min_gain": min_gain, "learners": list(learners),
           "rmse_oof_rsm": rmse0, "seed": sm.root, "seeds": sm.as_dict(), "oof": oof.info}
    return HybridModel(base, learner, chosen, lam_final, rep, oof, cfg)


def fit(dataset: ExperimentDataset, response: str, model: str = "quadratic", selection: dict | None = None,
        **kw) -> HybridModel:
    """Hybrid model of one response of an ExperimentDataset (training runs only)."""
    if response not in dataset.response_names:
        raise DesignError(f"dataset has no response {response!r}.")
    mask = dataset.training_mask & ~np.isnan(dataset.get_response(response))
    if mask.sum() < 6:
        raise DesignError("need at least 6 training runs with a measured response.")
    f = RSMFitter(model, selection, factor_names=dataset.factor_names)
    h = fit_hybrid(dataset.coded[mask], dataset.get_response(response)[mask], dataset.factor_names, fitter=f, **kw)
    h.base.factors = dataset.factors
    h.base.response = response
    return h


def fit_responses(dataset: ExperimentDataset, responses, **kw) -> dict:
    """One independent hybrid model per response (each response is modelled and judged on its own)."""
    return {r: fit(dataset, r, **kw) for r in responses}
