"""Fixed evaluation set (research_methodology §5.3).

A set of points covering the design region (Sobol / LHS, constraints respected), generated ONCE from a
seed and independent of the training data.  Two situations are kept apart:
* truth available (simulator / analytic function / pre-collected pool, i.e. benchmarks): the true
  RMSE / MAE / R^2 on the evaluation set are reported;
* truth NOT available (real experiments): only the model's mean predictive variance is reported --
  evaluation points get no invented responses, and the CV estimate is a separate quantity.
`check_disjoint` guarantees that no training point is also an evaluation point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.seed import derive_seed, resolve_seed
from ..design.region import Region
from ..exceptions import DesignError


@dataclass
class EvalReport:
    per_response: dict
    has_truth: bool
    n_points: int

    @property
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.per_response).T

    @property
    def mean_r2(self) -> float:
        v = [d["r2"] for d in self.per_response.values() if d["r2"] is not None and np.isfinite(d["r2"])]
        return float(np.mean(v)) if v and len(v) == len(self.per_response) else float("nan")


@dataclass
class EvaluationSet:
    points: np.ndarray
    seed: int
    acceptance_rate: float = 1.0
    truth: dict | None = field(default=None, repr=False)

    @classmethod
    def build(cls, factors, m: int = 1000, *, constraints=None, region=None, seed=None, method="sobol"):
        if m < 10:
            raise DesignError("an evaluation set needs at least 10 points.")
        region = region or Region.from_factors(factors, constraints)
        s = resolve_seed(seed)
        P, rate = region.sample(m, seed=derive_seed(s, "evalset"), method=method)
        return cls(P, s, float(rate))

    def with_truth(self, fn) -> "EvaluationSet":
        """Attach true responses computed ONCE by fn(points) -> {response: values}."""
        t = fn(self.points)
        truth = {}
        for k, v in t.items():
            v = np.asarray(v, float).ravel()
            if v.shape != (len(self.points),) or not np.all(np.isfinite(v)):
                raise DesignError(f"truth for {k!r} must be finite with one value per evaluation point.")
            truth[k] = v
        return EvaluationSet(self.points, self.seed, self.acceptance_rate, truth)

    def check_disjoint(self, train_coded, tol: float = 1e-9) -> None:
        T = np.atleast_2d(np.asarray(train_coded, float))
        if len(T) == 0:
            return
        d = np.max(np.abs(self.points[:, None, :] - T[None, :, :]), axis=2).min(axis=1)
        n = int((d <= tol).sum())
        if n:
            raise DesignError(f"{n} evaluation point(s) coincide with training points: the evaluation set "
                              f"must be independent of the training data.")

    def evaluate(self, models: dict) -> EvalReport:
        """models: {response: object with predict(X) -> (mean, std)}."""
        out = {}
        for name, mdl in models.items():
            mean, std = mdl.predict(self.points)
            mpv = float(np.mean(std ** 2))
            dv = getattr(mdl, "data_variance", None)
            rec = {"mean_pred_var": mpv,
                   "mean_pred_var_normalised": mpv / dv if dv and dv > 0 else None,
                   "rmse": None, "mae": None, "r2": None, "r2_status": "no_truth"}
            if self.truth is not None and name in self.truth:
                t = self.truth[name]
                e = t - mean
                sst = float(np.sum((t - t.mean()) ** 2))
                rec["rmse"], rec["mae"] = float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))
                if np.ptp(t) <= 1e-12 * float(np.max(np.abs(t))):          # constant relative to its magnitude
                    rec["r2"], rec["r2_status"] = float("nan"), "sst_zero"
                else:
                    rec["r2"], rec["r2_status"] = 1.0 - float(np.sum(e ** 2)) / sst, "ok"      # may be negative
            out[name] = rec
        return EvalReport(out, self.truth is not None, len(self.points))
