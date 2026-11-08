"""Multi-response stopping rule (ADR-009, research_methodology §6.2).

Per round, per response m:  R2_m (pooled cross-validated predictive R^2, see cv.py) must satisfy
    R2_m >= tau_m                         (every response, individually)      AND, if requested,
    R2_agg >= tau_agg                     R2_agg = mean_m R2_m   or   sum_m w_m R2_m  (w >= 0, sum w = 1)
The aggregate NEVER replaces the individual thresholds.  A response whose R^2 is undefined (constant
response, too few samples, non-finite predictions) counts as "not met" and the reason is reported.
Negative R^2 values are kept as they are.

Stopping (evaluated after every round):
    n_runs >= N_min  AND  the last `consecutive` rounds all met    ->  'target_reached'
    otherwise  n_runs >= N_max                                     ->  'budget_exhausted' (target NOT met)
    otherwise                                                      ->  'continue'
Defaults (ADR-009): tau = 0.95, N_min = k(k-1)/2, consecutive = 3.  N_max has NO default.
Comparisons use an absolute tolerance of 1e-12 so that e.g. mean(0.96, 0.95, 0.94) = 0.95 counts as 0.95.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..exceptions import DesignError

TOL = 1e-12


@dataclass
class RoundCheck:
    per_response: dict
    aggregate: float | None
    aggregate_threshold: float | None
    aggregate_method: str
    aggregate_met: bool | None
    met: bool
    unmet: list = field(default_factory=list)


@dataclass
class StopDecision:
    stop: bool
    reason: str                 # 'target_reached' | 'budget_exhausted' | 'continue'
    message: str
    target_met: bool
    n_runs: int
    n_min: int
    n_max: int
    streak: int
    consecutive: int
    unmet: list = field(default_factory=list)


def validate_weights(weights, names) -> dict:
    """Weights must cover exactly `names`, be non-negative and sum to 1."""
    if weights is None:
        raise DesignError("weighted_mean needs `weights` (non-negative, summing to 1).")
    w = {str(k): float(v) for k, v in dict(weights).items()}
    if set(w) != set(names):
        raise DesignError(f"weights must be given for exactly these responses: {sorted(names)}.")
    if any((not np.isfinite(v)) or v < 0 for v in w.values()):
        raise DesignError("weights must be finite and non-negative.")
    if abs(sum(w.values()) - 1.0) > 1e-9:
        raise DesignError(f"weights must sum to 1 (got {sum(w.values())}).")
    return w


def _unpack(v):
    if hasattr(v, "value") and hasattr(v, "status"):
        return float(v.value), v.status, getattr(v, "reason", "")
    x = float(v)
    return (x, "ok", "") if np.isfinite(x) else (x, "nonfinite_prediction", "R^2 is not finite")


def check_round(r2: dict, *, response_min_r2=0.95, aggregate_r2=None, aggregate_method="mean",
                weights=None) -> RoundCheck:
    names = list(r2)
    if not names:
        raise DesignError("no responses given.")
    if aggregate_method not in ("mean", "weighted_mean"):
        raise DesignError("aggregate_method must be 'mean' or 'weighted_mean'.")
    if isinstance(response_min_r2, dict):
        if set(response_min_r2) != set(names):
            raise DesignError(f"response_min_r2 must name exactly these responses: {sorted(names)}.")
        thr = {k: float(v) for k, v in response_min_r2.items()}
    else:
        thr = {k: float(response_min_r2) for k in names}
    per, unmet = {}, []
    for k in names:
        val, status, reason = _unpack(r2[k])
        ok = status == "ok" and np.isfinite(val)
        met = bool(ok and val >= thr[k] - TOL)
        per[k] = {"r2": val if ok else float("nan"), "status": status, "threshold": thr[k], "met": met,
                  "reason": reason}
        if not ok:
            unmet.append(f"{k}: R^2 undefined ({status}: {reason})")
        elif not met:
            unmet.append(f"{k}: R^2 = {val:.4f} < {thr[k]:g}")
    agg, agg_met = None, None
    if aggregate_r2 is not None:
        vals = [per[k]["r2"] for k in names]
        if not all(np.isfinite(vals)):
            agg, agg_met = float("nan"), False
            unmet.append("aggregate R^2 undefined because at least one response R^2 is undefined")
        else:
            if aggregate_method == "mean":
                agg = float(np.mean(vals))
            else:
                w = validate_weights(weights, names)
                agg = float(sum(w[k] * per[k]["r2"] for k in names))
            agg_met = bool(agg >= aggregate_r2 - TOL)
            if not agg_met:
                unmet.append(f"aggregate ({aggregate_method}) R^2 = {agg:.4f} < {aggregate_r2:g}")
    met = all(p["met"] for p in per.values()) and (agg_met is None or agg_met)
    return RoundCheck(per, agg, aggregate_r2, aggregate_method, agg_met, bool(met), unmet)


class MultiResponseStopping:
    def __init__(self, n_max: int, response_min_r2=0.95, aggregate_r2=None, aggregate_method="mean",
                 weights=None, consecutive: int = 3, n_min: int | None = None, batch_size: int = 1):
        for nm, v, lo in (("n_max", n_max, 1), ("consecutive", consecutive, 1), ("batch_size", batch_size, 1)):
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < lo:
                raise DesignError(f"{nm} must be an integer >= {lo}.")
        if n_min is not None and (isinstance(n_min, bool) or not isinstance(n_min, (int, np.integer)) or n_min < 0):
            raise DesignError("n_min must be a non-negative integer or None.")
        if aggregate_method not in ("mean", "weighted_mean"):
            raise DesignError("aggregate_method must be 'mean' or 'weighted_mean'.")
        if aggregate_method == "weighted_mean" and aggregate_r2 is not None and weights is None:
            raise DesignError("weighted_mean needs `weights`.")
        ts = list(response_min_r2.values()) if isinstance(response_min_r2, dict) else [response_min_r2]
        for t in ts + ([aggregate_r2] if aggregate_r2 is not None else []):
            if not np.isfinite(t) or t > 1.0:
                raise DesignError("R^2 thresholds must be finite and <= 1.")
        self.n_max, self.consecutive, self.batch_size = int(n_max), int(consecutive), int(batch_size)
        self.n_min = None if n_min is None else int(n_min)
        self.response_min_r2, self.aggregate_r2 = response_min_r2, aggregate_r2
        self.aggregate_method, self.weights = aggregate_method, weights

    def n_min_for(self, k: int) -> int:
        return self.n_min if self.n_min is not None else k * (k - 1) // 2

    def remaining(self, n_runs: int) -> int:
        return max(self.n_max - int(n_runs), 0)

    def next_batch_size(self, n_runs: int) -> int:
        return min(self.batch_size, self.remaining(n_runs))

    def check(self, r2: dict) -> RoundCheck:
        return check_round(r2, response_min_r2=self.response_min_r2, aggregate_r2=self.aggregate_r2,
                           aggregate_method=self.aggregate_method, weights=self.weights)

    def evaluate(self, checks: list, n_runs: int, k: int) -> StopDecision:
        n_min = self.n_min_for(k)
        if n_min > self.n_max:
            raise DesignError(f"N_min = {n_min} exceeds N_max = {self.n_max}: the target could never be "
                              f"declared reached. Raise n_max or lower n_min.")
        if not checks:
            raise DesignError("evaluate() needs at least one RoundCheck.")
        streak = 0
        for c in reversed(checks):
            if not c.met:
                break
            streak += 1
        unmet = list(checks[-1].unmet)
        base = dict(n_runs=int(n_runs), n_min=n_min, n_max=self.n_max, streak=streak,
                    consecutive=self.consecutive, unmet=unmet)
        if n_runs >= n_min and streak >= self.consecutive:
            return StopDecision(True, "target_reached", f"已達標：連續 {streak} 輪所有響應都滿足 R² 門檻。", True, **base)
        if n_runs >= self.n_max:
            return StopDecision(True, "budget_exhausted", "已達最大 run 數但目標「未達成」，不可視為成功。", False, **base)
        why = []
        if n_runs < n_min:
            why.append(f"run 數 {n_runs} < N_min {n_min}")
        if streak < self.consecutive:
            why.append(f"連續達標 {streak}/{self.consecutive} 輪")
        return StopDecision(False, "continue", "繼續：" + "；".join(why), False, **base)
