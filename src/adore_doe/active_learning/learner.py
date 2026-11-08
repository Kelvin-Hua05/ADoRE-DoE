"""ActiveLearner - Module 1 loop (research_methodology §5.1).

    assess()     fit one GPR per response -> cross-validated predictive R^2 per response (leak-free, see cv.py)
                 -> stopping rule (ADR-009) -> (optional) metrics on the fixed evaluation set
    recommend()  candidates (excluding existing runs, min-distance) -> acquisition -> a Selection
                 (NOTHING is simulated; the dataset is not changed)
    update()     the user supplies the simulated responses for the recommended (or any feasible) points;
                 they are appended to the dataset with source 'AL-iter-<round>'
    run(sim)     convenience loop for benchmarks / synthetic functions

Rules enforced here: one `assess()` per new data state (calling it again on unchanged data does NOT create a
new round, so the 'consecutive rounds' counter cannot be inflated); every training run must have all responses;
the run budget N_max is never exceeded (the last batch is trimmed); a stop with `budget_exhausted` is never
reported as success.  This module depends on core/design only -- never on rsm or residual (ADR-005).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.dataset import ExperimentDataset
from ..core.seed import SeedManager, derive_seed
from ..design.region import Region
from ..exceptions import DesignError
from .acquisition import STRATEGIES, Selection, select_batch
from .candidates import build_candidates
from .cv import gpr_cv_r2
from .evaluation import EvaluationSet
from .gpr import GaussianProcess
from .stopping import MultiResponseStopping, StopDecision

_CV_DEFAULT = {"scheme": "kfold", "n_splits": 5, "repeats": 1, "refit_hyperparameters": True}


class ActiveLearner:
    def __init__(self, dataset: ExperimentDataset, responses, stopping: MultiResponseStopping, *,
                 strategy: str = "ivr", batch_size: int | None = None, constraints=None, region=None,
                 kernel: str = "matern52", cv: dict | None = None, n_candidates: int = 500,
                 candidate_method: str = "sobol", n_integration: int = 512, min_distance: float = 0.0,
                 weights=None, aggregate: str = "mean", reestimate_hyperparameters: bool = True,
                 n_restarts: int = 3, eval_set: EvaluationSet | None = None, poly_model="quadratic", seed=None):
        if strategy not in STRATEGIES:
            raise DesignError(f"strategy must be one of {STRATEGIES}.")
        self.responses = list(responses)
        if not self.responses or len(set(self.responses)) != len(self.responses):
            raise DesignError("give a non-empty list of distinct response names.")
        miss = [r for r in self.responses if r not in dataset.response_names]
        if miss:
            raise DesignError(f"dataset has no response(s) {miss}.")
        if not isinstance(stopping, MultiResponseStopping):
            raise DesignError("stopping must be a MultiResponseStopping.")
        self.ds, self.stopping, self.strategy = dataset, stopping, strategy
        self.batch_size = int(batch_size or stopping.batch_size)
        self.region = region or Region.from_factors(dataset.factors, constraints)
        self.kernel, self.n_candidates, self.candidate_method = kernel, int(n_candidates), candidate_method
        self.n_integration, self.min_distance = int(n_integration), float(min_distance)
        self.weights, self.aggregate = weights, aggregate
        self.reestimate, self.n_restarts, self.eval_set, self.poly_model = bool(reestimate_hyperparameters), int(n_restarts), eval_set, poly_model
        self.cv = {**_CV_DEFAULT, **(cv or {})}
        self.sm = SeedManager(seed)
        self.round, self.history, self.checks = 0, [], []
        self.models: dict = {}
        self._theta: dict = {}
        self._last_n = None
        self._decision: StopDecision | None = None
        self.pending: Selection | None = None

    # ------------------------------------------------------------------ data
    @property
    def k(self) -> int:
        return self.ds.n_factors

    def _train_rows(self):
        mask = self.ds.training_mask
        if not self.ds.complete_mask(self.responses)[mask].all():
            raise DesignError("some runs have missing responses; fill them (update / add_response) before assess().")
        X = self.ds.coded[mask]
        return X, {r: self.ds.get_response(r)[mask] for r in self.responses}

    @property
    def n_runs(self) -> int:
        return int(self.ds.training_mask.sum())

    # ------------------------------------------------------------------- fit
    def fit(self):
        X, y = self._train_rows()
        if len(X) < 3:
            raise DesignError("need at least 3 complete runs to fit the surrogate models.")
        root = self.sm.root
        for r in self.responses:
            gp = GaussianProcess(self.kernel, n_restarts=self.n_restarts,
                                 seed=derive_seed(root, f"gpr-{r}-{self.round}"))
            prev = self._theta.get(r)
            if prev is None:
                gp.fit(X, y[r])
            elif self.reestimate:
                gp.fit(X, y[r], init_theta=prev)            # warm start from the PREVIOUS round (ADR-002)
            else:
                gp.fit(X, y[r], fixed_theta=prev)           # hyper-parameters frozen after the first fit
            self.models[r] = gp
            self._theta[r] = gp.theta
        return self.models

    # ---------------------------------------------------------------- assess
    def assess(self) -> StopDecision:
        n = self.n_runs
        if self._last_n == n and self._decision is not None:
            return self._decision                           # same data -> same round (no double counting)
        self.round += 1
        self.fit()
        X, y = self._train_rows()
        root, cv = self.sm.root, self.cv
        r2, cvi = {}, {}
        for r in self.responses:
            res, _, info = gpr_cv_r2(X, y[r], scheme=cv["scheme"], n_splits=cv["n_splits"], repeats=cv["repeats"],
                                     refit_hyperparameters=cv["refit_hyperparameters"], kernel=self.kernel,
                                     n_restarts=min(self.n_restarts, 2), seed=derive_seed(root, f"cv-{r}-{self.round}"))
            r2[r], cvi[r] = res, info
        check = self.stopping.check(r2)
        self.checks.append(check)
        decision = self.stopping.evaluate(self.checks, n, self.k)
        ev = None
        if self.eval_set is not None:
            self.eval_set.check_disjoint(X)
            ev = self.eval_set.evaluate(self.models)
        tr = self.ds.training()
        self.history.append({
            "round": self.round, "n_runs": n, "n_unique_points": tr.n_unique_points,
            "n_replicated_runs": tr.n_replicated_runs, "r2": {r: r2[r] for r in self.responses},
            "aggregate_r2": check.aggregate, "round_met": check.met, "unmet": list(check.unmet),
            "decision": decision, "strategy": self.strategy,
            "hyperparameters": {r: self.models[r].hyperparameters() for r in self.responses},
            "gpr_fit": {r: self.models[r].fit_info for r in self.responses}, "cv": cvi,
            "eval": ev, "recommended": None})
        self._last_n, self._decision = n, decision
        return decision

    # ------------------------------------------------------------- recommend
    def recommend(self) -> Selection:
        if self._decision is None or self._last_n != self.n_runs:
            raise DesignError("call assess() on the current data before recommend().")
        if self._decision.stop:
            raise DesignError(f"stopping criterion reached ({self._decision.reason}); no further runs are recommended.")
        bs = min(self.batch_size, self.stopping.remaining(self.n_runs))
        existing = self.ds.coded[self.ds.training_mask]
        cand, cinfo = build_candidates(self.region, self.n_candidates, self.candidate_method,
                                       seed=derive_seed(self.sm.root, f"candidates-{self.round}"),
                                       existing=existing, min_distance=self.min_distance)
        sel = select_batch(cand, bs, self.strategy, models=self.models, existing=existing, region=self.region,
                           weights=self.weights, aggregate=self.aggregate, min_distance=self.min_distance,
                           n_integration=self.n_integration, poly_model=self.poly_model,
                           seed=derive_seed(self.sm.root, f"select-{self.round}"))
        sel.info["candidate_set"] = cinfo
        self.pending = sel
        self.history[-1]["recommended"] = sel.points.copy()
        return sel

    # ---------------------------------------------------------------- update
    def update(self, points, values: dict) -> np.ndarray:
        P = np.atleast_2d(np.asarray(points, float))
        if P.shape[1] != self.k:
            raise DesignError(f"points must have {self.k} columns.")
        if not self.region.contains(P).all():
            raise DesignError("some points lie outside the design region.")
        if set(values) != set(self.responses):
            raise DesignError(f"values must contain exactly the responses {self.responses}.")
        vals = {}
        for r in self.responses:
            v = np.asarray(values[r], float).ravel()
            if v.shape != (len(P),) or not np.all(np.isfinite(v)):
                raise DesignError(f"values[{r!r}] must be finite with one entry per point.")
            vals[r] = v
        if self.n_runs + len(P) > self.stopping.n_max:
            raise DesignError(f"adding {len(P)} runs would exceed n_max = {self.stopping.n_max}.")
        ids = self.ds.append_runs(P, source=f"AL-iter-{self.round}", point_type="active_learning", values=vals)
        self.pending = None
        return ids

    # ------------------------------------------------------------------- run
    def run(self, simulator, max_rounds: int = 200) -> StopDecision:
        """simulator(points_coded) -> {response: values}.  Loops assess -> recommend -> update."""
        for _ in range(max_rounds):
            d = self.assess()
            if d.stop:
                return d
            sel = self.recommend()
            if len(sel) == 0:
                raise DesignError("no admissible candidate left (min_distance too large?).")
            self.update(sel.points, simulator(sel.points))
        raise DesignError(f"max_rounds={max_rounds} reached without a stopping decision.")

    # --------------------------------------------------------------- results
    def history_df(self) -> pd.DataFrame:
        rows = []
        for h in self.history:
            row = {"round": h["round"], "n_runs": h["n_runs"], "n_unique": h["n_unique_points"],
                   "n_replicated": h["n_replicated_runs"], "aggregate_r2": h["aggregate_r2"],
                   "round_met": h["round_met"], "decision": h["decision"].reason}
            for r, v in h["r2"].items():
                row[f"r2_{r}"] = v.value
                row[f"r2_status_{r}"] = v.status
                row[f"rmse_cv_{r}"] = v.rmse
            if h["eval"] is not None:
                for r, d in h["eval"].per_response.items():
                    row[f"eval_rmse_{r}"], row[f"eval_r2_{r}"] = d["rmse"], d["r2"]
                    row[f"eval_meanvar_{r}"] = d["mean_pred_var_normalised"]
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        d = self._decision
        return {"rounds": self.round, "n_runs": self.n_runs, "stopped": bool(d and d.stop),
                "reason": d.reason if d else None, "target_met": bool(d and d.target_met),
                "message": d.message if d else None, "n_unique_points": self.ds.training().n_unique_points,
                "n_replicated_runs": self.ds.training().n_replicated_runs, "seed": self.sm.root}
