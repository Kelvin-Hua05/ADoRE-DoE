"""Acquisition strategies and greedy batch selection (ADR-002 / ADR-003).

strategy        what it optimises                                        needs
'i_optimal'     MAIN strategy (ADR-003): classical I-optimal augmentation  existing runs + a polynomial model
                of a polynomial model -- reduction of tr[(X'X)^-1 B]; it depends on the design only, NOT on
                the responses and NOT on any GPR variance.
'ivr'           integrated GPR variance reduction over region points      fitted GPR per response
'max_variance'  largest GPR predictive variance (BASELINE)                fitted GPR per response
'random'        uniform random candidate (BASELINE)                       -

GPR strategies, several responses: every response's variance is divided by that response's data variance
(ADR-002), so a response measured in different units cannot dominate; the per-response scores are then
combined by a weighted mean (weights >= 0, sum 1; default equal) or by the maximum ('max' = worst response).

Batches are greedy and EXACT for fixed hyper-parameters: the latent GP posterior covariance does not depend on
the (yet unknown) responses, so after choosing candidate c the covariance of every response is updated by
        K <- K - K[:,c] K[c,:] / (K[c,c] + noise)
(rank-1).  The same holds for I-optimal (Sherman-Morrison).  A minimum distance (coded units) to existing
runs and to already chosen batch points prevents clustering.  Nothing here ever runs a simulation.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.seed import derive_seed, make_rng, resolve_seed
from ..design.criteria import check_design
from ..design.model import as_model
from ..design.moment import IntegrationDomain
from ..design.optimal import RIDGE, candidate_scores
from ..design.region import Region
from ..exceptions import DesignError, InferenceWarning
from .candidates import distances_to
from .stopping import validate_weights

STRATEGIES = ("i_optimal", "ivr", "max_variance", "random")


@dataclass
class Selection:
    points: np.ndarray
    indices: list
    scores: list
    strategy: str
    aggregate: str
    info: dict = field(default_factory=dict)

    def actual_df(self, factors) -> pd.DataFrame:
        return pd.DataFrame({f.name: f.from_coded(self.points[:, j]) for j, f in enumerate(factors)})

    def to_csv(self, path, factors):
        df = self.actual_df(factors)
        for j, f in enumerate(factors):
            df[f"{f.name}[coded]"] = self.points[:, j]
        df.to_csv(path, index=False)
        return path

    def __len__(self):
        return len(self.points)


def _combine(per_resp: dict, weights: dict, aggregate: str) -> np.ndarray:
    S = np.vstack([per_resp[k] for k in per_resp])
    if aggregate == "max":
        return S.max(axis=0)
    w = np.array([weights[k] for k in per_resp])[:, None]
    return (w * S).sum(axis=0)


def select_batch(candidates, batch_size: int, strategy: str, *, models: dict | None = None, existing=None,
                 region: Region | None = None, weights=None, aggregate: str = "mean",
                 min_distance: float = 0.0, n_integration: int = 512, poly_model="quadratic",
                 seed=None) -> Selection:
    if strategy not in STRATEGIES:
        raise DesignError(f"strategy must be one of {STRATEGIES}.")
    if aggregate not in ("mean", "max"):
        raise DesignError("aggregate must be 'mean' or 'max'.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise DesignError("batch_size must be an integer >= 1.")
    if min_distance < 0:
        raise DesignError("min_distance must be >= 0.")
    C = np.atleast_2d(np.asarray(candidates, float))
    Nc, k = C.shape
    if Nc == 0:
        raise DesignError("no candidate points.")
    ex = np.zeros((0, k)) if existing is None else np.atleast_2d(np.asarray(existing, float))
    if ex.shape[1] != k and len(ex):
        raise DesignError("existing runs and candidates have different numbers of factors.")
    root = resolve_seed(seed)
    region = region or Region.unit_cube(k)
    allowed = np.ones(Nc, bool)
    if min_distance > 0 and len(ex):
        allowed &= distances_to(C, ex) >= min_distance
    info = {"strategy": strategy, "n_candidates": Nc, "n_allowed_initial": int(allowed.sum()),
            "min_distance": float(min_distance), "seed": root, "truncated": False}
    chosen, scores, per_pick = [], [], []

    def next_allowed(j):
        allowed[j] = False
        if min_distance > 0:
            np.logical_and(allowed, distances_to(C, C[j][None, :]) >= min_distance, out=allowed)

    # ---------------------------------------------------------------- per-strategy state
    if strategy == "random":
        rng = make_rng(derive_seed(root, "random"))
    elif strategy == "i_optimal":
        if len(ex) == 0:
            raise DesignError("i_optimal augmentation needs the existing runs (`existing`).")
        spec = as_model(poly_model, k)
        F0, Fc = spec.expand(ex), spec.expand(C)
        if check_design(F0).singular:
            warnings.warn(f"existing runs do not identify the {spec.kind} model (rank "
                          f"{check_design(F0).rank} < {spec.n_terms}); the ridge-regularised information "
                          f"matrix is used for scoring.", InferenceWarning, stacklevel=2)
        dom = IntegrationDomain(region, m=max(int(n_integration), 16), seed=derive_seed(root, "integration"))
        B = dom.matrix(spec).B
        A = F0.T @ F0 + RIDGE * np.eye(spec.n_terms)
        info.update({"poly_model": spec.kind, "responses_used": False})
    else:
        if not models:
            raise DesignError(f"strategy {strategy!r} needs fitted GPR models ({{response: GaussianProcess}}).")
        names = list(models)
        dv = {n: models[n].data_variance for n in names}
        for n in names:
            if not (dv[n] > 0 and np.isfinite(dv[n])):
                raise DesignError(f"response {n!r} is constant: its variance cannot be used for normalisation.")
        w = ({n: 1.0 / len(names) for n in names} if weights is None else validate_weights(weights, names))
        info.update({"aggregate": aggregate, "weights": w, "normalisers": dv, "responses": names})
        if strategy == "ivr":
            dom = IntegrationDomain(region, m=max(int(n_integration), 16), seed=derive_seed(root, "integration"))
            Pint, _ = dom.points(max(int(n_integration), 16))
            Z = np.vstack([C, Pint])
            info["n_integration"] = len(Pint)
        else:
            Z = C
        Kp = {n: models[n].posterior_cov(Z) for n in names}

    # ------------------------------------------------------------------ greedy loop
    for _ in range(int(batch_size)):
        if not allowed.any():
            info["truncated"] = True
            break
        if strategy == "random":
            j = int(rng.choice(np.flatnonzero(allowed)))
            s, pp = 0.0, {}
        elif strategy == "i_optimal":
            sc = candidate_scores(np.linalg.inv(A), Fc, B, "I")
            sc = np.where(allowed, sc, -np.inf)
            j = int(np.argmax(sc))
            s, pp = float(sc[j]), {}
            A = A + np.outer(Fc[j], Fc[j])
        else:
            per = {}
            for n in names:
                K = Kp[n]
                if strategy == "max_variance":
                    per[n] = np.diag(K)[:Nc] / dv[n]
                else:
                    denom = np.diag(K)[:Nc] + models[n].noise_var
                    per[n] = (K[Nc:, :Nc] ** 2).mean(axis=0) / denom / dv[n]
            agg = _combine(per, w, aggregate)
            agg = np.where(allowed, agg, -np.inf)
            j = int(np.argmax(agg))
            s, pp = float(agg[j]), {n: float(per[n][j]) for n in names}
            for n in names:                                    # rank-1 posterior update (response-independent)
                K = Kp[n]
                kc = K[:, j].copy()
                Kp[n] = K - np.outer(kc, kc) / (kc[j] + models[n].noise_var)
        chosen.append(j)
        scores.append(s)
        per_pick.append(pp)
        next_allowed(j)
    info["per_pick_scores"] = per_pick
    if len(chosen) < batch_size:
        info["truncated"] = True
    return Selection(C[chosen] if chosen else np.zeros((0, k)), chosen, scores, strategy, aggregate, info)
