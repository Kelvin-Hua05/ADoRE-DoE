"""Candidate points for Active Learning: generated, never evaluated, never simulated.

Wraps design.generate_candidates and adds (i) exclusion of points that already exist (completed or
pending runs) and (ii) a minimum-distance rule (Euclidean, coded units) to existing points.
"""
from __future__ import annotations

import numpy as np

from ..design.candidates import generate_candidates
from ..exceptions import DesignError


def distances_to(P, others) -> np.ndarray:
    P, O = np.atleast_2d(P), np.atleast_2d(others)
    if len(O) == 0:
        return np.full(len(P), np.inf)
    return np.sqrt(((P[:, None, :] - O[None, :, :]) ** 2).sum(axis=2)).min(axis=1)


def build_candidates(region, n: int = 500, method: str = "sobol", seed=None, existing=None,
                     min_distance: float = 0.0, points=None):
    """Return (candidate_array, info).  `existing`: coded runs already used."""
    if min_distance < 0:
        raise DesignError("min_distance must be >= 0.")
    ex = None if existing is None or len(np.atleast_2d(existing)) == 0 else np.atleast_2d(np.asarray(existing, float))
    cs = generate_candidates(region, "user" if points is not None else method, n=n, seed=seed,
                             points=points, exclude=ex)
    P, removed_md = cs.points, 0
    if min_distance > 0 and ex is not None:
        keep = distances_to(P, ex) >= min_distance
        removed_md = int((~keep).sum())
        P = P[keep]
        if len(P) == 0:
            raise DesignError("no candidate satisfies the minimum-distance rule; lower min_distance.")
    info = {"method": cs.method, "n_generated": len(cs.points) + cs.n_removed_existing,
            "n_removed_existing": cs.n_removed_existing, "n_removed_min_distance": removed_md,
            "n_candidates": len(P), "acceptance_rate": cs.acceptance_rate, "seed": cs.seed,
            "min_distance": float(min_distance)}
    return P, info
