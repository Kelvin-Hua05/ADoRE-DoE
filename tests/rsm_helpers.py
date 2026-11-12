"""Shared data / helpers for the RSM tests (imported as a top-level module by pytest's default path rules)."""
import numpy as np

from adore_doe.rsm import terms as T

A = 1.414
# Montgomery, Design & Analysis of Experiments, RSM chapter (chemical-process example).  Data recalled from the
# book; the arrangement of the four factorial responses was fixed by exhaustive search: of all 576 arrangements
# this is the unique one that reproduces all six published coefficients (79.94, 0.995, 0.515, -1.376, -1.001, 0.25).
Z2 = np.array([(-1, -1), (1, -1), (-1, 1), (1, 1), (-A, 0), (A, 0), (0, -A), (0, A)] + [(0, 0)] * 5)
Y2 = np.array([76.5, 78.0, 77.0, 79.5, 75.6, 78.4, 77.0, 78.5, 79.9, 80.3, 80.0, 79.7, 79.8])
# first-order design with centre points (same book): coefficients 40.44, 0.775, 0.325; SS_PE = 0.1720
Z1 = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)] + [(0, 0)] * 5)
Y1 = np.array([39.3, 40.0, 40.9, 41.5, 40.3, 40.5, 40.7, 40.2, 40.6])


def rand_fit(n=20, k=3, seed=0, kind="quadratic", noise=0.3):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-1, 1, (n, k))
    terms = T.universe(kind, k)
    beta = rng.normal(0, 1, len(terms))
    y = T.expand(Z, terms) @ beta + noise * rng.standard_normal(n)
    return Z, y, terms, [f"x{i+1}" for i in range(k)]
