"""Polynomial terms for RSM models (coded coordinates).

A term is a tuple of exponents (one per factor); the intercept is the all-zero tuple.
Hierarchy: the PARENTS of x_i x_j are x_i and x_j; the parent of x_i^2 is x_i; linear terms and
the intercept have no parents.  `strong` hierarchy = a term may be in the model only if all its
parents are.
"""
from __future__ import annotations

import numpy as np

from ..design.model import ModelSpec


def order_key(t) -> tuple:
    """Natural display order = the order of ModelSpec: intercept, A, B, C, A*B, A*C, B*C, A^2, B^2, C^2."""
    return (sum(t), 1 if 2 in t else 0, tuple(-x for x in t))


def intercept(k: int) -> tuple:
    return (0,) * k


def unit(k: int, i: int) -> tuple:
    return tuple(1 if j == i else 0 for j in range(k))


def universe(kind: str, k: int) -> list:
    """All terms of the model kind, intercept first."""
    return [tuple(int(v) for v in t) for t in ModelSpec(kind, k).terms]


def term_name(e, names) -> str:
    parts = [f"{names[j]}" if p == 1 else f"{names[j]}^{p}" for j, p in enumerate(e) if p]
    return "*".join(parts) if parts else "Intercept"


def parents(e: tuple) -> list:
    k = len(e)
    nz = [j for j, p in enumerate(e) if p]
    if sum(e) <= 1:
        return []
    if len(nz) == 1 and e[nz[0]] == 2:
        return [unit(k, nz[0])]
    if len(nz) == 2 and all(e[j] == 1 for j in nz):
        return [unit(k, j) for j in nz]
    raise ValueError(f"unsupported term {e}")


def children(term: tuple, present) -> list:
    return [t for t in present if term in parents(t)]


def closure(selected, k: int) -> list:
    """`selected` plus all missing parents (strong-hierarchy repair), intercept included."""
    out = {intercept(k), *map(tuple, selected)}
    for t in list(out):
        out.update(parents(t))
    return sorted(out, key=order_key)


def missing_parents(terms) -> dict:
    s = set(map(tuple, terms))
    return {t: [p for p in parents(t) if p not in s] for t in s if any(p not in s for p in parents(t))}


def expand(Z, terms) -> np.ndarray:
    Z = np.atleast_2d(np.asarray(Z, float))
    return np.column_stack([np.prod(np.power(Z, np.array(e)), axis=1) for e in terms])
