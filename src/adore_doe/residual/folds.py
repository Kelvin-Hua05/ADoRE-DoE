"""Seeded (repeated) K-fold splits -- a local copy so that `residual` does not depend on `active_learning` (ADR-005)."""
from __future__ import annotations

import numpy as np

from ..core.seed import derive_seed, make_rng, resolve_seed
from ..exceptions import DesignError


def make_folds(n: int, n_splits: int = 5, repeats: int = 1, seed=None):
    """list (one per repeat) of lists of (train_idx, test_idx); n_splits is capped at n."""
    if n < 2:
        raise DesignError("cross-validation needs at least 2 observations.")
    if n_splits < 2 or repeats < 1:
        raise DesignError("n_splits must be >= 2 and repeats >= 1.")
    k, root, out = min(int(n_splits), n), resolve_seed(seed), []
    for r in range(int(repeats)):
        perm = make_rng(derive_seed(root, "folds", str(r))).permutation(n)
        chunks = np.array_split(perm, k)
        out.append([(np.concatenate([c for j, c in enumerate(chunks) if j != f]), np.sort(chunks[f]))
                    for f in range(k)])
    return out
