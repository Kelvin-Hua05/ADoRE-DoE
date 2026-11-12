"""rsm.fit(dataset, response, ...): RSM model of one response from an ExperimentDataset."""
from __future__ import annotations

import numpy as np

from ..core.dataset import ExperimentDataset
from ..exceptions import DesignError
from . import terms as T
from .model import RSMResult, fit_arrays
from .selection import select_terms


def fit(dataset: ExperimentDataset, response: str, model: str = "quadratic", selection: dict | None = None,
        alpha: float = 0.05, terms=None, on_rank_deficient: str = "error") -> RSMResult:
    """Fit the polynomial model of `response` on the TRAINING runs (validation runs are excluded).

    `model`: 'linear' | 'interaction' | 'quadratic' (the full candidate set).  `terms`: explicit list of
    exponent tuples (overrides `model`).  `selection`: e.g. {"method": "stepwise", "criterion": "p",
    "alpha_in": 0.1, "alpha_out": 0.1, "hierarchy": "strong"} selects a sub-model of `model` first."""
    if response not in dataset.response_names:
        raise DesignError(f"dataset has no response {response!r}.")
    mask = dataset.training_mask & ~np.isnan(dataset.get_response(response))
    if mask.sum() < 3:
        raise DesignError("need at least 3 training runs with a measured response.")
    Z = dataset.coded[mask]
    y = dataset.get_response(response)[mask]
    ro = dataset.run_order[mask]
    k = Z.shape[1]
    sel = None
    if terms is not None:
        use = [tuple(t) for t in terms]
    elif selection:
        sel = select_terms(Z, y, kind=model, **selection)
        use = sel.terms
    else:
        use = T.universe(model, k)
    res = fit_arrays(Z, y, use, dataset.factor_names, dataset.factors, ro, alpha, response, on_rank_deficient)
    res.selection = sel
    res.model_kind = model
    return res
