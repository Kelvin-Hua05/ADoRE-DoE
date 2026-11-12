"""Final RSM modelling (Phase 4): polynomial OLS with the usual report (coefficients, ANOVA, lack of fit,
fit statistics, explicit equation), term selection with hierarchy, and diagnostics.  See docs/rsm.md."""
from .anova import anova_table, pure_error, replicate_groups
from .fit import fit
from .model import RSMResult, fit_arrays, info_criteria, natural_polynomial
from .selection import SelectionResult, select_terms
from .terms import closure, expand, missing_parents, parents, term_name, universe

__all__ = ["fit", "fit_arrays", "RSMResult", "select_terms", "SelectionResult", "anova_table", "pure_error",
           "replicate_groups", "info_criteria", "natural_polynomial", "universe", "parents", "closure",
           "missing_parents", "expand", "term_name"]
