"""Module 1 - Active Learning (Phase 3): GPR surrogate, leak-free cross-validated predictive R^2,
multi-response stopping rule, fixed evaluation set, acquisition strategies (I-optimal augmentation as the
main strategy; IVR, max-variance, random as comparisons) and the ActiveLearner loop.
This package must not import `rsm` or `residual` (ADR-005).  See docs/active_learning.md."""
from .acquisition import STRATEGIES, Selection, select_batch
from .candidates import build_candidates, distances_to
from .cv import R2Result, gpr_cv_r2, make_folds, oof_predict, predictive_r2
from .evaluation import EvalReport, EvaluationSet
from .gpr import GaussianProcess, kernel_matrix
from .learner import ActiveLearner
from .stopping import (MultiResponseStopping, RoundCheck, StopDecision, check_round,
                       validate_weights)

__all__ = ["GaussianProcess", "kernel_matrix", "gpr_cv_r2", "predictive_r2", "make_folds", "oof_predict",
           "R2Result", "EvaluationSet", "EvalReport", "build_candidates", "distances_to", "select_batch",
           "Selection", "STRATEGIES", "MultiResponseStopping", "StopDecision", "RoundCheck", "check_round",
           "validate_weights", "ActiveLearner"]
