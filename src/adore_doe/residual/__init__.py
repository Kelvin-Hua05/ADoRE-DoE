"""Module 2 - ML residual correction (Phase 5): out-of-fold residual targets, GP / MLP / SVR learners with a
'no correction' fallback, and the hybrid model  y_RSM + lambda * e_ML.  Independent of `active_learning` (ADR-005).
See docs/residual.md."""
from .folds import make_folds
from .hybrid import HybridModel, fit, fit_hybrid, fit_responses
from .learners import LEARNERS, GPLearner, MLPLearner, NoneLearner, SVRLearner, make_learner
from .oof import OOFResult, RSMFitter, oof_rsm

__all__ = ["fit", "fit_hybrid", "fit_responses", "HybridModel", "RSMFitter", "oof_rsm", "OOFResult", "make_folds",
           "make_learner", "LEARNERS", "GPLearner", "MLPLearner", "SVRLearner", "NoneLearner"]
