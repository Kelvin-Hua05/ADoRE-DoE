"""ADoRE-DoE - Adaptive Design of Experiments with Residual Enhancement."""
from . import active_learning, core, design, diagnostics, residual, rsm, screening
from .core import Experiment, ExperimentDataset, Factor, Response
from .exceptions import (DesignError, InferenceError, InferenceWarning,
                         InfeasibleRegionError, RankDeficientError)

__version__ = "0.1.0.dev0"
__all__ = ["Factor", "Response", "ExperimentDataset", "Experiment", "core", "design", "screening", "active_learning", "rsm", "diagnostics", "residual",
           "DesignError", "RankDeficientError", "InferenceError", "InferenceWarning",
           "InfeasibleRegionError"]
