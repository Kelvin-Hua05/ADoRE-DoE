"""Residual / influence diagnostics for RSM models (Phase 4)."""
from .influence import DiagnosticsResult, compute_diagnostics

__all__ = ["compute_diagnostics", "DiagnosticsResult"]
