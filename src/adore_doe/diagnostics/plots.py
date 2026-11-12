"""Diagnostic plots (matplotlib is an optional extra: pip install 'adore-doe[plot]')."""
from __future__ import annotations

import numpy as np
from scipy import stats

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("Plotting needs matplotlib: pip install 'adore-doe[plot]'") from e


def _ax(ax):
    return plt.subplots(figsize=(5.5, 4.5))[1] if ax is None else ax


def _save(ax, save_path):
    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def normal_plot(values, ax=None, save_path=None, title="Normal plot of residuals"):
    """Ordered (externally studentised) residuals vs normal quantiles at plotting positions (i - 0.5)/n."""
    v = np.sort(np.asarray(values, float))
    v = v[np.isfinite(v)]
    n = len(v)
    q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax = _ax(ax)
    ax.scatter(q, v, edgecolors="k", s=30, zorder=3)
    if n >= 4:                                     # line through the quartiles (robust reference)
        q1, q3 = np.percentile(v, [25, 75])
        z1, z3 = stats.norm.ppf([0.25, 0.75])
        slope = (q3 - q1) / (z3 - z1)
        xs = np.array([q[0], q[-1]])
        ax.plot(xs, q1 + slope * (xs - z1), color="#E57373", label="quartile line")
        ax.legend(fontsize=8)
    ax.set_xlabel("Theoretical normal quantile")
    ax.set_ylabel("Externally studentised residual")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    return _save(ax, save_path)


def scatter(x, y, xlabel, ylabel, hline=None, identity=False, ax=None, save_path=None, title=None):
    ax = _ax(ax)
    x, y = np.asarray(x, float), np.asarray(y, float)
    ax.scatter(x, y, edgecolors="k", s=30, zorder=3)
    if hline is not None:
        ax.axhline(hline, color="gray", lw=1)
    if identity:
        lo, hi = np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])
        ax.plot([lo, hi], [lo, hi], color="#E57373")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{ylabel} vs {xlabel}", fontsize=10)
    ax.grid(alpha=0.3)
    return _save(ax, save_path)
