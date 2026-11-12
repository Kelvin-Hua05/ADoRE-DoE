"""Term selection (forward / backward / stepwise) with hierarchy control.

criterion 'p'     : enter the candidate with the smallest t-test p-value if p < alpha_in;
                    remove the term with the largest p-value if p > alpha_out   (defaults 0.10 / 0.10)
criterion 'aicc' / 'bic' (or 'aic'): move to the model with the lowest criterion value as long as it
                    improves (smaller than the current one by more than 1e-9).
hierarchy 'strong' (default): a term enters TOGETHER with its missing parents (the candidate is evaluated in the
                    model augmented by those parents; for criterion 'p' the p-value is that of the candidate
                    itself); a term cannot be removed while a child that needs it is in the model.
hierarchy 'none'  : no restriction.
The intercept is always kept; `force` terms are never removed.  Every step is recorded.  The search is a
greedy heuristic: it does not guarantee the best subset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from . import terms as T
from .model import aliased_columns, info_criteria

CRITERIA = ("p", "aicc", "bic", "aic")
METHODS = ("stepwise", "forward", "backward")


@dataclass
class SelectionResult:
    terms: list
    history: list = field(default_factory=list)
    stop_reason: str = ""
    config: dict = field(default_factory=dict)

    def names(self, factor_names):
        return [T.term_name(t, factor_names) for t in self.terms]


def _ols(Z, y, terms):
    X = T.expand(Z, terms)
    n, p = X.shape
    if n < p:
        return None
    _, rank = aliased_columns(X, [str(i) for i in range(p)])
    if rank < p:
        return None
    Q, R = np.linalg.qr(X)
    beta = np.linalg.solve(R, Q.T @ y)
    r = y - X @ beta
    rss = float(r @ r)
    out = {"rss": rss, "p": p, "n": n, "beta": beta}
    if n - p >= 1:
        Rinv = np.linalg.inv(R)
        se = np.sqrt(np.diag(Rinv @ Rinv.T) * rss / (n - p))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = beta / se
        out["pval"] = 2 * stats.t.sf(np.abs(t), n - p)
    else:
        out["pval"] = np.full(p, np.nan)
    return out


def _crit(fit, kind):
    ic = info_criteria(fit["rss"], fit["n"], fit["p"])
    return ic[kind]


def select_terms(Z, y, kind: str = "quadratic", method: str = "stepwise", criterion: str = "p",
                 alpha_in: float = 0.10, alpha_out: float = 0.10, hierarchy: str = "strong",
                 start=None, force=(), max_steps: int = 100) -> SelectionResult:
    if method not in METHODS or criterion not in CRITERIA or hierarchy not in ("strong", "none"):
        raise ValueError(f"method in {METHODS}, criterion in {CRITERIA}, hierarchy in ('strong', 'none').")
    if not (0 < alpha_in < 1 and 0 < alpha_out < 1):
        raise ValueError("alpha_in and alpha_out must be in (0, 1).")
    Z = np.atleast_2d(np.asarray(Z, float))
    y = np.asarray(y, float).ravel()
    n, k = Z.shape
    icpt = T.intercept(k)
    full = T.universe(kind, k)
    cand_all = [t for t in full if t != icpt]
    forced = {tuple(t) for t in force}
    if start is None:
        cur = set(full) if method == "backward" else {icpt}
    else:
        cur = {icpt, *map(tuple, start)}
    cur |= forced
    if hierarchy == "strong":
        cur = set(T.closure(cur, k))
    key = lambda s: sorted(s, key=T.order_key)
    hist, seen = [], {frozenset(cur)}
    cfg = dict(kind=kind, method=method, criterion=criterion, alpha_in=alpha_in, alpha_out=alpha_out,
               hierarchy=hierarchy, force=sorted(forced), n=n)
    reason = f"max_steps={max_steps} reached"

    def candidates_to_add():
        out = []
        for c in cand_all:
            if c in cur:
                continue
            add = {c} | (set(T.parents(c)) - cur if hierarchy == "strong" else set())
            new = cur | add
            f = _ols(Z, y, key(new))
            if f is None or (criterion != "p" and f["n"] - f["p"] < 1) or (criterion == "p" and f["n"] - f["p"] < 1):
                continue
            idx = key(new).index(c)
            out.append((c, add - {c}, f, f["pval"][idx]))
        return out

    def removable():
        out = []
        for t in cur:
            if t == icpt or t in forced:
                continue
            if hierarchy == "strong" and any(t in T.parents(ch) for ch in cur if ch != t):
                continue
            out.append(t)
        return out

    def try_remove(curfit):
        rem = removable()
        if not rem:
            return None
        if criterion == "p":
            order = key(cur)
            worst = max(rem, key=lambda t: (np.nan_to_num(curfit["pval"][order.index(t)], nan=1.0)))
            pv = curfit["pval"][order.index(worst)]
            if np.isfinite(pv) and pv > alpha_out:
                return worst, float(pv)
            return None
        best = None
        for t in rem:
            f = _ols(Z, y, key(cur - {t}))
            if f is None:
                continue
            v = _crit(f, criterion)
            if best is None or v < best[1]:
                best = (t, v)
        return best if best and best[1] < _crit(curfit, criterion) - 1e-9 else None

    for step in range(1, max_steps + 1):
        curfit = _ols(Z, y, key(cur))
        if curfit is None:
            reason = "current model not identifiable"
            break
        acted = False
        if method in ("forward", "stepwise"):
            cands = candidates_to_add()
            pick = None
            if cands:
                if criterion == "p":
                    c = min(cands, key=lambda r: np.nan_to_num(r[3], nan=1.0))
                    if np.isfinite(c[3]) and c[3] < alpha_in:
                        pick = (c[0], c[1], float(c[3]))
                else:
                    c = min(cands, key=lambda r: _crit(r[2], criterion))
                    if _crit(c[2], criterion) < _crit(curfit, criterion) - 1e-9:
                        pick = (c[0], c[1], float(_crit(c[2], criterion)))
            if pick:
                cur |= {pick[0]} | set(pick[1])
                hist.append({"step": step, "action": "add", "term": pick[0], "parents_added": sorted(pick[1]),
                             "value": pick[2], "terms": key(cur)})
                acted = True
        if not acted and method in ("backward", "stepwise"):
            rm = try_remove(curfit)
            if rm:
                cur = cur - {rm[0]}
                hist.append({"step": step, "action": "remove", "term": rm[0], "parents_added": [],
                             "value": float(rm[1]), "terms": key(cur)})
                acted = True
        if not acted:
            reason = "converged: no term to add or remove"
            break
        fs = frozenset(cur)
        if fs in seen:
            reason = "cycle detected; stopped"
            break
        seen.add(fs)
        if method == "stepwise" and criterion == "p":
            pass
    return SelectionResult(key(cur), hist, reason, cfg)
