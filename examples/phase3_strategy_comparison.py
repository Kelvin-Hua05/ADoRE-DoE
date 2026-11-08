"""Phase 3 acceptance experiment: do the global-oriented acquisition strategies beat the baselines?

SYNTHETIC benchmark on known analytic functions (3 factors, coded [-1, 1]).  For every (function, seed):
  * initial design = 12-point Latin hypercube (seeded; different per seed),
  * then 12 sequential single-point rounds with each strategy, refitting a GPR (warm start) each round,
  * metric = TRUE RMSE of the GPR on a fixed 1000-point evaluation set, divided by the SD of the true values
    (NRMSE, comparable across functions); the evaluation set is independent of all training points.
Strategies: random (baseline), max_variance (baseline), ivr, i_optimal (classical polynomial I-optimal
augmentation, ADR-003 main strategy).  Everything is seeded; results are reported as they come out.

  python phase3_strategy_comparison.py [--seeds 8] [--steps 12] [--out output/phase3_strategy_comparison.csv]
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import qmc

from adore_doe import Factor
from adore_doe.active_learning import EvaluationSet, GaussianProcess, build_candidates, select_batch
from adore_doe.core.seed import derive_seed
from adore_doe.design.region import Region

FUNCS = {
    "smooth (quadratic+interaction)": lambda Z: 1 + 2 * Z[:, 0] - Z[:, 1] + 0.5 * Z[:, 2] + 1.5 * Z[:, 0] * Z[:, 1] + Z[:, 2] ** 2,
    "nonlinear (sin/cos)": lambda Z: np.sin(2.5 * Z[:, 0]) + np.cos(2 * Z[:, 1]) * Z[:, 2] + 0.3 * Z[:, 0] ** 2,
    "localized (gaussian bump)": lambda Z: np.exp(-8 * ((Z[:, 0] - 0.4) ** 2 + (Z[:, 1] + 0.3) ** 2)) + 0.3 * Z[:, 2],
}
STRATEGIES = ["random", "max_variance", "ivr", "i_optimal"]


def run_one(fn, strategy, seed, steps, n0, region, es, n_cand, n_int):
    X = 2 * qmc.LatinHypercube(d=3, seed=seed).random(n0) - 1
    y = fn(X)
    theta, curve = None, []
    for step in range(steps + 1):
        gp = GaussianProcess(n_restarts=2, seed=derive_seed(seed, f"gp{step}")).fit(X, y, init_theta=theta)
        theta = gp.theta
        m, _ = gp.predict(es.points)
        curve.append(np.sqrt(np.mean((m - es.truth["y"]) ** 2)) / es.truth["y"].std())
        if step == steps:
            break
        C, _ = build_candidates(region, n_cand, "sobol", seed=derive_seed(seed, f"c{step}"), existing=X)
        sel = select_batch(C, 1, strategy, models={"y": gp}, existing=X, region=region,
                           n_integration=n_int, seed=derive_seed(seed, f"s{step}"))
        X = np.vstack([X, sel.points])
        y = np.append(y, fn(sel.points))
    return np.array(curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--n0", type=int, default=12)
    ap.add_argument("--out", default="output/phase3_strategy_comparison.csv")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    fs = [Factor(c, 0, 1) for c in "ABC"]
    region = Region.from_factors(fs)
    rows, t0 = [], time.time()
    for fname, fn in FUNCS.items():
        es = EvaluationSet.build(fs, 1000, seed=12345).with_truth(lambda P, fn=fn: {"y": fn(P)})
        for st in STRATEGIES:
            for s in range(a.seeds):
                c = run_one(fn, st, 100 + s, a.steps, a.n0, region, es, 300, 256)
                rows.append({"function": fname, "strategy": st, "seed": s, "final_nrmse": c[-1], "mean_nrmse": c.mean(),
                             **{f"nrmse_{i}": v for i, v in enumerate(c)}})
        print(f"{fname}: done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    g = df.groupby(["function", "strategy"])[["final_nrmse", "mean_nrmse"]].agg(["mean", "std"]).round(4)
    print(g.reindex(STRATEGIES, level=1).to_string())
    print("\npaired comparison on the final NRMSE (positive diff = strategy is BETTER than the baseline):")
    for fname in FUNCS:
        sub = df[df.function == fname].pivot(index="seed", columns="strategy", values="final_nrmse")
        for st in ("ivr", "i_optimal", "max_variance"):
            for base in ("random", "max_variance"):
                if st == base:
                    continue
                d = sub[base] - sub[st]
                print(f"  {fname:34s} {st:12s} vs {base:12s}: mean diff {d.mean():+.4f}, wins {(d > 0).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
