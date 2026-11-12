"""Phase 5 acceptance experiment: does the ML residual correction help, and does it do harm when there is nothing
to learn?  SYNTHETIC benchmark (3 factors, coded [-1, 1]); every number is reproducible from the seeds.

Truth = quadratic polynomial  +  (a structural residual the quadratic RSM cannot represent)  +  noise (sd 0.05).
Scenarios:  smooth structural (1.0 sin(2.5 x1) cos(2 x2));  localized bump (1.5 exp(-6 ((x1-.4)^2 + (x2+.3)^2)));
            pure quadratic (no structure at all).
Training: Latin hypercube with n = 30 or 45 runs (seeded, different for every seed); independent validation:
1024 Sobol points (never used in any fit).  Compared on the validation RMSE:
  RSM                the plain quadratic model,
  Hybrid             y_RSM + lambda e_ML with OOF residuals, learner chosen among gp/mlp/svr/none (this package),
  always-GP (lam=1)  OOF residuals + GP, NO comparison with 'none' (what a naive correction would do).

  python phase5_residual_experiment.py [--seeds 20] [--out output/phase5_residual_experiment.csv]
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import qmc

from adore_doe import residual
from adore_doe.residual import GPLearner, RSMFitter, oof_rsm

QUAD = lambda Z: 1 + 2 * Z[:, 0] - Z[:, 1] + 0.5 * Z[:, 2] ** 2 + 0.8 * Z[:, 0] * Z[:, 1]
SCEN = {
    "smooth structural": lambda Z: QUAD(Z) + 1.0 * np.sin(2.5 * Z[:, 0]) * np.cos(2 * Z[:, 1]),
    "localized bump": lambda Z: QUAD(Z) + 1.5 * np.exp(-6 * ((Z[:, 0] - 0.4) ** 2 + (Z[:, 1] + 0.3) ** 2)),
    "pure quadratic": QUAD,
}


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="output/phase5_residual_experiment.csv")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    V = 2 * qmc.Sobol(3, seed=99).random(1024) - 1
    rows, t0 = [], time.time()
    for name, f in SCEN.items():
        yv = f(V)
        for n in (30, 45):
            for s in range(a.seeds):
                Z = 2 * qmc.LatinHypercube(d=3, seed=s).random(n) - 1
                y = f(Z) + 0.05 * np.random.default_rng(s).standard_normal(n)
                h = residual.fit_hybrid(Z, y, ["a", "b", "c"], seed=s)
                fitter = RSMFitter(factor_names=["a", "b", "c"])
                oof = oof_rsm(fitter, Z, y, 5, 3, seed=s)
                gp = GPLearner().fit(Z, oof.residual, seed=s)
                naive = h.predict_rsm(V) + gp.predict(V)
                rows.append({"scenario": name, "n": n, "seed": s, "rmse_rsm": rmse(yv, h.predict_rsm(V)),
                             "rmse_hybrid": rmse(yv, h.predict(V)), "rmse_always_gp": rmse(yv, naive),
                             "chosen": h.learner_name, "lambda": h.lam})
        print(f"{name}: done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print("\nvalidation RMSE, mean over seeds (std in brackets); d = Hybrid - RSM (negative = improvement)")
    for (name, n), g in df.groupby(["scenario", "n"], sort=False):
        d = g.rmse_hybrid - g.rmse_rsm
        dn = g.rmse_always_gp - g.rmse_rsm
        print(f"{name:18s} n={n}: RSM {g.rmse_rsm.mean():.4f} ({g.rmse_rsm.std():.4f}) | Hybrid {g.rmse_hybrid.mean():.4f} "
              f"({g.rmse_hybrid.std():.4f}) | always-GP {g.rmse_always_gp.mean():.4f} ({g.rmse_always_gp.std():.4f})")
        print(f"{'':25s} Hybrid-RSM mean {d.mean():+.4f}, better {(d < -1e-9).sum()}/{len(d)}, worse {(d > 1e-9).sum()}/{len(d)}, "
              f"fell back to none {(g.chosen == 'none').sum()}/{len(d)} | always-GP-RSM mean {dn.mean():+.4f}, "
              f"worse {(dn > 1e-9).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
