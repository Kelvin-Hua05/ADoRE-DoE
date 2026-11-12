import ast
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import qmc

import adore_doe as ad
from adore_doe import design, residual
from adore_doe.residual import (GPLearner, MLPLearner, NoneLearner, RSMFitter, SVRLearner, fit_hybrid, make_folds,
                                make_learner, oof_rsm)
from adore_doe.rsm import fit_arrays, select_terms, universe

NAMES = ["a", "b", "c"]
QUAD = lambda Z: 1 + 2 * Z[:, 0] - Z[:, 1] + 0.5 * Z[:, 2] ** 2 + 0.8 * Z[:, 0] * Z[:, 1]
SMOOTH = lambda Z: QUAD(Z) + 1.0 * np.sin(2.5 * Z[:, 0]) * np.cos(2 * Z[:, 1])
V = 2 * qmc.Sobol(3, seed=99).random(512) - 1


def data(f, n, seed, noise=0.05):
    Z = 2 * qmc.LatinHypercube(d=3, seed=seed).random(n) - 1
    return Z, f(Z) + noise * np.random.default_rng(seed).standard_normal(n)


# ------------------------------------------------------------------ folds
def test_folds_partition_and_are_seeded():
    reps = make_folds(13, 5, 3, seed=2)
    assert len(reps) == 3
    for folds in reps:
        assert sorted(np.concatenate([te for _, te in folds])) == list(range(13))
        assert all(not set(tr) & set(te) for tr, te in folds)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(make_folds(10, 5, 1, 7)[0], make_folds(10, 5, 1, 7)[0]))
    assert len(make_folds(4, 10, 1, 0)[0]) == 4
    for bad in ((1,), (10, 1), (10, 5, 0)):
        with pytest.raises(ad.DesignError):
            make_folds(*bad)


# ------------------------------------------------------------------ OOF residual targets
def test_oof_predictions_are_what_the_fold_models_predict():
    Z, y = data(SMOOTH, 24, 1)
    fitter = RSMFitter(factor_names=NAMES, selection={"method": "stepwise", "criterion": "p"})
    o = oof_rsm(fitter, Z, y, n_splits=4, repeats=2, seed=3)
    manual = np.zeros((2, 24))
    for r, rep in enumerate(make_folds(24, 4, 2, 3)):
        for tr, te in rep:
            manual[r, te] = fitter.fit(Z[tr], y[tr]).predict(Z[te])
    assert np.allclose(o.per_repeat, manual) and np.allclose(o.pred, manual.mean(axis=0))
    assert np.allclose(o.residual, y - o.pred) and o.rmse == pytest.approx(np.sqrt(np.mean((y - o.pred) ** 2)))
    assert o.info["repeats"] == 2 and o.info["n_splits"] == 4


def test_oof_residual_does_not_depend_on_its_own_response_even_with_selection_inside_folds():
    """Constructive no-leakage test (ADR-006): changing y_i must not change the OOF prediction of run i.
    The term selection is redone inside every fold; selecting terms once on ALL data (the leaky shortcut) does leak."""
    rng = np.random.default_rng(0)
    Z = rng.uniform(-1, 1, (24, 3))
    y = 2 * Z[:, 0] + 0.3 * rng.standard_normal(24)
    sel = {"method": "stepwise", "criterion": "p", "alpha_in": 0.2, "alpha_out": 0.2}
    inside = RSMFitter(factor_names=NAMES, selection=sel)
    base = oof_rsm(inside, Z, y, 5, 2, seed=1).pred

    class Fixed:                                           # leaky: terms chosen on the full data first
        def __init__(self, terms):
            self.terms = terms

        def fit(self, Zt, yt):
            return fit_arrays(Zt, yt, self.terms, NAMES, on_rank_deficient="drop")

    leak_changes = []
    for i in range(24):
        y2 = y.copy()
        y2[i] += 8.0
        assert oof_rsm(inside, Z, y2, 5, 2, seed=1).pred[i] == pytest.approx(base[i], abs=1e-9)   # never changes
        t0 = select_terms(Z, y, kind="quadratic", **sel).terms
        t1 = select_terms(Z, y2, kind="quadratic", **sel).terms
        p0 = oof_rsm(Fixed(t0), Z, y, 5, 2, seed=1).pred[i]
        p1 = oof_rsm(Fixed(t1), Z, y2, 5, 2, seed=1).pred[i]
        leak_changes.append(abs(p1 - p0) > 1e-6)
    assert any(leak_changes)                               # the test can detect the leak it guards against


def test_rank_deficient_folds_are_repaired_by_dropping_terms_and_counted():
    # 7 points of a 3x3 grid: identifiable for the quadratic model, but leaving out (1, -1) -- the only point with
    # x1 = 1 -- makes x1^2 collinear with x1 in that fold.
    Z = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1)], float)
    y = 1 + Z[:, 0] + Z[:, 1] ** 2
    o = oof_rsm(RSMFitter(factor_names=["a", "b"]), Z, y, n_splits=7, repeats=1, seed=1)        # leave-one-out
    assert np.isfinite(o.pred).all() and o.info["n_folds_with_dropped_terms"] == 1
    strict = RSMFitter(factor_names=["a", "b"], on_rank_deficient="error")
    with pytest.raises(ad.RankDeficientError, match="not identifiable"):
        oof_rsm(strict, Z, y, n_splits=7, repeats=1, seed=1)


# ------------------------------------------------------------------ learners
@pytest.mark.parametrize("name", ["gp", "svr", "mlp"])
def test_learners_fit_predict_reproducibly(name):
    Z, _ = data(SMOOTH, 40, 2)
    e = np.sin(2.5 * Z[:, 0]) * np.cos(2 * Z[:, 1])
    m1, m2 = make_learner(name).fit(Z, e, seed=3), make_learner(name).fit(Z, e, seed=3)
    P = V[:50]
    assert m1.predict(P).shape == (50,) and np.isfinite(m1.predict(P)).all()
    assert np.allclose(m1.predict(P), m2.predict(P))                                   # same seed -> same model
    truth = np.sin(2.5 * P[:, 0]) * np.cos(2 * P[:, 1])
    if name in ("gp", "svr"):
        assert np.mean((truth - m1.predict(P)) ** 2) < 0.5 * np.var(truth)             # learns the structure


def test_learner_edge_cases_and_registry():
    Z = V[:20]
    for name in ("gp", "svr", "mlp"):
        const = make_learner(name).fit(Z, np.full(20, 0.7), seed=1)
        assert np.allclose(const.predict(Z[:5]), 0.7)                                  # constant target -> constant
    assert np.allclose(NoneLearner().fit(Z, np.ones(20)).predict(Z), 0.0)
    assert make_learner("none").name == "none" and set(residual.LEARNERS) == {"gp", "mlp", "svr"}
    with pytest.raises(ad.DesignError, match="Random Forest is deferred"):
        make_learner("rf")
    with pytest.raises(ad.DesignError):
        make_learner("xgb")
    assert isinstance(make_learner("svr"), SVRLearner) and isinstance(make_learner("mlp"), MLPLearner)
    assert isinstance(make_learner("gp"), GPLearner)


def test_scikit_learn_is_optional_for_importing_the_package():
    src = os.path.dirname(os.path.dirname(ad.__file__))
    code = ("import sys; sys.modules['sklearn'] = None\n"
            "import adore_doe, adore_doe.residual\n"
            "from adore_doe.residual import make_learner\n"
            "try:\n    make_learner('gp').fit([[0.0],[1.0],[2.0]], [1.0, 2.0, 4.0])\n"
            "except ImportError as e:\n    print('MSG', e)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": src})
    assert out.returncode == 0, out.stderr
    assert "adore-doe[ml]" in out.stdout


# ------------------------------------------------------------------ hybrid model
def test_hybrid_is_rsm_plus_lambda_times_learner_and_reports_the_comparison():
    Z, y = data(SMOOTH, 45, 3)
    h = fit_hybrid(Z, y, NAMES, seed=1)
    assert np.allclose(h.predict(V), h.predict_rsm(V) + h.lam * h.learner.predict(V))
    assert 0.0 <= h.lam <= 1.0 and h.learner_name in {"gp", "mlp", "svr", "none"}
    assert set(h.report.columns) >= {"learner", "oof_rmse_lambda1", "lambda", "oof_rmse_at_lambda", "gain", "gain_lambda1", "chosen"}
    assert set(h.report.learner) == {"gp", "mlp", "svr"} and h.report.chosen.sum() == (1 if h.corrected else 0)
    assert h.config["rmse_oof_rsm"] == pytest.approx(h.oof.rmse) and h.config["seed"] == 1
    ev = h.evaluate(V, SMOOTH(V))
    assert list(ev.index) == ["RSM", "Hybrid", "Hybrid - RSM"] and {"RMSE", "MAE", "R2"} <= set(ev.columns)
    assert ev.loc["Hybrid - RSM", "RMSE"] == pytest.approx(ev.loc["Hybrid", "RMSE"] - ev.loc["RSM", "RMSE"])


def test_fixed_lambda_min_gain_and_validation():
    Z, y = data(SMOOTH, 45, 3)
    h = fit_hybrid(Z, y, NAMES, lam=0.5, seed=1)
    assert h.lam in (0.0, 0.5)
    assert fit_hybrid(Z, y, NAMES, min_gain=0.99, seed=1).learner_name == "none"       # nothing clears an absurd bar
    z = fit_hybrid(Z, y, NAMES, min_gain=0.99, seed=1)
    assert z.lam == 0.0 and np.allclose(z.predict(V), z.predict_rsm(V)) and not z.corrected
    assert np.allclose(z.evaluate(V, SMOOTH(V)).loc["Hybrid - RSM", ["RMSE", "MAE"]], 0.0)
    for bad in (dict(lam=1.5), dict(lam=-0.1), dict(lam="x"), dict(min_gain=-1)):
        with pytest.raises(ad.DesignError):
            fit_hybrid(Z, y, NAMES, **bad)
    with pytest.raises(ad.DesignError):
        fit_hybrid(Z, y, NAMES, learners=("gp", "rf"), seed=1)


def test_same_seed_same_model_and_report():
    Z, y = data(SMOOTH, 36, 4)
    a, b = fit_hybrid(Z, y, NAMES, seed=5), fit_hybrid(Z, y, NAMES, seed=5)
    assert a.report.equals(b.report) and a.learner_name == b.learner_name and a.lam == b.lam
    assert np.allclose(a.predict(V[:30]), b.predict(V[:30]))
    assert set(a.config["seeds"]) >= {"root", "oof", "inner"}


# ------------------------------------------------------------------ the acceptance behaviour (statistical, seeded)
def test_no_correction_when_there_is_nothing_to_learn():
    """Pure quadratic truth + noise: the hybrid must fall back to the plain RSM (measured 40/40 in the experiment)."""
    none, worse = 0, 0
    for s in range(6):
        Z, y = data(QUAD, 36, s)
        h = fit_hybrid(Z, y, NAMES, seed=s)
        none += h.learner_name == "none"
        d = h.evaluate(V, QUAD(V)).loc["Hybrid - RSM", "RMSE"]
        worse += d > 1e-9
    assert none >= 5 and worse <= 1


def test_correction_improves_independent_validation_error_when_a_structural_residual_exists():
    wins, deltas = 0, []
    for s in range(6):
        Z, y = data(SMOOTH, 45, s)
        h = fit_hybrid(Z, y, NAMES, seed=s)
        d = h.evaluate(V, SMOOTH(V)).loc["Hybrid - RSM", "RMSE"]
        deltas.append(d)
        wins += d < -1e-9
    assert wins >= 5 and np.mean(deltas) < -0.05 and max(deltas) < 0.05


# ------------------------------------------------------------------ independence of validation data / datasets
def test_validation_data_never_enters_any_fit():
    Z, y = data(SMOOTH, 36, 2)
    a = fit_hybrid(Z, y, NAMES, seed=1)
    pred_before = a.predict(V)
    wild = SMOOTH(V) * 100 + 5                                  # whatever the validation responses are ...
    ev1, ev2 = a.evaluate(V, SMOOTH(V)), a.evaluate(V, wild)
    assert np.allclose(a.predict(V), pred_before)               # ... the fitted model is unchanged by evaluation
    assert ev1.loc["RSM", "RMSE"] != pytest.approx(ev2.loc["RSM", "RMSE"])


def test_dataset_wrapper_uses_training_runs_only_and_is_per_response():
    fs = [ad.Factor("A", 0, 10), ad.Factor("B", 0, 10), ad.Factor("C", 0, 10)]
    ds = design.i_optimal(fs, "quadratic", 36, seed=1, n_starts=4)
    Zc = ds.coded
    ds.add_response("y1", SMOOTH(Zc) + 0.02 * np.random.default_rng(0).standard_normal(36))
    ds.add_response("y2", QUAD(Zc) * 1e6 + 0.02e6 * np.random.default_rng(1).standard_normal(36))
    a = residual.fit(ds, "y1", seed=1)
    ds2 = ds.copy()
    ds2.append_runs(np.array([[0.1, 0.2, 0.3], [-0.5, 0.5, 0.0]]), source="validation", values={"y1": [999.0, -999.0], "y2": [1.0, 2.0]})
    b = residual.fit(ds2, "y1", seed=1)
    assert np.allclose(a.base.beta, b.base.beta) and a.learner_name == b.learner_name and a.base.n == 36
    both = residual.fit_responses(ds, ["y1", "y2"], seed=1)
    assert set(both) == {"y1", "y2"} and both["y2"].base.response == "y2" and both["y1"].base.factors is ds.factors
    assert both["y2"].learner_name == "none" or both["y2"].corrected                  # each response decided separately
    with pytest.raises(ad.DesignError):
        residual.fit(ds, "nope")
    small = design.ccd(fs[:2], center_points=1)
    small.add_response("y", np.arange(small.n_runs, dtype=float))
    small.append_runs(np.zeros((0, 2)))
    with pytest.raises(ad.DesignError, match="at least 6"):
        residual.fit(small.subset(np.arange(small.n_runs) < 5), "y")


# ------------------------------------------------------------------ ADR-005
def test_residual_does_not_import_active_learning_and_vice_versa():
    root = pathlib.Path(residual.__file__).parent
    bad = []
    for f in root.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods = []
            if isinstance(node, ast.ImportFrom):
                mods = [("." * node.level) + (node.module or "")] + [("." * node.level) + (node.module or "") + "." + a.name for a in node.names]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                tail = m.replace("adore_doe", "").lstrip(".")
                if tail.split(".")[0] == "active_learning":
                    bad.append((f.name, m))
    assert bad == []
