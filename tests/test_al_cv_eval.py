import numpy as np
import pytest

import adore_doe as ad
from adore_doe.active_learning.candidates import build_candidates, distances_to
from adore_doe.active_learning.cv import gpr_cv_r2, make_folds, oof_predict, predictive_r2
from adore_doe.active_learning.evaluation import EvaluationSet
from adore_doe.active_learning.gpr import GaussianProcess
from adore_doe.design.region import Region


def F(k=3):
    return [ad.Factor(chr(65 + i), 0.0, 10.0) for i in range(k)]


def smooth(Z):
    return np.sin(2 * Z[:, 0]) + 0.5 * Z[:, 1] ** 2 + Z[:, 0] * Z[:, -1]


# ------------------------------------------------------------------ folds
@pytest.mark.parametrize("scheme,kw", [("kfold", dict(n_splits=5)), ("kfold", dict(n_splits=3)),
                                       ("repeated_kfold", dict(n_splits=4, repeats=3)), ("loo", {})])
def test_folds_partition_the_data_and_are_disjoint(scheme, kw):
    n = 13
    reps = make_folds(n, scheme, seed=1, **kw)
    assert len(reps) == (kw.get("repeats", 1) if scheme == "repeated_kfold" else 1)
    for folds in reps:
        tests = np.concatenate([te for _, te in folds])
        assert sorted(tests) == list(range(n))                                  # each point held out exactly once
        for tr, te in folds:
            assert not set(tr) & set(te) and len(tr) + len(te) == n
    if scheme == "loo":
        assert len(reps[0]) == n


def test_folds_reproducible_capped_and_repeats_differ():
    a, b = make_folds(10, "kfold", 5, seed=3), make_folds(10, "kfold", 5, seed=3)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a[0], b[0]))
    assert len(make_folds(4, "kfold", 10, seed=0)[0]) == 4                       # n_splits capped at n
    r = make_folds(12, "repeated_kfold", 4, 2, seed=1)
    assert not all(np.array_equal(x[1], y[1]) for x, y in zip(r[0], r[1]))
    for bad in (dict(scheme="bootstrap"), dict(n=1), dict(n_splits=1), dict(scheme="repeated_kfold", repeats=0)):
        kw = dict(n=10, scheme="kfold", n_splits=5, repeats=1); kw.update(bad)
        with pytest.raises(ad.DesignError):
            make_folds(**kw)


# ------------------------------------------------------------------ predictive R^2
def test_predictive_r2_definition_and_negative_values_are_kept():
    y = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    assert predictive_r2(y, y[None, :]).value == pytest.approx(1.0)
    assert predictive_r2(y, np.full((1, 5), y.mean())).value == pytest.approx(0.0, abs=1e-12)
    r = predictive_r2(y, (2 * y.mean() - y)[None, :])                           # worse than the mean
    sst = np.sum((y - y.mean()) ** 2)
    assert r.value == pytest.approx(1 - np.sum((y - (2 * y.mean() - y)) ** 2) / sst) and r.value < 0
    assert r.valid and r.status == "ok" and r.rmse > 0 and r.mae > 0
    rep = predictive_r2(y, np.vstack([y, np.full(5, y.mean())]))               # pooled over repeats = mean of R^2
    assert rep.value == pytest.approx(0.5)


def test_undefined_r2_is_reported_with_a_reason_not_zero():
    const = predictive_r2(np.full(6, 2.0), np.full((1, 6), 2.0))
    assert const.status == "sst_zero" and np.isnan(const.value) and "SST = 0" in const.reason and not const.valid
    few = predictive_r2(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))
    assert few.status == "insufficient_samples" and np.isnan(few.value) and not few.valid
    bad = predictive_r2(np.arange(5.0), np.array([[0, 1, np.nan, 3, 4.0]]))
    assert bad.status == "nonfinite_prediction" and np.isnan(bad.value)


# ------------------------------------------------------------------ GP cross-validation
def test_gpr_cv_r2_is_high_for_smooth_functions_and_low_for_noise():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (30, 3))
    good, _, info = gpr_cv_r2(X, smooth(X), seed=1, n_restarts=1)
    assert good.valid and good.value > 0.8 and info["scheme"] == "kfold" and not info["leaky"]
    bad, _, _ = gpr_cv_r2(X, rng.standard_normal(30), seed=1, n_restarts=1)
    assert bad.value < 0.3                                                       # no signal -> no predictive power


@pytest.mark.parametrize("scheme", ["kfold", "loo"])
def test_cv_is_leak_free_changing_a_response_never_changes_its_own_oof_prediction(scheme):
    """Constructive no-leakage test: y_i must not influence the out-of-fold prediction of observation i,
    even though the hyper-parameters are re-optimised inside every fold."""
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (10, 2))
    y = smooth(np.column_stack([X, X[:, :1]]))
    base = gpr_cv_r2(X, y, scheme=scheme, n_splits=5, n_restarts=1, seed=2)[1][0]
    for i in (0, 4, 9):
        y2 = y.copy()
        y2[i] += 5.0
        oof2 = gpr_cv_r2(X, y2, scheme=scheme, n_splits=5, n_restarts=1, seed=2)[1][0]
        assert oof2[i] == pytest.approx(base[i], abs=1e-9)
        others_changed = np.abs(np.delete(oof2, i) - np.delete(base, i)).max()
        assert others_changed > 1e-6                                              # the test can detect dependence


def test_fixed_hyperparameters_option_is_leaky_and_flagged():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (10, 2))
    y = smooth(np.column_stack([X, X[:, :1]]))
    base, _, info = (lambda r: (r[1][0], r[0], r[2]))(gpr_cv_r2(X, y, refit_hyperparameters=False, n_restarts=1, seed=2))
    assert info["leaky"] is True
    y2 = y.copy(); y2[3] += 5.0
    oof2 = gpr_cv_r2(X, y2, refit_hyperparameters=False, n_restarts=1, seed=2)[1][0]
    assert abs(oof2[3] - base[3]) > 1e-6                                          # hyper-parameters saw y_3: leakage


def test_cv_reproducible_and_small_samples_reported():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, (12, 2)); y = smooth(np.column_stack([X, X[:, :1]]))
    a = gpr_cv_r2(X, y, n_restarts=1, seed=5)[0].value
    assert a == gpr_cv_r2(X, y, n_restarts=1, seed=5)[0].value
    tiny = gpr_cv_r2(X[:2], y[:2], seed=1)[0]
    assert tiny.status == "insufficient_samples"
    assert gpr_cv_r2(X, np.full(12, 4.0), n_restarts=1, seed=1)[0].status == "sst_zero"
    rep = gpr_cv_r2(X, y, scheme="repeated_kfold", repeats=2, n_restarts=1, seed=5)
    assert rep[1].shape == (2, 12) and rep[2]["repeats"] == 2


def test_oof_predict_uses_only_training_folds():
    X = np.arange(10, dtype=float)[:, None]
    seen = []
    def fp(Xtr, ytr, Xte, s):
        seen.append((set(Xtr.ravel()), set(Xte.ravel())))
        return np.full(len(Xte), ytr.mean())
    oof = oof_predict(X, np.arange(10.0), fp, "kfold", 5, seed=0)
    assert all(not tr & te for tr, te in seen) and oof.shape == (1, 10) and np.isfinite(oof).all()


# ------------------------------------------------------------------ evaluation set
def test_evaluation_set_is_seeded_feasible_and_independent_of_training():
    cons = [({"A": 1, "B": 1}, "<=", 12)]
    a = EvaluationSet.build(F(2), 300, constraints=cons, seed=4)
    b = EvaluationSet.build(F(2), 300, constraints=cons, seed=4)
    assert np.array_equal(a.points, b.points) and a.seed == 4 and 0 < a.acceptance_rate < 1
    assert Region.from_factors(F(2), cons).contains(a.points).all()
    assert not np.array_equal(a.points, EvaluationSet.build(F(2), 300, constraints=cons, seed=5).points)
    a.check_disjoint(np.array([[0.123, 0.456]]))                                   # fine
    with pytest.raises(ad.DesignError, match="coincide"):
        a.check_disjoint(np.vstack([a.points[7], [0.1, 0.1]]))
    with pytest.raises(ad.DesignError):
        EvaluationSet.build(F(2), 5)


class _Stub:
    def __init__(self, mean, std, dv=2.0):
        self._m, self._s, self.data_variance = mean, std, dv

    def predict(self, X):
        n = len(X)
        return np.full(n, self._m) if np.isscalar(self._m) else self._m(X), np.full(n, self._s)


def test_evaluation_metrics_with_and_without_truth():
    es = EvaluationSet.build(F(2), 200, seed=1)
    rep0 = es.evaluate({"y": _Stub(0.0, 0.5)})
    r = rep0.per_response["y"]
    assert not rep0.has_truth and r["rmse"] is None and r["r2"] is None and r["r2_status"] == "no_truth"
    assert r["mean_pred_var"] == pytest.approx(0.25) and r["mean_pred_var_normalised"] == pytest.approx(0.125)
    truth = es.with_truth(lambda P: {"y": P[:, 0] * 2.0})
    perfect = truth.evaluate({"y": _Stub(lambda X: X[:, 0] * 2.0, 0.1)}).per_response["y"]
    assert perfect["rmse"] == pytest.approx(0) and perfect["r2"] == pytest.approx(1.0)
    t = truth.truth["y"]
    worse = truth.evaluate({"y": _Stub(lambda X: -X[:, 0] * 2.0, 0.1)})
    r2 = worse.per_response["y"]["r2"]
    assert r2 == pytest.approx(1 - np.sum((2 * t) ** 2) / np.sum((t - t.mean()) ** 2)) and r2 < 0   # negative kept
    assert worse.has_truth and worse.table.shape[0] == 1 and np.isfinite(worse.per_response["y"]["mae"])
    const = es.with_truth(lambda P: {"y": np.ones(len(P))}).evaluate({"y": _Stub(1.0, 0.1)}).per_response["y"]
    assert const["r2_status"] == "sst_zero" and np.isnan(const["r2"])
    with pytest.raises(ad.DesignError):
        es.with_truth(lambda P: {"y": np.ones(3)})


def test_real_gp_against_known_truth_on_evaluation_set():
    X = np.random.default_rng(0).uniform(-1, 1, (25, 3))
    g = GaussianProcess(seed=1).fit(X, smooth(X))
    es = EvaluationSet.build(F(3), 400, seed=2).with_truth(lambda P: {"y": smooth(P)})
    es.check_disjoint(X)
    rep = es.evaluate({"y": g})
    assert rep.per_response["y"]["r2"] > 0.9 and rep.mean_r2 == pytest.approx(rep.per_response["y"]["r2"])


# ------------------------------------------------------------------ candidates
def test_candidates_exclude_existing_and_respect_min_distance():
    region = Region.unit_cube(2)
    existing = np.array([[0.0, 0.0], [0.5, 0.5]])
    P, info = build_candidates(region, n=400, method="sobol", seed=3, existing=existing, min_distance=0.3)
    assert distances_to(P, existing).min() >= 0.3 - 1e-12
    assert info["n_removed_min_distance"] > 0 and info["n_candidates"] == len(P) and info["seed"] == 3
    P2, _ = build_candidates(region, n=400, method="sobol", seed=3, existing=existing, min_distance=0.3)
    assert np.array_equal(P, P2)                                                  # reproducible
    g, ig = build_candidates(region, method="grid", existing=np.array([[0.0, 0.0]]))
    assert ig["n_removed_existing"] == 1 and not any(np.allclose(p, 0) for p in g)
    u, iu = build_candidates(region, points=[[0.1, 0.2], [0.0, 0.0]], existing=[[0.0, 0.0]])
    assert len(u) == 1 and iu["method"] == "user"
    with pytest.raises(ad.DesignError):
        build_candidates(region, min_distance=-1)
    with pytest.raises(ad.DesignError):
        build_candidates(region, method="grid", existing=[[0, 0]], min_distance=10.0)


def test_candidates_respect_constrained_regions():
    r = Region.from_factors(F(2), [({"A": 1, "B": 1}, "<=", 8)])
    P, info = build_candidates(r, n=300, seed=1, existing=[[-1.0, -1.0]])
    assert r.contains(P).all() and 0 < info["acceptance_rate"] <= 1


@pytest.mark.parametrize("scale", [1e-15, 1e-6, 1e9])
def test_r2_and_evaluation_are_scale_aware_not_absolute(scale):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (24, 3))
    y = smooth(X)
    a = gpr_cv_r2(X, y, n_restarts=1, seed=1)[0]
    b = gpr_cv_r2(X, y * scale + 2 * scale, n_restarts=1, seed=1)[0]
    assert b.status == "ok" and b.value == pytest.approx(a.value, abs=5e-3)               # tiny/huge units: same R^2
    assert predictive_r2(np.full(6, 3 * scale), np.full((1, 6), 3 * scale)).status == "sst_zero"
    es = EvaluationSet.build(F(2), 100, seed=1).with_truth(lambda P: {"y": (P[:, 0] + 2) * scale})
    ok = es.evaluate({"y": _Stub(lambda X_: (X_[:, 0] + 2) * scale, 0.0)}).per_response["y"]
    assert ok["r2_status"] == "ok" and ok["r2"] == pytest.approx(1.0)
