import numpy as np
import pytest
from scipy import stats

import adore_doe as ad
from adore_doe import design, rsm
from adore_doe.rsm import terms as T
from adore_doe.rsm.model import info_criteria, natural_polynomial

from rsm_helpers import A, Y1, Y2, Z1, Z2, rand_fit  # noqa: E402


# ------------------------------------------------------------------ textbook examples
def test_montgomery_second_order_coefficients_reproduce_the_published_model():
    r = rsm.fit_arrays(Z2, Y2, T.universe("quadratic", 2), ["x1", "x2"])
    got = dict(zip(r.term_names, r.beta))
    assert got["Intercept"] == pytest.approx(79.94, abs=5e-3)
    assert got["x1"] == pytest.approx(0.995, abs=5e-3) and got["x2"] == pytest.approx(0.515, abs=5e-3)
    assert got["x1^2"] == pytest.approx(-1.376, abs=5e-3) and got["x2^2"] == pytest.approx(-1.001, abs=5e-3)
    assert got["x1*x2"] == pytest.approx(0.25, abs=1e-9)
    assert r.n == 13 and r.p == 6 and r.df_resid == 7 and 0.95 < r.r2 < 0.99


def test_montgomery_first_order_with_centre_points_anova():
    r = rsm.fit_arrays(Z1, Y1, [(0, 0), (1, 0), (0, 1)], ["x1", "x2"])
    assert np.allclose(r.beta, [40.4444, 0.775, 0.325], atol=5e-5)
    at = r.anova().set_index("Source")
    assert at.loc["Pure Error", "SS"] == pytest.approx(0.1720, abs=5e-5) and at.loc["Pure Error", "df"] == 4
    assert at.loc["x1", "SS"] == pytest.approx(2.4025, abs=5e-5) and at.loc["x2", "SS"] == pytest.approx(0.4225, abs=5e-5)
    # curvature SS = (ybar_f - ybar_c)^2 * n_f n_c / (n_f + n_c) = 0.0027 (independent formula)
    yf, yc = Y1[:4], Y1[4:]
    assert (yf.mean() - yc.mean()) ** 2 * 4 * 5 / 9 == pytest.approx(0.0027, abs=5e-5)


# ------------------------------------------------------------------ OLS against independent computations
def test_ols_matches_numpy_and_scipy_formulas():
    Z, y, terms, names = rand_fit()
    r = rsm.fit_arrays(Z, y, terms, names)
    X = T.expand(Z, terms)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    n, p = X.shape
    s2 = np.sum((y - X @ beta) ** 2) / (n - p)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * s2)
    assert np.allclose(r.beta, beta) and np.allclose(r.se, se) and r.df_resid == n - p
    assert np.allclose(r.t, beta / se) and np.allclose(r.pvalue, 2 * stats.t.sf(np.abs(beta / se), n - p))
    tc = stats.t.ppf(0.975, n - p)
    assert np.allclose(r.ci, np.column_stack([beta - tc * se, beta + tc * se]))
    assert r.sigma == pytest.approx(np.sqrt(s2)) and r.rss == pytest.approx(np.sum((y - X @ beta) ** 2))
    tab = r.coef_table()
    assert list(tab.columns) == ["Term", "Coefficient", "df", "StdErr", "CI_low", "CI_high", "t", "p", "VIF"]


def test_fit_statistics_identities_press_and_leverage():
    Z, y, terms, names = rand_fit(seed=3)
    r = rsm.fit_arrays(Z, y, terms, names)
    n, p = r.n, r.p
    sst = np.sum((y - y.mean()) ** 2)
    assert r.r2 == pytest.approx(1 - r.rss / sst)
    assert r.r2_adj == pytest.approx(1 - (r.rss / (n - p)) / (sst / (n - 1)))
    assert r.leverage.sum() == pytest.approx(p) and np.all((r.leverage > 0) & (r.leverage < 1))
    # PRESS equals an explicit leave-one-out refit
    loo = []
    for i in range(n):
        keep = np.arange(n) != i
        b = np.linalg.lstsq(T.expand(Z[keep], terms), y[keep], rcond=None)[0]
        loo.append(y[i] - T.expand(Z[i:i + 1], terms) @ b)
    assert r.press == pytest.approx(float(np.sum(np.square(loo))), rel=1e-9)
    assert r.r2_pred == pytest.approx(1 - r.press / sst) and r.r2_pred < r.r2
    fs = r.fit_statistics()
    assert fs["Std. Dev."] == pytest.approx(r.sigma) and fs["C.V. %"] == pytest.approx(100 * r.sigma / y.mean())
    assert r.adeq_precision == pytest.approx((r.fitted.max() - r.fitted.min()) / np.sqrt(p * r.sigma2 / n))
    ic = info_criteria(r.rss, n, p)
    assert ic["aic"] == pytest.approx(n * np.log(r.rss / n) + 2 * p)
    assert ic["aicc"] == pytest.approx(ic["aic"] + 2 * p * (p + 1) / (n - p - 1))
    assert ic["bic"] == pytest.approx(n * np.log(r.rss / n) + p * np.log(n))
    assert ic["aic_sm"] - ic["aic"] == pytest.approx(n * (1 + np.log(2 * np.pi)))


def test_vif_equals_one_over_one_minus_r2_of_each_column():
    Z, y, terms, names = rand_fit(seed=5)
    r = rsm.fit_arrays(Z, y, terms, names)
    X = T.expand(Z, terms)
    for j in range(1, X.shape[1]):
        others = np.delete(X, j, axis=1)                        # includes the intercept column
        b = np.linalg.lstsq(others, X[:, j], rcond=None)[0]
        r2 = 1 - np.sum((X[:, j] - others @ b) ** 2) / np.sum((X[:, j] - X[:, j].mean()) ** 2)
        assert r.vif[j] == pytest.approx(1 / (1 - r2), rel=1e-6)
    assert np.isnan(r.vif[0])                                   # no VIF for the intercept
    d = design.ccd([ad.Factor("A", 0, 1), ad.Factor("B", 0, 1), ad.Factor("C", 0, 1)], center_points=3)
    rc = rsm.fit_arrays(d.coded, np.random.default_rng(0).normal(size=d.n_runs), T.universe("quadratic", 3), ["A", "B", "C"])
    assert np.nanmax(rc.vif) < 3                                 # a good design has small VIFs


def test_prediction_and_its_standard_error():
    Z, y, terms, names = rand_fit(seed=7)
    r = rsm.fit_arrays(Z, y, terms, names)
    assert np.allclose(r.predict(Z), r.fitted)
    P = np.random.default_rng(1).uniform(-1, 1, (6, 3))
    m, se = r.predict(P, return_se=True)
    F = T.expand(P, terms)
    X = T.expand(Z, terms)
    assert np.allclose(se, r.sigma * np.sqrt(np.einsum("ij,jk,ik->i", F, np.linalg.inv(X.T @ X), F)))
    assert np.allclose(m, F @ r.beta)


# ------------------------------------------------------------------ explicit equations
def test_coded_and_natural_equations_are_consistent():
    fs = [ad.Factor("Width", 10, 50), ad.Factor("Spacing", 5, 30), ad.Factor("T", 20, 80)]
    Z, y, terms, _ = rand_fit(n=25, seed=2)
    r = rsm.fit_arrays(Z, y, terms, [f.name for f in fs], factors=fs, response="C")
    eq = r.equation(coded=True)
    assert eq.startswith("C = ") and "Width*Spacing" in eq and "Width^2" in eq
    nat = natural_polynomial({t: float(b) for t, b in zip(terms, r.beta)}, fs)
    P = np.random.default_rng(3).uniform(-1, 1, (8, 3))
    X_nat = np.column_stack([f.from_coded(P[:, j]) for j, f in enumerate(fs)])
    val = sum(c * np.prod(np.power(X_nat, np.array(e)), axis=1) for e, c in nat.items())
    assert np.allclose(val, r.predict(P), rtol=1e-9, atol=1e-9)           # equation in actual units == model
    assert "C = " in r.equation(coded=False)
    with pytest.raises(ValueError):
        rsm.fit_arrays(Z, y, terms, ["a", "b", "c"]).equation(coded=False)
    assert natural_polynomial({(0, 0, 0): 1.0}, fs) == {(0, 0, 0): 1.0}


# ------------------------------------------------------------------ identifiability
def test_non_identifiable_models_are_reported_not_pseudo_inverted():
    fs = [ad.Factor(c, 0, 1) for c in "ABC"]
    d = ad.screening.fractional_factorial(fs, generators=["C=AB"])           # 4 runs: A*B aliased with C
    y = np.arange(4.0)
    with pytest.raises(ad.RankDeficientError, match="not identifiable"):
        rsm.fit_arrays(d.coded, y, T.universe("interaction", 3), ["A", "B", "C"])
    with pytest.raises(ad.RankDeficientError, match="runs"):
        rsm.fit_arrays(d.coded[:3], y[:3], T.universe("quadratic", 3), ["A", "B", "C"])
    X = d.coded
    r = rsm.fit_arrays(X, y, T.universe("interaction", 3), ["A", "B", "C"], on_rank_deficient="drop")
    assert r.dropped_terms and r.p < 7 and np.isfinite(r.beta).all()


def test_no_residual_df_gives_nan_inference_and_a_warning():
    Z = np.array([[-1.0], [0.0], [1.0]])
    with pytest.warns(ad.InferenceWarning, match="No residual degrees of freedom"):
        r = rsm.fit_arrays(Z, np.array([1.0, 3.0, 2.0]), T.universe("quadratic", 1), ["x"])
    assert r.df_resid == 0 and np.isnan(r.se).all() and np.isnan(r.pvalue).all() and np.isnan(r.sigma)
    assert np.isnan(r.r2_pred) and np.isinf(r.press) and r.r2 == pytest.approx(1.0)


# ------------------------------------------------------------------ terms and hierarchy
def test_terms_universe_parents_and_closure():
    assert [len(T.universe(k, 3)) for k in ("linear", "interaction", "quadratic")] == [4, 7, 10]
    assert T.parents((1, 1, 0)) == [(1, 0, 0), (0, 1, 0)] and T.parents((0, 2, 0)) == [(0, 1, 0)]
    assert T.parents((1, 0, 0)) == [] and T.parents((0, 0, 0)) == []
    cl = T.closure([(1, 1, 0), (0, 0, 2)], 3)
    assert {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 0, 2)} == set(cl)
    assert T.missing_parents([(0, 0, 0), (1, 1, 0)]) == {(1, 1, 0): [(1, 0, 0), (0, 1, 0)]}
    assert T.missing_parents(T.universe("quadratic", 3)) == {}
    assert T.children((1, 0), T.universe("quadratic", 2)) == [(1, 1), (2, 0)]
    assert T.term_name((1, 1, 0), ["A", "B", "C"]) == "A*B" and T.term_name((0, 0, 2), ["A", "B", "C"]) == "C^2"


# ------------------------------------------------------------------ dataset wrapper
def test_fit_from_dataset_uses_training_runs_only():
    fs = [ad.Factor("A", 0, 10), ad.Factor("B", 0, 10)]
    d = design.ccd(fs, center_points=3)
    y = 5 + 2 * d.coded[:, 0] - d.coded[:, 1] ** 2 + 0.01 * np.random.default_rng(0).standard_normal(d.n_runs)
    d.add_response("y", y)
    d.append_runs(np.array([[0.3, 0.3]]), source="validation", values={"y": [999.0]})
    r = rsm.fit(d, "y")
    assert r.n == d.n_runs - 1 and r.response == "y" and abs(r.beta[0] - 5) < 0.1
    assert r.factors is d.factors and r.run_order is not None
    assert rsm.fit(d, "y", model="linear").p == 3
    assert rsm.fit(d, "y", terms=[(0, 0), (1, 0)]).p == 2
    with pytest.raises(ad.DesignError):
        rsm.fit(d, "nope")
    e = design.ccd(fs, center_points=1)
    e.add_response("z", [1.0, 2.0] + [np.nan] * (e.n_runs - 2))
    with pytest.raises(ad.DesignError, match="at least 3"):
        rsm.fit(e, "z")


# ------------------------------------------------------------------ statsmodels cross-check (runs where installed)
def test_cross_check_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    Z, y, terms, names = rand_fit(n=22, seed=11)
    r = rsm.fit_arrays(Z, y, terms, names)
    X = T.expand(Z, terms)
    res = sm.OLS(y, X).fit()
    assert np.allclose(r.beta, res.params) and np.allclose(r.se, res.bse)
    assert np.allclose(r.t, res.tvalues) and np.allclose(r.pvalue, res.pvalues)
    assert np.allclose(r.ci, res.conf_int(alpha=0.05))
    assert r.r2 == pytest.approx(res.rsquared) and r.r2_adj == pytest.approx(res.rsquared_adj)
    assert r.sigma2 == pytest.approx(res.mse_resid)
    ic = info_criteria(r.rss, r.n, r.p)
    assert ic["aic_sm"] == pytest.approx(res.aic) and ic["bic_sm"] == pytest.approx(res.bic)
    infl = res.get_influence()
    assert np.allclose(r.leverage, infl.hat_matrix_diag)
    assert r.press == pytest.approx(float(np.sum(np.asarray(infl.resid_press) ** 2)))
