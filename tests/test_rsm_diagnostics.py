import numpy as np
import pytest
from scipy import stats

import adore_doe as ad
from adore_doe import rsm
from adore_doe.diagnostics import compute_diagnostics
from adore_doe.rsm import terms as T
from rsm_helpers import Y2, Z2, rand_fit


def fit(seed=0, n=24, noise=0.3, kind="quadratic"):
    Z, y, terms, names = rand_fit(n=n, seed=seed, noise=noise, kind=kind)
    return rsm.fit_arrays(Z, y, terms, names, run_order=np.random.default_rng(seed).permutation(n) + 1), Z, y


def test_formulas_against_manual_leave_one_out():
    r, Z, y = fit(seed=2)
    d = compute_diagnostics(r).table
    n, p = r.n, r.p
    X = r.X
    h = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
    ri = r.resid / (r.sigma * np.sqrt(1 - h))
    assert np.allclose(d.Leverage, h) and np.allclose(d.IntStud, ri)
    for i in (0, 5, 13):                                                       # external = leave-one-out sigma
        keep = np.arange(n) != i
        b = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        s_i = np.sqrt(np.sum((y[keep] - X[keep] @ b) ** 2) / (n - p - 1))
        assert d.ExtStud[i] == pytest.approx(r.resid[i] / (s_i * np.sqrt(1 - h[i])), rel=1e-9)
    assert np.allclose(d.CooksD, ri ** 2 * h / (p * (1 - h)))
    assert np.allclose(d.DFFITS, d.ExtStud * np.sqrt(h / (1 - h)))
    assert list(d.Run) == list(r.run_order) and np.allclose(d.Actual, y) and np.allclose(d.Fitted, r.fitted)


def test_injected_outlier_is_flagged_and_clean_data_is_not():
    rng = np.random.default_rng(3)
    Z = rng.uniform(-1, 1, (30, 2))
    terms = T.universe("quadratic", 2)
    y = T.expand(Z, terms) @ np.array([1, 2, -1, 0.5, 1, -1.0]) + 0.1 * rng.standard_normal(30)
    clean = compute_diagnostics(rsm.fit_arrays(Z, y, terms, ["a", "b"]))
    assert not clean.table.Outlier.any()
    y2 = y.copy(); y2[7] += 3.0
    bad = compute_diagnostics(rsm.fit_arrays(Z, y2, terms, ["a", "b"]))
    assert bad.table.Outlier[7] and bad.table.Outlier.sum() == 1 and bad.table.CooksD[7] == bad.table.CooksD.max()
    assert "outliers=1" in bad.summary()


def test_normality_and_variance_checks():
    r, _, _ = fit(seed=4, n=40)
    d = compute_diagnostics(r)
    W, p = stats.shapiro(d.table.ExtStud.to_numpy())
    assert d.normality["W"] == pytest.approx(W) and d.normality["p"] == pytest.approx(p)   # is Shapiro-Wilk on t_i
    # Calibration (n = 40, p = 10): Shapiro-Wilk on studentised residuals is APPROXIMATE -- measured false alarms
    # ~10 % at the nominal 5 %, power ~75 % against exponential noise.  Assert only the order of magnitude.
    fa, pw = [], []
    for s in range(60):
        Z, y, terms, nm = rand_fit(n=40, seed=s, noise=0.3)
        fa.append(compute_diagnostics(rsm.fit_arrays(Z, y, terms, nm)).normality["p"] < 0.05)
        rng = np.random.default_rng(1000 + s)
        Z2 = rng.uniform(-1, 1, (40, 3))
        y2 = T.expand(Z2, terms) @ np.random.default_rng(s).normal(0, 1, len(terms)) + rng.exponential(1.0, 40)
        pw.append(compute_diagnostics(rsm.fit_arrays(Z2, y2, terms, nm)).normality["p"] < 0.05)
    assert np.mean(fa) < 0.25 and np.mean(pw) > 0.5
    rng = np.random.default_rng(0)
    Z = rng.uniform(-1, 1, (60, 1))
    y = 1 + Z[:, 0] + (0.05 + 0.6 * (Z[:, 0] + 1)) * rng.standard_normal(60)       # variance grows with x
    dh = compute_diagnostics(rsm.fit_arrays(Z, y, T.universe("linear", 1), ["x"]))
    assert dh.variance_trend["rho"] > 0.3 and dh.variance_trend["p"] < 0.05


def test_leverage_one_and_missing_df_are_reported_not_invented():
    Z = np.array([[-1.0], [0.0], [1.0]])
    with pytest.warns(ad.InferenceWarning):
        r = rsm.fit_arrays(Z, np.array([1.0, 3.0, 2.0]), T.universe("quadratic", 1), ["x"])
    d = compute_diagnostics(r)
    assert d.table.ExtStud.isna().all() and d.table.CooksD.isna().all() and any("no residual degrees" in n for n in d.notes)
    assert np.isnan(d.normality["p"]) and d.normality["reason"]
    Z2_ = np.array([[-1.0], [-0.5], [0.0], [0.5], [3.0]])                        # one point with huge leverage
    r2 = rsm.fit_arrays(Z2_, np.array([1.0, 1.4, 2.0, 2.6, 9.0]), T.universe("linear", 1), ["x"])
    d2 = compute_diagnostics(r2).table
    assert d2.HighLeverage.iloc[-1] and d2.Leverage.iloc[-1] > 0.8


def test_textbook_data_diagnostics_run():
    r = rsm.fit_arrays(Z2, Y2, T.universe("quadratic", 2), ["x1", "x2"])
    d = r.diagnostics()
    assert len(d.table) == 13 and np.isfinite(d.table.ExtStud).all() and not d.table.Outlier.any()
    assert d.table.Leverage.sum() == pytest.approx(6)


def test_diagnostic_plots_draw_the_computed_numbers(tmp_path):
    import matplotlib.pyplot as plt
    r, Z, y = fit(seed=5)
    d = r.diagnostics()
    ax = d.plot_normal(save_path=tmp_path / "n.png")
    xs, ys = ax.collections[0].get_offsets()[:, 0], ax.collections[0].get_offsets()[:, 1]
    assert np.allclose(np.sort(ys), np.sort(d.table.ExtStud)) and np.all(np.diff(xs) > 0) and (tmp_path / "n.png").exists()
    ax = d.plot_residual_vs_predicted()
    assert np.allclose(ax.collections[0].get_offsets()[:, 0], d.table.Fitted) and ax.get_xlabel() == "Predicted"
    ax = d.plot_predicted_vs_actual()
    assert ax.get_xlabel() == "Actual" and ax.get_ylabel() == "Predicted" and len(ax.lines) == 1
    ax = d.plot_residual_vs_run()
    assert sorted(ax.collections[0].get_offsets()[:, 0]) == sorted(r.run_order)
    ax = d.plot_residual_vs_factor("x2")
    assert np.allclose(ax.collections[0].get_offsets()[:, 0], Z[:, 1])
    with pytest.raises(ValueError):
        d.plot_residual_vs_factor("zzz")
    noorder = rsm.fit_arrays(Z, y, T.universe("quadratic", 3), ["x1", "x2", "x3"]).diagnostics()
    with pytest.raises(ValueError, match="run order"):
        noorder.plot_residual_vs_run()                                              # no run order known -> refuse
    plt.close("all")


def test_cross_check_influence_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    r, Z, y = fit(seed=8, n=26)
    d = compute_diagnostics(r).table
    infl = sm.OLS(y, r.X).fit().get_influence()
    assert np.allclose(d.Leverage, infl.hat_matrix_diag)
    assert np.allclose(d.IntStud, infl.resid_studentized_internal)
    assert np.allclose(d.ExtStud, infl.resid_studentized_external)
    assert np.allclose(d.CooksD, infl.cooks_distance[0]) and np.allclose(d.DFFITS, infl.dffits[0])
