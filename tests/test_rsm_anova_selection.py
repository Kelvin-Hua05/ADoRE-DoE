import numpy as np
import pytest
from scipy import stats

import adore_doe as ad
from adore_doe import design, rsm
from adore_doe.rsm import terms as T
from adore_doe.rsm.anova import pure_error, replicate_groups
from adore_doe.rsm.selection import select_terms
from rsm_helpers import Y1, Y2, Z1, Z2, rand_fit  # noqa: E402


def sub(res):
    return res.anova().set_index("Source")


# ------------------------------------------------------------------ ANOVA identities
def test_anova_sums_of_squares_and_f_tests():
    Z, y, terms, names = rand_fit(n=24, seed=4)
    r = rsm.fit_arrays(Z, y, terms, names)
    a = sub(r)
    assert a.loc["Model", "SS"] + a.loc["Residual", "SS"] == pytest.approx(a.loc["Cor Total", "SS"])
    assert a.loc["Model", "df"] + a.loc["Residual", "df"] == a.loc["Cor Total", "df"] == r.n - 1
    mse = r.rss / r.df_resid
    assert a.loc["Model", "F"] == pytest.approx((a.loc["Model", "SS"] / (r.p - 1)) / mse)
    assert a.loc["Model", "p"] == pytest.approx(stats.f.sf(a.loc["Model", "F"], r.p - 1, r.df_resid))
    coef = r.coef_table().set_index("Term")
    for j, t in enumerate(r.terms):
        if sum(t) == 0:
            continue
        nm = r.term_names[j]
        keep = [i for i in range(r.p) if i != j]                             # partial SS = RSS(without) - RSS(full)
        X = r.X[:, keep]
        rss_wo = np.sum((r.y - X @ np.linalg.lstsq(X, r.y, rcond=None)[0]) ** 2)
        assert a.loc[nm, "SS"] == pytest.approx(rss_wo - r.rss) and a.loc[nm, "df"] == 1
        assert a.loc[nm, "F"] == pytest.approx(coef.loc[nm, "t"] ** 2)          # F = t^2 for a 1-df term
        assert a.loc[nm, "p"] == pytest.approx(coef.loc[nm, "p"])


# ------------------------------------------------------------------ lack of fit
def test_lack_of_fit_with_replicates_matches_independent_pure_error():
    r = rsm.fit_arrays(Z2, Y2, T.universe("quadratic", 2), ["x1", "x2"])
    a = sub(r)
    sspe = float(np.sum((Y2[8:] - Y2[8:].mean()) ** 2))                        # the 5 centre replicates
    assert a.loc["Pure Error", "SS"] == pytest.approx(sspe) and a.loc["Pure Error", "df"] == 4
    assert a.loc["Lack of Fit", "SS"] == pytest.approx(r.rss - sspe) and a.loc["Lack of Fit", "df"] == 3
    F = (a.loc["Lack of Fit", "SS"] / 3) / (sspe / 4)
    assert a.loc["Lack of Fit", "F"] == pytest.approx(F) and a.loc["Lack of Fit", "p"] == pytest.approx(stats.f.sf(F, 3, 4))
    assert a.loc["Lack of Fit", "p"] > 0.05                                    # the quadratic model is adequate here


def test_lack_of_fit_is_not_available_without_replicates():
    fs = [ad.Factor("A", 0, 1), ad.Factor("B", 0, 1), ad.Factor("C", 0, 1)]
    d = design.ccd(fs, center_points=1)                                       # ADR-010: one centre point only
    y = d.coded[:, 0] + d.coded[:, 1] ** 2 + 0.1 * np.random.default_rng(0).standard_normal(d.n_runs)
    r = rsm.fit_arrays(d.coded, y, T.universe("quadratic", 3), ["A", "B", "C"])
    row = r.anova().set_index("Source").loc["Lack of Fit"]
    assert np.isnan(row["SS"]) and np.isnan(row["p"]) and "NOT AVAILABLE" in row["note"] and "no replicated" in row["note"]
    assert sub(r).loc["Pure Error", "df"] == 0
    r3 = rsm.fit_arrays(*[ad_ for ad_ in (design.ccd(fs, center_points=3).coded,)], np.arange(17.0) ** 0.5 + np.random.default_rng(1).normal(0, .1, 17),
                        T.universe("quadratic", 3), ["A", "B", "C"])
    assert np.isfinite(sub(r3).loc["Lack of Fit", "SS"])                       # 3 centre points -> LOF exists


def test_lack_of_fit_unavailable_when_model_fits_every_distinct_point():
    Z = np.array([[-1.0], [-1.0], [0.0], [0.0], [1.0], [1.0]])                 # 3 distinct points, quadratic p = 3
    r = rsm.fit_arrays(Z, np.array([1.0, 1.2, 0.0, 0.1, 1.1, 0.9]), T.universe("quadratic", 1), ["x"])
    row = sub(r).loc["Lack of Fit"]
    assert np.isnan(row["SS"]) and row["df"] == 0 and "df_LOF = 0" in row["note"]
    assert sub(r).loc["Pure Error", "df"] == 3


def test_lack_of_fit_is_calibrated_and_detects_a_wrong_model():
    """Simulation (1000 data sets, 7 levels x 3 replicates, quadratic model): a CORRECT model is rejected about
    5 % of the time at alpha = 0.05 (measured 4.6 % with 3000 runs); a cubic truth is always detected."""
    z = np.repeat(np.linspace(-1, 1, 7), 3)[:, None]
    terms = T.universe("quadratic", 1)
    false_alarm, power = [], []
    for s in range(1000):
        rng = np.random.default_rng(10_000 + s)
        ok = rsm.fit_arrays(z, 1 + z[:, 0] + 0.05 * rng.standard_normal(len(z)), terms, ["x"])
        false_alarm.append(sub(ok).loc["Lack of Fit", "p"] < 0.05)
        if s < 100:
            bad = rsm.fit_arrays(z, 2 * z[:, 0] ** 3 + 0.05 * rng.standard_normal(len(z)), terms, ["x"])
            power.append(sub(bad).loc["Lack of Fit", "p"] < 0.01)
    assert 0.025 < np.mean(false_alarm) < 0.075 and np.mean(power) == 1.0


def test_replicate_groups_and_pure_error_helpers():
    Z = np.array([[0, 0], [1, 1], [0, 0], [0, 0], [-1, 0]], float)
    g = replicate_groups(Z)
    assert g[0] == g[2] == g[3] and len(set(g)) == 3
    ss, df, G = pure_error(Z, np.array([1.0, 5.0, 2.0, 3.0, 9.0]))
    assert df == 2 and G == 3 and ss == pytest.approx(np.sum((np.array([1, 2, 3]) - 2.0) ** 2))


def test_anova_without_residual_df_is_marked():
    Z = np.array([[-1.0], [0.0], [1.0]])
    with pytest.warns(ad.InferenceWarning):
        r = rsm.fit_arrays(Z, np.array([1.0, 3.0, 2.0]), T.universe("quadratic", 1), ["x"])
    row = sub(r).loc["Residual"]
    assert row["df"] == 0 and "no residual df" in row["note"]
    assert np.isnan(sub(r).loc["Model", "F"])


# ------------------------------------------------------------------ term selection
NAMES2 = ["A", "B"]


def names(res, f=NAMES2):
    return {T.term_name(t, f) for t in res.terms}


def test_textbook_data_stepwise_p_selection_and_history():
    r = select_terms(Z2, Y2, "quadratic", "stepwise", "p", alpha_in=0.10, alpha_out=0.10)
    assert names(r) == {"Intercept", "A", "B", "A^2", "B^2"}                  # A*B (p = 0.1025) is not kept
    assert r.stop_reason.startswith("converged") and r.config["criterion"] == "p"
    assert all(h["action"] in ("add", "remove") and "terms" in h for h in r.history)
    assert any(h["parents_added"] for h in r.history)                         # A^2 entered together with A
    r2 = select_terms(Z2, Y2, "quadratic", "stepwise", "p", alpha_in=0.11, alpha_out=0.11)
    assert "A*B" in names(r2)                                                  # 0.1025 < 0.11: now kept
    bic = select_terms(Z2, Y2, "quadratic", "stepwise", "bic")
    assert "A*B" in names(bic)                                                 # criteria can differ: reported, not hidden


@pytest.mark.parametrize("method", ["forward", "backward", "stepwise"])
@pytest.mark.parametrize("criterion", ["p", "aicc", "bic"])
def test_strong_hierarchy_always_holds_and_history_is_consistent(method, criterion):
    for seed in range(12):
        Z, y, terms, _ = rand_fit(n=22, k=3, seed=seed, noise=0.8)
        r = select_terms(Z, y, "quadratic", method, criterion)
        assert T.missing_parents(r.terms) == {}
        assert (0, 0, 0) in r.terms and len(set(r.terms)) == len(r.terms)
        assert r.history == [] or r.history[-1]["terms"] == r.terms
        if criterion in ("aicc", "bic") and method == "forward":
            vals = [h["value"] for h in r.history if h["action"] == "add"]
            assert all(b < a for a, b in zip(vals, vals[1:]))                # every accepted move improves


def test_hierarchy_none_can_keep_an_interaction_without_its_parents():
    rng = np.random.default_rng(0)
    Z = rng.uniform(-1, 1, (30, 2))
    y = 3 * Z[:, 0] * Z[:, 1] + 0.05 * rng.standard_normal(30)
    none = select_terms(Z, y, "interaction", "forward", "p", hierarchy="none")
    first = none.history[0]
    assert first["term"] == (1, 1) and first["parents_added"] == [] and first["terms"] == [(0, 0), (1, 1)]
    assert T.missing_parents(first["terms"]) != {}                         # the first model violates hierarchy
    strong = select_terms(Z, y, "interaction", "forward", "p", hierarchy="strong")
    assert {(1, 1), (1, 0), (0, 1)} <= set(strong.terms) and T.missing_parents(strong.terms) == {}
    assert strong.history[0]["term"] == (1, 1) and sorted(strong.history[0]["parents_added"]) == [(0, 1), (1, 0)]


def test_selection_finds_the_true_terms_and_respects_force():
    rng = np.random.default_rng(1)
    Z = rng.uniform(-1, 1, (30, 3))
    y = 1 + 2 * Z[:, 0] - 3 * Z[:, 1] ** 2 + 0.05 * rng.standard_normal(30)
    r = select_terms(Z, y, "quadratic", "stepwise", "p")
    assert {(1, 0, 0), (0, 1, 0), (0, 2, 0)} <= set(r.terms) and (1, 0, 1) not in r.terms
    f = select_terms(Z, y, "quadratic", "stepwise", "p", force=[(0, 0, 1)])
    assert (0, 0, 1) in f.terms                                              # forced term is never removed
    b = select_terms(Z, y, "quadratic", "backward", "p")
    assert b.history and b.history[0]["action"] == "remove" and (1, 0, 0) in b.terms
    assert select_terms(Z, y, "quadratic", "forward", "p").history[0]["action"] == "add"


def test_selection_uses_a_provided_start_model_and_validates_arguments():
    Z, y, _, _ = rand_fit(n=25, k=2, seed=3)
    r = select_terms(Z, y, "quadratic", "forward", "bic", start=[(1, 0), (0, 1)])
    assert {(1, 0), (0, 1)} <= set(r.terms)
    for kw in (dict(method="best"), dict(criterion="r2"), dict(hierarchy="weak"), dict(alpha_in=0.0), dict(alpha_out=1.5)):
        with pytest.raises(ValueError):
            select_terms(Z, y, "quadratic", **kw)


def test_selection_never_adds_a_term_that_leaves_no_residual_df():
    Z = np.array([[-1.0, -1], [1, -1], [-1, 1], [1, 1], [0, 0], [0.5, 0.2]])
    y = np.array([1.0, 2.0, 2.5, 5.0, 2.0, 2.4])
    r = select_terms(Z, y, "quadratic", "forward", "p", alpha_in=0.99)
    assert len(r.terms) <= len(y) - 1


def test_rsm_fit_with_selection_from_a_dataset():
    fs = [ad.Factor("A", 0, 10), ad.Factor("B", 0, 10), ad.Factor("C", 0, 10)]
    d = design.ccd(fs, center_points=3)
    rng = np.random.default_rng(2)
    d.add_response("y", 4 + 3 * d.coded[:, 0] - 2 * d.coded[:, 1] ** 2 + 0.05 * rng.standard_normal(d.n_runs))
    r = rsm.fit(d, "y", selection={"method": "stepwise", "criterion": "p", "alpha_in": 0.1, "alpha_out": 0.1})
    assert r.selection is not None and r.selection.stop_reason and r.p < 10
    assert T.missing_parents(r.terms) == {} and "A" in names(r, ["A", "B", "C"])
    assert r.r2 > 0.99


def test_selected_terms_and_equations_use_the_natural_factor_order():
    fs = [ad.Factor(c, 0, 1) for c in ("Width", "Spacing", "Thickness")]
    d = design.ccd(fs, center_points=3)
    y = 1 + d.coded[:, 0] - 2 * d.coded[:, 1] + 0.5 * d.coded[:, 2] ** 2
    r = rsm.fit(d, "y" if False else _add(d, y), selection={"method": "stepwise", "criterion": "p"})
    assert r.term_names[:4] == ["Intercept", "Width", "Spacing", "Thickness"]            # not reversed
    eq = rsm.fit(d, "y", model="interaction").equation()
    assert eq.index("Width") < eq.index("Spacing") < eq.index("Thickness")
    assert T.closure([(0, 1, 0), (1, 0, 0)], 3) == [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    full = T.universe("quadratic", 3)
    assert sorted(full, key=T.order_key) == full                                      # same order as ModelSpec


def _add(d, y):
    d.add_response("y", y)
    return "y"
