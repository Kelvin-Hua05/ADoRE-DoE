import numpy as np
import pytest

import adore_doe as ad
from adore_doe.active_learning.acquisition import STRATEGIES, Selection, select_batch
from adore_doe.active_learning.candidates import distances_to
from adore_doe.active_learning.gpr import GaussianProcess
from adore_doe.core.seed import derive_seed, resolve_seed
from adore_doe.design.criteria import i_criterion
from adore_doe.design.model import ModelSpec
from adore_doe.design.moment import IntegrationDomain
from adore_doe.design.region import Region


def f1(Z):
    return np.sin(2 * Z[:, 0]) + 0.5 * Z[:, 1] ** 2 + Z[:, 0] * Z[:, -1]


def setup(n=12, k=2, nc=60, seed=0, two=False):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, k))
    C = Region.unit_cube(k).sample(nc, seed=seed + 1)[0]
    m = {"a": GaussianProcess(seed=1).fit(X, f1(X))}
    if two:
        m["b"] = GaussianProcess(seed=2).fit(X, np.cos(3 * X[:, 0]) + X[:, 1])
    return X, C, m


def iv_after(model, X_obs, y_obs, adds, P):
    g = GaussianProcess(seed=0).fit(np.vstack([X_obs, adds]), np.append(y_obs, 100.0 + np.arange(len(adds))),
                                    fixed_theta=model.theta)
    return g, np.mean(np.diag(g.posterior_cov(P))) / g.y_std ** 2


# ------------------------------------------------------------------ exactness against brute force
@pytest.mark.parametrize("strategy", ["ivr", "max_variance"])
def test_greedy_gpr_batch_equals_sequential_refit_with_fixed_hyperparameters(strategy):
    X, C, models = setup(n=10, nc=40)
    gp = models["a"]
    y = f1(X)
    seed, n_int = 3, 128
    sel = select_batch(C, 3, strategy, models=models, existing=X, seed=seed, n_integration=n_int)
    P = IntegrationDomain(Region.unit_cube(2), m=n_int, seed=derive_seed(resolve_seed(seed), "integration")).points(n_int)[0]
    adds = np.zeros((0, 2))
    for step in range(3):
        best, best_score = None, -np.inf
        for j in range(len(C)):
            if any(np.allclose(C[j], a) for a in adds):
                continue
            g, iv = iv_after(gp, X, y, np.vstack([adds, C[j]]), P)
            if strategy == "ivr":
                g0, iv0 = iv_after(gp, X, y, adds, P) if len(adds) else (gp, np.mean(np.diag(gp.posterior_cov(P))) / gp.y_std ** 2)
                score = iv0 - iv
            else:
                g0 = GaussianProcess(seed=0).fit(np.vstack([X, adds]), np.append(y, 100.0 + np.arange(len(adds))),
                                                 fixed_theta=gp.theta) if len(adds) else gp
                score = float(g0.posterior_cov(C[j][None, :])[0, 0] / g0.y_std ** 2)
            if score > best_score:
                best, best_score = j, score
        assert sel.indices[step] == best
        assert sel.scores[step] == pytest.approx(best_score, rel=1e-6, abs=1e-10)
        adds = np.vstack([adds, C[best]])


def test_i_optimal_augmentation_equals_exhaustive_best_addition_and_ignores_responses():
    X, C, _ = setup(n=12, nc=50)
    sel = select_batch(C, 3, "i_optimal", existing=X, seed=2)                    # no models needed
    spec = ModelSpec("quadratic", 2)
    B = IntegrationDomain(Region.unit_cube(2), m=512, seed=derive_seed(resolve_seed(2), "integration")).matrix(spec).B
    cur = X.copy()
    for step in range(3):
        vals = [i_criterion(spec.expand(np.vstack([cur, c])), B) for c in C]
        for j in sel.indices[:step]:
            vals[j] = np.inf
        assert sel.indices[step] == int(np.argmin(vals))
        cur = np.vstack([cur, C[sel.indices[step]]])
    assert sel.info["responses_used"] is False and sel.info["poly_model"] == "quadratic"
    # the same picks whatever the responses are (design-based, not response-driven)
    other = select_batch(C, 3, "i_optimal", existing=X, seed=2, models=setup(n=12, nc=50, two=True)[2])
    assert other.indices == sel.indices


def test_max_variance_first_pick_is_the_largest_normalised_variance():
    X, C, models = setup(two=True)
    sel = select_batch(C, 1, "max_variance", models=models, seed=1)
    nv = [models[n].predict(C)[1] ** 2 / models[n].data_variance for n in models]
    assert sel.indices[0] == int(np.argmax(np.mean(nv, axis=0)))
    assert sel.scores[0] == pytest.approx(np.max(np.mean(nv, axis=0)))
    assert set(sel.info["per_pick_scores"][0]) == {"a", "b"}


# ------------------------------------------------------------------ multi-response behaviour
@pytest.mark.parametrize("strategy", ["ivr", "max_variance"])
def test_units_of_a_response_cannot_dominate(strategy):
    X, C, m1 = setup(two=True)
    y_b = np.cos(3 * X[:, 0]) + X[:, 1]
    m2 = {"a": m1["a"], "b": GaussianProcess(seed=2).fit(X, y_b * 1e6 + 5e6)}      # same shape, other units
    s1 = select_batch(C, 3, strategy, models=m1, seed=1)
    s2 = select_batch(C, 3, strategy, models=m2, seed=1)
    assert s1.indices == s2.indices and np.allclose(s1.scores, s2.scores, rtol=1e-3)


def test_weights_and_max_aggregation():
    X, C, models = setup(two=True)
    only_a = select_batch(C, 3, "ivr", models={"a": models["a"]}, seed=1)
    w1 = select_batch(C, 3, "ivr", models=models, weights={"a": 1.0, "b": 0.0}, seed=1)
    assert w1.indices == only_a.indices
    only_b = select_batch(C, 3, "ivr", models={"b": models["b"]}, seed=1)
    w0 = select_batch(C, 3, "ivr", models=models, weights={"a": 0.0, "b": 1.0}, seed=1)
    assert w0.indices == only_b.indices
    mx = select_batch(C, 1, "max_variance", models=models, aggregate="max", seed=1)
    worst = np.max([models[n].predict(C)[1] ** 2 / models[n].data_variance for n in models], axis=0)
    assert mx.indices[0] == int(np.argmax(worst)) and mx.aggregate == "max"
    for bad in ({"a": 0.5, "b": 0.6}, {"a": 1.0}):
        with pytest.raises(ad.DesignError):
            select_batch(C, 1, "ivr", models=models, weights=bad, seed=1)


# ------------------------------------------------------------------ batches, distances, reproducibility
def test_greedy_batch_is_distinct_and_not_clustered():
    X, C, models = setup(n=8, nc=300)
    sel = select_batch(C, 4, "ivr", models=models, existing=X, seed=1)
    assert len(set(sel.indices)) == 4 and sel.points.shape == (4, 2)
    sc = select_batch(C, 1, "ivr", models=models, seed=1)                        # best single-round scores
    one = models["a"].posterior_cov(np.vstack([C, np.random.default_rng(0).uniform(-1, 1, (64, 2))]))
    naive = np.argsort(-(one[64 * 0 + len(C):, :len(C)] ** 2).mean(axis=0))[:4]
    pd = lambda P: min(np.linalg.norm(P[i] - P[j]) for i in range(len(P)) for j in range(i))
    assert pd(C[sel.indices]) >= pd(C[naive]) - 1e-12                              # spread out, not stacked


def test_min_distance_to_existing_and_within_batch_and_truncation():
    X, C, models = setup(n=8, nc=200)
    for strat in ("ivr", "max_variance", "i_optimal", "random"):
        sel = select_batch(C, 4, strat, models=models, existing=X, min_distance=0.3, seed=1)
        assert distances_to(sel.points, X).min() >= 0.3 - 1e-12
        D = np.linalg.norm(sel.points[:, None] - sel.points[None], axis=2) + np.eye(len(sel)) * 9
        assert D.min() >= 0.3 - 1e-12 and len(sel) == 4
    big = select_batch(C, 6, "max_variance", models=models, existing=X, min_distance=1.2, seed=1)
    assert len(big) < 6 and big.info["truncated"] and big.info["min_distance"] == 1.2
    tiny = select_batch(C[:3], 5, "random", seed=0)
    assert len(tiny) == 3 and tiny.info["truncated"]


def test_reproducible_deterministic_strategies_and_seeded_random():
    X, C, models = setup()
    for strat in ("ivr", "max_variance", "i_optimal"):
        a = select_batch(C, 3, strat, models=models, existing=X, seed=1)
        b = select_batch(C, 3, strat, models=models, existing=X, seed=2)
        assert a.indices == b.indices                                           # no randomness (integration points aside)
    r1, r2, r3 = (select_batch(C, 4, "random", seed=s).indices for s in (5, 5, 6))
    assert r1 == r2 and r1 != r3


def test_with_fixed_hyperparameters_the_picks_do_not_depend_on_the_response_values():
    """The latent posterior covariance is independent of y.  Two GPs with the SAME hyper-parameters but
    completely different responses must therefore produce identical (normalised) scores and picks."""
    X, C, models = setup(nc=50)
    th = models["a"].theta
    other_y = np.random.default_rng(11).standard_normal(len(X)) * 40 + 9
    g2 = GaussianProcess(seed=0).fit(X, other_y, fixed_theta=th)
    for strat in ("ivr", "max_variance"):
        a = select_batch(C, 3, strat, models=models, seed=1)
        b = select_batch(C, 3, strat, models={"a": g2}, seed=1)
        assert a.indices == b.indices and np.allclose(a.scores, b.scores, rtol=1e-9)


# ------------------------------------------------------------------ misc / validation
def test_selection_object_and_constrained_region(tmp_path):
    fs = [ad.Factor("A", 0, 10), ad.Factor("B", 0, 10)]
    X, C, models = setup(nc=80)
    sel = select_batch(C, 2, "ivr", models=models, existing=X, seed=1)
    df = sel.actual_df(fs)
    assert list(df.columns) == ["A", "B"] and len(df) == 2 and df.min().min() >= 0 and df.max().max() <= 10
    p = sel.to_csv(tmp_path / "next.csv", fs)
    assert p.exists() and isinstance(sel, Selection) and len(sel) == 2
    r = Region.from_factors(fs, [({"A": 1, "B": 1}, "<=", 12)])
    Cc = r.sample(80, seed=1)[0]
    s2 = select_batch(Cc, 3, "ivr", models=models, existing=X, region=r, seed=1)
    assert r.contains(s2.points).all()


def test_validation_errors():
    X, C, models = setup()
    assert set(STRATEGIES) == {"i_optimal", "ivr", "max_variance", "random"}
    with pytest.raises(ad.DesignError):
        select_batch(C, 1, "ei", models=models)
    with pytest.raises(ad.DesignError):
        select_batch(C, 0, "ivr", models=models)
    with pytest.raises(ad.DesignError):
        select_batch(C, 1, "ivr")                                               # needs models
    with pytest.raises(ad.DesignError):
        select_batch(C, 1, "i_optimal")                                         # needs existing runs
    with pytest.raises(ad.DesignError):
        select_batch(C, 1, "ivr", models=models, aggregate="median")
    with pytest.raises(ad.DesignError):
        select_batch(C, 1, "ivr", models=models, min_distance=-1)
    const = GaussianProcess(seed=0).fit(X, np.full(len(X), 2.0))
    with pytest.raises(ad.DesignError, match="constant"):
        select_batch(C, 1, "max_variance", models={"c": const})
    with pytest.raises(ad.DesignError):
        select_batch(C, 1, "i_optimal", existing=np.zeros((3, 5)))
    with pytest.raises(ad.DesignError):
        select_batch(np.zeros((0, 2)), 1, "random")


def test_unidentifiable_existing_design_warns_for_i_optimal():
    C = Region.unit_cube(2).sample(30, seed=1)[0]
    with pytest.warns(ad.InferenceWarning, match="do not identify"):
        sel = select_batch(C, 2, "i_optimal", existing=C[:3], seed=1)
    assert len(sel) == 2
