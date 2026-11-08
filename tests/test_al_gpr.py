import numpy as np
import pytest

import adore_doe as ad
from adore_doe.active_learning.gpr import GaussianProcess, kernel_matrix


def data(n=18, d=3, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, d))
    f = lambda Z: np.sin(2 * Z[:, 0]) + 0.5 * Z[:, 1] ** 2 + Z[:, 0] * Z[:, -1]
    return X, f(X) + noise * rng.standard_normal(n), f


# ------------------------------------------------------------------ kernels
@pytest.mark.parametrize("kind", ["matern52", "rbf"])
def test_kernel_is_symmetric_psd_with_diagonal_signal_variance(kind):
    X = np.random.default_rng(1).uniform(-1, 1, (25, 3))
    K = kernel_matrix(X, X, np.array([0.7, 1.5, 3.0]), 2.5, kind)
    assert np.allclose(K, K.T) and np.allclose(np.diag(K), 2.5)
    assert np.linalg.eigvalsh(K).min() > -1e-9


def test_matern52_and_rbf_match_closed_forms_and_ard_scales_dimensions():
    A, B = np.array([[0.0, 0.0]]), np.array([[0.6, 0.8]])
    ls = np.array([1.0, 1.0])
    r = 1.0
    assert kernel_matrix(A, B, ls, 3.0, "matern52")[0, 0] == pytest.approx(
        3.0 * (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r))
    assert kernel_matrix(A, B, ls, 3.0, "rbf")[0, 0] == pytest.approx(3.0 * np.exp(-0.5))
    long_y = kernel_matrix(A, B, np.array([1.0, 100.0]), 1.0, "rbf")[0, 0]
    short_y = kernel_matrix(A, B, np.array([1.0, 0.1]), 1.0, "rbf")[0, 0]
    assert long_y > kernel_matrix(A, B, ls, 1.0, "rbf")[0, 0] > short_y        # ARD: per-dimension scale
    with pytest.raises(ad.DesignError):
        kernel_matrix(A, B, ls, 1.0, "linear")


# ------------------------------------------------------------------ fit / predict
def test_noiseless_data_is_interpolated_and_uncertainty_vanishes_at_data():
    X, y, f = data()
    g = GaussianProcess(seed=1).fit(X, y)
    m, s = g.predict(X)
    assert np.abs(m - y).max() < 1e-3 and s.max() < 1e-2
    Xt = np.random.default_rng(5).uniform(-1, 1, (400, 3))
    mt, st = g.predict(Xt)
    assert np.sqrt(np.mean((mt - f(Xt)) ** 2)) < 0.3 * np.std(f(Xt))           # a real surrogate
    assert st.min() >= 0 and st.mean() > s.mean()                                # more uncertain away from data


def test_variance_grows_with_distance_to_data_and_saturates_at_prior():
    X = np.array([[0.0, 0.0]])
    X = np.vstack([X, np.random.default_rng(2).uniform(-0.2, 0.2, (6, 2))])
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    g = GaussianProcess(seed=0, noise=1e-6).fit(X, y)
    d = np.array([[0.0, 0.0], [0.4, 0.4], [0.8, 0.8], [1.0, 1.0]])
    _, s = g.predict(d)
    assert s[0] < s[1] < s[3] + 1e-9
    assert s[3] <= np.sqrt(g.signal_var) * (1 + 1e-6)                           # never above the prior sd


def test_posterior_covariance_is_consistent_with_predict_and_psd():
    X, y, _ = data(12, 2)
    g = GaussianProcess(seed=0).fit(X, y)
    P = np.random.default_rng(3).uniform(-1, 1, (40, 2))
    C = g.posterior_cov(P)
    _, s = g.predict(P)
    assert np.allclose(np.diag(C), s ** 2, atol=1e-10) and np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > -1e-8
    Q = np.random.default_rng(4).uniform(-1, 1, (7, 2))
    assert np.allclose(g.posterior_cov(P, Q), g.posterior_cov(np.vstack([P, Q]))[:40, 40:], atol=1e-10)


def test_rank1_update_equals_refit_with_fixed_hyperparameters():
    """The latent posterior covariance does not depend on y: adding a point c and refitting with the
    SAME hyper-parameters equals the rank-1 formula  K - K[:,c]K[c,:]/(K[c,c]+noise)."""
    X, y, f = data(14, 2)
    g = GaussianProcess(seed=0, noise=None).fit(X, y)
    P = np.random.default_rng(7).uniform(-1, 1, (30, 2))
    c = np.array([[0.3, -0.4]])
    U = np.vstack([P, c])
    K = g.posterior_cov(U)
    upd = K[:30, :30] - np.outer(K[:30, 30], K[30, :30]) / (K[30, 30] + g.noise_var)
    # refit with an arbitrary y at c, identical (standardised-scale) hyper-parameters
    X2, y2 = np.vstack([X, c]), np.append(y, 123.0)
    g2 = GaussianProcess(seed=0).fit(X2, y2, fixed_theta=g.theta)
    ref = g2.posterior_cov(P) / g2.y_std ** 2 * g.y_std ** 2                     # compare on the same scale
    assert np.allclose(upd, ref, atol=1e-9)
    # and the value of the response at c indeed does not matter
    g3 = GaussianProcess(seed=0).fit(X2, np.append(y, -77.0), fixed_theta=g.theta)
    assert np.allclose(g3.posterior_cov(P) / g3.y_std ** 2, g2.posterior_cov(P) / g2.y_std ** 2, atol=1e-10)


# ------------------------------------------------------------------ hyper-parameters
def test_ard_identifies_the_relevant_dimension():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (30, 3))
    y = np.sin(3 * X[:, 0])
    g = GaussianProcess(seed=1).fit(X, y)
    ls = g.lengthscales
    assert ls[1] > 3 * ls[0] and ls[2] > 3 * ls[0]


def test_bounds_fixed_noise_fixed_theta_and_warm_start():
    X, y, _ = data(15, 2)
    g = GaussianProcess(seed=1, ls_bounds=(0.5, 2.0), sf2_bounds=(0.1, 10.0), noise_bounds=(1e-4, 1e-2)).fit(X, y)
    assert np.all(g.lengthscales >= 0.5 - 1e-9) and np.all(g.lengthscales <= 2.0 + 1e-9)
    assert 1e-4 <= g._nv <= 1e-2 + 1e-12
    gn = GaussianProcess(seed=1, noise=0.05).fit(X, y)
    assert gn.noise_var == pytest.approx(0.05, rel=1e-6)
    gf = GaussianProcess().fit(X, y, fixed_theta=g.theta)
    assert gf.fit_info["fixed"] and np.allclose(gf.theta, g.theta)
    g0 = GaussianProcess(seed=1).fit(X, y)
    gw = GaussianProcess(seed=2, n_restarts=0).fit(X, y, init_theta=g0.theta)
    assert gw.fit_info["n_starts"] == 2                                          # warm start + default start
    assert gw.fit_info["nlml"] <= g0.fit_info["nlml"] + 1e-6                     # warm start is never worse


def test_reproducible_with_seed_and_scale_invariant():
    X, y, _ = data(16, 3)
    a, b = GaussianProcess(seed=4).fit(X, y), GaussianProcess(seed=4).fit(X, y)
    assert np.array_equal(a.theta, b.theta) and np.array_equal(a.predict(X[:3])[0], b.predict(X[:3])[0])
    big = GaussianProcess(seed=4).fit(X, y * 1e6 + 5e6)
    assert np.allclose(big.theta, a.theta, atol=1e-3)                           # standardised internally
    Xt = np.random.default_rng(9).uniform(-1, 1, (100, 3))
    dev = np.abs(big.predict(Xt)[0] - (a.predict(Xt)[0] * 1e6 + 5e6)).max()
    assert dev < 1e-3 * np.ptp(y) * 1e6
    assert big.data_variance == pytest.approx(a.data_variance * 1e12)


def test_optimizer_failure_falls_back_with_warning():
    X, y, _ = data(10, 2)
    g = GaussianProcess(seed=0)
    g._nlml = lambda *a, **k: 1e25
    with pytest.warns(ad.InferenceWarning, match="failed from every start"):
        g.fit(X, y)
    assert g.fit_info["fallback"] and np.isfinite(g.predict(X)[0]).all()


def test_constant_response_and_validation_errors():
    X = np.random.default_rng(0).uniform(-1, 1, (8, 2))
    g = GaussianProcess(seed=0).fit(X, np.full(8, 3.0))
    assert g.constant_response and g.data_variance == 0.0
    assert np.allclose(g.predict(X)[0], 3.0)
    for bad in (dict(kernel="linear"), dict(ls_bounds=(2.0, 1.0)), dict(noise=-1.0), dict(sf2_bounds=(0, 1))):
        with pytest.raises(ad.DesignError):
            GaussianProcess(**bad)
    with pytest.raises(ad.DesignError):
        GaussianProcess().predict(X)
    with pytest.raises(ad.DesignError):
        GaussianProcess().fit(X[:1], [1.0])
    with pytest.raises(ad.DesignError):
        GaussianProcess().fit(X, np.r_[np.nan, np.ones(7)])
    with pytest.raises(ad.DesignError):
        GaussianProcess().fit(X, np.ones(7))


def test_hyperparameter_record_and_rbf_option():
    X, y, _ = data(14, 2)
    g = GaussianProcess(kernel="rbf", seed=0).fit(X, y)
    h = g.hyperparameters()
    assert h["kernel"] == "rbf" and len(h["lengthscales"]) == 2 and h["noise_var"] > 0 and len(h["theta"]) == 4
    assert g.fit_info["n_starts"] == 1 + 3 and "nlml" in g.fit_info          # default start + 3 random


@pytest.mark.parametrize("scale,offset", [(1e-15, 2e-15), (1e-9, 0.0), (1e9, 3e9)])
def test_physical_scales_are_not_mistaken_for_constant_responses(scale, offset):
    X, y, _ = data(16, 3)
    g = GaussianProcess(seed=1).fit(X, y * scale + offset)
    assert not g.constant_response and g.data_variance > 0
    ref = GaussianProcess(seed=1).fit(X, y)
    Xt = np.random.default_rng(3).uniform(-1, 1, (50, 3))
    assert np.abs(g.predict(Xt)[0] - (ref.predict(Xt)[0] * scale + offset)).max() < 1e-3 * np.ptp(y) * abs(scale)
    assert GaussianProcess(seed=1).fit(X, np.full(16, 2e-15)).constant_response            # truly constant stays constant
