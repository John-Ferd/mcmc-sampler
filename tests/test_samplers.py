import numpy as np

from mcmc_sampler import BayesLinReg, gibbs_sampler, make_synthetic, metropolis_hastings


def _model(n=400, seed=0):
    X, y, beta, s2 = make_synthetic(n=n, seed=seed)
    return BayesLinReg.with_default_prior(X, y), X, y, beta, s2


def test_beta_conditional_matches_ols_under_flat_prior():
    model, X, y, _, _ = _model()
    model = BayesLinReg(X, y, np.zeros(3), 1e10 * np.eye(3))
    m_n, V_n = model.beta_conditional(sigma2=1.3)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(m_n, ols, rtol=1e-6)
    np.testing.assert_allclose(V_n, 1.3 * np.linalg.inv(X.T @ X), rtol=1e-6)


def test_log_posterior_differences_match_closed_form_in_beta():
    # For fixed sigma2 the log posterior is quadratic in beta with the
    # conditional precision as curvature; check one finite-difference identity.
    model, *_ = _model()
    theta = np.array([0.5, -1.0, 0.2, np.log(0.8)])
    m_n, V_n = model.beta_conditional(0.8)
    e = np.array([1e-3, 0, 0, 0])
    second = model.log_post(theta + e) - 2 * model.log_post(theta) + model.log_post(theta - e)
    assert np.isclose(second / 1e-6, -np.linalg.inv(V_n)[0, 0], rtol=1e-4)


def test_gibbs_recovers_parameters():
    model, X, y, beta, s2 = _model(n=2000)
    draws = gibbs_sampler(model, n_iter=3000, burn_in=500, seed=1)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(draws[:, :-1].mean(axis=0), ols, atol=0.02)
    assert abs(draws[:, -1].mean() - s2) < 0.15


def test_mh_agrees_with_gibbs_posterior_means():
    model, *_ = _model()
    g = gibbs_sampler(model, n_iter=4000, burn_in=500, seed=2)
    m, acc, _ = metropolis_hastings(model, n_iter=20_000, burn_in=5000, seed=3)
    np.testing.assert_allclose(m.mean(axis=0), g.mean(axis=0), atol=0.05)
    assert 0.1 < acc < 0.45


def test_samplers_are_reproducible_with_a_seed():
    model, *_ = _model()
    a = gibbs_sampler(model, n_iter=50, burn_in=10, seed=7)
    b = gibbs_sampler(model, n_iter=50, burn_in=10, seed=7)
    np.testing.assert_array_equal(a, b)
