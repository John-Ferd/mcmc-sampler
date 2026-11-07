import numpy as np

from mcmc_sampler.diagnostics import autocorrelation, ess, split_rhat


def test_autocorrelation_lag0_is_one_and_iid_is_near_zero():
    x = np.random.default_rng(0).normal(size=5000)
    rho = autocorrelation(x)
    assert rho[0] == 1.0
    assert np.all(np.abs(rho[1:20]) < 0.05)


def test_ess_of_iid_draws_is_close_to_n():
    x = np.random.default_rng(1).normal(size=(4, 2500))
    assert 0.85 * 10_000 < ess(x) < 1.15 * 10_000


def test_ess_matches_ar1_theory():
    # AR(1) with coefficient phi has integrated autocorrelation time (1+phi)/(1-phi).
    rng = np.random.default_rng(2)
    phi, n = 0.9, 50_000
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    expected = n * (1 - phi) / (1 + phi)
    assert 0.7 * expected < ess(x) < 1.3 * expected


def test_rhat_near_one_for_agreeing_chains():
    x = np.random.default_rng(3).normal(size=(4, 2000))
    assert split_rhat(x) < 1.01


def test_rhat_flags_chains_stuck_in_different_places():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4, 2000)) + np.array([[0.0], [0.0], [3.0], [3.0]])
    assert split_rhat(x) > 1.1


def test_split_rhat_catches_a_drifting_chain():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 2000))
    x[0] += np.linspace(0, 4, 2000)           # one chain trends within itself
    assert split_rhat(x) > 1.05
