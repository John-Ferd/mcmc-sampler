"""
Random-walk Metropolis-Hastings on theta = (beta, log sigma2).

The proposal is N(theta, s^2 * I). During burn-in only, s is adapted toward a
target acceptance rate (Robbins-Monro on log s); adaptation is then frozen so
the kept chain is a valid, time-homogeneous Markov chain. A symmetric
proposal means the Hastings ratio is just the posterior ratio.

An isotropic proposal is deliberately simple: with correlated posteriors it
mixes slowly, which is exactly the contrast with Gibbs the project measures.
"""
from __future__ import annotations

import numpy as np

from .model import BayesLinReg


def metropolis_hastings(model: BayesLinReg, n_iter: int = 5000, burn_in: int = 5000,
                        step: float = 0.1, target_accept: float = 0.234,
                        theta_init: np.ndarray | None = None, seed: int = 0):
    """Return (draws, acceptance_rate, final_step).

    draws has shape (n_iter, p + 1) with columns beta_1..beta_p, sigma2
    (sigma2 is transformed back from log sigma2).
    """
    rng = np.random.default_rng(seed)
    d = model.p + 1
    theta = np.zeros(d) if theta_init is None else np.asarray(theta_init, dtype=float).copy()
    lp = model.log_post(theta)
    log_step = np.log(step)
    out = np.empty((n_iter, d))
    accepted = 0
    for t in range(burn_in + n_iter):
        prop = theta + np.exp(log_step) * rng.normal(size=d)
        lp_prop = model.log_post(prop)
        accept = np.log(rng.random()) < lp_prop - lp
        if accept:
            theta, lp = prop, lp_prop
        if t < burn_in:
            # Robbins-Monro: nudge log step so acceptance -> target.
            log_step += (float(accept) - target_accept) / np.sqrt(t + 1)
        else:
            accepted += accept
            out[t - burn_in, :-1] = theta[:-1]
            out[t - burn_in, -1] = np.exp(theta[-1])
    return out, accepted / n_iter, float(np.exp(log_step))
