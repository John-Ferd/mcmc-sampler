"""Gibbs sampler: alternate exact draws from the two full conditionals."""
from __future__ import annotations

import numpy as np

from .model import BayesLinReg


def gibbs_sampler(model: BayesLinReg, n_iter: int = 5000, burn_in: int = 1000,
                  sigma2_init: float | None = None, seed: int = 0) -> np.ndarray:
    """Return draws of shape (n_iter, p + 1): columns beta_1..beta_p, sigma2."""
    rng = np.random.default_rng(seed)
    sigma2 = float(sigma2_init) if sigma2_init is not None else float(np.var(model.y))
    out = np.empty((n_iter, model.p + 1))
    for t in range(burn_in + n_iter):
        m_n, V_n = model.beta_conditional(sigma2)
        beta = rng.multivariate_normal(m_n, V_n)
        shape, scale = model.sigma2_conditional(beta)
        sigma2 = scale / rng.gamma(shape)          # InvGamma(shape, scale) draw
        if t >= burn_in:
            out[t - burn_in, :-1] = beta
            out[t - burn_in, -1] = sigma2
    return out
