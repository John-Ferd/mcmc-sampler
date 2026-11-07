"""
Bayesian linear regression with a semi-conjugate prior.

    y | beta, sigma2  ~  N(X beta, sigma2 I)
    beta              ~  N(m0, V0)
    sigma2            ~  InvGamma(a0, b0)

The prior on beta does not scale with sigma2, so the joint posterior has no
closed form, but both full conditionals do (this is what Gibbs exploits):

    beta   | sigma2, y  ~  N(m_n, V_n),
        V_n = (V0^{-1} + X^T X / sigma2)^{-1},
        m_n = V_n (V0^{-1} m0 + X^T y / sigma2)
    sigma2 | beta,   y  ~  InvGamma(a0 + n/2, b0 + ||y - X beta||^2 / 2)

Metropolis-Hastings instead only needs the unnormalized log posterior, which
it evaluates on the unconstrained parameter theta = (beta, log sigma2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BayesLinReg:
    X: np.ndarray
    y: np.ndarray
    m0: np.ndarray
    V0: np.ndarray
    a0: float = 2.0
    b0: float = 1.0

    @classmethod
    def with_default_prior(cls, X, y, prior_var: float = 100.0, a0: float = 2.0, b0: float = 1.0):
        p = X.shape[1]
        return cls(X, y, np.zeros(p), prior_var * np.eye(p), a0, b0)

    def __post_init__(self):
        self.V0_inv = np.linalg.inv(self.V0)
        self.XtX = self.X.T @ self.X
        self.Xty = self.X.T @ self.y
        self.n, self.p = self.X.shape

    # ---- full conditionals (used by Gibbs) --------------------------------
    def beta_conditional(self, sigma2: float):
        """Mean and covariance of beta | sigma2, y."""
        prec = self.V0_inv + self.XtX / sigma2
        V_n = np.linalg.inv(prec)
        m_n = V_n @ (self.V0_inv @ self.m0 + self.Xty / sigma2)
        return m_n, V_n

    def sigma2_conditional(self, beta: np.ndarray):
        """Shape and scale of sigma2 | beta, y (inverse-gamma)."""
        r = self.y - self.X @ beta
        return self.a0 + self.n / 2.0, self.b0 + 0.5 * float(r @ r)

    # ---- log posterior on theta = (beta, log sigma2) (used by MH) ---------
    def log_post(self, theta: np.ndarray) -> float:
        beta, log_s2 = theta[:-1], theta[-1]
        s2 = np.exp(log_s2)
        r = self.y - self.X @ beta
        loglik = -0.5 * self.n * log_s2 - 0.5 * float(r @ r) / s2
        d = beta - self.m0
        logprior_beta = -0.5 * float(d @ self.V0_inv @ d)
        # InvGamma(a0, b0) density on s2, times the Jacobian |ds2/dlog_s2| = s2
        logprior_s2 = -(self.a0 + 1) * log_s2 - self.b0 / s2 + log_s2
        return loglik + logprior_beta + logprior_s2


def make_synthetic(n: int = 200, beta=(1.0, -2.0, 0.5), sigma: float = 1.0, seed: int = 0):
    """Design with an intercept column and correlated covariates."""
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=float)
    p = len(beta)
    Z = rng.normal(size=(n, p - 1))
    Z[:, 1:] += 0.6 * Z[:, :1]          # correlated predictors make MH harder
    X = np.column_stack([np.ones(n), Z])
    y = X @ beta + sigma * rng.normal(size=n)
    return X, y, beta, sigma**2
