"""Metropolis-Hastings and Gibbs samplers for Bayesian linear regression,
with convergence diagnostics (split R-hat, ESS) written from scratch."""

from .diagnostics import autocorrelation, ess, split_rhat, summarize
from .gibbs import gibbs_sampler
from .mh import metropolis_hastings
from .model import BayesLinReg, make_synthetic

__all__ = [
    "BayesLinReg",
    "autocorrelation",
    "ess",
    "gibbs_sampler",
    "make_synthetic",
    "metropolis_hastings",
    "split_rhat",
    "summarize",
]
__version__ = "0.1.0"
