"""Frequentist coverage of the Gibbs 95% credible intervals.

One dataset's interval can miss the truth by chance (that is what 95% means).
The meaningful check is repeated simulation: draw many datasets from the model,
fit each, and count how often each interval contains the true value. With a
weak prior the rate should be close to 95%. Writes artifacts/coverage.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mcmc_sampler import BayesLinReg, gibbs_sampler, make_synthetic

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--datasets", type=int, default=200)
ap.add_argument("--n-iter", type=int, default=2000)
args = ap.parse_args()

hits = []
for s in range(args.datasets):
    X, y, beta, s2 = make_synthetic(n=200, seed=1000 + s)
    draws = gibbs_sampler(BayesLinReg.with_default_prior(X, y), args.n_iter, burn_in=500, seed=s)
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    truth = np.append(beta, s2)
    hits.append((lo <= truth) & (truth <= hi))
rate = np.mean(hits, axis=0)
names = [f"beta_{j}" for j in range(len(rate) - 1)] + ["sigma2"]
# Monte Carlo standard error of a coverage rate near 0.95
se = float(np.sqrt(0.95 * 0.05 / args.datasets))
out = {"datasets": args.datasets, "nominal": 0.95, "mc_standard_error": round(se, 4),
       "coverage": {k: round(float(v), 3) for k, v in zip(names, rate)}}
Path(__file__).resolve().parents[1].joinpath("artifacts", "coverage.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
