"""Gibbs vs Metropolis-Hastings on synthetic Bayesian linear regression.

Runs 4 chains of each sampler from dispersed starting points and writes
artifacts/results.json, artifacts/traces.png and artifacts/acf.png.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcmc_sampler import BayesLinReg, autocorrelation, gibbs_sampler, make_synthetic, metropolis_hastings, summarize
from mcmc_sampler.env import environment

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--n-iter", type=int, default=5000)
ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

out_dir = Path(__file__).resolve().parents[1] / "artifacts"
out_dir.mkdir(exist_ok=True)

X, y, beta_true, s2_true = make_synthetic(n=200, seed=args.seed)
model = BayesLinReg.with_default_prior(X, y)
names = [f"beta_{j}" for j in range(model.p)] + ["sigma2"]
truth = np.append(beta_true, s2_true)
rng = np.random.default_rng(args.seed)
inits = rng.normal(scale=3.0, size=(args.chains, model.p + 1))   # dispersed starts

t0 = time.perf_counter()
g = np.stack([gibbs_sampler(model, args.n_iter, burn_in=1000, sigma2_init=np.exp(inits[c, -1]),
                            seed=args.seed + c) for c in range(args.chains)])
t_gibbs = time.perf_counter() - t0

t0 = time.perf_counter()
mh_runs = [metropolis_hastings(model, args.n_iter, burn_in=5000, theta_init=inits[c], seed=args.seed + c)
           for c in range(args.chains)]
t_mh = time.perf_counter() - t0
m = np.stack([r[0] for r in mh_runs])

ols = np.linalg.lstsq(X, y, rcond=None)[0]
results = {
    "setup": {"n_obs": int(model.n), "n_params": int(model.p + 1), "chains": args.chains,
              "draws_per_chain": args.n_iter, "seed": args.seed},
    "ols_beta": ols.round(4).tolist(),
    "gibbs": {"seconds": round(t_gibbs, 2), "summary": summarize(g, names, truth)},
    "mh": {"seconds": round(t_mh, 2),
           "acceptance": [round(float(r[1]), 3) for r in mh_runs],
           "step": [round(r[2], 4) for r in mh_runs],
           "summary": summarize(m, names, truth)},
}
results["environment"] = environment()
with open(out_dir / "results.json", "w") as fh:
    json.dump(results, fh, indent=2)

print(f"{'param':8s} {'truth':>7s} | {'Gibbs mean':>10s} {'95% CI':>18s} {'R-hat':>6s} {'ESS':>7s} | "
      f"{'MH mean':>8s} {'R-hat':>6s} {'ESS':>7s}")
for rg, rm in zip(results["gibbs"]["summary"], results["mh"]["summary"]):
    print(f"{rg['param']:8s} {rg['truth']:7.3f} | {rg['mean']:10.3f} [{rg['ci_lo']:7.3f},{rg['ci_hi']:7.3f}] "
          f"{rg['rhat']:6.3f} {rg['ess']:7.0f} | {rm['mean']:8.3f} {rm['rhat']:6.3f} {rm['ess']:7.0f}")
print(f"Gibbs {t_gibbs:.1f}s, MH {t_mh:.1f}s, MH acceptance {results['mh']['acceptance']}")

# Trace plots (first two chains) and ACF of beta_1 for both samplers.
fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
for ax, draws, label in [(axes[0], g, "Gibbs"), (axes[1], m, "Metropolis-Hastings")]:
    for c in range(min(2, args.chains)):
        ax.plot(draws[c, :1000, 1], lw=0.6, label=f"chain {c}")
    ax.axhline(truth[1], color="k", ls="--", lw=1, label="true value")
    ax.set_title(f"{label}: trace of beta_1 (first 1000 kept draws)")
    ax.legend(fontsize=7, loc="upper right")
axes[1].set_xlabel("iteration")
fig.tight_layout()
fig.savefig(out_dir / "traces.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.5))
for draws, label in [(g, "Gibbs"), (m, "Metropolis-Hastings")]:
    ax.plot(np.mean([autocorrelation(draws[c, :, 1])[:60] for c in range(args.chains)], axis=0), label=label)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("lag")
ax.set_ylabel("autocorrelation of beta_1")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "acf.png", dpi=130)
plt.close(fig)
