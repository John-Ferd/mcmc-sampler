"""Gibbs vs Metropolis-Hastings on real data: the diabetes study (Efron et al. 2004).

Part 1, full data. Bayesian linear regression of disease progression on the 10
standardized baseline measurements (plus intercept), with the same weak prior
as the synthetic experiment. 4 chains of each sampler from dispersed starts;
split R-hat and ESS for every coefficient; posterior means vs least squares.

Part 2, held-out check (real data has no "true" coefficients, so this is how
the uncertainty is validated). 20 random 80/20 splits: standardize with the
training rows only, run Gibbs on the training rows, and count how often the
95% posterior predictive interval contains each held-out patient's outcome.

Writes artifacts/diabetes.json, diabetes_coefficients.png, diabetes_traces.png.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcmc_sampler import BayesLinReg, autocorrelation, gibbs_sampler, metropolis_hastings, summarize
from mcmc_sampler.data import load_diabetes, posterior_predictive_interval, standardize

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--n-iter", type=int, default=5000)
ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--splits", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
root = Path(__file__).resolve().parents[1]
out_dir = root / "artifacts"
out_dir.mkdir(exist_ok=True)

X_raw, y_raw, feats = load_diabetes(root / "data")
names = ["intercept"] + feats + ["sigma2"]

# ------------------------------------------------------------------ part 1: full data
s = standardize(X_raw, y_raw)
X, y = s["X_train"], s["y_train"]
model = BayesLinReg.with_default_prior(X, y)
rng = np.random.default_rng(args.seed)
inits = rng.normal(scale=3.0, size=(args.chains, model.p + 1))

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
sg, sm = summarize(g, names), summarize(m, names)
for row, o in zip(sg, list(ols) + [None]):
    if o is not None:
        row["ols"] = float(o)
    row["ci_excludes_zero"] = bool(row["ci_lo"] > 0 or row["ci_hi"] < 0) if row["param"] != "sigma2" else None

# ------------------------------------------------------------------ part 2: held-out predictive check
cover, width, rmse_bayes, rmse_ols = [], [], [], []
for k in range(args.splits):
    perm = np.random.default_rng(1000 + k).permutation(len(y_raw))
    tr, te = perm[: int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]
    d = standardize(X_raw[tr], y_raw[tr], X_raw[te], y_raw[te])
    draws = gibbs_sampler(BayesLinReg.with_default_prior(d["X_train"], d["y_train"]), 3000, burn_in=500, seed=k)
    lo, hi, mean = posterior_predictive_interval(d["X_test"], draws, seed=k)
    yt = d["y_test"]
    cover.append(float(((lo <= yt) & (yt <= hi)).mean()))
    width.append(float((hi - lo).mean() * d["y_sd"]))
    rmse_bayes.append(float(np.sqrt(np.mean((mean - yt) ** 2)) * d["y_sd"]))
    b = np.linalg.lstsq(d["X_train"], d["y_train"], rcond=None)[0]
    rmse_ols.append(float(np.sqrt(np.mean((d["X_test"] @ b - yt) ** 2)) * d["y_sd"]))
n_test = len(y_raw) - int(0.8 * len(y_raw))
mc_se = float(np.sqrt(0.95 * 0.05 / (n_test * args.splits)))

ess_g = {r["param"]: r["ess"] for r in sg}
ess_m = {r["param"]: r["ess"] for r in sm}
ratio = {p: ess_g[p] / ess_m[p] for p in ess_g}
res = {
    "data": "diabetes (Efron, Hastie, Johnstone & Tibshirani 2004): 442 patients, 10 features",
    "scale": "features and outcome standardized; coefficients are per standard deviation",
    "setup": {"chains": args.chains, "draws_per_chain": args.n_iter, "seed": args.seed},
    "gibbs": {"seconds": round(t_gibbs, 2), "summary": sg},
    "mh": {"seconds": round(t_mh, 2), "acceptance": [round(float(r[1]), 3) for r in mh_runs], "summary": sm},
    "ess_ratio_gibbs_over_mh": {"min": round(min(ratio.values()), 1), "max": round(max(ratio.values()), 1),
                                "by_param": {p: round(v, 1) for p, v in ratio.items()}},
    "max_rhat": {"gibbs": round(max(r["rhat"] for r in sg), 4), "mh": round(max(r["rhat"] for r in sm), 4)},
    "max_abs_posterior_mean_minus_ols": round(float(np.max(np.abs(np.array([r["mean"] for r in sg[:-1]]) - ols))), 4),
    "held_out": {"splits": args.splits, "test_patients_per_split": n_test,
                 "coverage_95pct_predictive_interval": round(float(np.mean(cover)), 4),
                 "coverage_min_max_over_splits": [round(min(cover), 3), round(max(cover), 3)],
                 "monte_carlo_se_of_coverage": round(mc_se, 4),
                 "mean_interval_width": round(float(np.mean(width)), 1),
                 "rmse_posterior_mean": round(float(np.mean(rmse_bayes)), 2),
                 "rmse_least_squares": round(float(np.mean(rmse_ols)), 2),
                 "outcome_sd": round(float(y_raw.std()), 1)},
}
(out_dir / "diabetes.json").write_text(json.dumps(res, indent=2))

print(f"{'param':10s} {'Gibbs mean':>10s} {'95% CI':>18s} {'OLS':>7s} {'R-hat':>6s} {'ESS':>7s} | {'MH R-hat':>8s} "
      f"{'MH ESS':>7s} {'ratio':>6s}")
for rg, rm in zip(sg, sm):
    o = f"{rg['ols']:7.3f}" if "ols" in rg else " " * 7
    print(f"{rg['param']:10s} {rg['mean']:10.3f} [{rg['ci_lo']:7.3f},{rg['ci_hi']:7.3f}] {o} {rg['rhat']:6.3f} "
          f"{rg['ess']:7.0f} | {rm['rhat']:8.3f} {rm['ess']:7.0f} {ratio[rg['param']]:6.1f}")
print(json.dumps({k: res[k] for k in ("ess_ratio_gibbs_over_mh", "max_rhat", "max_abs_posterior_mean_minus_ols",
                                      "held_out")}, indent=1))
print(f"Gibbs {t_gibbs:.1f}s, MH {t_mh:.1f}s, MH acceptance {res['mh']['acceptance']}")

# ------------------------------------------------------------------ plots
coef = sg[1:-1]
fig, ax = plt.subplots(figsize=(6.5, 4))
yy = np.arange(len(coef))[::-1]
ax.hlines(yy, [r["ci_lo"] for r in coef], [r["ci_hi"] for r in coef], lw=2, label="Gibbs 95% credible interval")
ax.plot([r["mean"] for r in coef], yy, "o", label="posterior mean")
ax.plot([r["ols"] for r in coef], yy, "x", color="k", label="least squares")
ax.axvline(0, color="gray", lw=0.8)
ax.set_yticks(yy, [r["param"] for r in coef])
ax.set_xlabel("effect of +1 SD on disease progression (in SDs)")
ax.set_title("Diabetes: posterior of the regression coefficients")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(out_dir / "diabetes_coefficients.png", dpi=130)
plt.close(fig)

j = names.index("S1")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [2, 1]})
for draws, label in [(g, "Gibbs"), (m, "Metropolis-Hastings")]:
    axes[0].plot(draws[0, :1500, j], lw=0.6, label=label)
    axes[1].plot(np.mean([autocorrelation(draws[c, :, j])[:100] for c in range(args.chains)], axis=0), label=label)
axes[0].set_title("trace of the S1 coefficient (chain 1, first 1,500 kept draws)")
axes[0].set_xlabel("iteration")
axes[1].set_title("autocorrelation, S1 coefficient")
axes[1].set_xlabel("lag")
axes[1].axhline(0, color="k", lw=0.5)
for ax in axes:
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out_dir / "diabetes_traces.png", dpi=130)
plt.close(fig)
