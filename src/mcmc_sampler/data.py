"""
Real data: the diabetes study from Efron, Hastie, Johnstone & Tibshirani,
"Least Angle Regression", Annals of Statistics (2004).

442 patients; 10 baseline measurements (age, sex, body-mass index, mean
blood pressure and six blood-serum measurements S1-S6) and a quantitative
measure of disease progression one year later (Y). The serum measurements are
strongly correlated (S1 and S2: r = 0.90; S1's variance inflation factor is
about 59), which makes the regression posterior highly correlated: a hard case
for random-walk Metropolis-Hastings and an easy one for Gibbs.

The original tab-separated file is downloaded once from the authors' data page
and cached in data/ (the same numbers ship with scikit-learn as
``load_diabetes(scaled=False)``).

Unit tests use small simulated data where the true parameters are known; the
real-data experiment is scripts/run_diabetes.py.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

URL = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt"
FEATURES = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]


def load_diabetes(data_dir: str | Path = "data", path: str | Path | None = None):
    """Return (X of shape (442, 10), y of shape (442,), feature names)."""
    if path is None:
        path = Path(data_dir) / "diabetes.tab.txt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL, path)
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().split()
    if header[:10] != FEATURES or header[10] != "Y":
        raise ValueError(f"unexpected columns: {header}")
    d = np.loadtxt(path, skiprows=1)
    return d[:, :10], d[:, 10], list(FEATURES)


def standardize(X_train, y_train, X_test=None, y_test=None):
    """Scale features and target with TRAINING means and standard deviations only,
    and prepend an intercept column. Returns the scaled arrays plus the
    statistics needed to map predictions back to the original units."""
    mx, sx = X_train.mean(0), X_train.std(0)
    my, sy = y_train.mean(), y_train.std()

    def design(X):
        return np.column_stack([np.ones(len(X)), (X - mx) / sx])

    out = {"X_train": design(X_train), "y_train": (y_train - my) / sy, "y_mean": my, "y_sd": sy}
    if X_test is not None:
        out["X_test"] = design(X_test)
    if y_test is not None:
        out["y_test"] = (y_test - my) / sy
    return out


def posterior_predictive_interval(X_new, draws, level: float = 0.95, seed: int = 0):
    """Equal-tailed posterior predictive interval for new responses.

    ``draws`` has columns beta_1..beta_p, sigma2 (as returned by the samplers).
    For each draw, simulate y_new ~ N(X_new beta, sigma2): the interval then
    includes both parameter uncertainty and observation noise.
    """
    rng = np.random.default_rng(seed)
    beta, s2 = draws[:, :-1], draws[:, -1]
    mean = X_new @ beta.T                                     # (n_new, n_draws)
    sim = mean + np.sqrt(s2)[None, :] * rng.normal(size=mean.shape)
    a = (1 - level) / 2
    lo, hi = np.quantile(sim, [a, 1 - a], axis=1)
    return lo, hi, mean.mean(axis=1)
