import numpy as np
import pytest

from mcmc_sampler.data import FEATURES, load_diabetes, posterior_predictive_interval, standardize


def test_loader_reads_the_original_tab_format(tmp_path):
    f = tmp_path / "diabetes.tab.txt"
    rows = ["\t".join(FEATURES + ["Y"]), "59\t2\t32.1\t101\t157\t93.2\t38\t4\t4.8598\t87\t151",
            "48\t1\t21.6\t87\t183\t103.2\t70\t3\t3.8918\t69\t75"]
    f.write_text("\n".join(rows) + "\n")
    X, y, names = load_diabetes(path=f)
    assert X.shape == (2, 10) and names == FEATURES
    np.testing.assert_allclose(y, [151, 75])


def test_loader_rejects_unexpected_columns(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("A B\n1 2\n")
    with pytest.raises(ValueError):
        load_diabetes(path=f)


def test_standardize_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(5, 2, (50, 3)), rng.normal(100, 2, (10, 3))
    ytr, yte = rng.normal(10, 3, 50), rng.normal(50, 3, 10)
    d = standardize(Xtr, ytr, Xte, yte)
    np.testing.assert_allclose(d["X_train"][:, 0], 1.0)                 # intercept column
    np.testing.assert_allclose(d["X_train"][:, 1:].mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(d["X_test"][:, 1:], (Xte - Xtr.mean(0)) / Xtr.std(0))
    np.testing.assert_allclose(d["y_test"], (yte - ytr.mean()) / ytr.std())


def test_predictive_interval_has_nominal_coverage_when_parameters_are_known():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(4000), rng.normal(size=4000)])
    beta, s2 = np.array([1.0, -2.0]), 0.5
    y = X @ beta + np.sqrt(s2) * rng.normal(size=4000)
    draws = np.tile(np.append(beta, s2), (2000, 1))                     # "posterior" concentrated on the truth
    lo, hi, mean = posterior_predictive_interval(X, draws, seed=2)
    assert np.all(lo < mean) and np.all(mean < hi)
    assert abs(((lo <= y) & (y <= hi)).mean() - 0.95) < 0.015
