PY ?= python

.PHONY: setup test lint experiment coverage diabetes

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

experiment:
	$(PY) scripts/run_experiment.py

coverage:
	$(PY) scripts/coverage_check.py

diabetes:     # real data (one-time download into data/), ~15 seconds
	$(PY) scripts/run_diabetes.py
