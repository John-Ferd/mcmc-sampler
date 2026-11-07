PY ?= python

.PHONY: setup test lint experiment coverage

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
