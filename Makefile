PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help check test test-all

help:
	@echo 'check     Validate the pinned Python environment'
	@echo 'test      Run the code-only unit suite'
	@echo 'test-all  Run all tests; retained experiment artifacts are required'

check:
	$(PYTHON) scripts/check_env.py

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_public_*.py'

test-all:
	$(PYTHON) -m unittest discover -s tests
