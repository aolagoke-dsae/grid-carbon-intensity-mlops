.PHONY: test audit run-local monitor deploy

test:
	ruff check src tests notebooks scripts
	pytest -v

audit:
	python scripts/mutation_audit.py

run-local:
	python -m carbon_mlops.inference.batch_score --mode local

monitor:
	python -m carbon_mlops.monitoring.compute_drift

deploy:
	databricks bundle deploy -t dev
