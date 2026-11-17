.PHONY: test audit run-local monitor dashboards deploy

test:
	ruff check src tests notebooks scripts
	pytest -v

audit:
	python scripts/mutation_audit.py

run-local:
	python -m carbon_mlops.inference.batch_score --mode local

monitor:
	python -m carbon_mlops.monitoring.compute_drift

dashboards:
	python -m carbon_mlops.inference.dashboard_extracts

deploy:
	databricks bundle deploy -t dev
