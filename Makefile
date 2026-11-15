.PHONY: test audit run-local deploy

test:
	ruff check src tests notebooks scripts
	pytest -v

audit:
	python scripts/mutation_audit.py

run-local:
	python -m carbon_mlops.inference.batch_score --mode local

deploy:
	databricks bundle deploy -t dev
