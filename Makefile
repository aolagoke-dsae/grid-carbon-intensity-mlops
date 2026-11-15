.PHONY: test run-local deploy

test:
ruff check src tests
pytest -v

run-local:
python -m carbon_mlops.inference.batch_score --mode local

deploy:
databricks bundle deploy -t dev
