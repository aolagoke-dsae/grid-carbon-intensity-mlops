.PHONY: test audit run-local monitor dashboards slides-data slides deploy

test:
	ruff check src tests notebooks scripts dashboards/slides
	pytest -v

audit:
	python scripts/mutation_audit.py

run-local:
	python -m carbon_mlops.inference.batch_score --mode local

monitor:
	python -m carbon_mlops.monitoring.compute_drift

# Rewriting the result tables also refreshes the slide data, so the two cannot drift apart.
dashboards:
	python -m carbon_mlops.inference.dashboard_extracts
	python dashboards/slides/export_deck_data.py

slides-data:
	python dashboards/slides/export_deck_data.py

# Needs the optional packages: pip install -e ".[slides]"
slides: slides-data
	python dashboards/slides/build_deck.py
	python dashboards/slides/build_deck.py --interview

deploy:
	databricks bundle deploy -t dev
