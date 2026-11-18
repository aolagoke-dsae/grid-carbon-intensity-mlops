param(
    [Parameter(Position = 0)]
    [ValidateSet("test", "audit", "run-local", "monitor", "dashboards", "slides-data", "slides", "deploy")]
    [string]$Task = "test"
)

$ErrorActionPreference = "Stop"

switch ($Task) {
    "test" {
        python -m ruff check src tests notebooks scripts dashboards/slides
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        python -m pytest -v
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "audit" {
        python scripts/mutation_audit.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "run-local" {
        python -m carbon_mlops.inference.batch_score --mode local
    }
    "monitor" {
        python -m carbon_mlops.monitoring.compute_drift
    }
    "dashboards" {
        # Rewriting the result tables also refreshes the slide data, so the two cannot drift apart.
        python -m carbon_mlops.inference.dashboard_extracts
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        python dashboards/slides/export_deck_data.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "slides-data" {
        python dashboards/slides/export_deck_data.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "slides" {
        # Needs the optional packages: pip install -e ".[slides]"
        python dashboards/slides/export_deck_data.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        python dashboards/slides/build_deck.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        python dashboards/slides/build_deck.py --interview
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "deploy" {
        databricks bundle deploy -t dev
    }
}