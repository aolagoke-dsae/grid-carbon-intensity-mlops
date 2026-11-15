param(
    [Parameter(Position = 0)]
    [ValidateSet("test", "run-local", "deploy")]
    [string]$Task = "test"
)

$ErrorActionPreference = "Stop"

switch ($Task) {
    "test" {
        python -m ruff check src tests notebooks
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        python -m pytest -v
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "run-local" {
        python -m carbon_mlops.inference.batch_score --mode local
    }
    "deploy" {
        databricks bundle deploy -t dev
    }
}