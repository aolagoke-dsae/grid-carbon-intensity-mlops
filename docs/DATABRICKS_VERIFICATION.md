# Databricks Verification Checklist (Phase G)

**Status: nothing in this document has been executed on a Databricks workspace.** The deployment
code was written and checked locally. Each step below must be run on the Free Edition workspace
and its result recorded here before Phase G can be called done.

## What was verified locally, and what that does and does not show

| Check | Shows | Does not show |
| :--- | :--- | :--- |
| `tests/test_jobs.py`: every bundle task names an entry point the wheel has, references a defined serverless environment, and has no key at the wrong indentation | The bundle is internally consistent with `pyproject.toml` | That Databricks accepts the bundle |
| Wheel built with the bundle's own build command and inspected | It contains the package, six entry points and the pinned snapshot (4.6 MB) | That the wheel installs on serverless environment 5 without dependency conflicts |
| All six entry points run from the installed wheel in an empty folder (no repository): ingest, features, train, gate, recalibrate, score (local mode) | The job logic, the packaged-snapshot fallback and the registry flow work end to end | Anything about Spark, Delta, Unity Catalog or `spark_udf` |
| Integration test drives train, gate and recalibration through the job functions on a temporary registry | Aliases, tags and the recalibration lineage behave as designed | Unity Catalog model registry behaviour |

Code paths that have **never executed anywhere**, because they need Spark or Unity Catalog:

- `io/store.py`: `CREATE SCHEMA`, `saveAsTable`, `spark.table(...).toPandas()`.
- `features/build_features.py`: `FeatureEngineeringClient.create_table` and `write_table(mode="merge")`.
- `inference/batch_score.py`: `run_databricks_batch_scoring` (the `spark_udf`, the Gold table write, the volume export).
- `models/train.py`: registration against the `databricks-uc` registry.

## Assumptions to confirm

Each is a guess based on documentation, not an observation.

| # | Assumption | If it is wrong |
| :--- | :--- | :--- |
| 1 | `environment_version: "5"` (Python 3.12) is available on Free Edition | Change the version in `databricks.yml`; versions 3 to 6 are Python 3.12 |
| 2 | `./dist/*.whl` is resolved relative to `databricks.yml` | The official example uses `../dist/*.whl` from a file in `resources/`; adjust the path |
| 3 | `python -m pip wheel . --no-deps --wheel-dir dist` works as the artifact build command on the machine that deploys | Replace with `uv build --wheel`, the command the official example uses |
| 4 | Installing `mlflow>=2.16`, `lightgbm`, `pandas>=2.2` and `numpy>=1.26` from the wheel's requirements does not clash with packages preinstalled on serverless | Pin to the preinstalled versions, or depend on `mlflow-skinny` |
| 5 | `databricks-feature-engineering` is not preinstalled on serverless, so it is listed as a dependency | Harmless if it is preinstalled |
| 6 | `FeatureEngineeringClient.create_table(primary_keys=...)` accepts a DataFrame whose key columns are nullable in the Spark schema | Create the table with SQL (`NOT NULL` keys and a `PRIMARY KEY` constraint) and use `write_table` only |
| 7 | `spark.createDataFrame(pandas_df)` handles about 840,000 rows by 27 columns over Spark Connect | Write the features in chunks, or stage a Parquet file in a volume and read it with Spark |
| 8 | `mlflow.pyfunc.spark_udf` can load the model on serverless using the code stored with it (`code_paths`) and the listed `pip_requirements` | Pass `env_manager="local"` if the project wheel is visible to the UDF, or adjust the requirements |
| 9 | The job can create the volume `workspace.carbon_mlops.exports` and write a file to `/Volumes/...` with pandas | Create the volume by hand, or write the extract with Spark |
| 10 | `DATABRICKS_RUNTIME_VERSION` is set in serverless jobs, which is how the code detects Databricks | Detect serverless another way in `io/store.py` |

## Steps to run

Run from the repository root with the Databricks CLI authenticated against the workspace.

| # | Command or action | Expected result | Observed (fill in) |
| :--- | :--- | :--- | :--- |
| 1 | `databricks bundle validate -t dev` | No errors and no "unknown field" warnings | |
| 2 | `databricks bundle deploy -t dev` | A wheel is built into `dist/` and two jobs appear in the workspace | |
| 3 | `databricks bundle run -t dev carbon_training_pipeline` | Four tasks succeed in order: ingest, features, train, gate | |
| 4 | In Catalog Explorer, open `workspace.carbon_mlops` | Tables `bronze_entsoe_hourly` (35,082 rows), `grid_horizon_training`, and `grid_horizon_features` with a primary key on `zone`, `origin_timestamp`, `horizon_h` | |
| 5 | Open the registered model `workspace.carbon_mlops.conformal_carbon_forecaster` | Version 1 holds both `challenger` and `champion`; its tag `gate_decision` is `promoted` | |
| 6 | Compare the gate task's log with the local run | Coverage DE-LU 0.8951 and FR 0.8899 on Gate-Val, as in `docs/REVIEW_ISSUES.md` | |
| 7 | `databricks bundle run -t dev carbon_batch_inference` | Two tasks succeed: recalibrate, then score | |
| 8 | Open the registered model again | Version 2 holds `champion`, with tags `recalibrated_from_version = 1` and `calibrated_through = 2025-12-31 23:00:00+00:00` | |
| 9 | `SELECT zone, count(*), min(origin_timestamp) FROM workspace.carbon_mlops.gold_forecasts GROUP BY zone` | 24 rows per zone, origin `2025-12-31 22:00` for both | |
| 10 | Check `p05 <= p50 <= p95` and `p05 >= 0` on every row of `gold_forecasts` | Holds on all 48 rows | |
| 11 | Browse the volume `workspace.carbon_mlops.exports` | `latest_24h_forecasts.csv` with 48 data rows | |
| 12 | Re-run step 3 | Succeeds again; a new challenger version is compared against the current champion | |

If a step fails, record the error text in the "Observed" column before changing anything. The job
schedule is paused by design: the snapshot is static, so a recurring run would rescore the same hours.

## GitHub Actions

Bundle validation in CI runs only when the repository secrets `DATABRICKS_HOST` and
`DATABRICKS_TOKEN` exist. Until they are added, CI skips it, so a green run says nothing about
the bundle.
