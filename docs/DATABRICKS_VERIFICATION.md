# Databricks Verification Checklist (Phase G)

**Status (2026-11-15): steps 1 to 11 pass on the Free Edition workspace.** Both jobs run end to end.
Step 3 needed one fix (the MLflow experiment); the inference job passed on its first attempt. Still
open: step 12 (re-running training against an existing champion) has not been run, the model-version
tags were not confirmed because the CLI does not return them, and step 6 shows a small numerical
difference from the local run.

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
| 1 | `databricks bundle validate -t dev` | No errors and no "unknown field" warnings | **Pass**, 2026-11-15, CLI v1.26.0: `Validation OK!`, no warnings |
| 2 | `databricks bundle deploy -t dev` | A wheel is built into `dist/` and two jobs appear in the workspace | **Pass**, 2026-11-15: wheel built and uploaded, `Resources: 2 created`, 34 s |
| 3 | `databricks bundle run -t dev carbon_training_pipeline` | Four tasks succeed in order: ingest, features, train, gate | **Pass on attempt 2**, run `293770853415496`: ingest 48 s, features 64 s, train 80 s, gate 39 s. Attempt 1 (run `261932530366971`) failed at train; see run log |
| 4 | In Catalog Explorer, open `workspace.carbon_mlops` | Tables `bronze_entsoe_hourly` (35,082 rows), `grid_horizon_training`, and `grid_horizon_features` with a primary key on `zone`, `origin_timestamp`, `horizon_h` | **Pass on structure**, checked through the CLI: all three Delta tables exist; `grid_horizon_features` has constraint `grid_horizon_features_pk` on the three key columns, all `NOT NULL`, 24 columns, no target column. Row counts checked by SQL and equal to the local run: Bronze 17,543 (DE-LU) and 17,539 (FR); training table 839,904 rows of which 838,823 have a target; feature table 839,904 rows, all keys distinct |
| 5 | Open the registered model `workspace.carbon_mlops.conformal_carbon_forecaster` | Version 1 holds both `challenger` and `champion`; its tag `gate_decision` is `promoted` | **Pass on aliases**: version 1 is `READY` and holds both aliases. The gate task logged "Promoted @challenger v1 to @champion". The tag itself was not confirmed: the CLI does not return model-version tags |
| 6 | Compare the gate task's log with the local run | Coverage DE-LU 0.8951 and FR 0.8899 on Gate-Val, as in `docs/REVIEW_ISSUES.md` | **Close, not equal**: Databricks gives DE-LU 0.8941 and FR 0.8880. See "Numerical difference" below |
| 7 | `databricks bundle run -t dev carbon_batch_inference` | Two tasks succeed: recalibrate, then score | **Pass on attempt 1**, run `807621109176368`: recalibrate 77 s, score 77 s |
| 8 | Open the registered model again | Version 2 holds `champion`, with tags `recalibrated_from_version = 1` and `calibrated_through = 2025-12-31 23:00:00+00:00` | **Pass on aliases**: versions 1 and 2 are `READY`; `champion` is version 2, `challenger` is version 1. The task logged "Recalibrated ... v1 -> v2 through 2025-12-31 23:00:00". Tags not confirmed through the CLI |
| 9 | `SELECT zone, count(*), min(origin_timestamp) FROM workspace.carbon_mlops.gold_forecasts GROUP BY zone` | 24 rows per zone, origin `2025-12-31 22:00` for both | **Pass**: 24 rows per zone, one origin (`2025-12-31 22:00 UTC`) in each, lead times 1 to 24 all present |
| 10 | Check `p05 <= p50 <= p95` and `p05 >= 0` on every row of `gold_forecasts` | Holds on all 48 rows | **Pass**: 0 ordering violations, 0 negative lower bounds, 0 nulls, and `interval_width` equals `p95 - p05` on every row |
| 11 | Browse the volume `workspace.carbon_mlops.exports` | `latest_24h_forecasts.csv` with 48 data rows | **Pass**: file present (20,906 bytes), 48 data rows, 24 per zone, with `p05`, `p50` and `p95` |
| 12 | Re-run step 3 | Succeeds again; a new challenger version is compared against the current champion | Not run |

If a step fails, record the error text in the "Observed" column before changing anything. The job
schedule is paused by design: the snapshot is static, so a recurring run would rescore the same hours.

## Run log

### 2026-11-15, attempt 1 of the training job (run `261932530366971`)

| Task | Result |
| :--- | :--- |
| `ingest_bronze` | SUCCESS, 87 s |
| `materialize_uc_features` | SUCCESS, 75 s |
| `train_challenger` | FAILED after one automatic retry |
| `promotion_gate` | skipped (upstream failed) |

Error, verbatim, raised by `mlflow.start_run` inside `train.register_forecaster`:

```
RestException: RESOURCE_DOES_NOT_EXIST: Could not find experiment with ID None.
```

Cause: a notebook has a default MLflow experiment, a wheel task does not, so `start_run` had no
experiment to attach to. This was not among the ten assumptions listed above; it is an eleventh,
unforeseen, failure. Model fitting itself completed before the error: the failure is at logging.

Assumptions settled by this attempt:

| # | Assumption | Outcome |
| :--- | :--- | :--- |
| 1 | `environment_version: "5"` is available on Free Edition | confirmed |
| 2 | `./dist/*.whl` resolves relative to `databricks.yml` | confirmed |
| 3 | The `pip wheel` build command works | confirmed |
| 4 | The wheel's requirements install without clashing | confirmed for ingest, features and train (all imported and ran) |
| 5 | `databricks-feature-engineering` as a dependency | confirmed (the feature client imported and ran) |
| 6 | `create_table` accepts nullable key columns | confirmed: the table was created with `NOT NULL` keys and a primary key |
| 7 | 840,000 rows from pandas to Spark in one call | confirmed |
| 10 | `DATABRICKS_RUNTIME_VERSION` is set in serverless jobs | confirmed (the Delta branch ran) |
| 8, 9 | `spark_udf` model loading; volume export | not reached |

### 2026-11-15, fix and attempt 2 (run `293770853415496`)

Fix: `train.ensure_mlflow_experiment` selects the experiment explicitly when running as a Databricks
job (`/Users/<current user>/grid-carbon-intensity-mlops`, or `MLFLOW_EXPERIMENT_NAME` if set). It is
covered by a unit test and a breakage-audit case (G15).

| Task | Result |
| :--- | :--- |
| `ingest_bronze` | SUCCESS, 48 s |
| `materialize_uc_features` | SUCCESS, 64 s |
| `train_challenger` | SUCCESS, 80 s; registered `workspace.carbon_mlops.conformal_carbon_forecaster` v1 as `challenger` |
| `promotion_gate` | SUCCESS, 39 s; promoted v1 to `champion` |

This confirms, on the workspace: registration in the Unity Catalog model registry, loading the
registered model back inside the gate task from the code stored with it, and the gate's re-scoring.

### Numerical difference between Databricks and the local run

| Gate-Val coverage | Local (Windows ARM, Python 3.14) | Databricks (Linux, Python 3.12) |
| :--- | :--- | :--- |
| DE-LU | 0.8951 | 0.8941 |
| FR | 0.8899 | 0.8880 |

The input data is identical: every row count above matches. The difference is therefore in model
fitting. The likely cause is that gradient-boosted trees are not bit-identical across processor
architectures and library builds, so a few split decisions differ. That explanation is plausible but
**not verified**; the LightGBM, NumPy and pandas versions installed on serverless were not recorded.

Consequences:

- Both runs pass the gate, and the conclusion is the same.
- The locked-test figures in `docs/LOCKED_TEST_RESULTS.md` are exact for the local environment and
  reproduce there bit for bit. On another platform expect differences of about 0.1 to 0.2 percentage
  points. This is far smaller than the bootstrap intervals (about plus or minus 3 points), but the
  claim "reproducible" should be read as "reproducible on the same platform".

### 2026-11-15, inference job, attempt 1 (run `807621109176368`)

| Task | Result |
| :--- | :--- |
| `recalibrate_champion` | SUCCESS, 77 s; registered v2 as `champion`, recalibrated from v1 through 2025-12-31 23:00 UTC |
| `spark_udf_batch_score` | SUCCESS, 77 s; scored 48 rows into `workspace.carbon_mlops.gold_forecasts` and wrote the extract to the volume |

Assumptions settled:

| # | Assumption | Outcome |
| :--- | :--- | :--- |
| 8 | `spark_udf` can load the model on serverless | confirmed. MLflow used `env_manager="local"` (its default) and warned that this does not recreate the training environment; here training and scoring share the same serverless environment, so the warning does not apply |
| 9 | The job can create the volume and write a file to `/Volumes/...` with pandas | confirmed |

Forecast written for origin 2025-12-31 22:00 UTC (rounded, gCO2eq/kWh):

| Zone | Lead time | Origin intensity | p05 | p50 | p95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DE-LU | 1 h | 269.8 | 245.7 | 283.8 | 297.8 |
| DE-LU | 12 h | 269.8 | 156.9 | 287.1 | 439.4 |
| DE-LU | 24 h | 269.8 | 126.5 | 337.2 | 561.4 |
| FR | 1 h | 41.4 | 30.9 | 35.5 | 46.8 |
| FR | 12 h | 41.4 | 27.4 | 45.3 | 72.3 |
| FR | 24 h | 41.4 | 22.6 | 42.1 | 69.0 |

The snapshot ends at that hour, so these forecasts cannot be checked against outcomes.

### Still unverified after these runs

- Step 12: a second training run, where the gate compares a new challenger with an existing champion
  inside Unity Catalog. Locally this path is tested; on the workspace the gate has only run with no
  champion present.
- The model-version tags (`gate_decision`, `recalibrated_from_version`, `calibrated_through`).
- `fe.write_table(mode="merge")` has run exactly once: the first training attempt created the
  feature table with `create_table`, and the second attempt merged the same rows into it. A merge
  that changes or adds rows has not been exercised.
- The `prod` target, and the schedule (paused by design).
- Library versions installed on serverless were not recorded.

## GitHub Actions

Bundle validation in CI runs only when the repository secrets `DATABRICKS_HOST` and
`DATABRICKS_TOKEN` exist. Until they are added, CI skips it, so a green run says nothing about
the bundle.
