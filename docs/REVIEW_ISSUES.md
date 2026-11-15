# Pre-Freeze Review Issue Register

Source: adversarial pre-publication review of 2026-11-15 (commit `dc33e7d`).
Phases are listed in fix order; each phase changes what the next one measures.
Cite the ID in commit messages, e.g. `fix(B3): timestamp-based lags`.

Status: `open`, `done`, `deferred` (with reason), `user` (needs an action only the author can take).

## Reference numbers (committed code at `dc33e7d`, locked test)

| | Reported | Reproduced | 95% CI (7-day block bootstrap) |
| :--- | :--- | :--- | :--- |
| DE-LU coverage | 91.63% | 89.89% | 85.4 – 94.0 |
| FR coverage | 89.26% | 80.33% | 74.7 – 85.9 |
| Combined | 90.45% | 85.11% | 80.7 – 89.1 |

Gate-Val: DE-LU 92.46%, FR 87.37%, pooled 89.92%.

## Phase A: Make verification trustworthy

| ID | Issue | Where | Status |
| :--- | :--- | :--- | :--- |
| A1 | Bare `pytest` fails at collection (`No module named 'tests'`) | `tests/test_pyfunc_contract.py:17`, `ci_cd.yml` | done |
| A2 | No git remote; CI has never run | repo | user |
| A3 | `bundle validate` uses a dummy host/token and cannot pass | `ci_cd.yml` | done |
| A4 | `pytest`, `ruff`, `openai` are runtime dependencies | `pyproject.toml` | done |
| A5 | `Makefile` and `dev.ps1` lint different paths than CI (`notebooks` missing) | `Makefile`, `scripts/dev.ps1` | done |
| A6 | `Makefile` has a UTF-8 BOM and no tab-indented recipes, so `make` cannot parse it | `Makefile` | done |

## Phase B: Data and provenance

| ID | Issue | Where | Status |
| :--- | :--- | :--- | :--- |
| B1 | No export script; snapshot lineage is not reproducible | new `scripts/export_bronze_snapshot.py` | done |
| B2 | 15 DE-LU partial hours re-admitted with a recomputed CI; `is_partial` unused | export script, `build_features.py` | done |
| B3 | 4 missing FR hours; row-based `shift()` gives 48 wrong `ci_lag_24h` values | `build_features.py:49-54` | done |
| B4 | 2 FR hours cover 30/45 minutes but appear as full hours | export script | done |
| B5 | Three undocumented denominators (`total`, `priced`, `gen_total`); no `priced_share` column | snapshot `README.md`, export script | done |
| B6 | G1 hash test compares the file to itself; pipeline passes `expected_sha256=None` | `test_ingestion_provenance.py:18`, `ingest_entsoe.py` | done |
| B7 | README says data ends 23:00 (it ends 22:00); hash grain is zone-year, not row | snapshot `README.md` | done |
| B8 | Timestamps are tz-naive | `ingest_entsoe.py` | done |
| B9 | Prices, load and flows dropped although the concept note promises day-ahead signals | export script, `CONCEPT_NOTE.md` | deferred (multi-day change; revisit after Dec 1) |
| B10 | Monte Carlo factor uncertainty dropped; for FR (44 g wide) it exceeds the forecast interval (19 g) | `CONCEPT_NOTE.md` | done |
| B11 | Derived `data/lakehouse_local/*.parquet` was tracked despite `.gitignore`, so every run dirtied the tree | `data/lakehouse_local/` | done |
| B12 | `upstream_payload_sha256` stored a hash of the upstream hash, which did not resolve to the upstream manifest | export script | done |

Phase B outcome: lags are now hour-exact and 17 untrusted hours are masked. Locked-test coverage is unchanged
(DE-LU 89.85%, FR 80.33%), as expected for hygiene fixes; the calibration problem is Phase D.

## Phase C: Splits and leakage gates

| ID | Issue | Where | Status |
| :--- | :--- | :--- | :--- |
| C1 | No 24h purge; 600 rows per boundary have targets in the next split | `train.py:35-63` | done |
| C2 | G2 assertions are tautological and miss the lag bug | `build_features.py:135-141`, `test_no_leakage.py` | done |
| C3 | EDA notebook reads the locked-test period | `notebooks/01_eda_and_data_quality.py` | done (plots regenerated on pre-test data; the earlier look at the test period is disclosed under F11) |
| C4 | Calibration season (spring/summer) differs from test season (autumn/winter) | design; addressed by D2 | open |
| C5 | EDA notebook crashed at the heatmap (`axes.set_ylabel` on an array), so the committed plots could not be regenerated | `notebooks/01_eda_and_data_quality.py:68` | done |
| C6 | Local scoring picked the last Gate-Val origin, which after purging has a single horizon | `batch_score.py` | done |

Phase C outcome: every split is purged at its boundary and no target hour is shared between splits.
Locked-test coverage after purging: DE-LU 89.67%, FR 80.00% (Gate-Val: DE-LU 92.25%, FR 87.00%).

## Phase D: Model and calibration

| ID | Issue | Where | Status |
| :--- | :--- | :--- | :--- |
| D1 | Base model underfit: raw coverage 71–76%, loses to persistence at h=1 (MAE 23.2 vs 16.3) | `train.py:134-145` | open |
| D2 | Static offsets do not transfer: FR test coverage 80.3% (rolling 60-day recalibration: 85.6%) | `train.py:149-173` | open |
| D3 | Bucket-level offsets hide per-horizon miscalibration (94.7% at h=1, 89.2% at h=6) | `train.py`, `config.py` | open |
| D4 | DE-LU coverage is 72% in the top renewable quartile, all misses below p05 | `train.py` | open (optional before freeze) |
| D5 | Unknown bucket silently gets a zero offset | `pyfunc_wrapper.py:87` | open |
| D6 | Dead `else` branch that would crash on a float | `pyfunc_wrapper.py:48-53` | open |
| D7 | `horizon_bucket` is trusted from input instead of derived from `horizon_h` | `pyfunc_wrapper.py:82-86` | open |
| D8 | Code is asymmetric CQR; concept note describes the symmetric score | `CONCEPT_NOTE.md` §3 | open |
| D9 | "Guaranteed finite-sample coverage" is an overclaim for autocorrelated data | `CONCEPT_NOTE.md` §1 | open |

## Phase E: Promotion gate

| ID | Issue | Where | Status |
| :--- | :--- | :--- | :--- |
| E1 | Pooled coverage lets FR (87.4%) pass on DE-LU's surplus | `evaluate_gate.py:49-53` | open |
| E2 | Gate reads self-reported run metrics; champion and challenger are not scored on the same window | `evaluate_gate.py` | open |
| E3 | No persistence baseline in G3 | `evaluate_gate.py`, `train.py` | open |
| E4 | Gate test injects metrics with `log_metric` instead of testing real ones | `test_pyfunc_contract.py:87-108` | open |

## Phase F: Hypothesis evaluation (after D and E freeze the model)

| ID | Issue | Where | Status |
| :--- | :--- | :--- | :--- |
| F1 | `02_cqr_calibration_analysis.py` and `hypothesis_results.csv` do not exist | `notebooks/`, `dashboards/data/` | open |
| F2 | Reported coverage does not reproduce (see reference numbers) | notebook, README | open |
| F3 | No confidence intervals on coverage | notebook | open |
| F4 | H1 width result is a level artifact; needs relative width and conditional coverage | notebook | open |
| F5 | H1 pseudo-replication: each target hour appears 24 times | notebook | open |
| F6 | H2 was decided before testing (0 of 17,539 hours have DE-LU below FR) and the window has no summer; label it descriptive | notebook, `CONCEPT_NOTE.md` | open |
| F7 | H3 p-value is an n-floor; zones likely pooled as independent | notebook | open |
| F8 | H3 width is partly built from calibration offsets; test realized error instead | notebook | open |
| F9 | H3 uses only 00:00 origins, confounding lead time with time of day | notebook | open |
| F10 | H3 permutation importance: no per-bucket models exist, and features correlate at 0.99 | notebook | open |
| F11 | Locked test has been evaluated more than once; disclose it | README, `CONCEPT_NOTE.md` | open |

## Phase G: Databricks deployment (parallel to B–F, except G3)

| ID | Issue | Where | Status |
| :--- | :--- | :--- | :--- |
| G1 | No wheel artifact, libraries or Serverless environment | `databricks.yml` | open |
| G2 | `parameters` mis-indented, so scoring likely runs in local mode | `databricks.yml:44` | open |
| G3 | No train or gate task or entry point; `@champion` is never registered | `databricks.yml`, `pyproject.toml` | open |
| G4 | `carbon-ingest` reads a relative path not packaged in the wheel | `ingest_entsoe.py`, `pyproject.toml` | open |
| G5 | Column compared to the list from `.collect()` | `batch_score.py:72-73` | open |
| G6 | Feature table written as plain Delta, then merged as a feature table with no primary key | `build_features.py:143-177` | open |
| G7 | No `CREATE SCHEMA`; two-level and three-level table names are mixed | `store.py`, `batch_score.py` | open |
| G8 | `databricks-feature-engineering` not declared (verify whether Serverless preinstalls it) | `pyproject.toml` | open |
| G9 | Model pickles `carbon_mlops` by reference with no `code_paths` | `train.py:223-230` | open |
| G10 | Dashboard CSV written to a relative path on an ephemeral driver | `batch_score.py:104-105` | open |
| G11 | "Latest origin" is a global max over a static snapshot; hourly schedule rescores the same rows | `batch_score.py:71-73` | open |

## Dependencies

- B before D: fixing lags and partial hours changes the training data.
- C1 before D: the purge changes which rows are in calibration.
- D and E before F: the hypothesis notebook should run once, on the final model.
- G is independent of B–F except G3, which needs the final train and gate functions.
