# Pre-Freeze Review Issue Register

Source: adversarial pre-publication review of 2026-11-15 (commit `be2d8bb`).
Phases are listed in fix order; each phase changes what the next one measures.
Cite the ID in commit messages, e.g. `fix(B3): timestamp-based lags`.

Status: `open`, `done`, `deferred` (with reason), `user` (needs an action only the author can take).

An item is `done` only when all of the following hold. Until then it is `implemented` or `verified`:

1. **Implemented:** the change is in the code.
2. **Verified:** `ruff` and the full test suite pass; the fix is confirmed on the real snapshot by
   independent code; `python scripts/mutation_audit.py` reports every breakage caught; docs and this
   register match the code.
3. **Done:** verified, and GitHub Actions is green on the pushed commit.

## Reference numbers (committed code at `be2d8bb`, locked test)

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
| A2 | No git remote; CI has never run | repo | done (GitHub remote; CI green) |
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
| C4 | Calibration season (spring/summer) differs from test season (autumn/winter) | design; addressed by D2 | done (addressed by D2) |
| C5 | EDA notebook crashed at the heatmap (`axes.set_ylabel` on an array), so the committed plots could not be regenerated | `notebooks/01_eda_and_data_quality.py:68` | done |
| C6 | Local scoring picked the last Gate-Val origin, which after purging has a single horizon | `batch_score.py` | done |

Phase C outcome: every split is purged at its boundary and no target hour is shared between splits.
Locked-test coverage after purging: DE-LU 89.67%, FR 80.00% (Gate-Val: DE-LU 92.25%, FR 87.00%).

## Phase D: Model and calibration

| ID | Issue | Where | Status |
| :--- | :--- | :--- | :--- |
| D1 | Base model underfit: raw coverage 71–76%, loses to persistence at h=1 (MAE 23.2 vs 16.3) | `train.py:134-145` | done |
| D2 | Static offsets do not transfer: FR test coverage 80.3% (rolling 60-day recalibration: 85.6%) | `train.py:149-173` | done |
| D3 | Bucket-level offsets hide per-horizon miscalibration (94.7% at h=1, 89.2% at h=6) | `train.py`, `config.py` | done |
| D4 | DE-LU coverage is 72% in the top renewable quartile, all misses below p05 | `train.py` | open (optional before freeze) |
| D5 | Unknown bucket silently gets a zero offset | `pyfunc_wrapper.py:87` | done |
| D6 | Dead `else` branch that would crash on a float | `pyfunc_wrapper.py:48-53` | done |
| D7 | `horizon_bucket` is trusted from input instead of derived from `horizon_h` | `pyfunc_wrapper.py:82-86` | done |
| D8 | Code is asymmetric CQR; concept note describes the symmetric score | `CONCEPT_NOTE.md` §3 | done |
| D9 | "Guaranteed finite-sample coverage" is an overclaim for autocorrelated data | `CONCEPT_NOTE.md` §1 | done |
| D10 | Conformal quantile via `np.quantile(method="higher")` landed one rank too high (intervals slightly too wide) | `models/conformal.py` | done |
| D11 | FR at h=1 narrowly loses to persistence on Gate-Val (MAE 1.398 vs 1.372); every bucket still wins | `train.py` | accepted: reported as a limitation; the bucket-level baseline check is enforced by E3 |

Phase D outcome (all choices made on Gate-Val only; the locked test was not scored):

| Gate-Val walk-forward | Static, 3 buckets | Rolling 30 days, per horizon |
| :--- | :--- | :--- |
| DE-LU coverage | 92.3% | 89.5% |
| FR coverage | 87.0% | 89.0% |
| Per-horizon coverage range | 84% – 97.5% | 87% – 92% |

The 600-tree model had about 3% lower pinball loss on Gate-Val but takes 26 s to fit; the 200-tree
model was kept to stay inside the 15 s local-run budget.

## Audit of Phases A–D (2026-11-15)

Each fix was deliberately broken in a scratch copy and the test suite run against it.
24 of 28 breakages were caught. The 4 survivors exposed missing tests, now added. The audit is
repeatable: `python scripts/mutation_audit.py` (or `make audit`).

| Survivor | Gap | Closed by |
| :--- | :--- | :--- |
| A generation-mix feature reading the future | Leakage test perturbed only intensity; G2 audit checked only intensity lags | Audit now covers all origin-state features and the rolling std; perturbation test perturbs every input column |
| Training on the level while predicting as a change | No test of forecast accuracy | Trained median must beat persistence on Gate-Val |
| Conformity score with the wrong sign | No direct test of `add_conformity_scores` | Sign test with a stub model |
| Gate metrics computed from other intervals | Metrics not tied to predictions | Metrics must equal a recomputation from the walk-forward predictions |

Independent re-verification on the real snapshot found 0 flagged hours used as origin or target,
0 mismatches across lags, rolling statistics and origin-state features, and 0 target hours shared
between splits.

## Model freeze (2026-11-15)

The Phase D selection was repeated with the final code (after the D10 quantile fix and the split
purge), on Gate-Val only: 5 model sizes x 2 target formulations x 4 window lengths = 40 configurations.

| Decision | Rule stated in advance | Result with final code |
| :--- | :--- | :--- |
| Target formulation | Lower Gate-Val pinball loss | Change-based beats level-based for all 5 model sizes |
| Recalibration window | Smallest worst-zone gap from 90% coverage | 30 days wins for all 5 model sizes |
| Model size | Lowest pinball loss within the 15 s local-run budget | 200 trees confirmed (table below) |

| Model (change-based, 30-day window) | DE-LU cov | FR cov | DE-LU pinball | FR pinball | Fit time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 120 trees | 89.3% | 88.9% | 16.78 | 0.863 | 2.9 s |
| 150 trees | 89.5% | 89.1% | 16.84 | 0.861 | 3.3 s |
| **200 trees (chosen)** | **89.5%** | **89.0%** | **16.55** | **0.851** | **4.1 s** |
| 300 trees | 89.6% | 88.7% | 16.51 | 0.848 | 9.2 s |
| 600 trees | 89.4% | 88.8% | 16.06 | 0.835 | 11.8 s |

Every one of the 40 configurations beats persistence in every zone and horizon bucket.
The 300- and 600-tree models are slightly more accurate but push the local run past 15 s.

**The model is frozen from here.** Gate-Val has now informed about 40 configurations, so its coverage
figure is mildly optimistic. No further tuning on Gate-Val; a later change needs a reason stated in
advance, not a better score.

Local run time (`python -m carbon_mlops.inference.batch_score --mode local`, full process): 15.2-16.0 s
before, 12.6-13.8 s after (5 runs). The saving comes from not scoring Calib rows older than the first
calibration window; gate metrics, gate predictions and final offsets are identical to scoring the
full history (checked on the real snapshot and guarded by a test).

Breakage audit after these changes: 29 of 29 caught, 0 survived, 0 stale.

| Open item | State |
| :--- | :--- |
| D11: FR at h=1 narrowly loses to persistence | open; not tuned away, the baseline check moves into the gate (E3) |
| Databricks code paths (Spark scoring, feature store, registry on Unity Catalog) | unverified: never executed; only verifiable on the workspace (Phase G) |

## Phase E: Promotion gate

| ID | Issue | Where | Status |
| :--- | :--- | :--- | :--- |
| E1 | Pooled coverage lets FR (87.4%) pass on DE-LU's surplus | `evaluate_gate.py:49-53` | done |
| E2 | Gate reads self-reported run metrics; champion and challenger are not scored on the same window | `evaluate_gate.py` | done |
| E3 | No persistence baseline in G3 | `evaluate_gate.py`, `train.py` | done |
| E4 | Gate test injects metrics with `log_metric` instead of testing real ones | `test_pyfunc_contract.py:87-108` | done |
| E5 | Pooled pinball comparison was decided by DE-LU alone (its losses are about 20x FR's) | `evaluate_gate.py` | done |
| E6 | The gate test wrote model artifacts into `./mlruns` inside the repo on every run | `tests/test_pyfunc_contract.py` | done |

Phase E outcome. Phases D and E are `done` (CI green on `a34d8ee`, `0ce762a` and `cae0ba3`). The gate now re-scores
`@challenger` and `@champion` itself on the same Gate-Val window and requires: per-zone coverage in
`[0.87, 0.96]`, a median that beats persistence in every zone and horizon bucket, and a scale-free
pinball ratio versus the champion of at most 1. Each decision is written to the model version as the
tags `gate_decision` and `gate_reason`.

Acceptance checks (all met):

| Issue | Check | Evidence |
| :--- | :--- | :--- |
| E1 | Pooled coverage in bounds but one zone out of bounds is rejected | `test_gate_requires_every_zone_inside_coverage_bounds` |
| E2 | Overwriting the logged run metrics does not change the decision | integration test poisons the metrics of both runs |
| E3 | Losing to persistence in any zone and bucket is rejected | `test_gate_rejects_a_model_that_loses_to_persistence` |
| E4 | A really worse model is rejected against a really better champion | one-tree challenger vs 15-tree champion, no injected numbers |

Gate on the real snapshot with a throwaway registry (Gate-Val only):

| Challenger | Coverage DE-LU / FR | Pinball ratio vs champion | Decision |
| :--- | :--- | :--- | :--- |
| Production model, no champion yet | 89.5% / 89.0% | n/a | promoted |
| Identical retrain | 89.5% / 89.0% | 1.000 | promoted (tie) |
| 20 trees, logged pinball falsified to 0.001 | 89.5% / 90.1% | 1.442 | rejected |
| Level-based target | 87.1% / 86.7% | 1.021 | rejected (FR coverage, pinball) |

Breakage audit after Phase E: 37 of 37 caught, 0 survived, 0 stale.

Known limits of the gate: the coverage bounds are the pre-existing `[0.87, 0.96]`; no per-horizon
floor was added, because any new threshold would have been chosen after seeing Gate-Val. The gate is
not yet wired into a Databricks job (G3).

## Phase F: Hypothesis evaluation (after D and E freeze the model)

| ID | Issue | Where | Status |
| :--- | :--- | :--- | :--- |
| F1 | `02_cqr_calibration_analysis.py` and `hypothesis_results.csv` do not exist | `notebooks/`, `dashboards/data/` | done (verified, CI green): notebook, results CSV, calibration CSV and run record are committed |
| F2 | Reported coverage does not reproduce (see reference numbers) | notebook, README | done (verified, CI green): old figures retired; reproducible figures in `docs/LOCKED_TEST_RESULTS.md` |
| F3 | No confidence intervals on coverage | notebook | done (verified, CI green): 7-day block-bootstrap intervals on every coverage figure |
| F4 | H1 width result is a level artifact; needs relative width and conditional coverage | notebook | done (verified, CI green): relative width, error and coverage by regime reported as context |
| F5 | H1 pseudo-replication: each target hour appears 24 times | notebook | done (verified, CI green): each target hour counted once |
| F6 | H2 was decided before testing (0 of 17,539 hours have DE-LU below FR) and the window has no summer; label it descriptive | notebook, `CONCEPT_NOTE.md` | done (verified, CI green): reported as descriptive, with both scope limits |
| F7 | H3 p-value is an n-floor; zones likely pooled as independent | notebook | done (verified, CI green): zones tested separately; pooled figure shown with its caveat |
| F8 | H3 width is partly built from calibration offsets; test realized error instead | notebook | done (verified, CI green): corrected test uses realized absolute error |
| F9 | H3 uses only 00:00 origins, confounding lead time with time of day | notebook | done (verified, CI green): corrected test averages over all 24 origin hours |
| F10 | H3 permutation importance: no per-bucket models exist, and features correlate at 0.99 | notebook | done (verified, CI green): grouped permutation importance with bootstrap intervals |
| F11 | Locked test has been evaluated more than once; disclose it | README, `CONCEPT_NOTE.md` | done (verified, CI green): disclosed in the plan, the concept note and the results document; root `README.md` written, with tests tying its numbers and links to the committed files |

Phase F outcome (2026-11-15). The Locked Test was scored once with the frozen model, under the plan.

| Step | Evidence |
| :--- | :--- |
| Plan approved and committed before any analysis code | `6c8e730` |
| Analysis code committed and pushed before the run | `292c883`, on GitHub at 20:31 +0100; run started 20:32 |
| Single execution (`--run`) | outputs committed unedited as `601a320` |
| Reproduction (`--check`) | recomputed from scratch; both CSV files and the run record reproduced exactly |
| Write-up | `docs/LOCKED_TEST_RESULTS.md`, generated by `scripts/make_results_doc.py` so no number is typed by hand |

Headline results (full tables in `docs/LOCKED_TEST_RESULTS.md`):

| | DE-LU | FR |
| :--- | :--- | :--- |
| Coverage of the 90% interval | 89.0% (85.5 – 92.2) | 87.4% (83.9 – 90.3) |
| Inside the reference band [87%, 96%] | yes | yes, narrowly |
| Day-ahead bucket | 88.8% | 86.3%, outside the band, interval excludes 90% |
| Median beats persistence in every bucket | yes | yes |

H1, H2 and H3 are all falsified, with the registered and corrected analyses agreeing in each case.
H3 fails on the importance reversal; its second part (larger errors at day-ahead lead times) holds
in both zones.

Limitations established by the run. None is acted on: the model is frozen.

| ID | Finding | State |
| :--- | :--- | :--- |
| F15 | FR day-ahead coverage is 86.3% and its interval excludes 90%; FR coverage decays from 89.8% at 1 h to 85.2% at 17 h | open limitation, reported |
| F16 | Calibration is not uniform by regime: DE-LU coverage is 83.3% in the highest renewable-share quartile, where the median is also worse than persistence | open limitation, reported (the deferred D4) |
| F17 | FR at h=1 loses to persistence on the Locked Test as well (0.989 vs 0.771 pinball), confirming D11 | open limitation, reported |
| F18 | The coverage figures are not a fully independent confirmation: settings were chosen on Gate-Val and scheduled recalibration was prompted by an earlier look at this split | disclosed |

Verification of the analysis code, none of it on the Locked Test:

- 15 unit tests on synthetic data with hand-computed answers (bootstrap, calibration table, H1, H2
  including the daylight-saving window, H3a, H3b, verdict rule, locked-test replay).
- Full dry run of the notebook on **Gate-Val** (`--dry-run-gate-val`), twice, with 10,000 resamples:
  about 4.6 minutes each, and the calibration CSV, results CSV and run record are byte-identical
  between the two runs.
- Breakage audit: 49 of 49 caught, 0 survived, 0 stale (12 new breakages for the analysis).

Found while building it:

| ID | Issue | State |
| :--- | :--- | :--- |
| F12 | Draft plan shuffled H3a features "as whole days", which never shuffles time-of-day features | fixed in the plan before it was committed |
| F13 | Bootstrap percentile of infinite ratios (zero calendar importance) returned NaN, silently failing the H3a rule | fixed, with a test |
| F14 | A recalibration window shorter than the step plus one day leaves the longest lead time without offsets at a split boundary; the run stops with an explicit error | documented; the frozen 30-day / 7-day setting is unaffected |

Run time of the test suite is now 11–13 s against the 15 s standard (47 tests).

## Phase G: Databricks deployment (parallel to B–F, except G3)

| ID | Issue | Where | Status |
| :--- | :--- | :--- | :--- |
| G1 | No wheel artifact, libraries or Serverless environment | `databricks.yml` | verified on the workspace 2026-11-15: bundle validates, deploys and both jobs run on serverless environment 5 |
| G2 | `parameters` mis-indented, so scoring likely runs in local mode | `databricks.yml:44` | verified on the workspace 2026-11-15: the scoring task received `--mode databricks` and ran the Spark path |
| G3 | No train or gate task or entry point; `@champion` is never registered | `databricks.yml`, `pyproject.toml` | verified on the workspace 2026-11-15: train, gate and recalibrate ran; the gate ran both without and with an existing champion (run `617140816488603`) |
| G4 | `carbon-ingest` reads a relative path not packaged in the wheel | `ingest_entsoe.py`, `pyproject.toml` | verified on the workspace 2026-11-15: ingest read the snapshot from the wheel; 35,082 rows |
| G5 | Column compared to the list from `.collect()` | `batch_score.py:72-73` | verified on the workspace 2026-11-15: scoring selected the newest origin per zone |
| G6 | Feature table written as plain Delta, then merged as a feature table with no primary key | `build_features.py:143-177` | verified on the workspace 2026-11-15: feature table created with its primary key; no clash; second run merged into it |
| G7 | No `CREATE SCHEMA`; two-level and three-level table names are mixed | `store.py`, `batch_score.py` | verified on the workspace 2026-11-15: schema created, three-level names resolve |
| G8 | `databricks-feature-engineering` not declared (verify whether Serverless preinstalls it) | `pyproject.toml` | verified on the workspace 2026-11-15: the feature client installed and ran |
| G9 | Model pickles `carbon_mlops` by reference with no `code_paths` | `train.py:223-230` | verified on the workspace 2026-11-15: the gate and the Spark UDF both loaded the model from its stored code |
| G10 | Dashboard CSV written to a relative path on an ephemeral driver | `batch_score.py:104-105` | verified on the workspace 2026-11-15: extract in the volume, 48 rows |
| G11 | "Latest origin" is a global max over a static snapshot; hourly schedule rescores the same rows | `batch_score.py:71-73` | verified on the workspace 2026-11-15: 24 rows per zone for origin 2025-12-31 22:00 UTC |
| G12 | Wheel entry points returned a DataFrame; a console entry point's return value becomes the exit status | `pyproject.toml`, new `jobs.py` | verified on the workspace 2026-11-15: all six tasks exit cleanly |
| G13 | Model name was resolved differently in training and in the gate | `train.resolve_model_name` | verified on the workspace 2026-11-15 |
| G14 | The breakage audit's scratch copy lacked `databricks.yml`, so a bundle test would have failed on every breakage and made all of them look caught | `scripts/mutation_audit.py` | fixed; the audit now requires the untouched copy to pass first |
| G15 | A wheel task has no default MLflow experiment, so `mlflow.start_run` failed on Databricks with "Could not find experiment with ID None" (found on the first workspace run; not foreseen) | `train.ensure_mlflow_experiment` | verified on the workspace: training job run `293770853415496` succeeds |
| G16 | Gate-Val coverage differs slightly between platforms (local 0.8951 / 0.8899, Databricks 0.8941 / 0.8880) with identical data | `docs/DATABRICKS_VERIFICATION.md` | open: a cross-platform difference, not run-to-run noise (two workspace runs agree to every digit); the exact cause is not verified. Results are reproducible on the same platform |

Phase G state (2026-11-15). **Both jobs run end to end on the Free Edition workspace.** Steps 1 to 11
of `docs/DATABRICKS_VERIFICATION.md` pass; training run `293770853415496`, inference run
`807621109176368`. Step 12 passes as run `617140816488603`, and a second inference run (`424886171434319`) left the
registry in its operating state: champion is version 4, recalibrated through 2025-12-31.
Not yet `done`: the model-version tags are unconfirmed, G16 (small cross-platform numerical
difference) is open. CI is green on the current head, which contains these commits (confirmed
2026-11-18).

What changed:

- `databricks.yml`: wheel artifact, a serverless environment per job, six tasks, `parameters` inside
  `python_wheel_task`. Training job: ingest, features, train, gate. Inference job: recalibrate, score.
- `jobs.py`: one entry point per task. `run_recalibration_job` is the scheduled recalibration the
  evaluation assumes: it recomputes the champion's offsets from the latest realized errors and
  registers the result as the new champion, tagged with the version it came from.
- Storage uses three-level names and creates the schema; the snapshot ships inside the wheel.

Local evidence:

- Wheel built with the bundle's build command: 27 files, six entry points, the 4.6 MB snapshot.
- All six entry points run from the installed wheel in an empty folder with no repository:
  ingest 0.2 s, features 4.1 s, train 10.4 s, gate 3.5 s, recalibrate 1.3 s, score (local) 10.5 s.
  The gate promoted version 1; recalibration produced version 2, calibrated through
  2025-12-31 23:00 UTC with all 48 offset cells; it scored origin 2025-12-31 22:00 for both zones.
- 53 tests in about 12.5 s. Breakage audit: 58 of 58 caught, with the baseline check passing.

Verification state at commit `ec99624` (2026-11-15, on mains power):

| Check | Result |
| :--- | :--- |
| `ruff check src tests notebooks scripts` | clean |
| Test suite | 57 tests, 11.0 to 11.9 s over five runs |
| Breakage audit | 59 of 59 caught, 0 survived, 0 stale, baseline passing (429 s) |
| Databricks checklist | 12 steps run, 11 pass, step 6 records a cross-platform difference |

The test suite takes 45 to 60 s when the laptop is on battery with the processor throttled. That is
an effect of the machine's power state, not of the code: the same suite runs in 11 s on mains.

## Monitoring (US-3.2), added 2026-11-16

New work, not part of the original review. Status follows the same three-state rule.

| ID | Item | Where | Status |
| :--- | :--- | :--- | :--- |
| M1 | Feature drift: PSI per zone and feature, trailing 30 days against Train and against the same 30 days a year earlier | `monitoring/compute_drift.py` | done (verified locally, CI green) |
| M2 | Rolling 7-day coverage per zone and horizon bucket, miss direction, width, skill against persistence | `monitoring/compute_drift.py` | done (verified locally, CI green) |
| M3 | Monitoring task in the Databricks inference job (`carbon-monitor`) | `jobs.py`, `databricks.yml` | verified on the workspace 2026-11-17 (run `963815134813799`); see `docs/DATABRICKS_VERIFICATION.md` |
| M4 | PSI against the whole Train period is high almost always (321 of 468 values at alert), because a 30-day window against 14 months mostly measures the season | design | addressed by adding the same-season reference; the original measure is kept because the sprint plan specifies it |
| M5 | Same-season PSI is still at alert in 249 of 468 values. The 0.10 / 0.25 thresholds assume independent samples; a month of hourly data is strongly autocorrelated | design | open limitation, documented in the README; thresholds deliberately not retuned |
| M6 | Skill against persistence falls from about 0.6 (DE-LU) and 0.4 (FR) in July to about 0.05 and 0.2 in December, with three negative weeks. Season and model age cannot be separated with one year of data. Base models are never refitted, only recalibrated | finding | open: no retraining schedule exists |

Verification (2026-11-16): 10 unit tests with hand-computed answers; lint clean; 67 tests in about
12.7 s. Breakage audit: the full 67-case run caught 66; the survivor ("coverage ignores the upper
interval bound") showed that the synthetic forecasts only ever missed on the low side. A test with
misses above the interval was added and the 8 monitoring cases were re-run: 8 of 8 caught. The other
59 cases were not re-run after that, since adding a test cannot stop an existing test from failing.

The monitored period includes the held-out test period, by the author's decision of 2026-11-16. It
recomputes forecasts already published in `docs/LOCKED_TEST_RESULTS.md` and is used for reporting only.

What the first real output shows (`dashboards/data/gold_monitoring.csv`, 26 weekly dates, 1,352 rows):

- Weekly coverage is outside the band in 10 of 26 weeks for DE-LU and 14 of 26 for FR; both zones are
  below it in the same week five times; the longest run of consecutive below-band weeks is 3 in each zone.
- FR drifts more than DE-LU year on year: its intensity, fossil share and 24-hour mean are at alert
  in 88% to 96% of weeks, against 35% to 85% for the same DE-LU features.

## Dashboard extracts (US-3.3), added 2026-11-17

US-3.3 was revised from four Tableau views to one Tableau and one Power BI dashboard with no shared
visuals. The dashboards themselves are built by the author; this section covers their data.

| ID | Item | Where | Status |
| :--- | :--- | :--- | :--- |
| W1 | Window ranking (window means, cleanest start, hindsight rank), shared by the extracts and the QA agent | `analysis/windows.py` | done (verified locally, CI green) |
| W2 | Forecast history, candidate windows per strategy, and strategy summary | `inference/dashboard_extracts.py` | done (verified locally, CI green) |
| W3 | Extract task in the Databricks inference job (`carbon-extracts`) | `jobs.py`, `databricks.yml` | verified on the workspace 2026-11-17 (run `963815134813799`); see `docs/DATABRICKS_VERIFICATION.md` |
| W4 | Per-visual build notes, colours, source line and refresh steps | `dashboards/README.md` | written; revised 2026-11-17 to the current design (business-value visuals, a load control, plain wording, and "upper bound" as the displayed name of the `cautious` strategy); the dashboards do not exist yet |
| W5 | CSV files written on the workspace carry no UTC offset and write midnight as a date only; the local files carry `+00:00`. Same instants, different text | `inference/dashboard_extracts.py`, `monitoring/compute_drift.py` | fixed 2026-11-17 and verified locally: all four CSV extracts (including `latest_24h_forecasts.csv`, which had the same defect) go through `write_csv_extract` in `io/store.py`, which writes UTC text with `+00:00` whatever the input; 2 tests, 3 breakage cases (77 of 77 caught), and the regenerated local files are byte-identical. Confirmed on the workspace 2026-11-17 (run `448998131378416`): every timestamp in the five downloaded files carries `+00:00`, and the timestamp columns of three files equal the local ones as text |
| G17 | Forecasts from the workspace champion differ from the local model's: mean absolute difference in `p50` of 6.8 gCO2eq/kWh in DE-LU (about 2%), 0.5 in FR. Model-independent outputs are identical. Same cause as G16, larger than its coverage figures suggested | `docs/DATABRICKS_VERIFICATION.md` | open; found 2026-11-17. Conclusions agree; published figures come from the local model |

The extracts read the held-out period, for reporting only, on the basis approved for monitoring on
2026-11-16. Daily forecasts are those issued at 00:00 UTC.

First real output, 3-hour window, held-out period, 103 comparable days per zone: a window chosen from
the median forecast was 18.9% (DE-LU) and 14.2% (FR) cleaner than the day's average, against 12.9%
and 9.5% for "repeat yesterday's cleanest hours" and 25.4% and 18.7% in hindsight.

### Business value columns, added 2026-11-17

| ID | Item | Where | Status |
| :--- | :--- | :--- | :--- |
| W6 | Outcome in decision units: CO2 avoided per MWh moved for every candidate window; per strategy the mean and total avoided, the share of days the choice was dirtier than the day's average, the worst day, and the share of hindsight's value captured | `inference/dashboard_extracts.py` | verified on the workspace 2026-11-17 (same run as W7) |
| W7 | Weekly value per strategy (`window_value_weekly.csv`, `gold_window_value_weekly`), to set beside weekly skill | `inference/dashboard_extracts.py` | verified on the workspace 2026-11-17 (run `466275646757399`); see `docs/DATABRICKS_VERIFICATION.md` |
| W8 | The "cautious" strategy (choose by the upper bound) does not beat the median: in DE-LU it avoids 63.5 against 66.2 kg per MWh and is dirtier than the day's average on 12.6% of days against 6.8% | `dashboards/data/window_strategy_summary.csv` | finding, open. The interval is not shown to improve the choice of window; stated in the README. Reproduced with the workspace model (61.4 against 67.4 kg; 13.6% against 6.8%) |

Held-out period, 3-hour window, 103 days: the median strategy avoided 66.2 kg CO2 per MWh moved in
DE-LU and 5.2 in FR (yesterday rule: 46.6 and 3.4; hindsight: 89.2 and 6.7), captured 74% and 77% of
hindsight's value, and was dirtier than the day's average on 6.8% and 9.7% of days (yesterday rule:
22.3% and 15.5%). No annual or monetary figure is derived.

Verification (2026-11-17): hand-computed tests for the new columns and for the weekly table, and the
README test now checks the kg, bad-day and worst-day figures against the CSV; lint clean; 79 tests in
about 13.6 s. Breakage audit: 82 of 82 caught, including 5 new cases for W6 and W7. The weekly table,
weighted by its days, reproduces the period totals of the strategy summary to within 1%.

Verification (2026-11-17): 7 unit tests with hand-computed answers plus a README consistency test;
lint clean; 76 tests in about 12.8 s. Breakage audit: 74 of 74 caught, including 7 new cases for W1
and W2. The 3-hour savings and the share of days cleaner than the day mean were recomputed from
`forecast_history.csv` alone by separate code sharing nothing with the package, and agree to six
decimals for both zones (103 days each). `databricks bundle validate -t dev` passes with the new task.

## QA agent (US-4.1, gate G4), added 2026-11-18

Design rule: a language model is never allowed near a number. A question is routed to one tool; the
tool computes; a fixed template writes the sentence; a check re-reads the sentence and refuses it if
it contains a number that is not one of the tool's facts.

| ID | Item | Where | Status |
| :--- | :--- | :--- | :--- |
| Q1 | Three deterministic tools, each traced with `mlflow.trace`: cleanest window, grid comparison, model health. They read the Gold tables on Databricks and their committed snapshots locally, and share the window ranking with the dashboard extracts | `agent/tools.py` | done 2026-11-18, verified locally and on the workspace, CI green on `ce73405`: the Gold-table branch ran on Databricks in runs `868631481796785` and `296832682509939`, see Q8 |
| Q2 | Keyword router, answer templates, and refusals that contain no figures (out of scope, zone not named, request the tools cannot serve) | `agent/qa_agent.py` | done (verified locally, CI green) |
| Q3 | Grounding check: every number in a finished answer must equal a tool fact at the precision written; otherwise the answer is withheld | `agent/qa_agent.py` | done (verified locally, CI green) |
| Q4 | Model router: a chat model may return only a tool name and arguments, validated against a fixed schema; its prose is discarded; any invalid reply or backend failure falls back to the keyword router | `agent/qa_agent.py` | run against four Databricks hosted models on 2026-11-18 (see Q7 and the table below); not run against Ollama |
| Q5 | MLflow traces of a real question | | checked locally 2026-11-18: one question from the command line produced one trace with an agent span (`qa_agent.answer`) containing a tool span (`check_model_health`). Tracing is switched off in the tests. Verified on the workspace 2026-11-18: three questions run from a laptop were logged to the experiment `carbon-qa-agent-traces` (id 3346646492912694) and read back with independent code. Each trace has state OK and an agent span holding the question and the answer text, identical to the answer returned locally; the two answered questions each have a child tool span with the tool's arguments and its 10 facts; the question with no zone has no tool span and the clarifying reply. The stored traces contain no credential (searched for the access token). **Not covered:** the model call is not a span, so a trace does not show the model's raw reply (Q6); and these three traces came from the committed snapshots. Traces from inside Databricks are under Q8 |
| Q6 | `mlflow.openai.autolog()` for the model router, named in the sprint plan | | not implemented |
| Q8 | The agent run inside Databricks: the task `carbon-agent-check` (job `carbon_qa_agent_check`) asks the 18 evaluation questions with the keyword router, reading the Gold tables, saves each answer with its facts and source to the export volume, and fails if any answer is wrong or ungrounded | `jobs.py`, `databricks.yml` | done 2026-11-18, verified on the workspace, CI green on `ce73405`: 18 of 18 in both runs. Checked from outside: the tools re-run on the volume's CSV copies of the two tables give the same facts (8 of 11 answers exactly, the 3 model-health answers to the 6 significant figures the CSV keeps), and the DE-LU window was worked out by hand with pandas (start 00:00 UTC on 1 January 2026, mean 274.558 against a day mean of 309.784). **Not covered:** the model router inside Databricks, and live data (the tables come from the static snapshot) |
| Q9 | Found by the first workspace run: 14 of 18 traces arrived. The task ended while the last four were still queued for sending. Fix: the task flushes the trace queue before it ends | `jobs.run_agent_check_job` | done 2026-11-18, fixed and verified on the workspace, CI green on `ce73405`: the second run added exactly 18 traces, all with state OK, 11 with a tool span and 7 without |
| Q7 | Found by the first live run: with no zone in the question, all four models chose DE-LU instead of leaving the agent to ask; and asked for a 30-hour window, one model answered for 24 instead of letting the agent refuse. Fix: a model route is rejected if its zone is not the one the question names, or if its window length differs from one the question states in digits; the keyword router then asks or refuses in fixed words. The stand-in tests had not caught either, because the stand-in only replied what the test told it to | `agent/qa_agent.py` (`llm_route`) | done: fixed, verified locally and on the four models, CI green |

Verification (2026-11-18): 38 tests, including hand-computed tool results, an independent
recomputation on the committed snapshots, 18 evaluation questions (`agent/eval_questions.json`), and
tampering tests for the grounding check; lint clean; 117 tests in about 12.7 s. Breakage audit: 90 of
90 caught, including 8 new cases for G4. The README quotes one real answer, and a test fails if the
agent stops giving it. After the Q7 fix: 42 agent tests, 122 in total in about 14 s, and the
breakage audit catches 92 of 92, including 2 new cases for the Q7 guards.

Live model run (2026-11-18, Databricks model serving, temperature 0, the 18 evaluation questions):

| Model endpoint | Before the Q7 fix | After | Routed by the model, after |
| :--- | :--- | :--- | :--- |
| `databricks-meta-llama-3-3-70b-instruct` (the default) | 16 of 18 | 18 of 18 | 11 of 11 answerable |
| `databricks-meta-llama-3-1-8b-instruct` | 16 of 18 | 18 of 18 | 11 of 11 |
| `databricks-gpt-oss-20b` | 17 of 18 | 18 of 18 | 11 of 11 |
| `databricks-qwen3-next-80b-a3b-instruct` | 16 of 18 | 18 of 18 | 11 of 11 |

"Before" is the one run of all four models made before any fix: every wrong answer was a guessed
zone. `gpt-oss-20b` shortened the 30-hour window in the next run, made with only the zone guard in
place (17 of 18), and not in the first, so a single run does not show everything a model can do.
The other 7 questions have no tool answer, and are refused or clarified by the keyword router after
the model declines or is rejected.
The tool description for model health was also widened, because all four models first declined
"What was the coverage in Germany last week?". The unpaced first run hit the workspace rate limit (HTTP
429) on 6 calls; each fell back to the keyword router and still answered correctly. This is one run
per model on 18 questions written by the author, so it shows the router works, not how often it
would err on other wording.

Limits: the agent answers three kinds of question and refuses the rest. Locally it reads the forecast
snapshot written by the local pipeline (origin 12 September 2025), so its "latest forecast" is a past
one. On the workspace it reads the Gold table, whose latest forecast was issued at 22:00 UTC on
31 December 2025, the end of the static snapshot. A model router that picks the wrong valid tool gives a grounded answer to the wrong question;
the grounding check cannot catch that. The Q7 guards cover the zone and a window length written in
digits; a length written in words ("two hours") is still the model's to read.

## Slide data (added 2026-11-18)

The numbers behind the presentation's charts, and the code that draws them, are tracked in
`dashboards/slides/` so that anyone can check or redraw them. The PowerPoint files stay untracked.

| ID | Item | Where | Status |
| :--- | :--- | :--- | :--- |
| S1 | Every charted number is computed by `export_deck_data.py` from the result tables in `dashboards/data/`, the committed snapshot and this register, and stored in `deck_data.json`. Figures that were typed into slide text are looked up from the same file | `dashboards/slides/` | done for the charts (verified locally, CI green): every chart series comes from `deck_data.json`, and the restructured export gives a file identical to the one the decks were built from. Still typed by hand: the champion version and the number of recalibration runs (in the export), and in slide text the test run time ("about 13 seconds") and the lead-time bands where FR coverage holds |
| S2 | Staleness check: `deck_data.json` records a SHA-256 of each input file; a test compares them with the current files and re-reads the counts and quoted figures, and names what changed | `tests/test_slides_data.py` | done (verified locally, CI green) |
| S3 | The `dashboards` task refreshes the slide data after rewriting the result tables; `slides-data` and `slides` tasks refresh it and rebuild the decks | `Makefile`, `scripts/dev.ps1` | partly verified: `dev.ps1 slides-data` and `dev.ps1 dashboards` were run on Windows 2026-11-18, and the `dashboards` run rewrote the result tables and the slide data byte for byte. **Not run:** the `slides` task as one command (it needs one environment holding both the package and the `slides` extra; the decks were built in a separate one), and the three new `make` targets (no `make` on the Windows machine; CI runs only `make test`) |
| S4 | Limit: the check compares inputs, it does not recompute. A change to package code that leaves every input file unchanged is not flagged by it | | accepted; the package's tests cover that code |
| S5 | Limit: adding a test or a register item makes the slide data stale by design, so the export has to be re-run before such a commit passes | | accepted |

Verification (2026-11-18): lint clean; 126 tests in about 14.4 s, 4 of them new; breakage audit 94 of
94 caught, including 2 new cases that switch off each half of the staleness check. The recorded
fingerprints equal those of the files as Git stores them, so the check gives the same answer on a
Windows checkout and in CI.

## Dependencies

- B before D: fixing lags and partial hours changes the training data.
- C1 before D: the purge changes which rows are in calibration.
- D and E before F: the hypothesis notebook should run once, on the final model.
- G is independent of B–F except G3, which needs the final train and gate functions.
