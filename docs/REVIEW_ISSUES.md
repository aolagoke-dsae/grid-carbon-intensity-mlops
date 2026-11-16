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
| F1 | `02_cqr_calibration_analysis.py` and `hypothesis_results.csv` do not exist | `notebooks/`, `dashboards/data/` | verified (pending CI): notebook, results CSV, calibration CSV and run record are committed |
| F2 | Reported coverage does not reproduce (see reference numbers) | notebook, README | verified (pending CI): old figures retired; reproducible figures in `docs/LOCKED_TEST_RESULTS.md` |
| F3 | No confidence intervals on coverage | notebook | verified (pending CI): 7-day block-bootstrap intervals on every coverage figure |
| F4 | H1 width result is a level artifact; needs relative width and conditional coverage | notebook | verified (pending CI): relative width, error and coverage by regime reported as context |
| F5 | H1 pseudo-replication: each target hour appears 24 times | notebook | verified (pending CI): each target hour counted once |
| F6 | H2 was decided before testing (0 of 17,539 hours have DE-LU below FR) and the window has no summer; label it descriptive | notebook, `CONCEPT_NOTE.md` | verified (pending CI): reported as descriptive, with both scope limits |
| F7 | H3 p-value is an n-floor; zones likely pooled as independent | notebook | verified (pending CI): zones tested separately; pooled figure shown with its caveat |
| F8 | H3 width is partly built from calibration offsets; test realized error instead | notebook | verified (pending CI): corrected test uses realized absolute error |
| F9 | H3 uses only 00:00 origins, confounding lead time with time of day | notebook | verified (pending CI): corrected test averages over all 24 origin hours |
| F10 | H3 permutation importance: no per-bucket models exist, and features correlate at 0.99 | notebook | verified (pending CI): grouped permutation importance with bootstrap intervals |
| F11 | Locked test has been evaluated more than once; disclose it | README, `CONCEPT_NOTE.md` | verified (pending CI): disclosed in the plan, the concept note and the results document; root `README.md` written, with tests tying its numbers and links to the committed files |

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
difference) is open, and CI has not been confirmed on these commits.

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
| M1 | Feature drift: PSI per zone and feature, trailing 30 days against Train and against the same 30 days a year earlier | `monitoring/compute_drift.py` | verified locally (pending CI) |
| M2 | Rolling 7-day coverage per zone and horizon bucket, miss direction, width, skill against persistence | `monitoring/compute_drift.py` | verified locally (pending CI) |
| M3 | Monitoring task in the Databricks inference job (`carbon-monitor`) | `jobs.py`, `databricks.yml` | implemented; bundle consistency tested; **not deployed, never run on the workspace** |
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

## Dependencies

- B before D: fixing lags and partial hours changes the training data.
- C1 before D: the purge changes which rows are in calibration.
- D and E before F: the hypothesis notebook should run once, on the final model.
- G is independent of B–F except G3, which needs the final train and gate functions.
