# Scrum-Adapted Implementation Guide (Nov 15 – Dec 10, 2026)

This plan organizes delivery into **four 1-week Sprints**. Every Sprint delivers a working, demo-able vertical slice verified against a strict **Definition of Done (DoD)** so the project remains presentation-ready at the end of every week.

---

## Product Backlog Overview

| Epic | Sprint | Window | Primary Deliverable |
| :--- | :--- | :--- | :--- |
| **Epic 1: Walking Skeleton** | **Sprint 1** | Nov 15 – Nov 21 | Local & Databricks end-to-end vertical slice + EDA plots (`Gates G1, G2`) |
| **Epic 2: MLOps Governance & CI/CD** | **Sprint 2** | Nov 22 – Nov 28 | Unity Catalog Feature Store, `@champion` gate on Gate-Val, Spark UDF scoring & DAB (`Gate G3`) |
| **Epic 3: Hypotheses, Drift & BI** | **Sprint 3** | Nov 29 – Dec 5 | Autocorrelation-safe `H1–H3` evaluation on Locked Test, PSI monitoring, one Tableau and one Power BI dashboard |
| **Epic 4: Agentic QA & Freeze** | **Sprint 4** | Dec 6 – Dec 10 | Grounded QA Copilot (`@mlflow.trace`), Dec 8 code freeze & Dec 11 demo rehearsal (`Gate G4`) |

---

## Sprint 1 (Nov 15 – Nov 21): Walking Skeleton & Data Provenance
**Sprint Goal:** Ship a runnable, unit-tested end-to-end vertical slice from Bronze Parquet snapshot to calibrated `p05–p95` predictions and initial EDA visuals.

### Sprint 1 Backlog
* **US-1.1 — Unified Config, Dual Task Runners & Storage Adapter:**
  * *Story:* As a developer, I want `pyproject.toml`, `Makefile` (Linux/CI), `scripts/dev.ps1` (Windows PowerShell), `src/carbon_mlops/config.py`, and `src/carbon_mlops/io/store.py` so `test` and `run-local` execute identically locally (Parquet) and on Databricks (Unity Catalog Delta).
  * *Tasks:* Scaffold repo directories; implement `CarbonMLOpsConfig` with 4-way chronological split (`60/15/10/15`); implement `read_table()` / `write_table()` Spark-or-Parquet switch.
* **US-1.2 — Provenance Ingestion & Exploratory Diagnostics (`Gate G1`):**
  * *Story:* As a data auditor, I want `ingest_entsoe.py` to load `data/bronze_snapshot/entsoe_generation_sample.parquet` with SHA-256 checksums, and `notebooks/01_eda_and_data_quality.py` to visualize `DE-LU` vs. `FR` regimes and autocorrelation lags.
  * *Tasks:* Export 12–24 months hourly snapshot from `grid-carbon-intensity`; implement `ingest_entsoe.py` + `tests/test_ingestion_provenance.py`; export `eda_zone_distributions.png`, `eda_diurnal_heatmap.png`, and `eda_acf_lags.png` to `docs/images/`.
* **US-1.3 — Leakage-Free Horizon Features & Split-CQR Baseline (`Gate G2`):**
  * *Story:* As an ML engineer, I want `build_features.py` (`t+1..t+24h`), `pyfunc_wrapper.py` (with explicit Serverless `pip_requirements`), `train.py`, and `viz/diagnostics.py` producing valid non-crossing `[p05, p50, p95]` forecasts and fan charts.
  * *Tasks:* Implement `build_horizon_features()` + `tests/test_no_leakage.py`; implement `ConformalCarbonForecaster` + `tests/test_pyfunc_contract.py`; implement `plot_probabilistic_forecast()` + `tests/test_viz.py`.

### Sprint 1 Definition of Done (Review: Nov 21)
* `.\scripts\dev.ps1 test` (and `make test` in CI) passes `ruff` and all 4 Sprint 1 test suites (`G1`, `G2`, `pyfunc`, `viz`).
* `.\scripts\dev.ps1 run-local` completes end-to-end on the Bronze snapshot in <15 seconds and saves fan chart PNGs to `docs/images/`.

---

## Sprint 2 (Nov 22 – Nov 28): Databricks Unity Catalog Governance & CI/CD
**Sprint Goal:** Operationalize the pipeline on Databricks Unity Catalog with automated `@champion` model promotion on Gate-Val, Spark UDF batch scoring, and GitHub Actions CI/CD.

### Sprint 2 Backlog
* **US-2.1 — Unity Catalog Feature Store Materialization:**
  * *Story:* As an ML engineer, I want `build_features.py` to register `grid_horizon_features` via `FeatureEngineeringClient` with composite primary keys `["zone", "origin_timestamp", "horizon_h"]`.
* **US-2.2 — Model Registry & Automated `@champion` Gate (`Gate G3`):**
  * *Story:* As a model owner, I want `train.py` to register `ConformalCarbonForecaster` (with pinned `pip_requirements` for Serverless compatibility) as `@challenger` in Unity Catalog while logging Gate-Val metrics to the MLflow Run, and `evaluate_gate.py` to promote `@champion` only when **Gate-Val** coverage is in `[0.87, 0.96]` in every zone, the median beats persistence in every zone and horizon bucket, and the scale-free pinball ratio versus `@champion` improves/ties, with both models re-scored by the gate on the same window (keeping Locked Test untouched).
* **US-2.3 — Distributed Batch Scoring, DAB & GitHub Actions:**
  * *Story:* As a platform engineer, I want `batch_score.py` (`mlflow.pyfunc.spark_udf` -> `gold_forecasts` + batch CSV extract `dashboards/data/latest_24h_forecasts.csv`), `databricks.yml`, and `.github/workflows/ci_cd.yml` automated.

### Sprint 2 Definition of Done (Review: Nov 28)
* GitHub Actions CI passes (`make test` + `databricks bundle validate -t dev`).
* Both DAB jobs (`carbon_training_pipeline` and `carbon_batch_inference`) succeed on Databricks Free Edition (Serverless), registering `@champion` in Unity Catalog and exporting `latest_24h_forecasts.csv`.

---

## Sprint 3 (Nov 29 – Dec 5): Hypothesis Validation, Drift Monitoring & BI Dashboards
**Sprint Goal:** Formally test pre-registered hypotheses (`H1–H3`) on the Locked Test split, automate feature PSI & coverage drift monitoring, and build two complementary dashboards, one in Tableau and one in Power BI.

### Sprint 3 Backlog
* **US-3.1 — Statistical Hypothesis Evaluation (`H1`, `H2`, `H3`):**
  * *Story:* As a quantitative researcher, I want `notebooks/02_cqr_calibration_analysis.py` to run autocorrelation-safe Mann-Whitney U (with block bootstrap) and daily/non-overlapping paired Wilcoxon tests for `H1–H3` on the **Locked Holdout Test split** and output summary tables/plots.
* **US-3.2 — Automated Drift & Coverage Monitoring:**
  * *Story:* As an MLOps engineer, I want `monitoring/compute_drift.py` integrated into `carbon_batch_inference` to compute feature PSI and rolling 7-day empirical coverage into `gold_monitoring` (`dashboards/data/gold_monitoring.csv`).
* **US-3.3 — Two Complementary Dashboards (Tableau and Power BI):**
  * *Story:* As an energy operator, I want a Tableau dashboard (`dashboards/carbon_forecast.twbx`, published to Tableau Public) that answers "when should I run my load, and how sure is the forecast?": headline numbers, a fan chart with the recommended window, every possible start hour, a calendar of realized savings, and the two grids side by side. As a model owner, I want a Power BI dashboard (`dashboards/model_health.pbix`) that answers "is the model still trustworthy?": status cards, a coverage control chart, coverage by lead time and by renewable regime, a drift matrix, and skill over time. The two share no visuals. Revised 2026-11-17 from four Tableau views, to show both tools without duplication; per-visual build notes are in `dashboards/README.md`.

### Sprint 3 Definition of Done (Review: Dec 5)
* `H1–H3` statistical results (p-values, effect sizes, falsification verdicts) documented on Locked Test.
* Tableau dashboard published to Tableau Public and Power BI dashboard saved as `.pbix`, both carrying the data source line; extract refresh workflow documented in `dashboards/README.md`; screenshots saved to `docs/images/tableau_dashboard.png` and `docs/images/powerbi_dashboard.png`.

---

## Sprint 4 (Dec 6 – Dec 10): Grounded Agentic QA Copilot, Code Freeze & Rehearsal
**Sprint Goal:** Ship the zero-hallucination QA Copilot with MLflow Tracing, enforce a hard code freeze on **December 8**, and rehearse the **December 11** presentation.

### Sprint 4 Backlog
* **US-4.1 — Grounded Grid & MLOps QA Copilot (`Gate G4`):**
  * *Story:* As a dispatch user, I want `src/carbon_mlops/agent/tools.py` (`find_greenest_window`, `compare_zones_24h`, `check_model_health` decorated with `@mlflow.trace`) and `qa_agent.py` (Databricks Foundation Model API + Ollama `localhost:11434/v1` toggle via `mlflow.openai.autolog()`) validated against `eval_questions.json` in `tests/test_agent_tools.py`.
* **US-4.2 — Hard Code Freeze & Documentation Polish (Target: Dec 8):**
  * *Story:* As a presenter, I want all code frozen by end of Dec 8, MLflow Trace & Databricks Job DAG screenshots saved to `docs/images/`, and root `README.md` finalized with architecture diagram, `H1–H3` findings, and Tableau Public link.
* **US-4.3 — Demo Backup & Presentation Rehearsal (Dec 9 – Dec 10):**
  * *Story:* As a speaker, I want 30-second fallback screen recordings of the Databricks Job run, MLflow Trace waterfall, and Tableau dashboard, plus two timed 10–12 minute dry runs.

### Sprint 4 Definition of Done (Review: Dec 10)
* Clean `git status`, green GitHub Actions badge, live Tableau Public link, verified MLflow Traces (`Gate G4`), and rehearsed deck ready for **December 11**.
