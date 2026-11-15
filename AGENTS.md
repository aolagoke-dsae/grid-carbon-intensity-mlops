# Grid Carbon Intensity MLOps - Architecture & Peer-Reviewed Standards

## Domain & Statistical Scope (`DE-LU` & `FR`)
- **Zones:** `["DE-LU", "FR"]` (Germany-Luxembourg vs. France electricity grids).
- **Horizons:** `1` to `24` hours ahead (`t+1h` .. `t+24h`), grouped into horizon buckets (`1-6h`, `7-12h`, `13-24h`) for reporting. Split Conformalized Quantile Regression (CQR) offsets are calibrated per `(zone, horizon_h)` and recalibrated on a schedule (trailing 30 days, every 7 days; `models/conformal.py`).
- **Target & Quantiles:** Grid carbon intensity (`gCO2eq/kWh`), predicting calibrated `[p05, p50, p95]` (nominal 90% prediction interval).
- **4-Way Chronological Split:** Strict time-ordered split (`Train 60%` / `Calib 15%` / `Gate-Val 10%` / `Locked Test 15%`). `evaluate_gate.py` uses `Gate-Val` only; `Locked Test` is reserved for `H1-H3` in Sprint 3, evaluated as a walk-forward under the recalibration schedule, using daily/non-overlapping aggregation to prevent autocorrelation bias. Splits are purged so that no target hour is shared between them.

## Canonical Project Layout & Names
- Root helper: `Makefile` (Linux CI) and `scripts/dev.ps1` (Windows PowerShell `test` / `run-local`).
- Core package under `src/carbon_mlops/`: `config.py` (`CarbonMLOpsConfig`), `io/store.py` (`read_table`, `write_table`), `ingestion/ingest_entsoe.py`, `features/build_features.py` (`build_horizon_features`), `models/pyfunc_wrapper.py` (`ConformalCarbonForecaster` with explicit `pip_requirements`), `models/train.py`, `models/conformal.py`, `models/evaluate_gate.py`, `jobs.py` (one entry point per Databricks task), `analysis/hypotheses.py`, `inference/batch_score.py`, `monitoring/compute_drift.py`, `viz/diagnostics.py` (`plot_probabilistic_forecast`), and `agent/{tools.py,qa_agent.py,eval_questions.json}`.
- Databricks notebooks under `notebooks/` MUST start with `# Databricks notebook source` at line 1.

## Core MLOps & CQR Quality Gates (`G1–G4`)
- **G1 (Cryptographic Provenance):** Preserve upstream payload hashes and append `source_file_sha256` + UTC ingestion timestamp with zero hardcoded secrets (`tests/test_ingestion_provenance.py`).
- **G2 (Zero Temporal Leakage):** Enforce `max_feature_timestamp <= origin_timestamp` (including any known-future day-ahead schedules published before `origin_timestamp`) and `target_timestamp == origin_timestamp + horizon_h` (`tests/test_no_leakage.py`).
- **G3 (Conformal Calibration & Promotion):** Enforce `0 <= p05 <= p50 <= p95` and explicit `pip_requirements` in `ConformalCarbonForecaster` (`tests/test_pyfunc_contract.py`), and in `evaluate_gate.py` re-score `@challenger` and `@champion` on the same Gate-Val window, then require: coverage within `[0.87, 0.96]` in every zone (not pooled), a median that beats persistence in every zone and horizon bucket, and a scale-free pinball ratio versus `@champion` of at most 1. Run `python scripts/mutation_audit.py` at the end of each phase.
- **G4 (Zero-Hallucination Agent QA):** Every number cited by `agent/qa_agent.py` must come strictly from deterministic `@mlflow.trace` tools (`find_greenest_window`, `compare_zones_24h`, `check_model_health`) in `agent/tools.py` over Gold tables/snapshots (`tests/test_agent_tools.py`).

## Python, Linting & Workflow Standards
- Code must pass `ruff check` and `pytest` locally in <15 seconds.
- Use explicit Python type hints, `dataclasses` for `CarbonMLOpsConfig`, and structured `logging` (never `print()` in `src/`).
- Summarize planned file changes in 2–3 bullet points before modifying existing files.
