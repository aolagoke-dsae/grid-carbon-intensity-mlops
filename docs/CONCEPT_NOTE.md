# Concept Note: 24-Hour Probabilistic Grid Carbon Intensity Forecasting & MLOps Pipeline (`DE-LU` & `FR`)

- **Author:** Dr. Adewole Olagoke (`@aolagoke-dsae`)
- **Target Presentation:** GreenBootcamps Capstone Defense — December 11, 2026 (Code Freeze: December 8, 2026)
- **Delivery Methodology:** Scrum-Adapted 4-Sprint Iterative Plan (Nov 15 – Dec 10, 2026)
- **Platform & Stack:** Databricks (Unity Catalog, Feature Engineering, MLflow Registry & Tracing, Asset Bundles), LightGBM + Split Conformalized Quantile Regression (CQR), Tableau Public (packaged extract), GitHub Actions

---

## 1. Business Case & Operational Context
Decarbonizing flexible industrial and municipal electricity demand—such as commercial EV fleets, heat pumps, cold-storage facilities, and batch compute workloads—requires scheduling consumption into low-carbon hours **24 hours in advance**.

Standard scheduling tools rely on **deterministic point forecasts** of grid carbon intensity (`gCO2eq/kWh`). In grids with high variable renewable energy (VRE) penetration like Germany–Luxembourg (`DE-LU`), wind and solar forecast errors create asymmetric tail risks: scheduling a 3-hour industrial batch around an optimistic point forecast can inadvertently trigger peak-hour fossil dispatch (`lignite`/`gas`).

This project operationalizes an end-to-end **Databricks MLOps & Uncertainty-Aware Decision System** for `DE-LU` and France (`FR`) that:
1. Delivers calibrated 24-hour probabilistic forecasts (`p05`, `p50`, `p95`) with guaranteed finite-sample coverage.
2. Publishes batch-exported Gold forecasts and drift diagnostics to **Tableau Public** (via packaged `.twbx` extracts) for operational planning.
3. Provides a **grounded, tool-calling QA Copilot** (with MLflow Tracing) that answers risk-constrained dispatch and model-health questions using deterministic queries over Gold tables—never LLM-invented numbers.

---

## 2. Core Research & Operational Questions
1. **Grid Regime Dynamics (`Sprint 1`):** How do diurnal, seasonal, and renewable-penetration regimes shape the distribution and autocorrelation structure of carbon intensity in `DE-LU` versus nuclear-dominated `FR`?
2. **Horizon & Regime Uncertainty (`Sprint 1–3`):** How does predictive uncertainty (`p95 - p05` width) evolve across lead times (`t+1h` through `t+24h`) and renewable ramp regimes, and can per-`(zone, horizon_bucket)` Split CQR maintain nominal 90% coverage without look-ahead leakage?
3. **Operational Load Shifting (`Sprint 3`):** When and under what upper-bound risk tolerances (`p95`) does domestic `DE-LU` load shifting outperform `FR` carbon parity?
4. **Production Reliability & Grounded QA (`Sprint 2–4`):** How can automated promotion gates (`@champion`), feature drift (PSI) tracking, and trace-audited tool calling prevent silent model degradation and numeric hallucination?

---

## 3. Experimental Design & 4-Way Chronological Split
To prevent repeated model-selection peeking from contaminating statistical hypothesis testing, the historical timeline is partitioned strictly in chronological order into **four non-overlapping blocks**:
1. **Train Split (60%):** Fits base LightGBM quantile regressors (`tau in {0.05, 0.50, 0.95}`).
2. **Conformal Calibration Split (15%):** Computes nonconformity residuals `E_i = max(q_0.05(x_i) - y_i, y_i - q_0.95(x_i))` per `(zone, horizon_bucket)`.
3. **Validation / Promotion Gate Split (10%):** Used exclusively by `evaluate_gate.py` (`Gate G3`) to evaluate `@challenger` vs. `@champion` coverage and pinball loss prior to promotion.
4. **Locked Holdout Test Split (15%):** Sealed until **Sprint 3 (`US-3.1`)** and evaluated once for pre-registered hypotheses `H1–H3`.

*Feature Scope Note:* Features combine strictly causal origin-state grid telemetry (`feature_ts <= origin_timestamp`), target calendar encodings (`target_hour`, `target_dow`, `target_month`), and—where available in the upstream snapshot—day-ahead TSO generation/load schedules published prior to `origin_timestamp`.

---

## 4. Pre-Registered Data-Driven Hypotheses (`H1–H3`)
All hypotheses concern empirical physical properties of the `DE-LU` and `FR` electricity grids and are evaluated strictly on the **Locked Holdout Test split** during **Sprint 3 (`US-3.1`)**, using daily/non-overlapping aggregation to prevent serial autocorrelation from inflating test statistics:

* **H1 — Renewable Penetration & Volatility Regime (`DE-LU`):**
  * *Claim:* In `DE-LU`, hours in the top quartile of renewable generation share (`>= Q75`) exhibit both a **strictly lower median carbon intensity** and a **strictly wider calibrated 90% prediction interval** (steeper ramp uncertainty) than hours in the bottom quartile (`<= Q25`).
  * *Test & Falsification:* One-sided Mann-Whitney U test with block-bootstrap robustness check comparing `p50` and interval width `W = p95 - p05` between `>= Q75` and `<= Q25` regimes (`alpha = 0.01`). Falsified if either difference fails significance.

* **H2 — Midday Cross-Border Decarbonization Parity (`DE-LU` vs. `FR`):**
  * *Claim:* Although `FR` has a lower overall mean carbon intensity due to nuclear baseload, `DE-LU` achieves **strictly lower realized carbon intensity than `FR` during midday windows (`11:00–15:00 CET`)** on days when `DE-LU` solar+wind share exceeds its seasonal median.
  * *Test & Falsification:* Paired Wilcoxon signed-rank test on **daily midday mean** carbon intensity (`CI_DE-LU,d < CI_FR,d`, one independent pair per qualifying Test day `d`) (`alpha = 0.01`). Falsified if `FR` remains cleaner even during high-renewable midday windows.

* **H3 — Horizon Decay of Autoregressive Persistence:**
  * *Claim:* Origin-state grid features (`carbon_intensity_origin`, `wind_share_origin`) dominate permutation feature importance at short lead times (`h = 1..6h`), whereas deterministic diurnal/calendar and day-ahead structural signals overtake origin-state features at day-ahead lead times (`h = 13..24h`), accompanied by a **statistically significant widening of 90% interval width** (`W_13-24h > W_1-6h`).
  * *Test & Falsification:* Permutation importance ratio reversal between `1–6h` and `13–24h` models plus paired Wilcoxon test on interval width across **non-overlapping daily `00:00 UTC` forecast origins** (`alpha = 0.01`).

---

## 5. Engineering & MLOps Quality Gates (`G1–G4`)

| Gate | Sprint | Contract Description | Verification Mechanism |
| :--- | :--- | :--- | :--- |
| **G1: Cryptographic Provenance** | Sprint 1 | Every Bronze snapshot ingested preserves upstream payload hashes and appends `source_file_sha256` + UTC ingestion timestamp with zero hardcoded secrets. | `tests/test_ingestion_provenance.py` |
| **G2: Zero Temporal Leakage** | Sprint 1 | For every `(origin_timestamp, horizon_h)` row, `max_feature_timestamp <= origin_timestamp` and `target_timestamp == origin_timestamp + horizon_h`. | `tests/test_no_leakage.py` |
| **G3: Conformal Calibration & Promotion** | Sprint 1–2 | `@challenger` achieves empirical **Gate-Val** coverage within `[0.87, 0.96]`, satisfies `0 <= p05 <= p50 <= p95`, declares explicit Serverless `pip_requirements`, and beats/ties `@champion` pinball loss before promotion. | `evaluate_gate.py` & `tests/test_pyfunc_contract.py` |
| **G4: Zero-Hallucination Agent QA** | Sprint 4 | Every numeric figure cited by the Grid & MLOps QA Copilot is strictly traceable to `@mlflow.trace` deterministic tool outputs over Gold tables. | `tests/test_agent_tools.py` & MLflow Tracing |

---

## 6. Sprint-Aligned Product Epics (Nov 15 – Dec 10, 2026)
* **Epic 1 (`Sprint 1`):** Walking Skeleton — Bronze snapshot ingestion, leakage-free horizon features, Split-CQR baseline, and EDA diagnostics.
* **Epic 2 (`Sprint 2`):** Lakehouse Governance & CI/CD — Unity Catalog Feature Store, `@champion` promotion gate on Gate-Val, Spark UDF batch scoring, DAB, and GitHub Actions.
* **Epic 3 (`Sprint 3`):** Empirical Validation & BI — Formal `H1–H3` hypothesis testing on Locked Test, PSI/coverage drift monitoring, and packaged Tableau Public dashboard.
* **Epic 4 (`Sprint 4`):** Grounded Agentic QA & Defense — Tool-calling QA Copilot with MLflow Tracing, Dec 8 code freeze, and Dec 11 GreenBootcamps rehearsal.
