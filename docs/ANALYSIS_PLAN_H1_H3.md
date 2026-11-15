# Locked-Test Analysis Plan for H1–H3 (Phase F)

**Status: APPROVED by the author on 2026-11-15, with the four proposed decisions of section 7.
No number in this plan has been computed on the Locked Test with the current model.** This file is
committed before any analysis code reads the Locked Test, and the commit hash is recorded with the
results.

## 1. What this plan can and cannot restore

The Locked Test is not blind, and this plan does not pretend otherwise.

- The hypotheses H1–H3 and their tests were written in `docs/CONCEPT_NOTE.md` before the split
  was scored. They are reproduced unchanged in section 5 as the **registered** analyses.
- An earlier evaluation produced H1–H3 results that cannot be reproduced: the notebook was never
  committed. Those figures (including 91.63% / 89.26% coverage) are **retired** and must not be cited.
- During the pre-freeze review of 2026-11-15, the Locked Test was scored several times against the
  earlier model (static calibration, bucket offsets). That review looked at coverage by zone,
  bucket, horizon and renewable-share quartile, at interval width and absolute error by regime, at
  the DE-LU versus FR midday comparison, and at width and error by lead time. **The direction of
  all three hypotheses is therefore already known for the earlier model.**
- The Sprint 1 EDA originally plotted the test period. It has been regenerated without it.

What the plan does restore: every analysis choice for the **current, frozen** model is fixed here
before that model is scored on the Locked Test, the registered tests are run as written so nothing
is quietly swapped, and every correction added by the review is labelled as such.

## 2. Model and evaluation protocol (fixed)

- **Model:** the frozen model of `docs/REVIEW_ISSUES.md`, "Model freeze": change-based target,
  200 trees, per-`(zone, horizon_h)` offsets, 30-day window, 7-day step. Base models are fitted on
  Train only and are **not** refitted on Calib or Gate-Val.
- **Evaluation:** walk-forward over the Locked Test. Every 7 days from the first test origin, the
  offsets are recomputed from the trailing 30 days of errors whose targets were already observed
  (initially Gate-Val rows, then test rows as they realize), and applied to the next 7 days of
  origins. This is the same replay the promotion gate uses on Gate-Val.
- **Period:** origins `2025-09-13 10:00 UTC` to `2025-12-31 21:00 UTC` (autumn and early winter only).
- **Data rules:** as in Phases B and C. Partial and incomplete hours are never an origin, a lag
  or a target; the 4 missing FR hours stay missing.
- **One run.** The analysis is executed once on the Locked Test. Whatever it shows, the model, the
  calibration settings and this plan are not changed afterwards. A later model change would need
  new data to be evaluated on, which this project does not have.

## 3. Unit of analysis and uncertainty (fixed)

Hourly forecasts are strongly autocorrelated, and each target hour appears up to 24 times (once
per lead time). Treating rows as independent overstates the evidence.

- **Unit:** the UTC calendar day of the origin, unless a section says otherwise. Hypothesis tests
  use one value per day. The test period has about 109 days.
- **Intervals:** moving-block bootstrap over days, block length 7 days, 10,000 resamples, seed
  `20261115`. Reported as 95% percentile intervals.
- **Significance:** `alpha = 0.01`, one-sided, as registered. A hypothesis made of two claims is
  supported only if both hold. No further multiplicity correction, as registered; secondary
  analyses never decide a verdict.

## 4. Headline calibration results (not a hypothesis)

Reported for each zone separately, never pooled as the main figure:

| Quantity | Breakdown |
| :--- | :--- |
| Empirical coverage of `[p05, p95]` with 95% block-bootstrap interval | per zone; per zone and horizon bucket; per zone and `horizon_h` |
| Miss direction (share below `p05`, share above `p95`) | per zone |
| Mean interval width | per zone and horizon bucket |
| Pinball loss of `p05`, `p50`, `p95`; persistence pinball loss of `p50` | per zone and horizon bucket |
| Coverage by renewable-share quartile at the target hour | per zone (conditional calibration) |
| Coverage by calendar month and by target hour of day | per zone |

**Reference band stated in advance:** the promotion gate's `[0.87, 0.96]` per zone. A zone outside
the band is reported as a calibration failure of the deployed method. It does not trigger retuning.

## 5. Hypotheses

For each hypothesis there are two analyses. The **registered** analysis is the test as written in
the concept note. The **corrected** analysis fixes a validity problem found by the review and was
added after the split had been seen for the earlier model.

**Verdict rule (approved):** the verdict comes from the corrected analysis. The
registered analysis is reported beside it. If the two disagree, the verdict is reported as
"sensitive to method", not as supported or falsified.

### H1: renewable penetration and volatility (DE-LU)

*Claim (registered):* DE-LU hours in the top quartile of renewable share have both a lower median
carbon intensity and a wider calibrated 90% interval than hours in the bottom quartile.

Fixed definitions:
- Renewable share is `renewable_share` at the **target** hour. Quartile thresholds `Q25`, `Q75` are
  those of the DE-LU target hours in the Locked Test. Sensitivity: thresholds from Train.
- Each target hour is used once: its interval width is the mean over the lead times available for it.

| | Registered | Corrected |
| :--- | :--- | :--- |
| H1a lower intensity | One-sided Mann-Whitney U on hourly `p50`, top versus bottom quartile | Difference in median **realized** intensity, top minus bottom; 7-day block bootstrap; supported if the 99% upper bound is below 0 |
| H1b wider interval | One-sided Mann-Whitney U on hourly width `W = p95 - p05` | Difference in mean width `W`, top minus bottom; 7-day block bootstrap; supported if the 99% lower bound is above 0 |

Reported as context, deciding nothing: relative width `W / p50`, mean absolute error and coverage
in each quartile, and H1b by horizon bucket. These answer the question the review raised, namely
whether a narrower absolute interval in high-renewable hours reflects lower uncertainty or only a
lower level.

Known limitation: H1a is close to definitional, because a higher renewable share mechanically
lowers the intensity computed from the generation mix.

### H2: midday parity, DE-LU versus FR

*Claim (registered):* on days when the DE-LU solar plus wind share exceeds its seasonal median,
DE-LU has lower realized intensity than FR during `11:00–15:00` local time.

Fixed definitions:
- Local time is `Europe/Berlin`, daylight-saving aware. The window is the four hours starting
  11:00, 12:00, 13:00 and 14:00 local.
- Daily values are the mean realized intensity over the window, per zone. A day counts only if
  all four hours are present and trusted in both zones.
- "Seasonal median" is the median daily midday solar plus wind share of DE-LU over the test
  period. Sensitivity: the median of the same calendar dates in 2024.

| | Registered | Corrected |
| :--- | :--- | :--- |
| H2 | One-sided paired Wilcoxon signed-rank on qualifying days, `DE-LU < FR` | Same test, plus the count of days with `DE-LU < FR` and the smallest daily gap |

**This is reported as a descriptive result, not a test.** Its outcome is known in advance: in the
whole 2024–2025 snapshot there is no hour in which DE-LU is below FR, and the smallest gap in Train
alone is 85 gCO2eq/kWh. The write-up must also state two scope limits: the test period contains
no summer, which is the regime the hypothesis was about, and the intensity is production-based, so
it says nothing about cross-border flows.

### H3: horizon decay of persistence

*Claim (registered):* origin-state features dominate permutation importance at `1–6h`, calendar
and day-ahead structural signals overtake them at `13–24h`, and the 90% interval is significantly
wider at `13–24h` than at `1–6h`.

Deviation that applies to both analyses: the model has no day-ahead structural features (see
`data/bronze_snapshot/README.md`), so H3a compares origin-state features with calendar features only.

**H3a, importance reversal.** Fixed definitions:
- Groups, permuted together because their members are highly correlated:
  origin-state = `carbon_intensity_origin`, `ci_lag_*`, `ci_roll24_*`, the five `*_share_origin`
  columns and `gen_total_mwh_origin`; calendar = `target_hour_sin`, `target_hour_cos`,
  `target_dow`, `target_month`, `is_weekend`. `horizon_h` is never permuted.
- Importance of a group = increase in pinball loss of the median when the group's values are
  shuffled between origins within a zone, 20 repeats, seed `20261115`. Every row takes the group's
  values from a donor origin at the same lead time, and all lead times of an origin share one donor.
  Only origins with all 24 lead times are used.
- `R = importance(origin-state) / importance(calendar)`, per zone and per horizon bucket, with a
  95% interval from the 7-day block bootstrap.

| | Registered | Corrected |
| :--- | :--- | :--- |
| H3a | "Importance ratio reversal between the `1–6h` and `13–24h` models" (single-feature importances; no such separate models exist) | Supported in a zone if the interval of `R` at `1–6h` lies above 1 and the interval of `R` at `13–24h` lies below 1 |

The registered version cannot be run as written. It is approximated by ungrouped single-feature
importances from the one model per zone, and reported with that caveat.

**H3b, growth of uncertainty with lead time.**

| | Registered | Corrected |
| :--- | :--- | :--- |
| Quantity | Interval width `W` | Absolute error of the median, `abs(y - p50)` |
| Origins | 00:00 UTC only | All 24 origin hours, averaged within the day |
| Daily value | mean `W` at `13–24h` minus mean `W` at `1–6h` | mean error at `13–24h` minus mean error at `1–6h` |
| Test | One-sided paired Wilcoxon, zones as registered | One-sided Wilcoxon signed-rank on daily values, **each zone separately**; supported only if both zones pass |

Why the correction: the width is partly set by the calibration offsets, so it can widen with lead
time by construction, and using only 00:00 origins compares night hours at short lead times with
afternoon hours at long ones. Realized error over all origin hours removes both problems. The
registered width test is still reported, per zone and pooled.

H3 is supported only if H3a holds in both zones and H3b holds in both zones.

## 6. Implementation and controls

1. **Code before data.** The analysis is written as tested functions in
   `src/carbon_mlops/analysis/hypotheses.py` that take a predictions table. They are developed
   and debugged on **Gate-Val** predictions and on synthetic data only.
2. **Tests.** Each statistic has a unit test with a hand-computed answer. The walk-forward over
   the Locked Test gets the same causality tests as Gate-Val, and matching entries in
   `scripts/mutation_audit.py`.
3. **Single execution.** `notebooks/02_cqr_calibration_analysis.py` calls those functions once on
   the Locked Test and writes `dashboards/data/hypothesis_results.csv`,
   `dashboards/data/locked_test_calibration.csv` and the figures.
4. **Reproducibility.** Seeds are fixed. Re-running the notebook must regenerate both CSV files
   identically; the notebook's `--check` mode recomputes them and fails on any difference.
5. **Record.** The results file stores the commit hash of this plan, the model settings and the
   run date. `README.md`, the concept note and the register carry the disclosure of section 1.

## 7. Decisions taken by the author (2026-11-15)

1. **Verdict rule:** the corrected analysis decides; the registered analysis is reported beside
   it; disagreement is reported as "sensitive to method".
2. **H3a threshold:** the interval of `R` must lie entirely above 1 at `1–6h` and entirely below 1
   at `13–24h`.
3. **H2:** reported as a descriptive result.
4. **Base model:** kept fitted on Train only.

## 8. Corrections made to the draft before this commit

Both were found while designing the implementation, before any Locked-Test data was read.

- **H3a shuffle rule.** The draft shuffled features "as whole days". That keeps the hour of day
  fixed, so the time-of-day features would never be shuffled and the calendar group's importance
  would be understated. It now shuffles between origins.
- **Reproducibility check.** The draft made it a unit test. That would re-read the Locked Test on
  every test run and break the 15-second test budget. It is now a `--check` mode of the notebook.
