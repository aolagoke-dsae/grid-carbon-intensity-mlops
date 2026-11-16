# Locked-Test Results: Calibration and H1–H3

Generated from `dashboards/data/locked_test_calibration.csv`, `dashboards/data/hypothesis_results.csv`
and `dashboards/data/locked_test_run.json`. Every number below is read from those files.

## How to read this

- **Plan:** `docs/ANALYSIS_PLAN_H1_H3.md`, commit `6c8e730`, approved and pushed before the run.
- **Run:** one execution of `notebooks/02_cqr_calibration_analysis.py --run` on 2026-11-15, then `--check`,
  which recomputed everything and reproduced the files exactly.
- **Period:** origins `2025-09-13 10:00` to `2025-12-31 21:00` UTC, 124,955 forecast rows,
  16 weekly recalibrations. Autumn and early winter only.
- **Model:** frozen; base models fitted on Train only; change-based target; per-lead-time offsets from the
  trailing 30 days, recomputed every 7 days.
- **Not blind.** The Locked Test had been scored several times against an earlier model during the review of
  2026-11-15, and an early EDA included its period. See section 1 of the plan. Earlier reported figures
  (91.63% / 89.26% coverage and the first H1–H3 results) came from an uncommitted notebook and are retired.
- **Intervals** are 95% moving-block bootstrap intervals over days (7-day blocks, 10,000 resamples).

## 1. Calibration of the 90% interval

| Zone | Coverage | 95% interval | Below p05 | Above p95 | Mean width | Reference band [87%, 96%] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DE-LU | 89.0% | 85.5% – 92.2% | 5.8% | 5.2% | 266.7 | inside |
| FR | 87.4% | 83.9% – 90.3% | 6.0% | 6.6% | 19.3 | inside |

Both zones are inside the reference band, FR only just (87.35% against a floor of 87%). Both intervals
contain the nominal 90%, FR's barely (upper end 90.3%). With the earlier static calibration the same
period gave FR about 80%.

### By horizon bucket

| Zone | Bucket | Coverage | 95% interval | Mean width | Median pinball | Persistence pinball | Band |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DE-LU | `short_1_6h` | 89.7% | 87.0% – 92.0% | 151.2 | 16.95 | 22.67 | inside |
| DE-LU | `mid_7_12h` | 88.8% | 84.6% – 92.1% | 248.5 | 30.86 | 38.99 | inside |
| DE-LU | `day_ahead_13_24h` | 88.8% | 84.8% – 92.7% | 333.9 | 41.23 | 47.66 | inside |
| FR | `short_1_6h` | 89.1% | 85.9% – 92.1% | 14.1 | 1.63 | 2.13 | inside |
| FR | `mid_7_12h` | 87.8% | 84.4% – 90.6% | 19.2 | 2.37 | 3.29 | inside |
| FR | `day_ahead_13_24h` | 86.3% | 82.2% – 89.7% | 21.9 | 2.96 | 3.73 | **outside** |

**FR day-ahead under-covers.** At `13–24h` FR reaches 86.3%, outside the band, and its interval
(82.2% – 89.7%) excludes 90%. FR coverage falls from 89.8% at 1 hour to
85.2% at 17 hours. DE-LU stays between 88.3% and 90.3% at every lead time.
See `docs/images/locked_test_coverage_by_horizon.png`.

The median beats persistence in every zone and bucket. By single lead time it loses in FR at 1 h (0.989 vs 0.771).

### Conditional on renewable share at the target hour

| Zone | Quartile | Coverage | 95% interval | Below p05 | Above p95 | Median pinball | Persistence pinball |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DE-LU | Q1 | 85.5% | 79.9% – 90.5% | 0.4% | 14.1% | 31.33 | 40.75 |
| DE-LU | Q2 | 94.1% | 91.2% – 96.1% | 1.8% | 4.1% | 28.79 | 40.95 |
| DE-LU | Q3 | 93.1% | 88.4% – 96.2% | 4.6% | 2.3% | 31.30 | 38.92 |
| DE-LU | Q4 | 83.3% | 77.0% – 91.0% | 16.4% | 0.2% | 38.77 | 36.24 |
| FR | Q1 | 85.5% | 80.8% – 89.5% | 4.0% | 10.5% | 2.75 | 3.83 |
| FR | Q2 | 84.0% | 79.1% – 88.6% | 4.6% | 11.4% | 3.07 | 3.87 |
| FR | Q3 | 90.5% | 87.5% – 93.0% | 5.9% | 3.5% | 2.25 | 2.78 |
| FR | Q4 | 89.4% | 83.9% – 93.7% | 9.5% | 1.1% | 1.86 | 2.40 |

**Calibration is not uniform across regimes.** In DE-LU the highest-renewable quartile (Q4) is covered
83.3% of the time, and 16.4% of its hours fall below `p05`: the model over-predicts intensity
in the greenest hours. In that quartile the median is also worse than persistence (38.77 vs 36.24).
The lowest quartile (Q1) is covered 85.5%, with 14.1% of hours above `p95`. The marginal 89% is an average
of over-coverage in the middle and under-coverage at both ends.

### By month

| Zone | 2025-09 | 2025-10 | 2025-11 | 2025-12 |
| :--- | :--- | :--- | :--- | :--- |
| DE-LU | 91.2% | 85.1% | 91.6% | 89.2% |
| FR | 91.0% | 84.9% | 90.8% | 84.3% |

## 2. Hypotheses

Verdict rule (approved): the corrected analysis decides; the registered analysis is shown beside it; a
disagreement is reported as "sensitive to method". Significance level 0.01, one-sided.

| Hypothesis | Registered | Corrected | Verdict |
| :--- | :--- | :--- | :--- |
| H1 | not supported | not supported | **falsified** |
| H2 | not supported | not supported | **falsified (descriptive; outcome known in advance)** |
| H3 | not supported | not supported | **falsified** |

All three are falsified, and in each case the registered and corrected analyses agree.

### H1: renewable penetration and volatility (DE-LU): falsified

| Part | Analysis | Estimate (top − bottom quartile) | 95% interval | Result |
| :--- | :--- | :--- | :--- | :--- |
| a: lower intensity | corrected, median realized | -379.5 | -412.1 to -340.8 | supported |
| a: lower intensity | registered, Mann-Whitney on `p50` | -279.2 | p = 3.3e-209 | supported |
| b: wider interval | corrected, mean width | -33.1 | -56.3 to -6.5 | **not supported** |
| b: wider interval | registered, Mann-Whitney on width | -33.1 | p = 1.00 | **not supported** |

Each quartile holds 657 target hours, each counted once. The interval is *narrower*, not wider, in
high-renewable hours, so the two-part claim fails. Part a is close to definitional, since a higher
renewable share lowers the intensity computed from the mix.

Context, not part of the verdict:

| | Top quartile (high renewables) | Bottom quartile |
| :--- | :--- | :--- |
| Mean width | 238.5 | 271.6 |
| Width relative to the median forecast | 0.98 | 0.54 |
| Mean absolute error of the median | 77.0 | 62.6 |
| Coverage | 83.5% | 85.5% |

The realized error is *larger* in high-renewable hours while the interval is narrower. So the falsification
does not show that high-renewable hours are less uncertain. It shows that this model's intervals do not
widen where its errors grow. That is a limitation of the calibration method, which applies one offset per
lead time regardless of regime.

### H2: midday parity, DE-LU versus FR: falsified (descriptive)

Of 106 days with complete midday data, 53 had a DE-LU solar plus wind share above the period median.
DE-LU was below FR on 0 of those 53 days. Mean midday intensity was 207.4 gCO2eq/kWh in DE-LU
against 26.8 in FR, and the smallest daily gap was 69.9 gCO2eq/kWh.
With the threshold taken from the same dates one year earlier the result is the same (59 qualifying days,
217.2 against 28.0).

This was known before the run: in the whole 2024–2025 snapshot no hour has DE-LU below FR. Two limits
apply. The period has no summer, which is the regime the hypothesis was about. And the intensity is
production-based, so it says nothing about what a consumer in either zone draws once cross-border flows are counted.

### H3: horizon decay of persistence: falsified

H3 needs both an importance reversal (a) and growing uncertainty (b). Part b holds; part a does not.

**H3a, importance ratio (origin-state / calendar).** A value above 1 means origin-state features matter more.

| Zone | `1–6h` ratio | 95% interval | `13–24h` ratio | 95% interval | Reversal |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DE-LU | 7.44 | 5.17 – 12.02 | 4.67 | 2.97 – 8.37 | no |
| FR | 5.37 | 4.34 – 6.87 | 4.08 | 2.98 – 6.01 | no |

The ratio falls with lead time in both zones, which is the direction the hypothesis expected, but it never
drops below 1. Origin-state features still dominate at day-ahead lead times. The registered approximation
(single-feature importances) gives the same answer. Two caveats: the model has no day-ahead structural
features, so only calendar features could overtake; and with a change-based target the origin intensity
enters every forecast additively, which makes it important by construction.

**H3b, growth of uncertainty with lead time.**

| Analysis | Zone | Mean daily difference (`13–24h` − `1–6h`) | Days | Share of days positive | p |
| :--- | :--- | :--- | :--- | :--- | :--- |
| corrected: absolute error, all origin hours | DE-LU | 47.24 | 109 | 0.982 | 1.0e-19 |
| corrected: absolute error, all origin hours | FR | 2.62 | 109 | 0.890 | 1.9e-17 |
| registered: width, 00:00 UTC origins | DE-LU | 189.93 | 106 | 1.000 | 2.0e-19 |
| registered: width, 00:00 UTC origins | FR | 10.08 | 106 | 0.991 | 2.1e-19 |
| registered: width, 00:00 UTC origins | pooled | 100.00 | 212 | n/a | 7.9e-37 |

Errors are larger at day-ahead than at short lead times in both zones, on the corrected measure that does
not depend on how the interval is built. The pooled registered p-value is small mainly because nearly every
pair has the same sign and the two zones are counted as independent; it should not be read as effect size.

## 3. What these results support, and what they do not

- **Supported:** with weekly recalibration the 90% interval covers 89.0% in DE-LU and 87.4% in FR over an
  held-out autumn and early winter, and the median beats persistence in every zone and bucket.
- **Not supported:** uniform calibration. FR day-ahead (86.3%) is under-covered. So are the lowest and highest
  renewable-share quartiles in DE-LU (85.5% and 83.3%) and the two lowest in FR (85.5% and 84.0%).
  A user scheduling load into the greenest DE-LU hours is in that zone's worst-calibrated regime.
- **Not supported:** any of H1–H3 as registered.
- **Scope:** one period of about 3.5 months, two zones, production-based intensity at median emission factors.
  The interval excludes emission-factor uncertainty, which for FR is larger than the forecast interval.
- **Selection effects:** the model's settings were chosen on Gate-Val, and the decision to recalibrate on a
  schedule was prompted by an earlier look at this same split. The coverage figures are therefore not a
  fully independent confirmation.
- **No changes follow from this run.** The model stays frozen, as the plan requires.
