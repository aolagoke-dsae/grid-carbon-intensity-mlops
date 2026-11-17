# Dashboards: build notes

Two dashboards, one per tool, with different jobs and no shared visuals.

| | Tableau | Power BI |
| :--- | :--- | :--- |
| Question | When should I run my load tomorrow, and how sure is the forecast? | Is the model still trustworthy? |
| Reader | An operator of flexible demand | Whoever maintains the model |
| Files | `forecast_history.csv`, `window_candidates.csv`, `window_strategy_summary.csv` | `gold_monitoring.csv`, `locked_test_calibration.csv` |

All files are in `dashboards/data/`. Neither dashboard computes a forecast or a statistic that
matters: every number is produced by tested code and the tools only display it.

## Rules for both

- **Source line, on every dashboard:** `Source: ENTSO-E Transparency Platform; data: 50Hertz, Amprion,
  TenneT, TransnetBW (DE-LU) and RTE (FR). Intensity values and forecasts are the author's own
  computations.` Naming the source is a condition of the data owners' terms (see the root `README.md`).
- **Zone colours, everywhere:** DE-LU `#d95f02` (orange), FR `#1b9e77` (green). These match the plots
  in `docs/images/`.
- **Status is never colour alone.** Show the status word or an icon next to the colour.
- **One screen, no scrolling.**
- **The data is not live.** The snapshot ends on 31 December 2025. Title the date picker "Forecast
  date (historical)" so nobody reads "tomorrow" literally.
- **All times are UTC.** A daily forecast is issued at 00:00 UTC and covers the following 24 hours.

## Refreshing the data

```bash
python -m carbon_mlops.inference.dashboard_extracts   # forecast history, windows, strategy summary
python -m carbon_mlops.monitoring.compute_drift       # gold_monitoring.csv
```

or `make dashboards` and `make monitor` (`.\\scripts\\dev.ps1 dashboards` / `monitor` on Windows).
On Databricks the inference job writes the same files to the volume `workspace.carbon_mlops.exports`.
Then refresh the extract in each tool. `locked_test_calibration.csv` is fixed: it is the result of
the one held-out evaluation and is never regenerated.

---

## Tableau: "When should I run tomorrow?"

### Data model

Connect `window_candidates.csv` as the main source and relate `forecast_history.csv` to it on
`zone` and `forecast_date`. Add `window_strategy_summary.csv` as a separate source for the
strategy comparison.

Parameters to create:

| Parameter | Values | Default |
| :--- | :--- | :--- |
| Zone | DE-LU, FR | DE-LU |
| Forecast date (historical) | all `forecast_date` values | latest |
| Window length | 2, 3, 4, 6 | 3 |
| Risk | median, cautious | median |

One calculated field, `Is recommended`:
`IF [Risk] = "median" THEN [recommended_median] ELSE [recommended_cautious] END`.
Filter every sheet on Zone, Forecast date and Window length.

### Visuals

| # | Visual | Sheet type | Build |
| :--- | :--- | :--- | :--- |
| 1 | Headline numbers | Three text tiles | Filter `window_candidates` to `Is recommended`. Tile A: `start_timestamp` to `end_timestamp` (format `HH:mm`). Tile B: `expected_p50` with `expected_p95` beneath as "up to". Tile C: `saving_vs_day_mean` as a percentage, labelled "cleaner than the day's average (realized)" |
| 2 | Fan chart | Dual-axis line and area | From `forecast_history`: `target_timestamp` on columns; area between `p05` and `p95`; line for `p50`; dashed line for `realized`. Add a reference band on the time axis from `start_timestamp` to `end_timestamp` of the recommended window |
| 3 | Every possible start hour | Bar chart | From `window_candidates`: `start_horizon_h` on columns, `expected_p50` (or `expected_p95` when Risk is cautious) on rows. Colour by `Is recommended`. Add `realized_mean` as a dot on the same axis |
| 4 | Did it pay off? | Calendar heatmap | From `window_candidates` filtered to `Is recommended`, ignoring the date parameter: week number on columns, weekday on rows, colour by `saving_vs_day_mean` with a diverging palette centred on zero |
| 5 | Two grids, one day | Two small fan charts side by side | Same as visual 2 for both zones at once, **each with its own axis**. FR sits near 30 and DE-LU near 370, so a shared axis flattens FR |

Add a small table under visual 4 from `window_strategy_summary.csv` (filter `period = held_out` and
the chosen window length): `strategy`, `mean_saving_vs_day_mean`, `share_of_days_cleaner_than_day_mean`.
It shows the forecast against the "repeat yesterday's cleanest hours" rule and against hindsight.

### What to say about it

- `saving_vs_day_mean` compares the chosen window with running at an arbitrary time that day. It is
  a relative saving in intensity, not in tonnes: converting needs the load in MWh.
- `hindsight` is the best window knowing the outcome. No forecast can reach it; it is the ceiling.
- The summary compares strategies only on days where all of them made a choice.
- Visual 5 contrasts two grids. It is not a choice between them: nobody can move a factory.

---

## Power BI: "Is the model still trustworthy?"

### Data model

Load `gold_monitoring.csv` as `Monitoring` and `locked_test_calibration.csv` as `Calibration`. They
are not related; each feeds its own visuals. Set `monitor_date` to type Date. Slicers: `zone`,
`monitor_date` (range), and for the calibration visuals `breakdown`.

`Monitoring` is in long format: one row per `monitor_date`, `zone` and `metric`. Filter a visual to
the metric it shows. Useful measures:

```DAX
Latest Date = MAX ( Monitoring[monitor_date] )

Latest Weekly Coverage =
CALCULATE (
    AVERAGE ( Monitoring[value] ),
    Monitoring[metric] = "coverage_7d",
    Monitoring[monitor_date] = [Latest Date]
)

Weeks Below Band =
CALCULATE (
    COUNTROWS ( Monitoring ),
    Monitoring[metric] = "coverage_7d",
    Monitoring[status] = "below_band"
)

Drift Alerts Latest Week =
CALCULATE (
    COUNTROWS ( Monitoring ),
    Monitoring[metric_group] = "feature_drift_same_season",
    Monitoring[status] = "alert",
    Monitoring[monitor_date] = [Latest Date]
)

Latest Skill =
CALCULATE (
    AVERAGE ( Monitoring[value] ),
    Monitoring[metric] = "median_skill_vs_persistence",
    Monitoring[monitor_date] = [Latest Date]
)
```

### Visuals

| # | Visual | Type | Build |
| :--- | :--- | :--- | :--- |
| 1 | Status cards | Four cards | `Latest Weekly Coverage`, overall coverage (`Calibration[coverage]` where `breakdown = "all"`), `Latest Skill`, `Drift Alerts Latest Week`. Conditional formatting on the first from the band 0.87 to 0.96 |
| 2 | Control chart | Line chart | `Monitoring` filtered to `metric = "coverage_7d"`: `monitor_date` on the axis, `value` as values, `zone` as legend. Constant lines at 0.87, 0.90 and 0.96; shade between the outer two. Markers on, coloured by `status` |
| 3 | Coverage by lead time | Line chart with error bars | `Calibration` filtered to `breakdown = "horizon_h"`: `level` (as a whole number) on the axis, `coverage` as values, `zone` as legend. Error bars from `coverage_ci_low` and `coverage_ci_high`. Constant line at 0.90 |
| 4 | Coverage by renewable regime | Clustered column chart | `Calibration` filtered to `breakdown = "renewable_quartile"`: `level` (Q1 to Q4) on the axis, `coverage` as values, `zone` as legend. Constant line at 0.90 |
| 5 | Drift matrix | Matrix | `Monitoring` filtered to `metric_group = "feature_drift_same_season"` and one zone: `metric` on rows, `monitor_date` on columns, `value` as values. Background colour by rules: below 0.10, 0.10 to 0.25, 0.25 and above |
| 6 | Skill over time | Line chart | `Monitoring` filtered to `metric = "median_skill_vs_persistence"`: `monitor_date` on the axis, `value` as values, `zone` as legend. Constant line at 0 |

### What to say about it

- **Weekly coverage is noisy.** A week holds a few thousand strongly correlated forecasts. Single
  weeks outside the band are expected; a run of them is the signal.
- **The drift thresholds assume independent samples**, which a month of hourly data is not, so the
  matrix shows a lot of "alert". Read it by comparing features and zones, not cell by cell.
- **Visuals 3 and 4 come from the one held-out evaluation** and do not change. Visuals 1, 2, 5 and 6
  come from the monitor and refresh.
- Skill below zero means the model did worse than assuming no change from now.

---

## Columns

`forecast_history.csv`: `zone`, `forecast_date`, `period` (`validation` or `held_out`),
`origin_timestamp`, `horizon_h`, `target_timestamp`, `target_hour_utc`, `p05`, `p50`, `p95`,
`interval_width`, `realized`, `covered`, `calibrated_as_of`.

`window_candidates.csv`: one row per zone, day, window length and start hour. `start_horizon_h` is
the lead time of the window's first hour (1 means 01:00 UTC). `expected_p50` and `expected_p95` are
the forecast means over the window; `realized_mean` is what happened. `saving_vs_day_mean` is
`1 - realized_mean / day_mean_realized`. `regret` is `realized_mean` minus that of the cleanest
window in hindsight. `hindsight_rank` is 1 for the cleanest window of the day. The flags
`recommended_median`, `recommended_cautious`, `recommended_yesterday` and `best_in_hindsight` mark
each strategy's choice.

`window_strategy_summary.csv`: per zone, period, window length and strategy: `n_days`,
`mean_saving_vs_day_mean`, `share_of_days_cleaner_than_day_mean`, `mean_realized_intensity`,
`mean_regret`, `share_of_days_best_window`, `median_hindsight_rank`.

`gold_monitoring.csv` and `locked_test_calibration.csv` are described in the root `README.md` and in
`docs/LOCKED_TEST_RESULTS.md`.
