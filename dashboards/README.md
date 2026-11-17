# Dashboards: build notes

Two dashboards, one per tool, with different jobs and no shared visuals.

| | Tableau | Power BI |
| :--- | :--- | :--- |
| Question | When should I run my load, and what is it worth? | Is the model still trustworthy? |
| Reader | An operator of flexible demand | Whoever maintains the model |
| Files | `forecast_history.csv`, `window_candidates.csv`, `window_strategy_summary.csv` | `gold_monitoring.csv`, `locked_test_calibration.csv`, `window_strategy_summary.csv`, `window_value_weekly.csv` |

All files are in `dashboards/data/`. Neither dashboard computes a forecast or a statistic that
matters: every number is produced by tested code and the tools only display it.

## Rules for both

- **Source line, on every dashboard:** `Source: ENTSO-E Transparency Platform; data: 50Hertz, Amprion,
  TenneT, TransnetBW (DE-LU) and RTE (FR). Intensity values and forecasts are the author's own
  computations.` Naming the source is a condition of the data owners' terms (see the root `README.md`).
- **Plain words for a reader who was not in the room:**
  - Say "carbon intensity", never "intensity" alone, and define it once on each dashboard: grams of
    CO2-equivalent emitted for each kWh of electricity generated.
  - Say once what the zone codes are: DE-LU is the Germany-Luxembourg market zone, FR is France.
  - Label axes with the unit in words, "g CO2 per kWh", and lead time as "hours ahead".
  - Name the strategies as shown in the table below, not by their column names.
- **Zone colours, everywhere:** DE-LU `#d95f02` (orange), FR `#1b9e77` (green). These match the plots
  in `docs/images/`. Where colour separates strategies instead of zones, use the zone colour for the
  forecast and grey for the others.
- **Status is never colour alone.** Show the status word or a different marker shape next to the colour.
- **One screen, no scrolling.**
- **The data is not live.** The snapshot ends on 31 December 2025. Title the date picker "Forecast
  date (historical)" so nobody reads "tomorrow" literally.
- **All times are UTC.** A daily forecast is issued at 00:00 UTC and covers the following 24 hours.
- **No money and no annual totals.** The project has no price data, and the held-out period covers
  September to December only.

Strategy names to display:

| In the data (`strategy`) | Show as | What it is |
| :--- | :--- | :--- |
| `median` | Forecast, median | The window with the lowest median forecast |
| `cautious` | Forecast, upper bound | The window with the lowest upper bound (p95). The name in the data is historical: on the held-out period this choice was **not** safer than the median, so do not label it "cautious" |
| `yesterday` | Repeat yesterday | Yesterday's cleanest hours, reused today |
| `hindsight` | Hindsight (best possible) | The cleanest window knowing the outcome; the ceiling |

## Refreshing the data

```bash
python -m carbon_mlops.inference.dashboard_extracts   # forecast history, windows, strategy summary, weekly value
python -m carbon_mlops.monitoring.compute_drift       # gold_monitoring.csv
```

or `make dashboards` and `make monitor` (`.\\scripts\\dev.ps1 dashboards` / `monitor` on Windows).
On Databricks the inference job writes the same files to the volume `workspace.carbon_mlops.exports`.
Then refresh the extract in each tool. `locked_test_calibration.csv` is fixed: it is the result of
the one held-out evaluation and is never regenerated.

---

## Tableau: "When should I run my load?"

### Data model

Connect `window_candidates.csv`, `forecast_history.csv` and `window_strategy_summary.csv` as three
separate sources. The controls are parameters, which work across sources without relationships.

Parameters to create:

| Parameter | Values | Default |
| :--- | :--- | :--- |
| Zone | DE-LU, FR | DE-LU |
| Forecast date (historical) | all `forecast_date` values | latest |
| Window length | 2, 3, 4, 6 | 3 |
| Choose by | median, upper bound | median |
| Load (MWh per day) | 0.1 to 100 | 1 |

One calculated field, `Is recommended`:
`IF [Choose by] = "median" THEN [recommended_median] ELSE [recommended_cautious] END`.
Filter visuals 1 to 4 on Zone and Window length, and visuals 1 to 3 on Forecast date as well.

**Units.** Carbon intensity in gCO2eq/kWh is the same number as kg CO2 per MWh, so
`avoided_kg_per_mwh` is the CO2 avoided by moving 1 MWh into a window instead of running at an
arbitrary time of day. Multiply by `Load (MWh per day)` for kg avoided.

### Visuals

| # | Visual | Sheet type | Build |
| :--- | :--- | :--- | :--- |
| 1 | Headline numbers | Three text tiles | Filter `window_candidates` to `Is recommended`. Tile A, "Cleanest window": `start_timestamp` to `end_timestamp` (format `HH:mm`). Tile B, "Expected carbon intensity in that window": `expected_p50` with `expected_p95` beneath as "up to". Tile C, "What happened": `avoided_kg_per_mwh * Load` as kg CO2 avoided, with `saving_vs_day_mean` beneath as a percentage |
| 2 | Forecast for the day | Line and band | From `forecast_history`: `target_timestamp` on columns; band between `p05` and `p95`; line for `p50`; dashed line for `realized`. Shade the recommended window on the time axis from its `start_timestamp` to its `end_timestamp`. Y axis "g CO2 per kWh" |
| 3 | Every possible start hour | Bar chart | From `window_candidates`: `start_horizon_h` on columns, `expected_p50` (or `expected_p95` when choosing by the upper bound) on rows. Highlight the recommended bar. Add `realized_mean` as a dot on the same axis |
| 4 | Did it pay off? | Calendar heatmap | From `window_candidates` filtered to `Is recommended`, ignoring the date: week on columns, weekday on rows, colour by `saving_vs_day_mean` with a diverging palette centred on zero |
| 5 | What it was worth | Bar chart, both zones on one axis | From `window_strategy_summary` filtered to `period = held_out` and the chosen window length: strategy on rows, `total_avoided_kg_per_daily_mwh * Load / 1000` as tonnes CO2 avoided, coloured by zone. Keep one shared axis: FR's small bars are the message |
| 6 | How often it backfired | Bar chart | Same source and filters, without `hindsight`: `share_of_days_dirtier_than_day_mean` per strategy and zone, labelled with `worst_day_saving_vs_day_mean` as "worst day" |

Visuals 5 and 6 ignore Zone, Forecast date and Choose by; visual 5 scales with Load and visual 6 does not.

### What to say about it

- "Avoided" compares the chosen window with running at an arbitrary time that day, not with an
  existing schedule.
- Hindsight is the best window knowing the outcome. No forecast can reach it; it is the ceiling.
- The strategy charts compare strategies only on days where all of them made a choice.
- Choosing by the upper bound did not help: on the held-out period it avoided less and backfired more
  often than choosing by the median. Say so; it is a finding, not a flaw in the chart.
- The two grids differ by a factor of more than ten. That is a contrast, not a choice: nobody can
  move a factory.

---

## Power BI: "Is the model still trustworthy?"

### Data model

Load `gold_monitoring.csv` as `Monitoring`, `locked_test_calibration.csv` as `Calibration`,
`window_strategy_summary.csv` as `Strategy` and `window_value_weekly.csv` as `ValueWeekly`. They need
no relationships with each other; add a small `Zones` table (DE-LU, FR) related to each, so one
slicer filters every visual and the zone colours are set in one place. Set `monitor_date` and
`week_start` to type Date.

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

Value Delivered =
CALCULATE (
    AVERAGE ( Strategy[mean_avoided_kg_per_mwh] ),
    Strategy[period] = "held_out",
    Strategy[window_hours] = 3,
    Strategy[strategy] = "median"
)
```

### Visuals

| # | Visual | Type | Build |
| :--- | :--- | :--- | :--- |
| 1 | Status cards | Five cards, each showing both zones with a status word | `Latest Weekly Coverage`; held-out coverage with its interval (`Calibration` where `breakdown = "all"`); `Latest Skill`; `Drift Alerts Latest Week`; `Value Delivered` with `share_of_hindsight_value_captured` |
| 2 | Weekly coverage (control chart) | Line chart | `Monitoring` filtered to `metric = "coverage_7d"`: `monitor_date` on the axis, `value` as values, one line per zone. Show the accepted band 0.87 to 0.96 and the target 0.90. Mark weeks outside the band with a different marker shape, from `status` |
| 3 | Coverage by lead time | Line chart with error bars | `Calibration` filtered to `breakdown = "horizon_h"`: `level` (as a whole number) on the axis, labelled "hours ahead"; `coverage` as values; one line per zone. Error bars from `coverage_ci_low` and `coverage_ci_high`. Constant line at 0.90 |
| 4 | Coverage by renewable share | Markers with error bars, not columns | `Calibration` filtered to `breakdown = "renewable_quartile"`: `level` (Q1 lowest to Q4 highest) on the axis, `coverage` as values, error bars as in visual 3. Columns would need an axis from zero, which hides differences of a few points |
| 5 | Feature drift by week | Matrix | `Monitoring` filtered to `metric_group = "feature_drift_same_season"` and one zone: `metric` on rows, `monitor_date` on columns, `value` as values. Background by rules in one hue from light to dark: below 0.10, 0.10 to 0.25, 0.25 and above |
| 6 | Skill over time | Line chart | `Monitoring` filtered to `metric = "median_skill_vs_persistence"`: `monitor_date` on the axis, `value` as values, one line per zone. Constant line at 0 |
| 7 | Value over time | Line chart, one zone at a time | `ValueWeekly` filtered to `window_hours = 3` and `strategy` in `median`, `yesterday`: `week_start` on the axis, `mean_avoided_kg_per_mwh` as values, one line per strategy. Constant line at 0. It shows whether value fades when skill does |

### What to say about it

- **Weekly coverage is noisy.** A week holds a few thousand strongly correlated forecasts. Single
  weeks outside the band are expected; a run of them is the signal.
- **The drift thresholds assume independent samples**, which a month of hourly data is not, so the
  matrix shows a lot of "alert". Read it by comparing features and zones, not cell by cell.
- **Visuals 3 and 4 and the second and fifth cards come from the one held-out evaluation** and do not
  change. The rest come from the monitor and the extracts and refresh.
- Skill is the share of the error of a "no change" forecast that the model removes. Below zero means
  the model did worse than assuming no change from now.

---

## Columns

`forecast_history.csv`: `zone`, `forecast_date`, `period` (`validation` or `held_out`),
`origin_timestamp`, `horizon_h`, `target_timestamp`, `target_hour_utc`, `p05`, `p50`, `p95`,
`interval_width`, `realized`, `covered`, `calibrated_as_of`.

`window_candidates.csv`: one row per zone, day, window length and start hour. `start_horizon_h` is
the lead time of the window's first hour (1 means 01:00 UTC). `expected_p50` and `expected_p95` are
the forecast means over the window; `realized_mean` is what happened. `saving_vs_day_mean` is
`1 - realized_mean / day_mean_realized`. `regret` is `realized_mean` minus that of the cleanest
window in hindsight. `avoided_kg_per_mwh` is `day_mean_realized - realized_mean`: kg CO2 avoided per
MWh moved into the window. `hindsight_rank` is 1 for the cleanest window of the day. The flags
`recommended_median`, `recommended_cautious`, `recommended_yesterday` and `best_in_hindsight` mark
each strategy's choice.

`window_strategy_summary.csv`: per zone, period, window length and strategy: `n_days`,
`mean_saving_vs_day_mean`, `share_of_days_cleaner_than_day_mean`, `mean_realized_intensity`,
`mean_regret`, `share_of_days_best_window`, `median_hindsight_rank`, and the value columns:
`mean_avoided_kg_per_mwh`; `total_avoided_kg_per_daily_mwh` (what a load of 1 MWh a day avoided over
the period); `share_of_days_dirtier_than_day_mean`; `worst_day_saving_vs_day_mean` (negative means
dirtier); `share_of_hindsight_value_captured` (the strategy's total as a share of hindsight's).

`window_value_weekly.csv`: per zone, window length, week (`week_start`, a Monday) and strategy:
`n_days`, `mean_avoided_kg_per_mwh`, `mean_saving_vs_day_mean`,
`share_of_days_dirtier_than_day_mean`. Only days on which every strategy made a choice are counted,
so a week can hold fewer than seven days.

`gold_monitoring.csv` and `locked_test_calibration.csv` are described in the root `README.md` and in
`docs/LOCKED_TEST_RESULTS.md`.
