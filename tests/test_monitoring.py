"""Unit tests for drift and coverage monitoring (US-3.2), on synthetic data only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import build_horizon_features
from carbon_mlops.models.train import train_split_cqr
from carbon_mlops.monitoring.compute_drift import (
    MONITORED_FEATURES,
    MONITORING_COLUMNS,
    PSI_ALERT,
    PSI_WARN,
    compute_monitoring,
    coverage_status,
    feature_psi_table,
    monitoring_dates,
    population_stability_index,
    psi_status,
    rolling_coverage_table,
    summarize_latest,
    write_monitoring,
)
from tests.test_no_leakage import _make_synthetic_bronze

CFG = CarbonMLOpsConfig()
UTC = "UTC"


# ----------------------------------------------------------------------------- PSI


def test_psi_matches_a_hand_computed_two_bin_example() -> None:
    reference = np.array([0.0] * 50 + [1.0] * 50)  # shares 0.5 / 0.5, edge at 0.5
    current = np.array([0.0] * 90 + [1.0] * 10)  # shares 0.9 / 0.1
    expected = (0.9 - 0.5) * np.log(0.9 / 0.5) + (0.1 - 0.5) * np.log(0.1 / 0.5)
    assert population_stability_index(reference, current, n_bins=2) == pytest.approx(expected)
    assert expected == pytest.approx(0.8789, abs=1e-4)


def test_psi_edge_cases() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(size=5000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-12)
    # Two samples of one distribution give a small value; a shifted one a large value.
    assert population_stability_index(sample, rng.normal(size=5000)) < 0.02
    assert population_stability_index(sample, rng.normal(loc=2.0, size=5000)) > PSI_ALERT
    # A feature that never varies (nuclear share in DE-LU) is stable, not undefined.
    assert population_stability_index(np.zeros(100), np.zeros(40)) == 0.0
    # Missing values are ignored; no data on either side is reported as NaN.
    with_gaps = np.concatenate([sample, [np.nan, np.nan]])
    assert population_stability_index(with_gaps, sample) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(population_stability_index(np.array([]), sample))
    assert np.isnan(population_stability_index(sample, np.array([np.nan])))
    # An empty bin is floored, so the value is finite.
    assert np.isfinite(population_stability_index(np.arange(100.0), np.full(50, 99.0)))


def test_status_thresholds_are_the_documented_conventions() -> None:
    assert (PSI_WARN, PSI_ALERT) == (0.10, 0.25)
    assert [psi_status(v) for v in (0.0, 0.0999, 0.10, 0.2499, 0.25, 3.0)] == [
        "ok",
        "ok",
        "warn",
        "warn",
        "alert",
        "alert",
    ]
    assert psi_status(float("nan")) == "no_data"
    bounds = (0.87, 0.96)
    assert [coverage_status(v, bounds) for v in (0.8699, 0.87, 0.90, 0.96, 0.9601)] == [
        "below_band",
        "ok",
        "ok",
        "ok",
        "above_band",
    ]
    assert coverage_status(float("nan"), bounds) == "no_data"


def test_monitoring_dates_are_weekly_midnights_with_a_full_step_behind_them() -> None:
    dates = monitoring_dates(
        pd.Timestamp("2025-07-02 10:00", tz=UTC), pd.Timestamp("2025-07-24 05:00", tz=UTC), 7
    )
    assert list(dates) == [
        pd.Timestamp("2025-07-10", tz=UTC),
        pd.Timestamp("2025-07-17", tz=UTC),
        pd.Timestamp("2025-07-24", tz=UTC),
    ]


def _origin_features(start: str, days: int, level: float, zone: str = "FR") -> pd.DataFrame:
    """Feature rows for hourly origins, each repeated for two lead times."""
    origins = pd.date_range(start, periods=24 * days, freq="h", tz=UTC)
    frame = pd.DataFrame({"zone": zone, "origin_timestamp": np.repeat(origins, 2)})
    frame["horizon_h"] = np.tile([1, 2], len(origins))
    hour = frame["origin_timestamp"].dt.hour
    for feature in MONITORED_FEATURES:
        frame[feature] = level + hour  # a fixed daily pattern around `level`
    return frame


def test_feature_psi_uses_train_as_reference_and_counts_each_origin_once() -> None:
    cfg = CarbonMLOpsConfig(zones=("FR",))
    train = _origin_features("2024-01-01", days=420, level=100.0)
    later = _origin_features("2025-03-01", days=60, level=100.0)
    features = pd.concat([train, later], ignore_index=True)
    reference_end = train["origin_timestamp"].max()
    dates = pd.DatetimeIndex([pd.Timestamp("2025-04-15", tz=UTC)])

    table = feature_psi_table(features, reference_end, dates, cfg, window_days=30)

    assert list(table.columns) == list(MONITORING_COLUMNS)
    assert set(table["metric_group"]) == {"feature_drift", "feature_drift_same_season"}
    assert len(table) == 2 * len(MONITORED_FEATURES)
    # 30 days of hourly origins, not twice that for the two lead times.
    assert (table["n"] == 30 * 24).all()
    assert np.allclose(table["value"], 0.0, atol=1e-12)
    assert (table["status"] == "ok").all()

    # Shifting only the data after Train moves the PSI: the reference did not move with it.
    shifted = pd.concat([train, _origin_features("2025-03-01", 60, level=500.0)], ignore_index=True)
    drifted = feature_psi_table(shifted, reference_end, dates, cfg, window_days=30)
    assert (drifted["value"] > PSI_ALERT).all()
    assert (drifted["status"] == "alert").all()
    # Exactly the value obtained with Train alone as the reference.
    feature = MONITORED_FEATURES[0]
    train_values = train.drop_duplicates("origin_timestamp")[feature].to_numpy()
    current_values = np.full(30 * 24, 500.0) + np.tile(np.arange(24), 30)
    against_train = drifted[drifted["metric"] == f"psi_{feature}"]["value"].iloc[0]
    assert against_train == pytest.approx(population_stability_index(train_values, current_values))


def test_same_season_psi_holds_the_season_fixed_and_needs_a_complete_reference() -> None:
    cfg = CarbonMLOpsConfig(zones=("FR",))
    # A strong annual cycle that repeats exactly from one year to the next.
    origins = pd.date_range("2024-01-01", periods=24 * 640, freq="h", tz=UTC)
    seasonal = 100.0 + 80.0 * np.sin(2 * np.pi * origins.dayofyear / 365.0)
    features = pd.DataFrame({"zone": "FR", "origin_timestamp": origins, "horizon_h": 1})
    for feature in MONITORED_FEATURES:
        features[feature] = seasonal
    reference_end = pd.Timestamp("2025-02-28 23:00", tz=UTC)
    dates = pd.DatetimeIndex(
        [pd.Timestamp("2025-08-01", tz=UTC), pd.Timestamp("2025-01-20", tz=UTC)]
    )
    table = feature_psi_table(features, reference_end, dates, cfg, window_days=30)
    august = table[table["monitor_date"] == dates[0]].set_index("metric_group")["value"]

    # Against all of Train a summer month looks like drift; against last summer it does not.
    assert august.loc["feature_drift"].min() > PSI_ALERT
    assert august.loc["feature_drift_same_season"].max() < PSI_WARN

    # One year before 20 January 2025 the series has only 20 of the 30 days: not reported.
    january = table[
        (table["monitor_date"] == dates[1]) & (table["metric_group"] == "feature_drift_same_season")
    ]
    assert january["value"].isna().all()
    assert (january["status"] == "no_data").all()


# ----------------------------------------------------------------------------- coverage


def _forecasts(covered_until: str) -> pd.DataFrame:
    """Hourly targets for two weeks; covered before `covered_until`, missed (too low) after."""
    targets = pd.date_range("2025-07-01", periods=24 * 14, freq="h", tz=UTC)
    rows = []
    for bucket, horizon in (("short_1_6h", 1), ("day_ahead_13_24h", 24)):
        frame = pd.DataFrame({"target_timestamp": targets})
        frame["origin_timestamp"] = frame["target_timestamp"] - pd.Timedelta(hours=horizon)
        frame["zone"], frame["horizon_h"], frame["horizon_bucket"] = "FR", horizon, bucket
        rows.append(frame)
    frame = pd.concat(rows, ignore_index=True)
    frame["p05"], frame["p50"], frame["p95"] = 90.0, 100.0, 110.0
    frame["interval_width"] = 20.0
    frame["carbon_intensity_origin"] = 104.0
    hit = frame["target_timestamp"] < pd.Timestamp(covered_until, tz=UTC)
    frame["target_carbon_intensity"] = np.where(hit, 102.0, 80.0)
    return frame


def test_rolling_coverage_matches_hand_computed_windows() -> None:
    cfg = CarbonMLOpsConfig(zones=("FR",))
    # Everything is covered up to 11 July 12:00, half-way through the second week's 4th day.
    preds = _forecasts(covered_until="2025-07-11 12:00")
    dates = pd.DatetimeIndex(
        [pd.Timestamp("2025-07-08", tz=UTC), pd.Timestamp("2025-07-15", tz=UTC)]
    )
    table = rolling_coverage_table(preds, dates, cfg, window_days=7)
    value = table.set_index(["monitor_date", "metric"])["value"]
    count = table.set_index(["monitor_date", "metric"])["n"]

    # Week 1 (targets 1-7 July): all covered. Week 2 (8-14 July): 3.5 of 7 days covered.
    assert value[(dates[0], "coverage_7d")] == 1.0
    assert value[(dates[1], "coverage_7d")] == pytest.approx(0.5)
    assert count[(dates[0], "coverage_7d")] == 7 * 24 * 2
    assert value[(dates[1], "coverage_7d_short_1_6h")] == pytest.approx(0.5)
    assert count[(dates[1], "coverage_7d_day_ahead_13_24h")] == 7 * 24
    # A bucket with no forecasts is reported as missing, not as zero.
    assert np.isnan(value[(dates[1], "coverage_7d_mid_7_12h")])

    assert value[(dates[1], "share_below_p05")] == pytest.approx(0.5)
    assert value[(dates[1], "share_above_p95")] == 0.0
    assert value[(dates[1], "mean_interval_width")] == 20.0
    # Week 1: model error |102 - 100| = 2, persistence error |102 - 104| = 2: no skill.
    assert value[(dates[0], "median_skill_vs_persistence")] == pytest.approx(0.0)
    # Week 2: half the hours are 80. Model error (2 + 20) / 2, persistence (2 + 24) / 2.
    assert value[(dates[1], "median_skill_vs_persistence")] == pytest.approx(1 - 11 / 13)

    status = table.set_index(["monitor_date", "metric"])["status"]
    assert status[(dates[0], "coverage_7d")] == "above_band"
    assert status[(dates[1], "coverage_7d")] == "below_band"
    assert status[(dates[1], "mean_interval_width")] == "info"


def test_rolling_coverage_counts_misses_above_the_interval_too() -> None:
    cfg = CarbonMLOpsConfig(zones=("FR",))
    preds = _forecasts(covered_until="2025-07-20")  # nothing falls below the interval
    # The first two days of week 1 come in above p95 (110).
    high = preds["target_timestamp"] < pd.Timestamp("2025-07-03", tz=UTC)
    preds.loc[high, "target_carbon_intensity"] = 130.0
    dates = pd.DatetimeIndex([pd.Timestamp("2025-07-08", tz=UTC)])
    value = rolling_coverage_table(preds, dates, cfg, window_days=7).set_index("metric")["value"]

    assert value["coverage_7d"] == pytest.approx(5 / 7)
    assert value["share_above_p95"] == pytest.approx(2 / 7)
    assert value["share_below_p05"] == 0.0
    assert value["coverage_7d"] + value["share_above_p95"] + value["share_below_p05"] == (
        pytest.approx(1.0)
    )


def test_rolling_coverage_uses_only_outcomes_observed_before_the_date() -> None:
    cfg = CarbonMLOpsConfig(zones=("FR",))
    preds = _forecasts(covered_until="2025-07-11 12:00")
    date = pd.Timestamp("2025-07-08", tz=UTC)
    dates = pd.DatetimeIndex([date])
    before = rolling_coverage_table(preds, dates, cfg, window_days=7)

    # Changing every outcome from the monitoring date onward must not change its row.
    tampered = preds.copy()
    later = tampered["target_timestamp"] >= date
    tampered.loc[later, "target_carbon_intensity"] = -999.0
    pd.testing.assert_frame_equal(before, rolling_coverage_table(tampered, dates, cfg, 7))

    # A forecast issued inside the window but due after the date is not counted yet.
    in_window = (preds["target_timestamp"] >= date - pd.Timedelta(days=7)) & (
        preds["target_timestamp"] < date
    )
    assert before.set_index("metric").loc["coverage_7d", "n"] == int(in_window.sum())
    issued_before_due_after = (preds["origin_timestamp"] < date) & (
        preds["target_timestamp"] >= date
    )
    assert issued_before_due_after.sum() > 0


# ----------------------------------------------------------------------------- end to end


def test_compute_and_write_monitoring_end_to_end(tmp_path: Path) -> None:
    cfg = CarbonMLOpsConfig(
        recalibration_window_days=4,
        recalibration_step_days=1,
        local_data_dir=tmp_path,
        dashboard_data_dir=tmp_path / "dash",
    )
    bronze = _make_synthetic_bronze(hours=480)
    feats = build_horizon_features(bronze, cfg=cfg, write_output=False, base_dir=tmp_path)
    artifacts = train_split_cqr(feats, cfg=cfg, n_estimators=5)

    table = compute_monitoring(
        artifacts.forecaster, feats, cfg, coverage_window_days=1, psi_window_days=2
    )

    assert list(table.columns) == list(MONITORING_COLUMNS)
    assert set(table["zone"]) == set(cfg.zones)
    assert {"coverage", "forecast_quality", "feature_drift", "feature_drift_same_season"} == set(
        table["metric_group"]
    )
    # Monitoring starts after Train and Calib: at Gate-Val, and runs through the last target.
    gate_start = artifacts.splits["gate_val"]["origin_timestamp"].min()
    assert table["monitor_date"].min() > gate_start
    assert table["monitor_date"].max() <= feats["target_timestamp"].max() + pd.Timedelta(days=1)
    assert table["monitor_date"].nunique() >= 4
    coverage = table[table["metric"] == "coverage_1d"]
    assert coverage["value"].between(0.0, 1.0).all()
    assert (coverage["n"] > 0).all()
    # 20 days of data cannot have a same-season reference a year earlier.
    seasonal = table[table["metric_group"] == "feature_drift_same_season"]
    assert seasonal["value"].isna().all()

    csv_path = Path(write_monitoring(table, cfg))
    assert csv_path == tmp_path / "dash" / "gold_monitoring.csv"
    written = pd.read_csv(csv_path)
    assert len(written) == len(table)
    assert list(written.columns) == list(MONITORING_COLUMNS)
    assert (tmp_path / "lakehouse_local" / "carbon_mlops_gold_monitoring.parquet").exists()

    flagged = summarize_latest(table)
    assert (flagged["monitor_date"] == table["monitor_date"].max()).all()
    assert not flagged["status"].isin(["ok", "info"]).any()
