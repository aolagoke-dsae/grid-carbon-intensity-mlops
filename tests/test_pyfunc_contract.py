"""Unit tests for ConformalCarbonForecaster & automated Gate-Val promotion (Gate G3)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import CAUSAL_FEATURE_COLS, build_horizon_features
from carbon_mlops.jobs import (
    run_gate_job,
    run_monitoring_job,
    run_recalibration_job,
    run_training_job,
)
from carbon_mlops.models.conformal import (
    add_conformity_scores,
    calibrate_offsets,
    conformal_quantile,
    select_calibration_window,
    walk_forward_calibrated_predictions,
)
from carbon_mlops.models.evaluate_gate import (
    GateEvaluation,
    decide_promotion,
    pinball_ratio,
    summarize_gate_predictions,
)
from carbon_mlops.models.pyfunc_wrapper import (
    ConformalCarbonForecaster,
    enforce_cqr_invariants,
)
from carbon_mlops.models.train import (
    compute_interval_metrics,
    recalibrate_forecaster,
    train_split_cqr,
)
from tests.test_no_leakage import _make_synthetic_bronze


class _CrossingStubModel:
    """Stub regressor intentionally producing crossing & negative predictions to test invariants."""

    def __init__(self, shift: float) -> None:
        self.shift = shift

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.shift, dtype=float)


def test_gate_g3_pyfunc_non_crossing_and_non_negative() -> None:
    models = {
        "DE-LU": {
            "q05": _CrossingStubModel(250.0),
            "q50": _CrossingStubModel(180.0),
            "q95": _CrossingStubModel(-15.0),
        },
        "FR": {
            "q05": _CrossingStubModel(20.0),
            "q50": _CrossingStubModel(30.0),
            "q95": _CrossingStubModel(45.0),
        },
    }
    offsets = {
        ("DE-LU", 1): (12.5, 18.0),
        ("FR", 2): (4.0, 6.0),
    }
    forecaster = ConformalCarbonForecaster(
        models_by_zone=models, conformal_offsets=offsets, target_mode="level"
    )

    assert "lightgbm>=4.3.0" in forecaster.pip_requirements
    assert "scipy>=1.13.0" in forecaster.pip_requirements

    sample = pd.DataFrame(
        {
            "zone": ["DE-LU", "FR"],
            "horizon_h": list(range(1, 3)),
            "horizon_bucket": ["short_1_6h", "short_1_6h"],
            **{col: list(range(1, 3)) for col in CAUSAL_FEATURE_COLS if col != "horizon_h"},
        }
    )

    preds = forecaster.predict(context=None, model_input=sample)
    assert list(preds.columns) == ["p05", "p50", "p95", "interval_width"]
    assert (preds["p05"] >= 0.0).all()
    assert (preds["p05"] <= preds["p50"]).all()
    assert (preds["p50"] <= preds["p95"]).all()
    assert np.allclose(preds["interval_width"], preds["p95"] - preds["p05"])

    # A caller-supplied bucket label is ignored: offsets follow horizon_h alone.
    relabelled = sample.assign(horizon_bucket="day_ahead_13_24h")
    pd.testing.assert_frame_equal(forecaster.predict(context=None, model_input=relabelled), preds)

    # A horizon without a calibrated offset is an error, never a silent zero offset.
    with pytest.raises(ValueError, match="No conformal offsets"):
        forecaster.predict(context=None, model_input=sample.assign(horizon_h=[5, 6]))


def test_delta_target_mode_adds_origin_intensity() -> None:
    models = {"FR": {k: _CrossingStubModel(v) for k, v in [("q05", -3), ("q50", 0), ("q95", 4)]}}
    sample = pd.DataFrame(
        {
            "zone": ["FR", "FR"],
            "horizon_h": [1, 1],
            **{col: [30.0, 50.0] for col in CAUSAL_FEATURE_COLS if col != "horizon_h"},
        }
    )
    forecaster = ConformalCarbonForecaster(
        models_by_zone=models, conformal_offsets={("FR", 1): (1.0, 2.0)}, target_mode="delta"
    )
    preds = forecaster.predict(context=None, model_input=sample)
    assert np.allclose(preds["p50"], [30.0, 50.0])
    assert np.allclose(preds["p05"], [30.0 - 3 - 1, 50.0 - 3 - 1])
    assert np.allclose(preds["p95"], [30.0 + 4 + 2, 50.0 + 4 + 2])

    with pytest.raises(ValueError, match="target_mode"):
        ConformalCarbonForecaster(models_by_zone=models, conformal_offsets={}, target_mode="x")


def test_reported_p50_is_the_middle_value_when_the_model_median_leaves_its_bounds() -> None:
    # Rows: median inside the calibrated bounds, below the lower one, above the upper one.
    raw_p05 = np.array([50.0, 50.0, 50.0])
    raw_p50 = np.array([60.0, 40.0, 99.0])
    raw_p95 = np.array([90.0, 90.0, 90.0])
    p05, p50, p95 = enforce_cqr_invariants(
        raw_p05, raw_p50, raw_p95, q_low=np.full(3, 2.0), q_high=np.full(3, 3.0)
    )

    # Calibrated bounds are 48 and 93. Inside them the model median is reported unchanged.
    assert (p05[0], p50[0], p95[0]) == (48.0, 60.0, 93.0)
    # Outside them the interval stretches to hold the model median, and the published p50 is
    # the calibrated bound next to it. It is then not the model's own median.
    assert (p05[1], p50[1], p95[1]) == (40.0, 48.0, 93.0)
    assert (p05[2], p50[2], p95[2]) == (48.0, 93.0, 99.0)


def test_conformal_quantile_is_the_finite_sample_order_statistic() -> None:
    scores = np.arange(1.0, 101.0)
    # ceil((100 + 1) * 0.95) = 96th smallest of 100 scores.
    assert conformal_quantile(scores, 0.95) == 96.0
    # With too few scores the finite-sample rule falls back to the largest one.
    assert conformal_quantile(np.array([1.0, 5.0, 2.0]), 0.95) == 5.0
    with pytest.raises(ValueError, match="empty"):
        conformal_quantile(np.array([]), 0.95)


class _FixedRawQuantiles:
    """Stub exposing predict_raw_quantiles with constant level-scale quantiles."""

    def predict_raw_quantiles(self, model_input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"raw_q05": 90.0, "raw_q50": 100.0, "raw_q95": 110.0}, index=model_input.index
        )


def test_conformity_scores_are_positive_exactly_when_truth_is_outside() -> None:
    labelled = pd.DataFrame({"target_carbon_intensity": [80.0, 100.0, 125.0]})
    scored = add_conformity_scores(_FixedRawQuantiles(), labelled)

    # Truth below the lower quantile: positive low score. Above the upper: positive high score.
    assert scored["low_score"].tolist() == [10.0, -10.0, -35.0]
    assert scored["high_score"].tolist() == [-30.0, -10.0, 15.0]

    with pytest.raises(ValueError, match="realized targets"):
        unlabelled = pd.DataFrame({"target_carbon_intensity": [np.nan]})
        add_conformity_scores(_FixedRawQuantiles(), unlabelled)


def _scored_history(n_hours: int = 24 * 40) -> pd.DataFrame:
    """Synthetic scored rows for one zone and one horizon, one origin per hour."""
    origins = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 10.0, size=n_hours)
    return pd.DataFrame(
        {
            "zone": "FR",
            "horizon_h": 3,
            "origin_timestamp": origins,
            "target_timestamp": origins + pd.Timedelta(hours=3),
            "target_carbon_intensity": 100.0 + noise,
            "carbon_intensity_origin": 100.0,
            "raw_q05": 100.0,
            "raw_q50": 100.0,
            "raw_q95": 100.0,
            "low_score": -noise,
            "high_score": noise,
        }
    )


def test_calibration_window_uses_only_targets_observed_before_as_of() -> None:
    scored = _scored_history()
    as_of = pd.Timestamp("2025-02-01", tz="UTC")
    window = select_calibration_window(scored, as_of=as_of, window_days=7)

    assert (window["target_timestamp"] < as_of).all()
    assert (window["origin_timestamp"] >= as_of - pd.Timedelta(days=7)).all()
    # Origins whose 3h-ahead target is not yet realized at as_of are excluded.
    assert window["origin_timestamp"].max() == as_of - pd.Timedelta(hours=4)


def test_walk_forward_never_calibrates_on_the_rows_it_predicts() -> None:
    cfg = CarbonMLOpsConfig(recalibration_window_days=14, recalibration_step_days=7)
    scored = _scored_history()
    eval_start = pd.Timestamp("2025-01-20", tz="UTC")
    eval_end = scored["origin_timestamp"].max()
    preds = walk_forward_calibrated_predictions(scored, eval_start, eval_end, cfg=cfg)

    assert len(preds) == int((scored["origin_timestamp"] >= eval_start).sum())
    assert (preds["calibrated_as_of"] <= preds["origin_timestamp"]).all()
    assert preds["calibrated_as_of"].nunique() == 3
    covered = (preds["target_carbon_intensity"] >= preds["p05"]) & (
        preds["target_carbon_intensity"] <= preds["p95"]
    )
    assert 0.85 <= covered.mean() <= 0.97

    # Corrupting the evaluated rows' own scores must not change their intervals.
    tampered = scored.copy()
    last_block = tampered["origin_timestamp"] >= preds["calibrated_as_of"].max()
    tampered.loc[last_block, ["low_score", "high_score"]] = 1e6
    preds_tampered = walk_forward_calibrated_predictions(tampered, eval_start, eval_end, cfg=cfg)
    pd.testing.assert_series_equal(preds["p95"], preds_tampered["p95"])


def test_calibrate_offsets_is_per_zone_and_horizon() -> None:
    scored = _scored_history()
    other = scored.assign(horizon_h=12, low_score=scored["low_score"] * 3, high_score=7.0)
    offsets = calibrate_offsets(pd.concat([scored, other]), nominal_coverage=0.90)

    assert set(offsets) == {("FR", 3), ("FR", 12)}
    assert offsets[("FR", 12)][1] == 7.0
    assert offsets[("FR", 12)][0] > offsets[("FR", 3)][0]
    with pytest.raises(ValueError, match="empty"):
        calibrate_offsets(scored.iloc[0:0], nominal_coverage=0.90)


def test_us22_mlflow_registry_and_champion_promotion_gate(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    # Keep model artifacts in the test's temp folder; the default would be ./mlruns in the repo.
    experiment_id = mlflow.create_experiment(
        "gate_test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    cfg = CarbonMLOpsConfig()

    bronze = _make_synthetic_bronze(hours=480)
    feats = build_horizon_features(bronze, cfg=cfg, write_output=False, base_dir=tmp_path)
    artifacts = run_training_job(
        cfg=cfg, features_df=feats, n_estimators=15, tracking_uri=tracking_uri
    )
    assert artifacts.run_id is not None
    assert artifacts.model_version is not None

    # Gate-Val predictions are walk-forward: calibrated strictly before each origin.
    gate_preds = artifacts.gate_val_predictions
    assert (gate_preds["calibrated_as_of"] <= gate_preds["origin_timestamp"]).all()
    assert (gate_preds["p05"] <= gate_preds["p50"]).all()
    assert (gate_preds["p50"] <= gate_preds["p95"]).all()
    metrics = artifacts.gate_val_metrics
    for zone_key in ("de_lu", "fr"):
        assert f"gate_val_coverage_{zone_key}" in metrics
        # The trained median must be a better forecast than "no change from now".
        assert (
            metrics[f"gate_val_pinball_p50_{zone_key}"]
            < metrics[f"gate_val_persistence_pinball_p50_{zone_key}"]
        )

    # Logged metrics are exactly what the walk-forward predictions imply.
    truth = gate_preds["target_carbon_intensity"]
    recomputed = float(((truth >= gate_preds["p05"]) & (truth <= gate_preds["p95"])).mean())
    assert metrics["gate_val_coverage"] == pytest.approx(recomputed)
    assert metrics == compute_interval_metrics(gate_preds, cfg, prefix="gate_val")
    assert 0.0 < metrics["gate_val_coverage"] < 1.0

    # The registered forecaster scores every horizon and can be recalibrated on new data.
    full_origin = feats.dropna(subset=["target_carbon_intensity"]).groupby(
        ["zone", "origin_timestamp"]
    )
    one_origin = full_origin.filter(lambda g: len(g) == 24).head(24)
    assert len(artifacts.forecaster.predict(context=None, model_input=one_origin)) == 24
    labelled = artifacts.splits["gate_val"]
    as_of = labelled["target_timestamp"].max() + pd.Timedelta(hours=1)
    recalibrated = recalibrate_forecaster(artifacts.forecaster, labelled, as_of=as_of, cfg=cfg)
    assert recalibrated.models_by_zone is artifacts.forecaster.models_by_zone
    assert recalibrated.calibrated_through == str(as_of)

    # The MLflow-loaded model reproduces the in-memory forecaster.
    loaded = mlflow.pyfunc.load_model(f"runs:/{artifacts.run_id}/model")
    sample = one_origin[["zone", *CAUSAL_FEATURE_COLS]].astype(
        {c: "float64" for c in ("horizon_h", "target_dow", "target_month", "is_weekend")}
    )
    np.testing.assert_allclose(
        loaded.predict(sample)["p50"].to_numpy(),
        artifacts.forecaster.predict(context=None, model_input=sample)["p50"].to_numpy(),
    )

    # ---- Promotion gate with real models (no injected numbers decide anything) ----
    # The synthetic series is almost noise-free, so coverage bounds are opened here; the
    # bound logic itself is tested on its own below.
    gate_cfg = CarbonMLOpsConfig(coverage_bounds=(0.0, 1.0))
    model_name = "conformal_carbon_forecaster"
    client = MlflowClient(tracking_uri=tracking_uri)

    # Poison the run's own logged metrics: the gate must ignore them and re-score.
    client.log_metric(artifacts.run_id, "gate_val_coverage_fr", 0.10)
    client.log_metric(artifacts.run_id, "gate_val_mean_pinball", 9999.0)

    decision = run_gate_job(cfg=gate_cfg, features_df=feats, client=client)
    assert decision.promoted is True, decision.reason
    assert decision.champion is None
    assert decision.challenger.n_rows == len(gate_preds)
    for zone, zone_key in (("DE-LU", "de_lu"), ("FR", "fr")):
        assert decision.challenger.coverage_by_zone[zone] == pytest.approx(
            metrics[f"gate_val_coverage_{zone_key}"]
        )
        assert decision.challenger.mean_pinball_by_zone[zone] == pytest.approx(
            metrics[f"gate_val_mean_pinball_{zone_key}"]
        )
    champ = client.get_model_version_by_alias(model_name, "champion")
    assert str(champ.version) == str(artifacts.model_version)
    assert client.get_model_version(model_name, champ.version).tags["gate_decision"] == "promoted"

    # A genuinely worse challenger (a single tree) whose run claims to be excellent.
    artifacts_bad = run_training_job(
        cfg=cfg, features_df=feats, n_estimators=1, tracking_uri=tracking_uri
    )
    assert artifacts_bad.run_id is not None
    client.log_metric(artifacts_bad.run_id, "gate_val_mean_pinball", 0.001)

    rejected = run_gate_job(cfg=gate_cfg, features_df=feats, client=client)
    assert rejected.promoted is False
    assert rejected.champion is not None
    assert rejected.champion_version == str(artifacts.model_version)
    assert rejected.pinball_ratio_vs_champion is not None
    assert rejected.pinball_ratio_vs_champion > 1.0
    assert any("pinball ratio" in check for check in rejected.failed_checks)
    champ_after = client.get_model_version_by_alias(model_name, "champion")
    assert str(champ_after.version) == str(artifacts.model_version)
    bad_tags = client.get_model_version(model_name, artifacts_bad.model_version).tags
    assert bad_tags["gate_decision"] == "rejected"

    # Champion and challenger were scored on exactly the same Gate-Val rows.
    assert rejected.champion.n_rows == rejected.challenger.n_rows

    # ---- Scheduled recalibration of the champion (the batch-inference job's first task) ----
    new_version = run_recalibration_job(cfg=cfg, features_df=feats, client=client)
    recalibrated_mv = client.get_model_version_by_alias(model_name, "champion")
    assert str(recalibrated_mv.version) == new_version
    assert int(new_version) > int(artifacts_bad.model_version)
    tags = client.get_model_version(model_name, new_version).tags
    assert tags["recalibrated_from_version"] == str(artifacts.model_version)

    labelled = feats.dropna(subset=["target_carbon_intensity"])
    newest = labelled["target_timestamp"].max() + pd.Timedelta(hours=1)
    assert tags["calibrated_through"] == str(newest)

    recalibrated = mlflow.pyfunc.load_model(f"models:/{model_name}@champion").unwrap_python_model()
    # Same base models as the gated champion: identical raw quantiles, only offsets moved.
    pd.testing.assert_frame_equal(
        recalibrated.predict_raw_quantiles(one_origin),
        artifacts.forecaster.predict_raw_quantiles(one_origin),
    )
    assert len(recalibrated.conformal_offsets) == len(cfg.zones) * len(cfg.horizons)
    assert recalibrated.conformal_offsets != artifacts.forecaster.conformal_offsets
    assert recalibrated.calibrated_through == str(newest)
    # ---- Monitoring of the champion (the inference job's monitoring task) ----
    monitor_cfg = CarbonMLOpsConfig(local_data_dir=tmp_path, dashboard_data_dir=tmp_path / "dash")
    monitoring_csv = Path(run_monitoring_job(cfg=monitor_cfg, features_df=feats))
    assert monitoring_csv == tmp_path / "dash" / "gold_monitoring.csv"
    assert monitoring_csv.exists()
    # Twenty days of data hold no full weekly window after Gate-Val starts: an empty table,
    # not an error.
    assert list(pd.read_csv(monitoring_csv).columns)[:3] == ["monitor_date", "zone", "metric_group"]

    # The rejected challenger keeps its alias and never became the recalibrated champion.
    challenger_mv = client.get_model_version_by_alias(model_name, "challenger")
    assert str(challenger_mv.version) == str(artifacts_bad.model_version)


def _evaluation(
    coverage: dict[str, float],
    pinball: dict[str, float] | None = None,
    losing: tuple[str, ...] = (),
) -> GateEvaluation:
    return GateEvaluation(
        coverage_by_zone=coverage,
        mean_pinball_by_zone=pinball or {"DE-LU": 16.0, "FR": 0.80},
        cells_losing_to_persistence=losing,
        n_rows=1000,
    )


def test_gate_requires_every_zone_inside_coverage_bounds() -> None:
    cfg = CarbonMLOpsConfig()
    assert decide_promotion(_evaluation({"DE-LU": 0.90, "FR": 0.89}), None, cfg) == ()

    # Pooled coverage would be 0.90, inside the bounds, but FR alone is below the floor.
    failed = decide_promotion(_evaluation({"DE-LU": 0.95, "FR": 0.85}), None, cfg)
    assert len(failed) == 1
    assert "coverage[FR]" in failed[0]

    too_wide = decide_promotion(_evaluation({"DE-LU": 0.97, "FR": 0.90}), None, cfg)
    assert len(too_wide) == 1
    assert "coverage[DE-LU]" in too_wide[0]

    # The bounds themselves are inclusive.
    assert decide_promotion(_evaluation({"DE-LU": 0.87, "FR": 0.96}), None, cfg) == ()


def test_gate_rejects_a_model_that_loses_to_persistence() -> None:
    cfg = CarbonMLOpsConfig()
    challenger = _evaluation({"DE-LU": 0.90, "FR": 0.90}, losing=("FR/short_1_6h",))
    failed = decide_promotion(challenger, None, cfg)
    assert failed == ("median does not beat persistence in FR/short_1_6h",)


def test_gate_pinball_comparison_is_scale_free_and_ties_promote() -> None:
    cfg = CarbonMLOpsConfig()
    coverage = {"DE-LU": 0.90, "FR": 0.90}
    champion = _evaluation(coverage, {"DE-LU": 16.0, "FR": 0.80})

    assert decide_promotion(champion, champion, cfg) == ()

    # Slightly better in DE-LU, clearly worse in FR. The pooled mean is unchanged
    # (15.9 + 0.9 == 16.0 + 0.8), so a pooled rule would wave this through.
    lopsided = _evaluation(coverage, {"DE-LU": 15.9, "FR": 0.90})
    assert pinball_ratio(lopsided, champion) == pytest.approx((15.9 / 16.0 + 0.90 / 0.80) / 2)
    failed = decide_promotion(lopsided, champion, cfg)
    assert len(failed) == 1
    assert "pinball ratio" in failed[0]

    better = _evaluation(coverage, {"DE-LU": 15.0, "FR": 0.78})
    assert decide_promotion(better, champion, cfg) == ()


def test_summarize_gate_predictions_measures_each_zone_and_bucket() -> None:
    cfg = CarbonMLOpsConfig()
    preds = pd.DataFrame(
        {
            "zone": ["DE-LU", "DE-LU", "FR", "FR"],
            "horizon_bucket": ["short_1_6h", "day_ahead_13_24h", "short_1_6h", "short_1_6h"],
            "target_carbon_intensity": [100.0, 200.0, 30.0, 40.0],
            "carbon_intensity_origin": [110.0, 201.0, 30.5, 39.5],
            "p05": [90.0, 150.0, 31.0, 35.0],
            "p50": [101.0, 190.0, 32.0, 43.0],
            "p95": [120.0, 260.0, 36.0, 45.0],
        }
    )
    evaluation = summarize_gate_predictions(preds, cfg)

    assert evaluation.n_rows == 4
    # FR: the first truth (30) lies below p05 (31), the second is inside.
    assert evaluation.coverage_by_zone == {"DE-LU": 1.0, "FR": 0.5}
    # DE-LU short: model error 1 beats persistence error 10. DE-LU day-ahead: 10 loses to 1.
    # FR short: model errors (2, 3) lose to persistence errors (0.5, 0.5).
    assert evaluation.cells_losing_to_persistence == ("DE-LU/day_ahead_13_24h", "FR/short_1_6h")
    assert evaluation.mean_pinball_by_zone["DE-LU"] > evaluation.mean_pinball_by_zone["FR"]

    with pytest.raises(ValueError, match="no rows for zone"):
        summarize_gate_predictions(preds[preds["zone"] == "FR"], cfg)


def test_scoring_only_recent_calib_rows_changes_no_result(tmp_path: Path) -> None:
    # A 2-day window on 20 days of data: most of Calib is older than the first window.
    cfg = CarbonMLOpsConfig(recalibration_window_days=2, recalibration_step_days=1)
    bronze = _make_synthetic_bronze(hours=480)
    feats = build_horizon_features(bronze, cfg=cfg, write_output=False, base_dir=tmp_path)
    artifacts = train_split_cqr(feats, cfg=cfg, n_estimators=15)

    calib, gate = artifacts.splits["calib"], artifacts.splits["gate_val"]
    window_start = gate["origin_timestamp"].min() - pd.Timedelta(days=2)
    assert (calib["origin_timestamp"] < window_start).any(), "test needs Calib rows to skip"

    full_history = pd.concat([calib, gate], ignore_index=True)
    scored = add_conformity_scores(artifacts.forecaster, full_history)
    reference = walk_forward_calibrated_predictions(
        scored, gate["origin_timestamp"].min(), gate["origin_timestamp"].max(), cfg=cfg
    )
    assert compute_interval_metrics(reference, cfg) == artifacts.gate_val_metrics
