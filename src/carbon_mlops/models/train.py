"""Chronological 4-way split training, per-horizon rolling Split CQR, and MLflow logging."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import lightgbm as lgb
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.features.build_features import CAUSAL_FEATURE_COLS
from carbon_mlops.io.store import is_databricks_runtime
from carbon_mlops.models.conformal import (
    add_conformity_scores,
    calibrate_offsets,
    select_calibration_window,
    walk_forward_calibrated_predictions,
)
from carbon_mlops.models.pyfunc_wrapper import ConformalCarbonForecaster

logger = logging.getLogger(__name__)


QUANTILE_LEVELS: dict[str, float] = {"q05": 0.05, "q50": 0.50, "q95": 0.95}

# Chosen on Gate-Val only (walk-forward pinball loss and coverage), within the 15 s
# local-run budget. See docs/REVIEW_ISSUES.md, Phase D.
DEFAULT_LGBM_PARAMS: dict[str, float | int] = {
    "n_estimators": 200,
    "learning_rate": 0.045,
    "num_leaves": 31,
    "min_child_samples": 100,
}


@dataclass
class TrainingArtifacts:
    """Trained forecaster, splits, Gate-Val walk-forward predictions/metrics, and MLflow info."""

    forecaster: ConformalCarbonForecaster
    splits: dict[str, pd.DataFrame]
    gate_val_metrics: dict[str, float]
    gate_val_predictions: pd.DataFrame
    run_id: str | None = None
    model_version: str | None = None


SPLIT_ORDER: tuple[str, ...] = ("train", "calib", "gate_val", "locked_test")


def compute_split_boundaries(
    features_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig | None = None,
) -> dict[str, pd.Timestamp]:
    """Return the last origin_timestamp of Train, Calib and Gate-Val (Locked Test follows)."""
    cfg = cfg or CarbonMLOpsConfig()
    labelled = features_df.dropna(subset=["target_carbon_intensity"])
    origins = pd.DatetimeIndex(labelled["origin_timestamp"].unique()).sort_values()
    n = len(origins)

    r_train, r_calib, r_gate, _ = cfg.split_ratios
    return {
        "train": origins[int(n * r_train) - 1],
        "calib": origins[int(n * (r_train + r_calib)) - 1],
        "gate_val": origins[int(n * (r_train + r_calib + r_gate)) - 1],
    }


def assert_splits_do_not_overlap(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if any split's targets reach into the origins of the split that follows it."""
    present = [name for name in SPLIT_ORDER if name in splits and not splits[name].empty]
    for earlier, later in zip(present[:-1], present[1:], strict=True):
        last_target = splits[earlier]["target_timestamp"].max()
        first_origin = splits[later]["origin_timestamp"].min()
        if last_target >= first_origin:
            raise AssertionError(
                f"Split leakage: {earlier} targets reach {last_target}, "
                f"but {later} origins start at {first_origin}"
            )


def chronological_four_way_split(
    features_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Partition supervised horizon rows by origin_timestamp into Train/Calib/GateVal/Test.

    Rows are purged at every boundary: a row belongs to a split only if its target also
    falls inside that split. Without this, the last 24 hours of origins in one split carry
    labels that the next split uses as features and as its own labels.
    """
    cfg = cfg or CarbonMLOpsConfig()
    df = features_df.dropna(subset=["target_carbon_intensity"]).copy()
    ends = compute_split_boundaries(df, cfg=cfg)
    origin, target = df["origin_timestamp"], df["target_timestamp"]

    splits = {
        "train": df[target <= ends["train"]],
        "calib": df[(origin > ends["train"]) & (target <= ends["calib"])],
        "gate_val": df[(origin > ends["calib"]) & (target <= ends["gate_val"])],
        "locked_test": df[origin > ends["gate_val"]],
    }
    splits = {name: part.reset_index(drop=True) for name, part in splits.items()}
    assert_splits_do_not_overlap(splits)
    return splits


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Compute mean quantile pinball loss for quantile level alpha."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def _zone_key(zone: str) -> str:
    return zone.lower().replace("-", "_")


def compute_interval_metrics(
    preds_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig,
    prefix: str = "gate_val",
) -> dict[str, float]:
    """Coverage, width and pinball loss of calibrated predictions, pooled and per zone.

    Per zone it also reports the worst per-horizon coverage and the pinball loss of the
    persistence forecast (the origin intensity), the baseline the median must beat.
    """
    y = preds_df["target_carbon_intensity"].to_numpy()
    p05, p50, p95 = (preds_df[c].to_numpy() for c in ("p05", "p50", "p95"))
    covered = (y >= p05) & (y <= p95)

    metrics: dict[str, float] = {
        f"{prefix}_coverage": float(np.mean(covered)),
        f"{prefix}_mean_width": float(np.mean(p95 - p05)),
        f"{prefix}_pinball_p05": pinball_loss(y, p05, 0.05),
        f"{prefix}_pinball_p50": pinball_loss(y, p50, 0.50),
        f"{prefix}_pinball_p95": pinball_loss(y, p95, 0.95),
    }
    metrics[f"{prefix}_mean_pinball"] = float(
        np.mean([metrics[f"{prefix}_pinball_{q}"] for q in ("p05", "p50", "p95")])
    )

    persistence = preds_df["carbon_intensity_origin"].to_numpy()
    horizons = preds_df["horizon_h"].to_numpy()
    for zone in cfg.zones:
        z_mask = (preds_df["zone"] == zone).to_numpy()
        if not z_mask.any():
            continue
        key = _zone_key(zone)
        y_z, p05_z, p50_z, p95_z = y[z_mask], p05[z_mask], p50[z_mask], p95[z_mask]
        metrics[f"{prefix}_coverage_{key}"] = float(np.mean(covered[z_mask]))
        metrics[f"{prefix}_width_{key}"] = float(np.mean(p95_z - p05_z))
        per_horizon = pd.Series(covered[z_mask]).groupby(horizons[z_mask]).mean()
        metrics[f"{prefix}_min_horizon_coverage_{key}"] = float(per_horizon.min())
        metrics[f"{prefix}_mean_pinball_{key}"] = float(
            np.mean(
                [
                    pinball_loss(y_z, p05_z, 0.05),
                    pinball_loss(y_z, p50_z, 0.50),
                    pinball_loss(y_z, p95_z, 0.95),
                ]
            )
        )
        metrics[f"{prefix}_pinball_p50_{key}"] = pinball_loss(y_z, p50_z, 0.50)
        metrics[f"{prefix}_persistence_pinball_p50_{key}"] = pinball_loss(
            y_z, persistence[z_mask], 0.50
        )
    return metrics


def fit_quantile_models(
    train_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig,
    lgbm_params: dict[str, float | int],
) -> dict[str, dict[str, lgb.LGBMRegressor]]:
    """Fit one LightGBM quantile regressor per zone and quantile level on the Train split."""
    models_by_zone: dict[str, dict[str, lgb.LGBMRegressor]] = {}
    for zone in cfg.zones:
        z_train = train_df[train_df["zone"] == zone]
        x_tr = z_train[list(CAUSAL_FEATURE_COLS)]
        y_tr = z_train["target_carbon_intensity"].to_numpy()
        if cfg.target_mode == "delta":
            y_tr = y_tr - z_train["carbon_intensity_origin"].to_numpy()

        z_models: dict[str, lgb.LGBMRegressor] = {}
        for key, alpha in QUANTILE_LEVELS.items():
            reg = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                subsample_for_bin=20000,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                **lgbm_params,
            )
            reg.fit(x_tr, y_tr)
            z_models[key] = reg
        models_by_zone[zone] = z_models
    return models_by_zone


def recalibrate_forecaster(
    forecaster: ConformalCarbonForecaster,
    labelled_df: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: CarbonMLOpsConfig | None = None,
) -> ConformalCarbonForecaster:
    """Return the forecaster with offsets recomputed from the window that ends at `as_of`.

    This is the scheduled recalibration step: the base models are untouched, only the
    conformal offsets move, and only targets observed before `as_of` are used.
    """
    cfg = cfg or CarbonMLOpsConfig()
    scored = add_conformity_scores(forecaster, labelled_df)
    window = select_calibration_window(scored, as_of, cfg.recalibration_window_days)
    offsets = calibrate_offsets(window, cfg.nominal_coverage)
    return forecaster.with_offsets(offsets, calibrated_through=str(as_of))


def train_split_cqr(
    features_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig | None = None,
    n_estimators: int | None = None,
    register_mlflow: bool = False,
    tracking_uri: str | None = None,
) -> TrainingArtifacts:
    """Fit base quantile models, evaluate the recalibration schedule on Gate-Val, log to MLflow.

    Gate-Val metrics come from a walk-forward replay: offsets are recomputed every
    `recalibration_step_days` from the trailing window (which starts inside Calib) and
    applied to the following origins, exactly as in operation. The returned forecaster
    carries the offsets of the last recalibration before the Locked Test begins.
    """
    cfg = cfg or CarbonMLOpsConfig()
    lgbm_params = dict(DEFAULT_LGBM_PARAMS)
    if n_estimators is not None:
        lgbm_params["n_estimators"] = n_estimators

    splits = chronological_four_way_split(features_df, cfg=cfg)
    gate_df = splits["gate_val"]

    models_by_zone = fit_quantile_models(splits["train"], cfg, lgbm_params)
    uncalibrated = ConformalCarbonForecaster(
        models_by_zone=models_by_zone, conformal_offsets={}, target_mode=cfg.target_mode
    )

    # Only Calib origins inside the first trailing window can ever be used for calibration,
    # so older Calib rows are not scored. This changes no result, only the run time.
    gate_start = gate_df["origin_timestamp"].min()
    window_start = gate_start - pd.Timedelta(days=cfg.recalibration_window_days)
    recent_calib = splits["calib"][splits["calib"]["origin_timestamp"] >= window_start]
    history = pd.concat([recent_calib, gate_df], ignore_index=True)
    scored = add_conformity_scores(uncalibrated, history)
    gate_preds = walk_forward_calibrated_predictions(
        scored,
        eval_start=gate_df["origin_timestamp"].min(),
        eval_end=gate_df["origin_timestamp"].max(),
        cfg=cfg,
    )
    metrics = compute_interval_metrics(gate_preds, cfg, prefix="gate_val")
    logger.info("Gate-Val walk-forward CQR metrics: %s", metrics)

    calibrated_through = gate_df["target_timestamp"].max() + pd.Timedelta(hours=1)
    final_window = select_calibration_window(
        scored, as_of=calibrated_through, window_days=cfg.recalibration_window_days
    )
    forecaster = uncalibrated.with_offsets(
        calibrate_offsets(final_window, cfg.nominal_coverage),
        calibrated_through=str(calibrated_through),
    )

    run_id: str | None = None
    model_version: str | None = None

    if register_mlflow:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if is_databricks_runtime():
            mlflow.set_registry_uri("databricks-uc")

        model_name = f"{cfg.catalog}.{cfg.schema}.conformal_carbon_forecaster"
        if not is_databricks_runtime() and (
            not tracking_uri or tracking_uri.startswith(("file:", "sqlite:"))
        ):
            model_name = "conformal_carbon_forecaster"

        sample_in = (
            gate_df[["zone", *CAUSAL_FEATURE_COLS]]
            .head(10)
            .astype(
                {
                    "horizon_h": "float64",
                    "target_dow": "float64",
                    "target_month": "float64",
                    "is_weekend": "float64",
                }
            )
        )
        sample_out = forecaster.predict(context=None, model_input=sample_in)
        signature = infer_signature(sample_in, sample_out)

        with mlflow.start_run(run_name="split_cqr_challenger") as run:
            run_id = run.info.run_id
            mlflow.log_params(
                {
                    **lgbm_params,
                    "nominal_coverage": cfg.nominal_coverage,
                    "cqr_mode": "asymmetric_two_sided_per_horizon",
                    "target_mode": cfg.target_mode,
                    "recalibration_window_days": cfg.recalibration_window_days,
                    "recalibration_step_days": cfg.recalibration_step_days,
                    "calibrated_through": str(calibrated_through),
                    "zones": ",".join(cfg.zones),
                }
            )
            mlflow.log_metrics(metrics)
            logged = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=forecaster,
                signature=signature,
                input_example=sample_in,
                pip_requirements=forecaster.pip_requirements,
                registered_model_name=model_name,
            )
            model_version = str(logged.registered_model_version)
            client = MlflowClient()
            client.set_registered_model_alias(
                name=model_name,
                alias="challenger",
                version=model_version,
            )
            logger.info(
                "Registered %s v%s as @challenger (run_id=%s)", model_name, model_version, run_id
            )

    return TrainingArtifacts(
        forecaster=forecaster,
        splits=splits,
        gate_val_metrics=metrics,
        gate_val_predictions=gate_preds,
        run_id=run_id,
        model_version=model_version,
    )
