"""Chronological 4-way split training, Asymmetric Split CQR, and MLflow Registry logging."""

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
from carbon_mlops.models.pyfunc_wrapper import ConformalCarbonForecaster

logger = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    """Container holding trained CQR forecaster, splits, Gate-Val metrics, and MLflow run info."""

    forecaster: ConformalCarbonForecaster
    splits: dict[str, pd.DataFrame]
    gate_val_metrics: dict[str, float]
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


def _compute_gate_val_metrics(
    forecaster: ConformalCarbonForecaster,
    gate_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig,
) -> dict[str, float]:
    """Evaluate Gate-Val coverage (overall & per-zone) and quantile pinball losses."""
    gate_preds = forecaster.predict(context=None, model_input=gate_df)
    y_gate = gate_df["target_carbon_intensity"].to_numpy()
    covered = (y_gate >= gate_preds["p05"].to_numpy()) & (y_gate <= gate_preds["p95"].to_numpy())

    metrics: dict[str, float] = {
        "gate_val_coverage": float(np.mean(covered)),
        "gate_val_mean_width": float(gate_preds["interval_width"].mean()),
        "gate_val_pinball_p05": pinball_loss(y_gate, gate_preds["p05"].to_numpy(), 0.05),
        "gate_val_pinball_p50": pinball_loss(y_gate, gate_preds["p50"].to_numpy(), 0.50),
        "gate_val_pinball_p95": pinball_loss(y_gate, gate_preds["p95"].to_numpy(), 0.95),
    }
    metrics["gate_val_mean_pinball"] = float(
        np.mean(
            [
                metrics["gate_val_pinball_p05"],
                metrics["gate_val_pinball_p50"],
                metrics["gate_val_pinball_p95"],
            ]
        )
    )

    for zone in cfg.zones:
        z_mask = (gate_df["zone"] == zone).to_numpy()
        if z_mask.any():
            z_key = zone.lower().replace("-", "_")
            metrics[f"gate_val_coverage_{z_key}"] = float(np.mean(covered[z_mask]))
            metrics[f"gate_val_width_{z_key}"] = float(
                gate_preds.loc[z_mask, "interval_width"].mean()
            )
    return metrics


def train_split_cqr(
    features_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig | None = None,
    n_estimators: int = 120,
    register_mlflow: bool = False,
    tracking_uri: str | None = None,
) -> TrainingArtifacts:
    """Fit per-zone LightGBM quantile models, calibrate two-sided Split CQR, and log to MLflow."""
    cfg = cfg or CarbonMLOpsConfig()
    splits = chronological_four_way_split(features_df, cfg=cfg)
    train_df = splits["train"]
    calib_df = splits["calib"]
    gate_df = splits["gate_val"]

    models_by_zone: dict[str, dict[str, lgb.LGBMRegressor]] = {}
    quantile_keys = {"q05": 0.05, "q50": 0.50, "q95": 0.95}

    for zone in cfg.zones:
        z_train = train_df[train_df["zone"] == zone]
        x_tr = z_train[list(CAUSAL_FEATURE_COLS)]
        y_tr = z_train["target_carbon_intensity"].to_numpy()

        z_models: dict[str, lgb.LGBMRegressor] = {}
        for key, alpha in quantile_keys.items():
            reg = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=n_estimators,
                learning_rate=0.06,
                num_leaves=31,
                subsample_for_bin=20000,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            reg.fit(x_tr, y_tr)
            z_models[key] = reg
        models_by_zone[zone] = z_models

    conformal_offsets: dict[tuple[str, str], tuple[float, float]] = {}
    tail_alpha = (1.0 - cfg.nominal_coverage) / 2.0

    for zone in cfg.zones:
        z_cal = calib_df[calib_df["zone"] == zone]
        x_cal = z_cal[list(CAUSAL_FEATURE_COLS)]
        q05_cal = models_by_zone[zone]["q05"].predict(x_cal)
        q95_cal = models_by_zone[zone]["q95"].predict(x_cal)
        y_cal = z_cal["target_carbon_intensity"].to_numpy()

        low_scores = q05_cal - y_cal
        high_scores = y_cal - q95_cal

        for bucket in cfg.horizon_buckets:
            mask = (z_cal["horizon_bucket"] == bucket).to_numpy()
            lo_b = low_scores[mask]
            hi_b = high_scores[mask]
            n_b = len(lo_b)
            if n_b == 0:
                conformal_offsets[(zone, bucket)] = (0.0, 0.0)
                continue
            q_level = min(1.0, np.ceil((n_b + 1) * (1.0 - tail_alpha)) / n_b)
            q_lo = float(np.quantile(lo_b, q_level, method="higher"))
            q_hi = float(np.quantile(hi_b, q_level, method="higher"))
            conformal_offsets[(zone, bucket)] = (q_lo, q_hi)

    forecaster = ConformalCarbonForecaster(
        models_by_zone=models_by_zone,
        conformal_offsets=conformal_offsets,
    )
    metrics = _compute_gate_val_metrics(forecaster, gate_df, cfg)
    logger.info("Gate-Val Asymmetric CQR metrics: %s", metrics)

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
            gate_df[["zone", "horizon_bucket", *CAUSAL_FEATURE_COLS]]
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
                    "n_estimators": n_estimators,
                    "nominal_coverage": cfg.nominal_coverage,
                    "cqr_mode": "asymmetric_two_sided",
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
        run_id=run_id,
        model_version=model_version,
    )
