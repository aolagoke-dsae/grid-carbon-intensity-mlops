"""Automated Gate-Val promotion gate (@challenger -> @champion) protecting Locked Test (Gate G3).

The gate does not trust metrics logged by a training run. It loads the @challenger (and the
@champion, if one exists) from the registry and re-scores both itself on the same Gate-Val
window, by replaying the recalibration schedule with each model's base models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.io.store import is_databricks_runtime
from carbon_mlops.models.pyfunc_wrapper import ConformalCarbonForecaster
from carbon_mlops.models.train import (
    chronological_four_way_split,
    evaluate_on_gate_val,
    pinball_loss,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateEvaluation:
    """What the gate measured for one model on Gate-Val."""

    coverage_by_zone: dict[str, float]
    mean_pinball_by_zone: dict[str, float]
    # (zone, horizon_bucket) cells where the median is not better than persistence
    cells_losing_to_persistence: tuple[str, ...]
    n_rows: int


@dataclass(frozen=True)
class GateDecision:
    """Outcome of judging @challenger (and comparing it with @champion) on Gate-Val."""

    promoted: bool
    reason: str
    challenger_version: str
    failed_checks: tuple[str, ...]
    challenger: GateEvaluation
    champion: GateEvaluation | None = None
    champion_version: str | None = None
    pinball_ratio_vs_champion: float | None = None


def summarize_gate_predictions(preds_df: pd.DataFrame, cfg: CarbonMLOpsConfig) -> GateEvaluation:
    """Reduce walk-forward Gate-Val predictions to the quantities the gate decides on."""
    y = preds_df["target_carbon_intensity"].to_numpy()
    p05, p50, p95 = (preds_df[c].to_numpy() for c in ("p05", "p50", "p95"))
    persistence = preds_df["carbon_intensity_origin"].to_numpy()
    covered = (y >= p05) & (y <= p95)
    zones = preds_df["zone"].to_numpy()
    buckets = preds_df["horizon_bucket"].to_numpy()

    coverage: dict[str, float] = {}
    mean_pinball: dict[str, float] = {}
    losing: list[str] = []
    for zone in cfg.zones:
        z_mask = zones == zone
        if not z_mask.any():
            raise ValueError(f"Gate-Val has no rows for zone {zone}")
        coverage[zone] = float(np.mean(covered[z_mask]))
        mean_pinball[zone] = float(
            np.mean(
                [
                    pinball_loss(y[z_mask], p05[z_mask], 0.05),
                    pinball_loss(y[z_mask], p50[z_mask], 0.50),
                    pinball_loss(y[z_mask], p95[z_mask], 0.95),
                ]
            )
        )
        for bucket in cfg.horizon_buckets:
            cell = z_mask & (buckets == bucket)
            if not cell.any():
                continue
            model_loss = pinball_loss(y[cell], p50[cell], 0.50)
            baseline_loss = pinball_loss(y[cell], persistence[cell], 0.50)
            if model_loss >= baseline_loss:
                losing.append(f"{zone}/{bucket}")
    return GateEvaluation(
        coverage_by_zone=coverage,
        mean_pinball_by_zone=mean_pinball,
        cells_losing_to_persistence=tuple(losing),
        n_rows=len(preds_df),
    )


def pinball_ratio(challenger: GateEvaluation, champion: GateEvaluation) -> float:
    """Mean over zones of challenger / champion pinball loss (below 1 means better).

    A ratio per zone keeps the comparison scale-free: DE-LU losses are about twenty times
    larger than FR losses, so a pooled mean would be decided by DE-LU alone.
    """
    ratios = [
        challenger.mean_pinball_by_zone[zone] / champion.mean_pinball_by_zone[zone]
        for zone in challenger.mean_pinball_by_zone
    ]
    return float(np.mean(ratios))


def decide_promotion(
    challenger: GateEvaluation,
    champion: GateEvaluation | None,
    cfg: CarbonMLOpsConfig,
) -> tuple[str, ...]:
    """Return the checks the challenger fails; an empty tuple means it may be promoted.

    1. Coverage of every zone, not the pooled figure, must lie inside cfg.coverage_bounds.
    2. The median must beat the persistence forecast in every zone and horizon bucket.
    3. Against an existing champion, scored on the same window, the scale-free pinball
       ratio must not exceed 1 (ties promote).
    """
    failed: list[str] = []
    low, high = cfg.coverage_bounds
    for zone, cov in challenger.coverage_by_zone.items():
        if not (low <= cov <= high):
            failed.append(f"coverage[{zone}]={cov:.4f} outside [{low:.2f}, {high:.2f}]")
    for cell in challenger.cells_losing_to_persistence:
        failed.append(f"median does not beat persistence in {cell}")
    if champion is not None:
        ratio = pinball_ratio(challenger, champion)
        if ratio > 1.0:
            failed.append(f"pinball ratio vs champion={ratio:.4f} exceeds 1.0")
    return tuple(failed)


def _load_forecaster(model_name: str, alias: str) -> ConformalCarbonForecaster:
    """Load the registered model behind an alias and return the wrapped Python model."""
    loaded = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
    forecaster = loaded.unwrap_python_model()
    if not isinstance(forecaster, ConformalCarbonForecaster):
        raise TypeError(f"{model_name}@{alias} is not a ConformalCarbonForecaster")
    return forecaster


def _warn_if_logged_metrics_differ(
    client: MlflowClient, run_id: str, evaluation: GateEvaluation
) -> None:
    """Log a warning when the training run's own coverage disagrees with the gate's."""
    logged = client.get_run(run_id).data.metrics
    for zone, cov in evaluation.coverage_by_zone.items():
        key = f"gate_val_coverage_{zone.lower().replace('-', '_')}"
        if key in logged and abs(logged[key] - cov) > 1e-6:
            logger.warning(
                "Run %s logged %s=%.4f but the gate measured %.4f; the gate's value is used",
                run_id,
                key,
                logged[key],
                cov,
            )


def evaluate_and_promote_champion(
    features_df: pd.DataFrame,
    cfg: CarbonMLOpsConfig | None = None,
    model_name: str | None = None,
    client: MlflowClient | None = None,
) -> GateDecision:
    """Re-score @challenger (and @champion) on Gate-Val and promote iff Gate G3 passes."""
    cfg = cfg or CarbonMLOpsConfig()
    client = client or MlflowClient()
    if is_databricks_runtime():
        mlflow.set_registry_uri("databricks-uc")

    if model_name is None:
        model_name = (
            f"{cfg.catalog}.{cfg.schema}.conformal_carbon_forecaster"
            if is_databricks_runtime()
            else "conformal_carbon_forecaster"
        )

    splits = chronological_four_way_split(features_df, cfg=cfg)

    challenger_mv = client.get_model_version_by_alias(name=model_name, alias="challenger")
    challenger_version = str(challenger_mv.version)
    _, challenger_preds = evaluate_on_gate_val(
        _load_forecaster(model_name, "challenger"), splits, cfg
    )
    challenger_eval = summarize_gate_predictions(challenger_preds, cfg)
    _warn_if_logged_metrics_differ(client, challenger_mv.run_id, challenger_eval)

    champion_eval: GateEvaluation | None = None
    champion_version: str | None = None
    try:
        champion_mv = client.get_model_version_by_alias(name=model_name, alias="champion")
        champion_version = str(champion_mv.version)
    except MlflowException:
        champion_mv = None
    if champion_mv is not None and champion_version != challenger_version:
        _, champion_preds = evaluate_on_gate_val(
            _load_forecaster(model_name, "champion"), splits, cfg
        )
        champion_eval = summarize_gate_predictions(champion_preds, cfg)

    failed = decide_promotion(challenger_eval, champion_eval, cfg)
    ratio = pinball_ratio(challenger_eval, champion_eval) if champion_eval is not None else None
    promoted = not failed

    if promoted:
        client.set_registered_model_alias(
            name=model_name, alias="champion", version=challenger_version
        )
        reason = (
            f"Promoted @challenger v{challenger_version} to @champion "
            f"(coverage={challenger_eval.coverage_by_zone}, "
            f"pinball_ratio_vs_champion={ratio}, previous_champion=v{champion_version})"
        )
        logger.info(reason)
    else:
        reason = f"Rejected @challenger v{challenger_version}: " + "; ".join(failed)
        logger.warning(reason)

    client.set_model_version_tag(
        name=model_name,
        version=challenger_version,
        key="gate_decision",
        value="promoted" if promoted else "rejected",
    )
    client.set_model_version_tag(
        name=model_name, version=challenger_version, key="gate_reason", value=reason[:4000]
    )

    return GateDecision(
        promoted=promoted,
        reason=reason,
        challenger_version=challenger_version,
        failed_checks=failed,
        challenger=challenger_eval,
        champion=champion_eval,
        champion_version=champion_version,
        pinball_ratio_vs_champion=ratio,
    )
