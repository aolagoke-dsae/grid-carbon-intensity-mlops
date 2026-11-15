"""Automated Gate-Val promotion gate (@challenger -> @champion) protecting Locked Test (Gate G3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.io.store import is_databricks_runtime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateDecision:
    """Outcome of comparing @challenger against @champion on Gate-Val metrics."""

    promoted: bool
    reason: str
    challenger_version: str
    challenger_coverage: float
    challenger_pinball: float
    champion_pinball: float | None = None


def evaluate_and_promote_champion(
    cfg: CarbonMLOpsConfig | None = None,
    model_name: str | None = None,
    client: MlflowClient | None = None,
) -> GateDecision:
    """Promote @challenger to @champion iff Gate-Val coverage & pinball pass Gate G3."""
    cfg = cfg or CarbonMLOpsConfig()
    client = client or MlflowClient()

    if model_name is None:
        model_name = (
            f"{cfg.catalog}.{cfg.schema}.conformal_carbon_forecaster"
            if is_databricks_runtime()
            else "conformal_carbon_forecaster"
        )

    challenger = client.get_model_version_by_alias(name=model_name, alias="challenger")
    challenger_run = client.get_run(challenger.run_id)
    c_metrics = challenger_run.data.metrics

    cov = float(c_metrics["gate_val_coverage"])
    pinball = float(c_metrics["gate_val_mean_pinball"])
    low_cov, high_cov = cfg.coverage_bounds

    if not (low_cov <= cov <= high_cov):
        reason = (
            f"Rejected @challenger v{challenger.version}: gate_val_coverage={cov:.4f} "
            f"outside [{low_cov:.2f}, {high_cov:.2f}]"
        )
        logger.warning(reason)
        return GateDecision(
            promoted=False,
            reason=reason,
            challenger_version=str(challenger.version),
            challenger_coverage=cov,
            challenger_pinball=pinball,
        )

    champion_pinball: float | None = None
    try:
        champion = client.get_model_version_by_alias(name=model_name, alias="champion")
        champ_run = client.get_run(champion.run_id)
        champion_pinball = float(champ_run.data.metrics["gate_val_mean_pinball"])
    except MlflowException:
        champion_pinball = None

    if champion_pinball is not None and pinball > champion_pinball:
        reason = (
            f"Rejected @challenger v{challenger.version}: gate_val_mean_pinball={pinball:.4f} "
            f"worse than @champion ({champion_pinball:.4f})"
        )
        logger.warning(reason)
        return GateDecision(
            promoted=False,
            reason=reason,
            challenger_version=str(challenger.version),
            challenger_coverage=cov,
            challenger_pinball=pinball,
            champion_pinball=champion_pinball,
        )

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=str(challenger.version),
    )
    reason = (
        f"Promoted @challenger v{challenger.version} to @champion "
        f"(coverage={cov:.4f}, pinball={pinball:.4f}, prev_champion={champion_pinball})"
    )
    logger.info(reason)
    return GateDecision(
        promoted=True,
        reason=reason,
        challenger_version=str(challenger.version),
        challenger_coverage=cov,
        challenger_pinball=pinball,
        champion_pinball=champion_pinball,
    )
