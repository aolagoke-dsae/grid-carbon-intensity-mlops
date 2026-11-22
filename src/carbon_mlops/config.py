"""Centralized configuration for the 24-hour probabilistic grid carbon intensity pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def project_root() -> Path | None:
    """The repository checkout this package runs from, or None when installed as a wheel."""
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else None


def repo_path(relative: str) -> Path:
    """A path inside the repository that is valid from any working directory.

    Run from the repository root it is the plain relative path, so nothing written to logs or
    tables changes. From another folder it points back to the checkout. An installed wheel has
    no checkout (a Databricks job, for one), and the path stays relative to the working
    directory.
    """
    root = project_root()
    if root is None:
        return Path(relative)
    try:
        return Path(os.path.relpath(root / relative, Path.cwd()))
    except ValueError:  # Windows: the working directory is on another drive
        return root / relative


@dataclass(frozen=True)
class CarbonMLOpsConfig:
    """Configuration governing zones, horizons, 4-way splits, CQR gates, and I/O."""

    zones: tuple[str, ...] = ("DE-LU", "FR")
    horizons: tuple[int, ...] = tuple(range(1, 25))
    horizon_buckets: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "short_1_6h": (1, 6),
            "mid_7_12h": (7, 12),
            "day_ahead_13_24h": (13, 24),
        }
    )
    quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95)
    nominal_coverage: float = 0.90
    coverage_bounds: tuple[float, float] = (0.87, 0.96)

    # Base-model target: "delta" predicts the change from the origin intensity, so persistence
    # is the zero prediction; "level" predicts the intensity itself.
    target_mode: str = "delta"

    # Scheduled recalibration of the conformal offsets. Every `recalibration_step_days` the
    # per-(zone, horizon_h) offsets are recomputed from the trailing
    # `recalibration_window_days` of realized errors. Window chosen on Gate-Val only
    # (candidates 30/45/60/90 days; rule: smallest worst-zone gap from nominal coverage).
    recalibration_window_days: int = 30
    recalibration_step_days: int = 7

    # 4-way chronological split ratios (Train / Calib / Gate-Val / Locked Test)
    split_ratios: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15)

    # Unity Catalog & local Parquet paths
    catalog: str = "workspace"
    schema: str = "carbon_mlops"
    local_data_dir: Path = field(default_factory=lambda: repo_path("data"))
    bronze_snapshot_path: Path = field(
        default_factory=lambda: repo_path(
            "data/bronze_snapshot/entsoe_generation_sample.parquet"
        )
    )
    # Pinned SHA-256 of the committed snapshot (Gate G1). Regenerate the file with
    # scripts/export_bronze_snapshot.py and update this value and the snapshot README together.
    bronze_snapshot_sha256: str = "c3dc7796be7589eb72c5363277057b164493f909f81500f982d4f3160eea34a9"
    dashboard_data_dir: Path = field(default_factory=lambda: repo_path("dashboards/data"))
    images_dir: Path = field(default_factory=lambda: repo_path("docs/images"))

    # Serverless-safe pip requirements for MLflow PyFunc UDF
    udf_pip_requirements: tuple[str, ...] = (
        "lightgbm>=4.3.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scipy>=1.13.0",
    )

    # Grounded QA Agent LLM backend settings ("ollama" or "databricks")
    llm_backend: str = field(default_factory=lambda: os.getenv("CARBON_LLM_BACKEND", "ollama"))
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3:8b"

    def __post_init__(self) -> None:
        total = round(sum(self.split_ratios), 6)
        if total != 1.0:
            raise ValueError(f"split_ratios must sum to 1.0, got {total}")

    def bucket_for_horizon(self, horizon_h: int) -> str:
        """Return the horizon bucket name for a given lead time in hours."""
        for name, (low, high) in self.horizon_buckets.items():
            if low <= horizon_h <= high:
                return name
        raise ValueError(f"Unsupported horizon_h={horizon_h}; expected 1..24")
