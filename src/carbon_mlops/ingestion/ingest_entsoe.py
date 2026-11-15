"""Bronze ENTSO-E snapshot ingestion with SHA-256 provenance verification (Gate G1)."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from carbon_mlops.config import CarbonMLOpsConfig
from carbon_mlops.io.store import write_table

logger = logging.getLogger(__name__)

REQUIRED_BRONZE_COLUMNS: tuple[str, ...] = (
    "zone",
    "timestamp_utc",
    "total_mwh",
    "priced_mwh",
    "emissions_kg",
    "is_partial",
    "carbon_intensity_g_kwh",
    "gen_total_mwh",
    "wind_share",
    "solar_share",
    "renewable_share",
    "nuclear_share",
    "fossil_share",
    "upstream_payload_sha256",
)


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 checksum of a file on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bronze_schema(df: pd.DataFrame, allowed_zones: tuple[str, ...]) -> None:
    """Verify schema contract, zone codes, and physically plausible carbon intensity bounds."""
    missing = [col for col in REQUIRED_BRONZE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required Bronze columns: {missing}")

    if df.empty:
        raise ValueError("Bronze snapshot DataFrame is empty")

    observed_zones = set(df["zone"].unique())
    invalid_zones = observed_zones - set(allowed_zones)
    if invalid_zones:
        raise ValueError(f"Unexpected zone identifiers in Bronze snapshot: {invalid_zones}")

    ci = df["carbon_intensity_g_kwh"].dropna()
    if (ci < 0).any() or (ci > 2000).any():
        raise ValueError("carbon_intensity_g_kwh out of physical bounds [0, 2000] gCO2eq/kWh")

    if df["upstream_payload_sha256"].isna().any() or (df["upstream_payload_sha256"] == "").any():
        raise ValueError("Missing upstream_payload_sha256 provenance hashes")


def ingest_bronze_snapshot(
    cfg: CarbonMLOpsConfig | None = None,
    snapshot_path: Path | None = None,
    expected_sha256: str | None = None,
    base_dir: Path | None = None,
) -> pd.DataFrame:
    """Load Bronze Parquet snapshot, enforce Gate G1 provenance, and write bronze table."""
    cfg = cfg or CarbonMLOpsConfig()
    src_path = snapshot_path or cfg.bronze_snapshot_path
    target_base = base_dir or cfg.local_data_dir

    if not src_path.exists():
        raise FileNotFoundError(f"Bronze snapshot not found: {src_path}")

    actual_sha256 = compute_file_sha256(src_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Gate G1 SHA-256 mismatch for {src_path}: expected={expected_sha256}, "
            f"actual={actual_sha256}"
        )

    df = pd.read_parquet(src_path)
    validate_bronze_schema(df, cfg.zones)

    ingested_ts = datetime.now(timezone.utc).isoformat()
    df = df.copy()
    df["source_file_sha256"] = actual_sha256
    df["ingested_at_utc"] = ingested_ts

    table_name = f"{cfg.schema}.bronze_entsoe_hourly"
    write_table(df, table_name=table_name, mode="overwrite", base_dir=target_base)
    logger.info(
        "Ingested %d Bronze rows (sha256=%s) into %s",
        len(df),
        actual_sha256[:12],
        table_name,
    )
    return df
