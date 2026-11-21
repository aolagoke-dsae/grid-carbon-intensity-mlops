"""Rebuild the hourly Bronze snapshot from the upstream `grid-carbon-intensity` marts (Gate G1).

Reads `fact_generation.parquet` and `fact_intensity.parquet` from the upstream
`exports/` folder and writes `data/bronze_snapshot/entsoe_generation_sample.parquet`.

Aggregation rules (one row per zone and UTC hour):

* Only `direction == "generation"` rows are used; consumption legs (pumping) are dropped.
* `total_mwh`      = all generation, every fuel, including pumped storage (B10).
* `gen_total_mwh`  = generation excluding hydro pumped storage (B10); denominator of the shares.
* `priced_mwh`     = generation of fuels with an emission factor; denominator of the intensity.
* `carbon_intensity_g_kwh` = sum(emissions_kg) / sum(priced_mwh), i.e. energy-weighted over steps.
* `hours_covered`  = summed step length of the upstream steps in the hour (1.0 when complete).
* `n_partial_steps` / `is_partial` = upstream steps at which a published series was missing.
  Upstream withholds the intensity of such steps; this snapshot keeps the recomputed value
  but flags it so the feature pipeline can mask it.
* `upstream_payload_sha256` = the upstream `source_sha256`, the key of the raw retrieval in
  `raw/manifest.csv` of the upstream repository.

Usage:
    python scripts/export_bronze_snapshot.py --upstream-repo ../grid-carbon-intensity
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("export_bronze_snapshot")

ZONE_BY_EIC: dict[str, str] = {
    "10Y1001A1001A82H": "DE-LU",
    "10YFR-RTE------C": "FR",
}

FUEL_GROUPS: dict[str, tuple[str, ...]] = {
    "wind_mwh": ("B18", "B19"),
    "solar_mwh": ("B16",),
    "renewable_mwh": ("B01", "B09", "B11", "B12", "B15", "B16", "B18", "B19"),
    "nuclear_mwh": ("B14",),
    "fossil_mwh": ("B02", "B03", "B04", "B05", "B06"),
}
PUMPED_STORAGE_PSR = "B10"

OUTPUT_COLUMNS: tuple[str, ...] = (
    "zone",
    "timestamp_utc",
    "total_mwh",
    "priced_mwh",
    "emissions_kg",
    "is_partial",
    "carbon_intensity_g_kwh",
    "gen_total_mwh",
    "wind_mwh",
    "solar_mwh",
    "renewable_mwh",
    "nuclear_mwh",
    "fossil_mwh",
    "upstream_payload_sha256",
    "wind_share",
    "solar_share",
    "renewable_share",
    "nuclear_share",
    "fossil_share",
    "priced_share",
    "hours_covered",
    "n_partial_steps",
)


def _upstream_commit(upstream_repo: Path) -> str:
    """Return the HEAD commit of the upstream repository, or 'unknown' outside git."""
    try:
        out = subprocess.run(
            ["git", "-C", str(upstream_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _join_hashes(hashes: pd.Series) -> str:
    """Return the single upstream hash of an hour, or a digest when several payloads meet."""
    unique = sorted(set(hashes))
    if len(unique) == 1:
        return unique[0]
    return hashlib.sha256("|".join(unique).encode("utf-8")).hexdigest()


def build_snapshot(
    fact_generation: pd.DataFrame,
    fact_intensity: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate upstream step-level facts into one row per (zone, UTC hour) in [start, end)."""
    gen = fact_generation[
        (fact_generation["direction"] == "generation")
        & fact_generation["zone_eic"].isin(list(ZONE_BY_EIC))
        & (fact_generation["ts_utc"] >= start)
        & (fact_generation["ts_utc"] < end)
    ].copy()
    gen["zone"] = gen["zone_eic"].map(ZONE_BY_EIC)
    gen["timestamp_utc"] = gen["ts_utc"].dt.floor("h")
    keys = ["zone", "timestamp_utc"]

    by_fuel = gen.pivot_table(
        index=keys, columns="psr_type", values="mwh", aggfunc="sum", fill_value=0.0
    )
    out = pd.DataFrame(index=by_fuel.index)
    out["total_mwh"] = by_fuel.sum(axis=1)
    out["priced_mwh"] = (
        gen[gen["is_priced_generation"]].groupby(keys)["mwh"].sum().reindex(out.index).fillna(0.0)
    )
    out["emissions_kg"] = gen.groupby(keys)["emissions_kg"].sum().reindex(out.index).fillna(0.0)
    out["gen_total_mwh"] = out["total_mwh"] - by_fuel.get(PUMPED_STORAGE_PSR, 0.0)
    for col, psr_types in FUEL_GROUPS.items():
        present = [p for p in psr_types if p in by_fuel.columns]
        out[col] = by_fuel[present].sum(axis=1)
    out["upstream_payload_sha256"] = gen.groupby(keys)["source_sha256"].agg(_join_hashes)

    steps = fact_intensity[
        fact_intensity["zone_eic"].isin(list(ZONE_BY_EIC))
        & (fact_intensity["ts_utc"] >= start)
        & (fact_intensity["ts_utc"] < end)
    ].copy()
    steps["zone"] = steps["zone_eic"].map(ZONE_BY_EIC)
    steps["timestamp_utc"] = steps["ts_utc"].dt.floor("h")
    step_agg = steps.groupby(keys).agg(
        hours_covered=("step_hours", "sum"),
        n_partial_steps=("is_partial", "sum"),
    )
    out = out.join(step_agg, how="left")
    out["hours_covered"] = out["hours_covered"].round(6)
    out["n_partial_steps"] = out["n_partial_steps"].fillna(0).astype("int64")
    out["is_partial"] = out["n_partial_steps"] > 0

    out["carbon_intensity_g_kwh"] = np.where(
        out["priced_mwh"] > 0, out["emissions_kg"] / out["priced_mwh"], np.nan
    )
    for col in FUEL_GROUPS:
        out[col.replace("_mwh", "_share")] = out[col] / out["gen_total_mwh"]
    out["priced_share"] = out["priced_mwh"] / out["total_mwh"]

    out = out.reset_index().sort_values(keys).reset_index(drop=True)
    return out[list(OUTPUT_COLUMNS)]


def compare_with_existing(new_df: pd.DataFrame, existing_path: Path) -> None:
    """Log whether the numeric columns shared with an existing snapshot are unchanged."""
    old = pd.read_parquet(existing_path)
    keys = ["zone", "timestamp_utc"]
    shared = [
        c
        for c in old.columns
        if c in new_df.columns and c not in keys and pd.api.types.is_numeric_dtype(old[c])
    ]
    merged = old[keys + shared].merge(new_df[keys + shared], on=keys, suffixes=("_old", "_new"))
    if len(merged) != len(old) or len(merged) != len(new_df):
        logger.warning(
            "Row sets differ: existing=%d new=%d matched=%d", len(old), len(new_df), len(merged)
        )
    worst = 0.0
    for col in shared:
        a = merged[f"{col}_old"].astype(float)
        b = merged[f"{col}_new"].astype(float)
        diff = float((a - b).abs().max())
        worst = max(worst, diff)
        if not np.allclose(a, b, rtol=1e-12, atol=1e-9, equal_nan=True):
            logger.warning(
                "Column %s differs from existing snapshot (max abs diff %.6g)", col, diff
            )
    logger.info(
        "Compared %d shared numeric columns with %s: max abs diff %.3g",
        len(shared),
        existing_path,
        worst,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream-repo", type=Path, default=Path("../grid-carbon-intensity"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/bronze_snapshot/entsoe_generation_sample.parquet")
    )
    parser.add_argument("--start", default="2024-01-01", help="Inclusive UTC start")
    parser.add_argument("--end", default="2026-01-01", help="Exclusive UTC end")
    parser.add_argument(
        "--dry-run", action="store_true", help="Build and compare, but do not write the file"
    )
    args = parser.parse_args()

    exports = args.upstream_repo / "exports"
    fact_generation = pd.read_parquet(exports / "fact_generation.parquet")
    fact_intensity = pd.read_parquet(exports / "fact_intensity.parquet")
    snapshot = build_snapshot(
        fact_generation, fact_intensity, pd.Timestamp(args.start), pd.Timestamp(args.end)
    )

    if args.output.exists():
        compare_with_existing(snapshot, args.output)

    logger.info(
        "Built %d rows | %s | %s -> %s | partial hours=%d | incomplete hours=%d | upstream=%s",
        len(snapshot),
        snapshot.groupby("zone").size().to_dict(),
        snapshot["timestamp_utc"].min(),
        snapshot["timestamp_utc"].max(),
        int(snapshot["is_partial"].sum()),
        int((snapshot["hours_covered"] < 1.0).sum()),
        _upstream_commit(args.upstream_repo),
    )
    if args.dry_run:
        logger.info("Dry run: nothing written")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(args.output, index=False)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    logger.info("Wrote %s (sha256=%s)", args.output, digest)
    logger.info("Update bronze_snapshot_sha256 in src/carbon_mlops/config.py and the README")
    logger.info("Then record the new file for DVC: dvc commit -f bronze_snapshot")


if __name__ == "__main__":
    main()
