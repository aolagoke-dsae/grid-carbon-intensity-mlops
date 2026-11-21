# Bronze Snapshot Provenance (Gate G1)

- **File:** `entsoe_generation_sample.parquet`
- **SHA-256 Checksum:** `c3dc7796be7589eb72c5363277057b164493f909f81500f982d4f3160eea34a9`
  (pinned in `src/carbon_mlops/config.py` as `bronze_snapshot_sha256`; ingestion fails on a mismatch)
- **DVC record:** stage `bronze_snapshot` in the root `dvc.yaml`; `dvc.lock` holds the file's MD5
  (`efcd01a5a5e32d7f7b9a6e034c2b0073`) and size (4,573,033 bytes). The file stays in Git
  (`cache: false`), and `dvc status` reports whether it still matches. After a deliberate rebuild run
  `dvc commit -f bronze_snapshot` and update the SHA-256 above and in the configuration
- **Source:** ENTSO-E Transparency Platform; data: 50Hertz, Amprion, TenneT, TransnetBW (DE-LU) and
  RTE (FR). Actual generation per production type (A75), used on the owners' terms, which require the
  source to be named (see the root `README.md`). Intensity values are the author's computations
- **Upstream Repository:** `grid-carbon-intensity` (`commit: 7caaee395e3303c5eea26c714645001b7dda01cd`)
- **Built by:** `python scripts/export_bronze_snapshot.py --upstream-repo ../grid-carbon-intensity`
  from the upstream marts `exports/fact_generation.parquet` and `exports/fact_intensity.parquet`
- **Coverage Window:** `2024-01-01 00:00 UTC` to `2025-12-31 22:00 UTC`, one row per zone and UTC hour,
  `35,082` rows (`DE-LU` 17,543; `FR` 17,539). The upstream yearly pull ends at local midnight
  (`22:45 UTC`), so the hour `23:00 UTC` of 2025-12-31 is not part of it.

## How the hourly rows are aggregated

Upstream publishes 15-minute steps for `DE-LU` and a mix of 15-minute and hourly steps for `FR`.
Each snapshot row sums the steps that start within the UTC hour.

- Only `direction == "generation"` rows are used. Consumption legs (pumping) are dropped.
- Intensity is energy-weighted across the steps: `sum(emissions_kg) / sum(priced_mwh)`,
  not the mean of the step intensities.
- Emission factors are the upstream **median** life-cycle factors (IPCC AR5 and ZHAW).

### Three different denominators

| Column | What it sums | Used as denominator of |
| :--- | :--- | :--- |
| `total_mwh` | All generation, every production type, including pumped storage (`B10`) | `priced_share` |
| `gen_total_mwh` | Generation excluding pumped storage (`B10`) | the five `*_share` columns |
| `priced_mwh` | Generation of production types that carry an emission factor (excludes `B06` oil, `B10`, `B15`, `B17`, `B20`, `B25`) | `carbon_intensity_g_kwh` |

`carbon_intensity_g_kwh` therefore describes only the priced part of generation. `priced_share`
states how much that is: it is at least 0.91 in `FR` and falls to 0.76 in `DE-LU` in hours with
heavy pumped-storage discharge. The renewable, nuclear and fossil shares do not sum
to 1, because waste (`B17`), other (`B20`) and battery storage (`B25`) belong to no group.

### Fuel groups

| Column | ENTSO-E production types |
| :--- | :--- |
| `wind_mwh` | `B18` offshore, `B19` onshore |
| `solar_mwh` | `B16` |
| `renewable_mwh` | `B01` biomass, `B09` geothermal, `B11` run-of-river, `B12` reservoir, `B15` other renewable, `B16`, `B18`, `B19` |
| `nuclear_mwh` | `B14` |
| `fossil_mwh` | `B02` lignite, `B03` coal-derived gas, `B04` gas, `B05` hard coal, `B06` oil |

## Data-quality flags and known gaps

- **Partial hours (`is_partial`, `n_partial_steps`):** 15 `DE-LU` hours contain at least one step at
  which a series the zone publishes was missing. Upstream withholds the intensity of such steps.
  The snapshot keeps the recomputed hourly value so the row is auditable, and the feature pipeline
  masks it: a partial hour is never a forecast origin, a lag value or a target.
- **Incomplete hours (`hours_covered` < 1):** 2 `FR` hours (`2025-12-10 14:00` and `18:00 UTC`)
  hold only 30 and 45 minutes of data. Their energy columns are understated; they are masked
  the same way.
- **Missing hours:** 4 `FR` hours have no row (`2025-10-25 12:00` and `2025-12-10 15:00–17:00 UTC`).
  The feature pipeline rebuilds a gap-free hourly grid before computing lags.
- **Timestamps:** stored without a timezone and always UTC. Ingestion makes this explicit by
  converting `timestamp_utc` to timezone-aware UTC.

## What the snapshot leaves out

- **Emission-factor uncertainty.** Upstream carries a Monte Carlo interval per step for the
  uncertainty of the life-cycle factors. It is not exported. Forecast intervals built on this
  snapshot are conditional on the median factors. For `FR` the factor interval (about 44
  gCO2eq/kWh wide on average) is wider than the mean intensity itself (about 32).
- **Day-ahead prices, load and cross-border flows.** Available upstream, not exported. The
  intensity is production-based, not consumption-based.

## Schema Contract

| Column | Type | Description |
| :--- | :--- | :--- |
| `zone` | `string` | Bidding zone (`DE-LU` or `FR`) |
| `timestamp_utc` | `datetime64[us]` | Hourly UTC bucket start (timezone-naive, UTC by contract) |
| `total_mwh`, `gen_total_mwh`, `priced_mwh` | `float64` | Generation under the three definitions above (MWh) |
| `emissions_kg` | `float64` | Life-cycle emissions of the priced generation at median factors (kg CO2eq) |
| `carbon_intensity_g_kwh` | `float64` | `emissions_kg / priced_mwh` (`gCO2eq/kWh`) |
| `wind_mwh`, `solar_mwh`, `renewable_mwh`, `nuclear_mwh`, `fossil_mwh` | `float64` | Generation per fuel group (MWh) |
| `wind_share`, `solar_share`, `renewable_share`, `nuclear_share`, `fossil_share` | `float64` | Fuel group over `gen_total_mwh` (`[0, 1]`) |
| `priced_share` | `float64` | `priced_mwh / total_mwh` (`[0, 1]`) |
| `is_partial` | `bool` | At least one upstream step in the hour was partial |
| `n_partial_steps` | `int64` | Number of partial upstream steps in the hour |
| `hours_covered` | `float64` | Summed length of the upstream steps in the hour (1.0 when complete) |
| `upstream_payload_sha256` | `string` | Upstream `source_sha256`: the key of the raw ENTSO-E retrieval in the upstream `raw/manifest.csv`. One value per zone and yearly pull, so 4 distinct values. |
