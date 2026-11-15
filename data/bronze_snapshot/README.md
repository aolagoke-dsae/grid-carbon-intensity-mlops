# Bronze Snapshot Provenance (Gate G1)

- **File:** entsoe_generation_sample.parquet
- **SHA-256 Checksum:** $sha
- **Upstream Repository:** grid-carbon-intensity (commit: 7caaee395e3303c5eea26c714645001b7dda01cd)
- **Coverage Window:** 2024-01-01 00:00:00 UTC to 2025-12-31 23:00:00 UTC (hourly resampled, 35,082 rows across DE-LU and FR)

## Schema Contract

| Column | Type | Description |
| :--- | :--- | :--- |
| zone | string | Bidding zone (DE-LU or FR) |
| 	imestamp_utc | datetime64[us] | Hourly UTC bucket start |
| 	otal_mwh / priced_mwh | loat64 | Total & emission-priced generation (MWh) |
| emissions_kg | loat64 | Total direct + lifecycle grid emissions (kg CO2eq) |
| carbon_intensity_g_kwh | loat64 | Realized grid carbon intensity (gCO2eq/kWh) |
| wind_share, solar_share, enewable_share, 
uclear_share, ossil_share | loat64 | Share of total hourly generation ([0, 1]) |
| upstream_payload_sha256 | string | Deterministic SHA-256 digest of upstream ENTSO-E payload hashes |