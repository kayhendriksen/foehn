# Collections

MeteoSwiss organises its open data into five categories. Category **B** (atmosphere measurements -- radio soundings, ceilometer, ozone, etc.) is **not yet released** (B1 radio soundings expected first half of 2026).

---

## A -- Ground-based measurements

Station-level time series in CSV, split into time slices (`historical`, `recent`, `now`). Converted to Parquet.

| Dataset | ID | Description | Frequencies | Stations | Parameters |
|---|---|---|---|---|---|
| `smn` | A1 | **Automatic weather stations** -- the core SwissMetNet network. ~160 stations across Switzerland measuring temperature, humidity, pressure, precipitation, wind, radiation, sunshine, soil temperature, and dew point. | 10-min, hourly, daily, monthly, yearly | 158 | 181 |
| `smn_precip` | A2 | **Automatic precipitation stations** -- rain-gauge-only network. Reports precipitation totals at multiple granularities. | 10-min, hourly, daily, monthly, yearly | 141 | 6 |
| `smn_tower` | A3 | **Tower stations** -- tall mast measurements for temperature, humidity, wind (scalar + gusts), radiation, and sunshine at tower height. | 10-min, hourly, daily, monthly, yearly | 4 | 46 |
| `nime` | A5 | **Manual precipitation stations** -- observer-read gauges reporting daily precipitation, plus fresh snow depth and snow cover. | daily, monthly, yearly | 270 | 17 |
| `tot` | A6 | **Totaliser precipitation** -- remote alpine rain gauges read once per year, reporting precipitation reduced to hydrological year (Oct 1 -- Sep 30). | yearly | 57 | 1 |
| `pollen` | A7 | **Pollen stations** -- airborne pollen concentrations for 7 taxa: alder, birch, hazel, beech, ash, oak, and grasses (_Poaceae_). | hourly, daily, yearly | 16 | 28 |
| `obs` | A8 | **Visual / meteorological observations** -- human-observed daily cloud cover, counts of days with rain, snowfall, hail, fog, and snow coverage. | daily, monthly, yearly | 8 | 27 |
| `phenology` | A9 | **Phenological observations** -- day-of-year for lifecycle events (leaf unfolding, flowering, fruit maturity, leaf colouring, leaf drop) across 26 plant species including horse chestnut, beech, cherry, apple, grape vine, and larch. | yearly | 175 | 71 |

---

## C -- Climate data

| Key | ID | Description | Format |
|---|---|---|---|
| `nbcn` | C1 | **Homogeneous climate stations** -- break-adjusted daily, monthly, and yearly series for temperature, pressure, precipitation, sunshine, and cloud cover (29 stations). Used for long-term trend analysis. | CSV -> Parquet |
| `nbcn_precip` | C2 | **Homogeneous precipitation** -- break-adjusted monthly and yearly precipitation series (46 stations). | CSV -> Parquet |
| `surface_derived_grid` | C3 | **Ground-based spatial analyses** -- gridded fields of precipitation, temperature, and sunshine duration derived from station interpolation. | NetCDF (opt-in) |
| `satellite_derived_grid` | C4 | **Satellite-based spatial analyses** -- gridded radiation, cloud cover, and land surface temperature derived from satellite. | NetCDF (opt-in) |
| `climate_normals` | C6 | **Station normals** -- 30-year reference averages for 1961--1990 and 1991--2020. Monthly values per station. | TXT -> Parquet |
| `climate_normals_*` | C7 | **Spatial normals** -- gridded 30-year reference maps for precipitation, sunshine, and temperature (both reference periods). | NetCDF / GeoTIFF (opt-in) |
| `climate_scenarios` | C8 | **CH2025 local scenarios** -- daily station-level climate projections, one column per climate model. Dates are a nominal 30-year period (0001--0030 on a 365-day calendar), not real calendar dates. | CSV -> Parquet |
| `climate_scenarios_grid` | C9 | **CH2025 gridded scenarios** -- spatially gridded climate projections. | NetCDF (opt-in) |
| `climate_scenarios_indoor` | -- | **CH2025 indoor climate** -- hourly indoor-climate scenario series per station, scenario (RCP), and variant, delivered as a single ZIP of CSVs. | CSV+ZIP -> Parquet |

---

## D -- Radar data

| Key | ID | Description | Format |
|---|---|---|---|
| `radar_precip` | D1 | **Precipitation radar** -- composite precipitation grids at 5--10 min intervals. | HDF5 (opt-in) |
| `radar_hail` | D3 | **Hail radar** -- probability-of-hail grids at 5 min intervals. | HDF5 (opt-in) |

Radar collections are large and require `--grids` to download.

---

## E -- Forecast data

| Key | ID | Description | Format |
|---|---|---|---|
| `forecast_icon_ch1` | E2 | **ICON-CH1-EPS** -- 1 km ensemble forecast model over Switzerland. | GRIB2 (opt-in) |
| `forecast_icon_ch2` | E3 | **ICON-CH2-EPS** -- 2.1 km ensemble forecast model. | GRIB2 (opt-in) |
| `forecast_local` | E4 | **Local point forecasts** -- hourly and daily forecasts for ~5,600 points (stations + postal codes) covering temperature, precipitation, wind, radiation, and more (32 parameters). | CSV -> Parquet |
| `analysis_kenda_ch1` | E5 | **KENDA-CH1 numerical weather analysis** -- 1 km gridded analysis of temperature, humidity, wind, pressure, radiation, pollen, and more from the KENDA-CH1 data assimilation system, plus 1-hour First Guess estimates. Updated hourly, last 24 hours only. | GRIB2 (opt-in) |

GRIB2 forecast and analysis collections are large and require `--grids` to download.

---

## Hail hazard maps

Static spatial reference grids showing expected hail grain size (cm) at different return periods. These are not categorised under A--E because they are static hazard assessments, not measured or forecasted time series -- they represent probabilistic climatological analyses published as fixed reference maps.

| Key | Description | Format |
|---|---|---|
| `hail_hazard_10y` | Hail grain size -- 10-year return period | NetCDF / GeoTIFF (opt-in) |
| `hail_hazard_20y` | Hail grain size -- 20-year return period | NetCDF / GeoTIFF (opt-in) |
| `hail_hazard_50y` | Hail grain size -- 50-year return period | NetCDF / GeoTIFF (opt-in) |
| `hail_hazard_100y` | Hail grain size -- 100-year return period | NetCDF / GeoTIFF (opt-in) |

---

## Time slices

MeteoSwiss splits CSV data into three time slices, encoded in the filename:

| Slice | Range | Update frequency | Frequencies |
|---|---|---|---|
| `recent` | Jan 1 this year to yesterday | Daily at 12:00 UTC | 10-min, hourly, daily, monthly |
| `historical` | Start of measurement to Dec 31 last year | Once per year (early January) | 10-min, hourly, daily, monthly |
| `now` | Yesterday 12:00 UTC to now | Every 10 minutes | 10-min, hourly only |

Some collections (phenology, totaliser, yearly aggregates) don't use time slices -- they publish a single file per station.

All timestamps are UTC. For 10-min and hourly data the timestamp marks the **end** of the interval (16:00 = 15:50:01--16:00:00). For daily, monthly, and yearly data the timestamp marks the **start** (2023-06-01 = the whole of June).
