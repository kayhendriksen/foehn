<h1 align="center">
  <img src="https://raw.githubusercontent.com/kayhendriksen/foehn/main/assets/banner.svg" alt="foehn" width="600">
</h1>

<p align="center">
  <strong>MeteoSwiss Open Data → Parquet → Databricks Delta tables</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/foehn/">
    <img src="https://img.shields.io/pypi/v/foehn.svg" alt="PyPI Latest Release">
  </a>
  <a href="https://pypi.org/project/foehn/">
    <img src="https://img.shields.io/pypi/pyversions/foehn.svg" alt="Python Versions">
  </a>
  <a href="https://github.com/kayhendriksen/foehn/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">
  </a>
  <a href="https://pypi.org/project/foehn/">
    <img src="https://img.shields.io/pypi/dm/foehn.svg" alt="Monthly Downloads">
  </a>
</p>

---

foehn downloads every [MeteoSwiss OGD](https://github.com/MeteoSwiss/opendata) collection via the STAC API, converts CSV/TXT to Parquet with [Polars](https://pola.rs), and optionally ingests everything into [Databricks](https://www.databricks.com) Unity Catalog Delta tables on a daily schedule.

## Why foehn?

- **20+ collections in one command** — weather stations, radar, hail maps, forecasts, climate scenarios, and more
- **Significantly smaller on disk** — columnar Parquet with Zstandard compression vs. raw CSVs
- **Incremental by default** — only re-downloads files that changed since your last run, tracked via `_last_run.json`
- **No Spark required locally** — download + conversion uses Polars only; Spark is optional for Delta ingestion
- **Ships a Declarative Automation Bundle** — ready-to-deploy daily job and historical backfill, no pipeline config needed

---

## Quick start

```bash
pip install foehn
foehn
```

Recent data (Jan 1 → yesterday) is downloaded and converted to Parquet under `./data/meteoswiss/`.

---

## Collections

MeteoSwiss organises its open data into five categories. Category **B** (atmosphere measurements — radio soundings, ceilometer, ozone, etc.) is **not yet released** (B1 radio soundings expected first half of 2026).

### A — Ground-based measurements

Station-level time series in CSV, split into time slices (`historical`, `recent`, `now`). Converted to Parquet.

| Key | ID | Description | Granularities | Stations | Parameters |
|---|---|---|---|---|---|
| `smn` | A1 | **Automatic weather stations** — the core SwissMetNet network. ~160 stations across Switzerland measuring temperature, humidity, pressure, precipitation, wind, radiation, sunshine, soil temperature, and dew point. | 10-min, hourly, daily, monthly, yearly | 158 | 181 |
| `smn_precip` | A2 | **Automatic precipitation stations** — rain-gauge-only network. Reports precipitation totals at multiple granularities. | 10-min, hourly, daily, monthly, yearly | 141 | 6 |
| `smn_tower` | A3 | **Tower stations** — tall mast measurements for temperature, humidity, wind (scalar + gusts), radiation, and sunshine at tower height. | 10-min, hourly, daily, monthly, yearly | 4 | 46 |
| `nime` | A5 | **Manual precipitation stations** — observer-read gauges reporting daily precipitation, plus fresh snow depth and snow cover. | daily, monthly, yearly | 273 | 17 |
| `tot` | A6 | **Totaliser precipitation** — remote alpine rain gauges read once per year, reporting precipitation reduced to hydrological year (Oct 1 – Sep 30). | yearly | 57 | 1 |
| `pollen` | A7 | **Pollen stations** — airborne pollen concentrations for 7 taxa: alder, birch, hazel, beech, ash, oak, and grasses (_Poaceae_). | hourly, daily, yearly | 16 | 28 |
| `obs` | A8 | **Visual / meteorological observations** — human-observed daily cloud cover, counts of days with rain, snowfall, hail, fog, and snow coverage. | daily, monthly, yearly | 20 | 27 |
| `phenology` | A9 | **Phenological observations** — day-of-year for lifecycle events (leaf unfolding, flowering, fruit maturity, leaf colouring, leaf drop) across 26 plant species including horse chestnut, beech, cherry, apple, grape vine, and larch. | yearly | 175 | 71 |
### C — Climate data

| Key | ID | Description | Format |
|---|---|---|---|
| `nbcn` | C1 | **Homogeneous climate stations** — break-adjusted series for temperature, pressure, precipitation, sunshine, and cloud cover (29 stations). Used for long-term trend analysis. | CSV → Parquet |
| `nbcn_precip` | C2 | **Homogeneous precipitation** — break-adjusted precipitation series (46 stations). | CSV → Parquet |
| `surface_derived_grid` | C3 | **Ground-based spatial analyses** — gridded fields of precipitation, temperature, and sunshine duration derived from station interpolation. | NetCDF (opt-in) |
| `satellite_derived_grid` | C4 | **Satellite-based spatial analyses** — gridded radiation, cloud cover, and land surface temperature derived from satellite. | NetCDF (opt-in) |
| `climate_normals` | C6 | **Station normals** — 30-year reference averages for 1961–1990 and 1991–2020. Monthly values per station. | TXT → Parquet |
| `climate_normals_*` | C7 | **Spatial normals** — gridded 30-year reference maps for precipitation, sunshine, and temperature (both reference periods). | NetCDF / GeoTIFF (opt-in) |
| `climate_scenarios` | C8 | **CH2025 local scenarios** — station-level climate projections. | CSV → Parquet |
| `climate_scenarios_grid` | C9 | **CH2025 gridded scenarios** — spatially gridded climate projections. | NetCDF (opt-in) |

### D — Radar data

| Key | ID | Description | Format |
|---|---|---|---|
| `radar_precip` | D1 | **Precipitation radar** — composite precipitation grids at 5–10 min intervals. | HDF5 (opt-in) |
| `radar_hail` | D3 | **Hail radar** — probability-of-hail grids at 5 min intervals. | HDF5 (opt-in) |

Radar collections are large and require `--grids` to download.

### E — Forecast data

| Key | ID | Description | Format |
|---|---|---|---|
| `forecast_icon_ch1` | E2 | **ICON-CH1-EPS** — 1 km ensemble forecast model over Switzerland. | GRIB2 (opt-in) |
| `forecast_icon_ch2` | E3 | **ICON-CH2-EPS** — 2.1 km ensemble forecast model. | GRIB2 (opt-in) |
| `forecast_local` | E4 | **Local point forecasts** — forecasts for ~5,600 points (stations + postal codes) covering temperature, precipitation, wind, radiation, and more (32 parameters). | CSV → Parquet |

GRIB2 forecast collections are large and require `--grids` to download.

### Hail hazard maps

Static spatial reference grids showing expected hail grain size (cm) at different return periods. These are **not categorised under A–E** because they are static hazard assessments, not measured or forecasted time series — they represent probabilistic climatological analyses published as fixed reference maps.

| Key | Description | Format |
|---|---|---|
| `hail_hazard_10y` | Hail grain size — 10-year return period | NetCDF / GeoTIFF (opt-in) |
| `hail_hazard_20y` | Hail grain size — 20-year return period | NetCDF / GeoTIFF (opt-in) |
| `hail_hazard_50y` | Hail grain size — 50-year return period | NetCDF / GeoTIFF (opt-in) |
| `hail_hazard_100y` | Hail grain size — 100-year return period | NetCDF / GeoTIFF (opt-in) |

---

## Time slices

MeteoSwiss splits CSV data into three time slices, encoded in the filename:

| Slice | Range | Update frequency | Granularities |
|---|---|---|---|
| `recent` | Jan 1 this year → yesterday | Daily at 12:00 UTC | 10-min, hourly, daily, monthly |
| `historical` | Start of measurement → Dec 31 last year | Once per year (early January) | 10-min, hourly, daily, monthly |
| `now` | Yesterday 12:00 UTC → now | Every 10 minutes | 10-min, hourly only |

Some collections (phenology, totaliser, yearly aggregates) don't use time slices — they publish a single file per station.

All timestamps are UTC. For 10-min and hourly data the timestamp marks the **end** of the interval (16:00 = 15:50:01–16:00:00). For daily, monthly, and yearly data the timestamp marks the **start** (2023-06-01 = the whole of June).

---

## Installation

**From PyPI:**
```bash
pip install foehn
```

**From source:**
```bash
git clone https://github.com/kayhendriksen/foehn
cd foehn
pip install -e .
```

**With Databricks extras** (PySpark + Delta):
```bash
pip install "foehn[databricks]"
```

Requires Python ≥ 3.10.

---

## Python API

Use foehn directly from notebooks or scripts:

```python
import foehn

# List all available datasets
foehn.list_datasets()
# [{'key': 'smn', 'collection_id': 'ch.meteoschweiz.ogd-smn', 'category': 'A',
#   'subcategory': 'A1', 'description': 'Automatic weather stations',
#   'format': 'CSV', 'granularities': ['t', 'h', 'd', 'm'],
#   'time_slices': ['historical', 'recent', 'now']}, ...]

# Load data directly into a Polars DataFrame (nothing written to disk)
df = foehn.load("smn", station="BER", granularity="d")

# Filter by multiple stations and granularities
df = foehn.load("smn", station=["BER", "ZUR"], granularity=["d", "h"])

# Include historical data
df = foehn.load("smn", station="BER", granularity="d", data_types=["historical", "recent"])

# Download a single dataset to disk
foehn.download("smn", data_dir="./data/meteoswiss")

# Download with specific time slices
foehn.download("smn", data_types=["historical", "recent"])

# Convert downloaded CSVs to Parquet
foehn.convert("smn", data_dir="./data/meteoswiss")
```

---

## CLI reference

```
foehn [options]
```

**Time range** — recent (Jan 1 this year → yesterday) is always included; flags extend it:

| Flag | Description |
|---|---|
| *(none)* | Recent only — Jan 1 this year → yesterday, updated daily at 12 UTC |
| `--historical` | Also fetch full archive — start of measurement → Dec 31 last year |
| `--now` | Also fetch realtime slice — yesterday 12 UTC → now, 10-min updates |
| `--all` | All three slices: historical + recent + now |

**Behaviour:**

| Flag | Description |
|---|---|
| `--full-refresh` | Ignore incremental tracking, re-download everything |
| `--convert-only` | Convert existing CSVs to Parquet without downloading |

**Output:**

| Flag | Description |
|---|---|
| `--list` | List available collections and exit |
| `--grids` | Also fetch GRIB2, radar HDF5, NetCDF, GeoTIFF (large) |
| `--no-parquet` | Skip conversion, keep raw CSVs only |
| `--data-dir PATH` | Output root (default: `./data/meteoswiss`) |

Parquet files land in `<data-dir>/parquet/<collection>/`.

---

## Environment variables

Settings can also be configured via environment variables. CLI flags always take precedence.

| Variable | Equivalent | Description |
|---|---|---|
| `FOEHN_DATA_DIR` | `--data-dir` | Root data directory |
| `FOEHN_FULL_REFRESH` | `--full-refresh` | Set to `1`, `true`, or `yes` to ignore incremental tracking |

---

## Databricks pipeline

The recommended setup uses Declarative Automation Bundles.

**1. Set variables:**
```bash
export BUNDLE_VAR_host=https://adb-xxx.azuredatabricks.net
export BUNDLE_VAR_alert_email=you@example.com
```

**2. Deploy:**
```bash
pip install databricks-cli
databricks bundle validate
databricks bundle deploy -t prod
```

This deploys two jobs:

- **`foehn_daily`** — runs at 13:30 UTC every day; downloads recent data and refreshes Delta tables
- **`foehn_historical`** — paused by default; trigger manually for first run or on Jan 1 for the annual archive slice

---

## Data sources

| | |
|---|---|
| STAC API | https://data.geo.admin.ch/api/stac/v1 |
| Documentation | https://opendatadocs.meteoswiss.ch |
| MeteoSwiss OGD | https://github.com/MeteoSwiss/opendata |

---

## License

MIT
