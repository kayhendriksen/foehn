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
- **Significantly smaller on disk** — columnar Parquet with Snappy compression vs. raw CSVs
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

| Key | Description | Format |
|---|---|---|
| `smn` | Automatic weather stations (A1) | CSV → Parquet |
| `smn_precip` | Automatic precipitation stations (A2) | CSV → Parquet |
| `smn_tower` | Tower stations (A3) | CSV → Parquet |
| `nime` | Manual precipitation stations (A5) | CSV → Parquet |
| `tot` | Totaliser precipitation (A6) | CSV → Parquet |
| `obs` | Visual observations (A8) | CSV → Parquet |
| `pollen` | Pollen stations (A7) | CSV → Parquet |
| `phenology` | Phenological observations (A9) | CSV → Parquet |
| `nbcn` | Homogeneous climate stations (C1) | CSV → Parquet |
| `nbcn_precip` | Homogeneous precipitation (C2) | CSV → Parquet |
| `climate_normals` | Station normals 1961–1990 / 1991–2020 (C6) | TXT → Parquet |
| `climate_normals_*` | Spatial normals (C7) | NetCDF / GeoTIFF |
| `surface_derived_grid` | Spatial analyses — precip, temp, sunshine (C3/C4) | NetCDF |
| `satellite_derived_grid` | Spatial analyses — radiation, clouds (C5) | NetCDF |
| `climate_scenarios` | CH2025 local scenarios (C8) | CSV → Parquet |
| `climate_scenarios_grid` | CH2025 gridded scenarios (C9) | NetCDF |
| `hail_hazard_*` | Hail hazard maps | NetCDF / ZIP |
| `forecast_local` | Local point forecasts (E4) | CSV → Parquet |
| `forecast_icon_ch1/ch2` | ICON-CH1/CH2-EPS (E2/E3) | GRIB2 (opt-in) |
| `radar_precip/hail` | Precipitation + hail radar (D1/D3) | HDF5 (opt-in) |

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

# List all available collections
foehn.list_collections()
# [{'category': 'CSV', 'key': 'smn', 'collection_id': 'ch.meteoschweiz.ogd-smn'}, ...]

# Download a single collection
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
