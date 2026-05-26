<h1 align="center">
  <img src="https://raw.githubusercontent.com/kayhendriksen/foehn/main/assets/banner.svg" alt="foehn" width="600">
</h1>

<p align="center">
  <strong>MeteoSwiss Open Data — Python API, CLI, MCP server, Parquet & Delta tables</strong>
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
  <a href="https://scorecard.dev/viewer/?uri=github.com/kayhendriksen/foehn">
    <img src="https://api.scorecard.dev/projects/github.com/kayhendriksen/foehn/badge" alt="OpenSSF Scorecard">
  </a>
  <a href="https://pypi.org/project/foehn/">
    <img src="https://img.shields.io/pypi/dm/foehn.svg" alt="Monthly Downloads">
  </a>
</p>

---

foehn downloads every [MeteoSwiss OGD](https://github.com/MeteoSwiss/opendata) collection via the STAC API, converts CSV/TXT to Parquet with [Polars](https://pola.rs), and optionally ingests everything into [Databricks](https://www.databricks.com) Unity Catalog Delta tables on a daily schedule. It also ships an [MCP server](https://modelcontextprotocol.io) so LLMs can query Swiss weather data directly.

<p align="center">
  <img src="https://raw.githubusercontent.com/kayhendriksen/foehn/main/assets/mcp_demo.png" alt="Daily weather in Bern, powered by foehn" width="700">
</p>
<p align="center">
  <em>Daily weather in Bern, powered by foehn's MCP server and MeteoSwiss open data.</em>
</p>

## Why foehn?

- **20+ collections in one command** — weather stations, radar, hail maps, forecasts, climate scenarios, and more
- **MCP server for LLMs** — give your favorite LLM live access to MeteoSwiss data with the MCP server
- **Significantly smaller on disk** — columnar Parquet with Zstandard compression vs. raw CSVs
- **Incremental by default** — only re-downloads files that changed since your last run, tracked via `_last_run.json`
- **No Spark required locally** — download + conversion uses Polars only; Spark is optional for Delta ingestion
- **Ships a Declarative Automation Bundle** — ready-to-deploy daily job and historical backfill, no pipeline config needed

---

## Quick start

```bash
pip install foehn
foehn download
```

Recent data (Jan 1 to yesterday) is downloaded and converted to Parquet under `./data/meteoswiss/`.

<img src="https://raw.githubusercontent.com/kayhendriksen/foehn/main/assets/cli_demo.gif" alt="foehn CLI demo" width="800">

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

**With extras:**
```bash
pip install "foehn[databricks]"   # PySpark + Delta
pip install "foehn[mcp]"          # MCP server
pip install "foehn[grids]"        # xarray + Zarr for all gridded data (NetCDF, GRIB2, radar)
```

Requires Python 3.11 or later.

---

## Python API

```python
import foehn

df = foehn.load("smn", station="BER", frequency="d")
```

Load data directly into Polars DataFrames, explore metadata, download to disk, and convert to Parquet — all from Python. See the [full Python API documentation](docs/python-api.md).

---

## CLI

```bash
foehn download smn pollen
foehn load smn --station BER --frequency d
```

The CLI mirrors the Python API with subcommands for downloading, converting, loading, and inspecting metadata. See the [full CLI documentation](docs/cli.md).

---

## Gridded data

```python
ds = foehn.open_dataset("surface_derived_grid", match="rhiresd")  # NetCDF climate grid
ds = foehn.open_dataset("forecast_icon_ch1", match="202605231500-0-t_2m-ctrl")  # one GRIB2 field
ds = foehn.open_dataset("radar_precip", match="cpc2613000000")    # one radar composite
foehn.to_zarr("surface_derived_grid", match="rhiresd")            # Zarr store
```

NetCDF climate grids/normals/scenarios, GRIB2 forecasts (ICON-CH1/CH2, KENDA), and HDF5/ODIM radar composites all open as xarray Datasets instead of DataFrames. One extra covers them: `pip install "foehn[grids]"`. See the [gridded data documentation](docs/grids.md).

---

## MCP server

```json
{
  "mcpServers": {
    "foehn": {
      "command": "foehn",
      "args": ["mcp"]
    }
  }
}
```

Give any MCP-compatible LLM live access to MeteoSwiss data. See the [full MCP server documentation](docs/mcp-server.md).

---

## Documentation

| | |
|---|---|
| [Collections](docs/collections.md) | All 20+ MeteoSwiss datasets, categories, and time slice conventions |
| [Python API](docs/python-api.md) | Loading data, metadata, downloading, and Parquet conversion |
| [Gridded data](docs/grids.md) | NetCDF grids as xarray Datasets and Zarr stores |
| [CLI](docs/cli.md) | All subcommands, flags, and environment variables |
| [MCP Server](docs/mcp-server.md) | Setup, configuration, and available tools |
| [Databricks Pipeline](docs/databricks.md) | Declarative Automation Bundle deployment |

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

<!-- mcp-name: io.github.kayhendriksen/foehn -->
