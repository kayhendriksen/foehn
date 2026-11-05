<p align="center">
  <img src="https://raw.githubusercontent.com/kayhendriksen/foehn/main/assets/banner.svg" alt="foehn" width="560">
</p>

<p align="center">
  <strong>MeteoSwiss Open Data — Python API, CLI and MCP server.</strong><br>
  Tabular data as Polars DataFrames and Parquet, gridded data as xarray and Zarr.
</p>

---

foehn downloads every [MeteoSwiss OGD](https://github.com/MeteoSwiss/opendata)
collection through the STAC API, converts station CSVs to Parquet with
[Polars](https://pola.rs), and opens gridded collections — NetCDF climate grids,
GRIB2 forecasts and ODIM radar composites — as [xarray](https://xarray.dev)
Datasets or [Zarr](https://zarr.dev) stores. It can ingest everything into
[Databricks](https://www.databricks.com) Delta tables on a schedule, and ships an
[MCP server](https://modelcontextprotocol.io) so LLMs can query Swiss weather
data directly.

## Highlights

- **20+ collections, one interface** — weather stations, pollen, radar, hail
  maps, forecasts, climate grids and scenarios.
- **Tabular and gridded** — CSV station data as DataFrames or Parquet; NetCDF,
  GRIB2 and radar as xarray Datasets or Zarr stores.
- **Incremental by default** — only files that changed since the last run are
  downloaded, tracked per dataset and time slice.
- **Scriptable CLI** — every command prints a table for people and CSV/JSON for
  pipelines, with diagnostics kept on stderr.
- **MCP server for LLMs** — give any MCP client live access to MeteoSwiss data.
- **No Spark required** — download and conversion use Polars only; Spark is
  optional, for Delta ingestion.

## Installation

```bash
pip install foehn                 # or: uv add foehn
pip install "foehn[grids]"        # xarray + Zarr for NetCDF, GRIB2 and radar
pip install "foehn[mcp]"          # the MCP server
pip install "foehn[databricks]"   # PySpark + Delta ingestion
```

foehn requires Python 3.11 or later.

## Quick start

=== "Python"

    ```python
    import foehn

    # Recent daily observations for Bern, as a Polars DataFrame
    df = foehn.load("smn", station="BER", frequency="d")

    # What does each column mean?
    foehn.parameters("smn")

    # One radar composite as an xarray Dataset (needs foehn[grids])
    ds = foehn.open_dataset("radar_precip", match="cpcYYDDDHHMM")
    ```

=== "CLI"

    ```bash
    foehn list                                         # every dataset
    foehn load smn --station BER --frequency d         # preview in the terminal
    foehn load smn --station BER --frequency d -o ber.parquet
    foehn download smn pollen                          # to ./data/meteoswiss as Parquet
    ```

=== "MCP"

    ```json
    {
      "mcpServers": {
        "foehn": { "command": "foehn", "args": ["mcp"] }
      }
    }
    ```

## Where next

| | |
|---|---|
| [Collections](collections.md) | Every MeteoSwiss dataset, its categories and time slices |
| [Python API guide](python-api.md) | Loading data, metadata, downloading and Parquet conversion |
| [Gridded data](grids.md) | NetCDF, GRIB2 and radar as xarray Datasets and Zarr stores |
| [Command line](cli.md) | Every subcommand, flag, output format and exit status |
| [MCP server](mcp-server.md) | Setup, configuration and the available tools |
| [Databricks pipeline](databricks.md) | The Declarative Automation Bundle |
| [API reference](reference/api.md) | Every public function, generated from the docstrings |

## Data sources

Data is provided by [MeteoSwiss Open Government Data](https://opendatadocs.meteoswiss.ch)
through the [geo.admin.ch STAC API](https://data.geo.admin.ch/api/stac/v1).
foehn is an independent project and is not affiliated with MeteoSwiss.
