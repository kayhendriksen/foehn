# MCP Server

foehn includes an [MCP server](https://modelcontextprotocol.io) that gives LLMs live access to MeteoSwiss data. It is published on the [MCP Registry](https://registry.modelcontextprotocol.io).

<p align="center">
  <img src="../assets/mcp_demo.png" alt="Daily weather in Bern, powered by foehn" width="700">
</p>
<p align="center">
  <em>Daily weather in Bern, powered by foehn's MCP server and MeteoSwiss open data.</em>
</p>

---

## Installation

```bash
pip install "foehn[mcp]"
```

---

## Configuration

Add foehn to your MCP client config:

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

---

## Available tools

| Tool | Description |
|---|---|
| `list_datasets` | Browse all MeteoSwiss datasets with metadata |
| `load_data` | Fetch weather measurements as rows |
| `describe_data` | Get summary statistics for a dataset |
| `describe_grid` | Inspect a gridded dataset's dimensions and variables (NetCDF, GRIB2, radar) |
| `get_parameters` | Look up what each column measures |
| `get_stations` | Find station abbreviations and locations |
| `get_inventory` | Check data availability per station |

`describe_grid` is the gridded counterpart to `describe_data`, covering every
gridded collection — NetCDF, GRIB2 forecasts, and HDF5/ODIM radar (GRIB2 and
radar need a single-file `match`). It caches the source file locally on first
use (download-then-lazy) and needs the `grids` extra (`pip install
"foehn[grids]"`). It returns structure only — grid *values* aren't streamed into
the LLM context, and writing Zarr stores isn't exposed over MCP (use the Python
API or `foehn to-zarr` CLI).
