# Gridded data (NetCDF → xarray / Zarr)

Most MeteoSwiss datasets are tabular CSV (station measurements) — see the
[Python API](python-api.md) for `foehn.load()`. A handful are *gridded*:
N-dimensional spatial fields stored as NetCDF. These have no station/row shape,
so they are read as [xarray](https://docs.xarray.dev) Datasets instead of Polars
DataFrames.

`foehn.open_dataset()` is the gridded analog of `foehn.load()`, and
`foehn.to_zarr()` is the analog of `foehn.to_parquet()`.

---

## Installation

The grid path needs optional dependencies (xarray, netCDF4, h5netcdf, zarr):

```bash
pip install "foehn[grids]"
```

Core foehn stays at two dependencies; nothing above is imported until you call a
grid function. `rechunk=` additionally needs `dask` (`pip install dask`), which
is deliberately **not** bundled in the extra.

---

## Which datasets are gridded?

NetCDF collections (readable today): the spatial climate analyses
(`surface_derived_grid`, `satellite_derived_grid`), the `climate_normals_*`
reference grids, `climate_scenarios_grid`, the `hail_hazard_*` maps, and
`climate_scenarios_indoor`. Find them with:

```python
import foehn

[d["dataset"] for d in foehn.list_datasets() if d["format"] == "NetCDF"]
```

GRIB2 forecasts (`forecast_icon_ch1/ch2`, `analysis_kenda_ch1`) and HDF5 radar
(`radar_precip`, `radar_hail`) are **not** wired into this read path yet —
`open_dataset()` raises `NotImplementedError` for them. Download the raw files
with `foehn download <dataset> --grids`.

---

## Opening a dataset

```python
import foehn

# Open a whole collection
ds = foehn.open_dataset("surface_derived_grid")

# A multi-file collection mixes parameters/resolutions that can't be combined.
# Narrow to one coherent set with match= (substring of the source filename):
ds = foehn.open_dataset("surface_derived_grid", match="rhiresd")

# Restrict to specific data variables
ds = foehn.open_dataset("climate_scenarios_grid", match="_pr_", variables="pr")
```

> **Download-then-lazy, not cloud-lazy.** The first call downloads the *entire*
> NetCDF to `<data_dir>/bronze/<dataset>/` before anything is read — for
> `climate_scenarios_grid` that's roughly 900 MB up front. Array *values* are
> then read lazily from the local copy; subsequent calls reuse the cache. There
> is no byte-range read of the remote file. Use `match=` to avoid pulling
> parameters you don't need.

Once open, it's an ordinary xarray Dataset — slice it with `.sel()`/`.isel()`,
reduce, plot, etc. (these grids use the Swiss LV95 projection, so spatial
coordinates are in metres, not lat/lon):

```python
ds["TabsD"].mean(dim="time")          # time-mean field
ds.isel(time=0)                        # first time step
ds["TabsD"].sel(x=2_600_000, y=1_200_000, method="nearest")  # nearest grid cell
```

| Parameter | Type | Description |
|---|---|---|
| `dataset` | `str` | Grid dataset name (must be a NetCDF collection) |
| `variables` | `str` or `list[str]` | Restrict to these data variable(s) |
| `match` | `str` | Keep only source files whose name contains this substring |
| `data_dir` | `str` or `Path` | Root data directory (default `./data/meteoswiss`) |
| `engine` | `str` | xarray backend; default auto-detects NetCDF-3 vs NetCDF-4/HDF5 |

---

## Writing to Zarr

`foehn.to_zarr()` materialises a dataset to a Zarr store under
`<data_dir>/zarr/<dataset>.zarr`:

```python
store = foehn.to_zarr("surface_derived_grid", match="rhiresd")
# -> data/meteoswiss/zarr/surface_derived_grid.zarr

# Re-chunk before writing (requires `pip install dask`)
foehn.to_zarr("surface_derived_grid", match="rhiresd", rechunk={"time": 24})
```

Non-CF time axes (MeteoSwiss climate normals label theirs `years since
1991-01-01`, which CF decoding rejects) are sanitised on write so the resulting
store always re-opens cleanly with `xarray.open_zarr()`.

---

## CLI

```bash
# Print an xarray summary of a gridded dataset
foehn open surface_derived_grid --match rhiresd

# Restrict to specific variables
foehn open climate_scenarios_grid --match _pr_ --variables pr

# Write a Zarr store
foehn to-zarr surface_derived_grid --match rhiresd
```

---

## MCP

The MCP server exposes the read/inspect path as the `describe_grid` tool: it
returns a grid's dimensions, coordinates, and variables without downloading
array values into the LLM context (it still caches the source file — same
download-then-lazy caveat). Writing Zarr stores (`to_zarr`) is intentionally not
an MCP tool, since the server is read-only. See the
[MCP server docs](mcp-server.md).
