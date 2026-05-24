# Gridded data (NetCDF / GRIB2 → xarray / Zarr)

Most MeteoSwiss datasets are tabular CSV (station measurements) — see the
[Python API](python-api.md) for `foehn.load()`. A handful are *gridded*:
N-dimensional spatial fields stored as NetCDF (climate grids) or GRIB2
(forecasts). These have no station/row shape, so they are read as
[xarray](https://docs.xarray.dev) Datasets instead of Polars DataFrames.

`foehn.open_dataset()` is the gridded analog of `foehn.load()`, and
`foehn.to_zarr()` is the analog of `foehn.to_parquet()`.

---

## Installation

NetCDF grids need the `grids` extra (xarray, netCDF4, h5netcdf, zarr):

```bash
pip install "foehn[grids]"
```

GRIB2 forecasts need the separate `grib` extra (xarray, cfgrib, eccodes):

```bash
pip install "foehn[grib]"          # forecasts only
pip install "foehn[grids,grib]"    # both, plus Zarr writing
```

`grib` is kept separate because **eccodes** ships a system C library whose
install can be fragile on some platforms — isolating it means a failed eccodes
build can't break NetCDF support. Core foehn stays at two dependencies; nothing
above is imported until you call a grid function. `rechunk=` additionally needs
`dask` (`pip install dask`), deliberately **not** bundled in either extra.

---

## Which datasets are gridded?

**NetCDF** (`grids` extra): the spatial climate analyses (`surface_derived_grid`,
`satellite_derived_grid`), the `climate_normals_*` reference grids,
`climate_scenarios_grid`, and the `hail_hazard_*` maps.

**GRIB2** (`grib` extra): the forecasts `forecast_icon_ch1`, `forecast_icon_ch2`,
and the analysis `analysis_kenda_ch1`. See [Forecasts (GRIB2)](#forecasts-grib2)
below — they behave differently from the static NetCDF grids.

List the gridded collections with:

```python
import foehn

[d["dataset"] for d in foehn.list_datasets() if d["format"] in ("NetCDF", "GRIB2")]
```

HDF5 radar (`radar_precip`, `radar_hail`) is **not** readable yet —
`open_dataset()` raises `NotImplementedError` (it needs `xradar`). Download the
raw files with `foehn download <dataset> --grids`.

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
| `dataset` | `str` | Grid dataset name (NetCDF or GRIB2 collection) |
| `variables` | `str` or `list[str]` | Restrict to these data variable(s) |
| `match` | `str` | Keep only source files whose name contains this substring (required for GRIB2) |
| `data_dir` | `str` or `Path` | Root data directory (default `./data/meteoswiss`) |
| `engine` | `str` | xarray backend; default auto-detects (NetCDF-3/4) or uses cfgrib (GRIB2) |

---

## Forecasts (GRIB2)

ICON-CH1/CH2 forecasts and KENDA analysis are GRIB2 (`grib` extra) and behave
differently from the static NetCDF grids:

- **`match` is required and must resolve to a single file.** A forecast
  collection is *thousands* of files — one per variable × ensemble member ×
  lead time × reference time. `match` is a substring of the source filename,
  which looks like
  `icon-ch1-eps-<reftime>-<leadtime>-<variable>-<ctrl|perturb>.grib2`, so include
  the reference time and lead time:

  ```python
  ds = foehn.open_dataset("forecast_icon_ch1", match="202605231500-0-t_2m-ctrl")
  ```

  If `match` is omitted, or matches more than one file, `open_dataset()` raises a
  `ValueError` **before downloading anything** (and lists example filenames so
  you can narrow it).

- **Native unstructured grid.** ICON runs on an icosahedral grid, so cfgrib
  returns a 1-D `values` dimension (~1.1M cells) with **no lat/lon attached**
  (the grid-definition file ships separately and is not joined here). Because
  that dimension has no coordinate, files **cannot** be stacked across lead
  times / reference times yet — multi-file consolidation (concat-along-step,
  kerchunk) is a planned follow-up. For now you read one field at a time.

- Writing to Zarr works the same way (and inherits the single-file `match`
  requirement):

  ```python
  foehn.to_zarr("forecast_icon_ch1", match="202605231500-0-t_2m-ctrl")
  # -> data/meteoswiss/zarr/forecast_icon_ch1__202605231500_0_t_2m_ctrl.zarr
  ```

---

## Writing to Zarr

`foehn.to_zarr()` materialises a dataset to a Zarr store under
`<data_dir>/zarr/`. The default name encodes `match`, so different filtered
slices of a collection don't overwrite each other:

```python
store = foehn.to_zarr("surface_derived_grid", match="rhiresd")
# -> data/meteoswiss/zarr/surface_derived_grid__rhiresd.zarr

foehn.to_zarr("surface_derived_grid", match="tabsd")
# -> data/meteoswiss/zarr/surface_derived_grid__tabsd.zarr  (distinct store)

# Unfiltered keeps the bare name
foehn.to_zarr("hail_hazard_50y")
# -> data/meteoswiss/zarr/hail_hazard_50y.zarr

# Override the location explicitly
foehn.to_zarr("surface_derived_grid", match="rhiresd", store="out/rain.zarr")

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

# Write a Zarr store (default name: surface_derived_grid__rhiresd.zarr)
foehn to-zarr surface_derived_grid --match rhiresd

# Write to an explicit path
foehn to-zarr surface_derived_grid --match rhiresd --out out/rain.zarr
```

---

## MCP

The MCP server exposes the read/inspect path as the `describe_grid` tool: it
returns a grid's dimensions, coordinates, and variables without downloading
array values into the LLM context (it still caches the source file — same
download-then-lazy caveat). Writing Zarr stores (`to_zarr`) is intentionally not
an MCP tool, since the server is read-only. See the
[MCP server docs](mcp-server.md).
