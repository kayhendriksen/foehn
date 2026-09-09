# Gridded data (NetCDF / GRIB2 / radar → xarray / Zarr)

Most MeteoSwiss datasets are tabular CSV (station measurements) — see the
[Python API](python-api.md) for `foehn.load()`. A handful are *gridded*:
N-dimensional spatial fields stored as NetCDF (climate grids), GRIB2 (forecasts),
or HDF5/ODIM (radar composites). These have no station/row shape, so they are
read as [xarray](https://docs.xarray.dev) Datasets instead of Polars DataFrames.

`foehn.open_dataset()` is the gridded analog of `foehn.load()`, and
`foehn.to_zarr()` is the analog of `foehn.to_parquet()`.

---

## Installation

A single `grids` extra covers all three gridded formats — NetCDF (xarray,
netCDF4, h5netcdf), GRIB2 (cfgrib, eccodes), HDF5/ODIM radar (h5py, pyproj), plus
zarr for writing:

```bash
pip install "foehn[grids]"
```

Core foehn stays at two dependencies; nothing above is imported until you call a
grid function. One caveat: **eccodes** (pulled in for GRIB2) ships a system C
library whose wheel can fail to build on some platforms — if so, that breaks the
whole `grids` install, not just GRIB2. `rechunk=` additionally needs `dask`
(`pip install dask`), deliberately **not** bundled in the extra.

---

## Which datasets are gridded?

All three install via `pip install "foehn[grids]"`.

**NetCDF**: the spatial climate analyses (`surface_derived_grid`,
`satellite_derived_grid`, `radar_derived_grid` -- which also carries the hail
return-period and hail-day grids), the `climate_normals_grid` reference grids,
and `climate_scenarios_grid`.

**GRIB2**: the forecasts `forecast_icon_ch1`, `forecast_icon_ch2`, and the
analysis `analysis_kenda_ch1`. See [Forecasts (GRIB2)](#forecasts-grib2).

**HDF5/ODIM radar**: `radar_precip` (CombiPrecip) and `radar_hail`. See
[Radar (HDF5/ODIM)](#radar-hdf5odim).

GRIB2 and radar behave differently from the static NetCDF grids (see their
sections). List the gridded collections with:

```python
import foehn

[d["dataset"] for d in foehn.list_datasets() if d["format"] in ("NetCDF", "GRIB2", "HDF5")]
```

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
> then read lazily from the local copy. On later calls foehn checks the current
> STAC asset metadata: an unchanged asset reuses the cache and a newer `updated`
> timestamp refreshes it. If that metadata check is temporarily offline, an
> existing complete local file remains usable. There is no byte-range read of
> the remote file. Use `match=` to avoid pulling parameters you don't need.

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
| `dataset` | `str` | Grid dataset name (NetCDF, GRIB2, or HDF5/radar collection) |
| `variables` | `str` or `list[str]` | Restrict to these data variable(s) |
| `match` | `str` | Keep only source files whose name contains this substring (required for GRIB2 and radar, where it must select one file) |
| `data_dir` | `str` or `Path` | Root data directory (default `./data/meteoswiss`) |

---

## Forecasts (GRIB2)

ICON-CH1/CH2 forecasts and KENDA analysis are GRIB2 (read via cfgrib) and behave
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

- **Native unstructured grid, with lat/lon joined.** ICON runs on an icosahedral
  grid, so cfgrib returns a 1-D `values` dimension (~1.1M cells). foehn fetches
  the collection's `horizontal_constants_*.grib2` (a collection-level asset,
  cached once) and attaches the cell-centre **`lat`/`lon`** as coordinates, so
  the field is geo-referenced even though it's not a regular grid:

  ```python
  ds = foehn.open_dataset("forecast_icon_ch1", match="202605231500-0-t_2m-ctrl")
  # cfgrib names the variable from the GRIB, not the filename token — check ds.data_vars
  ds["t2m"]           # dims (values,), with ds.lat / ds.lon (degrees) attached
  ```

  (Best-effort: if the constants file can't be reached, the field still opens,
  just without `lat`/`lon`, and a warning is emitted.)

- `open_dataset` reads a single field. Writing one field to Zarr works the same
  way (and inherits the single-file `match` requirement):

  ```python
  foehn.to_zarr("forecast_icon_ch1", match="202605231500-0-t_2m-ctrl")
  # -> data/meteoswiss/zarr/forecast_icon_ch1__202605231500_0_t_2m_ctrl.zarr
  ```

### Cubing forecasts (`stack=True`)

To assemble a *cube* instead of one field, use `to_zarr(..., stack=True)` with
a broader `match`. It opens every matched file, promotes whichever forecast axes
**vary** (`number` / `time` / `step`) into dimensions, and merges them with
`combine_by_coords` — so e.g. all lead times of one variable+member across runs
becomes a `(time, step, values)` cube (carrying the joined `lat`/`lon`):

```python
store = foehn.to_zarr("forecast_icon_ch1", match="-t_2m-ctrl", stack=True)
cube = xarray.open_zarr(store)        # dims e.g. (time, step, values)
cube["t2m"].isel(time=-1)             # latest run, all lead times (var named by cfgrib)
```

`stack=True` works for **any** gridded format — each dataset kind carries its own
cube builder: the GRIB2 combine above, radar's incremental time-stack (below), or,
for NetCDF, none at all (a multi-file `match` already combines on read, so `stack`
is a no-op there). Unlike
radar's incremental path, the GRIB2 combine loads the whole matched set into
memory at once, so it's **capped at 1000 files** — narrow `match` if you hit
that. A cloud-lazy kerchunk path (to avoid the in-memory load) remains future work.

---

## Radar (HDF5/ODIM)

`radar_precip` (CombiPrecip, gauge-adjusted precipitation) and `radar_hail`
(probability of hail) are ODIM-H5 **Cartesian composites** (`object=COMP`) — a
single 2-D grid per file on the Swiss projection. They are *not* polar radar
volumes, so `xradar` doesn't apply; foehn reads them with a small h5py-based
reader (h5py + pyproj, part of the `grids` extra).

```python
ds = foehn.open_dataset("radar_precip", match="cpc2613000000")
ds["acrr"]  # accumulated rainfall (mm), dims (y, x) on Swiss LV95
```

- **`match` is required and must select a single file**, like GRIB2 — there's one
  composite per ~5-minute timestep, so a collection is thousands of files. Match
  a filename prefix (e.g. `cpc<timestamp>` for precip, `bzc<timestamp>` for hail).
  An over-broad or missing match raises **before downloading**.
- The reader applies the ODIM `gain`/`offset` scaling, maps `nodata` (outside
  radar coverage) to `NaN` and `undetect` (nothing detected) to `0`, and derives
  **Swiss LV95 `x`/`y` coordinates** (metres, EPSG:2056) from the file's
  projection metadata via `pyproj` — so radar lines up with the NetCDF Swiss
  grids and `.sel(x=…, y=…, method="nearest")` works. The variable is named by
  the ODIM quantity (`acrr`, `poh`).

### Stacking a time series

`open_dataset` reads one timestep. To assemble a whole day/range into a single
**`(time, y, x)`** Zarr cube, use `to_zarr(..., stack=True)` with a `match` that
selects the timesteps:

```python
store = foehn.to_zarr("radar_precip", match="cpc26130", stack=True)
cube = xarray.open_zarr(store)        # dims (time, y, x)
cube["acrr"].sel(x=2_600_000, y=1_200_000, method="nearest")  # rain time-series at a point
```

The cube is written **incrementally** — one timestep appended at a time along
`time` — so it stays dask-free and peak memory is a single file regardless of how
many timesteps the match spans.

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
foehn.to_zarr("climate_normals_grid")
# -> data/meteoswiss/zarr/climate_normals_grid.zarr

# Override the location explicitly
foehn.to_zarr("surface_derived_grid", match="rhiresd", store="out/rain.zarr")

# Re-chunk before writing (requires `pip install dask`)
foehn.to_zarr("surface_derived_grid", match="rhiresd", rechunk={"time": 24})
```

Non-CF time axes (MeteoSwiss climate normals label theirs `years since
1991-01-01`, which CF decoding rejects) are sanitised on write so the resulting
store always re-opens cleanly with `xarray.open_zarr()`.

With the default `mode="w"`, the complete replacement store is staged beside its
destination and published only after a successful write. If opening, cubing,
rechunking, or writing fails, an existing complete store is preserved.

`mode="a"` writes directly into the existing store. This keeps append work and
temporary disk usage proportional to the new data instead of copying the entire
store first. Like other in-place append APIs, an interrupted append is not rolled
back automatically.

An append re-reads the listing for its `match`, and that listing is cumulative —
it returns everything published under the match, not only what is new. Timesteps
the store already holds are therefore skipped rather than written a second time,
so appending the same window twice is a no-op instead of a cube with duplicate
entries on its time axis. `mode="a"` against a destination that does not exist
yet simply creates it.

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

The MCP server exposes the inspect path as the `describe_grid` tool (any gridded
format): it returns a grid's dimensions, coordinates, and variables without
streaming array values into the LLM context (it may refresh the source cache when
STAC reports a newer asset — the same download-then-lazy caveat, hence it's
annotated `read_only_hint=False`).
Writing Zarr stores (`to_zarr`) is intentionally not exposed over MCP — conversion/
write tools aren't part of the MCP surface. See the [MCP server docs](mcp-server.md).
