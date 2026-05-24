# CLI Reference

The CLI uses subcommands that mirror the Python API.

---

## `foehn list`

List all available datasets.

```bash
foehn list
```

---

## `foehn download [DATASET...]`

Download datasets. Without arguments, downloads all CSV collections. Specify one or more datasets to download specific ones.

```bash
foehn download              # all CSV collections
foehn download smn pollen   # specific datasets only
```

| Flag | Description |
|---|---|
| `--historical` | Include historical time slice |
| `--now` | Include realtime "now" time slice |
| `--all` | Include all time slices (historical + recent + now) |
| `--full-refresh` | Ignore incremental tracking, re-download everything |
| `--grids` | Include grid/binary datasets (GRIB2, NetCDF) |
| `--no-parquet` | Skip CSV to Parquet conversion |
| `--data-dir PATH` | Output root (default: `./data/meteoswiss`) |

---

## `foehn to-parquet [DATASET...]`

Convert downloaded CSVs to Parquet. Without arguments, converts all collections.

```bash
foehn to-parquet            # all collections
foehn to-parquet smn        # single dataset
```

| Flag | Description |
|---|---|
| `--data-dir PATH` | Root data directory |

---

## `foehn metadata KIND DATASET`

Show dataset metadata fetched live from the API. `KIND` is one of `parameters`, `stations`, or `inventory`.

```bash
foehn metadata parameters smn   # what each column name means
foehn metadata stations smn     # station locations and info
foehn metadata inventory smn    # which station has which parameter
```

---

## `foehn load DATASET`

Load a dataset from the API and print a preview. No files are written to disk.

```bash
foehn load smn --station BER --frequency d
foehn load smn --station BER ZUR --frequency d h -n 50
foehn load smn --station BER --frequency d --year 2026 --month 1 2 3
foehn load smn --station BER --frequency d --date-from 2025-06-01 --date-to 2025-08-31
foehn load smn --station BER --frequency d --columns tre200d0 rre150d0 --sort desc
```

| Flag | Description |
|---|---|
| `--station` | Filter by station(s) |
| `--frequency` | Filter by frequency (t, h, d, m, y) |
| `--time-slice` | Time slices to include (default: recent) |
| `--year` | Filter by year(s) (e.g. 2025 2026) |
| `--month` | Filter by month(s) (1--12) |
| `--date-from` | Start date inclusive (YYYY-MM-DD) |
| `--date-to` | End date inclusive (YYYY-MM-DD) |
| `--columns` | Only return these columns |
| `--drop-null` | Drop rows where this column is null |
| `--sort` | Sort by timestamp: asc or desc |
| `-n` | Number of rows to show (default: 20) |

---

## `foehn open DATASET`

Open a gridded dataset and print its xarray summary. NetCDF grids need
`pip install "foehn[grids]"`; GRIB2 forecasts need `pip install "foehn[grib]"`
**and** a `--match` filter (a forecast collection is thousands of files). See the
[gridded data documentation](grids.md).

```bash
foehn open surface_derived_grid --match rhiresd
foehn open climate_scenarios_grid --match _pr_ --variables pr
foehn open forecast_icon_ch1 --match 202605231500-0-t_2m-ctrl   # one GRIB2 field (match must select 1 file)
```

| Flag | Description |
|---|---|
| `--variables` | Restrict to these data variable(s) |
| `--match` | Keep only source files whose name contains this substring (required for GRIB2) |

---

## `foehn to-zarr DATASET`

Write a gridded (NetCDF or GRIB2) dataset to a Zarr store under `<data_dir>/zarr/`.
The default name encodes `--match` (`<dataset>__<match>.zarr`), so different
slices don't overwrite each other; use `--out` for an explicit path. (GRIB2
collections require `--match`, like `foehn open`.)

```bash
foehn to-zarr surface_derived_grid --match rhiresd
foehn to-zarr surface_derived_grid --match rhiresd --out out/rain.zarr
foehn to-zarr forecast_icon_ch1 --match 202605231500-0-t_2m-ctrl
```

| Flag | Description |
|---|---|
| `--variables` | Restrict to these data variable(s) |
| `--match` | Keep only source files whose name contains this substring |
| `--out` | Explicit output path for the .zarr store |

---

## Environment variables

Settings can also be configured via environment variables. CLI flags always take precedence.

| Variable | Equivalent | Description |
|---|---|---|
| `FOEHN_DATA_DIR` | `--data-dir` | Root data directory |
| `FOEHN_FULL_REFRESH` | `--full-refresh` | Set to `1`, `true`, or `yes` to ignore incremental tracking |
