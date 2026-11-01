# CLI Reference

The CLI uses subcommands that mirror the Python API. Run `foehn --help`, or
`foehn <command> --help` for one command's flags and examples. `python -m foehn`
is the same command line.

```text
foehn [GLOBAL OPTIONS] <COMMAND> [OPTIONS]
```

| Command | What it does |
|---|---|
| [`list`](#foehn-list) | List the available datasets |
| [`load`](#foehn-load-dataset) | Load a dataset from the API and print it, or write it to a file |
| [`metadata`](#foehn-metadata-table-dataset) | Show a dataset's parameters, stations, forecast points or inventory |
| [`download`](#foehn-download-dataset) | Download datasets to disk and convert them to Parquet |
| [`to-parquet`](#foehn-to-parquet-dataset) | Convert downloaded CSVs to Parquet |
| [`open`](#foehn-open-dataset) | Open a gridded dataset and print its xarray summary |
| [`to-zarr`](#foehn-to-zarr-dataset) | Write a gridded dataset to a Zarr store |
| [`mcp`](#foehn-mcp) | Start the MCP server |
| [`version`](#foehn-version) | Show foehn's version, the platform and every dependency's version |

---

## Global options

Accepted before or after the subcommand (`foehn -q download` and
`foehn download -q` are the same).

| Flag | Description |
|---|---|
| `-q`, `--quiet` | Print no progress. Repeat (`-qq`) to silence warnings too; errors always print |
| `-v`, `--verbose` | Print debug detail from the library |
| `--color WHEN` | Colour diagnostics: `auto` (default — only on a terminal), `always`, or `never` |
| `-V`, `--version` | Print `foehn <version>` and exit (before the subcommand only) |
| `-h`, `--help` | Show help and exit |

## Output and exit status

**Data goes to stdout, diagnostics to stderr.** Tables, CSV/JSON output and the
version report are written to stdout; progress, warnings, errors and the
library's log go to stderr. So redirecting stdout captures only the data:

```bash
foehn load smn --station BER --frequency d --output-format csv > ber.csv
foehn metadata stations smn --output-format json | jq '.[].name'
```

An error is one line on stderr, with a hint where there is one:

```text
$ foehn download smm
foehn: error: unknown dataset 'smm'
  hint: did you mean 'smn'?
```

| Status | Meaning |
|---|---|
| `0` | Success |
| `1` | The command failed: bad input, an unreachable API, or a download/conversion that only partly succeeded |
| `2` | Usage error: an unknown flag, a missing argument, or an invalid value such as `--workers 0` |
| `130` | Interrupted (Ctrl-C) |
| `141` | stdout was closed by its reader, e.g. `foehn metadata inventory smn \| head` |

---

## `foehn list`

List all available datasets, grouped by MeteoSwiss category.

```bash
foehn list
foehn list --category A
foehn list --format GRIB2 --output-format json
```

| Flag | Description |
|---|---|
| `--category`, `-c` | Only datasets in this category (`A`, `C`, `D`, `E`) |
| `--format`, `-f` | Only datasets in this file format (`CSV`, `TXT`, `NetCDF`, `GRIB2`, `HDF5`) |
| `--output-format` | `table` (default), `json` (the `foehn.list_datasets()` rows) or `csv` |
| `--data-dir PATH` | Accepted for symmetry with the other commands; listing reads nothing from disk |

---

## `foehn load DATASET`

Load a dataset from the API and print a preview. No files are written to disk
unless you pass `--output`.

```bash
foehn load smn --station BER --frequency d
foehn load smn --station BER ZUR --frequency d h -n 50
foehn load smn --station BER --frequency d --year 2026 --month 1 2 3
foehn load smn --station BER --frequency d --date-from 2025-06-01 --date-to 2025-08-31
foehn load smn --station BER --frequency d --columns tre200d0 rre150d0 --sort desc
foehn load smn --station BER --frequency d --output-format csv > ber.csv
foehn load smn --station BER --frequency d --time-slice historical -o ber.parquet
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
| `--limit` | Cap the loaded DataFrame to N rows, after sorting. Bounds the result, not the bytes downloaded |
| `--workers` | Concurrent CSV downloads; must be positive (default: 8) |
| `-n` | Rows to show in the table preview (default: 20). Ignored by `--output-format` and `--output`, which write every row |
| `--output-format` | `table` (default, a preview), or every row as `csv`, `json` (one array of objects) or `ndjson` |
| `--output`, `-o` `PATH` | Write every row to `PATH` instead of printing. The suffix picks the format: `.parquet`, `.csv`, `.json`, `.ndjson` or `.jsonl` |

---

## `foehn metadata TABLE DATASET`

Show dataset metadata fetched live from the API. `TABLE` is one of `parameters`, `stations`, `points`, or `inventory`.

```bash
foehn metadata parameters smn   # what each column name means
foehn metadata stations smn     # station locations and info
foehn metadata inventory smn    # which station has which parameter
foehn metadata points forecast_local  # forecast points: stations, postal codes, ...
foehn metadata stations smn --output-format csv > stations.csv
```

| Flag | Description |
|---|---|
| `--output-format` | `table` (default), `csv`, `json` or `ndjson` |

---

## `foehn download [DATASET...]`

Download datasets. Without arguments, downloads every non-grid dataset. Specify
one or more datasets to download only those; a named grid dataset works without
`--grids`.

A GRIB2 forecast (`forecast_icon_ch1`, `forecast_icon_ch2`, `analysis_kenda_ch1`)
downloads the control member of its newest complete run: about 3,500 files and
8 GB for ICON-CH1, 7 GB for ICON-CH2, and under 100 MB for KENDA. A run is
published lead by lead over more than an hour, so one still arriving is passed
over for the run before it. The ensemble members are left out (a whole ICON run
is 85–145 GB); fetch one with `foehn open --match`.

```bash
foehn download              # all non-grid datasets
foehn download smn pollen   # specific datasets only
foehn download smn --all    # historical + recent + now
```

| Flag | Description |
|---|---|
| `--historical` | Include historical time slice |
| `--now` | Include realtime "now" time slice |
| `--all` | Include all time slices (historical + recent + now) |
| `--full-refresh` | Ignore incremental tracking and re-download everything, without sending stored ETags or trusting local files — repairs a damaged cache |
| `--grids` | Include NetCDF, GRIB2, and HDF5/radar datasets when no datasets are named |
| `--no-parquet` | Skip CSV to Parquet conversion |
| `--workers` | Concurrent downloads per dataset; must be positive (default: 8) |
| `--data-dir PATH` | Output root (default: `$FOEHN_DATA_DIR`, else `./data/meteoswiss`) |

---

## `foehn to-parquet [DATASET...]`

Convert downloaded tabular files to Parquet. Without arguments, visits all
non-grid datasets and converts the kinds that publish a Parquet representation.

```bash
foehn to-parquet            # all collections
foehn to-parquet smn        # single dataset
```

| Flag | Description |
|---|---|
| `--data-dir PATH` | Root data directory |

---

## `foehn open DATASET`

Open a gridded dataset and print its xarray summary. All gridded formats need
`pip install "foehn[grids]"`. GRIB2 and radar additionally require a `--match`
that selects a single file (those collections are thousands of files). See the
[gridded data documentation](grids.md).

```bash
foehn open surface_derived_grid --match rhiresd
foehn open climate_scenarios_grid --match _pr_ --variables pr
foehn open forecast_icon_ch1 --match YYYYMMDDHHMM-0-t_2m-ctrl   # one GRIB2 field
foehn open radar_precip --match cpcYYDDDHHMM                   # one radar composite
```

| Flag | Description |
|---|---|
| `--variables` | Restrict to these data variable(s) |
| `--match` | Keep only source files whose name contains this substring (required for GRIB2/radar, selects 1 file) |
| `--data-dir PATH` | Root data directory; source files are cached under its `bronze/` |

---

## `foehn to-zarr DATASET`

Write a gridded (NetCDF, GRIB2, or HDF5/radar) dataset to a Zarr store under
`<data_dir>/zarr/`. The default name encodes `--match` (`<dataset>__<match>.zarr`),
so different slices don't overwrite each other; use `--out` for an explicit path.
(GRIB2 and radar collections require `--match`, like `foehn open`.)

```bash
foehn to-zarr surface_derived_grid --match rhiresd
foehn to-zarr surface_derived_grid --match rhiresd --out out/rain.zarr
foehn to-zarr forecast_icon_ch1 --match YYYYMMDDHHMM-0-t_2m-ctrl
foehn to-zarr radar_precip --match cpcYYDDDHHMM              # one timestep
foehn to-zarr radar_precip --match cpcYYDDD --stack           # a day -> one (time, y, x) cube
foehn to-zarr forecast_icon_ch1 --match -t_2m-ctrl --stack    # GRIB2 -> (time, step, values) cube
```

| Flag | Description |
|---|---|
| `--variables` | Restrict to these data variable(s) |
| `--match` | Keep only source files whose name contains this substring |
| `--out` | Explicit output path for the .zarr store |
| `--stack` | Cube the matched files, by whichever method the dataset's kind uses (radar → time,y,x; GRIB2 → N-D). Takes no value; the v0.4 spellings `--stack auto` and `--stack time` are still accepted and select nothing the kind does not already decide |
| `--data-dir PATH` | Root data directory; the default store lands under its `zarr/` |

---

## `foehn mcp`

Start the [MCP server](mcp-server.md). Needs the `mcp` extra
(`pip install "foehn[mcp]"`). Under the default stdio transport, stdout carries
the protocol, so everything foehn logs goes to stderr.

| Flag | Description |
|---|---|
| `--transport` | `stdio` (default), `streamable-http`, or `sse` (deprecated by MCP in favour of `streamable-http`) |

---

## `foehn version`

Print foehn's version, the Python version and platform, where the data
directory resolves to, and the installed version of every dependency foehn can
use. Paste it into bug reports. `foehn.show_versions()` prints the same from
Python; `foehn --version` prints only the version.

```bash
foehn version
foehn version --output-format json
```

| Flag | Description |
|---|---|
| `--output-format` | `table` (default, the plain-text report) or `json` |

---

## Environment variables

Settings can also be configured via environment variables. CLI flags always take precedence.

| Variable | Equivalent | Description |
|---|---|---|
| `FOEHN_DATA_DIR` | `--data-dir` | Root data directory. Also honoured by the Python API, which uses it whenever `data_dir=` is not passed. |
| `FOEHN_FULL_REFRESH` | `--full-refresh` | Set to `1`, `true`, or `yes` to ignore incremental tracking |
| `NO_COLOR` | `--color never` | Any non-empty value turns colour off ([no-color.org](https://no-color.org)) |
| `FORCE_COLOR` | `--color always` | Any non-empty value turns colour on, even when stderr is not a terminal. `NO_COLOR` wins if both are set |
