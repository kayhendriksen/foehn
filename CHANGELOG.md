# Changelog

All notable changes to foehn are recorded here, newest first. foehn follows
[Semantic Versioning](https://semver.org/); while it is pre-1.0, a minor release
(0.x.0) may contain breaking changes, and each one is listed under **Breaking
changes**. The same notes are published with each
[GitHub release](https://github.com/kayhendriksen/foehn/releases).

## Unreleased

### Breaking changes

- **The CLI writes progress and log output to stderr**, not stdout. stdout now
  carries only a command's output (tables, CSV/JSON, the version report), so
  `foehn load ... --output-format csv > file.csv` captures data and nothing
  else. Scripts that parsed progress lines such as `Bronze data saved to:`
  from stdout must read stderr instead.
- **The `dev` extra is gone.** Development tools now live in a PEP 735
  `[dependency-groups]` table, which uv installs by default (`uv sync`).
  `pip install "foehn[dev]"` no longer installs ruff, ty and pytest into a
  user's environment.

### New features

- **Machine-readable output.** `foehn list`, `foehn metadata` and `foehn load`
  take `--output-format` (`table`, `csv`, `json`, plus `ndjson` for the latter
  two), and `foehn load -o PATH` writes every row to a `.parquet`, `.csv`,
  `.json` or `.ndjson` file.
- **Global `-q`/`--quiet`, `-v`/`--verbose` and `--color` options**, accepted
  before or after the subcommand. Colour honours `NO_COLOR` and `FORCE_COLOR`.
- **`foehn version` and `foehn.show_versions()`** print foehn's version, the
  platform, where the data directory resolves to, and every dependency's
  version for bug reports. `foehn -V` is a short `--version`.
- **`python -m foehn`** runs the CLI.
- **Typo suggestions.** An unknown dataset names the closest one, from the
  Python API (`Did you mean 'smn'?`), the CLI (`hint: did you mean 'smn'?`) and
  the MCP tools alike. On Python 3.14+ a mistyped subcommand is suggested too.
- **A User-Agent header** (`foehn/<version> (+https://github.com/kayhendriksen/foehn)`)
  on every request, so the MeteoSwiss and CSCS operators can identify foehn's
  traffic.

### Bug fixes

- **`foehn mcp` could corrupt its own protocol stream.** The CLI's log handler
  wrote to stdout, which is the JSON-RPC channel under the stdio transport; a
  `describe_grid` call that downloaded a file wrote a progress line into it.
- **`foehn load -n N` showed at most ten rows.** Polars truncates any frame to
  ten rows when printed, so the preview showed five, an ellipsis and five more.
- **`foehn load -n 0` showed twenty rows** instead of none.
- **Piping into `head` crashed with a `BrokenPipeError` traceback** (e.g.
  `foehn metadata inventory smn | head`). It now stops quietly with status 141.
- **`foehn download --workers 0`** printed its header and created the data
  directory before failing. Non-positive `--workers` and negative `-n`/`--limit`
  are now usage errors (status 2) before anything runs.
- **Dependabot could not bump `ty` without breaking CI.** Its pip ecosystem
  edits `pyproject.toml` but not `uv.lock`, and CI installs with
  `uv sync --locked`. A `uv` ecosystem now bumps both together.
- **docs/cli.md** was missing `foehn load --limit` and `--workers`, the
  `foehn list` filters, and `--data-dir` for `open` and `to-zarr`. A test now
  fails when a flag is added without documentation.

### Project

- Documentation site (MkDocs Material) with an API reference generated from the
  docstrings, built in CI with `--strict`.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, issue forms, a pull
  request template and `.editorconfig`.
- This changelog, assembled from every GitHub release.

## 0.5.0 (2026-10-25)

An architecture release. Almost every module behind the public API was rebuilt
around a small set of explicit seams, and that work surfaced a long list of
correctness fixes — most visibly for local forecasts, GRIB2 lookups, radar
cubes, incremental downloads and Databricks ingestion. Python 3.15 is now
supported.

The documented `foehn.*` calls from 0.4.0 keep working, but a few results change
shape and some failures that used to pass silently now raise. Please read
**Breaking changes** before upgrading.

### Breaking changes

- **`forecast_local` returns one row per point and time** (#95, #96). A forecast
  run is ~32 files, one per parameter, each covering every point. They were
  stacked, so each row held a single parameter and one point-hour was spread
  over ~23 rows. They are now joined, with every parameter as a column and
  `point_id`, `point_type_id`, `reference_timestamp` as the key columns. The
  Parquet file and the Delta table have the same shape and hold only the newest
  run.

  | `load("forecast_local")` | Before | After |
  |---|---|---|
  | Rows | 29,655,770 | 1,280,620 |
  | Non-null parameters per row | 1 | all |

- **Incremental cursors are per dataset and time slice** (#97). `_last_run.json`
  held one Workspace-wide timestamp, so downloading `smn` made the next `pollen`
  or `--historical smn` run skip items that had never been downloaded. Each
  dataset now has its own cursor, which advances only when that dataset
  downloads and converts cleanly. Expect one full listing per dataset on the
  first run after upgrading; ETags still prevent unchanged files from
  downloading again.
- **`download(since=...)` must include a timezone** (#97). It is now compared as
  an instant rather than as text, so `2026-10-25T10:30:00Z` correctly sorts after
  `2026-10-25T11:00:00+01:00`. A `since` without a timezone raises `ValueError`.
- **Stricter validation.** An unknown `frequency=` or `sort=` raises instead of
  matching nothing (#56). `load(station=[])` and `load(frequency=[])` mean "no
  filter" rather than "match nothing" (#51). An unfiltered NetCDF read is capped
  at 1,000 files (#67); `surface_derived_grid` alone is ~3,700 files / 30 GB.
- **The Python API honours `$FOEHN_DATA_DIR`** (#57). Only the CLI read it, so
  `foehn.download()` without `data_dir=` wrote to `./data/meteoswiss` even when
  the variable was set. Both now resolve the argument, then
  `$FOEHN_DATA_DIR`, then `./data/meteoswiss`.
- **`list_datasets()` changes** (#49, #57). `climate_scenarios_indoor` reports
  format `CSV` (was `CSV+ZIP`), and `climate_normals` is listed and accepted by
  `foehn.download()`.
- **`scripts/ingest_delta.py` exits 1 when any table fails** (#97). Write
  failures were counted as skips and the job reported success over stale tables.
- **`foehn[databricks]` installs `polars[rtcompat]`** instead of `polars-lts-cpu`
  (#99). Both distributions ship the same `polars` package, so installing them
  side by side overwrote each other's files and could break `import polars`.
  `polars-lts-cpu` has also been frozen at 1.33.1 since September 2025.
- **Internal modules reorganised.** `foehn.stac` is removed (#48).
  `foehn.client` is a deprecated shim: `load_etags`, `save_etags`,
  `load_last_run` and `save_last_run` still work with a `DeprecationWarning`, and
  the old `download_*` helpers raise an error naming the `foehn.download(...)`
  call that replaces them (#57, #60, #67). `ensure_grid_files` moved from
  `foehn.grids` to `foehn.gridfiles` (#61).

### New features

- **Forecast points** (#95). `foehn.points("forecast_local")`, `foehn metadata
  points forecast_local` and the MCP tool `get_points` list the ~5,800 forecast
  points (stations, postal code centres and points of interest). The MCP tool
  searches by name, station abbreviation or postal code.
- **`station=` works for `forecast_local`** (#95). It accepts station
  abbreviations and postal codes. It previously compared against item IDs, which
  are dates, so it never matched anything.
- **`climate_normals` is a regular dataset** (#57), downloadable and convertible
  like every other.
- **`foehn.metadata(dataset, table)`** (#60, #67) for code that picks a metadata
  table at runtime.
- **Pollen's `now` time slice** (#46).
- **Python 3.15** (#95). Tested in CI as a blocking job.
- **Polars 2.0.** The full suite passes on Polars 2.0.0; the requirement stays
  `polars>=1.18`.

### Bug fixes

- **GRIB2 forecast lookups missed the fast path** (#95). A STAC item's
  `datetime` is the valid time (run + lead time), not the run, so any match with
  a lead time above 0 fell back to listing the whole collection. A match for an
  expired run now names the runs the collection currently holds, with a real
  filename to copy, and the documentation examples use filename patterns instead
  of dates that had long expired.
- **GRIB2 cubes failed for control plus ensemble files** (#97) with `Dimension
  number already exists`.
- **Radar cubes duplicated observations** (#97). When CombiPrecip's quality code
  changes, MeteoSwiss keeps the original file beside the reanalysis. Both were
  appended as identical timestamps, and a later append could restore the
  original values over the correction. foehn now keeps one file per observation,
  highest quality code first, and the time axis stays unique. Appending a
  cumulative match also skips timesteps already in the store (#67).
- **Concurrent downloads could pair a file with the wrong ETag** (#97), after
  which every refresh kept the stale file.
- **Files deleted from the cache were never fetched again** (#97, #98) as long
  as nothing upstream had changed since the cursor.
- **Failed refreshes published partial output** (#97). A `climate_scenarios`
  conversion or Delta `climate_normals` write with one unreadable source
  replaced a complete output with a partial one that later runs treated as up to
  date. Chunked Delta writes now build in a `<table>__foehn_staging` table and
  publish in a single overwrite.
- **Writes are atomic** (#44, #62). Parquet files, Zarr stores and state files
  are written to a staging path and moved into place, so a failure leaves the
  previous version rather than a truncated file with a fresh timestamp.
- **Grid caches could mix file generations** (#66, #67). A refresh that failed
  part-way is detected and repaired rather than handed back as complete.
- **Metadata tables are parsed with their declared types** (#97), so a postal
  code such as `3011` stays text and no longer breaks the MCP `get_points` tool.
- **Clearer errors for metadata** (#95). `stations("climate_normals")` explains
  that the dataset is a direct ZIP instead of returning a bare 404, and a table a
  dataset doesn't publish lists the ones it does.
- **Quieter ICON reads** (#95). cfgrib's `FutureWarning`s and its misleading
  "no latitudes/longitudes" notice no longer reach the caller.
- **Corrupt state files** (#97). `_etags.json`, `_last_run.json` and archive
  markers that aren't valid UTF-8 are treated as absent instead of crashing.
- **Also fixed:** `date_to`, `columns=` filtering and the ZIP-size guard (#44),
  and explicit UTF-8 for every text read, ahead of Python 3.15's new default
  (#42).

### Performance

| Operation | Before | After |
|---|---|---|
| ICON-CH1 field, match with run and lead time | ~188 s | 1–6 s |
| ICON-CH1 field, match without a run | ~188 s | 12–35 s |
| `forecast_local`, one station and one parameter | ~25 s | ~1.4 s |

The forecast listing is narrowed to one valid time when the match allows it,
and otherwise fetched in parallel hour-wide windows (#95). For `forecast_local`,
`columns=` now decides which parameter files are downloaded at all (#95). Also:
parallel grid downloads, zero-copy CSV decoding, filters pushed into the CSV
scan, column projection, and a listing cache whose lifetime adapts to how long
the listing took (#45, #47).

### Architecture

The internals are organised around explicit seams, each with a single owner.
`CONTEXT.md` defines the vocabulary and layering tests enforce the module graph.

- **One network port** (#48). `fetch.Fetcher` is the only code that talks HTTP.
  Tests use an in-memory fetcher instead of ~200 `patch()` calls.
- **One dataset table** (#49, #57, #60). Each dataset has a `DatasetKind`, and
  `registry` holds one row per kind with its download, convert, load and grid
  adapters. Six routing ladders are gone; adding a dataset is one row.
- **One engine per stage** (#54, #56, #61, #64). `transfer.fetch_all` replaces
  five copies of the download loop, `convert.run_conversions` four copies of the
  conversion bookkeeping, and `assets` replaces 16 hand-written STAC asset walks.
- **Upstream conventions in one place** (#58, #61). `meteocsv` owns MeteoSwiss's
  CSV rules, `odim` and `icon` the radar and ICON grid formats, and `metadata`
  the metadata tables.
- **Explicit state and publication rules** (#62, #66). `atomicwrite` (staging),
  `coherence` (grid-set consistency), `_locking` (portable locks), `state`
  (ETags and cursors), `workspace` (directory layout) and `archives` (ZIP bomb
  and zip-slip guards).

### Upgrade notes

- **`forecast_local.parquet`** is rebuilt in the new shape when the next run
  lands. Delete it to rebuild immediately.
- **Radar Zarr stores** built with `mode="a"` before this release may repeat
  timestamps. Extending one now raises `ValueError`; rebuild it with `mode="w"`
  (the default).
- **Environments with both `polars` and `polars-lts-cpu`** can be repaired with:

  ```bash
  pip uninstall polars-lts-cpu
  pip install --force-reinstall --no-deps polars
  ```

### Known limitations

- On Python 3.15, the core package and the `mcp` extra install normally, but the
  `grids` extra does not yet: h5py, eccodes and numcodecs have no 3.15 wheels.
- `forecast_local` without `columns=` still downloads the whole run (~32 files),
  because each file covers every point.
- MeteoSwiss keeps forecast files for about a day and radar files for about two
  weeks; foehn cannot read runs or timesteps that have rolled off.

### Maintenance

- Test suite grown from ~350 to 870 tests, with the branch-coverage floor raised
  from 87% to 97% (#61, #67).
- Every function in `src/` is type-annotated (foehn ships `py.typed`), `ty` is
  pinned at 0.0.91, and ruff enforces a much stricter rule set (#94, #98).
- CI tools are pinned and maintained by Dependabot, and the packaging smoke test
  uses the same `build`/`twine` versions as the publish workflow (#41, #50, #55).
- PyJWT, a transitive dependency of `mcp`, updated in the lockfile to clear its
  security advisories (#95).
- Dependency and GitHub Actions updates via Dependabot.

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.5.0)

## 0.4.0 (2026-08-28)

### Fixed

**`time_slice` was silently ignored for 10-minute and hourly data** (#39)

MeteoSwiss splits the high-frequency historical series into decade chunks
(`ogd-smn_ber_t_historical_2000-2009.csv`). foehn's filename parser read only the
trailing segment, saw `2000-2009`, and classified the file as having no time slice —
which meant "always include". Every query silently pulled the full history.

`load("smn", station="BER", time_slice="now")` returned 1,152,843 rows spanning
2004–2026 instead of the 123 rows covering one day, transferring 152 MB to deliver
16.8 KB. Affects `smn`, `smn_precip`, and `smn_tower` at `t` and `h` frequency, in
both `load()` and `download()`. Daily, monthly, and yearly data were unaffected.

**Behaviour change:** if you called `load(..., time_slice="recent")` or `"now"` on
`t`/`h` data, you were receiving the full history and will now get only the slice
you asked for. Pass `time_slice="historical"` to keep the old result.

### Changed

**Requires `mcp>=2.0`** (#35) — the MCP server migrated from `FastMCP` to `MCPServer`.
All seven tools keep their existing names and signatures, so MCP clients need no
changes, but `foehn[mcp]` no longer installs against `mcp` 1.x.

### Internal

CI hardening (#36): type checking with `ty`, workflow auditing with `zizmor`,
a packaging smoke test, an 87% branch-coverage floor, dependency auditing extended
to every extra, a CycloneDX SBOM on each release, and a weekly canary against the
live MeteoSwiss API.

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.4.0)

## 0.3.4 (2026-07-27)

A performance release: converting `climate_scenarios` no longer holds the whole
collection in memory, so peak RSS stays flat as MeteoSwiss publishes more files.

### Performance

- **`climate_scenarios` conversion is now lazy and streaming** (#33). This was
  the only converter left that materialised every file before writing — an eager
  `read_csv` per CSV, all the DataFrames collected in a list, then one
  `write_parquet`. (`smn`-style groups and `climate_scenarios_indoor` already
  streamed; `climate_normals` writes one Parquet per file, so it's bounded.) It
  now uses `scan_csv` + `sink_parquet` like the rest. Peak RSS, measured on
  synthetic files matching the published shape — 10,950 rows × 30 model columns:

  | Input | Before | After |
  |---|---|---|
  | 60 MB / 30 files | 383 MB | 218 MB |
  | 238 MB / 120 files | 944 MB | 252 MB |

  The output Parquet is byte-identical: same schema, column order and rows.

  What kept this eager was the variable-length `KEY;VALUE` metadata preamble
  these files carry ahead of the real `DATE;<model>;...` header — the converter
  read each file into a string just to slice the preamble off. foehn now counts
  the preamble by reading only the head of the file and passes the offset to
  `scan_csv(skip_lines=...)`. That's `skip_lines`, not `skip_rows`: the latter
  skips by CSV parsing rules, so an unbalanced quote in a metadata value
  (`VARIABLE;Daily precipitation ("mm")`) shifts the header offset and the read
  fails with `NoDataError: empty CSV`.

### Requirements

- **polars >= 1.18** (was >= 1.0), for `scan_csv(skip_lines=...)`. Same floor for
  the `databricks` extra's `polars-lts-cpu`.

### Notes

- A `climate_scenarios` file that can't be read at all — no `DATE;` header, or a
  filename that doesn't carry station/variable/gwl — is still skipped and counted
  on its own, as before. Dtype errors, however, now surface when the stream is
  written rather than per file, so one bad file fails the whole Parquet instead
  of just itself. These files are ~10,950 rows, well inside the 20,000-row schema
  inference window, and `diagonal_relaxed` reconciles integer and float columns
  across files, so this shouldn't bite in practice.

### Maintenance

- CI action bumps (#32): `actions/checkout` 7.0.0 → 7.0.1,
  `github/codeql-action` 4.37.1 → 4.37.3 (init/analyze/upload-sarif),
  `ossf/scorecard-action` 2.4.3 → 2.4.4.

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.3.4)

## 0.3.3 (2026-07-21)

A correctness release: local forecasts actually return data, and ten dataset
keys that no longer exist upstream are gone.

### Bug fixes

- **`forecast_local` returned nothing.** A STAC item in this collection is one
  *day* — holding that day's ~24 hourly runs — not a single forecast. foehn kept
  only the latest item, but MeteoSwiss creates the newest day around 04:00 UTC
  and fills it as runs publish, so it was routinely empty. Both `download` and
  `load()` now select the newest *run* from the asset filenames, which is the
  unit MeteoSwiss actually publishes (one file per parameter, all ~5,600 points).
  Item selection also ranked on `properties.datetime`, a refresh timestamp
  identical across items to the microsecond — it ordered items by accident.
  Thanks to @Joelius300 for the report (#27, #30).

### Breaking changes

- **Ten retired collections removed** (#29). The `hagelgefaehrdung-korngroesse_*`
  and `klimanormwerte-*` collections now 404 on data.geo.admin.ch. Their data was
  folded into two collections that foehn already supports, so nothing is lost —
  but the old keys are gone from `COLLECTIONS`, `COLLECTION_META` and
  `NETCDF_COLLECTIONS`:

  | Removed | Now available as |
  |---|---|
  | `hail_hazard_{10,20,50,100}y` | `radar_derived_grid`, `archive-ch` item, `match=returnperiod{002..100}y{leha1,meshs}` |
  | `climate_normals_{temp,precip,sun}_{6190,9120}` | `climate_normals_grid`, `ch` item, `match={t,r,s}norm{m,y}{9120,6190}` |

  Both replacements are supersets — the hail grids gained 2/5/30/70-year return
  periods and a MESHS variant.

### Known limitations

- One `forecast_local` run is ~1 GB (32 parameters × ~30 MB). There is no
  per-parameter filter yet, so `download forecast_local` fetches all 32.
- MeteoSwiss retains only ~4 days of local forecasts and publishes no archive,
  so historical point forecasts can only be built by downloading on a schedule.

### Maintenance

- CI action bumps: `actions/checkout` 6.0.3 → 7.0.0, `actions/setup-python`, and
  a grouped update across 5 actions.
- Version is now declared consistently across `pyproject.toml`, `server.json` and
  `foehn.__version__`, with a test tying them together — `__version__` is not
  covered by the publish workflow's tag check and could previously drift (#31).

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.3.3)

## 0.3.2 (2026-06-10)

A reliability and security release — no new datasets, but downloads, parsing,
and the release pipeline are all hardened.

### Bug fixes

- **Fixed an infinite loop in CSV parsing.** The dtype-recovery fallback in
  `parse_csv_bytes()` retried forever when a column failed to parse even as
  Float64 (e.g. a stray non-numeric value past the schema-inference window).
  Reachable from `foehn.load()` and the MCP server's `load_data` — it now
  raises a clear error instead of hanging.
- **Climate normals (C6) recover from interrupted runs.** The skip check now
  looks at the extracted TXT files rather than the ZIP, so a run that died
  between download and extraction is retried instead of silently skipped
  forever.
- **State files can no longer brick the pipeline.** `_etags.json` and
  `_last_run.json` are written atomically, and a corrupt state file is treated
  as empty (with a warning) instead of crashing every subsequent run.

### Reliability

- All downloads are now atomic: CSVs and ZIPs join the binary assets in
  streaming to a `.part` file and renaming on completion (#21 + this release),
  so an interrupted transfer never leaves a truncated file behind.
- STAC listing and pagination use retrying HTTP sessions, and retries now also
  cover 429 rate limits (honouring `Retry-After`).
- The ETag store is pruned of stale entries on clean full runs — it no longer
  grows forever as forecast assets cycle.
- CSV assets with query strings (e.g. `?token=...`) are detected correctly, and
  time slices are parsed from the trailing filename segment so a coincidental
  "now" elsewhere in a URL can't be misread (#21).
- The library logs through standard `logging` (`foehn.*`, silent when imported);
  the CLI attaches its own stdout handler (#21).
- CSV decoding is total: the Windows-1252 fallback replaces unmappable bytes
  instead of raising.

### API

- `foehn.download()` gained a `force=` flag to re-download ZIP-shipped datasets
  (e.g. `climate_scenarios_indoor`) that would otherwise skip when already
  extracted:

  ```python
  import foehn

  foehn.download("climate_scenarios_indoor", force=True)
  ```

- `list_datasets()` no longer advertises frequencies for datasets where the
  `frequency` filter isn't supported (`forecast_local`, `climate_scenarios`,
  `climate_scenarios_indoor`) — the granularity is named in the description
  instead.
- All download functions return a `DownloadResult` summary (counts + new
  filenames) so callers can gate downstream work (#21).

### Security

- ZIP extraction now guards against decompression bombs (10 GiB declared-size
  cap) on top of the existing path-traversal checks — including the in-memory
  indoor-scenarios archive.
- The Databricks ingest escapes backslashes as well as quotes when setting
  column comments via Spark SQL.
- Release pipeline hardening: `mcp-publisher` is pinned by version and SHA-256,
  PyPI uploads explicitly enable PEP 740 attestations, build tooling is pinned,
  CodeQL runs the `security-extended` suite, and all workflow checkouts use
  `persist-credentials: false`.

**Full changelog**: https://github.com/kayhendriksen/foehn/compare/v0.3.1...v0.3.2

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.3.2)

## 0.3.1 (2026-06-02)

### New datasets

MeteoSwiss recently published their spatial climate analyses as open data. The
ground-based (`surface_derived_grid`) and satellite-based (`satellite_derived_grid`)
grids were already integrated in foehn — this release adds the remaining two:

- **`radar_derived_grid`** (C5) — radar-derived hail climate grids: hail-day counts (≥2 cm, ≥4 cm) and MESHS/LEHA return-period maps (2002 onward, summer half-year).
- **`climate_normals_grid`** (C7) — unified gridded climate normals: temperature, precipitation, and sunshine for 1961–1990 and 1991–2020, plus radiation and cloud averages for 2004–2020 (monthly and yearly).

Both are NetCDF and work with the existing `open_dataset()` / `to_zarr()` grid API:

```python
import foehn

foehn.open_dataset("climate_normals_grid", match="tnormy9120")  # yearly temp normal 1991–2020
foehn.open_dataset("radar_derived_grid", match="haildaysy")  # yearly hail days
```

### Docs

- README now surfaces gridded data (xarray/Zarr) outputs alongside the tabular ones.
- Collection and grids docs updated for the new datasets.

The existing `climate_normals_*` collections are unchanged and continue to work.

**Full changelog**: https://github.com/kayhendriksen/foehn/compare/v0.3.0...v0.3.1

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.3.1)

## 0.3.0 (2026-05-26)

Gridded MeteoSwiss data (NetCDF, GRIB2, radar) now opens as xarray Datasets, alongside new climate-scenario tabular collections. **Minimum Python is now 3.11.**

### Added

**Gridded data — the grid analog of the tabular API**
- `foehn.open_dataset(dataset, match=…, variables=…)` — open a collection as an xarray `Dataset` (mirrors `load()`).
- `foehn.to_zarr(…)` — materialise to a Zarr store (mirrors `to_parquet()`). `stack="auto"` assembles a matched set into one cube: radar → `(time, y, x)` (incremental, dask-free), GRIB2 → N-D over the varying number/time/step axes.
- **Formats:** NetCDF (climate grids, normals, scenarios), GRIB2 forecasts (ICON-CH1/CH2, KENDA — cell `lat`/`lon` joined from the horizontal-constants file), and HDF5/ODIM radar composites (CombiPrecip, hail) via a native reader, georeferenced to Swiss LV95.
- **CLI:** `foehn open` and `foehn to-zarr`.
- **MCP:** new `describe_grid` tool — inspect a grid's dimensions, coordinates, and variables without downloading array values.
- One optional extra covers it all: `pip install "foehn[grids]"`.

**New tabular collections**
- **Indoor climate scenarios** (CSV+ZIP) and **CH2025 climate scenarios** (metadata-preamble CSV) now load through `foehn.load()`.
- `forecast_local` now exposes a derived `reference_timestamp`.

### Breaking
- **Dropped Python 3.10** — minimum is now **3.11** (the `grids` extra needs zarr 3, which requires 3.11+).

### Install
```
pip install --upgrade foehn
pip install "foehn[grids]"   # xarray/Zarr for NetCDF + GRIB2 + radar
```

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.3.0)

## 0.2.23 (2026-05-21)

### Features

- **Added the KENDA-CH1 numerical weather analysis collection** (`analysis_kenda_ch1`). 1 km gridded analysis from MeteoSwiss's data assimilation system (temperature, humidity, wind, pressure, radiation, pollen, and more), category E5, GRIB2. Download with `foehn download --grids analysis_kenda_ch1`. ([#13](https://github.com/kayhendriksen/foehn/pull/13))

**Full changelog**: https://github.com/kayhendriksen/foehn/compare/v0.2.22...v0.2.23

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.23)

## 0.2.22 (2026-05-09)

### Features

- **`stations()` now returns Swiss LV95 coordinates** alongside WGS84. New columns `lv95_east` and `lv95_north` (EPSG:2056) are added to the DataFrame returned by `foehn.stations(dataset)`. The MCP `Station` model exposes the same fields. ([#9](https://github.com/kayhendriksen/foehn/pull/9))

### Behavior changes (potentially breaking)

- **`data_since` is no longer reformatted**: it is now kept in MeteoSwiss source format (`DD.MM.YYYY`) instead of being converted to `YYYY-MM-DD`. If you parse this column downstream, update your parser. ([#9](https://github.com/kayhendriksen/foehn/pull/9))

### Fixes

- **Corrected MeteoSwiss collection metadata** ([#7](https://github.com/kayhendriksen/foehn/pull/7)):
  - Added missing yearly frequency (`y`) to several SMN, climate, and pollen collections (`smn`, `smn_precip`, `smn_tower`, `obs`, `pollen`, `nbcn`, `nbcn_precip`).
  - Removed unsupported time slices on collections that don't actually publish them.
  - Fixed file format for two collections that were tagged as `GRIB2` but are actually `HDF5`.

### Internal / refactor

- Moved trusted-domain validation for STAC and download URLs into a shared `foehn._urls` helper (used by `client`, `stac`, and `api`).
- Added a `live` pytest marker (default deselected) for opt-in tests that hit the live MeteoSwiss STAC/S3. Run them with `pytest -m live`.

### Security & supply chain

- Added [OpenSSF Scorecard](https://github.com/ossf/scorecard) workflow + badge ([#3](https://github.com/kayhendriksen/foehn/pull/3), [#5](https://github.com/kayhendriksen/foehn/pull/5)).
- Improved security policy links ([#6](https://github.com/kayhendriksen/foehn/pull/6)).

### Dependencies & CI

- Bumped `github/codeql-action` and other workflow actions ([#4](https://github.com/kayhendriksen/foehn/pull/4), [#8](https://github.com/kayhendriksen/foehn/pull/8)).

### Tooling

- Added `.databricks/` and `.claude/` to `.gitignore` (local tool state).

**Full changelog**: https://github.com/kayhendriksen/foehn/compare/v0.2.21...v0.2.22

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.22)

## 0.2.21 (2026-04-25)

This release focuses on making downloads and conversions more reliable for real-world MeteoSwiss workflows. It improves parallel loading, fixes failure handling around Parquet conversion, strengthens thread safety for HTTP downloads, and tidies the MCP/Databricks packaging surface.

### Highlights

- Safer parallel downloads with thread-local HTTP sessions.
- Faster `load()` calls through parallel CSV fetching.
- More reliable CLI and Python API behavior when Parquet conversion fails.
- Improved incremental download safety so failed conversions do not advance the last-run cursor.
- MCP server metadata and development dependencies are now aligned.
- Databricks configuration/docs were cleaned up for `DATABRICKS_HOST`-based setup.

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.21)

## 0.2.20 (2026-04-18)

**Fix:** `to_parquet` no longer OOMs on large groups (historical slices, `climate_scenarios`).

**How:** Switched the per-group concat from in-memory (`read_csv` + `pl.concat` + `write_parquet`) to streaming (`scan_csv` + `concat(diagonal_relaxed)` + `sink_parquet`). Dtype drift is handled in-place — on the polars error we force the offending column to `Float64` and retry the sink.

**Result on full SMN (~12 GB CSV, 10 groups):** peak RSS 487 MB (was OOM at multi-GB on the Databricks driver).

No API change.

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.20)

## 0.2.19 (2026-04-17)

#### New

- **`DownloadResult` return type** — All `download_*` functions now return a
  `DownloadResult` dataclass with `total_assets`, `downloaded`, `skipped`, and
  `filenames`. Use `result.downloaded > 0` to gate expensive downstream work
  (e.g. Spark `MERGE INTO`) without scanning the bronze directory.
- **`since` filter for `download_netcdf`** — NetCDF/GeoTIFF/ZIP collections now
  support incremental updates, matching `download_collection` and
  `download_grib2`.
- **Exported `DownloadResult`** from the top-level `foehn` package.

#### Changed

- `foehn.download()` now returns `DownloadResult` (previously `None`), combining
  metadata + collection counts.
- Connection pool size in `_retry_session` now scales with `workers`, avoiding
  urllib3 "connection pool is full" warnings on high-concurrency downloads.

#### Migration

```python
### Before
foehn.download("smn")

### After — gate downstream work on actual changes
result = foehn.download("smn")
if result.downloaded:
    run_spark_merge(result.filenames)
```

No breaking changes — callers ignoring the return value continue to work.

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.19)

## 0.2.18 (2026-04-17)

Parallel downloads

Downloads are now fetched concurrently per dataset using a thread pool. A collection with hundreds of station CSVs (e.g. `smn`) is significantly faster, while staying polite to the MeteoSwiss / CSCS CDNs.

### Changes
- Per-asset downloads run concurrently in `download_collection`, `download_metadata`, `download_grib2`, and `download_netcdf`.
- New `--workers N` flag on `foehn download` (default: **8**).
- `foehn.download(...)` Python API gains a `workers` keyword argument.
- A single `requests.Session` is reused across threads per dataset, instead of being rebuilt per request.

### Usage
```bash
foehn download smn                  # default 8 concurrent downloads
foehn download smn --workers 16     # bump it up if your connection allows
foehn download smn --workers 1      # serial, old behaviour
```

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.18)

## 0.2.17 (2026-04-17)

### Breaking: download layout moved from `raw/` to `bronze/`

Downloads now land in `<data_dir>/bronze/<collection>/` instead of `<data_dir>/raw/<collection>/`, aligning with the medallion convention. No backwards compatibility — move or re-download existing data.

#### Migration

Local users:

```bash
mv ./data/meteoswiss/raw ./data/meteoswiss/bronze
```
Databricks users:
```bash
mv /Volumes/<catalog>/<schema>/<volume>/raw /Volumes/<catalog>/<schema>/<volume>/bronze
```
Other changes
convert_to_parquet(..., raw_dir, ...) parameter renamed to bronze_dir (positional callers unaffected)
Same for convert_climate_normals_to_parquet

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.17)

## 0.2.16 (2026-04-17)

### Features
- **Radar ingestion**: Added 5-minute Databricks bundle jobs for
  `radar_precip` (year-round) and `radar_hail` (Apr–Sep), downloading
  HDF5 files and MERGE-ing into the Delta catalog
  ([databricks.yml](https://github.com/kayhendriksen/foehn/blob/main/databricks.yml), [scripts/ingest_delta.py](https://github.com/kayhendriksen/foehn/blob/main/scripts/ingest_delta.py))
- **Metadata ingestion**: `*_meta_*.csv` files are now ingested as
  Delta tables ([scripts/ingest_delta.py](https://github.com/kayhendriksen/foehn/blob/main/scripts/ingest_delta.py))
- **CombiPrecip reanalysis**: client now compares STAC `updated`
  against local mtime so CPCH reanalysis overwrites stale CPC files
  ([src/foehn/client.py](https://github.com/kayhendriksen/foehn/blob/main/src/foehn/client.py))

### Fixes
- Relabeled surface/satellite grids as **C3/C4** (previously
  C3/C4/C5), added LST to the C4 description, and stubbed C5 as
  unreleased ([src/foehn/collections.py](https://github.com/kayhendriksen/foehn/blob/main/src/foehn/collections.py))

### Chores
- Pre-commit hook runs `ruff check --fix` + `ruff format`
  ([.pre-commit-config.yaml](https://github.com/kayhendriksen/foehn/blob/main/.pre-commit-config.yaml))
- Bumped GitHub Actions to Node.js 24
  ([.github/workflows/ci.yml](https://github.com/kayhendriksen/foehn/blob/main/.github/workflows/ci.yml),
  [.github/workflows/publish.yml](https://github.com/kayhendriksen/foehn/blob/main/.github/workflows/publish.yml))

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.16)

## 0.2.15 (2026-04-07)

### What's new

`foehn.load()` now supports filtering with `year`, `month`, `date_from`, `date_to`, `columns`, `drop_null`, and `sort` — across the Python API, CLI, and MCP server.

```python
df = foehn.load("smn", station="BER", frequency="d", year=2025, month=[6, 7, 8], sort="desc")
```

Other changes
Relaxed pyspark (>=3.5) and delta-spark (>=2.4) version bounds
Moved detailed docs to docs/

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.15)

## 0.2.14 (2026-04-04)

Added a few new features and filters to the MCP server to help LLM go trough the meteoswiss data

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.14)

## 0.2.13 (2026-04-04)

_No release notes were published for this version._

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.13)

## 0.2.12 (2026-04-04)

### What's new

- **MCP Registry support** — foehn is now published to the [MCP Registry](https://registry.modelcontextprotocol.io), making it discoverable by any MCP client
- **Structured tool outputs** — all MCP tools return typed Pydantic models (`Dataset`, `Parameter`, `Station`, `InventoryEntry`) instead of raw dicts, giving LLM clients richer schema information
- **Tool annotations** — tools are annotated as read-only and idempotent, helping clients make smarter decisions about when and how to call them
- **Proper error handling** — tools now raise `ValueError` instead of returning error dicts, following MCP conventions

### Improvements

- Automated MCP Registry publishing in CI pipeline via GitHub OIDC
- Added 50 tests for the MCP server
- CI lint coverage extended to `tests/`

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.12)

## 0.2.11 (2026-04-03)

### What's new

Adds a Model Context Protocol (MCP) server that exposes MeteoSwiss data to LLM clients (Claude Desktop, Cursor, etc.).

#### Tools

- `list_datasets` — browse available datasets with metadata
- `get_stations` — station names, coordinates, and canton
- `get_parameters` — column names, units, and measurement groups
- `get_inventory` — per-station data availability and time range
- `load_data` — fetch live measurements as tabular data

Also includes a `foehn://guide` reference resource and a `query_weather` prompt template.

#### Install

```pip install "foehn[mcp]"```

#### Start the server

foehn mcp                    # stdio (default, for Claude Desktop / MCP clients)
foehn mcp --transport sse    # SSE

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.11)

## 0.2.10 (2026-03-31)

### Ingest pipeline rewrite
The Databricks ingest step no longer requires a Parquet intermediate layer.
Raw CSVs are now read directly from the Unity Catalog Volume using Polars
`scan_csv` with `engine="streaming"`, then written to Delta tables via
Arrow → Spark.

- Large historical collections (SMN, SMN Precip, SMN Tower) use chunked
  writes to keep peak memory bounded (configurable via `--chunk-size`)
- Falls back to eager `parse_csv_bytes` if streaming collect fails due to
  mixed-type columns
- Column comments are automatically applied from `_meta_parameters.csv`
  English descriptions
- Local `spark-submit` is now supported: Unity Catalog DDL is skipped
  when `DATABRICKS_RUNTIME_VERSION` is not set
- Databricks job tasks updated: `download` now runs with `--no-parquet`,
  `ingest` receives `--historical` for the historical job

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.10)

## 0.2.9 (2026-03-31)

- **Combine per-station CSVs into grouped Parquet files** — Per-station CSVs are now grouped by frequency and time slice, then concatenated into a single Parquet file per group (e.g. all `ogd-smn_*_d_recent.csv` files become `smn_d_recent.parquet`)
- **Fix BOM handling in CSV decoding**
- **Add demo gif to README**

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.9)

## 0.2.8 (2026-03-28)

### What's new

- **Metadata API** — new `foehn.parameters()`, `foehn.stations()`, and `foehn.inventory()` functions to fetch dataset metadata live from the STAC API as Polars DataFrames
- **Metadata CLI** — new `foehn metadata` subcommand to inspect parameters, stations, and data inventory from the terminal
foehn metadata parameters smn
foehn metadata stations smn
foehn metadata inventory smn



### Breaking changes

- **API renames** — `data_types` → `time_slice`, `granularity` → `frequency`, `convert()` → `to_parquet()`, `key` → `dataset`

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.8)

## 0.2.7 (2026-03-27)

### Bug fixes

- Fix `data_types` accepting a string (e.g. `data_types="now"`) silently returning wrong data.
  The string was iterated character-by-character, matching unrelated time slices.
  `data_types` now accepts both `str` and `list[str]`.

- Add Polars smoke test at import time with a clear error message when binaries
  fail to load (e.g. on Databricks or systems without AVX2 support).

### Databricks support

- Add `polars-lts-cpu` to the `[databricks]` extra so `pip install foehn[databricks]`
  works out of the box on environments without AVX2/modern glibc.

### Improvements

- Improve `foehn list` output with grouping and filters
- Add Python 3.14 support
- Add ruff lint rules, fix lint issues, add SSL retry, and improve test coverage
- Add security CI job and expand test matrix

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.7)

## 0.2.6 (2026-03-24)

Refactor CLI to use subcommands (list, download, convert, load) matching the Python API.

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.6)

## 0.2.5 (2026-03-24)

### New features

- **`foehn.load()`** — load any CSV dataset directly into a Polars DataFrame without writing to disk. Supports `station`, `granularity`, and `data_types` filters.
- **Rich dataset metadata** — `list_datasets()` now returns category, subcategory, description, format, granularities, and available time slices for every dataset.

### API renames (breaking)

The public API has been renamed to follow Python data library conventions (HuggingFace, TFDS, etc.):

| Old (v0.2.4) | New (v0.2.5) |
|---|---|
| `list_collections()` | `list_datasets()` |
| `fetch()` | `download()` |
| `read()` | `load()` |
| `discover()` | removed (use `list_datasets()`) |

### Other changes

- Use metadata parameter files for CSV column type inference during conversion
- Refactor CSV parsing into reusable `parse_csv_bytes()` shared by `load()` and `convert()`
- Improved README with collection descriptions and `load()` examples
- 84 tests, 91% coverage

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.5)

## 0.2.4 (2026-03-24)

Add the Swiss National Supercomputing Centre (rgw.cscs.ch) to the list of trusted download domains, so collections hosted on CSCS object storage can be fetched without triggering the SSRF domain validation.

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.4)

## 0.2.3 (2026-03-23)

- Automatic retry on network errors — HTTP requests now use exponential backoff for transient 5xx errors instead of failing immediately
- Smarter CSV type inference — when a column inferred as integer contains float values beyond the inference window, foehn now detects the failing column and retries with targeted Float64 overrides instead of re-scanning the entire file
- Switch Parquet compression from Snappy to Zstd for smaller output files
- Switch conversion back to eager (read_csv/write_parquet) for more reliable schema error recovery
- Update download test mocks to match the new retry session API

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.3)

## 0.2.1 (2026-03-23)

### Changes

- Switch CSV-to-Parquet conversion from eager (read_csv/write_parquet) to streaming (scan_csv/sink_parquet) for lower memory usage on large files
- Add phase headers and clearer status labels in conversion logs

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.1)

## 0.2.0 (2026-03-22)

Python API, collection discovery, and security hardening

What's new
- Python API — use foehn directly from notebooks and scripts with foehn.fetch(), foehn.convert(), and foehn.list_collections()
- Collection discovery — foehn --list shows all available collections with their category and STAC collection ID
- Environment variables — FOEHN_DATA_DIR and FOEHN_FULL_REFRESH as alternatives to CLI flags

Security
- URL domain validation on all HTTP requests (SSRF protection)
- ZIP path traversal protection for climate normals extraction
- SQL injection prevention in Databricks ingestion script
- Narrowed exception handling in Delta table ingestion

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.2.0)

## 0.1.1 (2026-03-22)

- Fix banner image not rendering on PyPI by using an absolute URL
- Add a version guard to the publish workflow that catches tag/version mismatches before anything reaches PyPI

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.1.1)

## 0.1.0 (2026-03-21)

Download all 20+ MeteoSwiss Open Government Data collections via the STAC API, convert CSV/TXT to Parquet with Polars, and optionally ingest into Databricks Unity Catalog Delta tables.

pip install foehn

What's included
- Incremental downloader with ETag caching and last-run tracking
- CSV/TXT → Parquet conversion (5–10× smaller than raw CSV)
- CLI with --historical, --now, --all, and --grids flags
- Databricks Automation Bundle with daily and historical backfill jobs

[Release](https://github.com/kayhendriksen/foehn/releases/tag/v0.1.0)
