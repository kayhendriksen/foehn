# foehn

A Python toolkit for MeteoSwiss Open Government Data: it downloads what MeteoSwiss
publishes, and hands it back as Polars DataFrames, Parquet, xarray Datasets or Zarr.
This is the language foehn uses for MeteoSwiss's concepts and for its own.

## Language

### Datasets

**Dataset**:
foehn's short key for one body of MeteoSwiss data — `smn`, `radar_precip`,
`climate_scenarios`. The unit every public function takes.
_Avoid_: collection (that is the STAC id), source, feed, product

**Collection**:
The upstream identifier a **Dataset** maps to, e.g. `ch.meteoschweiz.ogd-smn`.
A STAC collection id for every **Dataset kind** but **Direct ZIP**, which has no
STAC collection and whose identifier is the geo.admin path its ZIP lives under.
One **Dataset** is exactly one **Collection**.
_Avoid_: dataset, endpoint

**Dataset kind**:
Which pipeline handles a **Dataset** — its download path, convert path and load
path. Seven values, listed below. Internal to foehn, and deliberately distinct
from **Format**.
_Avoid_: type, format, category, class

**Format**:
The wire format of a **Dataset**'s files — `CSV`, `NetCDF`, `GRIB2`, `HDF5`.
Public: it appears in `list_datasets()` and tells a caller whether they need the
`foehn[grids]` extra. Several **Dataset kinds** can share one **Format**.
_Avoid_: kind, type

**Category**:
MeteoSwiss's own A–E classification of its data (A ground-based, C climate,
D radar, E forecast). Public, and theirs — foehn reports it but never routes on it.
_Avoid_: kind, group, class

### The eight dataset kinds

**Standard CSV** (10 datasets):
Per-station CSV assets split by **Time slice** and **Granularity**. The main path:
`smn`, `nbcn`, `pollen`, and the rest of category A plus C1/C2.

**Preamble CSV** (`climate_scenarios`):
CSV carrying a `KEY;VALUE` metadata preamble before the real `DATE;<model>;…`
header, on nominal 30-year dates rather than calendar dates.

**Archive CSV** (`climate_scenarios_indoor`):
A single ZIP of per-station CSVs rather than per-station STAC assets.
_Avoid_: zip CSV — the archive-ness is what routes, not the compression.

**Forecast CSV** (`forecast_local`):
Point-forecast CSVs named by **Forecast run**, with no **Time slice** and no
`reference_timestamp` column of their own.

**NetCDF grid** (5 datasets):
Static gridded climate analyses, normals and scenarios. Combines across files.

**GRIB2 grid** (3 datasets):
ICON-CH1/CH2 forecasts and KENDA analysis, on an unstructured `values` grid.
One field per file.

**Radar grid** (`radar_precip`, `radar_hail`):
ODIM-H5 Cartesian composites, one per timestep.
_Avoid_: HDF5 grid, GRIB2 collection — lumping radar with GRIB2 was the original
routing mistake.

**Direct ZIP** (`climate_normals`):
A ZIP fetched from a fixed URL rather than listed on the STAC API. The only kind
with a download path and a convert path but neither a **Reader** nor a **Grid
reader**: its TXT files are a wide per-parameter table keyed by station *name*,
with twelve month columns and no timestamp, so what a normals DataFrame should
look like is an open question rather than a missing adapter.
_Avoid_: normals kind (the kind is the shipping shape, not the subject matter)

### Assets

**Item**:
One STAC entry inside a **Collection**. What an item *is* varies by **Dataset
kind**: a station for **Standard CSV**, a whole day of runs for **Forecast CSV**.

**Asset**:
One downloadable file, hanging off an **Item** or off the **Collection** itself.
Collection-level assets carry metadata (parameters, stations, inventory);
item-level assets carry the data.

**Time slice**:
Which span of a series an **Asset** holds: `historical` (start of record to the
end of last year), `recent` (this year to yesterday), `now` (last ~24h). Some
**Datasets** have none.
_Avoid_: data type, period, range

**Granularity**:
The aggregation interval of an **Asset**: `t` (10-minute), `h`, `d`, `m`, `y`.
MeteoSwiss's word, and foehn's. The public parameter is spelled `frequency` for
backwards compatibility; prefer granularity everywhere else.
_Avoid_: resolution, interval

**Forecast run**:
The `YYYYMMDDHHMM` issue time embedded in a forecast **Asset**'s filename — when
the forecast was made. Sorts lexicographically, so the newest run is `max()`.
_Avoid_: reference timestamp, forecast time

**Reference timestamp**:
The timestamp of a measurement or of a forecast step's interval, inside the file.
Not the **Forecast run**.

### Storage and transport

**Workspace**:
One data directory and every path foehn derives from it — **Bronze**, Parquet,
Zarr, the ETag store and the last-run cursor. Resolved once, by one rule: the
caller's `data_dir`, else `$FOEHN_DATA_DIR`, else `./data/meteoswiss`. The
public functions still take `data_dir=`; everything inside foehn takes the
**Workspace**, so no module derives one path from another.
_Avoid_: data dir (that is the argument, not the concept), root, storage

**Bronze**:
The local cache of raw downloaded files, at `<workspace>/bronze/<dataset>/`. What
`download()` writes and what the grid readers read from.
_Avoid_: raw, landing (the Databricks ingest script's word for the same place)

**Fetcher**:
The module that owns every HTTP call foehn makes to MeteoSwiss: STAC listing and
metadata, and file downloads. Retry policy, per-thread sessions, URL validation
and pagination live behind it. One call.
_Avoid_: client, session, transport

**Transfer**:
The module that owns turning a set of **Assets** into files: the worker pool,
destination de-duplication, per-asset failure isolation, ETag bookkeeping,
atomic writes and the counts in a **Download result**. Many calls, where the
**Fetcher** is one. Every download path goes through it.
_Avoid_: downloader, batch, pool

**Download result**:
What a download reports: `total_assets`, `downloaded`, `skipped`, `failed` and
the new filenames. The first is every **Asset** the call was given, and the
other three sum to it.

**Reader**:
How one **Dataset kind** becomes a Polars DataFrame — one per tabular kind,
selected from the registry exactly as its download and convert paths are. Owns
what its frame looks like as well as how it is fetched: which columns an explicit
`columns=` always keeps, and what `sort=` orders by. Hands back a finished frame,
so nothing above the registry filters it further.
_Avoid_: loader, parser (parsing is one step inside a reader)

**Grid reader**:
How one **Dataset kind** becomes an xarray Dataset — one per grid kind, selected
from the registry exactly as a **Reader** is. Carries what its kind needs opened
and, where the kind has one, how its matched files assemble into a single Zarr
cube. A **Dataset kind** has a **Reader** or a **Grid reader**, never both.
_Avoid_: engine, backend (those are xarray's, and one grid reader uses no xarray
engine at all), format reader

**Filters**:
One load query, normalised: stations and granularities lowercased, scalars
widened to tuples, an empty list read as "no filter". The public `load()`
keywords are packed into one of these at the seam, so nothing below restates
the eleven arguments.
_Avoid_: query, params, options

## Relationships

- A **Dataset** maps to exactly one **Collection** and has exactly one **Dataset kind**.
- A **Dataset kind** determines a **Format**; a **Format** does not determine a **Dataset kind**.
- A **Collection** holds many **Items**; an **Item** holds many **Assets**.
- A **Standard CSV** **Asset** is identified by its station, **Granularity** and **Time slice**.
- A **Forecast CSV** **Asset** is identified by its **Forecast run**.
- Every network read of a **Collection**, **Item** or **Asset** goes through the **Fetcher**.
- Every **Asset** written to **Bronze** goes through **Transfer**, which calls the **Fetcher**.
- Every path foehn reads or writes comes from a **Workspace**.
- A **Dataset kind** has one download path and one convert path, and at most one
  of a **Reader** (tabular) or a **Grid reader** (gridded) — never both, and
  neither for **Direct ZIP**.

## Flagged ambiguities

- "Collection" was used for both foehn's short key and the STAC id — including in
  the names `GRIB2_COLLECTIONS`, `NETCDF_COLLECTIONS` and friends, which are keyed
  by **Dataset**. Resolved: the short key is a **Dataset**; the STAC id is a
  **Collection**. Those sets are superseded by **Dataset kind**.
- `GRIB2_COLLECTIONS` contained the two **Radar grid** datasets, whose **Format**
  is HDF5. Resolved: **Radar grid** is its own **Dataset kind**.
- "Frequency" and "granularity" both refer to the aggregation interval. Resolved:
  the concept is **Granularity**; `frequency` survives only as a public parameter name.
- The valid **Granularity** and **Time slice** tokens were listed twice — once in
  `collections`, once again in the MCP layer, which could only agree or drift.
  Resolved: `collections.GRANULARITIES` and `collections.TIME_SLICES` are the
  vocabulary, `load()` enforces it, and the MCP tools restate nothing.
- `climate_normals` was a **Dataset** in the docs and in three callers' special
  cases, but not in the dataset table — so `foehn.list_datasets()` did not show
  it and `foehn.download("climate_normals")` raised "Unknown dataset". Resolved:
  it is a **Direct ZIP** **Dataset** like any other. What "download everything"
  means is now *not gridded* rather than *tabular*, which is what it always meant.
- The MCP guide restated `load()`'s filter vocabulary as prose and had drifted:
  it told callers `sort` defaults to `"asc"` when an omitted `sort` does not sort
  at all. Resolved: the granularity and time-slice tokens carry their own labels
  in `collections` and the guide renders them. The worked examples stay prose —
  they are guidance for an LLM, not a vocabulary that can drift.
- The default data directory was written at seven call sites and only the CLI
  read `$FOEHN_DATA_DIR`, so the same environment sent `foehn download` and
  `foehn.download()` to different places; the ETag store was placed at whatever
  `output_dir.parent` a caller passed. Resolved: **Workspace** owns the layout
  and the resolution rule.
- Which **Dataset kinds** are gridded was stated twice — as `collections.GRID_KINDS`
  beside the kind enum, and again as the keys of the grid path's own reader table.
  Resolved: a kind is gridded iff its registry row carries a **Grid reader**;
  `GRID_KINDS` and `is_grid` are gone.
