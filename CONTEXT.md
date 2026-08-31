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

**MeteoSwiss CSV**:
How MeteoSwiss writes a CSV and what foehn needs to read one back: the
UTF-8/Windows-1252 fallback, the semicolon separator, the dtypes named in a
**Collection**'s `_meta_parameters.csv`, the `KEY;VALUE` preamble, and the
filename rules that carry station, **Granularity**, **Time slice** and
**Forecast run**. Upstream's conventions, below every pipeline that reads them —
the load path, the convert stage and the Databricks ingest script alike.
_Avoid_: parsing, format (a **Format** is a value; this is the rules for one)

**Staging**:
Writing to a sibling temp file and moving it onto the target, so a write that
dies part-way leaves nothing rather than a truncated file with a fresh mtime —
which every skip rule above it reads as "already done". One rule, under the
**Fetcher**, **Transfer**, the convert stage and **Run state** alike.
_Avoid_: atomic (it is not atomic on every filesystem; the move is what matters)

**Metadata table**:
One collection-level **Asset** — parameters, stations, inventory — and the
columns foehn publishes from it. The suffix is MeteoSwiss's, the published names
are foehn's. Not a **Dataset kind** thing: nothing about it varies by kind, so it
is reached directly rather than through a registry row.
_Avoid_: metadata (unqualified — the convert stage means dtypes by it)

**Run state**:
What foehn remembers between runs, in the **Workspace**: the ETag store keyed by
asset href, and the last-run cursor the CLI advances only after a fully clean
run. Both reads are total — a corrupt file is treated as absent, because a lost
cursor costs one redundant download and a raised exception costs the run.
_Avoid_: cache (the **Fetcher**'s listing memo is a cache; this is not)

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
- Every module that reads a MeteoSwiss file goes through **MeteoSwiss CSV**; only
  the convert stage depends on the Parquet one.
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
- "Is this a known **Dataset**?" was answered by a bare `KeyError` below and
  re-worded as a sentence at all six of `api`'s entry points, and `load()` carried
  five more checks the **Dataset kind**'s row already answered. Resolved:
  `collections.collection_id` raises the sentence, and `registry.validate_load`
  refuses a query the row cannot answer.
- The module holding the download adapters was called `client` — the word the
  **Fetcher** entry tells you to avoid — on the one module in the tree that makes
  no HTTP call itself. It also held **Run state** and the ZIP guards, so the load
  path imported a download module to reach one of them. Resolved: `downloads`,
  `state` and `archives`, each named for what it owns.
- Upstream's CSV conventions and the Bronze → Parquet stage lived in one module,
  so `transfer` — the download engine — depended on the Parquet converter for one
  byte-level helper, and the gridded read path did too, through it. Five modules
  imported that one and four of them wanted only the conventions. Resolved:
  **MeteoSwiss CSV** is its own module and `convert` has a single consumer.
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
- The **Category** letters and their labels were written out in three modules —
  prose in `collections`, a dict in `cli.cmd_list`, a set in the MCP layer — plus
  a fourth time in the MCP catalogue's field description. Resolved:
  `collections.CATEGORY_LABELS` is the vocabulary and `CATEGORIES` its keys, the
  way `GRANULARITY_LABELS` and `TIME_SLICE_LABELS` already were. Category B is
  MeteoSwiss's and unreleased, so it has no row: a letter in the table is a
  letter something will offer as a filter.
- Which **Datasets** are loadable was rendered from the registry in the MCP
  guide but hand-typed in `load_data`'s docstring, which had fallen a dataset
  behind — it told an LLM `climate_scenarios_indoor` was not loadable while
  `load()` loaded it. Resolved: a tool docstring is an interface, so it is
  rendered from the same tables the guide uses. Placeholders are `$name`, filled
  by `_renders` under the `@mcp.tool` decorator.
- The gridded read path was one 800-line module: the STAC listing and fetch that
  puts a match's files in **Bronze**, upstream's ODIM and ICON file conventions,
  and the **Grid readers** themselves. So the code that parses a radar composite
  imported the download engine, and the ICON coordinate cache — the one piece of
  process state in the path — was a private global the test suite reached into
  by name. Resolved: `gridfiles` fetches (what **Transfer** is to the download
  paths), `odim` and `icon` carry upstream's conventions (what **MeteoSwiss
  CSV** is to the tabular path), and `grids` is the **Grid readers**. The same
  four-way split the tabular path already had.
- **Staging** was stated four times — the **Fetcher**'s stream, **Transfer**'s
  byte writer, the convert stage's Parquet writer and **Run state** — with three
  suffixes and three docstrings reasoning their way to the same rule. **Run
  state** reached into **Transfer**, the download engine, for a filesystem
  primitive that was never the download path's to own. Resolved: `atomicwrite` is
  a leaf and `test_layering` fails a module that hand-rolls the move itself.
- The default **Time slice** was the token `"recent"`, written at four code sites
  — the `Filters` field, its builder, `registry.download` and the CLI — and again
  as prose in four docstrings. Resolved: `collections.DEFAULT_TIME_SLICE` is the
  fact and every site reads it.
- `load()`'s docstring named the **Granularity** and **Time slice** tokens as
  prose, and so did `load_data`'s and `describe_data`'s — the latter two directly
  under the comment in `mcp_server` saying a tool docstring is rendered from the
  tables, never retyped beside them. The rule had been stated for one front end
  and half-applied there. Resolved: the renderer is `docstrings.renders`, a leaf
  under both front ends; `collections.options` renders a label table as prose;
  and `test_docstrings` asserts every offered token reaches every surface that
  offers it, so the rendering is load-bearing rather than a convention.
- The layering guards that could not be asked of the import graph were substring
  matches on the source — `".replace(path)"`, `"separator="`, `".chunk("`. Brittle
  both ways: `"dask"` matched a comment, and `".replace(path)"` would have missed
  `tmp.replace(target)`. Resolved: they read the tree — which names a module
  calls, which constants it passes as a keyword, whether it calls
  `<path>.replace(x)` with the one argument that tells `Path.replace` from
  `str.replace`.
- The **Standard CSV** kind's scan options were stated in the convert stage
  rather than in **MeteoSwiss CSV**, because the dtype-drift retry is wrapped
  around them. The last kind whose conventions sat above the module that owns
  them. Resolved: `meteocsv.scan_standard_csv` is the lazy half of
  `parse_csv_bytes`; the retry stays in the convert stage and passes the widened
  types back in. What convert still spells out is the **Direct ZIP** kind's
  tab-separated TXT, whose only reader it is.
- The **Archive CSV** kind's member files were read in two places — `readers`
  for the load path, `convert` for the Parquet stage — each spelling out
  upstream's separator and schema window and then calling the two `meteocsv`
  helpers in the right order. Every other kind already had its eager reader and
  its lazy scanner in **MeteoSwiss CSV**, each owning its own read options.
  Resolved: `indoor_station` answers whether a member is data and whose, and
  `parse_indoor_csv`/`scan_indoor_csv` read one. The load path now spells out no
  CSV convention at all.
- The Zarr store's name — `<dataset>` unfiltered, `<dataset>__<match>` otherwise —
  was derived in `api` and handed to `Workspace.zarr()` as a finished string, so
  that method documented the encoding as its caller's job. The one path foehn
  spelled out above the seam. Resolved: `Workspace.zarr(dataset, match)` takes
  the two facts the name is made of, and the **Workspace** owns this rule like
  every other one about the layout.
- `registry.write_zarr` routed to the right writer and then performed the write
  itself: move the non-CF time units aside, check for dask, rechunk, call
  `grids.write_zarr`. So the routing table knew the recipe, three names existed
  in `grids`' interface only so it could sequence them, and `cube_grib2` stated
  half the same recipe a second time inside `grids` on its own way out.
  Resolved: `grids.write_zarr` takes `rechunk=` and owns every step; `sanitize`
  and the dask check are private to it, and the registry picks the method.

- `api` — the public surface — held one stage outright rather than delegating:
  the **Metadata tables** were fetched, decoded and renamed inline, so it was the
  one entry point taking no **Fetcher** and reachable only through the
  process-wide default. `to_zarr` likewise chose cube-vs-single itself. Resolved:
  `metadata` owns the tables and their implementation, `registry.write_zarr`
  owns the choice, and `api` states the contract and delegates. It imported nine
  modules against `mcp_server`'s two; it imports seven now, and holds 51
  statements where it held 80.
