# Databricks Pipeline

foehn ships a Declarative Automation Bundle for deploying daily data ingestion into Databricks Unity Catalog Delta tables.

---

## Prerequisites

```bash
pip install "foehn[databricks]"
```

Install the current [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install)
using its supported installer, then check `databricks --version`. The legacy
Python package `databricks-cli` does not provide the `bundle` commands used here.

The `databricks` extra adds PySpark 4 with its SQL dependencies (Arrow, pandas,
and NumPy), Delta 4, and Polars' compatibility runtime
(`polars[rtcompat]`) for CPUs without AVX2; Polars uses it automatically when it
is installed. PySpark 4 is required: the ingest hands Spark Arrow tables, which
PySpark 3.5 cannot read. Running Spark locally needs Java 17 or 21. Earlier versions installed `polars-lts-cpu` instead, which ships
the same `polars` package as the regular build, so the two overwrite each
other's files (typically `ImportError: cannot import name ... from 'polars...'`).
In an environment that has both, remove the old one and repair Polars:

```bash
pip uninstall polars-lts-cpu
pip install --force-reinstall --no-deps polars
```

---

## Setup

Set your Databricks workspace URL via the standard unified-auth env var (or a `~/.databrickscfg` profile passed with `--profile`):

```bash
export DATABRICKS_HOST=https://adb-xxx.azuredatabricks.net
```

---

## Deploy

Run these from a checkout of the version you want to deploy, with
[uv](https://docs.astral.sh/uv/) installed:

```bash
databricks bundle validate
databricks bundle deploy -t prod
```

The deploy builds foehn's wheel from that checkout (`uv build --wheel --clear
--out-dir dist/bundle`) and installs it on every job, so the jobs always run the
code you deployed. `dist/bundle/` is emptied before each build; the `dist/` you
build releases into is left alone.

---

## Jobs

The bundle defines four jobs. In the `dev` target, development mode pauses
every schedule; in `prod`, each runs on the schedule below as soon as it is
deployed.

| Job | Schedule (Quartz, UTC) | Default | What it does |
|---|---|---|---|
| `foehn_daily` | `0 30 13 * * ?` — daily at 13:30 | Unpaused | Downloads the recent time slice and refreshes the Delta tables. Leaves the `*_historical` tables alone, even though the backfill's CSVs stay in the volume. |
| `foehn_radar_precip` | `0 */5 * * * ?` — every 5 minutes, all year | Unpaused | Downloads new precipitation radar composites (D1) and indexes them in the `radar_precip` table. Serverless, about 30 s per run. |
| `foehn_radar_hail` | `0 */5 * ? 4-9 *` — every 5 minutes, April–September | Unpaused | The same for the hail radar products (D3), which MeteoSwiss publishes only in hail season. |
| `foehn_historical` | `0 0 3 1 1 ?` — 03:00 on January 1 | Paused | The full historical backfill. Trigger it by hand for the first run, or unpause it to pick up each year's archive slice. The only job that writes the `*_historical` tables. |

Together the two radar jobs run 576 times a day from April to September and 288
times a day the rest of the year. If you do not want the radar archive, pause them before
deploying to prod — set `pause_status: PAUSED` on their `schedule` — or pause
them in the Jobs UI afterwards.

## Published table semantics

The Delta ingest path uses the same MeteoSwiss CSV readers as `foehn.load()` and
the Parquet conversion path. Standard station CSVs, preamble climate scenarios,
local forecasts, indoor archive members, and metadata CSVs therefore receive the
same column names, timestamps, encodings, and dtype normalisation before Spark
writes them. Where a name holds a character Delta
refuses — the climate normals' `Altitude [m asl]`, for one — the table is written
with Delta column mapping (`delta.columnMapping.mode = name`) so the name is kept;
such a table needs a Delta reader that supports column mapping, which Databricks
does. Overwrite passes replace the stored
schema and chunked append passes merge compatible additions, so a normalized
schema change can migrate an existing table.

The `climate_normals` Delta table combines only the English TXT variants from
the archive, so translations do not duplicate measurements. It retains the
English station and monthly-value columns and adds `parameter` (MeteoSwiss
shortcode), `unit`, `normal_period` (for example `1991-2020`), and `source_file`.
`normal_period` identifies the source's 30-year normal; the original
`Reference period` column describes the station's available reference interval.
Parameter identity must be included when selecting or aggregating observations.
The separate per-file climate-normal Parquet outputs retain their existing
schema and filenames.

A table is only replaced once all of its input has been read and written.
Chunked historical groups are built in a `<table>__foehn_staging_<id>` table,
unique to the run, and published with one overwrite at the end, so a failed
chunk leaves the previous complete table in place. A run killed outright can
leave its staging table behind; nothing publishes it, and it can be dropped by
hand. Any table that fails to read or write is reported with
its error and makes the job exit non-zero; a dataset with no input is reported
as skipped and does not.

An entire rolling group that stays unlisted upstream for seven days is reconciled
too, once a later successful download has recorded that. Its managed
`<dataset>_<frequency>_recent` or `<dataset>_<frequency>_now` Delta table is
emptied, preserving its schema, comments, and grants. Reserve these exact names
in the configured schema for foehn outputs. Historical, unsliced, metadata,
and unrelated tables are retained. Missing local files without a successful
listing do not authorize emptying a table.

## Rebuild an existing climate-normal table

Earlier versions combined parameters and languages without retaining their
identity. Those rows cannot be repaired reliably from the existing Delta table;
rebuild it from the TXT files in Bronze after deploying this version. On a
Databricks compute environment with the deployed wheel and script, run:

```bash
python scripts/ingest_delta.py --climate-normals-only \
  --catalog main --schema meteoswiss --volume landing
```

The same arguments can be supplied to a one-off Spark Python task pointing at
the deployed `scripts/ingest_delta.py`. This replaces only
`main.meteoswiss.climate_normals`, with its schema migrated in the overwrite;
the previous table stays available if parsing or writing fails. Missing Bronze
input makes this targeted rebuild fail instead of reporting a successful skip.
If input is missing, download it first on the same Databricks environment:

```bash
foehn download climate_normals --no-parquet \
  --data-dir /Volumes/main/meteoswiss/landing
```

After rebuilding, check that `parameter`, `unit`, `normal_period`, and
`source_file` are populated, and update downstream queries to include parameter
identity. Review station/parameter/normal-period duplicates before aggregating.
