# Databricks Pipeline

foehn ships a Declarative Automation Bundle for deploying daily data ingestion into Databricks Unity Catalog Delta tables.

---

## Prerequisites

```bash
pip install "foehn[databricks]"
pip install databricks-cli
```

The `databricks` extra adds PySpark, Delta, and Polars' compatibility runtime
(`polars[rtcompat]`) for CPUs without AVX2; Polars uses it automatically when it
is installed. Earlier versions installed `polars-lts-cpu` instead, which ships
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
local forecasts, indoor archive members, metadata CSVs, and climate-normal TXT
files therefore receive the same column names, timestamps, encodings, and dtype
normalisation before Spark writes them. Overwrite passes replace the stored
schema and chunked append passes merge compatible additions, so a normalized
schema change can migrate an existing table.

A table is only replaced once all of its input has been read and written.
Chunked historical groups are built in a `<table>__foehn_staging_<id>` table,
unique to the run, and published with one overwrite at the end, so a failed
chunk leaves the previous complete table in place. A run killed outright can
leave its staging table behind; nothing publishes it, and it can be dropped by
hand. Any table that fails to read or write is reported with
its error and makes the job exit non-zero; a dataset with no input is reported
as skipped and does not.
