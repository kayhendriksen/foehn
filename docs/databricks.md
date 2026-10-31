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

```bash
databricks bundle validate
databricks bundle deploy -t prod
```

---

## Jobs

This deploys two jobs:

**`foehn_daily`** -- runs at 13:30 UTC every day. Downloads the recent time slice and refreshes Delta tables. It leaves the `*_historical` tables alone, even though the backfill's CSVs stay in the volume.

**`foehn_historical`** -- paused by default. Trigger manually for the initial backfill or on January 1 for the annual archive slice. The only job that writes the `*_historical` tables.

## Published table semantics

The Delta ingest path uses the same MeteoSwiss CSV readers as `foehn.load()` and
the Parquet conversion path. Standard station CSVs, preamble climate scenarios,
local forecasts, indoor archive members, metadata CSVs, and climate-normal TXT
files therefore receive the same column names, timestamps, encodings, and dtype
normalisation before Spark writes them. Overwrite passes replace the stored
schema and chunked append passes merge compatible additions, so a normalized
schema change can migrate an existing table.

A table is only replaced once all of its input has been read and written.
Chunked historical groups are built in a `<table>__foehn_staging` table and
published with one overwrite at the end, so a failed chunk leaves the previous
complete table in place. Any table that fails to read or write is reported with
its error and makes the job exit non-zero; a dataset with no input is reported
as skipped and does not.
