# Databricks Pipeline

foehn ships a Declarative Automation Bundle for deploying daily data ingestion into Databricks Unity Catalog Delta tables.

---

## Prerequisites

```bash
pip install "foehn[databricks]"
pip install databricks-cli
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

**`foehn_daily`** -- runs at 13:30 UTC every day. Downloads the recent time slice and refreshes Delta tables.

**`foehn_historical`** -- paused by default. Trigger manually for the initial backfill or on January 1 for the annual archive slice.

## Published table semantics

The Delta ingest path uses the same MeteoSwiss CSV readers as `foehn.load()` and
the Parquet conversion path. Standard station CSVs, preamble climate scenarios,
local forecasts, indoor archive members, metadata CSVs, and climate-normal TXT
files therefore receive the same column names, timestamps, encodings, and dtype
normalisation before Spark writes them. Overwrite passes replace the stored
schema and chunked append passes merge compatible additions, so a normalized
schema change can migrate an existing table.
