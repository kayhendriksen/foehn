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

Set your Databricks workspace URL and alert email:

```bash
export BUNDLE_VAR_host=https://adb-xxx.azuredatabricks.net
export BUNDLE_VAR_alert_email=you@example.com
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
