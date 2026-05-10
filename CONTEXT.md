# foehn

Domain and architecture vocabulary for the foehn package — a MeteoSwiss Open Data downloader, converter, and Delta ingestor.

## Language

### Domain

**Dataset**:
A short foehn key (e.g. `smn`, `smn_precip`, `radar_precip`) that names a body of MeteoSwiss data. Maps 1:1 to a STAC collection ID via `foehn.collections.COLLECTIONS`.
_Avoid_: "key" alone, "source", "feed".

**Collection**:
A STAC collection — the upstream MeteoSwiss concept. Has JSON metadata, items, and assets.
_Avoid_: "collection" as a synonym for **dataset**. The dataset is the foehn-internal name; the collection is the STAC entity it points at.

**STAC item**:
One per-station record inside a STAC collection. The item ID is the 3-letter station abbreviation.

**Asset**:
A downloadable file URL inside a STAC item — CSV, TXT, or HDF5.
_Avoid_: "file", "resource".

**Station**:
A measurement site. Identified by a 3-letter abbreviation (e.g. `BER`, `ZUR`).

**Time slice**:
The recency window of a CSV asset, encoded in its filename suffix: `historical` (everything before this year), `recent` (this year), `now` (last 24 hours).
_Avoid_: "period", "window", "epoch".

**Frequency**:
The cadence of a measurement, encoded in the filename: `t` (10-minute), `h` (hourly), `d` (daily), `m` (monthly), `y` (yearly).
_Avoid_: "granularity", "interval", "resolution".

**Metadata CSV**:
A `_meta_parameters.csv`, `_meta_stations.csv`, or `_meta_datainventory.csv` file shipped alongside data assets in a collection. Each becomes a per-suffix Delta table on ingest.

**Bronze**:
The local directory tree of raw downloaded files (`bronze/{dataset}/...`). Lakehouse-medallion convention.

**Delta table**:
A target table in Unity Catalog, named `{catalog}.{schema}.{dataset}{_freq_slice}` (or `{dataset}_meta_{suffix}` for metadata).

### Architecture

**DeltaSink**:
The port at the seam between foehn's ingestion logic and Delta storage. Accepts a Polars DataFrame plus a fully-qualified table name and write mode. Has `SparkDeltaSink` (production) and `RecordingDeltaSink` (test) adapters.

**BinaryFileIndex**:
The port for radar HDF5 indexing. Accepts a directory and a table; upserts `(path, product, modification_time, size_bytes)` rows by path. Separate from **DeltaSink** because the operation is `binaryFile`-reader + `MERGE` SQL, not a frame write.

## Relationships

- A **dataset** maps to exactly one **collection**.
- A **collection** contains many **STAC items** (one per **station**) plus zero or more **metadata CSVs**.
- A **STAC item** contains one or more **assets**, each tagged with a **frequency** and **time slice**.
- A **dataset** ingests into one or more **Delta tables** — one per `(frequency, time slice)` group, plus one per **metadata CSV**.
- A **DeltaSink** is the only seam that touches Spark for tabular writes.
- A **BinaryFileIndex** is the only seam that touches Spark for radar.

## Example dialogue

> **Dev:** "If I add a new SMN **time slice**, do I need to touch the **dataset** entry in `COLLECTIONS`?"
> **Maintainer:** "No — the **dataset** points at one **collection**, and time slices are encoded in the asset filenames within that collection. The grouping logic picks them up automatically. You only edit `COLLECTIONS` when MeteoSwiss publishes a new STAC collection."

> **Dev:** "Can radar go through the **DeltaSink**?"
> **Maintainer:** "No — radar's operation is index-only (`binaryFile` + `MERGE`), not a frame write. It uses the **BinaryFileIndex** port. They're separate ports because they're different operations."

## Flagged ambiguities

- **"collection"** was used in the codebase to mean both the STAC entity and the foehn-internal key. Resolved: foehn-internal key is **dataset**; the STAC entity is **collection**.
- **"frequency" vs. "granularity"** — both used in MeteoSwiss documentation. Resolved: prefer **frequency**, since that's what the CSV filename suffix encodes.
