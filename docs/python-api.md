# Python API

Use foehn directly from notebooks or scripts. All functions are available from the top-level `foehn` module.

---

## Listing datasets

```python
import foehn

foehn.list_datasets()
# [{'dataset': 'smn', 'collection_id': 'ch.meteoschweiz.ogd-smn', 'category': 'A',
#   'subcategory': 'A1', 'description': 'Automatic weather stations',
#   'format': 'CSV', 'frequencies': ['t', 'h', 'd', 'm'],
#   'time_slices': ['historical', 'recent', 'now']}, ...]
```

---

## Loading data

`foehn.load()` fetches data directly into a Polars DataFrame without writing anything to disk.

```python
# Load daily data for a single station
df = foehn.load("smn", station="BER", frequency="d")

# Multiple stations and frequencies
df = foehn.load("smn", station=["BER", "ZUR"], frequency=["d", "h"])

# Include historical data alongside recent
df = foehn.load("smn", station="BER", frequency="d", time_slice=["historical", "recent"])
```

### Filtering

All filters can be combined.

```python
# Filter by year and month
df = foehn.load("smn", station="BER", frequency="d", year=2026, month=[1, 2, 3])

# Date range
df = foehn.load("smn", station="BER", frequency="d",
                time_slice="historical", date_from="2025-06-01", date_to="2025-08-31")

# Select specific columns (station_abbr and reference_timestamp are always kept)
df = foehn.load("smn", station="BER", frequency="d", columns=["tre200d0", "rre150d0"])

# Drop rows where a column is null
df = foehn.load("obs", frequency="y", drop_null="w3pnd2y0")

# Sort by timestamp (newest first)
df = foehn.load("smn", station="BER", frequency="d", sort="desc")
```

| Parameter | Type | Description |
|---|---|---|
| `station` | `str` or `list[str]` | Station abbreviation(s) |
| `frequency` | `str` or `list[str]` | Time frequency: t, h, d, m, y |
| `time_slice` | `str` or `list[str]` | Time period: historical, recent, now |
| `year` | `int` or `list[int]` | Filter by year(s) |
| `month` | `int` or `list[int]` | Filter by month(s), 1--12 |
| `date_from` | `str` | Start date inclusive, ISO format |
| `date_to` | `str` | End date inclusive, ISO format |
| `columns` | `list[str]` | Only return these columns |
| `drop_null` | `str` | Drop rows where this column is null |
| `sort` | `str` | Sort by timestamp: "asc" or "desc" |

---

## Metadata

Explore dataset metadata fetched live from the MeteoSwiss API.

```python
# Column name mappings: shortname, description, unit, type, ...
foehn.parameters("smn")

# Station info: abbreviation, name, canton, altitude, lat, lon, ...
foehn.stations("smn")

# What each station measures and since when
foehn.inventory("smn")
```

---

## Downloading to disk

```python
# Download a single dataset
foehn.download("smn", data_dir="./data/meteoswiss")

# Download with specific time slices
foehn.download("smn", time_slice=["historical", "recent"])
```

---

## Converting to Parquet

```python
foehn.to_parquet("smn", data_dir="./data/meteoswiss")
```

Parquet files are written to `<data_dir>/parquet/<collection>/`.
