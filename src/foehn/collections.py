"""
MeteoSwiss STAC collection IDs, routing sets, and constants.

TIME SLICES
-----------
MeteoSwiss slices the higher-volume CSV data by time, encoded in the filename.
Which slices exist depends on the granularity — the split tracks data volume and
update rate, so the finer the granularity, the more it is sliced:

  granularity    historical  recent  now
  ---------------------------------------
  t  (10-min)        y         y      y
  h  (hourly)        y         y      y
  d  (daily)         y         y      -
  m  (monthly)       -         -      -    single unsliced file
  y  (yearly)        -         -      -    single unsliced file

  "historical"  — From start of measurement → Dec 31 of last year.
                   Updated once a year (early January). t, h, d.
                   The t and h series are additionally chunked per decade
                   (ogd-smn_ber_t_historical_2000-2009.csv), so the slice is not
                   always the trailing filename segment — see
                   time_slice_from_filename below.

  "recent"      — From Jan 1 of this year → yesterday.
                   Rebuilt daily at 12:00 UTC. t, h, d.

  "now"          — From yesterday 12:00 UTC → now.
                   Refreshed every 10 minutes. t and h only.
                   It exists to bridge the gap between daily rebuilds of
                   "recent", which is why daily has none: a daily aggregate is
                   not complete until the day is over, and by then the next
                   "recent" rebuild already carries it. A "d_now" file would be
                   permanently empty or a duplicate.

  (no slice)     — m and y granularities, plus whole collections (phenology,
                   totaliser): one file per station, no time-slice suffix.

DATA GRANULARITY (suffix in filename: _t, _h, _d, _m, _y)
----------------------------------------------------------
  _t  10-minute values   — raw realtime data from SwissMetNet (SMN)
  _h  Hourly values      — aggregated from 10min or from instrument (e.g. pollen)
  _d  Daily values       — aggregated per WMO guidelines
  _m  Monthly values     — used in climatology, homogeneous series, normals
  _y  Yearly values      — used in climate scenarios

All timestamps are in UTC.
  t, h: timestamp = END of interval   (16:00 means 15:50:01–16:00:00)
  d, m, y: timestamp = START of interval (01.06.2023 means the whole of June)

CSV FORMAT
----------
  Separator:   semicolon (;)
  Encoding:    Windows-1252 (re-encoded to UTF-8 on download)
  Decimals:    full stop (.)
  Missing:     empty field

DATA CATEGORIES (per MeteoSwiss documentation)
----------------------------------------------
Named in CATEGORY_LABELS below — the four foehn carries. What each ships:
  A  CSV, time-sliced (historical/recent/now)
  C  CSV + NetCDF + TXT (varies by sub-type)
  D  HDF5 (binary grid data)
  E  GRIB2 (ICON models + KENDA analysis) + CSV (local forecasts)
  B  Atmosphere measurements — NOT YET RELEASED (radio soundings etc.), so it has
     no CATEGORY_LABELS row and nothing offers it as a filter.

WHERE COLLECTION METADATA LIVES (by format)
-------------------------------------------
A STAC collection exposes two asset levels: collection-level assets (one shared
set, at GET /collections/<id> -> "assets") and item-level assets (the per-item
data files, at /collections/<id>/items). Where the *metadata* sits differs by
format — this is what each foehn reader relies on:

  CSV     — Collection-level assets: <key>_meta_parameters.csv (column names,
            units, descriptions), <key>_meta_stations.csv (station name, canton,
            LV95 + WGS84 coords), <key>_meta_datainventory.csv (per-station date
            ranges). Surfaced by foehn.parameters() / stations() / inventory().
            The data CSVs themselves are item-level assets.

  NetCDF  — No collection-level metadata; everything is embedded as CF
            attributes inside each .nc item asset: per-variable ``units`` /
            ``long_name`` / ``grid_mapping`` (-> a CRS variable, e.g.
            ``swiss_lv95_coordinates`` with the projection WKT), plus ``lon``/
            ``lat`` coordinate variables. xarray surfaces these directly; foehn
            only special-cases non-CF time units ("months/years since ...",
            handled via a decode_times=False fallback + sanitise-on-write).

  GRIB2   — Per-field metadata (units, long_name, GRIB_* keys) lives in each
            message and is surfaced by cfgrib. Collection-level assets add:
            params_<model>.csv (the full parameter catalogue),
            horizontal_constants_<model>.grib2 (cell tlat/tlon on the unstructured
            ``values`` grid — foehn joins these as lat/lon), and
            vertical_constants_<model>.grib2 (level/height geometry; not used yet).

  HDF5    — ODIM-H5 attributes inside each item file: /what (object, date/time,
   (radar)  product), /where (projdef, xsize/ysize/xscale/yscale, corner lat/lon),
            /how (+ /how/MeteoSwiss: long_name, versions, gauge counts), and
            /dataset1/data1/what (quantity, gain/offset/nodata/undetect). foehn's
            ODIM reader extracts these to build the scaled, LV95-georeferenced grid.
"""

import re

from foehn.catalog import COLLECTION_META as COLLECTION_META
from foehn.catalog import COLLECTIONS as COLLECTIONS
from foehn.catalog import DATASETS as DATASETS
from foehn.catalog import KIND_OF as KIND_OF
from foehn.catalog import DatasetKind as DatasetKind
from foehn.catalog import DatasetSpec as DatasetSpec

STAC_API_BASE = "https://data.geo.admin.ch/api/stac/v1"

# Dataset facts are declared once in :mod:`foehn.catalog`.
#
# Kinds whose filenames do not carry the standard
# ogd-{key}_{station}_{granularity}[_{timeslice}].csv pattern, so there is no
# granularity to filter on. Derived from the kind rather than listed separately,
# which is what NO_GRANULARITY_COLLECTIONS used to be.
NO_GRANULARITY_KINDS = frozenset({DatasetKind.PREAMBLE_CSV, DatasetKind.ARCHIVE_CSV, DatasetKind.FORECAST_CSV})

# Kinds whose files carry no ``reference_timestamp`` column and derive one from
# something else in the row. Stated here because both the load path and the
# convert stage need it and neither can ask the registry — it sits above them.
# It was a ``dataset == "forecast_local"`` comparison at four sites, three of
# them in a function that had already resolved the kind.
DERIVED_TIMESTAMP_KINDS = frozenset({DatasetKind.FORECAST_CSV})


def collection_id(dataset: str) -> str:
    """Return a dataset's **Collection**, or raise the message every caller used to write.

    ``api`` spelled this guard out at six entry points because the layers below
    raised a bare ``KeyError``, which is not a sentence a caller can act on.
    """
    try:
        return COLLECTIONS[dataset]
    except KeyError:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use list_datasets() to see available datasets.") from None


def kind(dataset: str) -> DatasetKind:
    """Return a dataset's :class:`DatasetKind`, or raise for an unknown dataset."""
    collection_id(dataset)
    return KIND_OF[dataset]


# Whether a dataset is read as a grid is not listed here: it is whether its kind
# has a grid reader, which is ``registry.spec(dataset).is_grid``. A set naming
# the grid kinds beside the table that gives them their readers could only ever
# agree with it or drift from it.


# C6 climate normals — separate ZIP from opendata.swiss (not on STAC API).
# Contains 112 TXT files (tab-separated, Latin-1 encoding) with monthly/yearly
# station normals for periods 1961-1990 and 1991-2020. Converted to Parquet.
CLIMATE_NORMALS_ZIP_URL = "https://data.geo.admin.ch/ch.meteoschweiz.klima/normwerte/normwerte.zip"


# Time-slice tokens that appear as the trailing filename segment of standard
# CSV assets (ogd-{key}_{station}_{granularity}_{timeslice}.csv), each with what
# it covers. The MCP guide renders these rather than restating them: a token
# added here has to reach the LLM-facing documentation, and prose cannot be
# checked against a set.
TIME_SLICE_LABELS: dict[str, str] = {
    "now": "last ~24 hours, updated every 10 minutes — t and h only",
    "recent": "this calendar year through yesterday, updated daily",
    "historical": "start of measurements through Dec 31 of last year",
}

TIME_SLICES = frozenset(TIME_SLICE_LABELS)

DEFAULT_TIME_SLICE = "recent"
"""What a caller who names no time slice gets.

The token was written at four code sites — the :class:`~foehn.readers.Filters`
field, its builder, ``registry.download`` and the CLI — and again as prose in
four docstrings. One of them is the whole vocabulary's default; the rest were
copies of it.
"""

# The granularity segment's vocabulary — the ``_t``/``_h``/``_d``/``_m``/``_y``
# documented at the top of this module. Which of them a given dataset actually
# has is ``COLLECTION_META[...]["frequencies"]``; this is the whole alphabet, and
# the single source for it (the MCP layer used to carry its own copy of the
# tokens, and then its own prose gloss of them).
GRANULARITY_LABELS: dict[str, str] = {
    "t": "10-minute, near real-time",
    "h": "hourly",
    "d": "daily",
    "m": "monthly",
    "y": "yearly",
}

GRANULARITIES = frozenset(GRANULARITY_LABELS)

# MeteoSwiss's own A–E classification, and what each letter is called. foehn
# reports a category and filters on one but never routes on it — which is why
# this is a label table and not a :class:`DatasetKind`. Category B exists
# upstream but is unreleased, so it has no row: a letter here is a letter
# ``foehn list --category`` and the MCP catalogue will offer. Both used to keep
# their own copy, one as a dict and one as a set.
CATEGORY_LABELS: dict[str, str] = {
    "A": "Ground-based measurements",
    "C": "Climate data",
    "D": "Radar data",
    "E": "Forecast data",
}

CATEGORIES = frozenset(CATEGORY_LABELS)


def options(labels: dict[str, str]) -> str:
    """One of the label tables as inline prose: ``"t" (10-minute, near real-time), …``.

    What a docstring's Args block wants, where the MCP guide wants bullets. Both
    are rendered from the table; neither is retyped beside it, which is the only
    way prose and a set can be checked against each other.
    """
    return ", ".join(f'"{token}" ({label})' for token, label in labels.items())


# MeteoSwiss chunks the high-frequency historical series by decade, so the slice
# is not always the trailing segment: ``ogd-smn_ber_t_historical_2000-2009.csv``.
# Only a bare decade range is accepted as that trailing segment, which keeps the
# "match the end, not anywhere" guard below intact.
_DECADE_RANGE_RE = re.compile(r"^\d{4}-\d{4}$")


def time_slice_from_filename(filename: str) -> str | None:
    """Return the time slice of a standard CSV asset, or None if it has none.

    Detects the time slice from the trailing ``_``-separated filename segment
    (e.g. ``ogd-smn_ber_d_recent.csv`` → ``"recent"``) rather than matching the
    token anywhere in the path, so a coincidental substring (notably the bare
    ``"now"``) elsewhere in the URL can't be misread as a time slice.

    The ``t`` and ``h`` historical series are split per decade, in which case the
    slice sits one segment further back (``..._historical_2000-2009.csv``). Those
    are recognised too — without it the file parses as "no slice at all" and gets
    treated as unsliced data that every query must include, so ``time_slice`` is
    silently ignored and the full history ships on every call.
    """
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    parts = stem.rsplit("_", 2)
    last = parts[-1]
    if last in TIME_SLICES:
        return last
    if len(parts) >= 2 and _DECADE_RANGE_RE.match(last) and parts[-2] in TIME_SLICES:
        return parts[-2]
    return None


def granularity_from_filename(filename: str) -> str | None:
    """Return the granularity segment of a standard CSV asset, or None.

    Standard assets are ``ogd-{key}_{station}_{granularity}[_{timeslice}].csv``,
    so the granularity is the third underscore-separated segment
    (``ogd-smn_ber_d_recent.csv`` -> ``"d"``). Returns None for the kinds whose
    filenames do not carry one, rather than guessing at whatever sits in that
    position.
    """
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    parts = stem.split("_")
    return parts[2] if len(parts) > 2 else None


# Forecast CSV assets are named ``vnut12.lssw.<YYYYMMDDHHMM>.<param>.csv``, where
# the middle field is the model run time — i.e. when the forecast was issued. Note
# this is NOT the "reference timestamp" of MeteoSwiss's docs, which is the ``Date``
# column inside the file (the end of each forecast step's aggregation interval).
_FORECAST_RUN_RE = re.compile(r"^vnut\d+\.lssw\.(\d{12})\..+\.csv$")


def forecast_run_from_filename(filename: str) -> str | None:
    """Return the run timestamp of a forecast CSV asset, or None if unrecognised.

    e.g. ``vnut12.lssw.202607210600.dkl010h0.csv`` → ``"202607210600"``. The
    zero-padded ``YYYYMMDDHHMM`` form sorts lexicographically, so callers can
    pick the newest run by ``max()`` without parsing it into a datetime.
    """
    match = _FORECAST_RUN_RE.match(filename.rsplit("/", 1)[-1])
    return match.group(1) if match else None
