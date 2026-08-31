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
from enum import StrEnum

STAC_API_BASE = "https://data.geo.admin.ch/api/stac/v1"

# Maps our short key → STAC collection ID.
# Organised by MeteoSwiss category (A–E) and data format.
#
# FORMAT LEGEND:
#   CSV collections     → downloaded as CSV, converted to Parquet
#   GRIB2 collections   → binary grid (opt-in via --grids)
#   NETCDF collections  → binary grid (NetCDF/GeoTIFF/ZIP), always downloaded
#   C6 climate normals  → separate ZIP from opendata.swiss (not on STAC API)
COLLECTIONS = {
    # ── A: Ground-based measurements (CSV, time-sliced: historical/recent/now) ──
    "smn": "ch.meteoschweiz.ogd-smn",  # A1 — Automatic weather stations (t,h,d,m,y)
    "smn_precip": "ch.meteoschweiz.ogd-smn-precip",  # A2 — Automatic precipitation stations (t,h,d,m,y)
    "smn_tower": "ch.meteoschweiz.ogd-smn-tower",  # A3 — Automatic tower stations (t,h,d,m,y)
    # A4 — Automatic soil moisture stations                     NOT YET RELEASED
    "nime": "ch.meteoschweiz.ogd-nime",  # A5 — Manual precipitation stations (d,m,y)
    "tot": "ch.meteoschweiz.ogd-tot",  # A6 — Totaliser precipitation (y, no time-slice)
    "obs": "ch.meteoschweiz.ogd-obs",  # A8 — Meteorological visual observations (d,m,y)
    "pollen": "ch.meteoschweiz.ogd-pollen",  # A7 — Pollen stations (h,d,y)
    "phenology": "ch.meteoschweiz.ogd-phenology",  # A9 — Phenological observations (y, no time-slice)
    # ── B: Atmosphere measurements                             NOT YET RELEASED ──
    # B1 Radio soundings, B2 RALMO, B3 Ceilometer, B4-B5 Ozone, B6 SACRaM
    # ── C: Climate data ──────────────────────────────────────────────────────────
    # C1/C2 — Homogeneous series (CSV, time-sliced: historical/recent)
    "nbcn": "ch.meteoschweiz.ogd-nbcn",  # C1 — Climate stations, homogeneous (d,m,y)
    "nbcn_precip": "ch.meteoschweiz.ogd-nbcn-precip",  # C2 — Climate precipitation, homogeneous (m,y)
    # C3/C4/C5 — Spatial climate analyses (NetCDF, static grids)
    "surface_derived_grid": "ch.meteoschweiz.ogd-surface-derived-grid",  # C3 — Precipitation, temperature, sunshine
    "satellite_derived_grid": "ch.meteoschweiz.ogd-satellite-derived-grid",  # C4 — Radiation, clouds, LST
    "radar_derived_grid": "ch.meteoschweiz.ogd-radar-derived-grid",  # C5 — Hail (days + return periods), radar-derived
    # C6 — Station normals. The only dataset with no STAC collection: it ships as a
    # single ZIP under this geo.admin identifier, see CLIMATE_NORMALS_ZIP_URL.
    "climate_normals": "ch.meteoschweiz.klima",
    # C7 — Spatial climate normals (NetCDF/GeoTIFF, static reference grids)
    # Supersedes the retired per-parameter klimanormwerte-* map layers: the same
    # temp/precip/sun normals (1991-2020 and 1961-1990) are assets on this
    # collection's single "ch" item, e.g. match="tnormy9120".
    "climate_normals_grid": "ch.meteoschweiz.ogd-climate-normals-grid",  # C7 — Spatial normals (NetCDF)
    # C8 — Climate scenarios CH2025 local (CSV, no time-slice)
    "climate_scenarios": "ch.meteoschweiz.ogd-climate-scenarios-ch2025",
    # C9 — Climate scenarios CH2025 gridded (NetCDF, static)
    "climate_scenarios_grid": "ch.meteoschweiz.ogd-climate-scenarios-ch2025-grid",
    # NOTE: the hagelgefaehrdung-korngroesse_* hazard maps were retired from the
    # STAC API; their return-period grids are now assets on radar_derived_grid's
    # "archive-ch" item, e.g. match="returnperiod050yleha1".
    # ── Indoor climate scenarios (ZIP, static) ───────────────────────────────────
    "climate_scenarios_indoor": "ch.meteoschweiz.klimaszenarien-raumklima",
    # ── D: Radar data (HDF5, no time-slice, opt-in via --grids) ─────────────────
    "radar_precip": "ch.meteoschweiz.ogd-radar-precip",  # D1 — Precipitation radar (5-10 min)
    "radar_hail": "ch.meteoschweiz.ogd-radar-hail",  # D3 — Hail radar (5 min)
    # D2 Reflectivity, D4 Convection, D5 Polar 3D                NOT YET RELEASED
    # ── E: Forecast data ─────────────────────────────────────────────────────────
    "forecast_icon_ch1": "ch.meteoschweiz.ogd-forecasting-icon-ch1",  # E2 — ICON-CH1-EPS 1km GRIB2
    "forecast_icon_ch2": "ch.meteoschweiz.ogd-forecasting-icon-ch2",  # E3 — ICON-CH2-EPS 2.1km GRIB2
    "forecast_local": "ch.meteoschweiz.ogd-local-forecasting",  # E4 — Local point forecasts (CSV)
    "analysis_kenda_ch1": "ch.meteoschweiz.ogd-analysis-kenda-ch1",  # E5 — KENDA-CH1 numerical weather analysis GRIB2
    # E1 — Short-term forecast (nowcasting)                       NOT YET RELEASED
}

# Per-collection metadata: description, MeteoSwiss category, format, granularities,
# and available time slices.  Used by list_datasets() to expose a rich catalog.
COLLECTION_META: dict[str, dict] = {
    # ── A: Ground-based measurements ─────────────────────────────────────────
    "smn": {
        "category": "A",
        "subcategory": "A1",
        "description": "Automatic weather stations",
        "format": "CSV",
        "frequencies": ["t", "h", "d", "m", "y"],
        "time_slices": ["historical", "recent", "now"],
    },
    "smn_precip": {
        "category": "A",
        "subcategory": "A2",
        "description": "Automatic precipitation stations",
        "format": "CSV",
        "frequencies": ["t", "h", "d", "m", "y"],
        "time_slices": ["historical", "recent", "now"],
    },
    "smn_tower": {
        "category": "A",
        "subcategory": "A3",
        "description": "Automatic tower stations",
        "format": "CSV",
        "frequencies": ["t", "h", "d", "m", "y"],
        "time_slices": ["historical", "recent", "now"],
    },
    "nime": {
        "category": "A",
        "subcategory": "A5",
        "description": "Manual precipitation stations",
        "format": "CSV",
        "frequencies": ["d", "m", "y"],
        "time_slices": ["historical", "recent"],
    },
    "tot": {
        "category": "A",
        "subcategory": "A6",
        "description": "Totaliser precipitation",
        "format": "CSV",
        "frequencies": ["y"],
        "time_slices": [],
    },
    "obs": {
        "category": "A",
        "subcategory": "A8",
        "description": "Meteorological visual observations",
        "format": "CSV",
        "frequencies": ["d", "m", "y"],
        "time_slices": ["historical", "recent"],
    },
    "pollen": {
        "category": "A",
        "subcategory": "A7",
        "description": "Pollen stations",
        "format": "CSV",
        "frequencies": ["h", "d", "y"],
        # "now" belongs here: MeteoSwiss publishes ogd-pollen_*_h_now.csv for the
        # hourly series. Omitting it told list_datasets() callers — including the
        # MCP tools — that live pollen counts were unavailable, so nothing ever
        # asked for the slice that does in fact exist.
        "time_slices": ["historical", "recent", "now"],
    },
    "phenology": {
        "category": "A",
        "subcategory": "A9",
        "description": "Phenological observations",
        "format": "CSV",
        "frequencies": ["y"],
        "time_slices": [],
    },
    # ── C: Climate data ──────────────────────────────────────────────────────
    "nbcn": {
        "category": "C",
        "subcategory": "C1",
        "description": "Climate stations, homogeneous",
        "format": "CSV",
        "frequencies": ["d", "m", "y"],
        "time_slices": ["historical", "recent"],
    },
    "nbcn_precip": {
        "category": "C",
        "subcategory": "C2",
        "description": "Climate precipitation, homogeneous",
        "format": "CSV",
        "frequencies": ["m", "y"],
        "time_slices": [],
    },
    "surface_derived_grid": {
        "category": "C",
        "subcategory": "C3",
        "description": "Precipitation, temperature, sunshine grids",
        "format": "NetCDF",
        "frequencies": [],
        "time_slices": [],
    },
    "satellite_derived_grid": {
        "category": "C",
        "subcategory": "C4",
        "description": "Radiation, clouds, land surface temperature grids",
        "format": "NetCDF",
        "frequencies": [],
        "time_slices": [],
    },
    "radar_derived_grid": {
        "category": "C",
        "subcategory": "C5",
        "description": "Hail spatial climate analyses (hail days, return periods)",
        "format": "NetCDF",
        "frequencies": [],
        "time_slices": [],
    },
    "climate_normals": {
        "category": "C",
        "subcategory": "C6",
        "description": "Station climate normals, 1961-1990 and 1991-2020 (monthly + yearly)",
        "format": "TXT",
        # No reader, so no filter vocabulary to advertise: this dataset downloads
        # and converts to Parquet but is not reachable from load().
        "frequencies": [],
        "time_slices": [],
    },
    "climate_normals_grid": {
        "category": "C",
        "subcategory": "C7",
        "description": "Spatial climate normals (temp/precip/sunshine + radiation/clouds)",
        "format": "NetCDF",
        "frequencies": [],
        "time_slices": [],
    },
    # NOTE: collections whose filenames carry no granularity segment
    # (NO_GRANULARITY/CSV_ZIP) advertise no frequencies — the ``frequency``
    # filter is unsupported there, and this field doubles as its valid values.
    "climate_scenarios": {
        "category": "C",
        "subcategory": "C8",
        "description": "Climate scenarios CH2025 local (daily)",
        "format": "CSV",
        "frequencies": [],
        "time_slices": [],
    },
    "climate_scenarios_grid": {
        "category": "C",
        "subcategory": "C9",
        "description": "Climate scenarios CH2025 gridded",
        "format": "NetCDF",
        "frequencies": [],
        "time_slices": [],
    },
    "climate_scenarios_indoor": {
        "category": "C",
        "subcategory": "C",
        "description": "Indoor climate scenarios (hourly)",
        "format": "CSV",
        "frequencies": [],
        "time_slices": [],
    },
    # ── D: Radar data ────────────────────────────────────────────────────────
    "radar_precip": {
        "category": "D",
        "subcategory": "D1",
        "description": "Precipitation radar",
        "format": "HDF5",
        "frequencies": [],
        "time_slices": [],
    },
    "radar_hail": {
        "category": "D",
        "subcategory": "D3",
        "description": "Hail radar",
        "format": "HDF5",
        "frequencies": [],
        "time_slices": [],
    },
    # ── E: Forecast data ─────────────────────────────────────────────────────
    "forecast_icon_ch1": {
        "category": "E",
        "subcategory": "E2",
        "description": "ICON-CH1-EPS 1km",
        "format": "GRIB2",
        "frequencies": [],
        "time_slices": [],
    },
    "forecast_icon_ch2": {
        "category": "E",
        "subcategory": "E3",
        "description": "ICON-CH2-EPS 2.1km",
        "format": "GRIB2",
        "frequencies": [],
        "time_slices": [],
    },
    "forecast_local": {
        "category": "E",
        "subcategory": "E4",
        "description": "Local point forecasts (hourly + daily)",
        "format": "CSV",
        "frequencies": [],
        "time_slices": [],
    },
    "analysis_kenda_ch1": {
        "category": "E",
        "subcategory": "E5",
        "description": "Numerical weather analysis KENDA-CH1",
        "format": "GRIB2",
        "frequencies": [],
        "time_slices": [],
    },
}

# These sets control how each collection is downloaded and processed.
# A collection belongs to at most one of these sets.
# Everything not in any set is treated as a CSV collection (downloaded + Parquet).


# CSV collections that DON'T use "recent"/"historical"/"now" filename suffixes.
# These get all CSVs regardless of data_types filter. Only latest item is kept.
class DatasetKind(StrEnum):
    """Which pipeline handles a dataset: its download, convert and load paths.

    Distinct from ``format`` in COLLECTION_META, which says what the *bytes* are
    and is public. Several kinds share one format — smn, climate_scenarios and
    forecast_local are all CSV but need three different readers — so format
    cannot carry the routing. Kind is internal; nothing surfaces it to callers.
    """

    STANDARD_CSV = "standard_csv"
    """Per-station CSV assets split by time slice and granularity. The main path."""

    PREAMBLE_CSV = "preamble_csv"
    """CSV behind a ``KEY;VALUE`` metadata preamble, on nominal 30-year dates."""

    ARCHIVE_CSV = "archive_csv"
    """A single ZIP of per-station CSVs rather than per-station STAC assets."""

    FORECAST_CSV = "forecast_csv"
    """Point forecasts named by run, with no time slice and no reference_timestamp."""

    NETCDF_GRID = "netcdf_grid"
    """Static gridded analyses, normals and scenarios. Combines across files."""

    GRIB2_GRID = "grib2_grid"
    """ICON/KENDA on an unstructured grid, one field per file."""

    RADAR_GRID = "radar_grid"
    """ODIM-H5 Cartesian composites, one per timestep."""

    DIRECT_ZIP = "direct_zip"
    """A ZIP fetched from a fixed URL rather than listed on the STAC API.

    Climate normals only. It has a download path and a convert path but no
    reader: the TXT files are a wide per-parameter table keyed by station *name*
    with twelve month columns and no timestamp, so what a normals DataFrame
    should look like is an open question rather than a missing adapter.
    """


# One row per dataset. Adding a collection means adding it here and to
# COLLECTIONS; a kind missing from either is caught by the tests.
KIND_OF: dict[str, DatasetKind] = {
    # A: ground-based measurements
    "smn": DatasetKind.STANDARD_CSV,
    "smn_precip": DatasetKind.STANDARD_CSV,
    "smn_tower": DatasetKind.STANDARD_CSV,
    "nime": DatasetKind.STANDARD_CSV,
    "tot": DatasetKind.STANDARD_CSV,
    "obs": DatasetKind.STANDARD_CSV,
    "pollen": DatasetKind.STANDARD_CSV,
    "phenology": DatasetKind.STANDARD_CSV,
    # C: climate
    "nbcn": DatasetKind.STANDARD_CSV,
    "nbcn_precip": DatasetKind.STANDARD_CSV,
    "surface_derived_grid": DatasetKind.NETCDF_GRID,
    "satellite_derived_grid": DatasetKind.NETCDF_GRID,
    "radar_derived_grid": DatasetKind.NETCDF_GRID,
    "climate_normals_grid": DatasetKind.NETCDF_GRID,
    "climate_scenarios_grid": DatasetKind.NETCDF_GRID,
    "climate_normals": DatasetKind.DIRECT_ZIP,
    "climate_scenarios": DatasetKind.PREAMBLE_CSV,
    "climate_scenarios_indoor": DatasetKind.ARCHIVE_CSV,
    # D: radar
    "radar_precip": DatasetKind.RADAR_GRID,
    "radar_hail": DatasetKind.RADAR_GRID,
    # E: forecasts
    "forecast_icon_ch1": DatasetKind.GRIB2_GRID,
    "forecast_icon_ch2": DatasetKind.GRIB2_GRID,
    "analysis_kenda_ch1": DatasetKind.GRIB2_GRID,
    "forecast_local": DatasetKind.FORECAST_CSV,
}

# Kinds whose filenames do not carry the standard
# ogd-{key}_{station}_{granularity}[_{timeslice}].csv pattern, so there is no
# granularity to filter on. Derived from the kind rather than listed separately,
# which is what NO_GRANULARITY_COLLECTIONS used to be.
NO_GRANULARITY_KINDS = frozenset({DatasetKind.PREAMBLE_CSV, DatasetKind.FORECAST_CSV})

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
