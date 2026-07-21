"""
MeteoSwiss STAC collection IDs, routing sets, and constants.

TIME SLICES
-----------
MeteoSwiss splits CSV data into three time slices, encoded in the filename:

  "historical"  — From start of measurement → Dec 31 of last year
                   Updated once a year (early January).
                   Available for granularities: t, h, d, m

  "recent"      — From Jan 1 of this year → yesterday
                   Updated daily at 12:00 UTC.
                   Available for granularities: t, h, d, m

  "now"          — From yesterday 12:00 UTC → now
                   Updated every 10 minutes.
                   Available for granularities: t, h only

  (no type)      — Some data (e.g. yearly "y" granularity, phenology, totaliser)
                   don't use this split. These files have no time-slice suffix.

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
  A  Ground-based measurements       — CSV, time-sliced (historical/recent/now)
  B  Atmosphere measurements         — NOT YET RELEASED (radio soundings etc.)
  C  Climate data                    — CSV + NetCDF + TXT (varies by sub-type)
  D  Radar data                      — HDF5 (binary grid data)
  E  Forecast data                   — GRIB2 (ICON models + KENDA analysis) + CSV (local forecasts)

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
    # C6 — Climate normals → downloaded separately as ZIP, see CLIMATE_NORMALS_ZIP_URL
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
        "time_slices": ["historical", "recent"],
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
        "format": "CSV+ZIP",
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
FORECAST_CSV_COLLECTIONS = {"forecast_local"}

# Collections whose CSV filenames do NOT follow the standard
# ogd-{key}_{station}_{granularity}[_{timeslice}].csv pattern.
# Granularity filtering is not supported for these.
NO_GRANULARITY_COLLECTIONS = {"forecast_local", "climate_scenarios"}

# Binary grid collections (HDF5/GRIB2). Large, opt-in via --grids flag.
# Downloaded as binary blobs, NOT converted to Parquet.
GRIB2_COLLECTIONS = {
    "forecast_icon_ch1",
    "forecast_icon_ch2",
    "analysis_kenda_ch1",
    "radar_precip",
    "radar_hail",
}

# Spatial/static collections (NetCDF, GeoTIFF, ZIP). Always downloaded.
# NOT converted to Parquet — these are gridded/spatial data, not tabular.
NETCDF_COLLECTIONS = {
    "surface_derived_grid",
    "satellite_derived_grid",
    "radar_derived_grid",
    "climate_scenarios_grid",
    "climate_normals_grid",
}

# Tabular collections delivered as a single ZIP of CSVs (not per-station STAC
# assets), with their own separator/timestamp layout. Like the C6 climate
# normals, these get a bespoke download+convert path rather than the standard
# CSV flow.
CSV_ZIP_COLLECTIONS = {
    "climate_scenarios_indoor",
}

# CSV collections whose files carry a multi-row "KEY;VALUE" metadata preamble
# before the real "DATE;<model>;..." table (CH2025 climate scenarios). The
# standard reader would treat the preamble as the header, so these need a
# preamble-skipping parser. Downloads are standard per-file CSVs; only the
# parse differs.
PREAMBLE_CSV_COLLECTIONS = {
    "climate_scenarios",
}


# (The gridded read path's per-format engine/suffix config now lives in
# foehn.grids._GRID_READERS, keyed off each collection's COLLECTION_META format.)


# C6 climate normals — separate ZIP from opendata.swiss (not on STAC API).
# Contains 112 TXT files (tab-separated, Latin-1 encoding) with monthly/yearly
# station normals for periods 1961-1990 and 1991-2020. Converted to Parquet.
CLIMATE_NORMALS_ZIP_URL = "https://data.geo.admin.ch/ch.meteoschweiz.klima/normwerte/normwerte.zip"


# Time-slice tokens that appear as the trailing filename segment of standard
# CSV assets (ogd-{key}_{station}_{granularity}_{timeslice}.csv).
TIME_SLICES = frozenset({"historical", "recent", "now"})


def time_slice_from_filename(filename: str) -> str | None:
    """Return the time slice of a standard CSV asset, or None if it has none.

    Detects the time slice from the trailing ``_``-separated filename segment
    (e.g. ``ogd-smn_ber_d_recent.csv`` → ``"recent"``) rather than matching the
    token anywhere in the path, so a coincidental substring (notably the bare
    ``"now"``) elsewhere in the URL can't be misread as a time slice.
    """
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    last = stem.rsplit("_", 1)[-1]
    return last if last in TIME_SLICES else None
