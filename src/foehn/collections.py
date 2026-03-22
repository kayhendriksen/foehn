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
  E  Forecast data                   — GRIB2 (ICON models) + CSV (local forecasts)
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
    "smn": "ch.meteoschweiz.ogd-smn",  # A1 — Automatic weather stations (t,h,d,m)
    "smn_precip": "ch.meteoschweiz.ogd-smn-precip",  # A2 — Automatic precipitation stations (t,h,d,m)
    "smn_tower": "ch.meteoschweiz.ogd-smn-tower",  # A3 — Automatic tower stations (t,h,d,m)
    # A4 — Automatic soil moisture stations                     NOT YET RELEASED
    "nime": "ch.meteoschweiz.ogd-nime",  # A5 — Manual precipitation stations (d,m,y)
    "tot": "ch.meteoschweiz.ogd-tot",  # A6 — Totaliser precipitation (y, no time-slice)
    "obs": "ch.meteoschweiz.ogd-obs",  # A8 — Meteorological visual observations (t,m,y)
    "pollen": "ch.meteoschweiz.ogd-pollen",  # A7 — Pollen stations (h,d)
    "phenology": "ch.meteoschweiz.ogd-phenology",  # A9 — Phenological observations (y, no time-slice)
    # ── B: Atmosphere measurements                             NOT YET RELEASED ──
    # B1 Radio soundings, B2 RALMO, B3 Ceilometer, B4-B5 Ozone, B6 SACRaM
    # ── C: Climate data ──────────────────────────────────────────────────────────
    # C1/C2 — Homogeneous series (CSV, time-sliced: historical/recent)
    "nbcn": "ch.meteoschweiz.ogd-nbcn",  # C1 — Climate stations, homogeneous (d,m)
    "nbcn_precip": "ch.meteoschweiz.ogd-nbcn-precip",  # C2 — Climate precipitation, homogeneous (m)
    # C3/C4/C5 — Spatial climate analyses (NetCDF, static grids)
    "surface_derived_grid": "ch.meteoschweiz.ogd-surface-derived-grid",  # C3/C4 — Precipitation, temperature, sunshine
    "satellite_derived_grid": "ch.meteoschweiz.ogd-satellite-derived-grid",  # C5 — Radiation, clouds
    # C6 — Climate normals → downloaded separately as ZIP, see CLIMATE_NORMALS_ZIP_URL
    # C7 — Spatial climate normals (NetCDF/GeoTIFF, static reference grids)
    "climate_normals_precip_9120": "ch.meteoschweiz.klimanormwerte-niederschlag_aktuelle_periode",
    "climate_normals_sun_9120": "ch.meteoschweiz.klimanormwerte-sonnenscheindauer_aktuelle_periode",
    "climate_normals_temp_9120": "ch.meteoschweiz.klimanormwerte-temperatur_aktuelle_periode",
    "climate_normals_precip_6190": "ch.meteoschweiz.klimanormwerte-niederschlag_1961_1990",
    "climate_normals_sun_6190": "ch.meteoschweiz.klimanormwerte-sonnenscheindauer_1961_1990",
    "climate_normals_temp_6190": "ch.meteoschweiz.klimanormwerte-temperatur_1961_1990",
    # C8 — Climate scenarios CH2025 local (CSV, no time-slice)
    "climate_scenarios": "ch.meteoschweiz.ogd-climate-scenarios-ch2025",
    # C9 — Climate scenarios CH2025 gridded (NetCDF, static)
    "climate_scenarios_grid": "ch.meteoschweiz.ogd-climate-scenarios-ch2025-grid",
    # ── Hail hazard maps (NetCDF+ZIP, static reference grids) ────────────────────
    "hail_hazard_10y": "ch.meteoschweiz.hagelgefaehrdung-korngroesse_10_jahre",
    "hail_hazard_20y": "ch.meteoschweiz.hagelgefaehrdung-korngroesse_20_jahre",
    "hail_hazard_50y": "ch.meteoschweiz.hagelgefaehrdung-korngroesse_50_jahre",
    "hail_hazard_100y": "ch.meteoschweiz.hagelgefaehrdung-korngroesse_100_jahre",
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
    # E1 — Short-term forecast (nowcasting)                       NOT YET RELEASED
}

# These sets control how each collection is downloaded and processed.
# A collection belongs to at most one of these sets.
# Everything not in any set is treated as a CSV collection (downloaded + Parquet).

# CSV collections that DON'T use "recent"/"historical"/"now" filename suffixes.
# These get all CSVs regardless of data_types filter. Only latest item is kept.
FORECAST_CSV_COLLECTIONS = {"forecast_local"}

# Binary grid collections (HDF5/GRIB2). Large, opt-in via --grids flag.
# Downloaded as binary blobs, NOT converted to Parquet.
GRIB2_COLLECTIONS = {
    "forecast_icon_ch1",
    "forecast_icon_ch2",
    "radar_precip",
    "radar_hail",
}

# Spatial/static collections (NetCDF, GeoTIFF, ZIP). Always downloaded.
# NOT converted to Parquet — these are gridded/spatial data, not tabular.
NETCDF_COLLECTIONS = {
    "surface_derived_grid",
    "satellite_derived_grid",
    "climate_scenarios_grid",
    "climate_normals_precip_9120",
    "climate_normals_sun_9120",
    "climate_normals_temp_9120",
    "climate_normals_precip_6190",
    "climate_normals_sun_6190",
    "climate_normals_temp_6190",
    "hail_hazard_10y",
    "hail_hazard_20y",
    "hail_hazard_50y",
    "hail_hazard_100y",
    "climate_scenarios_indoor",
}


def discover() -> list[dict[str, str]]:
    """Return metadata about all available collections.

    Returns a list of dicts with keys: ``category``, ``key``, ``collection_id``.
    """
    result = []
    for key, collection_id in COLLECTIONS.items():
        if key in GRIB2_COLLECTIONS:
            category = "GRIB2"
        elif key in NETCDF_COLLECTIONS:
            category = "NetCDF"
        elif key in FORECAST_CSV_COLLECTIONS:
            category = "CSV (forecast)"
        else:
            category = "CSV"
        result.append({"category": category, "key": key, "collection_id": collection_id})
    return result


# C6 climate normals — separate ZIP from opendata.swiss (not on STAC API).
# Contains 112 TXT files (tab-separated, Latin-1 encoding) with monthly/yearly
# station normals for periods 1961-1990 and 1991-2020. Converted to Parquet.
CLIMATE_NORMALS_ZIP_URL = "https://data.geo.admin.ch/ch.meteoschweiz.klima/normwerte/normwerte.zip"
