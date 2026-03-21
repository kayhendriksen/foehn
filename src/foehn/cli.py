"""CLI entry point for the foehn package."""

from __future__ import annotations

import argparse
from pathlib import Path

from foehn.collections import (
    COLLECTIONS,
    GRIB2_COLLECTIONS,
    NETCDF_COLLECTIONS,
)
from foehn.convert import convert_climate_normals_to_parquet, convert_to_parquet
from foehn.download import (
    download_climate_normals_zip,
    download_collection,
    download_grib2,
    download_metadata,
    download_netcdf,
    load_last_run,
    save_last_run,
)


def main():
    parser = argparse.ArgumentParser(
        description="Download MeteoSwiss Open Data and convert to Parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Time range:
  (default)     recent      — Jan 1 this year → yesterday, updated daily at 12 UTC
  --historical  historical  — start of measurement → Dec 31 last year, updated once/year
  --now         now         — yesterday 12 UTC → now, updated every 10 min (t + h only)
  --all                     — historical + recent + now (full dataset)

Behaviour:
  --full-refresh            — ignore incremental tracking, re-download everything
  --convert-only            — convert existing CSVs to Parquet without downloading

Output:
  --grids                   — also fetch GRIB2, radar HDF5, NetCDF, GeoTIFF (large)
  --no-parquet              — skip conversion, keep raw CSVs only
  --data-dir PATH           — output root (default: ./data/meteoswiss)
""",
    )
    time_group = parser.add_argument_group("time range")
    time_group.add_argument(
        "--historical",
        action="store_true",
        help="Add 'historical' slice (start of measurement → Dec 31 last year)",
    )
    time_group.add_argument(
        "--now",
        action="store_true",
        help="Add 'now' realtime slice (yesterday 12 UTC → now, granularities t + h only)",
    )
    time_group.add_argument(
        "--all",
        action="store_true",
        help="Fetch all time slices: historical + recent + now",
    )
    behaviour_group = parser.add_argument_group("behaviour")
    behaviour_group.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignore _last_run.json timestamp, re-download all changed files",
    )
    behaviour_group.add_argument(
        "--convert-only",
        action="store_true",
        help="Skip downloading, only convert existing CSVs to Parquet",
    )
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--grids",
        action="store_true",
        help="Also download grid/binary data: ICON GRIB2, radar HDF5, NetCDF, GeoTIFF",
    )
    output_group.add_argument(
        "--no-parquet",
        action="store_true",
        help="Skip CSV/TXT → Parquet conversion",
    )
    output_group.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd() / "data" / "meteoswiss",
        help="Root data directory (default: ./data/meteoswiss)",
    )
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    parquet_dir = args.data_dir / "parquet"
    raw_dir.mkdir(parents=True, exist_ok=True)

    data_types = ["recent"]
    if args.all or args.now:
        data_types.append("now")
    if args.all or args.historical:
        data_types.insert(0, "historical")

    since = None
    if not args.full_refresh:
        since = load_last_run(args.data_dir)

    if since:
        print(f"Incremental update (last run: {since})", flush=True)
    else:
        print("Full download", flush=True)
    print(f"Data types: {data_types}", flush=True)

    if args.convert_only:
        # Just convert existing CSVs to Parquet, no downloads
        for key in COLLECTIONS:
            if key in GRIB2_COLLECTIONS or key in NETCDF_COLLECTIONS:
                continue
            convert_to_parquet(key, raw_dir, parquet_dir)
        convert_climate_normals_to_parquet(raw_dir, parquet_dir)
    else:
        for key in COLLECTIONS:
            if key in GRIB2_COLLECTIONS and not args.grids:
                continue
            if key in NETCDF_COLLECTIONS and not args.grids:
                continue
            if key not in GRIB2_COLLECTIONS and key not in NETCDF_COLLECTIONS:
                download_metadata(key, raw_dir)
            if key in GRIB2_COLLECTIONS:
                download_grib2(key, raw_dir, since=since)
            elif key in NETCDF_COLLECTIONS:
                download_netcdf(key, raw_dir)
            else:
                download_collection(key, raw_dir, data_types=data_types, since=since)
                if not args.no_parquet:
                    convert_to_parquet(key, raw_dir, parquet_dir)

        # C6 climate normals (ZIP from opendata.swiss, not STAC)
        download_climate_normals_zip(raw_dir, force=args.full_refresh)
        if not args.no_parquet:
            convert_climate_normals_to_parquet(raw_dir, parquet_dir)

        save_last_run(args.data_dir)

    print(f"\n\nRaw data saved to:      {raw_dir}")
    if not args.no_parquet:
        print(f"Parquet files saved to: {parquet_dir}")
    print("\nNext step: upload the parquet folder to a Databricks Volume.")
    print("  databricks fs cp -r data/meteoswiss/parquet/ \\")
    print("    dbfs:/Volumes/<catalog>/<schema>/meteoswiss/")
