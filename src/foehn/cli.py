"""CLI entry point for the foehn package."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from foehn.api import list_datasets
from foehn.client import (
    download_climate_normals_zip,
    download_collection,
    download_grib2,
    download_metadata,
    download_netcdf,
    load_last_run,
    save_last_run,
)
from foehn.collections import (
    COLLECTIONS,
    GRIB2_COLLECTIONS,
    NETCDF_COLLECTIONS,
)
from foehn.convert import convert_climate_normals_to_parquet, convert_to_parquet


def _resolve_data_dir(args_data_dir: Path | None) -> Path:
    if args_data_dir is not None:
        return args_data_dir
    env_dir = os.environ.get("FOEHN_DATA_DIR")
    return Path(env_dir) if env_dir else Path.cwd() / "data" / "meteoswiss"


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add flags shared across subcommands."""
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root data directory (default: $FOEHN_DATA_DIR or ./data/meteoswiss)",
    )


def _add_key_arg(parser: argparse.ArgumentParser) -> None:
    """Add optional positional KEY arguments."""
    parser.add_argument(
        "keys",
        nargs="*",
        metavar="KEY",
        help="Dataset key(s) to operate on (default: all). Use 'foehn list' to see options.",
    )


def _resolve_keys(keys: list[str], *, allow_grids: bool = False) -> list[str]:
    """Resolve key arguments to a list of collection keys."""
    if keys:
        for k in keys:
            if k not in COLLECTIONS:
                print(f"Error: unknown dataset key {k!r}. Run 'foehn list' to see options.", file=sys.stderr)
                sys.exit(1)
        return keys
    # Default: all collections (skip grids unless opted in)
    if allow_grids:
        return list(COLLECTIONS)
    return [k for k in COLLECTIONS if k not in GRIB2_COLLECTIONS and k not in NETCDF_COLLECTIONS]


def cmd_list(args: argparse.Namespace) -> None:
    rows = list_datasets()
    print(f"{'Category':<16} {'Key':<30} Collection ID")
    print("-" * 90)
    for row in rows:
        print(f"{row['category']:<16} {row['key']:<30} {row['collection_id']}")


def cmd_download(args: argparse.Namespace) -> None:
    data_dir = _resolve_data_dir(args.data_dir)
    raw_dir = data_dir / "raw"
    parquet_dir = data_dir / "parquet"
    raw_dir.mkdir(parents=True, exist_ok=True)

    full_refresh = args.full_refresh or os.environ.get("FOEHN_FULL_REFRESH", "").lower() in ("1", "true", "yes")

    data_types = ["recent"]
    if args.all or args.now:
        data_types.append("now")
    if args.all or args.historical:
        data_types.insert(0, "historical")

    since = None
    if not full_refresh:
        since = load_last_run(data_dir)

    if since:
        print(f"Incremental update (last run: {since})", flush=True)
    else:
        print("Full download", flush=True)
    print(f"Data types: {data_types}", flush=True)

    keys = _resolve_keys(args.keys, allow_grids=args.grids)

    for key in keys:
        if key in GRIB2_COLLECTIONS:
            download_grib2(key, raw_dir, since=since)
        elif key in NETCDF_COLLECTIONS:
            download_netcdf(key, raw_dir)
        else:
            download_metadata(key, raw_dir)
            download_collection(key, raw_dir, data_types=data_types, since=since)
            if not args.no_parquet:
                convert_to_parquet(key, raw_dir, parquet_dir)

    # C6 climate normals (ZIP from opendata.swiss, not STAC)
    if not args.keys:
        download_climate_normals_zip(raw_dir, force=full_refresh)
        if not args.no_parquet:
            convert_climate_normals_to_parquet(raw_dir, parquet_dir)

    save_last_run(data_dir)

    print(f"\nRaw data saved to:      {raw_dir}")
    if not args.no_parquet:
        print(f"Parquet files saved to: {parquet_dir}")


def cmd_convert(args: argparse.Namespace) -> None:
    data_dir = _resolve_data_dir(args.data_dir)
    raw_dir = data_dir / "raw"
    parquet_dir = data_dir / "parquet"

    keys = _resolve_keys(args.keys)

    for key in keys:
        if key in GRIB2_COLLECTIONS or key in NETCDF_COLLECTIONS:
            continue
        convert_to_parquet(key, raw_dir, parquet_dir)

    if not args.keys:
        convert_climate_normals_to_parquet(raw_dir, parquet_dir)

    print(f"Parquet files saved to: {parquet_dir}")


def cmd_load(args: argparse.Namespace) -> None:
    from foehn.api import load

    kwargs: dict = {}
    if args.station:
        kwargs["station"] = args.station
    if args.granularity:
        kwargs["granularity"] = args.granularity
    if args.data_types:
        kwargs["data_types"] = args.data_types

    df = load(args.key, **kwargs)

    n = args.n or 20
    print(df.head(n))
    print(f"\n[{df.shape[0]} rows x {df.shape[1]} columns]")


def main():
    parser = argparse.ArgumentParser(
        prog="foehn",
        description="Download MeteoSwiss Open Data and convert to Parquet.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- foehn list ---
    sub_list = subparsers.add_parser("list", help="List available datasets")
    _add_common_args(sub_list)
    sub_list.set_defaults(func=cmd_list)

    # --- foehn download ---
    sub_dl = subparsers.add_parser("download", help="Download datasets")
    _add_key_arg(sub_dl)
    _add_common_args(sub_dl)
    sub_dl.add_argument("--historical", action="store_true", help="Include historical time slice")
    sub_dl.add_argument("--now", action="store_true", help="Include realtime 'now' time slice")
    sub_dl.add_argument("--all", action="store_true", help="Include all time slices (historical + recent + now)")
    sub_dl.add_argument("--full-refresh", action="store_true", help="Ignore incremental tracking, re-download all")
    sub_dl.add_argument("--grids", action="store_true", help="Include grid/binary datasets (GRIB2, NetCDF)")
    sub_dl.add_argument("--no-parquet", action="store_true", help="Skip CSV → Parquet conversion")
    sub_dl.set_defaults(func=cmd_download)

    # --- foehn convert ---
    sub_conv = subparsers.add_parser("convert", help="Convert downloaded CSVs to Parquet")
    _add_key_arg(sub_conv)
    _add_common_args(sub_conv)
    sub_conv.set_defaults(func=cmd_convert)

    # --- foehn load ---
    sub_load = subparsers.add_parser("load", help="Load a dataset and print a preview")
    sub_load.add_argument("key", help="Dataset key (e.g. 'smn')")
    sub_load.add_argument("--station", nargs="+", help="Filter by station(s)")
    sub_load.add_argument("--granularity", nargs="+", help="Filter by granularity (t, h, d, m, y)")
    sub_load.add_argument("--data-types", nargs="+", help="Time slices (default: recent)")
    sub_load.add_argument("-n", type=int, default=None, help="Number of rows to show (default: 20)")
    sub_load.set_defaults(func=cmd_load)

    args = parser.parse_args()
    args.func(args)
