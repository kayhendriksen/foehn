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


def _add_dataset_arg(parser: argparse.ArgumentParser) -> None:
    """Add optional positional DATASET arguments."""
    parser.add_argument(
        "datasets",
        nargs="*",
        metavar="DATASET",
        help="Dataset(s) to operate on (default: all). Use 'foehn list' to see options.",
    )


def _resolve_datasets(datasets: list[str], *, allow_grids: bool = False) -> list[str]:
    """Resolve dataset arguments to a list of collection keys."""
    if datasets:
        for d in datasets:
            if d not in COLLECTIONS:
                print(f"Error: unknown dataset {d!r}. Run 'foehn list' to see options.", file=sys.stderr)
                sys.exit(1)
        return datasets
    # Default: all collections (skip grids unless opted in)
    if allow_grids:
        return list(COLLECTIONS)
    return [k for k in COLLECTIONS if k not in GRIB2_COLLECTIONS and k not in NETCDF_COLLECTIONS]


def cmd_list(args: argparse.Namespace) -> None:
    rows = list_datasets()

    if args.category:
        cat = args.category.upper()
        rows = [r for r in rows if r["category"] == cat]
    if args.format:
        fmt = args.format.upper()
        rows = [r for r in rows if r["format"].upper() == fmt]

    if not rows:
        print("No datasets match the given filters.")
        return

    # Group by category
    categories = {
        "A": "Ground-based measurements",
        "C": "Climate data",
        "D": "Radar data",
        "E": "Forecast data",
    }

    current_cat = None
    for row in rows:
        cat = row["category"]
        if cat != current_cat:
            if current_cat is not None:
                print()
            label = categories.get(cat, cat)
            print(f"── {cat}: {label} ──")
            print(f"  {'Dataset':<32} {'Format':<8} {'Frequency':<16} Description")
            current_cat = cat

        frequencies = ", ".join(row["frequencies"]) if row["frequencies"] else "—"
        print(f"  {row['dataset']:<32} {row['format']:<8} {frequencies:<16} {row['description']}")


def cmd_download(args: argparse.Namespace) -> None:
    data_dir = _resolve_data_dir(args.data_dir)
    raw_dir = data_dir / "raw"
    parquet_dir = data_dir / "parquet"
    raw_dir.mkdir(parents=True, exist_ok=True)

    full_refresh = args.full_refresh or os.environ.get("FOEHN_FULL_REFRESH", "").lower() in ("1", "true", "yes")

    time_slices = ["recent"]
    if args.all or args.now:
        time_slices.append("now")
    if args.all or args.historical:
        time_slices.insert(0, "historical")

    since = None
    if not full_refresh:
        since = load_last_run(data_dir)

    if since:
        print(f"Incremental update (last run: {since})", flush=True)
    else:
        print("Full download", flush=True)
    print(f"Time slices: {time_slices}", flush=True)

    datasets = _resolve_datasets(args.datasets, allow_grids=args.grids)

    for ds in datasets:
        if ds in GRIB2_COLLECTIONS:
            download_grib2(ds, raw_dir, since=since)
        elif ds in NETCDF_COLLECTIONS:
            download_netcdf(ds, raw_dir)
        else:
            download_metadata(ds, raw_dir)
            download_collection(ds, raw_dir, data_types=time_slices, since=since)
            if not args.no_parquet:
                convert_to_parquet(ds, raw_dir, parquet_dir)

    # C6 climate normals (ZIP from opendata.swiss, not STAC)
    if not args.datasets:
        download_climate_normals_zip(raw_dir, force=full_refresh)
        if not args.no_parquet:
            convert_climate_normals_to_parquet(raw_dir, parquet_dir)

    save_last_run(data_dir)

    print(f"\nRaw data saved to:      {raw_dir}")
    if not args.no_parquet:
        print(f"Parquet files saved to: {parquet_dir}")


def cmd_to_parquet(args: argparse.Namespace) -> None:
    data_dir = _resolve_data_dir(args.data_dir)
    raw_dir = data_dir / "raw"
    parquet_dir = data_dir / "parquet"

    datasets = _resolve_datasets(args.datasets)

    for ds in datasets:
        if ds in GRIB2_COLLECTIONS or ds in NETCDF_COLLECTIONS:
            continue
        convert_to_parquet(ds, raw_dir, parquet_dir)

    if not args.datasets:
        convert_climate_normals_to_parquet(raw_dir, parquet_dir)

    print(f"Parquet files saved to: {parquet_dir}")


def cmd_load(args: argparse.Namespace) -> None:
    from foehn.api import load

    kwargs: dict = {}
    if args.station:
        kwargs["station"] = args.station
    if args.frequency:
        kwargs["frequency"] = args.frequency
    if args.time_slice:
        kwargs["time_slice"] = args.time_slice

    df = load(args.dataset, **kwargs)

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
    sub_list.add_argument("--category", "-c", help="Filter by category (A, C, D, E)")
    sub_list.add_argument("--format", "-f", help="Filter by format (CSV, GRIB2, NetCDF)")
    _add_common_args(sub_list)
    sub_list.set_defaults(func=cmd_list)

    # --- foehn download ---
    sub_dl = subparsers.add_parser("download", help="Download datasets")
    _add_dataset_arg(sub_dl)
    _add_common_args(sub_dl)
    sub_dl.add_argument("--historical", action="store_true", help="Include historical time slice")
    sub_dl.add_argument("--now", action="store_true", help="Include realtime 'now' time slice")
    sub_dl.add_argument("--all", action="store_true", help="Include all time slices (historical + recent + now)")
    sub_dl.add_argument("--full-refresh", action="store_true", help="Ignore incremental tracking, re-download all")
    sub_dl.add_argument("--grids", action="store_true", help="Include grid/binary datasets (GRIB2, NetCDF)")
    sub_dl.add_argument("--no-parquet", action="store_true", help="Skip CSV → Parquet conversion")
    sub_dl.set_defaults(func=cmd_download)

    # --- foehn to-parquet ---
    sub_conv = subparsers.add_parser("to-parquet", help="Convert downloaded CSVs to Parquet")
    _add_dataset_arg(sub_conv)
    _add_common_args(sub_conv)
    sub_conv.set_defaults(func=cmd_to_parquet)

    # --- foehn load ---
    sub_load = subparsers.add_parser("load", help="Load a dataset and print a preview")
    sub_load.add_argument("dataset", help="Dataset name (e.g. 'smn')")
    sub_load.add_argument("--station", nargs="+", help="Filter by station(s)")
    sub_load.add_argument("--frequency", nargs="+", help="Filter by frequency (t, h, d, m, y)")
    sub_load.add_argument("--time-slice", nargs="+", help="Time slices (default: recent)")
    sub_load.add_argument("-n", type=int, default=None, help="Number of rows to show (default: 20)")
    sub_load.set_defaults(func=cmd_load)

    args = parser.parse_args()
    args.func(args)
