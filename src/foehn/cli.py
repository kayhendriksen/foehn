"""CLI entry point for the foehn package."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import polars as pl

from foehn import registry
from foehn.api import METADATA_TABLES, list_datasets, metadata
from foehn.collections import CATEGORIES, CATEGORY_LABELS, COLLECTIONS, DEFAULT_TIME_SLICE
from foehn.fetch import DEFAULT_WORKERS, default_fetcher
from foehn.state import load_last_run, run_watermark, save_last_run
from foehn.workspace import Workspace


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
    # Default: every dataset that is not a grid, unless grids are opted in.
    if allow_grids:
        return list(COLLECTIONS)
    return registry.non_grid_datasets()


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

    current_cat = None
    for row in rows:
        cat = row["category"]
        if cat != current_cat:
            if current_cat is not None:
                print()
            label = CATEGORY_LABELS.get(cat, cat)
            print(f"── {cat}: {label} ──")
            print(f"  {'Dataset':<32} {'Format':<8} {'Frequency':<16} Description")
            current_cat = cat

        frequencies = ", ".join(row["frequencies"]) if row["frequencies"] else "—"
        print(f"  {row['dataset']:<32} {row['format']:<8} {frequencies:<16} {row['description']}")


def cmd_download(args: argparse.Namespace) -> None:
    workspace = Workspace.resolve(args.data_dir)
    workspace.bronze().mkdir(parents=True, exist_ok=True)

    full_refresh = args.full_refresh or os.environ.get("FOEHN_FULL_REFRESH", "").lower() in ("1", "true", "yes")

    time_slices = [DEFAULT_TIME_SLICE]
    if args.all or args.now:
        time_slices.append("now")
    if args.all or args.historical:
        time_slices.insert(0, "historical")

    since = None
    if not full_refresh:
        since = load_last_run(workspace)

    if since:
        print(f"Incremental update (last run: {since})", flush=True)
    else:
        print("Full download", flush=True)
    print(f"Time slices: {time_slices}", flush=True)

    datasets = _resolve_datasets(args.datasets, allow_grids=args.grids)
    fetcher = default_fetcher()
    watermark = run_watermark()

    workers = args.workers
    failures = 0
    download_failures = 0
    for ds in datasets:
        download_failures += registry.download(
            ds,
            workspace,
            time_slice=time_slices,
            since=since,
            workers=workers,
            force=full_refresh,
            fetcher=fetcher,
        ).failed
        if not args.no_parquet:
            failures += registry.convert(ds, workspace)

    if failures == download_failures == 0:
        save_last_run(workspace, watermark)
    else:
        # Don't advance the incremental cursor if anything failed — otherwise the
        # next run filters out the still-broken items as "already seen". A failed
        # asset within an unchanged item would never be retried.
        reasons = []
        if download_failures:
            reasons.append(f"{download_failures} download failure(s)")
        if failures:
            reasons.append(f"{failures} conversion failure(s)")
        print(
            f"\n{', '.join(reasons)} — not advancing _last_run.json. Re-run after fixing.",
            file=sys.stderr,
            flush=True,
        )

    print(f"\nBronze data saved to:   {workspace.bronze()}")
    if not args.no_parquet:
        print(f"Parquet files saved to: {workspace.parquet()}")

    if failures or download_failures:
        sys.exit(1)


def cmd_to_parquet(args: argparse.Namespace) -> None:
    workspace = Workspace.resolve(args.data_dir)

    failures = 0
    for ds in _resolve_datasets(args.datasets):
        failures += registry.convert(ds, workspace)

    print(f"Parquet files saved to: {workspace.parquet()}")

    if failures:
        sys.exit(1)


def cmd_metadata(args: argparse.Namespace) -> None:
    # Which tables exist is api.METADATA_TABLES, and so is what argparse accepts
    # below — the ladder this replaced also carried an else branch that argparse's
    # own ``choices`` made unreachable.
    df = metadata(args.dataset, args.table)

    if df.is_empty():
        print(f"No {args.table} metadata found for {args.dataset!r}.")
        return

    # Table output similar to foehn list
    headers = df.columns
    # Render to strings once, in Polars, and measure and print from that same
    # projection. Measuring in Polars while printing Python's str() looked
    # equivalent but is not: a Datetime casts to 26 characters here
    # ("...00:00:00.000000") against str()'s 19, so the column came out padded
    # wider than anything in it. Doing both from one projection cannot drift —
    # and it keeps the measurement off the pure-Python path, which took seconds
    # on a large inventory.
    rendered = df.select(pl.col(c).cast(pl.Utf8).fill_null("—").alias(c) for c in headers)
    max_lens = rendered.select(pl.col(c).str.len_chars().max().alias(c) for c in headers).row(0)
    widths = [min(max(len(col), n or 0), 40) for col, n in zip(headers, max_lens, strict=True)]

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*(("─" * w) for w in widths)))
    for row in rendered.iter_rows():
        # Truncate long values
        values = [v[:w] if len(v) > w else v for v, w in zip(row, widths, strict=True)]
        print(fmt.format(*values))

    print(f"\n[{df.shape[0]} rows]")


def cmd_open(args: argparse.Namespace) -> None:
    from foehn.api import open_dataset

    ds = open_dataset(
        args.dataset,
        variables=args.variables,
        match=args.match,
        data_dir=Workspace.resolve(args.data_dir).root,
    )
    print(ds)


def cmd_to_zarr(args: argparse.Namespace) -> None:
    from foehn.api import to_zarr

    store = to_zarr(
        args.dataset,
        variables=args.variables,
        match=args.match,
        data_dir=Workspace.resolve(args.data_dir).root,
        store=args.out,
        stack=args.stack is not None,
    )
    print(f"Zarr store written to: {store}")


def cmd_mcp(args: argparse.Namespace) -> None:
    from foehn.mcp_server import run

    run(transport=args.transport)


def cmd_load(args: argparse.Namespace) -> None:
    from foehn.api import load

    # argparse leaves every unset option as None, which is what load() already
    # treats as "no filter" — so the options go straight across. ``workers`` is
    # the one exception: its default is a count, not None.
    df = load(
        args.dataset,
        station=args.station,
        frequency=args.frequency,
        time_slice=args.time_slice,
        year=args.year,
        month=args.month,
        date_from=args.date_from,
        date_to=args.date_to,
        columns=args.columns,
        drop_null=args.drop_null,
        sort=args.sort,
        limit=args.limit,
        workers=args.workers if args.workers is not None else DEFAULT_WORKERS,
    )

    n = args.n or 20
    print(df.head(n))
    print(f"\n[{df.shape[0]} rows x {df.shape[1]} columns]")


class _FoehnCliHandler(logging.StreamHandler):
    """Marker subclass so repeat invocations can find and reuse *our* handler.

    A plain ``StreamHandler`` would be indistinguishable from one an embedding
    application attached to the ``foehn`` logger itself.
    """


def _configure_logging() -> None:
    """Route the foehn library's logger to stdout for CLI use.

    The library logs through ``logging.getLogger("foehn.*")`` and ships no
    handler, so it is silent when imported. The CLI attaches a single stdout
    handler with a bare message format, preserving the previous ``print()`` look.

    Idempotent across repeated/embedded invocations: rather than bailing out
    when *any* handler exists, we look up our own tagged handler and refresh its
    stream/formatter (the previous one may point at a stale ``sys.stdout``).
    Level and propagation are always re-asserted.
    """
    foehn_logger = logging.getLogger("foehn")
    handler = next(
        (h for h in foehn_logger.handlers if isinstance(h, _FoehnCliHandler)),
        None,
    )
    if handler is None:
        handler = _FoehnCliHandler(sys.stdout)
        foehn_logger.addHandler(handler)
    else:
        # Repoint at the current stdout without setStream(), which would first
        # flush the old stream — fatal if a prior invocation's stream is closed.
        handler.stream = sys.stdout
    handler.setFormatter(logging.Formatter("%(message)s"))
    foehn_logger.setLevel(logging.INFO)
    foehn_logger.propagate = False


def main():
    _configure_logging()

    parser = argparse.ArgumentParser(
        prog="foehn",
        description="Download MeteoSwiss Open Data and convert to Parquet.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- foehn list ---
    sub_list = subparsers.add_parser("list", help="List available datasets")
    sub_list.add_argument("--category", "-c", help=f"Filter by category ({', '.join(sorted(CATEGORIES))})")
    sub_list.add_argument("--format", "-f", help="Filter by format (CSV, TXT, NetCDF, GRIB2, HDF5)")
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
    sub_dl.add_argument("--grids", action="store_true", help="Include grid datasets (NetCDF, GRIB2, HDF5/radar)")
    sub_dl.add_argument("--no-parquet", action="store_true", help="Skip CSV → Parquet conversion")
    sub_dl.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent downloads per dataset (default: 8)",
    )
    sub_dl.set_defaults(func=cmd_download)

    # --- foehn to-parquet ---
    sub_conv = subparsers.add_parser("to-parquet", help="Convert downloaded CSVs to Parquet")
    _add_dataset_arg(sub_conv)
    _add_common_args(sub_conv)
    sub_conv.set_defaults(func=cmd_to_parquet)

    # --- foehn metadata ---
    sub_meta = subparsers.add_parser("metadata", help="Show dataset metadata (parameters, stations, inventory)")
    sub_meta.add_argument("table", choices=sorted(METADATA_TABLES), help="Which metadata table to show")
    sub_meta.add_argument("dataset", help="Dataset name (e.g. 'smn')")
    sub_meta.set_defaults(func=cmd_metadata)

    # --- foehn load ---
    sub_load = subparsers.add_parser("load", help="Load a dataset and print a preview")
    sub_load.add_argument("dataset", help="Dataset name (e.g. 'smn')")
    sub_load.add_argument("--station", nargs="+", help="Filter by station(s)")
    sub_load.add_argument("--frequency", nargs="+", help="Filter by frequency (t, h, d, m, y)")
    sub_load.add_argument("--time-slice", nargs="+", help="Time slices (default: recent)")
    sub_load.add_argument("--year", nargs="+", type=int, help="Filter by year(s) (e.g. 2025 2026)")
    sub_load.add_argument("--month", nargs="+", type=int, help="Filter by month(s) (1-12)")
    sub_load.add_argument("--date-from", help="Start date inclusive (YYYY-MM-DD)")
    sub_load.add_argument("--date-to", help="End date inclusive (YYYY-MM-DD)")
    sub_load.add_argument("--columns", nargs="+", help="Only return these columns")
    sub_load.add_argument("--drop-null", help="Drop rows where this column is null")
    sub_load.add_argument("--sort", choices=["asc", "desc"], help="Sort by timestamp")
    sub_load.add_argument("--limit", type=int, default=None, help="Cap the loaded DataFrame to N rows (after sort)")
    sub_load.add_argument("--workers", type=int, default=None, help="Concurrent CSV downloads (default: 8)")
    sub_load.add_argument("-n", type=int, default=None, help="Number of rows to print in the preview (default: 20)")
    sub_load.set_defaults(func=cmd_load)

    # --- foehn open ---
    sub_open = subparsers.add_parser(
        "open", help="Open a gridded dataset (NetCDF, or GRIB2/radar with --match) and print its xarray summary"
    )
    sub_open.add_argument("dataset", help="Dataset name (e.g. 'surface_derived_grid', 'forecast_icon_ch1')")
    sub_open.add_argument("--variables", nargs="+", help="Restrict to these data variable(s)")
    sub_open.add_argument(
        "--match",
        help="Keep only source files whose name contains this substring (required for GRIB2/radar, selects 1 file)",
    )
    _add_common_args(sub_open)
    sub_open.set_defaults(func=cmd_open)

    # --- foehn to-zarr ---
    sub_zarr = subparsers.add_parser(
        "to-zarr", help="Write a gridded dataset (NetCDF, or GRIB2/radar with --match) to a Zarr store"
    )
    sub_zarr.add_argument("dataset", help="Dataset name (e.g. 'surface_derived_grid', 'forecast_icon_ch1')")
    sub_zarr.add_argument("--variables", nargs="+", help="Restrict to these data variable(s)")
    sub_zarr.add_argument(
        "--match",
        help="Keep only source files whose name contains this substring (required for GRIB2/radar, selects 1 file)",
    )
    sub_zarr.add_argument("--out", help="Explicit output path for the .zarr store (overrides the default location)")
    sub_zarr.add_argument(
        "--stack",
        nargs="?",
        const="auto",
        choices=["auto", "time"],
        default=None,
        # The dataset's kind now decides how to cube, so the value carries
        # nothing --stack on its own does not. It is still accepted because
        # v0.4.0 documented `--stack time` and `--stack auto`, and turning the
        # option into a bare flag made every such command line fail outright.
        help="Combine the matched files into one cube. Takes no value; 'auto' and 'time' are accepted "
        "for compatibility with v0.4 and choose nothing the dataset's kind does not already decide",
    )
    _add_common_args(sub_zarr)
    sub_zarr.set_defaults(func=cmd_to_zarr)

    # --- foehn mcp ---
    sub_mcp = subparsers.add_parser("mcp", help="Start the MCP server for LLM integration")
    sub_mcp.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    sub_mcp.set_defaults(func=cmd_mcp)

    args = parser.parse_args()
    args.func(args)
