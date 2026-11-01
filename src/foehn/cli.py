"""The ``foehn`` command line.

Every subcommand is a thin front end over :mod:`foehn.api`: it parses flags,
calls the function that does the work, and renders the result. Two conventions
hold throughout, the ones uv and ruff keep:

* **Data on stdout, diagnostics on stderr.** Tables, CSV/JSON output and the
  version report are what a command produces, so they go to stdout and can be
  piped or redirected. Progress, warnings, errors and the library's log go to
  stderr — so ``foehn load ... --output-format csv > out.csv`` never captures a
  log line, and ``foehn mcp`` never writes one into the JSON-RPC stream that its
  stdio transport carries on stdout.
* **Exit status.** 0 on success; 1 when the command failed (bad input, an
  unreachable API, a partial download); 2 for a usage error argparse rejects;
  130 when interrupted; 141 when the reader of stdout went away (``| head``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import polars as pl

from foehn import __version__, registry
from foehn.api import METADATA_TABLES, list_datasets, metadata
from foehn.collections import CATEGORIES, CATEGORY_LABELS, COLLECTIONS, DEFAULT_TIME_SLICE, suggest_dataset
from foehn.fetch import DEFAULT_WORKERS, FetchError, default_fetcher
from foehn.state import advance_cursor, load_cursor, run_watermark
from foehn.versions import format_versions, versions
from foehn.workspace import Workspace

DOCS_URL = "https://github.com/kayhendriksen/foehn/tree/main/docs"

FULL_REFRESH_ENV = "FOEHN_FULL_REFRESH"
"""Environment variable equivalent of ``foehn download --full-refresh``."""

# Rows ``foehn load`` previews when -n is not given.
_DEFAULT_PREVIEW_ROWS = 20

# What ``foehn load --output PATH`` writes, by the file's suffix.
_FILE_FORMATS = {
    ".parquet": "parquet",
    ".csv": "csv",
    ".json": "json",
    ".ndjson": "ndjson",
    ".jsonl": "ndjson",
}

# Exit statuses. 2 is argparse's own, for a usage error.
EXIT_FAILURE = 1
EXIT_INTERRUPTED = 130
EXIT_BROKEN_PIPE = 141


class CliError(Exception):
    """A failure the CLI reports as one ``error:`` line, optionally with a ``hint:``."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


# --- Output -------------------------------------------------------------------


def _use_color(when: str, stream: TextIO) -> bool:
    """Whether to colour *stream*: ``--color``, then ``NO_COLOR``/``FORCE_COLOR``, then a TTY check."""
    if when != "auto":
        return when == "always"
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


@dataclass(frozen=True)
class Console:
    """Where a command's diagnostics go, and how loudly.

    Everything here writes to stderr: stdout is reserved for the command's
    output. ``quiet`` silences progress; a second ``-q`` silences warnings too.
    """

    color: str = "auto"
    quiet: int = 0

    def paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if _use_color(self.color, sys.stderr) else text

    def status(self, message: str) -> None:
        if not self.quiet:
            print(message, file=sys.stderr, flush=True)

    def warning(self, message: str) -> None:
        if self.quiet < 2:
            print(f"{self.paint('warning', '1;33')}: {message}", file=sys.stderr, flush=True)

    def error(self, message: str, *, hint: str | None = None) -> None:
        print(f"foehn: {self.paint('error', '1;31')}: {message}", file=sys.stderr, flush=True)
        if hint:
            print(f"  {self.paint('hint', '1;36')}: {hint}", file=sys.stderr, flush=True)


def _console(args: argparse.Namespace) -> Console:
    return Console(color=args.color, quiet=args.quiet)


def _write_json(value: object) -> None:
    sys.stdout.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _write_frame(df: pl.DataFrame, output_format: str) -> None:
    """Write a whole DataFrame to stdout in a machine-readable format."""
    if output_format == "csv":
        sys.stdout.write(df.write_csv())
    elif output_format == "json":
        sys.stdout.write(df.write_json() + "\n")
    elif output_format == "ndjson":
        sys.stdout.write(df.write_ndjson())
    else:  # pragma: no cover - argparse's choices make this unreachable
        raise CliError(f"unsupported output format {output_format!r}")
    sys.stdout.flush()


def _save_frame(df: pl.DataFrame, path: Path) -> None:
    """Write a DataFrame to *path* in the format its suffix names."""
    file_format = _FILE_FORMATS.get(path.suffix.lower())
    if file_format is None:
        raise CliError(
            f"cannot tell which format to write {str(path)!r} in",
            hint=f"end the path in one of {', '.join(_FILE_FORMATS)}",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "parquet":
        df.write_parquet(path, compression="zstd")
    elif file_format == "csv":
        df.write_csv(path)
    elif file_format == "json":
        df.write_json(path)
    else:
        df.write_ndjson(path)


def _print_table(df: pl.DataFrame) -> None:
    """Render a metadata table as aligned plain-text columns."""
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


# --- Arguments ----------------------------------------------------------------


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be zero or more, got {value}")
    return number


def _global_options(*, root: bool) -> argparse.ArgumentParser:
    """Flags accepted both before and after the subcommand, as uv's are.

    The copies attached to each subcommand default to ``SUPPRESS``: argparse
    parses a subcommand into a fresh namespace and copies every attribute it set
    over the parent's, so a real default there would reset a ``-q`` given before
    the subcommand.
    """
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("global options")
    group.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0 if root else argparse.SUPPRESS,
        help="Print no progress; repeat (-qq) to silence warnings too",
    )
    group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0 if root else argparse.SUPPRESS,
        help="Print debug detail from the library",
    )
    group.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto" if root else argparse.SUPPRESS,
        metavar="WHEN",
        help="Colour diagnostics: auto (default), always, never. Honours NO_COLOR and FORCE_COLOR",
    )
    return parser


def _add_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Root data directory (default: $FOEHN_DATA_DIR or ./data/meteoswiss)",
    )


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "datasets",
        nargs="*",
        metavar="DATASET",
        help="Dataset(s) to operate on (default: all). Use 'foehn list' to see options.",
    )


def _add_output_format(parser: argparse.ArgumentParser, *formats: str) -> None:
    parser.add_argument(
        "--output-format",
        choices=["table", *formats],
        default="table",
        help="How to print the result (default: table)",
    )


def _add_grid_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", help="Dataset name (e.g. 'surface_derived_grid', 'forecast_icon_ch1')")
    parser.add_argument("--variables", nargs="+", help="Restrict to these data variable(s)")
    parser.add_argument(
        "--match",
        help="Keep only source files whose name contains this substring (required for GRIB2/radar, selects 1 file)",
    )


def _unknown_dataset(name: str) -> CliError:
    suggestion = suggest_dataset(name)
    hint = f"did you mean '{suggestion}'?" if suggestion else "run 'foehn list' to see the available datasets"
    return CliError(f"unknown dataset {name!r}", hint=hint)


def _resolve_datasets(datasets: list[str], *, allow_grids: bool = False) -> list[str]:
    """Resolve dataset arguments to a list of collection keys."""
    if datasets:
        for d in datasets:
            if d not in COLLECTIONS:
                raise _unknown_dataset(d)
        return datasets
    # Default: every dataset that is not a grid, unless grids are opted in.
    if allow_grids:
        return list(COLLECTIONS)
    return registry.non_grid_datasets()


# --- Commands -----------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> None:
    rows = list_datasets()

    if args.category:
        cat = args.category.upper()
        rows = [r for r in rows if r["category"] == cat]
    if args.format:
        fmt = args.format.upper()
        rows = [r for r in rows if r["format"].upper() == fmt]

    if args.output_format == "json":
        _write_json(rows)
        return
    if args.output_format == "csv":
        frame = pl.DataFrame(rows, schema=list(rows[0]) if rows else ["dataset"])
        lists = [name for name, dtype in frame.schema.items() if isinstance(dtype, pl.List)]
        _write_frame(frame.with_columns(pl.col(lists).list.join(",")), "csv")
        return

    if not rows:
        _console(args).status("No datasets match the given filters.")
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
    console = _console(args)
    # Resolved first, so an unknown name is refused before anything is created.
    datasets = _resolve_datasets(args.datasets, allow_grids=args.grids)
    workspace = Workspace.resolve(args.data_dir)
    workspace.bronze().mkdir(parents=True, exist_ok=True)

    full_refresh = args.full_refresh or os.environ.get(FULL_REFRESH_ENV, "").lower() in ("1", "true", "yes")

    time_slices = [DEFAULT_TIME_SLICE]
    if args.all or args.now:
        time_slices.append("now")
    if args.all or args.historical:
        time_slices.insert(0, "historical")

    console.status("Full download" if full_refresh else "Incremental update (per-dataset cursors)")
    console.status(f"Time slices: {time_slices}")

    fetcher = default_fetcher()
    watermark = run_watermark()

    failures = 0
    download_failures = 0
    held_back: list[str] = []
    for ds in datasets:
        # Each Dataset and Time slice keeps its own cursor. One shared instant let
        # a run of one Dataset filter the next Dataset's listing, so Items never
        # downloaded were dropped as "unchanged" — see foehn.state.
        since = None if full_refresh else load_cursor(workspace, ds, time_slices)
        console.status(f"  {ds}: {f'changes since {since}' if since else 'full listing'}")
        ds_download_failures = registry.download(
            ds,
            workspace,
            time_slice=time_slices,
            since=since,
            workers=args.workers,
            force=full_refresh,
            fetcher=fetcher,
        ).failed
        ds_failures = 0 if args.no_parquet else registry.convert(ds, workspace)
        download_failures += ds_download_failures
        failures += ds_failures
        if ds_download_failures == ds_failures == 0:
            advance_cursor(workspace, ds, time_slices, watermark)
        else:
            # Don't advance this Dataset's cursor if anything failed — otherwise
            # the next run filters out the still-broken items as "already seen". A
            # failed asset within an unchanged item would never be retried.
            held_back.append(ds)

    if held_back:
        reasons = []
        if download_failures:
            reasons.append(f"{download_failures} download failure(s)")
        if failures:
            reasons.append(f"{failures} conversion failure(s)")
        console.warning(
            f"{', '.join(reasons)} — not advancing the _last_run.json cursor for "
            f"{', '.join(held_back)}. Re-run after fixing."
        )

    console.status(f"\nBronze data saved to:   {workspace.bronze()}")
    if not args.no_parquet:
        console.status(f"Parquet files saved to: {workspace.parquet()}")

    if failures or download_failures:
        sys.exit(EXIT_FAILURE)


def cmd_to_parquet(args: argparse.Namespace) -> None:
    workspace = Workspace.resolve(args.data_dir)

    failures = 0
    for ds in _resolve_datasets(args.datasets):
        failures += registry.convert(ds, workspace)

    _console(args).status(f"Parquet files saved to: {workspace.parquet()}")

    if failures:
        sys.exit(EXIT_FAILURE)


def cmd_metadata(args: argparse.Namespace) -> None:
    # Which tables exist is api.METADATA_TABLES, and so is what argparse accepts
    # below — the ladder this replaced also carried an else branch that argparse's
    # own ``choices`` made unreachable.
    df = metadata(args.dataset, args.table)

    if args.output_format != "table":
        _write_frame(df, args.output_format)
    elif df.is_empty():
        _console(args).status(f"No {args.table} metadata found for {args.dataset!r}.")
    else:
        _print_table(df)


def cmd_load(args: argparse.Namespace) -> None:
    from foehn.api import load

    # argparse leaves every unset option as None, which is what load() already
    # treats as "no filter" — so the options go straight across.
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
        workers=args.workers,
    )

    if args.output is not None:
        _save_frame(df, args.output)
        _console(args).status(f"Wrote {df.height} rows x {df.width} columns to {args.output}")
    elif args.output_format != "table":
        _write_frame(df, args.output_format)
    else:
        n = _DEFAULT_PREVIEW_ROWS if args.n is None else args.n
        # Polars shows at most ten rows of any frame unless told otherwise, so
        # head(n) alone printed five, an ellipsis and five more for every n > 10.
        with pl.Config(tbl_rows=n):
            print(df.head(n))
        print(f"\n[{df.shape[0]} rows x {df.shape[1]} columns]")


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
    try:
        from foehn.mcp_server import run
    except ModuleNotFoundError as exc:
        # By package, not module: an mcp too old to have ``mcp.server.mcpserver``
        # is fixed by the same install.
        if (exc.name or "").split(".")[0] not in {"mcp", "pydantic"}:
            raise
        raise CliError(
            "the MCP server needs the optional 'mcp' dependencies",
            hint='install them with: pip install "foehn[mcp]"',
        ) from exc

    run(transport=args.transport)


def cmd_version(args: argparse.Namespace) -> None:
    if args.output_format == "json":
        _write_json(versions())
    else:
        print(format_versions())


# --- Logging ------------------------------------------------------------------


class _FoehnCliHandler(logging.StreamHandler):
    """Marker subclass so repeat invocations can find and reuse *our* handler.

    A plain ``StreamHandler`` would be indistinguishable from one an embedding
    application attached to the ``foehn`` logger itself.
    """


class _LevelFormatter(logging.Formatter):
    """A bare message for INFO and DEBUG; ``warning:``/``error:`` before anything louder."""

    def __init__(self, console: Console) -> None:
        super().__init__("%(message)s")
        self._console = console

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno >= logging.ERROR:
            return f"{self._console.paint('error', '1;31')}: {message}"
        if record.levelno >= logging.WARNING:
            return f"{self._console.paint('warning', '1;33')}: {message}"
        return message


def _log_level(*, quiet: int, verbose: int) -> int:
    if verbose:
        return logging.DEBUG
    return {0: logging.INFO, 1: logging.WARNING}.get(quiet, logging.ERROR)


def _configure_logging(console: Console | None = None, *, verbose: int = 0) -> None:
    """Route the foehn library's logger to stderr for CLI use.

    The library logs through ``logging.getLogger("foehn.*")`` and ships no
    handler, so it is silent when imported. The CLI attaches a single stderr
    handler with a bare message format. Never stdout: that is the command's
    output, and under ``foehn mcp`` it is the protocol channel itself.

    Idempotent across repeated/embedded invocations: rather than bailing out
    when *any* handler exists, we look up our own tagged handler and refresh its
    stream/formatter (the previous one may point at a stale ``sys.stderr``).
    Level and propagation are always re-asserted.
    """
    console = console or Console()
    foehn_logger = logging.getLogger("foehn")
    handler = next(
        (h for h in foehn_logger.handlers if isinstance(h, _FoehnCliHandler)),
        None,
    )
    if handler is None:
        handler = _FoehnCliHandler(sys.stderr)
        foehn_logger.addHandler(handler)
    else:
        # Repoint at the current stderr without setStream(), which would first
        # flush the old stream — fatal if a prior invocation's stream is closed.
        handler.stream = sys.stderr
    handler.setFormatter(_LevelFormatter(console))
    foehn_logger.setLevel(_log_level(quiet=console.quiet, verbose=verbose))
    foehn_logger.propagate = False


# --- Parser -------------------------------------------------------------------

_DESCRIPTION = """\
MeteoSwiss Open Government Data from the command line.

Download every MeteoSwiss collection and convert station data to Parquet, load
it straight into a table, inspect metadata, open gridded NetCDF/GRIB2/radar data
with xarray, or serve it all to an LLM over MCP."""

_EPILOG = f"""\
Use 'foehn <command> --help' for more information on a command.
Documentation: {DOCS_URL}"""


def _subcommand(
    subparsers: argparse._SubParsersAction,
    name: str,
    *,
    help: str,  # noqa: A002 - argparse's own keyword
    examples: str | None = None,
) -> argparse.ArgumentParser:
    return subparsers.add_parser(
        name,
        help=help,
        description=help[0].upper() + help[1:] + ".",
        epilog=f"examples:\n{examples}" if examples else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_global_options(root=False)],
    )


def build_parser() -> argparse.ArgumentParser:
    """The ``foehn`` argument parser. Public so the documentation can be checked against it."""
    parser = argparse.ArgumentParser(
        prog="foehn",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_global_options(root=True)],
    )
    if sys.version_info >= (3, 14):
        # Suggest the closest subcommand or choice for a typo ("foehn lsit").
        parser.suggest_on_error = True
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND", title="commands")

    # --- foehn list ---
    sub_list = _subcommand(
        subparsers,
        "list",
        help="list the available datasets",
        examples="  foehn list\n  foehn list --category A\n  foehn list --format GRIB2 --output-format json",
    )
    sub_list.add_argument("--category", "-c", help=f"Filter by category ({', '.join(sorted(CATEGORIES))})")
    sub_list.add_argument("--format", "-f", help="Filter by format (CSV, TXT, NetCDF, GRIB2, HDF5)")
    _add_output_format(sub_list, "json", "csv")
    _add_data_dir(sub_list)
    sub_list.set_defaults(func=cmd_list)

    # --- foehn load ---
    sub_load = _subcommand(
        subparsers,
        "load",
        help="load a dataset from the API and print it",
        examples=(
            "  foehn load smn --station BER --frequency d\n"
            "  foehn load smn --station BER ZUR --frequency h -n 50\n"
            "  foehn load smn --station BER --frequency d --output-format csv > ber.csv\n"
            "  foehn load smn --station BER --frequency d --year 2025 -o ber-2025.parquet"
        ),
    )
    sub_load.add_argument("dataset", help="Dataset name (e.g. 'smn')")
    sub_load.add_argument("--station", nargs="+", help="Filter by station(s)")
    sub_load.add_argument("--frequency", nargs="+", help="Filter by frequency (t, h, d, m, y)")
    sub_load.add_argument("--time-slice", nargs="+", help=f"Time slices (default: {DEFAULT_TIME_SLICE})")
    sub_load.add_argument("--year", nargs="+", type=int, help="Filter by year(s) (e.g. 2025 2026)")
    sub_load.add_argument("--month", nargs="+", type=int, help="Filter by month(s) (1-12)")
    sub_load.add_argument("--date-from", help="Start date inclusive (YYYY-MM-DD)")
    sub_load.add_argument("--date-to", help="End date inclusive (YYYY-MM-DD)")
    sub_load.add_argument("--columns", nargs="+", help="Only return these columns")
    sub_load.add_argument("--drop-null", help="Drop rows where this column is null")
    sub_load.add_argument("--sort", choices=["asc", "desc"], help="Sort by timestamp")
    sub_load.add_argument(
        "--limit", type=_non_negative_int, default=None, help="Cap the loaded DataFrame to N rows (after sort)"
    )
    sub_load.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent CSV downloads (default: {DEFAULT_WORKERS})",
    )
    sub_load.add_argument(
        "-n",
        type=_non_negative_int,
        default=None,
        help=f"Number of rows to print in the table preview (default: {_DEFAULT_PREVIEW_ROWS})",
    )
    _add_output_format(sub_load, "csv", "json", "ndjson")
    sub_load.add_argument(
        "--output",
        "-o",
        type=Path,
        metavar="PATH",
        help=f"Write every row to PATH instead of printing; the suffix picks the format ({', '.join(_FILE_FORMATS)})",
    )
    sub_load.set_defaults(func=cmd_load)

    # --- foehn metadata ---
    sub_meta = _subcommand(
        subparsers,
        "metadata",
        help="show a dataset's metadata (parameters, stations, points, inventory)",
        examples=(
            "  foehn metadata parameters smn\n"
            "  foehn metadata stations smn --output-format csv > stations.csv\n"
            "  foehn metadata points forecast_local"
        ),
    )
    sub_meta.add_argument("table", choices=sorted(METADATA_TABLES), help="Which metadata table to show")
    sub_meta.add_argument("dataset", help="Dataset name (e.g. 'smn')")
    _add_output_format(sub_meta, "csv", "json", "ndjson")
    sub_meta.set_defaults(func=cmd_metadata)

    # --- foehn download ---
    sub_dl = _subcommand(
        subparsers,
        "download",
        help="download datasets and convert them to Parquet",
        examples=(
            "  foehn download                 # every non-grid dataset, recent time slice\n"
            "  foehn download smn pollen      # specific datasets\n"
            "  foehn download smn --all       # historical + recent + now"
        ),
    )
    _add_dataset_args(sub_dl)
    _add_data_dir(sub_dl)
    sub_dl.add_argument("--historical", action="store_true", help="Include historical time slice")
    sub_dl.add_argument("--now", action="store_true", help="Include realtime 'now' time slice")
    sub_dl.add_argument("--all", action="store_true", help="Include all time slices (historical + recent + now)")
    sub_dl.add_argument("--full-refresh", action="store_true", help="Ignore incremental tracking, re-download all")
    sub_dl.add_argument("--grids", action="store_true", help="Include grid datasets (NetCDF, GRIB2, HDF5/radar)")
    sub_dl.add_argument("--no-parquet", action="store_true", help="Skip CSV → Parquet conversion")
    sub_dl.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent downloads per dataset (default: {DEFAULT_WORKERS})",
    )
    sub_dl.set_defaults(func=cmd_download)

    # --- foehn to-parquet ---
    sub_conv = _subcommand(
        subparsers,
        "to-parquet",
        help="convert downloaded CSVs to Parquet",
        examples="  foehn to-parquet\n  foehn to-parquet smn",
    )
    _add_dataset_args(sub_conv)
    _add_data_dir(sub_conv)
    sub_conv.set_defaults(func=cmd_to_parquet)

    # --- foehn open ---
    sub_open = _subcommand(
        subparsers,
        "open",
        help="open a gridded dataset (NetCDF, or GRIB2/radar with --match) and print its xarray summary",
        examples=(
            "  foehn open surface_derived_grid --match rhiresd\n"
            "  foehn open forecast_icon_ch1 --match YYYYMMDDHHMM-0-t_2m-ctrl"
        ),
    )
    _add_grid_args(sub_open)
    _add_data_dir(sub_open)
    sub_open.set_defaults(func=cmd_open)

    # --- foehn to-zarr ---
    sub_zarr = _subcommand(
        subparsers,
        "to-zarr",
        help="write a gridded dataset (NetCDF, or GRIB2/radar with --match) to a Zarr store",
        examples=(
            "  foehn to-zarr surface_derived_grid --match rhiresd\n"
            "  foehn to-zarr radar_precip --match cpcYYDDD --stack"
        ),
    )
    _add_grid_args(sub_zarr)
    sub_zarr.add_argument(
        "--out", metavar="PATH", help="Explicit output path for the .zarr store (overrides the default location)"
    )
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
    _add_data_dir(sub_zarr)
    sub_zarr.set_defaults(func=cmd_to_zarr)

    # --- foehn mcp ---
    sub_mcp = _subcommand(
        subparsers,
        "mcp",
        help="start the MCP server for LLM integration",
        examples="  foehn mcp\n  foehn mcp --transport streamable-http",
    )
    sub_mcp.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio). 'sse' is deprecated by MCP in favour of 'streamable-http'",
    )
    sub_mcp.set_defaults(func=cmd_mcp)

    # --- foehn version ---
    sub_version = _subcommand(
        subparsers,
        "version",
        help="show foehn's version, the platform and every dependency's version",
        examples="  foehn version\n  foehn version --output-format json",
    )
    _add_output_format(sub_version, "json")
    sub_version.set_defaults(func=cmd_version)

    return parser


def _silence_stdout() -> None:
    """Point stdout at /dev/null so the interpreter's exit-time flush cannot raise again."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError):  # pragma: no cover - stdout has no file descriptor (captured)
        pass


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = _console(args)
    _configure_logging(console, verbose=args.verbose)

    try:
        args.func(args)
    except CliError as exc:
        console.error(str(exc), hint=exc.hint)
        sys.exit(EXIT_FAILURE)
    except (ValueError, FetchError, ImportError) as exc:
        # The messages foehn raises for these name the fix already; a traceback
        # under them only buries it.
        console.error(str(exc))
        sys.exit(EXIT_FAILURE)
    except BrokenPipeError:
        # The reader of stdout went away (``foehn metadata inventory smn | head``).
        # Not an error worth a traceback: stop quietly, as a C tool killed by
        # SIGPIPE would, with the status that signal gives.
        _silence_stdout()
        sys.exit(EXIT_BROKEN_PIPE)
    except KeyboardInterrupt:
        sys.exit(EXIT_INTERRUPTED)
