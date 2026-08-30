"""What each :class:`~foehn.collections.DatasetKind` can do, and who does it.

One table replacing the routing ladders that used to sit in ``api``, ``cli`` and
the Databricks ingest script, each re-deriving from a different set of dataset
keys. Callers ask the registry instead of testing membership.

Layering: ``collections`` (dataset facts) → ``client``/``convert``/``readers``/
``grids`` (adapters) → this module → ``api``/``cli``/``mcp_server``. All four
pipeline stages route through the table: the load readers live in
``foehn.readers`` and the grid readers in ``foehn.grids``, both below this
module, so ``load`` and ``grid`` can sit in :class:`KindSpec` beside ``download``
and ``convert`` rather than as if-ladders in ``api`` and ``grids``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from foehn.client import (
    DownloadResult,
    already_current,
    download_indoor_zip,
    download_normals_zip,
    exists,
    stac_download,
)
from foehn.collections import COLLECTION_META, COLLECTIONS, KIND_OF, DatasetKind, kind
from foehn.convert import (
    convert_indoor_to_parquet,
    convert_normals_to_parquet,
    convert_preamble_to_parquet,
    convert_to_parquet,
)
from foehn.fetch import DEFAULT_WORKERS, Fetcher
from foehn.grids import (
    GridReader,
    cube_grib2,
    cube_radar,
    ensure_grid_files,
    open_grib2,
    open_netcdf,
    open_radar,
    require_grib2,
    require_netcdf,
    require_radar,
    select_variables,
)
from foehn.transfer import csv_to_disk

if TYPE_CHECKING:
    import polars as pl
    import xarray as xr

from foehn.readers import (
    Filters,
    Reader,
    read_archive,
    read_preamble,
    read_standard,
)


class DownloadAdapter(Protocol):
    """Every download adapter takes the same arguments and ignores what its kind
    does not use, so callers never have to know which ones apply. The alternative
    — a per-kind call shape — is the ladder we are removing.

    Spelled as a Protocol rather than ``Callable[..., DownloadResult]``: with
    ``...`` neither the adapters nor the call site were checked at all, so the
    "everyone takes everything" convention was documented but not enforced.
    """

    def __call__(
        self,
        dataset: str,
        bronze_dir: Path,
        *,
        time_slice: list[str],
        since: str | None,
        workers: int,
        force: bool,
        fetcher: Fetcher,
    ) -> DownloadResult: ...


ConvertAdapter = Callable[[str, Path, Path], int]


# The CSV kinds share one listing configuration; only whether the assets are
# time-sliced or narrowed to the newest forecast run differs.
_download_csv = stac_download(
    suffixes=(".csv",),
    title="Collection",
    label="CSV",
    write=csv_to_disk,
    etags=True,
    time_sliced=True,
    with_metadata=True,
)

_download_forecast_csv = stac_download(
    suffixes=(".csv",),
    title="Collection",
    label="CSV",
    write=csv_to_disk,
    etags=True,
    # Forecast filenames carry no time slice; the newest run bounds them instead.
    latest_run=True,
    with_metadata=True,
)

_download_netcdf = stac_download(
    suffixes=(".nc", ".tif", ".zip"),
    title="NetCDF collection",
    # These are static: an existing file is never restated upstream, so a plain
    # existence check is enough.
    skip=exists,
)

# The ephemeral collections only ever want the newest page, and MeteoSwiss
# overwrites their files in place (CombiPrecip reanalysis reuses a filename
# ~8 days later), so the skip rule compares timestamps rather than existence.
_download_grib2 = stac_download(
    suffixes=(".grib2", ".grib"),
    title="GRIB2 collection",
    label="binary file",
    skip=already_current,
    max_items=100,
)

_download_radar = stac_download(
    suffixes=(".h5", ".hdf5"),
    title="Radar collection",
    label="binary file",
    skip=already_current,
    max_items=100,
)


@dataclass(frozen=True)
class KindSpec:
    """Everything the routing needs to know about one kind.

    ``convert`` is None for the grid kinds: they have no Parquet path at all,
    which is a fact about the kind rather than a branch each caller re-derives.
    """

    download: DownloadAdapter
    convert: ConvertAdapter | None
    load: Reader | None
    """How this kind becomes a DataFrame. None for the grid kinds (use open_dataset)."""

    grid: GridReader | None
    """How this kind becomes an xarray Dataset. None for the tabular kinds (use load).

    Every row states all four pipeline stages, so ``tabular`` and ``is_grid``
    below are read off the row rather than set by hand beside it — a flag that
    can disagree with the adapter it describes is a flag that eventually will.
    """

    supports_granularity: bool
    """Whether ``frequency=`` applies — false where filenames carry no granularity."""

    supports_calendar_filters: bool
    """Whether year/month/date_from/date_to apply. False for nominal 30-year dates."""

    key_columns: tuple[str, ...] = ("station_abbr", "reference_timestamp")
    """Columns an explicit ``columns=`` selection always keeps."""

    sort_column: str = "reference_timestamp"
    """What ``sort=`` orders by. The nominal-date kind has no real timestamp."""

    @property
    def tabular(self) -> bool:
        """Loadable as a Polars DataFrame — i.e. this kind has a reader."""
        return self.load is not None

    @property
    def is_grid(self) -> bool:
        """Read as an xarray Dataset — i.e. this kind has a grid reader."""
        return self.grid is not None


KINDS: dict[DatasetKind, KindSpec] = {
    DatasetKind.STANDARD_CSV: KindSpec(
        download=_download_csv,
        convert=convert_to_parquet,
        load=read_standard,
        grid=None,
        supports_granularity=True,
        supports_calendar_filters=True,
    ),
    DatasetKind.PREAMBLE_CSV: KindSpec(
        download=_download_csv,
        convert=convert_preamble_to_parquet,
        load=read_preamble,
        grid=None,
        supports_granularity=False,
        # Dates are nominal (0001..0030 on a 365-day calendar), so the calendar
        # filters would silently match nothing.
        supports_calendar_filters=False,
        key_columns=("station_abbr", "variable", "gwl", "date"),
        # Nominal dates, so ``date`` is a lexically-ordered string, not a timestamp.
        sort_column="date",
    ),
    DatasetKind.ARCHIVE_CSV: KindSpec(
        download=download_indoor_zip,
        convert=convert_indoor_to_parquet,
        load=read_archive,
        grid=None,
        supports_granularity=False,
        supports_calendar_filters=True,
        key_columns=("station_abbr", "reference_timestamp", "period", "scenario", "variant"),
    ),
    DatasetKind.FORECAST_CSV: KindSpec(
        download=_download_forecast_csv,
        convert=convert_to_parquet,
        load=read_standard,
        grid=None,
        supports_granularity=False,
        supports_calendar_filters=True,
    ),
    DatasetKind.DIRECT_ZIP: KindSpec(
        download=download_normals_zip,
        convert=convert_normals_to_parquet,
        # Neither loadable nor gridded: it converts to Parquet but has no reader,
        # so it is the one dataset ``tabular_datasets()`` and ``grid_datasets()``
        # both leave out. See DatasetKind.DIRECT_ZIP.
        load=None,
        grid=None,
        supports_granularity=False,
        supports_calendar_filters=False,
    ),
    DatasetKind.NETCDF_GRID: KindSpec(
        download=_download_netcdf,
        convert=None,
        load=None,
        grid=GridReader(
            suffixes=(".nc",),
            require=require_netcdf,
            open=open_netcdf,
            # No cube builder: a multi-file ``match`` already combines on read,
            # so ``stack`` has nothing left to assemble.
            cube=None,
        ),
        supports_granularity=False,
        supports_calendar_filters=False,
    ),
    DatasetKind.GRIB2_GRID: KindSpec(
        download=_download_grib2,
        convert=None,
        load=None,
        grid=GridReader(
            suffixes=(".grib2", ".grib"),
            require=require_grib2,
            open=open_grib2,
            cube=cube_grib2,
            max_files=1,
            # The whole set is held in memory at once (no dask), so an over-broad
            # match is capped before anything is downloaded.
            cube_max_files=1000,
            match_example="202605231500-0-t_2m-ctrl",
            cube_match_example="-t_2m-ctrl",
            # A forecast item's STAC ``datetime`` is its model run, so a match
            # naming one narrows ~57,000 items to ~200 server-side.
            run_datetime=True,
        ),
        supports_granularity=False,
        supports_calendar_filters=False,
    ),
    DatasetKind.RADAR_GRID: KindSpec(
        download=_download_radar,
        convert=None,
        load=None,
        grid=GridReader(
            suffixes=(".h5",),
            require=require_radar,
            open=open_radar,
            cube=cube_radar,
            max_files=1,
            # Deliberately uncapped: the cube appends one timestep at a time, so
            # it wants every file the match spans.
            cube_max_files=None,
            match_example="cpc2613000000",
            cube_match_example="cpc26130",
        ),
        supports_granularity=False,
        supports_calendar_filters=False,
    ),
}


def spec(dataset: str) -> KindSpec:
    """Return the :class:`KindSpec` for *dataset*."""
    return KINDS[kind(dataset)]


def download(
    dataset: str,
    bronze_dir: Path,
    *,
    time_slice: list[str] | None = None,
    since: str | None = None,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
    fetcher: Fetcher,
) -> DownloadResult:
    """Download *dataset* by whichever path its kind uses."""
    return spec(dataset).download(
        dataset,
        bronze_dir,
        time_slice=time_slice or ["recent"],
        since=since,
        workers=workers,
        force=force,
        fetcher=fetcher,
    )


def unreadable_message(dataset: str) -> str:
    """Why *dataset* has no DataFrame, in terms of what the caller should do instead.

    Two reasons, not one: a grid dataset has ``open_dataset``, while the
    direct-ZIP kind has a Parquet path and no in-memory reader at all. Saying
    "binary/grid" for both was accurate until the second reason existed.
    """
    kind_spec = spec(dataset)
    if kind_spec.is_grid:
        fmt = COLLECTION_META[dataset]["format"]
        return f"Dataset {dataset!r} is a gridded ({fmt}) dataset. Use foehn.open_dataset() to get an xarray Dataset."
    return (
        f"Dataset {dataset!r} has no in-memory reader. Use foehn.download() and foehn.to_parquet() "
        "to materialise it, then read the Parquet."
    )


def load(dataset: str, filters: Filters, *, fetcher: Fetcher) -> pl.DataFrame:
    """Load *dataset* by whichever reader its kind uses.

    Callers guard on ``spec(dataset).tabular`` first — which kinds have no reader
    is a fact about the kind rather than a branch each caller re-derives.
    """
    reader = spec(dataset).load
    if reader is None:
        raise ValueError(unreadable_message(dataset))
    return reader(dataset, filters, fetcher=fetcher)


def _grid_reader(dataset: str) -> GridReader:
    """The dataset's :class:`~foehn.grids.GridReader`, or raise if it has none.

    The mirror of the guard in :func:`load`: a kind is read one way or the other,
    and which one is a fact about the kind rather than a branch each caller
    re-derives.
    """
    reader = spec(dataset).grid
    if reader is None:
        fmt = COLLECTION_META[dataset]["format"]
        raise ValueError(f"Dataset {dataset!r} is tabular ({fmt}). Use foehn.load() to get a Polars DataFrame instead.")
    return reader


def open_grid(
    dataset: str,
    *,
    match: str | None = None,
    variables: str | list[str] | None = None,
    bronze_dir: Path,
    fetcher: Fetcher,
) -> xr.Dataset:
    """Open *dataset* as an xarray Dataset, by whichever grid reader its kind uses.

    ``require`` runs before anything is fetched, so a missing optional dependency
    fails in milliseconds rather than after the download — ``climate_scenarios_grid``
    is ~900 MB.
    """
    reader = _grid_reader(dataset)
    if reader.max_files == 1 and match is None:
        fmt = COLLECTION_META[dataset]["format"]
        raise ValueError(
            f"Dataset {dataset!r} is a {fmt} collection of many single-field files; opening it "
            "unfiltered would download them all. Narrow to one file with match=, e.g. "
            f'foehn.open_dataset({dataset!r}, match="{reader.match_example}").'
        )
    reader.require()
    files = ensure_grid_files(
        dataset,
        bronze_dir,
        suffixes=reader.suffixes,
        match=match,
        max_files=reader.max_files,
        run_datetime=reader.run_datetime,
        fetcher=fetcher,
    )
    return select_variables(reader.open(files, dataset=dataset, bronze_dir=bronze_dir, fetcher=fetcher), variables)


def write_cube(
    dataset: str,
    store: Path,
    *,
    match: str | None = None,
    variables: str | list[str] | None = None,
    mode: str = "w",
    bronze_dir: Path,
    fetcher: Fetcher,
) -> None:
    """Assemble *dataset*'s matched files into one Zarr store at *store*.

    Callers guard on ``spec(dataset).grid.cube`` first — a kind with no cube
    builder needs none, which is a fact about the kind rather than a branch.
    """
    reader = _grid_reader(dataset)
    if reader.cube is None:
        fmt = COLLECTION_META[dataset]["format"]
        raise ValueError(f"Dataset {dataset!r} is {fmt}; a multi-file match= already combines on read.")
    if match is None:
        raise ValueError(
            f'stack= needs match= to scope the cube for {dataset!r}, e.g. match="{reader.cube_match_example}".'
        )
    reader.require()
    files = ensure_grid_files(
        dataset,
        bronze_dir,
        suffixes=reader.suffixes,
        match=match,
        max_files=reader.cube_max_files,
        run_datetime=reader.run_datetime,
        fetcher=fetcher,
    )
    reader.cube(
        files,
        store,
        dataset=dataset,
        bronze_dir=bronze_dir,
        fetcher=fetcher,
        variables=variables,
        mode=mode,
    )


def convert(dataset: str, bronze_dir: Path, parquet_dir: Path) -> int:
    """Convert *dataset* to Parquet, or return 0 for a kind that has no Parquet path."""
    converter = spec(dataset).convert
    return 0 if converter is None else converter(dataset, bronze_dir, parquet_dir)


def tabular_datasets() -> list[str]:
    """Every dataset loadable as a DataFrame, in declaration order."""
    return [key for key in COLLECTIONS if KINDS[KIND_OF[key]].tabular]


def grid_datasets() -> list[str]:
    """Every dataset read as a grid, in declaration order."""
    return [key for key in COLLECTIONS if KINDS[KIND_OF[key]].is_grid]


def non_grid_datasets() -> list[str]:
    """Every dataset that is not a grid, in declaration order.

    What "download everything by default" has always meant: skip the large
    binary collections unless asked for them. The CLI and the Databricks script
    used ``tabular_datasets()`` for it and then added ``climate_normals`` back by
    hand, because that dataset converts to Parquet without being loadable.
    """
    return [key for key in COLLECTIONS if not KINDS[KIND_OF[key]].is_grid]
