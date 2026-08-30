"""What each :class:`~foehn.collections.DatasetKind` can do, and who does it.

One table replacing the routing ladders that used to sit in ``api``, ``cli`` and
the Databricks ingest script, each re-deriving from a different set of dataset
keys. Callers ask the registry instead of testing membership.

Layering: ``collections`` (dataset facts) → ``client``/``convert`` (adapters) →
this module → ``api``/``cli``/``mcp_server``. The load readers deliberately stay
in ``api``: they are its implementation, and hoisting them here just to fill a
column would invert that dependency. What the registry says about loading is
which filters a kind supports, which is a fact about the data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from foehn.client import (
    DownloadResult,
    download_climate_scenarios_indoor,
    download_collection,
    download_grib2,
    download_metadata,
    download_netcdf,
)
from foehn.collections import COLLECTIONS, KIND_OF, DatasetKind, kind
from foehn.convert import (
    convert_climate_scenarios_indoor_to_parquet,
    convert_climate_scenarios_to_parquet,
    convert_to_parquet,
)
from foehn.fetch import DEFAULT_WORKERS, Fetcher

# Every download adapter takes the same arguments and ignores what its kind does
# not use, so callers never have to know which ones apply. The alternative — a
# per-kind call shape — is the ladder we are removing.
DownloadAdapter = Callable[..., DownloadResult]
ConvertAdapter = Callable[[str, Path, Path], int]


def _download_standard(
    dataset: str,
    bronze_dir: Path,
    *,
    time_slice: list[str],
    since: str | None,
    workers: int,
    force: bool,
    fetcher: Fetcher,
) -> DownloadResult:
    """Collection-level metadata plus the per-station CSVs, as one result.

    Both callers previously did this pairing themselves — ``api.download``
    summing the two results, the CLI adding their ``failed`` counts separately.
    """
    meta = download_metadata(dataset, bronze_dir, workers=workers, fetcher=fetcher)
    coll = download_collection(
        dataset, bronze_dir, data_types=time_slice, since=since, workers=workers, fetcher=fetcher
    )
    return DownloadResult(
        total_assets=meta.total_assets + coll.total_assets,
        downloaded=meta.downloaded + coll.downloaded,
        skipped=meta.skipped + coll.skipped,
        failed=meta.failed + coll.failed,
        filenames=meta.filenames + coll.filenames,
    )


def _download_archive(dataset: str, bronze_dir: Path, *, force: bool, fetcher: Fetcher, **_: object) -> DownloadResult:
    return download_climate_scenarios_indoor(bronze_dir, dataset, force=force, fetcher=fetcher)


def _download_grib2(
    dataset: str, bronze_dir: Path, *, since: str | None, workers: int, fetcher: Fetcher, **_: object
) -> DownloadResult:
    return download_grib2(dataset, bronze_dir, since=since, workers=workers, fetcher=fetcher)


def _download_netcdf(
    dataset: str, bronze_dir: Path, *, since: str | None, workers: int, fetcher: Fetcher, **_: object
) -> DownloadResult:
    return download_netcdf(dataset, bronze_dir, since=since, workers=workers, fetcher=fetcher)


def _convert_standard(dataset: str, bronze_dir: Path, parquet_dir: Path) -> int:
    return convert_to_parquet(dataset, bronze_dir, parquet_dir)


def _convert_preamble(_dataset: str, bronze_dir: Path, parquet_dir: Path) -> int:
    return convert_climate_scenarios_to_parquet(bronze_dir, parquet_dir)


def _convert_archive(_dataset: str, bronze_dir: Path, parquet_dir: Path) -> int:
    return convert_climate_scenarios_indoor_to_parquet(bronze_dir, parquet_dir)


@dataclass(frozen=True)
class KindSpec:
    """Everything the routing needs to know about one kind.

    ``convert`` is None for the grid kinds: they have no Parquet path at all,
    which is a fact about the kind rather than a branch each caller re-derives.
    """

    download: DownloadAdapter
    convert: ConvertAdapter | None
    tabular: bool
    """Loadable as a Polars DataFrame. False for the grid kinds (use open_dataset)."""

    supports_granularity: bool
    """Whether ``frequency=`` applies — false where filenames carry no granularity."""

    supports_calendar_filters: bool
    """Whether year/month/date_from/date_to apply. False for nominal 30-year dates."""

    key_columns: tuple[str, ...] = ("station_abbr", "reference_timestamp")
    """Columns an explicit ``columns=`` selection always keeps."""


KINDS: dict[DatasetKind, KindSpec] = {
    DatasetKind.STANDARD_CSV: KindSpec(
        download=_download_standard,
        convert=_convert_standard,
        tabular=True,
        supports_granularity=True,
        supports_calendar_filters=True,
    ),
    DatasetKind.PREAMBLE_CSV: KindSpec(
        download=_download_standard,
        convert=_convert_preamble,
        tabular=True,
        supports_granularity=False,
        # Dates are nominal (0001..0030 on a 365-day calendar), so the calendar
        # filters would silently match nothing.
        supports_calendar_filters=False,
        key_columns=("station_abbr", "variable", "gwl", "date"),
    ),
    DatasetKind.ARCHIVE_CSV: KindSpec(
        download=_download_archive,
        convert=_convert_archive,
        tabular=True,
        supports_granularity=False,
        supports_calendar_filters=True,
        key_columns=("station_abbr", "reference_timestamp", "period", "scenario", "variant"),
    ),
    DatasetKind.FORECAST_CSV: KindSpec(
        download=_download_standard,
        convert=_convert_standard,
        tabular=True,
        supports_granularity=False,
        supports_calendar_filters=True,
    ),
    DatasetKind.NETCDF_GRID: KindSpec(
        download=_download_netcdf,
        convert=None,
        tabular=False,
        supports_granularity=False,
        supports_calendar_filters=False,
    ),
    DatasetKind.GRIB2_GRID: KindSpec(
        download=_download_grib2,
        convert=None,
        tabular=False,
        supports_granularity=False,
        supports_calendar_filters=False,
    ),
    DatasetKind.RADAR_GRID: KindSpec(
        download=_download_grib2,
        convert=None,
        tabular=False,
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


def convert(dataset: str, bronze_dir: Path, parquet_dir: Path) -> int:
    """Convert *dataset* to Parquet, or return 0 for a kind that has no Parquet path."""
    converter = spec(dataset).convert
    return 0 if converter is None else converter(dataset, bronze_dir, parquet_dir)


def tabular_datasets() -> list[str]:
    """Every dataset loadable as a DataFrame, in declaration order."""
    return [key for key in COLLECTIONS if KINDS[KIND_OF[key]].tabular]


def grid_datasets() -> list[str]:
    """Every dataset read as a grid, in declaration order."""
    return [key for key in COLLECTIONS if not KINDS[KIND_OF[key]].tabular]
