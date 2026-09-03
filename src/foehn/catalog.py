"""The immutable Dataset catalogue.

Each Dataset is declared exactly once with its Collection, Dataset kind, and
published catalogue facts.  Compatibility views are derived at the bottom for
callers that still import ``COLLECTIONS``, ``COLLECTION_META``, or ``KIND_OF``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class DatasetKind(StrEnum):
    """Which pipeline handles a Dataset: download, convert, and read paths."""

    STANDARD_CSV = "standard_csv"
    PREAMBLE_CSV = "preamble_csv"
    ARCHIVE_CSV = "archive_csv"
    FORECAST_CSV = "forecast_csv"
    NETCDF_GRID = "netcdf_grid"
    GRIB2_GRID = "grib2_grid"
    RADAR_GRID = "radar_grid"
    DIRECT_ZIP = "direct_zip"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Every declared fact about one Dataset."""

    collection: str
    kind: DatasetKind
    category: str
    subcategory: str
    description: str
    format: str
    frequencies: tuple[str, ...] = ()
    time_slices: tuple[str, ...] = ()

    def published(self) -> dict[str, object]:
        """Return detached public catalogue data for this Dataset."""
        return {
            "category": self.category,
            "subcategory": self.subcategory,
            "description": self.description,
            "format": self.format,
            "frequencies": list(self.frequencies),
            "time_slices": list(self.time_slices),
        }


DATASETS: Mapping[str, DatasetSpec] = MappingProxyType(
    {
        "smn": DatasetSpec(
            "ch.meteoschweiz.ogd-smn",
            DatasetKind.STANDARD_CSV,
            "A",
            "A1",
            "Automatic weather stations",
            "CSV",
            ("t", "h", "d", "m", "y"),
            ("historical", "recent", "now"),
        ),
        "smn_precip": DatasetSpec(
            "ch.meteoschweiz.ogd-smn-precip",
            DatasetKind.STANDARD_CSV,
            "A",
            "A2",
            "Automatic precipitation stations",
            "CSV",
            ("t", "h", "d", "m", "y"),
            ("historical", "recent", "now"),
        ),
        "smn_tower": DatasetSpec(
            "ch.meteoschweiz.ogd-smn-tower",
            DatasetKind.STANDARD_CSV,
            "A",
            "A3",
            "Automatic tower stations",
            "CSV",
            ("t", "h", "d", "m", "y"),
            ("historical", "recent", "now"),
        ),
        "nime": DatasetSpec(
            "ch.meteoschweiz.ogd-nime",
            DatasetKind.STANDARD_CSV,
            "A",
            "A5",
            "Manual precipitation stations",
            "CSV",
            ("d", "m", "y"),
            ("historical", "recent"),
        ),
        "tot": DatasetSpec(
            "ch.meteoschweiz.ogd-tot",
            DatasetKind.STANDARD_CSV,
            "A",
            "A6",
            "Totaliser precipitation",
            "CSV",
            ("y",),
        ),
        "obs": DatasetSpec(
            "ch.meteoschweiz.ogd-obs",
            DatasetKind.STANDARD_CSV,
            "A",
            "A8",
            "Meteorological visual observations",
            "CSV",
            ("d", "m", "y"),
            ("historical", "recent"),
        ),
        "pollen": DatasetSpec(
            "ch.meteoschweiz.ogd-pollen",
            DatasetKind.STANDARD_CSV,
            "A",
            "A7",
            "Pollen stations",
            "CSV",
            ("h", "d", "y"),
            ("historical", "recent", "now"),
        ),
        "phenology": DatasetSpec(
            "ch.meteoschweiz.ogd-phenology",
            DatasetKind.STANDARD_CSV,
            "A",
            "A9",
            "Phenological observations",
            "CSV",
            ("y",),
        ),
        "nbcn": DatasetSpec(
            "ch.meteoschweiz.ogd-nbcn",
            DatasetKind.STANDARD_CSV,
            "C",
            "C1",
            "Climate stations, homogeneous",
            "CSV",
            ("d", "m", "y"),
            ("historical", "recent"),
        ),
        "nbcn_precip": DatasetSpec(
            "ch.meteoschweiz.ogd-nbcn-precip",
            DatasetKind.STANDARD_CSV,
            "C",
            "C2",
            "Climate precipitation, homogeneous",
            "CSV",
            ("m", "y"),
        ),
        "surface_derived_grid": DatasetSpec(
            "ch.meteoschweiz.ogd-surface-derived-grid",
            DatasetKind.NETCDF_GRID,
            "C",
            "C3",
            "Precipitation, temperature, sunshine grids",
            "NetCDF",
        ),
        "satellite_derived_grid": DatasetSpec(
            "ch.meteoschweiz.ogd-satellite-derived-grid",
            DatasetKind.NETCDF_GRID,
            "C",
            "C4",
            "Radiation, clouds, land surface temperature grids",
            "NetCDF",
        ),
        "radar_derived_grid": DatasetSpec(
            "ch.meteoschweiz.ogd-radar-derived-grid",
            DatasetKind.NETCDF_GRID,
            "C",
            "C5",
            "Hail spatial climate analyses (hail days, return periods)",
            "NetCDF",
        ),
        "climate_normals": DatasetSpec(
            "ch.meteoschweiz.klima",
            DatasetKind.DIRECT_ZIP,
            "C",
            "C6",
            "Station climate normals, 1961-1990 and 1991-2020 (monthly + yearly)",
            "TXT",
        ),
        "climate_normals_grid": DatasetSpec(
            "ch.meteoschweiz.ogd-climate-normals-grid",
            DatasetKind.NETCDF_GRID,
            "C",
            "C7",
            "Spatial climate normals (temp/precip/sunshine + radiation/clouds)",
            "NetCDF",
        ),
        "climate_scenarios": DatasetSpec(
            "ch.meteoschweiz.ogd-climate-scenarios-ch2025",
            DatasetKind.PREAMBLE_CSV,
            "C",
            "C8",
            "Climate scenarios CH2025 local (daily)",
            "CSV",
        ),
        "climate_scenarios_grid": DatasetSpec(
            "ch.meteoschweiz.ogd-climate-scenarios-ch2025-grid",
            DatasetKind.NETCDF_GRID,
            "C",
            "C9",
            "Climate scenarios CH2025 gridded",
            "NetCDF",
        ),
        "climate_scenarios_indoor": DatasetSpec(
            "ch.meteoschweiz.klimaszenarien-raumklima",
            DatasetKind.ARCHIVE_CSV,
            "C",
            "C",
            "Indoor climate scenarios (hourly)",
            "CSV",
        ),
        "radar_precip": DatasetSpec(
            "ch.meteoschweiz.ogd-radar-precip",
            DatasetKind.RADAR_GRID,
            "D",
            "D1",
            "Precipitation radar",
            "HDF5",
        ),
        "radar_hail": DatasetSpec(
            "ch.meteoschweiz.ogd-radar-hail",
            DatasetKind.RADAR_GRID,
            "D",
            "D3",
            "Hail radar",
            "HDF5",
        ),
        "forecast_icon_ch1": DatasetSpec(
            "ch.meteoschweiz.ogd-forecasting-icon-ch1",
            DatasetKind.GRIB2_GRID,
            "E",
            "E2",
            "ICON-CH1-EPS 1km",
            "GRIB2",
        ),
        "forecast_icon_ch2": DatasetSpec(
            "ch.meteoschweiz.ogd-forecasting-icon-ch2",
            DatasetKind.GRIB2_GRID,
            "E",
            "E3",
            "ICON-CH2-EPS 2.1km",
            "GRIB2",
        ),
        "forecast_local": DatasetSpec(
            "ch.meteoschweiz.ogd-local-forecasting",
            DatasetKind.FORECAST_CSV,
            "E",
            "E4",
            "Local point forecasts (hourly + daily)",
            "CSV",
        ),
        "analysis_kenda_ch1": DatasetSpec(
            "ch.meteoschweiz.ogd-analysis-kenda-ch1",
            DatasetKind.GRIB2_GRID,
            "E",
            "E5",
            "Numerical weather analysis KENDA-CH1",
            "GRIB2",
        ),
    }
)


class _ReadOnlyList(list):
    """A list that refuses mutation but is still a list to ``json`` and ``==``.

    The nested frequency and time-slice values. A tuple would be immutable but
    would also stop comparing equal to the list literals v0.4.0 callers wrote,
    which is the whole reason these views exist.
    """

    __slots__ = ()

    def _read_only(self, *_args: object, **_kwargs: object):
        raise TypeError("catalog values are read-only; change catalog.DATASETS instead.")

    __setitem__ = _read_only
    __delitem__ = _read_only
    __iadd__ = _read_only
    __imul__ = _read_only
    append = _read_only
    extend = _read_only
    insert = _read_only
    remove = _read_only
    pop = _read_only
    clear = _read_only
    sort = _read_only
    reverse = _read_only

    # copy/deepcopy/pickle hand back plain builtins: a caller who copies a view
    # wants something they can work with, and reconstructing this type would go
    # through the very methods that are closed off.
    def __copy__(self) -> list:
        return list(self)

    def __deepcopy__(self, memo: dict) -> list:
        import copy as _copy

        return [_copy.deepcopy(item, memo) for item in self]

    def __reduce__(self):
        return (list, (list(self),))


class _ReadOnlyDict(dict):
    """A dict that refuses mutation but is still a dict to ``json`` and ``==``.

    The compatibility views need both halves and ``MappingProxyType`` only gives
    one. As a proxy they were immutable but not value-compatible:
    ``json.dumps(COLLECTION_META)`` raised, and comparing a row against the
    literal a caller wrote at v0.4.0 was quietly False. As plain dicts they were
    value-compatible but no longer immutable, and one stray
    ``COLLECTIONS["smn"] = ...`` silently repointed a dataset for the rest of
    the process. A dict subclass is both: the C JSON encoder and ``==`` see the
    dict storage, while every mutating method is closed off.
    """

    __slots__ = ()

    def _read_only(self, *_args: object, **_kwargs: object):
        raise TypeError(
            f"{type(self).__name__} is a read-only view derived from catalog.DATASETS; "
            "add or change a dataset there instead."
        )

    __setitem__ = _read_only
    __delitem__ = _read_only
    # ``|=`` is a mutation like any other. dict supplies __ior__, so leaving it
    # alone left a way straight past every other guard here.
    __ior__ = _read_only
    update = _read_only
    setdefault = _read_only
    pop = _read_only
    popitem = _read_only
    clear = _read_only

    def __copy__(self) -> dict:
        return dict(self)

    def __deepcopy__(self, memo: dict) -> dict:
        import copy as _copy

        return {key: _copy.deepcopy(value, memo) for key, value in self.items()}

    def __reduce__(self):
        return (dict, (dict(self),))


# Backwards-compatible names: read-only views derived from DATASETS, which stays
# the one source. The *values* are the plain dicts and lists v0.4.0 published,
# because that is the entire point of a compatibility view — a view that is not
# value-compatible is just a second break.
COLLECTIONS: Mapping[str, str] = _ReadOnlyDict({key: row.collection for key, row in DATASETS.items()})
KIND_OF: Mapping[str, DatasetKind] = _ReadOnlyDict({key: row.kind for key, row in DATASETS.items()})
COLLECTION_META: Mapping[str, dict[str, object]] = _ReadOnlyDict(
    {
        key: _ReadOnlyDict(
            {
                "category": row.category,
                "subcategory": row.subcategory,
                "description": row.description,
                "format": row.format,
                "frequencies": _ReadOnlyList(row.frequencies),
                "time_slices": _ReadOnlyList(row.time_slices),
            }
        )
        for key, row in DATASETS.items()
    }
)


__all__ = [
    "COLLECTIONS",
    "COLLECTION_META",
    "DATASETS",
    "KIND_OF",
    "DatasetKind",
    "DatasetSpec",
]
