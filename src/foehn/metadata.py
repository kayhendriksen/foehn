"""A **Collection**'s metadata tables, and the columns foehn publishes from them.

The parameters, stations and inventory tables are collection-level **Assets**:
one shared set per **Collection**, fetched live rather than read from **Bronze**.
This is where they are turned into a frame.

Not routed through :mod:`foehn.registry` like the four kind-shaped stages, and
deliberately so: nothing about this varies by **Dataset kind**, so a row and an
adapter per kind would be a seam with nothing on either side of it. What it does
share with those stages is its shape — the **Fetcher** is passed in, so this is
reachable in a test without substituting the process-wide default.

The suffixes are MeteoSwiss's; the published names are foehn's. Stating the pair
once means a fourth ``_meta_*`` file is a row: it used to be three implied rename
maps in ``api``, an if-ladder in the CLI, and three models in the MCP layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import polars as pl

from foehn.assets import collection_assets
from foehn.collections import DatasetKind, collection_id, kind
from foehn.fetch import Fetcher
from foehn.meteocsv import decode_meteoswiss_csv


@dataclass(frozen=True)
class MetadataField:
    """One published field's source name, type, nullability, and description."""

    source: str
    published: str
    python_type: Any
    description: str
    nullable: bool = False

    @property
    def annotation(self) -> Any:
        return self.python_type | None if self.nullable else self.python_type


@dataclass(frozen=True)
class MetadataTable:
    """One collection-level metadata file, and the columns foehn publishes from it.

    The suffix is MeteoSwiss's; the renames are foehn's public column names, which
    is why the table lives here rather than in :mod:`foehn.meteocsv`. Stating it
    once means a fourth ``_meta_*`` file is a row — it used to be three implied
    rename maps here, an if-ladder in the CLI, and three models in the MCP layer.
    """

    suffix: str
    """Filename fragment identifying the file among a collection's assets."""

    fields: tuple[MetadataField, ...]

    @property
    def columns(self) -> Mapping[str, str]:
        """Immutable source → published compatibility view."""
        return MappingProxyType({field.source: field.published for field in self.fields})

    @property
    def schema_overrides(self) -> dict[str, type[pl.DataType]]:
        """Parse types for the fields whose declaration names exactly one.

        The CSV used to be read with inferred types while the MCP models were
        built from the declarations, and the two disagreed whenever inference
        guessed differently: a points table whose first postal code is ``3011``
        read that column as integers, which the ``Point`` model and the string
        search both rejected. Unions (``int | float``) are left to inference.
        """
        return {
            field.source: _POLARS_TYPES[field.python_type]
            for field in self.fields
            if field.python_type in _POLARS_TYPES
        }

    def field(self, published: str) -> MetadataField:
        for field in self.fields:
            if field.published == published:
                return field
        raise KeyError(published)

    def assert_model(self, model_fields: Mapping[str, object]) -> None:
        """Refuse an MCP Adapter whose names or annotations drift from this schema."""
        expected = {field.published: field.annotation for field in self.fields}
        actual = {name: getattr(value, "annotation", None) for name, value in model_fields.items()}
        if actual != expected:
            raise TypeError(f"Metadata model does not match {self.suffix}: expected {expected}, got {actual}")


_POLARS_TYPES: Mapping[Any, type[pl.DataType]] = MappingProxyType({str: pl.String, int: pl.Int64, float: pl.Float64})


def _field(
    source: str,
    published: str,
    python_type: Any,
    description: str,
    *,
    nullable: bool = False,
) -> MetadataField:
    return MetadataField(source, published, python_type, description, nullable)


TABLES: Mapping[str, MetadataTable] = MappingProxyType(
    {
        "parameters": MetadataTable(
            "_meta_parameters",
            (
                _field("parameter_shortname", "shortname", str, "Column name in data files (e.g. 'tre200s0')"),
                _field("parameter_description_en", "description", str, "Human-readable parameter description"),
                _field("parameter_unit", "unit", str, "Measurement unit (e.g. '°C', 'mm', 'hPa')"),
                _field("parameter_datatype", "type", str, "Data type"),
                _field("parameter_granularity", "granularity", str, "Temporal granularity"),
                _field("parameter_decimals", "decimals", int | str, "Number of decimal places"),
                _field("parameter_group_en", "group", str, "Parameter group (e.g. 'Temperature', 'Precipitation')"),
            ),
        ),
        "stations": MetadataTable(
            "_meta_stations",
            (
                _field("station_abbr", "abbr", str, "Station abbreviation (e.g. 'BER' for Bern)"),
                _field("station_name", "name", str, "Full station name"),
                _field("station_canton", "canton", str, "Swiss canton code (e.g. 'BE', 'ZH')"),
                _field("station_height_masl", "altitude", int | float, "Altitude in metres above sea level"),
                _field("station_coordinates_lv95_east", "lv95_east", float, "LV95 easting in metres (EPSG:2056)"),
                _field("station_coordinates_lv95_north", "lv95_north", float, "LV95 northing in metres (EPSG:2056)"),
                _field("station_coordinates_wgs84_lat", "lat", float, "WGS84 latitude"),
                _field("station_coordinates_wgs84_lon", "lon", float, "WGS84 longitude"),
                _field(
                    "station_data_since",
                    "data_since",
                    str,
                    "Date measurements started (DD.MM.YYYY, as published by MeteoSwiss)",
                ),
            ),
        ),
        # forecast_local publishes forecast points rather than stations: the
        # stations, plus one point per postal code and some points of interest.
        "points": MetadataTable(
            "_meta_point",
            (
                _field(
                    "point_id",
                    "point_id",
                    int,
                    "Forecast point ID; unique only together with point_type_id",
                ),
                _field("point_type_id", "point_type_id", int, "Point type ID, as in the point_type_id column"),
                _field(
                    "point_type_en",
                    "type",
                    str,
                    "Point type: 'Station', 'Postal code center' or 'Point of Interest'",
                ),
                _field(
                    "station_abbr",
                    "abbr",
                    str,
                    "Station abbreviation, or null when the point is not a station",
                    nullable=True,
                ),
                _field(
                    "postal_code",
                    "postal_code",
                    str,
                    "Swiss postal code (e.g. '3011'), or null when the point is not a postal code centre",
                    nullable=True,
                ),
                _field("point_name", "name", str, "Point name"),
                _field("point_height_masl", "altitude", int | float, "Altitude in metres above sea level"),
                _field("point_coordinates_lv95_east", "lv95_east", float, "LV95 easting in metres (EPSG:2056)"),
                _field("point_coordinates_lv95_north", "lv95_north", float, "LV95 northing in metres (EPSG:2056)"),
                _field("point_coordinates_wgs84_lat", "lat", float, "WGS84 latitude"),
                _field("point_coordinates_wgs84_lon", "lon", float, "WGS84 longitude"),
            ),
        ),
        "inventory": MetadataTable(
            "_meta_datainventory",
            (
                _field("station_abbr", "station", str, "Station abbreviation"),
                _field("parameter_shortname", "parameter", str, "Parameter shortname"),
                _field("data_since", "data_since", str, "Start of available data"),
                _field("data_till", "data_till", str, "End of available data, or null when still open", nullable=True),
                _field("owner", "owner", str, "Data owner"),
            ),
        ),
    }
)


def fetch_table(dataset: str, table: str, *, fetcher: Fetcher) -> pl.DataFrame:
    """Fetch one of a dataset's metadata tables from the STAC API.

    The single implementation behind ``foehn.parameters``, ``foehn.stations`` and
    ``foehn.inventory``, which differ only in which row they ask for.
    """
    if table not in TABLES:
        raise ValueError(f"Unknown metadata table {table!r}. Valid options: {', '.join(TABLES)}.")

    spec = TABLES[table]
    if kind(dataset) is DatasetKind.DIRECT_ZIP:
        # Not a STAC Collection at all, so asking the API for one is a bare 404.
        raise ValueError(
            f"Dataset {dataset!r} is a direct ZIP download rather than a STAC collection, "
            "so it publishes no metadata tables."
        )
    coll = fetcher.collection(collection_id(dataset))
    csv_assets = collection_assets(coll, suffixes=(".csv",))
    for asset in csv_assets:
        if spec.suffix not in asset.name:
            continue
        content = decode_meteoswiss_csv(fetcher.get(asset.href, timeout=60).body)
        df = pl.read_csv(content.encode("utf-8"), separator=";", schema_overrides=spec.schema_overrides)
        return df.select(pl.col(field.source).alias(field.published) for field in spec.fields)

    available = [name for name, other in TABLES.items() if any(other.suffix in a.name for a in csv_assets)]
    hint = f" It publishes: {', '.join(available)}." if available else ""
    raise ValueError(f"No {spec.suffix} metadata found for dataset {dataset!r}.{hint}")


__all__ = ["TABLES", "MetadataField", "MetadataTable", "fetch_table"]
