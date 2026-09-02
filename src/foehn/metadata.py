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
from foehn.collections import collection_id
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
    coll = fetcher.collection(collection_id(dataset))
    for asset in collection_assets(coll, suffixes=(".csv",), contains=spec.suffix):
        content = decode_meteoswiss_csv(fetcher.get(asset.href, timeout=60).body)
        df = pl.read_csv(content.encode("utf-8"), separator=";")
        return df.select(pl.col(field.source).alias(field.published) for field in spec.fields)

    raise ValueError(f"No {spec.suffix} metadata found for dataset {dataset!r}.")


__all__ = ["TABLES", "MetadataField", "MetadataTable", "fetch_table"]
