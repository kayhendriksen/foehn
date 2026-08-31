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

from dataclasses import dataclass

import polars as pl

from foehn.assets import collection_assets
from foehn.collections import collection_id
from foehn.fetch import Fetcher
from foehn.meteocsv import decode_meteoswiss_csv


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

    columns: dict[str, str]
    """Source column → published name, in the order the frame comes back."""


TABLES: dict[str, MetadataTable] = {
    "parameters": MetadataTable(
        "_meta_parameters",
        {
            "parameter_shortname": "shortname",
            "parameter_description_en": "description",
            "parameter_unit": "unit",
            "parameter_datatype": "type",
            "parameter_granularity": "granularity",
            "parameter_decimals": "decimals",
            "parameter_group_en": "group",
        },
    ),
    "stations": MetadataTable(
        "_meta_stations",
        {
            "station_abbr": "abbr",
            "station_name": "name",
            "station_canton": "canton",
            "station_height_masl": "altitude",
            "station_coordinates_lv95_east": "lv95_east",
            "station_coordinates_lv95_north": "lv95_north",
            "station_coordinates_wgs84_lat": "lat",
            "station_coordinates_wgs84_lon": "lon",
            "station_data_since": "data_since",
        },
    ),
    "inventory": MetadataTable(
        "_meta_datainventory",
        {
            "station_abbr": "station",
            "parameter_shortname": "parameter",
            "data_since": "data_since",
            "data_till": "data_till",
            "owner": "owner",
        },
    ),
}


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
        return df.select(pl.col(source).alias(published) for source, published in spec.columns.items())

    raise ValueError(f"No {spec.suffix} metadata found for dataset {dataset!r}.")


__all__ = ["TABLES", "MetadataTable", "fetch_table"]
