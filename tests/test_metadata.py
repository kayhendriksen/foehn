"""Tests at :mod:`foehn.metadata`'s interface.

The three public wrappers are covered through ``foehn.parameters`` and friends in
``test_api``, which is where callers meet them. These go at the module itself,
passing the fetcher in — which is what moving it out of ``api`` bought, and why
none of them touches the process-wide default.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

from foehn.metadata import TABLES, fetch_table
from tests.fakes import InMemoryFetcher, stac_collection

_PARAMETERS_CSV = (
    "parameter_shortname;parameter_description_en;parameter_unit;parameter_datatype;"
    "parameter_granularity;parameter_decimals;parameter_group_en\n"
    "tre200d0;Air temperature 2 m;°C;Float;d;1;Temperature\n"
)


def _fake(*hrefs, body=_PARAMETERS_CSV):
    fake = InMemoryFetcher()
    fake.any_collection = stac_collection("ch.meteoschweiz.ogd-smn", *hrefs)
    fake.default_body = body
    return fake


def test_a_table_is_published_under_foehns_column_names():
    """The suffix is MeteoSwiss's, the names are foehn's — that pair is the whole table."""
    fake = _fake("https://data.geo.admin.ch/x/ogd-smn_meta_parameters.csv")

    df = fetch_table("smn", "parameters", fetcher=fake)

    assert df.columns == list(TABLES["parameters"].columns.values())
    assert df["shortname"].to_list() == ["tre200d0"]
    assert df["unit"].to_list() == ["°C"]


def test_the_columns_come_back_in_the_declared_order():
    """Order is part of the table, not an accident of the source file's column order."""
    fake = _fake("https://data.geo.admin.ch/x/ogd-smn_meta_parameters.csv")

    df = fetch_table("smn", "parameters", fetcher=fake)

    assert df.columns == ["shortname", "description", "unit", "type", "granularity", "decimals", "group"]


def test_inventory_declares_an_open_ended_data_till():
    field = TABLES["inventory"].field("data_till")

    assert field.nullable is True
    assert field.annotation == str | None


def test_an_open_ended_inventory_row_remains_null():
    source = "station_abbr;parameter_shortname;data_since;data_till;owner\nBER;tre200d0;1864-01-01;;MeteoSchweiz\n"
    fake = _fake("https://data.geo.admin.ch/x/ogd-smn_meta_datainventory.csv", body=source)

    df = fetch_table("smn", "inventory", fetcher=fake)

    assert df["data_till"].to_list() == [None]


def test_a_table_that_does_not_exist_names_the_ones_that_do():
    with pytest.raises(ValueError, match="Unknown metadata table 'stations_v2'"):
        fetch_table("smn", "stations_v2", fetcher=_fake())

    with pytest.raises(ValueError, match="parameters, stations, points, inventory"):
        fetch_table("smn", "stations_v2", fetcher=_fake())


def test_a_collection_missing_the_asset_names_the_suffix_it_looked_for():
    """The collection is there and the file is not — which is upstream's problem, said plainly."""
    fake = _fake("https://data.geo.admin.ch/x/ogd-smn_meta_stations.csv")

    with pytest.raises(ValueError, match=re.escape("No _meta_parameters metadata found for dataset 'smn'")):
        fetch_table("smn", "parameters", fetcher=fake)


def test_an_unknown_dataset_is_refused_before_any_fetch():
    """``collection_id`` raises the sentence; nothing here restates it."""
    fake = _fake()

    with pytest.raises(ValueError, match="Unknown dataset"):
        fetch_table("not_a_dataset", "parameters", fetcher=fake)

    assert fake.collection_calls == []


def test_a_windows_1252_metadata_file_is_decoded():
    """MeteoSwiss ships both encodings; the degree sign is where it shows."""
    fake = _fake("https://data.geo.admin.ch/x/ogd-smn_meta_parameters.csv", body=_PARAMETERS_CSV.encode("windows-1252"))

    df = fetch_table("smn", "parameters", fetcher=fake)

    assert df["unit"].to_list() == ["°C"]


@pytest.mark.parametrize("table", sorted(TABLES))
def test_every_table_publishes_every_column_it_declares(table):
    """A rename map naming a column the source does not have fails at read, not at review."""
    spec = TABLES[table]
    # A value of each column's declared type: the declarations are the parse types.
    sample = {str: "x", int: "1", float: "1.5"}
    values = [sample.get(field.python_type, "1") for field in spec.fields]
    source = ";".join(spec.columns) + "\n" + ";".join(values) + "\n"
    fake = _fake(f"https://data.geo.admin.ch/x/ogd-smn{spec.suffix}.csv", body=source)

    df = fetch_table("smn", table, fetcher=fake)

    assert df.columns == list(spec.columns.values())
    assert df.height == 1
    assert isinstance(df, pl.DataFrame)


def test_a_direct_zip_dataset_says_it_has_no_metadata_instead_of_a_404():
    fake = InMemoryFetcher()
    fake.fail("ch.meteoschweiz.klima")

    with pytest.raises(ValueError, match="direct ZIP download"):
        fetch_table("climate_normals", "stations", fetcher=fake)
    assert fake.collection_calls == []


def test_a_missing_table_names_the_ones_the_collection_publishes():
    fake = _fake("https://data.geo.admin.ch/x/ogd-local-forecasting_meta_point.csv")

    with pytest.raises(ValueError, match="It publishes: points"):
        fetch_table("forecast_local", "stations", fetcher=fake)


def test_forecast_points_publish_their_columns():
    body = (
        b"point_id;point_type_id;station_abbr;postal_code;point_name;point_type_de;point_type_fr;"
        b"point_type_it;point_type_en;point_height_masl;point_coordinates_lv95_east;"
        b"point_coordinates_lv95_north;point_coordinates_wgs84_lat;point_coordinates_wgs84_lon\n"
        b"78;1;BER;;Bern / Zollikofen;Station;Station;Stazione;Station;553.0;2601934.0;1204410.0;46.99;7.46\n"
    )
    fake = _fake("https://data.geo.admin.ch/x/ogd-local-forecasting_meta_point.csv", body=body)

    df = fetch_table("forecast_local", "points", fetcher=fake)

    assert df.columns == list(TABLES["points"].columns.values())
    assert df.row(0, named=True)["abbr"] == "BER"


def test_points_parse_with_their_declared_types_whatever_the_rows_look_like():
    """A first postal code of 3011 used to read the column as integers.

    The MCP ``Point`` model and its string search both then rejected the table.
    Tested across the real CSV -> table -> MCP seam rather than a typed mock.
    """
    from unittest.mock import patch

    pytest.importorskip("mcp")
    from foehn.mcp_server import get_points

    body = (
        b"point_id;point_type_id;station_abbr;postal_code;point_name;point_type_en;point_height_masl;"
        b"point_coordinates_lv95_east;point_coordinates_lv95_north;point_coordinates_wgs84_lat;"
        b"point_coordinates_wgs84_lon\n"
        b"301100;2;;3011;Bern;Postal code center;550;2600000;1200000;46.9;7.4\n"
        b"78;1;BER;;Bern / Zollikofen;Station;553;2601934;1204410;46.99;7.46\n"
    )
    fake = _fake("https://data.geo.admin.ch/x/ogd-local-forecasting_meta_point.csv", body=body)

    df = fetch_table("forecast_local", "points", fetcher=fake)
    assert df.schema["postal_code"] == pl.String
    with patch("foehn.api.default_fetcher", return_value=fake):
        assert [p.point_id for p in get_points("forecast_local")] == [301100, 78]
        assert [p.point_id for p in get_points("forecast_local", query="3011")] == [301100]


def test_a_compressed_metadata_table_is_refused():
    import zlib

    body = zlib.compress(b"parameter_shortname;parameter_description_en\ntre200d0;Air temperature\n", 1)
    fake = _fake("https://data.geo.admin.ch/x/ogd-smn_meta_parameters.csv", body=body)

    with pytest.raises(ValueError, match="compressed data"):
        fetch_table("smn", "parameters", fetcher=fake)
