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


def test_a_table_that_does_not_exist_names_the_ones_that_do():
    with pytest.raises(ValueError, match="Unknown metadata table 'stations_v2'"):
        fetch_table("smn", "stations_v2", fetcher=_fake())

    with pytest.raises(ValueError, match="parameters, stations, inventory"):
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
    source = ";".join(spec.columns) + "\n" + ";".join("x" for _ in spec.columns) + "\n"
    fake = _fake(f"https://data.geo.admin.ch/x/ogd-smn{spec.suffix}.csv", body=source)

    df = fetch_table("smn", table, fetcher=fake)

    assert df.columns == list(spec.columns.values())
    assert df.height == 1
    assert isinstance(df, pl.DataFrame)
