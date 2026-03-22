"""Tests for the public Python API."""

from unittest.mock import patch

import pytest

import foehn
from foehn.api import convert, fetch, list_collections
from foehn.collections import COLLECTIONS


def test_import_from_foehn():
    assert callable(foehn.fetch)
    assert callable(foehn.convert)
    assert callable(foehn.list_collections)
    assert callable(foehn.discover)


def test_list_collections_returns_all():
    rows = list_collections()
    assert len(rows) == len(COLLECTIONS)
    for row in rows:
        assert set(row.keys()) == {"category", "key", "collection_id"}


def test_list_collections_categories():
    rows = {r["key"]: r["category"] for r in list_collections()}
    assert rows["smn"] == "CSV"
    assert rows["forecast_icon_ch1"] == "GRIB2"
    assert rows["surface_derived_grid"] == "NetCDF"
    assert rows["forecast_local"] == "CSV (forecast)"


def test_fetch_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown collection key"):
        fetch("nonexistent")


def test_fetch_grib2_key_raises():
    with pytest.raises(ValueError, match="binary/grid collection"):
        fetch("forecast_icon_ch1")


def test_fetch_netcdf_key_raises():
    with pytest.raises(ValueError, match="binary/grid collection"):
        fetch("surface_derived_grid")


@patch("foehn.api.download_collection")
@patch("foehn.api.download_metadata")
def test_fetch_calls_underlying_functions(mock_meta, mock_dl, tmp_path):
    fetch("smn", data_dir=tmp_path, data_types=["historical"])
    mock_meta.assert_called_once_with("smn", tmp_path / "raw")
    mock_dl.assert_called_once_with("smn", tmp_path / "raw", data_types=["historical"], since=None)


def test_convert_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown collection key"):
        convert("nonexistent")


@patch("foehn.api.convert_to_parquet")
def test_convert_calls_convert_to_parquet(mock_conv, tmp_path):
    convert("smn", data_dir=tmp_path)
    mock_conv.assert_called_once_with("smn", tmp_path / "raw", tmp_path / "parquet")
