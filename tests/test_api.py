"""Tests for the public Python API."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import foehn
from foehn.api import download, list_datasets, load, to_parquet
from foehn.collections import COLLECTIONS


def test_import_from_foehn():
    assert callable(foehn.download)
    assert callable(foehn.to_parquet)
    assert callable(foehn.list_datasets)


def test_list_datasets_returns_all():
    rows = list_datasets()
    assert len(rows) == len(COLLECTIONS)
    expected_keys = {
        "dataset",
        "collection_id",
        "category",
        "subcategory",
        "description",
        "format",
        "frequencies",
        "time_slices",
    }
    for row in rows:
        assert set(row.keys()) == expected_keys


def test_list_datasets_categories():
    rows = {r["dataset"]: r for r in list_datasets()}
    assert rows["smn"]["category"] == "A"
    assert rows["smn"]["format"] == "CSV"
    assert rows["forecast_icon_ch1"]["format"] == "GRIB2"
    assert rows["surface_derived_grid"]["format"] == "NetCDF"
    assert rows["forecast_local"]["format"] == "CSV"


def test_download_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        download("nonexistent")


def test_download_grib2_dataset_raises():
    with pytest.raises(ValueError, match="binary/grid dataset"):
        download("forecast_icon_ch1")


def test_download_netcdf_dataset_raises():
    with pytest.raises(ValueError, match="binary/grid dataset"):
        download("surface_derived_grid")


@patch("foehn.api.download_collection")
@patch("foehn.api.download_metadata")
def test_download_calls_underlying_functions(mock_meta, mock_dl, tmp_path):
    download("smn", data_dir=tmp_path, time_slice=["historical"])
    mock_meta.assert_called_once_with("smn", tmp_path / "raw")
    mock_dl.assert_called_once_with("smn", tmp_path / "raw", data_types=["historical"], since=None)


def test_to_parquet_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        to_parquet("nonexistent")


@patch("foehn.api.convert_to_parquet")
def test_to_parquet_calls_convert_to_parquet(mock_conv, tmp_path):
    to_parquet("smn", data_dir=tmp_path)
    mock_conv.assert_called_once_with("smn", tmp_path / "raw", tmp_path / "parquet")


# --- read() tests ---


def test_load_is_exported():
    assert callable(foehn.load)


def test_load_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load("nonexistent")


def test_load_grib2_dataset_raises():
    with pytest.raises(ValueError, match="binary/grid dataset"):
        load("forecast_icon_ch1")


def test_load_netcdf_dataset_raises():
    with pytest.raises(ValueError, match="binary/grid dataset"):
        load("surface_derived_grid")


def test_load_frequency_on_unsupported_dataset_raises():
    """Frequency filter on datasets without standard filenames should raise."""
    with pytest.raises(ValueError, match="does not support frequency"):
        load("climate_scenarios", frequency="d")

    with pytest.raises(ValueError, match="does not support frequency"):
        load("forecast_local", frequency="h")


def _mock_response(content, status_code=200):
    """Create a mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    if isinstance(content, str):
        resp.content = content.encode("utf-8")
    else:
        resp.content = content
    resp.headers = {}
    resp.raise_for_status = MagicMock()
    return resp


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_returns_dataframe(mock_session, mock_meta, mock_items):
    """load() should download CSVs in memory and return a concatenated DataFrame."""
    # No metadata assets
    mock_meta.return_value = {"assets": {}}

    # Two STAC items, each with one CSV asset
    mock_items.return_value = [
        {"assets": {"data": {"href": "https://data.geo.admin.ch/smn/file1_recent.csv"}}},
        {"assets": {"data": {"href": "https://data.geo.admin.ch/smn/file2_recent.csv"}}},
    ]

    csv1 = "station;temperature\nBER;20.5\nZUR;18.3\n"
    csv2 = "station;temperature\nGEN;22.1\n"

    session = MagicMock()
    responses = [_mock_response(csv1), _mock_response(csv2)]
    session.get = MagicMock(side_effect=responses)
    mock_session.return_value = session

    df = load("smn")

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 3
    assert "station" in df.columns
    assert "temperature" in df.columns


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_with_metadata_types(mock_session, mock_meta, mock_items):
    """load() should use metadata to infer column types."""
    meta_csv = "parameter_shortname;parameter_datatype\nvalue;float\n"
    mock_meta.return_value = {
        "assets": {"params": {"href": "https://data.geo.admin.ch/smn/ogd-smn_meta_parameters.csv"}}
    }

    mock_items.return_value = [
        {"assets": {"data": {"href": "https://data.geo.admin.ch/smn/file_recent.csv"}}},
    ]

    data_csv = "station;value\nBER;20.5\n"

    session = MagicMock()
    session.get = MagicMock(
        side_effect=[
            _mock_response(meta_csv),
            _mock_response(data_csv),
        ]
    )
    mock_session.return_value = session

    df = load("smn")

    assert isinstance(df, pl.DataFrame)
    assert df.schema["value"] == pl.Float64


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_filters_time_slice(mock_session, mock_meta, mock_items):
    """load() should only include CSVs matching the requested time_slice."""
    mock_meta.return_value = {"assets": {}}

    mock_items.return_value = [
        {
            "assets": {
                "recent": {"href": "https://data.geo.admin.ch/smn/file_recent.csv"},
                "historical": {"href": "https://data.geo.admin.ch/smn/file_historical.csv"},
            }
        },
    ]

    csv_data = "station;temp\nBER;20\n"

    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(csv_data))
    mock_session.return_value = session

    df = load("smn", time_slice=["recent"])

    assert isinstance(df, pl.DataFrame)
    # Only one CSV should have been fetched (the recent one)
    assert session.get.call_count == 1


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_no_csvs_raises(mock_session, mock_meta, mock_items):
    """load() should raise ValueError when no CSVs match."""
    mock_meta.return_value = {"assets": {}}
    mock_items.return_value = [{"assets": {"data": {"href": "https://data.geo.admin.ch/smn/file_historical.csv"}}}]

    session = MagicMock()
    mock_session.return_value = session

    with pytest.raises(ValueError, match="No CSV files found"):
        load("smn", time_slice=["now"])


def _smn_items(*stations):
    """Build STAC items with realistic SMN filenames for the given station abbrevs."""
    return [
        {
            "id": stn.lower(),
            "assets": {
                "d_recent": {"href": f"https://data.geo.admin.ch/smn/ogd-smn_{stn.lower()}_d_recent.csv"},
                "h_recent": {"href": f"https://data.geo.admin.ch/smn/ogd-smn_{stn.lower()}_h_recent.csv"},
                "t_recent": {"href": f"https://data.geo.admin.ch/smn/ogd-smn_{stn.lower()}_t_recent.csv"},
            },
        }
        for stn in stations
    ]


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_station_filter_single(mock_session, mock_meta, mock_items):
    """load(station='ber') should only download files for that station."""
    mock_meta.return_value = {"assets": {}}
    mock_items.return_value = _smn_items("ber", "zur", "gen")

    csv_data = "station;temp\nBER;20\n"
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(csv_data))
    mock_session.return_value = session

    df = load("smn", station="BER")

    assert isinstance(df, pl.DataFrame)
    # 3 granularities (d, h, t) for 1 station
    assert session.get.call_count == 3


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_station_filter_multiple(mock_session, mock_meta, mock_items):
    """load(station=['ber', 'zur']) should download files for both stations."""
    mock_meta.return_value = {"assets": {}}
    mock_items.return_value = _smn_items("ber", "zur", "gen")

    csv_data = "station;temp\nX;20\n"
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(csv_data))
    mock_session.return_value = session

    df = load("smn", station=["BER", "ZUR"])

    assert isinstance(df, pl.DataFrame)
    # 3 granularities × 2 stations = 6
    assert session.get.call_count == 6


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_frequency_filter(mock_session, mock_meta, mock_items):
    """load(frequency='d') should only download daily files."""
    mock_meta.return_value = {"assets": {}}
    mock_items.return_value = _smn_items("ber")

    csv_data = "station;temp\nBER;20\n"
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(csv_data))
    mock_session.return_value = session

    df = load("smn", station="BER", frequency="d")

    assert isinstance(df, pl.DataFrame)
    # Only 1 file: ber_d_recent
    assert session.get.call_count == 1


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_frequency_filter_multiple(mock_session, mock_meta, mock_items):
    """load(frequency=['d', 'h']) should download daily + hourly."""
    mock_meta.return_value = {"assets": {}}
    mock_items.return_value = _smn_items("ber")

    csv_data = "station;temp\nBER;20\n"
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(csv_data))
    mock_session.return_value = session

    df = load("smn", station="BER", frequency=["d", "h"])

    assert isinstance(df, pl.DataFrame)
    # 2 files: ber_d_recent + ber_h_recent
    assert session.get.call_count == 2


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_station_case_insensitive(mock_session, mock_meta, mock_items):
    """Station filter should be case-insensitive."""
    mock_meta.return_value = {"assets": {}}
    mock_items.return_value = _smn_items("ber")

    csv_data = "station;temp\nBER;20\n"
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(csv_data))
    mock_session.return_value = session

    df = load("smn", station="ber", frequency="d")
    assert isinstance(df, pl.DataFrame)
    assert session.get.call_count == 1
