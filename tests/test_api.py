"""Tests for the public Python API."""

import io
import zipfile
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

import foehn
from foehn.api import (
    download,
    inventory,
    list_datasets,
    load,
    parameters,
    stations,
    to_parquet,
)
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


@patch("foehn.api.download_grib2")
def test_download_grib2_dataset_routes_to_handler(mock_dl, tmp_path):
    from foehn.client import DownloadResult

    mock_dl.return_value = DownloadResult(total_assets=1, downloaded=1)
    res = download("forecast_icon_ch1", data_dir=tmp_path)
    mock_dl.assert_called_once_with("forecast_icon_ch1", tmp_path / "bronze", since=None, workers=8)
    assert res.downloaded == 1


@patch("foehn.api.download_netcdf")
def test_download_netcdf_dataset_routes_to_handler(mock_dl, tmp_path):
    from foehn.client import DownloadResult

    mock_dl.return_value = DownloadResult(total_assets=1, downloaded=1)
    res = download("surface_derived_grid", data_dir=tmp_path)
    mock_dl.assert_called_once_with("surface_derived_grid", tmp_path / "bronze", since=None, workers=8)
    assert res.downloaded == 1


_INDOOR_CSV = "time.yy,time.mm,time.dd,time.hh,tre200h0,ure200h0\n2035,1,1,0,0.3,94.5\n2035,1,1,1,-0.2,94.6\n"


def _make_indoor_zip(data_names):
    """Build an in-memory indoor .csv.zip with the given data CSVs + a metadata file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n in data_names:
            zf.writestr(n, _INDOOR_CSV)
        zf.writestr("Klimaszenarien-Raumklima_Metadata.csv", "a,b\n1,2\n")
    return buf.getvalue()


@patch("foehn.api.download_climate_scenarios_indoor")
def test_download_indoor_routes_to_handler(mock_dl, tmp_path):
    from foehn.client import DownloadResult

    mock_dl.return_value = DownloadResult(total_assets=1, downloaded=1)
    res = download("climate_scenarios_indoor", data_dir=tmp_path)
    mock_dl.assert_called_once_with(tmp_path / "bronze", "climate_scenarios_indoor", force=False)
    assert res.downloaded == 1


@patch("foehn.api.download_climate_scenarios_indoor")
def test_download_indoor_passes_force(mock_dl, tmp_path):
    from foehn.client import DownloadResult

    mock_dl.return_value = DownloadResult()
    download("climate_scenarios_indoor", data_dir=tmp_path, force=True)
    assert mock_dl.call_args.kwargs["force"] is True


@patch("foehn.api.convert_climate_scenarios_indoor_to_parquet")
def test_to_parquet_indoor_routes_to_converter(mock_conv, tmp_path):
    mock_conv.return_value = 0
    to_parquet("climate_scenarios_indoor", data_dir=tmp_path)
    mock_conv.assert_called_once_with(tmp_path / "bronze", tmp_path / "parquet")


@patch("foehn.api.convert_climate_scenarios_indoor_to_parquet")
def test_to_parquet_indoor_raises_on_failure(mock_conv, tmp_path):
    mock_conv.return_value = 1
    with pytest.raises(RuntimeError, match="did not convert"):
        to_parquet("climate_scenarios_indoor", data_dir=tmp_path)


def test_load_indoor_frequency_raises():
    with pytest.raises(ValueError, match="does not support frequency"):
        load("climate_scenarios_indoor", frequency="h")


@patch("foehn.api.get_collection_items")
@patch("foehn.api._retry_session")
def test_load_indoor_returns_dataframe(mock_session, mock_items):
    mock_items.return_value = [{"assets": {"d": {"href": "https://data.geo.admin.ch/x/raumklima.csv.zip"}}}]
    zip_bytes = _make_indoor_zip(["ABO_2035_RCP85_DRY.csv", "AIG_2060_RCP26_DRY.csv"])
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(zip_bytes))
    session.__enter__.return_value = session
    mock_session.return_value = session

    df = load("climate_scenarios_indoor")
    assert isinstance(df, pl.DataFrame)
    assert {"station_abbr", "period", "scenario", "variant", "reference_timestamp"} <= set(df.columns)
    assert not any(c.startswith("time.") for c in df.columns)
    assert set(df["station_abbr"].unique()) == {"ABO", "AIG"}
    # 2 rows per data file × 2 files; the metadata CSV is skipped
    assert len(df) == 4


@patch("foehn.api.get_collection_items")
@patch("foehn.api._retry_session")
def test_load_indoor_station_filter(mock_session, mock_items):
    mock_items.return_value = [{"assets": {"d": {"href": "https://data.geo.admin.ch/x/raumklima.csv.zip"}}}]
    zip_bytes = _make_indoor_zip(["ABO_2035_RCP85_DRY.csv", "AIG_2060_RCP26_DRY.csv"])
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(zip_bytes))
    session.__enter__.return_value = session
    mock_session.return_value = session

    df = load("climate_scenarios_indoor", station="ABO")
    assert set(df["station_abbr"].unique()) == {"ABO"}
    assert len(df) == 2


# --- climate_scenarios (C8: metadata preamble + wide model table) ---

_CS_CSV = (
    "TITLE;X\nVARIABLE;Daily precipitation sum\nGWL;GWL1.5\n\n"
    "DATE;MODEL_A;MODEL_B\n0001-01-01;0;24.2\n0001-01-02;1.5;0\n"
)


@patch("foehn.api.convert_climate_scenarios_to_parquet")
def test_to_parquet_climate_scenarios_routes(mock_conv, tmp_path):
    mock_conv.return_value = 0
    to_parquet("climate_scenarios", data_dir=tmp_path)
    mock_conv.assert_called_once_with(tmp_path / "bronze", tmp_path / "parquet")


def test_load_climate_scenarios_year_filter_raises():
    with pytest.raises(ValueError, match="nominal"):
        load("climate_scenarios", year=2025)


@patch("foehn.api.get_collection_items")
@patch("foehn.api._retry_session")
def test_load_climate_scenarios_returns_dataframe(mock_session, mock_items):
    href = "https://data.geo.admin.ch/x/ogd-climate-scenarios-ch2025_abe_pr_gwl1.5.csv"
    mock_items.return_value = [{"id": "abe", "assets": {"d": {"href": href}}}]
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(_CS_CSV))
    session.__enter__.return_value = session
    mock_session.return_value = session

    df = load("climate_scenarios")
    assert isinstance(df, pl.DataFrame)
    assert df.columns[:4] == ["station_abbr", "variable", "gwl", "date"]
    assert df["station_abbr"][0] == "abe"
    assert "MODEL_A" in df.columns
    assert df["date"].to_list() == ["0001-01-01", "0001-01-02"]


# --- forecast_local Date normalization ---

_FORECAST_LOCAL_CSV = "point_id;point_type_id;Date;dkl010h0\n1;1;202605202100;282\n1;1;202605202200;315\n"


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_forecast_local_adds_reference_timestamp(mock_session, mock_meta, mock_items):
    mock_meta.return_value = {"assets": {}}
    href = "https://data.geo.admin.ch/x/vnut12.lssw.202605210000.dkl010h0.csv"
    mock_items.return_value = [{"id": "x", "properties": {"datetime": "2026-05-21"}, "assets": {"d": {"href": href}}}]
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(_FORECAST_LOCAL_CSV))
    session.__enter__.return_value = session
    mock_session.return_value = session

    df = load("forecast_local")
    assert "reference_timestamp" in df.columns
    assert df["reference_timestamp"].dtype == pl.Datetime
    assert df["reference_timestamp"][0].year == 2026


def _forecast_item(item_id: str, runs: dict[str, list[str]]) -> dict:
    """Build a forecast STAC item: one item = one day, holding several hourly runs."""
    return {
        "id": item_id,
        # All items share a near-identical refresh timestamp upstream — it is not
        # the forecast date, so item selection must not rank on it.
        "properties": {"datetime": "2026-07-21T04:00:16.387170Z"},
        "assets": {
            f"{run}.{param}": {"href": f"https://data.geo.admin.ch/x/vnut12.lssw.{run}.{param}.csv"}
            for run, params in runs.items()
            for param in params
        },
    }


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_forecast_local_skips_empty_newest_item(mock_session, mock_meta, mock_items):
    """The newest day is created empty at ~04:00 UTC and filled as runs publish.

    Selecting it as "latest" returned zero CSVs — the cause of issue #27.
    """
    mock_meta.return_value = {"assets": {}}
    mock_items.return_value = [
        _forecast_item("20260720-ch", {"202607200600": ["dkl010h0"]}),
        _forecast_item("20260721-ch", {"202607210600": ["dkl010h0"]}),
        _forecast_item("20260722-ch", {}),  # newest day, not yet populated
    ]
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(_FORECAST_LOCAL_CSV))
    session.__enter__.return_value = session
    mock_session.return_value = session

    df = load("forecast_local")
    assert len(df) > 0
    # Newest *run* across the populated days, not the newest item.
    assert "202607210600" in session.get.call_args[0][0]


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_forecast_local_fetches_only_latest_run(mock_session, mock_meta, mock_items):
    """One run is ~32 files at ~30 MB; the retained window is ~40 runs (~40 GB)."""
    mock_meta.return_value = {"assets": {}}
    mock_items.return_value = [
        _forecast_item(
            "20260721-ch",
            {
                "202607210400": ["dkl010h0", "fu3010h0"],
                "202607210500": ["dkl010h0", "fu3010h0"],
                "202607210600": ["dkl010h0", "fu3010h0"],
            },
        )
    ]
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(_FORECAST_LOCAL_CSV))
    session.__enter__.return_value = session
    mock_session.return_value = session

    load("forecast_local")
    fetched = [c[0][0] for c in session.get.call_args_list]
    assert len(fetched) == 2
    assert all("202607210600" in url for url in fetched)


@patch("foehn.api.download_collection")
@patch("foehn.api.download_metadata")
def test_download_calls_underlying_functions(mock_meta, mock_dl, tmp_path):
    from foehn.client import DownloadResult

    mock_meta.return_value = DownloadResult(total_assets=1, downloaded=1, skipped=0, filenames=["m.csv"])
    mock_dl.return_value = DownloadResult(total_assets=2, downloaded=2, skipped=0, filenames=["a.csv", "b.csv"])

    result = download("smn", data_dir=tmp_path, time_slice=["historical"])
    mock_meta.assert_called_once_with("smn", tmp_path / "bronze", workers=8)
    mock_dl.assert_called_once_with("smn", tmp_path / "bronze", data_types=["historical"], since=None, workers=8)

    assert isinstance(result, DownloadResult)
    assert result.total_assets == 3
    assert result.downloaded == 3
    assert result.filenames == ["m.csv", "a.csv", "b.csv"]


def test_to_parquet_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        to_parquet("nonexistent")


@patch("foehn.api.convert_to_parquet")
def test_to_parquet_calls_convert_to_parquet(mock_conv, tmp_path):
    mock_conv.return_value = 0
    to_parquet("smn", data_dir=tmp_path)
    mock_conv.assert_called_once_with("smn", tmp_path / "bronze", tmp_path / "parquet")


@patch("foehn.api.convert_to_parquet")
def test_to_parquet_raises_on_convert_failure(mock_conv, tmp_path):
    """Python API should mirror the CLI: any conversion failure raises, not silent."""
    mock_conv.return_value = 2
    with pytest.raises(RuntimeError, match="2 group"):
        to_parquet("smn", data_dir=tmp_path)


@patch("foehn.api.convert_to_parquet")
def test_to_parquet_silent_when_no_failures(mock_conv, tmp_path):
    mock_conv.return_value = 0
    to_parquet("smn", data_dir=tmp_path)  # must not raise


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
    session.__enter__.return_value = session
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
    session.__enter__.return_value = session
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
    session.__enter__.return_value = session
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
    session.__enter__.return_value = session
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
    session.__enter__.return_value = session
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
    session.__enter__.return_value = session
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
    session.__enter__.return_value = session
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
    session.__enter__.return_value = session
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
    session.__enter__.return_value = session
    mock_session.return_value = session

    df = load("smn", station="ber", frequency="d")
    assert isinstance(df, pl.DataFrame)
    assert session.get.call_count == 1


# --- metadata tests ---

_PARAMS_CSV = (
    "parameter_shortname;parameter_description_de;parameter_description_fr;"
    "parameter_description_it;parameter_description_en;parameter_group_de;"
    "parameter_group_fr;parameter_group_it;parameter_group_en;"
    "parameter_granularity;parameter_decimals;parameter_datatype;parameter_unit\n"
    "tre200d0;Temp de;Temp fr;Temp it;Air temperature 2m daily mean;"
    "Temperatur;Température;Temperatura;Temperature;D;1;Float;°C\n"
)

_STATIONS_CSV = (
    "station_abbr;station_name;station_canton;station_wigos_id;station_type_de;"
    "station_type_fr;station_type_it;station_type_en;station_dataowner;"
    "station_data_since;station_height_masl;station_height_barometer_masl;"
    "station_coordinates_lv95_east;station_coordinates_lv95_north;"
    "station_coordinates_wgs84_lat;station_coordinates_wgs84_lon;"
    "station_exposition_de;station_exposition_fr;station_exposition_it;"
    "station_exposition_en;station_url_de;station_url_fr;station_url_it;station_url_en\n"
    "BER;Bern;BE;0-20000-0-06631;Auto;Auto;Auto;Auto;MeteoSchweiz;"
    "01.01.1864;553.0;554.0;2601933.0;1199885.0;46.9508;7.4394;"
    "Ebene;Plaine;Pianura;plain;url_de;url_fr;url_it;url_en\n"
)

_INVENTORY_CSV = (
    "station_abbr;parameter_shortname;meas_cat_nr;data_since;data_till;owner\n"
    "BER;tre200d0;1;01.01.1864 00:00;;MeteoSchweiz\n"
)


def _stac_assets(suffix, href):
    return {"assets": {suffix: {"href": href}}}


@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_parameters_returns_dataframe(mock_session, mock_meta):
    mock_meta.return_value = _stac_assets("params", "https://data.geo.admin.ch/smn/ogd-smn_meta_parameters.csv")
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(_PARAMS_CSV.encode("windows-1252")))
    session.__enter__.return_value = session
    mock_session.return_value = session

    df = parameters("smn")

    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["shortname", "description", "unit", "type", "granularity", "decimals", "group"]
    assert df["shortname"][0] == "tre200d0"
    assert df["description"][0] == "Air temperature 2m daily mean"
    assert df["unit"][0] == "°C"


@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_stations_returns_dataframe(mock_session, mock_meta):
    mock_meta.return_value = _stac_assets("stations", "https://data.geo.admin.ch/smn/ogd-smn_meta_stations.csv")
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(_STATIONS_CSV))
    session.__enter__.return_value = session
    mock_session.return_value = session

    df = stations("smn")

    assert isinstance(df, pl.DataFrame)
    assert df.columns == [
        "abbr",
        "name",
        "canton",
        "altitude",
        "lv95_east",
        "lv95_north",
        "lat",
        "lon",
        "data_since",
    ]
    assert df["abbr"][0] == "BER"
    assert df["name"][0] == "Bern"
    assert df["canton"][0] == "BE"
    assert df["lv95_east"][0] == 2601933.0
    assert df["lv95_north"][0] == 1199885.0
    assert df["data_since"][0] == "01.01.1864"


@pytest.mark.live
def test_stations_live_returns_lv95_and_source_date_format():
    df = stations("smn")

    assert isinstance(df, pl.DataFrame)
    assert df.columns == [
        "abbr",
        "name",
        "canton",
        "altitude",
        "lv95_east",
        "lv95_north",
        "lat",
        "lon",
        "data_since",
    ]
    assert df.height > 0

    dates = df["data_since"].drop_nulls().to_list()
    assert dates
    assert all(len(date) == 10 and date[2] == "." and date[5] == "." for date in dates)

    ber = df.filter(pl.col("abbr") == "BER")
    assert ber.height == 1
    row = ber.to_dicts()[0]
    assert row["name"].startswith("Bern")
    assert row["canton"] == "BE"
    assert 2_400_000 < row["lv95_east"] < 2_900_000
    assert 1_000_000 < row["lv95_north"] < 1_400_000
    assert 45.0 < row["lat"] < 48.0
    assert 5.0 < row["lon"] < 11.0
    assert row["data_since"] == "01.01.1864"


@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_inventory_returns_dataframe(mock_session, mock_meta):
    mock_meta.return_value = _stac_assets("inv", "https://data.geo.admin.ch/smn/ogd-smn_meta_datainventory.csv")
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(_INVENTORY_CSV))
    session.__enter__.return_value = session
    mock_session.return_value = session

    df = inventory("smn")

    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["station", "parameter", "data_since", "data_till", "owner"]
    assert df["station"][0] == "BER"
    assert df["parameter"][0] == "tre200d0"


def test_parameters_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        parameters("nonexistent")


def test_stations_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        stations("nonexistent")


def test_inventory_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        inventory("nonexistent")


@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_metadata_no_matching_asset_raises(mock_session, mock_meta):
    mock_meta.return_value = {"assets": {}}
    mock_session.return_value = MagicMock()

    with pytest.raises(ValueError, match="No _meta_parameters metadata found"):
        parameters("smn")


def test_metadata_exported_from_foehn():
    assert callable(foehn.parameters)
    assert callable(foehn.stations)
    assert callable(foehn.inventory)


# --- post-load filter tests ---

# CSV with reference_timestamp for filter tests.
_FILTER_CSV = (
    "station_abbr;reference_timestamp;temp;precip\n"
    "BER;2024-07-01T00:00:00;22.0;5.0\n"
    "BER;2025-01-15T00:00:00;0.0;10.0\n"
    "BER;2025-06-15T00:00:00;20.0;\n"
    "BER;2025-07-15T00:00:00;25.0;8.0\n"
    "BER;2025-12-01T00:00:00;-1.0;3.0\n"
)


def _setup_filter_mocks(mock_session, mock_meta, mock_items):
    """Set up common mocks for filter tests."""
    mock_meta.return_value = {"assets": {}}
    mock_items.return_value = [
        {"id": "BER", "assets": {"data": {"href": "https://data.geo.admin.ch/smn/ogd-smn_ber_d_recent.csv"}}},
    ]
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(_FILTER_CSV))
    session.__enter__.return_value = session
    mock_session.return_value = session


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_year_filter(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", year=2025)
    assert len(df) == 4
    assert all(ts.year == 2025 for ts in df["reference_timestamp"].to_list())


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_year_filter_list(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", year=[2024, 2025])
    assert len(df) == 5


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_month_filter(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", month=[6, 7])
    assert len(df) == 3
    assert all(ts.month in (6, 7) for ts in df["reference_timestamp"].to_list())


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_year_and_month_combined(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", year=2025, month=7)
    assert len(df) == 1
    assert df["temp"][0] == 25.0


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_date_from_filter(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", date_from="2025-06-01")
    assert len(df) == 3


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_date_to_filter(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", date_to="2025-01-15")
    assert len(df) == 2


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_date_range_filter(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", date_from="2025-06-01", date_to="2025-08-31")
    assert len(df) == 2


def test_date_filter_on_date_typed_column_does_not_raise():
    """date_from/date_to must work when reference_timestamp parsed as Date, not Datetime."""
    from datetime import date

    from foehn.api import _apply_post_filters

    df = pl.DataFrame(
        {
            "station_abbr": ["BER", "BER", "BER"],
            "reference_timestamp": [date(2025, 1, 1), date(2025, 6, 1), date(2025, 12, 1)],
        }
    )
    assert df["reference_timestamp"].dtype == pl.Date  # guard: column really is Date-typed

    out = _apply_post_filters(df, date_from="2025-03-01", date_to="2025-08-31")
    assert len(out) == 1
    assert out["reference_timestamp"][0] == date(2025, 6, 1)


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_drop_null_filter(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", drop_null="precip")
    assert len(df) == 4
    assert df["precip"].null_count() == 0


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_drop_null_nonexistent_column_ignored(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", drop_null="nonexistent")
    assert len(df) == 5


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_sort_desc(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", sort="desc")
    timestamps = df["reference_timestamp"].to_list()
    assert timestamps == sorted(timestamps, reverse=True)


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_sort_asc(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", sort="asc")
    timestamps = df["reference_timestamp"].to_list()
    assert timestamps == sorted(timestamps)


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_columns_filter(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", columns=["temp"])
    assert set(df.columns) == {"station_abbr", "reference_timestamp", "temp"}


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_columns_filter_ignores_nonexistent(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", columns=["temp", "nonexistent"])
    assert set(df.columns) == {"station_abbr", "reference_timestamp", "temp"}


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_all_filters_combined(mock_session, mock_meta, mock_items):
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load(
        "smn",
        station="BER",
        frequency="d",
        year=2025,
        month=[6, 7],
        columns=["temp"],
        sort="desc",
    )
    assert len(df) == 2
    assert set(df.columns) == {"station_abbr", "reference_timestamp", "temp"}
    assert df["temp"][0] == 25.0  # July first (desc)


# --- limit + concurrent fetching ---


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_limit_caps_rows(mock_session, mock_meta, mock_items):
    """limit param should cap the returned DataFrame to N rows."""
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", limit=2)
    assert len(df) == 2


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_limit_applied_after_sort(mock_session, mock_meta, mock_items):
    """limit + sort='desc' should give the N newest rows."""
    _setup_filter_mocks(mock_session, mock_meta, mock_items)
    df = load("smn", station="BER", frequency="d", sort="desc", limit=1)
    assert len(df) == 1
    # newest row in _FILTER_CSV is 2025-12-01
    assert df["reference_timestamp"][0].year == 2025
    assert df["reference_timestamp"][0].month == 12


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_workers_one_uses_serial_path(mock_session, mock_meta, mock_items):
    """workers=1 should still produce a correct DataFrame (covers the serial branch)."""
    mock_meta.return_value = {"assets": {}}
    mock_items.return_value = _smn_items("ber")

    csv_data = "station;temp\nBER;20\n"
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(csv_data))
    session.__enter__.return_value = session
    mock_session.return_value = session

    df = load("smn", station="BER", frequency="d", workers=1)
    assert isinstance(df, pl.DataFrame)


@patch("foehn.api.get_collection_items")
@patch("foehn.api.get_collection_metadata")
@patch("foehn.api._retry_session")
def test_load_concurrent_fetch_multiple_files(mock_session, mock_meta, mock_items):
    """With multiple CSVs and workers>1, all are fetched and concatenated."""
    mock_meta.return_value = {"assets": {}}
    mock_items.return_value = _smn_items("ber", "zur", "gen")

    csv_data = "station;temp\nX;20\n"
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response(csv_data))
    session.__enter__.return_value = session
    mock_session.return_value = session

    df = load("smn", frequency="d", workers=4)
    # 3 stations × 1 frequency = 3 fetches
    assert session.get.call_count == 3
    assert len(df) == 3
