"""Tests for the public Python API."""

import io
import zipfile
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from conftest import CLIMATE_SCENARIOS_CSV

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
from foehn.fetch import default_fetcher
from foehn.workspace import Workspace


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


@pytest.mark.parametrize("dataset", ["smn", "forecast_icon_ch1", "surface_derived_grid", "climate_scenarios_indoor"])
@patch("foehn.registry.download")
def test_download_delegates_to_the_registry(mock_dl, dataset, tmp_path):
    """Whichever kind it is, download() hands it to the registry and returns the result."""
    from foehn.client import DownloadResult

    mock_dl.return_value = DownloadResult(total_assets=1, downloaded=1)

    res = download(dataset, data_dir=tmp_path)

    assert mock_dl.call_args.args == (dataset, Workspace(tmp_path))
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


@patch("foehn.registry.download")
def test_download_passes_force_through(mock_dl, tmp_path):
    from foehn.client import DownloadResult

    mock_dl.return_value = DownloadResult()
    download("climate_scenarios_indoor", data_dir=tmp_path, force=True)
    assert mock_dl.call_args.kwargs["force"] is True


@patch("foehn.registry.convert")
def test_to_parquet_raises_when_the_registry_reports_failures(mock_conv, tmp_path):
    mock_conv.return_value = 1
    with pytest.raises(RuntimeError, match="did not convert"):
        to_parquet("climate_scenarios_indoor", data_dir=tmp_path)


def test_load_indoor_frequency_raises():
    with pytest.raises(ValueError, match="does not support frequency"):
        load("climate_scenarios_indoor", frequency="h")


def test_load_indoor_returns_dataframe(fetcher):
    fetcher.any_items = [{"assets": {"d": {"href": "https://data.geo.admin.ch/x/raumklima.csv.zip"}}}]
    zip_bytes = _make_indoor_zip(["ABO_2035_RCP85_DRY.csv", "AIG_2060_RCP26_DRY.csv"])
    fetcher.default_body = zip_bytes

    df = load("climate_scenarios_indoor")
    assert isinstance(df, pl.DataFrame)
    assert {"station_abbr", "period", "scenario", "variant", "reference_timestamp"} <= set(df.columns)
    assert not any(c.startswith("time.") for c in df.columns)
    assert set(df["station_abbr"].unique()) == {"ABO", "AIG"}
    # 2 rows per data file × 2 files; the metadata CSV is skipped
    assert len(df) == 4


def test_load_indoor_station_filter(fetcher):
    fetcher.any_items = [{"assets": {"d": {"href": "https://data.geo.admin.ch/x/raumklima.csv.zip"}}}]
    zip_bytes = _make_indoor_zip(["ABO_2035_RCP85_DRY.csv", "AIG_2060_RCP26_DRY.csv"])
    fetcher.default_body = zip_bytes

    df = load("climate_scenarios_indoor", station="ABO")
    assert set(df["station_abbr"].unique()) == {"ABO"}
    assert len(df) == 2


# --- climate_scenarios (C8: metadata preamble + wide model table) ---


def test_load_climate_scenarios_year_filter_raises():
    with pytest.raises(ValueError, match="nominal"):
        load("climate_scenarios", year=2025)


def test_load_climate_scenarios_returns_dataframe(fetcher):
    href = "https://data.geo.admin.ch/x/ogd-climate-scenarios-ch2025_abe_pr_gwl1.5.csv"
    fetcher.any_items = [{"id": "abe", "assets": {"d": {"href": href}}}]
    fetcher.default_body = CLIMATE_SCENARIOS_CSV

    df = load("climate_scenarios")
    assert isinstance(df, pl.DataFrame)
    assert df.columns[:4] == ["station_abbr", "variable", "gwl", "date"]
    assert df["station_abbr"][0] == "abe"
    assert "MODEL_A" in df.columns
    assert df["date"].to_list() == ["0001-01-01", "0001-01-02"]


# --- forecast_local Date normalization ---

_FORECAST_LOCAL_CSV = "point_id;point_type_id;Date;dkl010h0\n1;1;202605202100;282\n1;1;202605202200;315\n"


def test_load_forecast_local_adds_reference_timestamp(fetcher):
    fetcher.any_collection = {"assets": {}}
    href = "https://data.geo.admin.ch/x/vnut12.lssw.202605210000.dkl010h0.csv"
    fetcher.any_items = [{"id": "x", "properties": {"datetime": "2026-05-21"}, "assets": {"d": {"href": href}}}]
    fetcher.default_body = _FORECAST_LOCAL_CSV

    df = load("forecast_local")
    assert "reference_timestamp" in df.columns
    assert df["reference_timestamp"].dtype == pl.Datetime
    assert df["reference_timestamp"][0].year == 2026


def test_load_forecast_local_date_filter_applies(fetcher):
    """forecast_local rows are filtered by date even though its timestamp is derived.

    Its reference_timestamp is synthesised from the compact Date column, which
    used to happen only after the concat — so the per-frame filter pass has to
    derive it first or it would silently skip filtering this dataset.
    """
    fetcher.any_collection = {"assets": {}}
    href = "https://data.geo.admin.ch/x/vnut12.lssw.202605210000.dkl010h0.csv"
    fetcher.any_items = [{"id": "x", "properties": {"datetime": "2026-05-21"}, "assets": {"d": {"href": href}}}]
    fetcher.default_body = _FORECAST_LOCAL_CSV

    both = load("forecast_local")
    assert len(both) == 2

    # The fixture's two rows are 2026-05-20 21:00 and 22:00 — an exact bound splits them.
    narrowed = load("forecast_local", date_to="2026-05-20 21:00:00")
    assert len(narrowed) == 1
    assert narrowed["reference_timestamp"][0].hour == 21


def test_load_time_filters_match_across_multiple_stations(fetcher):
    """Filtering per-frame must give the same rows as filtering after the concat."""
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = [
        {"id": s, "assets": {"data": {"href": f"https://data.geo.admin.ch/smn/ogd-smn_{s.lower()}_d_recent.csv"}}}
        for s in ("BER", "ZUR", "GVE")
    ]
    fetcher.default_body = _FILTER_CSV

    everything = load("smn", frequency="d")
    narrowed = load("smn", frequency="d", year=2025, month=[6, 7])

    expected = everything.filter(
        pl.col("reference_timestamp").dt.year().eq(2025) & pl.col("reference_timestamp").dt.month().is_in([6, 7])
    )
    assert narrowed.sort("temp").to_dicts() == expected.sort("temp").to_dicts()
    assert len(narrowed) == 6  # 3 stations x 2 matching rows


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


def test_load_forecast_local_skips_empty_newest_item(fetcher):
    """The newest day is created empty at ~04:00 UTC and filled as runs publish.

    Selecting it as "latest" returned zero CSVs — the cause of issue #27.
    """
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = [
        _forecast_item("20260720-ch", {"202607200600": ["dkl010h0"]}),
        _forecast_item("20260721-ch", {"202607210600": ["dkl010h0"]}),
        _forecast_item("20260722-ch", {}),  # newest day, not yet populated
    ]
    fetcher.default_body = _FORECAST_LOCAL_CSV

    df = load("forecast_local")
    assert len(df) > 0
    # Newest *run* across the populated days, not the newest item.
    assert "202607210600" in fetcher.gets[-1]


def test_load_forecast_local_fetches_only_latest_run(fetcher):
    """One run is ~32 files at ~30 MB; the retained window is ~40 runs (~40 GB)."""
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = [
        _forecast_item(
            "20260721-ch",
            {
                "202607210400": ["dkl010h0", "fu3010h0"],
                "202607210500": ["dkl010h0", "fu3010h0"],
                "202607210600": ["dkl010h0", "fu3010h0"],
            },
        )
    ]
    fetcher.default_body = _FORECAST_LOCAL_CSV

    load("forecast_local")
    fetched = fetcher.gets
    assert len(fetched) == 2
    assert all("202607210600" in url for url in fetched)


@patch("foehn.registry.download")
def test_download_passes_the_requested_time_slice(mock_dl, tmp_path):
    from foehn.client import DownloadResult

    mock_dl.return_value = DownloadResult()
    download("smn", data_dir=tmp_path, time_slice=["historical"])

    assert mock_dl.call_args.kwargs["time_slice"] == ["historical"]


def test_to_parquet_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        to_parquet("nonexistent")


@patch("foehn.registry.convert")
def test_to_parquet_delegates_to_the_registry(mock_conv, tmp_path):
    mock_conv.return_value = 0
    to_parquet("smn", data_dir=tmp_path)
    mock_conv.assert_called_once_with("smn", Workspace(tmp_path))


@patch("foehn.registry.convert")
def test_to_parquet_raises_on_convert_failure(mock_conv, tmp_path):
    """Python API should mirror the CLI: any conversion failure raises, not silent."""
    mock_conv.return_value = 2
    with pytest.raises(RuntimeError, match="2 group"):
        to_parquet("smn", data_dir=tmp_path)


@patch("foehn.registry.convert")
def test_to_parquet_silent_when_no_failures(mock_conv, tmp_path):
    mock_conv.return_value = 0
    to_parquet("smn", data_dir=tmp_path)  # must not raise


# --- read() tests ---


def test_load_is_exported():
    assert callable(foehn.load)


def test_load_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load("nonexistent")


def test_load_dataset_without_a_reader_says_what_to_do_instead():
    """climate_normals is neither loadable nor gridded — the old message said "binary/grid"."""
    with pytest.raises(ValueError, match="no in-memory reader"):
        foehn.load("climate_normals")


def test_load_grib2_dataset_raises():
    with pytest.raises(ValueError, match="gridded"):
        load("forecast_icon_ch1")


def test_load_netcdf_dataset_raises():
    with pytest.raises(ValueError, match="gridded"):
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


def test_load_returns_dataframe(fetcher):
    """load() should download CSVs in memory and return a concatenated DataFrame."""
    # No metadata assets
    fetcher.any_collection = {"assets": {}}

    # Two STAC items, each with one CSV asset
    fetcher.any_items = [
        {"assets": {"data": {"href": "https://data.geo.admin.ch/smn/file1_recent.csv"}}},
        {"assets": {"data": {"href": "https://data.geo.admin.ch/smn/file2_recent.csv"}}},
    ]

    fetcher.add_body("https://data.geo.admin.ch/smn/file1_recent.csv", "station;temperature\nBER;20.5\nZUR;18.3\n")
    fetcher.add_body("https://data.geo.admin.ch/smn/file2_recent.csv", "station;temperature\nGEN;22.1\n")

    df = load("smn")

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 3
    assert "station" in df.columns
    assert "temperature" in df.columns


def test_load_with_metadata_types(fetcher):
    """load() should use metadata to infer column types."""
    meta_csv = "parameter_shortname;parameter_datatype\nvalue;float\n"
    fetcher.any_collection = {
        "assets": {"params": {"href": "https://data.geo.admin.ch/smn/ogd-smn_meta_parameters.csv"}}
    }

    fetcher.any_items = [
        {"assets": {"data": {"href": "https://data.geo.admin.ch/smn/file_recent.csv"}}},
    ]

    fetcher.add_body("https://data.geo.admin.ch/smn/ogd-smn_meta_parameters.csv", meta_csv)
    fetcher.add_body("https://data.geo.admin.ch/smn/file_recent.csv", "station;value\nBER;20.5\n")

    df = load("smn")

    assert isinstance(df, pl.DataFrame)
    assert df.schema["value"] == pl.Float64


def test_load_filters_time_slice(fetcher):
    """load() should only include CSVs matching the requested time_slice."""
    fetcher.any_collection = {"assets": {}}

    fetcher.any_items = [
        {
            "assets": {
                "recent": {"href": "https://data.geo.admin.ch/smn/file_recent.csv"},
                "historical": {"href": "https://data.geo.admin.ch/smn/file_historical.csv"},
            }
        },
    ]

    csv_data = "station;temp\nBER;20\n"

    fetcher.default_body = csv_data

    df = load("smn", time_slice=["recent"])

    assert isinstance(df, pl.DataFrame)
    # Only one CSV should have been fetched (the recent one)
    assert len(fetcher.gets) == 1


def test_load_no_csvs_raises(fetcher):
    """load() should raise ValueError when no CSVs match."""
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = [{"assets": {"data": {"href": "https://data.geo.admin.ch/smn/file_historical.csv"}}}]

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


def test_load_station_filter_single(fetcher):
    """load(station='ber') should only download files for that station."""
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = _smn_items("ber", "zur", "gen")

    csv_data = "station;temp\nBER;20\n"
    fetcher.default_body = csv_data

    df = load("smn", station="BER")

    assert isinstance(df, pl.DataFrame)
    # 3 granularities (d, h, t) for 1 station
    assert len(fetcher.gets) == 3


def test_load_station_filter_multiple(fetcher):
    """load(station=['ber', 'zur']) should download files for both stations."""
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = _smn_items("ber", "zur", "gen")

    csv_data = "station;temp\nX;20\n"
    fetcher.default_body = csv_data

    df = load("smn", station=["BER", "ZUR"])

    assert isinstance(df, pl.DataFrame)
    # 3 granularities × 2 stations = 6
    assert len(fetcher.gets) == 6


def test_load_frequency_filter(fetcher):
    """load(frequency='d') should only download daily files."""
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = _smn_items("ber")

    csv_data = "station;temp\nBER;20\n"
    fetcher.default_body = csv_data

    df = load("smn", station="BER", frequency="d")

    assert isinstance(df, pl.DataFrame)
    # Only 1 file: ber_d_recent
    assert len(fetcher.gets) == 1


def test_load_frequency_filter_multiple(fetcher):
    """load(frequency=['d', 'h']) should download daily + hourly."""
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = _smn_items("ber")

    csv_data = "station;temp\nBER;20\n"
    fetcher.default_body = csv_data

    df = load("smn", station="BER", frequency=["d", "h"])

    assert isinstance(df, pl.DataFrame)
    # 2 files: ber_d_recent + ber_h_recent
    assert len(fetcher.gets) == 2


def test_load_station_case_insensitive(fetcher):
    """Station filter should be case-insensitive."""
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = _smn_items("ber")

    csv_data = "station;temp\nBER;20\n"
    fetcher.default_body = csv_data

    df = load("smn", station="ber", frequency="d")
    assert isinstance(df, pl.DataFrame)
    assert len(fetcher.gets) == 1


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


def test_parameters_returns_dataframe(fetcher):
    fetcher.any_collection = _stac_assets("params", "https://data.geo.admin.ch/smn/ogd-smn_meta_parameters.csv")
    fetcher.default_body = _PARAMS_CSV.encode("windows-1252")

    df = parameters("smn")

    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["shortname", "description", "unit", "type", "granularity", "decimals", "group"]
    assert df["shortname"][0] == "tre200d0"
    assert df["description"][0] == "Air temperature 2m daily mean"
    assert df["unit"][0] == "°C"


def test_stations_returns_dataframe(fetcher):
    fetcher.any_collection = _stac_assets("stations", "https://data.geo.admin.ch/smn/ogd-smn_meta_stations.csv")
    fetcher.default_body = _STATIONS_CSV

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
def test_declared_time_slices_match_what_meteoswiss_publishes():
    """COLLECTION_META's time_slices must match the real listing.

    These are surfaced by list_datasets(), the CLI and the MCP tools, so a
    missing slice tells callers a slice does not exist when it does — pollen's
    hourly "now" series was invisible this way.
    """
    from foehn import registry
    from foehn._urls import asset_filename
    from foehn.collections import (
        COLLECTION_META,
        COLLECTIONS,
        DatasetKind,
        kind,
        time_slice_from_filename,
    )

    # The archive kind ships one ZIP rather than per-slice CSV assets, so it has
    # no time slices to compare.
    tabular = [k for k in registry.tabular_datasets() if kind(k) is not DatasetKind.ARCHIVE_CSV]
    mismatches = {}
    for key in tabular:
        found = set()
        for item in default_fetcher().items(COLLECTIONS[key]):
            for asset in item.get("assets", {}).values():
                name = asset_filename(asset.get("href", ""))
                if name.endswith(".csv") and "_meta_" not in name:
                    found.add(time_slice_from_filename(name))
        actual = sorted(s for s in found if s)
        declared = sorted(COLLECTION_META[key]["time_slices"])
        if actual != declared:
            mismatches[key] = {"declared": declared, "actual": actual}

    assert not mismatches, f"COLLECTION_META time_slices out of step with MeteoSwiss: {mismatches}"


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


def test_inventory_returns_dataframe(fetcher):
    fetcher.any_collection = _stac_assets("inv", "https://data.geo.admin.ch/smn/ogd-smn_meta_datainventory.csv")
    fetcher.default_body = _INVENTORY_CSV

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


def test_metadata_no_matching_asset_raises(fetcher):
    fetcher.any_collection = {"assets": {}}

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


def _setup_filter_fetcher(fetcher):
    """Wire the fetcher every filter test shares."""
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = [
        {"id": "BER", "assets": {"data": {"href": "https://data.geo.admin.ch/smn/ogd-smn_ber_d_recent.csv"}}},
    ]
    fetcher.default_body = _FILTER_CSV


def test_load_year_filter(fetcher):
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", year=2025)
    assert len(df) == 4
    assert all(ts.year == 2025 for ts in df["reference_timestamp"].to_list())


def test_load_year_filter_list(fetcher):
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", year=[2024, 2025])
    assert len(df) == 5


def test_load_month_filter(fetcher):
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", month=[6, 7])
    assert len(df) == 3
    assert all(ts.month in (6, 7) for ts in df["reference_timestamp"].to_list())


def test_load_year_and_month_combined(fetcher):
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", year=2025, month=7)
    assert len(df) == 1
    assert df["temp"][0] == 25.0


def test_load_date_from_filter(fetcher):
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", date_from="2025-06-01")
    assert len(df) == 3


def test_load_date_to_filter(fetcher):
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", date_to="2025-01-15")
    assert len(df) == 2


def test_load_date_range_filter(fetcher):
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", date_from="2025-06-01", date_to="2025-08-31")
    assert len(df) == 2


def test_date_to_bare_date_includes_the_whole_final_day():
    """A bare "YYYY-MM-DD" date_to names the whole day, not its midnight.

    Comparing <= the parsed midnight is right for d/m/y (timestamps sit at
    00:00) but silently drops every 10-minute and hourly reading after 00:00,
    while the docstring promises an inclusive bound.
    """
    from foehn.readers import Filters, apply_time_filters

    df = pl.DataFrame(
        {
            "station_abbr": ["BER"] * 4,
            "reference_timestamp": [
                "2025-08-30 12:00",
                "2025-08-31 00:00",
                "2025-08-31 12:00",
                "2025-08-31 23:50",
            ],
        }
    ).with_columns(pl.col("reference_timestamp").str.to_datetime())

    out = apply_time_filters(df, Filters.build(date_from="2025-08-30", date_to="2025-08-31"))
    assert len(out) == 4

    # The day after is still excluded — the bound is the next midnight, exclusive.
    assert len(apply_time_filters(df, Filters.build(date_to="2025-08-30"))) == 1


def test_date_to_with_explicit_time_stays_an_exact_bound():
    """An explicit timestamp means exactly that instant, not the end of its day."""
    from foehn.readers import Filters, apply_time_filters

    df = pl.DataFrame(
        {
            "station_abbr": ["BER"] * 3,
            "reference_timestamp": ["2025-08-31 11:00", "2025-08-31 12:00", "2025-08-31 13:00"],
        }
    ).with_columns(pl.col("reference_timestamp").str.to_datetime())

    out = apply_time_filters(df, Filters.build(date_to="2025-08-31 12:00:00"))
    assert len(out) == 2


def test_date_filter_on_date_typed_column_does_not_raise():
    """date_from/date_to must work when reference_timestamp parsed as Date, not Datetime."""
    from datetime import date

    from foehn.readers import Filters, apply_time_filters

    df = pl.DataFrame(
        {
            "station_abbr": ["BER", "BER", "BER"],
            "reference_timestamp": [date(2025, 1, 1), date(2025, 6, 1), date(2025, 12, 1)],
        }
    )
    assert df["reference_timestamp"].dtype == pl.Date  # guard: column really is Date-typed

    out = apply_time_filters(df, Filters.build(date_from="2025-03-01", date_to="2025-08-31"))
    assert len(out) == 1
    assert out["reference_timestamp"][0] == date(2025, 6, 1)


def test_load_drop_null_filter(fetcher):
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", drop_null="precip")
    assert len(df) == 4
    assert df["precip"].null_count() == 0


def test_load_drop_null_nonexistent_column_raises(fetcher):
    # Silently ignoring the filter would return every null row it was asked to
    # drop — a plausible-looking wrong answer for a mistyped shortcode.
    _setup_filter_fetcher(fetcher)
    with pytest.raises(ValueError, match=r"Unknown column\(s\) \['nonexistent'\] in drop_null="):
        load("smn", station="BER", frequency="d", drop_null="nonexistent")


def test_load_sort_desc(fetcher):
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", sort="desc")
    timestamps = df["reference_timestamp"].to_list()
    assert timestamps == sorted(timestamps, reverse=True)


def test_load_sort_asc(fetcher):
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", sort="asc")
    timestamps = df["reference_timestamp"].to_list()
    assert timestamps == sorted(timestamps)


def test_load_columns_filter(fetcher):
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", columns=["temp"])
    assert set(df.columns) == {"station_abbr", "reference_timestamp", "temp"}


def test_load_columns_nonexistent_raises(fetcher):
    # Dropping the unknown name silently would hand back only the always-kept
    # key columns, with no signal that the requested parameter was a typo.
    _setup_filter_fetcher(fetcher)
    with pytest.raises(ValueError, match=r"Unknown column\(s\) \['nonexistent'\] in columns="):
        load("smn", station="BER", frequency="d", columns=["temp", "nonexistent"])


def test_load_all_filters_combined(fetcher):
    _setup_filter_fetcher(fetcher)
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


# --- column projection (parsed columns pushed into read_csv) ---


def test_load_columns_projection_matches_unprojected(fetcher):
    """Selecting columns up front must give exactly what selecting afterwards did."""
    _setup_filter_fetcher(fetcher)
    projected = load("smn", station="BER", frequency="d", columns=["temp"])

    _setup_filter_fetcher(fetcher)
    everything = load("smn", station="BER", frequency="d")

    assert projected.columns == ["station_abbr", "reference_timestamp", "temp"]
    assert projected.to_dicts() == everything.select(projected.columns).to_dicts()


def test_load_columns_projection_keeps_drop_null_column(fetcher):
    """drop_null must still work on a column the caller didn't ask to return.

    The projection has to keep it, or the filter silently loses its subject.
    """
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", columns=["temp"], drop_null="precip")

    assert df.columns == ["station_abbr", "reference_timestamp", "temp"]
    assert len(df) == 4  # the one null-precip row is gone


def test_load_columns_projection_handles_station_missing_the_column(fetcher):
    """A station whose file lacks the column still contributes rows, padded with null.

    Projecting per file must not turn a heterogeneous schema into a hard error —
    the diagonal concat filled these with nulls before and must keep doing so.
    """
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = [
        {"id": s, "assets": {"d": {"href": f"https://data.geo.admin.ch/smn/ogd-smn_{s.lower()}_d_recent.csv"}}}
        for s in ("BER", "ZUR")
    ]
    fetcher.add_body(
        "https://data.geo.admin.ch/smn/ogd-smn_ber_d_recent.csv",
        b"station_abbr;reference_timestamp;temp\nBER;2025-06-15T00:00:00;20.0\n",
    )
    fetcher.add_body(
        "https://data.geo.admin.ch/smn/ogd-smn_zur_d_recent.csv",
        b"station_abbr;reference_timestamp;precip\nZUR;2025-06-16T00:00:00;3.0\n",
    )

    df = load("smn", frequency="d", columns=["temp"], sort="asc")

    assert df.columns == ["station_abbr", "reference_timestamp", "temp"]
    assert df.to_dicts()[1]["station_abbr"] == "ZUR"
    assert df.to_dicts()[1]["temp"] is None


def test_load_columns_projection_keeps_forecast_local_date(fetcher):
    """forecast_local derives its timestamp from Date, so the projection must keep it."""
    fetcher.any_collection = {"assets": {}}
    href = "https://data.geo.admin.ch/x/vnut12.lssw.202605210000.dkl010h0.csv"
    fetcher.any_items = [{"id": "x", "properties": {"datetime": "2026-05-21"}, "assets": {"d": {"href": href}}}]
    fetcher.default_body = _FORECAST_LOCAL_CSV

    df = load("forecast_local", columns=["dkl010h0"])

    assert "reference_timestamp" in df.columns
    assert df["reference_timestamp"].null_count() == 0
    assert df["dkl010h0"].to_list() == [282, 315]


# --- limit + concurrent fetching ---


def test_load_limit_caps_rows(fetcher):
    """limit param should cap the returned DataFrame to N rows."""
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", limit=2)
    assert len(df) == 2


def test_load_limit_applied_after_sort(fetcher):
    """limit + sort='desc' should give the N newest rows."""
    _setup_filter_fetcher(fetcher)
    df = load("smn", station="BER", frequency="d", sort="desc", limit=1)
    assert len(df) == 1
    # newest row in _FILTER_CSV is 2025-12-01
    assert df["reference_timestamp"][0].year == 2025
    assert df["reference_timestamp"][0].month == 12


def test_load_workers_one_uses_serial_path(fetcher):
    """workers=1 should still produce a correct DataFrame (covers the serial branch)."""
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = _smn_items("ber")

    csv_data = "station;temp\nBER;20\n"
    fetcher.default_body = csv_data

    df = load("smn", station="BER", frequency="d", workers=1)
    assert isinstance(df, pl.DataFrame)


def test_load_concurrent_fetch_multiple_files(fetcher):
    """With multiple CSVs and workers>1, all are fetched and concatenated."""
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = _smn_items("ber", "zur", "gen")

    csv_data = "station;temp\nX;20\n"
    fetcher.default_body = csv_data

    df = load("smn", frequency="d", workers=4)
    # 3 stations × 1 frequency = 3 fetches
    assert len(fetcher.gets) == 3
    assert len(df) == 3
