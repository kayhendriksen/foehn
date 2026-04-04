"""Tests for the MCP server tools, resource, and prompt."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import polars as pl
import pytest

pytest.importorskip("mcp", reason="mcp not installed")

from foehn.collections import COLLECTIONS, GRIB2_COLLECTIONS, NETCDF_COLLECTIONS
from foehn.mcp_server import (
    _LOADABLE_DATASETS,
    _VALID_CATEGORIES,
    _VALID_FREQUENCIES,
    _VALID_TIME_SLICES,
    Dataset,
    DataSummary,
    InventoryEntry,
    Parameter,
    Station,
    describe_data,
    get_inventory,
    get_parameters,
    get_stations,
    list_datasets,
    load_data,
    mcp,
    query_weather,
    usage_guide,
)

# ── Constants ────────────────────────────────────────────────────────────────


class TestConstants:
    def test_loadable_datasets_excludes_grib2(self):
        for ds in GRIB2_COLLECTIONS:
            assert ds not in _LOADABLE_DATASETS

    def test_loadable_datasets_excludes_netcdf(self):
        for ds in NETCDF_COLLECTIONS:
            assert ds not in _LOADABLE_DATASETS

    def test_loadable_datasets_is_sorted(self):
        assert sorted(_LOADABLE_DATASETS) == _LOADABLE_DATASETS

    def test_loadable_datasets_all_exist_in_collections(self):
        for ds in _LOADABLE_DATASETS:
            assert ds in COLLECTIONS

    def test_valid_frequencies(self):
        assert {"t", "h", "d", "m", "y"} == _VALID_FREQUENCIES

    def test_valid_time_slices(self):
        assert {"historical", "recent", "now"} == _VALID_TIME_SLICES

    def test_valid_categories(self):
        assert {"A", "C", "D", "E"} == _VALID_CATEGORIES


# ── Pydantic models ─────────────────────────────────────────────────────────


class TestModels:
    def test_dataset_model(self):
        d = Dataset(
            dataset="smn",
            collection_id="ch.meteoschweiz.ogd-smn",
            category="A",
            subcategory="A1",
            description="Automatic weather stations",
            format="CSV",
            frequencies=["t", "h", "d", "m"],
            time_slices=["historical", "recent", "now"],
        )
        assert d.dataset == "smn"
        assert d.category == "A"
        assert d.frequencies == ["t", "h", "d", "m"]

    def test_parameter_model(self):
        p = Parameter(
            shortname="tre200s0",
            description="Air temperature 2m above ground",
            unit="°C",
            type="Float",
            granularity="T",
            decimals=1,
            group="Temperature",
        )
        assert p.shortname == "tre200s0"
        assert p.unit == "°C"

    def test_station_model(self):
        s = Station(
            abbr="BER",
            name="Bern",
            canton="BE",
            altitude=553,
            lat=46.9508,
            lon=7.4394,
            data_since="1864-01-01",
        )
        assert s.abbr == "BER"
        assert s.canton == "BE"

    def test_inventory_entry_model(self):
        e = InventoryEntry(
            station="BER",
            parameter="tre200d0",
            data_since="1864-01-01",
            data_till="2024-12-31",
            owner="MeteoSchweiz",
        )
        assert e.station == "BER"
        assert e.owner == "MeteoSchweiz"


# ── list_datasets ────────────────────────────────────────────────────────────


_FAKE_DATASETS = [
    {
        "dataset": "smn",
        "collection_id": "ch.meteoschweiz.ogd-smn",
        "category": "A",
        "subcategory": "A1",
        "description": "Automatic weather stations",
        "format": "CSV",
        "frequencies": ["t", "h", "d", "m"],
        "time_slices": ["historical", "recent", "now"],
    },
    {
        "dataset": "nbcn",
        "collection_id": "ch.meteoschweiz.ogd-nbcn",
        "category": "C",
        "subcategory": "C1",
        "description": "Climate stations, homogeneous",
        "format": "CSV",
        "frequencies": ["d", "m"],
        "time_slices": ["historical", "recent"],
    },
    {
        "dataset": "forecast_icon_ch1",
        "collection_id": "ch.meteoschweiz.ogd-forecasting-icon-ch1",
        "category": "E",
        "subcategory": "E2",
        "description": "ICON-CH1-EPS 1km",
        "format": "GRIB2",
        "frequencies": [],
        "time_slices": [],
    },
]


class TestListDatasets:
    @patch("foehn.mcp_server.foehn.list_datasets", return_value=_FAKE_DATASETS)
    def test_returns_all_datasets(self, mock_ld):
        result = list_datasets()
        assert len(result) == 3
        assert all(isinstance(d, Dataset) for d in result)
        mock_ld.assert_called_once()

    @patch("foehn.mcp_server.foehn.list_datasets", return_value=_FAKE_DATASETS)
    def test_filter_by_category(self, mock_ld):
        result = list_datasets(category="A")
        assert len(result) == 1
        assert result[0].dataset == "smn"

    @patch("foehn.mcp_server.foehn.list_datasets", return_value=_FAKE_DATASETS)
    def test_filter_by_category_case_insensitive(self, mock_ld):
        result = list_datasets(category="a")
        assert len(result) == 1
        assert result[0].dataset == "smn"

    @patch("foehn.mcp_server.foehn.list_datasets", return_value=_FAKE_DATASETS)
    def test_filter_by_category_no_match(self, mock_ld):
        result = list_datasets(category="D")
        assert len(result) == 0

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Invalid category"):
            list_datasets(category="Z")

    @patch("foehn.mcp_server.foehn.list_datasets", return_value=_FAKE_DATASETS)
    def test_no_category_returns_all(self, mock_ld):
        result = list_datasets(category=None)
        assert len(result) == 3


# ── load_data ────────────────────────────────────────────────────────────────


def _make_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


class TestLoadData:
    @patch("foehn.mcp_server.foehn.load")
    def test_basic_load(self, mock_load):
        mock_load.return_value = _make_df([{"station": "BER", "temp": 20.5}, {"station": "ZUR", "temp": 18.3}])
        result = load_data("smn")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["station"] == "BER"
        mock_load.assert_called_once_with("smn")

    @patch("foehn.mcp_server.foehn.load")
    def test_with_all_kwargs(self, mock_load):
        mock_load.return_value = _make_df([{"station": "BER", "temp": 20.5}])
        result = load_data("smn", station=["BER"], frequency="d", time_slice="recent", limit=10)
        assert len(result) == 1
        mock_load.assert_called_once_with("smn", station=["BER"], frequency="d", time_slice="recent")

    @patch("foehn.mcp_server.foehn.load")
    def test_limit_caps_at_500(self, mock_load):
        rows = [{"station": f"S{i}", "temp": float(i)} for i in range(600)]
        mock_load.return_value = _make_df(rows)
        result = load_data("smn", limit=9999)
        assert len(result) == 500

    @patch("foehn.mcp_server.foehn.load")
    def test_limit_minimum_is_1(self, mock_load):
        rows = [{"station": "BER", "temp": 20.5}, {"station": "ZUR", "temp": 18.3}]
        mock_load.return_value = _make_df(rows)
        result = load_data("smn", limit=-5)
        assert len(result) == 1

    @patch("foehn.mcp_server.foehn.load")
    def test_default_limit_is_50(self, mock_load):
        rows = [{"station": f"S{i}", "temp": float(i)} for i in range(100)]
        mock_load.return_value = _make_df(rows)
        result = load_data("smn")
        assert len(result) == 50

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_data("nonexistent")

    def test_grib2_dataset_raises(self):
        with pytest.raises(ValueError, match="binary/grid"):
            load_data("forecast_icon_ch1")

    def test_netcdf_dataset_raises(self):
        with pytest.raises(ValueError, match="binary/grid"):
            load_data("surface_derived_grid")

    def test_invalid_frequency_raises(self):
        with pytest.raises(ValueError, match="Invalid frequency"):
            load_data("smn", frequency="x")

    def test_invalid_time_slice_raises(self):
        with pytest.raises(ValueError, match="Invalid time_slice"):
            load_data("smn", time_slice="future")

    @patch("foehn.mcp_server.foehn.load")
    def test_none_kwargs_not_passed(self, mock_load):
        """When station/frequency/time_slice are None, they should not appear in kwargs."""
        mock_load.return_value = _make_df([{"x": 1}])
        load_data("smn", station=None, frequency=None, time_slice=None)
        mock_load.assert_called_once_with("smn")

    @patch("foehn.mcp_server.foehn.load")
    def test_returns_list_of_dicts(self, mock_load):
        mock_load.return_value = _make_df([{"station": "BER", "temp": 20.5}])
        result = load_data("smn")
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert "station" in result[0]

    @patch("foehn.mcp_server.foehn.load")
    def test_year_filter(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER", "BER"],
                "reference_timestamp": [
                    datetime(2023, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2025, 1, 1),
                ],
                "temp": [18.0, 19.0, 20.0],
            }
        )
        result = load_data("smn", year=[2025], limit=500)
        assert len(result) == 1
        assert result[0]["temp"] == 20.0

    @patch("foehn.mcp_server.foehn.load")
    def test_year_filter_multiple_years(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER", "BER"],
                "reference_timestamp": [
                    datetime(2023, 1, 1),
                    datetime(2024, 1, 1),
                    datetime(2025, 1, 1),
                ],
                "temp": [18.0, 19.0, 20.0],
            }
        )
        result = load_data("smn", year=[2024, 2025], limit=500)
        assert len(result) == 2

    @patch("foehn.mcp_server.foehn.load")
    def test_columns_filter(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER"],
                "reference_timestamp": [datetime(2025, 1, 1)],
                "temp": [20.0],
                "precip": [5.0],
                "wind": [10.0],
            }
        )
        result = load_data("smn", columns=["temp"], limit=500)
        assert len(result) == 1
        assert set(result[0].keys()) == {"station_abbr", "reference_timestamp", "temp"}

    @patch("foehn.mcp_server.foehn.load")
    def test_columns_filter_ignores_nonexistent(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER"],
                "reference_timestamp": [datetime(2025, 1, 1)],
                "temp": [20.0],
            }
        )
        result = load_data("smn", columns=["temp", "nonexistent"], limit=500)
        assert set(result[0].keys()) == {"station_abbr", "reference_timestamp", "temp"}

    @patch("foehn.mcp_server.foehn.load")
    def test_year_and_columns_combined(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER"],
                "reference_timestamp": [datetime(2024, 1, 1), datetime(2025, 1, 1)],
                "temp": [19.0, 20.0],
                "precip": [3.0, 5.0],
            }
        )
        result = load_data("smn", year=[2025], columns=["temp"], limit=500)
        assert len(result) == 1
        assert set(result[0].keys()) == {"station_abbr", "reference_timestamp", "temp"}
        assert result[0]["temp"] == 20.0

    @patch("foehn.mcp_server.foehn.load")
    def test_month_filter(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER", "BER"],
                "reference_timestamp": [
                    datetime(2025, 1, 15),
                    datetime(2025, 6, 15),
                    datetime(2025, 7, 15),
                ],
                "temp": [0.0, 20.0, 25.0],
            }
        )
        result = load_data("smn", month=[6, 7], limit=500)
        assert len(result) == 2
        assert result[0]["temp"] == 20.0
        assert result[1]["temp"] == 25.0

    @patch("foehn.mcp_server.foehn.load")
    def test_year_and_month_combined(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER", "BER", "BER"],
                "reference_timestamp": [
                    datetime(2024, 7, 1),
                    datetime(2025, 1, 1),
                    datetime(2025, 7, 1),
                    datetime(2025, 12, 1),
                ],
                "temp": [22.0, 0.0, 25.0, -2.0],
            }
        )
        result = load_data("smn", year=[2025], month=[7], limit=500)
        assert len(result) == 1
        assert result[0]["temp"] == 25.0

    @patch("foehn.mcp_server.foehn.load")
    def test_date_from_filter(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER", "BER"],
                "reference_timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 6, 1),
                    datetime(2025, 12, 1),
                ],
                "temp": [0.0, 20.0, -1.0],
            }
        )
        result = load_data("smn", date_from="2025-06-01", limit=500)
        assert len(result) == 2

    @patch("foehn.mcp_server.foehn.load")
    def test_date_to_filter(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER", "BER"],
                "reference_timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 6, 1),
                    datetime(2025, 12, 1),
                ],
                "temp": [0.0, 20.0, -1.0],
            }
        )
        result = load_data("smn", date_to="2025-06-01", limit=500)
        assert len(result) == 2

    @patch("foehn.mcp_server.foehn.load")
    def test_date_range_filter(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER", "BER", "BER"],
                "reference_timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 6, 1),
                    datetime(2025, 8, 31),
                    datetime(2025, 12, 1),
                ],
                "temp": [0.0, 20.0, 25.0, -1.0],
            }
        )
        result = load_data("smn", date_from="2025-06-01", date_to="2025-08-31", limit=500)
        assert len(result) == 2
        assert result[0]["temp"] == 20.0
        assert result[1]["temp"] == 25.0

    @patch("foehn.mcp_server.foehn.load")
    def test_drop_null_filter(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "ZUR", "GVE"],
                "reference_timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 1),
                ],
                "hail": [3, None, None],
            }
        )
        result = load_data("smn", drop_null="hail", limit=500)
        assert len(result) == 1
        assert result[0]["station_abbr"] == "BER"

    @patch("foehn.mcp_server.foehn.load")
    def test_drop_null_nonexistent_column_ignored(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER"],
                "reference_timestamp": [datetime(2025, 1, 1)],
                "temp": [20.0],
            }
        )
        result = load_data("smn", drop_null="nonexistent", limit=500)
        assert len(result) == 1

    @patch("foehn.mcp_server.foehn.load")
    def test_sort_desc(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER", "BER"],
                "reference_timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 6, 1),
                    datetime(2025, 12, 1),
                ],
                "temp": [0.0, 20.0, -1.0],
            }
        )
        result = load_data("smn", sort="desc", limit=500)
        assert len(result) == 3
        assert result[0]["temp"] == -1.0  # December first
        assert result[2]["temp"] == 0.0  # January last

    @patch("foehn.mcp_server.foehn.load")
    def test_sort_asc(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER"],
                "reference_timestamp": [
                    datetime(2025, 12, 1),
                    datetime(2025, 1, 1),
                ],
                "temp": [-1.0, 0.0],
            }
        )
        result = load_data("smn", sort="asc", limit=500)
        assert result[0]["temp"] == 0.0  # January first

    def test_invalid_sort_raises(self):
        with pytest.raises(ValueError, match="Invalid sort"):
            load_data("smn", sort="random")

    @patch("foehn.mcp_server.foehn.load")
    def test_sort_desc_with_limit(self, mock_load):
        """sort=desc + limit gets the most recent rows."""
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER"] * 5,
                "reference_timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 3, 1),
                    datetime(2025, 6, 1),
                    datetime(2025, 9, 1),
                    datetime(2025, 12, 1),
                ],
                "temp": [0.0, 5.0, 20.0, 15.0, -1.0],
            }
        )
        result = load_data("smn", sort="desc", limit=2)
        assert len(result) == 2
        assert result[0]["temp"] == -1.0  # December
        assert result[1]["temp"] == 15.0  # September


# ── describe_data ────────────────────────────────────────────────────────────


class TestDescribeData:
    @patch("foehn.mcp_server.foehn.load")
    def test_basic_summary(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "ZUR", "BER"],
                "reference_timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 1),
                    datetime(2025, 2, 1),
                ],
                "temp": [5.0, 3.0, 8.0],
            }
        )
        result = describe_data("smn")
        assert isinstance(result, DataSummary)
        assert result.dataset == "smn"
        assert result.total_rows == 3
        assert sorted(result.stations) == ["BER", "ZUR"]
        assert result.date_min is not None
        assert result.date_max is not None

    @patch("foehn.mcp_server.foehn.load")
    def test_column_summaries(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER"],
                "reference_timestamp": [datetime(2025, 1, 1), datetime(2025, 2, 1)],
                "temp": [5.0, None],
            }
        )
        result = describe_data("smn")
        col_names = [c.name for c in result.columns]
        assert "temp" in col_names
        temp_col = next(c for c in result.columns if c.name == "temp")
        assert temp_col.non_null_count == 1
        assert temp_col.null_count == 1

    @patch("foehn.mcp_server.foehn.load")
    def test_with_year_filter(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER"],
                "reference_timestamp": [datetime(2024, 1, 1), datetime(2025, 1, 1)],
                "temp": [19.0, 20.0],
            }
        )
        result = describe_data("smn", year=[2025])
        assert result.total_rows == 1

    @patch("foehn.mcp_server.foehn.load")
    def test_with_month_filter(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER", "BER"],
                "reference_timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 7, 1),
                    datetime(2025, 12, 1),
                ],
                "temp": [0.0, 25.0, -1.0],
            }
        )
        result = describe_data("smn", month=[7])
        assert result.total_rows == 1

    @patch("foehn.mcp_server.foehn.load")
    def test_with_date_range(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER", "BER"],
                "reference_timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 6, 15),
                    datetime(2025, 12, 1),
                ],
                "temp": [0.0, 20.0, -1.0],
            }
        )
        result = describe_data("smn", date_from="2025-06-01", date_to="2025-08-31")
        assert result.total_rows == 1

    @patch("foehn.mcp_server.foehn.load")
    def test_with_drop_null(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "ZUR", "GVE"],
                "reference_timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 1),
                ],
                "hail": [3, None, None],
            }
        )
        result = describe_data("smn", drop_null="hail")
        assert result.total_rows == 1
        assert result.stations == ["BER"]

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            describe_data("nonexistent")

    def test_grib2_dataset_raises(self):
        with pytest.raises(ValueError, match="binary/grid"):
            describe_data("forecast_icon_ch1")

    def test_invalid_frequency_raises(self):
        with pytest.raises(ValueError, match="Invalid frequency"):
            describe_data("smn", frequency="x")

    def test_invalid_time_slice_raises(self):
        with pytest.raises(ValueError, match="Invalid time_slice"):
            describe_data("smn", time_slice="future")

    @patch("foehn.mcp_server.foehn.load")
    def test_empty_dataframe(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": pl.Series([], dtype=pl.Utf8),
                "reference_timestamp": pl.Series([], dtype=pl.Datetime),
                "temp": pl.Series([], dtype=pl.Float64),
            }
        )
        result = describe_data("smn")
        assert result.total_rows == 0
        assert result.stations == []
        assert result.date_min is None
        assert result.date_max is None


# ── get_parameters ───────────────────────────────────────────────────────────


class TestGetParameters:
    @patch("foehn.mcp_server.foehn.parameters")
    def test_returns_parameter_list(self, mock_params):
        mock_params.return_value = pl.DataFrame(
            {
                "shortname": ["tre200s0"],
                "description": ["Air temperature 2m"],
                "unit": ["°C"],
                "type": ["Float"],
                "granularity": ["T"],
                "decimals": [1],
                "group": ["Temperature"],
            }
        )
        result = get_parameters("smn")
        assert len(result) == 1
        assert isinstance(result[0], Parameter)
        assert result[0].shortname == "tre200s0"
        assert result[0].unit == "°C"
        mock_params.assert_called_once_with("smn")

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_parameters("nonexistent")


# ── get_stations ─────────────────────────────────────────────────────────────


class TestGetStations:
    @patch("foehn.mcp_server.foehn.stations")
    def test_returns_station_list(self, mock_stations):
        mock_stations.return_value = pl.DataFrame(
            {
                "abbr": ["BER"],
                "name": ["Bern"],
                "canton": ["BE"],
                "altitude": [553],
                "lat": [46.9508],
                "lon": [7.4394],
                "data_since": ["1864-01-01"],
            }
        )
        result = get_stations("smn")
        assert len(result) == 1
        assert isinstance(result[0], Station)
        assert result[0].abbr == "BER"
        assert result[0].name == "Bern"
        mock_stations.assert_called_once_with("smn")

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_stations("nonexistent")


# ── get_inventory ────────────────────────────────────────────────────────────


class TestGetInventory:
    @patch("foehn.mcp_server.foehn.inventory")
    def test_returns_inventory_list(self, mock_inv):
        mock_inv.return_value = pl.DataFrame(
            {
                "station": ["BER"],
                "parameter": ["tre200d0"],
                "data_since": ["1864-01-01"],
                "data_till": ["2024-12-31"],
                "owner": ["MeteoSchweiz"],
            }
        )
        result = get_inventory("smn")
        assert len(result) == 1
        assert isinstance(result[0], InventoryEntry)
        assert result[0].station == "BER"
        assert result[0].parameter == "tre200d0"
        mock_inv.assert_called_once_with("smn")

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_inventory("nonexistent")


# ── usage_guide resource ─────────────────────────────────────────────────────


class TestUsageGuide:
    def test_returns_string(self):
        result = usage_guide()
        assert isinstance(result, str)

    def test_contains_header(self):
        result = usage_guide()
        assert "foehn — MeteoSwiss Data Access Guide" in result

    def test_contains_loadable_datasets(self):
        result = usage_guide()
        for ds in _LOADABLE_DATASETS:
            assert f"`{ds}`" in result

    def test_contains_binary_datasets(self):
        result = usage_guide()
        for ds in sorted(GRIB2_COLLECTIONS | NETCDF_COLLECTIONS):
            assert f"`{ds}`" in result

    def test_contains_workflow_steps(self):
        result = usage_guide()
        assert "list_datasets()" in result
        assert "get_stations(dataset)" in result
        assert "get_parameters(dataset)" in result
        assert "describe_data(" in result
        assert "load_data(" in result

    def test_contains_frequency_docs(self):
        result = usage_guide()
        for freq in ("10-minute", "hourly", "daily", "monthly", "yearly"):
            assert freq in result

    def test_contains_time_slice_docs(self):
        result = usage_guide()
        for ts in ("now", "recent", "historical"):
            assert f"`{ts}`" in result

    def test_contains_attribution(self):
        result = usage_guide()
        assert "MeteoSwiss" in result
        assert "foehn" in result


# ── query_weather prompt ─────────────────────────────────────────────────────


class TestQueryWeather:
    def test_returns_string_with_question(self):
        result = query_weather("What is the temperature in Bern?")
        assert isinstance(result, str)
        assert "What is the temperature in Bern?" in result

    def test_contains_workflow_instructions(self):
        result = query_weather("test question")
        assert "list_datasets()" in result
        assert "get_stations(dataset)" in result
        assert "get_parameters(dataset)" in result
        assert "load_data(" in result

    def test_mentions_foehn_guide(self):
        result = query_weather("test")
        assert "foehn://guide" in result


# ── MCP server instance ─────────────────────────────────────────────────────


class TestMCPInstance:
    def test_mcp_name(self):
        assert mcp.name == "foehn"

    def test_mcp_has_instructions(self):
        assert mcp.instructions is not None
        assert "foehn" in mcp.instructions


# ── run() ────────────────────────────────────────────────────────────────────


class TestRun:
    @patch.object(mcp, "run")
    def test_run_default_transport(self, mock_run):
        from foehn.mcp_server import run

        run()
        mock_run.assert_called_once_with(transport="stdio")

    @patch.object(mcp, "run")
    def test_run_custom_transport(self, mock_run):
        from foehn.mcp_server import run

        run(transport="sse")
        mock_run.assert_called_once_with(transport="sse")
