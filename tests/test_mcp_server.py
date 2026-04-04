"""Tests for the MCP server tools, resource, and prompt."""

from __future__ import annotations

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
    InventoryEntry,
    Parameter,
    Station,
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
