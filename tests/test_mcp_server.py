"""Tests for the MCP server tools, resource, and prompt."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import ANY, patch

import polars as pl
import pytest

pytest.importorskip("mcp", reason="mcp not installed")

from foehn import registry
from foehn.collections import CATEGORIES, COLLECTIONS, GRANULARITY_LABELS, TIME_SLICE_LABELS
from foehn.mcp_server import (
    _INSPECTABLE_GRIDS,
    _LOAD_BUDGET_BYTES,
    _LOADABLE_DATASETS,
    Dataset,
    DataSummary,
    GridSummary,
    InventoryEntry,
    Page,
    Parameter,
    Station,
    aggregate_data,
    describe_data,
    describe_grid,
    get_inventory,
    get_parameters,
    get_points,
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
        for ds in registry.grid_datasets():
            assert ds not in _LOADABLE_DATASETS

    def test_loadable_datasets_excludes_netcdf(self):
        for ds in registry.grid_datasets():
            assert ds not in _LOADABLE_DATASETS

    def test_loadable_datasets_is_sorted(self):
        assert sorted(_LOADABLE_DATASETS) == _LOADABLE_DATASETS

    def test_loadable_datasets_all_exist_in_collections(self):
        for ds in _LOADABLE_DATASETS:
            assert ds in COLLECTIONS

    def test_inspectable_grids_are_all_gridded(self):
        assert sorted(registry.grid_datasets()) == _INSPECTABLE_GRIDS

    def test_load_data_names_every_loadable_dataset(self):
        """The tool docstring is the interface a model reads, so it is rendered, not typed.

        It used to be typed, and had fallen a dataset behind: it told callers
        ``climate_scenarios_indoor`` was not loadable while ``load_data`` loaded it.
        """
        doc = load_data.__doc__
        for dataset in registry.tabular_datasets():
            assert dataset in doc

    def test_no_tool_docstring_is_left_holding_a_placeholder(self):
        """A ``$name`` that survives import is a fragment nobody rendered."""
        for tool in (list_datasets, load_data, aggregate_data):
            assert "$" not in tool.__doc__


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
            lv95_east=2601933.0,
            lv95_north=1199885.0,
            lat=46.9508,
            lon=7.4394,
            data_since="01.01.1864",
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

    def test_inventory_entry_accepts_an_open_ended_range(self):
        entry = InventoryEntry(
            station="BER",
            parameter="tre200d0",
            data_since="1864-01-01",
            data_till=None,
            owner="MeteoSchweiz",
        )

        assert entry.data_till is None


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


def _oversized_smn(fetcher, monkeypatch) -> None:
    """Five smn stations, against a budget smaller than one of their files."""
    fetcher.any_collection = {"assets": {}}
    fetcher.any_items = [
        {"id": s, "assets": {"data": {"href": f"https://data.geo.admin.ch/smn/ogd-smn_{s}_d_recent.csv"}}}
        for s in ("ber", "zur", "gve", "bas", "lug")
    ]
    fetcher.default_body = "station_abbr;reference_timestamp;temp\nBER;2025-01-01T00:00:00;1.0\n"
    monkeypatch.setattr("foehn.mcp_server._LOAD_BUDGET_BYTES", 10)


class TestLoadData:
    def test_every_query_has_a_download_budget(self):
        """A model's query is refused past this, rather than run until the host runs out of memory."""
        assert 0 < _LOAD_BUDGET_BYTES <= 4 * 1024**3

    @patch("foehn.mcp_server.foehn.load")
    def test_load_data_passes_the_budget(self, mock_load):
        mock_load.return_value = _make_df([{"x": 1}])
        load_data("smn")
        assert mock_load.call_args.kwargs["max_bytes"] == _LOAD_BUDGET_BYTES

    def test_an_oversized_query_is_refused_with_advice(self, fetcher, monkeypatch):
        _oversized_smn(fetcher, monkeypatch)
        with pytest.raises(ValueError, match="station="):
            load_data("smn", frequency="d", year=[2025])

    @patch("foehn.mcp_server.foehn.load")
    def test_basic_load(self, mock_load):
        mock_load.return_value = _make_df([{"station": "BER", "temp": 20.5}, {"station": "ZUR", "temp": 18.3}])
        result = load_data("smn")
        assert isinstance(result, Page)
        assert (result.total, result.count, result.has_more, result.next_offset) == (2, 2, False, None)
        assert result.items[0] == {"station": "BER", "temp": 20.5}
        assert mock_load.call_args.args == ("smn",)

    @patch("foehn.mcp_server.foehn.load")
    def test_the_page_is_cut_here_not_by_foehn_load(self, mock_load):
        """foehn.load parses every matching row anyway, so slicing here is free and gives a total."""
        mock_load.return_value = _make_df([{"x": 1}])
        load_data("smn", limit=10)
        assert "limit" not in mock_load.call_args.kwargs

    @patch("foehn.mcp_server.foehn.load")
    def test_with_all_kwargs(self, mock_load):
        mock_load.return_value = _make_df([{"station": "BER", "temp": 20.5}])
        result = load_data("smn", station=["BER"], frequency="d", time_slice="recent", limit=10)
        assert result.count == 1
        assert mock_load.call_args.args == ("smn",)
        assert mock_load.call_args.kwargs["station"] == ["BER"]
        assert mock_load.call_args.kwargs["frequency"] == "d"
        assert mock_load.call_args.kwargs["time_slice"] == "recent"

    @patch("foehn.mcp_server.foehn.load")
    def test_pages_through_rows(self, mock_load):
        mock_load.return_value = _make_df([{"station": "BER", "n": i} for i in range(120)])

        first = load_data("smn")
        assert (first.total, first.count, first.has_more, first.next_offset) == (120, 50, True, 50)
        assert first.items[0]["n"] == 0

        third = load_data("smn", offset=100)
        assert [row["n"] for row in third.items] == list(range(100, 120))
        assert (third.has_more, third.next_offset) == (False, None)

    @patch("foehn.mcp_server.foehn.load")
    def test_limit_caps_at_500(self, mock_load):
        mock_load.return_value = _make_df([{"station": f"S{i}", "temp": float(i)} for i in range(600)])
        result = load_data("smn", limit=9999)
        assert (result.count, result.total, result.next_offset) == (500, 600, 500)

    @patch("foehn.mcp_server.foehn.load")
    def test_limit_minimum_is_1(self, mock_load):
        mock_load.return_value = _make_df([{"station": "BER", "temp": 20.5}, {"station": "ZUR", "temp": 1.0}])
        assert load_data("smn", limit=-5).count == 1

    @patch("foehn.mcp_server.foehn.load")
    def test_default_limit_is_50(self, mock_load):
        mock_load.return_value = _make_df([{"station": f"S{i}", "temp": float(i)} for i in range(80)])
        assert load_data("smn").count == 50

    @patch("foehn.mcp_server.foehn.load")
    def test_a_single_station_or_column_need_not_be_a_list(self, mock_load):
        mock_load.return_value = _make_df([{"x": 1}])
        load_data("smn", station="BER", columns="tre200s0", year=2025, month=7)
        kwargs = mock_load.call_args.kwargs
        assert (kwargs["station"], kwargs["columns"], kwargs["year"], kwargs["month"]) == ("BER", ["tre200s0"], 2025, 7)

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_data("nonexistent")

    def test_grib2_dataset_raises(self):
        with pytest.raises(ValueError, match="gridded"):
            load_data("forecast_icon_ch1")

    def test_netcdf_dataset_raises(self):
        with pytest.raises(ValueError, match="gridded"):
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
        assert mock_load.call_args.args == ("smn",)

    @patch("foehn.mcp_server.foehn.load")
    def test_rows_are_dicts(self, mock_load):
        mock_load.return_value = _make_df([{"station": "BER", "temp": 20.5}])
        assert load_data("smn").items == [{"station": "BER", "temp": 20.5}]

    @patch("foehn.mcp_server.foehn.load")
    def test_year_filter_passed_to_load(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {"station_abbr": ["BER"], "reference_timestamp": [datetime(2025, 1, 1)], "temp": [20.0]}
        )
        load_data("smn", year=[2025], limit=500)
        assert mock_load.call_args.args == ("smn",)
        assert mock_load.call_args.kwargs["year"] == [2025]

    @patch("foehn.mcp_server.foehn.load")
    def test_month_filter_passed_to_load(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {"station_abbr": ["BER"], "reference_timestamp": [datetime(2025, 6, 1)], "temp": [20.0]}
        )
        load_data("smn", month=[6, 7], limit=500)
        assert mock_load.call_args.args == ("smn",)
        assert mock_load.call_args.kwargs["month"] == [6, 7]

    @patch("foehn.mcp_server.foehn.load")
    def test_date_range_passed_to_load(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {"station_abbr": ["BER"], "reference_timestamp": [datetime(2025, 6, 1)], "temp": [20.0]}
        )
        load_data("smn", date_from="2025-06-01", date_to="2025-08-31", limit=500)
        assert mock_load.call_args.args == ("smn",)
        assert mock_load.call_args.kwargs["date_from"] == "2025-06-01"
        assert mock_load.call_args.kwargs["date_to"] == "2025-08-31"

    @patch("foehn.mcp_server.foehn.load")
    def test_columns_passed_to_load(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {"station_abbr": ["BER"], "reference_timestamp": [datetime(2025, 1, 1)], "temp": [20.0]}
        )
        load_data("smn", columns=["temp"], limit=500)
        assert mock_load.call_args.args == ("smn",)
        assert mock_load.call_args.kwargs["columns"] == ["temp"]

    @patch("foehn.mcp_server.foehn.load")
    def test_drop_null_passed_to_load(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {"station_abbr": ["BER"], "reference_timestamp": [datetime(2025, 1, 1)], "hail": [3]}
        )
        load_data("smn", drop_null="hail", limit=500)
        assert mock_load.call_args.args == ("smn",)
        assert mock_load.call_args.kwargs["drop_null"] == "hail"

    @patch("foehn.mcp_server.foehn.load")
    def test_sort_passed_to_load(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {"station_abbr": ["BER"], "reference_timestamp": [datetime(2025, 1, 1)], "temp": [20.0]}
        )
        load_data("smn", sort="desc", limit=500)
        assert mock_load.call_args.args == ("smn",)
        assert mock_load.call_args.kwargs["sort"] == "desc"

    @patch("foehn.mcp_server.foehn.load")
    def test_all_filters_combined_passed_to_load(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {"station_abbr": ["BER"], "reference_timestamp": [datetime(2025, 7, 1)], "temp": [25.0]}
        )
        load_data(
            "smn",
            station=["BER"],
            frequency="d",
            time_slice="historical",
            year=[2025],
            month=[7],
            columns=["temp"],
            sort="desc",
            limit=500,
        )
        assert mock_load.call_args.args == ("smn",)
        assert (
            mock_load.call_args.kwargs
            | {
                "station": ["BER"],
                "frequency": "d",
                "time_slice": "historical",
                "year": [2025],
                "month": [7],
                "columns": ["temp"],
                "sort": "desc",
            }
            == mock_load.call_args.kwargs
        )

    def test_invalid_sort_raises(self):
        with pytest.raises(ValueError, match="Invalid sort"):
            load_data("smn", sort="random")

    @patch("foehn.mcp_server.foehn.load")
    def test_sort_desc_with_limit(self, mock_load):
        """sort=desc + limit gets the most recent rows: foehn.load sorts, this layer slices."""
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER"],
                "reference_timestamp": [datetime(2025, 12, 1), datetime(2025, 9, 1)],
                "temp": [-1.0, 15.0],
            }
        )
        result = load_data("smn", sort="desc", limit=1)
        assert [row["temp"] for row in result.items] == [-1.0]  # December
        assert mock_load.call_args.kwargs["sort"] == "desc"
        assert mock_load.call_args.args == ("smn",)
        assert mock_load.call_args.kwargs["sort"] == "desc"


# ── describe_data ────────────────────────────────────────────────────────────


class TestDescribeData:
    @patch("foehn.mcp_server.foehn.load")
    def test_describe_data_passes_the_budget(self, mock_load):
        """It downloads exactly what load_data would, so it is held to the same ceiling."""
        mock_load.return_value = pl.DataFrame({"x": [1]})
        describe_data("smn")
        assert mock_load.call_args.kwargs["max_bytes"] == _LOAD_BUDGET_BYTES

    def test_an_oversized_summary_is_refused(self, fetcher, monkeypatch):
        _oversized_smn(fetcher, monkeypatch)
        with pytest.raises(ValueError, match="station="):
            describe_data("smn", frequency="d")

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
    def test_with_year_filter_passed_to_load(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER"],
                "reference_timestamp": [datetime(2025, 1, 1)],
                "temp": [20.0],
            }
        )
        result = describe_data("smn", year=[2025])
        assert mock_load.call_args.args == ("smn",)
        assert mock_load.call_args.kwargs["year"] == [2025]
        assert result.total_rows == 1

    @patch("foehn.mcp_server.foehn.load")
    def test_with_month_filter_passed_to_load(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER"],
                "reference_timestamp": [datetime(2025, 7, 1)],
                "temp": [25.0],
            }
        )
        result = describe_data("smn", month=[7])
        assert mock_load.call_args.args == ("smn",)
        assert mock_load.call_args.kwargs["month"] == [7]
        assert result.total_rows == 1

    @patch("foehn.mcp_server.foehn.load")
    def test_with_date_range_passed_to_load(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER"],
                "reference_timestamp": [datetime(2025, 6, 15)],
                "temp": [20.0],
            }
        )
        result = describe_data("smn", date_from="2025-06-01", date_to="2025-08-31")
        assert mock_load.call_args.args == ("smn",)
        assert mock_load.call_args.kwargs["date_from"] == "2025-06-01"
        assert mock_load.call_args.kwargs["date_to"] == "2025-08-31"
        assert result.total_rows == 1

    @patch("foehn.mcp_server.foehn.load")
    def test_with_drop_null_passed_to_load(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER"],
                "reference_timestamp": [datetime(2025, 1, 1)],
                "hail": [3],
            }
        )
        result = describe_data("smn", drop_null="hail")
        assert mock_load.call_args.args == ("smn",)
        assert mock_load.call_args.kwargs["drop_null"] == "hail"
        assert result.total_rows == 1
        assert result.stations == ["BER"]

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            describe_data("nonexistent")

    def test_grib2_dataset_raises(self):
        with pytest.raises(ValueError, match="gridded"):
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


# ── aggregate_data ───────────────────────────────────────────────────────────


def _daily_smn() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "station_abbr": ["BER", "BER", "BER", "ZUR", "ZUR"],
            "reference_timestamp": [
                datetime(2024, 7, 1),
                datetime(2024, 7, 2),
                datetime(2025, 7, 1),
                datetime(2024, 7, 1),
                datetime(2025, 7, 1),
            ],
            "temp": [20.0, 24.0, 30.0, 18.0, None],
            "rain": [1.0, 0.0, 2.5, 4.0, 3.0],
        }
    )


class TestAggregateData:
    @patch("foehn.mcp_server.foehn.load")
    def test_means_per_station_by_default(self, mock_load):
        mock_load.return_value = _daily_smn()
        result = aggregate_data("smn", "temp")
        assert isinstance(result, Page)
        assert result.items == [
            {"station_abbr": "BER", "rows": 3, "temp_mean": 24.667},
            {"station_abbr": "ZUR", "rows": 2, "temp_mean": 18.0},
        ]

    @patch("foehn.mcp_server.foehn.load")
    def test_several_stats_and_columns_per_station_and_year(self, mock_load):
        mock_load.return_value = _daily_smn()
        result = aggregate_data("smn", ["temp", "rain"], stats=["max", "sum", "count"], group_by=["station", "year"])
        assert result.total == 4
        assert result.items[0] == {
            "station_abbr": "BER",
            "year": 2024,
            "rows": 2,
            "temp_max": 24.0,
            "temp_sum": 44.0,
            "temp_count": 2,
            "rain_max": 1.0,
            "rain_sum": 1.0,
            "rain_count": 2,
        }
        # A null is a missing reading: counted as a row, not as a value.
        zur_2025 = result.items[3]
        assert (zur_2025["rows"], zur_2025["temp_count"], zur_2025["temp_max"]) == (1, 0, None)

    @patch("foehn.mcp_server.foehn.load")
    def test_time_parts_are_derived_from_the_timestamp(self, mock_load):
        mock_load.return_value = _daily_smn()
        by_date = aggregate_data("smn", "rain", stats="sum", group_by=["date"])
        assert [(row["date"].isoformat(), row["rain_sum"]) for row in by_date.items] == [
            ("2024-07-01", 5.0),
            ("2024-07-02", 0.0),
            ("2025-07-01", 5.5),
        ]
        by_month = aggregate_data("smn", "rain", group_by=["month"])
        assert [row["month"] for row in by_month.items] == [7]

    @patch("foehn.mcp_server.foehn.load")
    def test_an_empty_group_by_is_one_row_over_everything(self, mock_load):
        mock_load.return_value = _daily_smn()
        result = aggregate_data("smn", "rain", stats="sum", group_by=[])
        assert result.items == [{"rows": 5, "rain_sum": 10.5}]

    @patch("foehn.mcp_server.foehn.load")
    def test_top_n_by_a_result_column(self, mock_load):
        mock_load.return_value = _daily_smn()
        result = aggregate_data(
            "smn", "temp", stats="max", group_by=["station", "date"], order_by="temp_max", sort="desc", limit=2
        )
        assert [row["temp_max"] for row in result.items] == [30.0, 24.0]
        assert (result.total, result.next_offset) == (5, 2)

    @patch("foehn.mcp_server.foehn.load")
    def test_ties_in_order_by_break_on_the_group_keys(self, mock_load):
        """Groups come back in no fixed order, so pages would otherwise shift between calls."""
        mock_load.return_value = pl.DataFrame(
            {"station_abbr": ["ZUR", "BER", "GVE"], "reference_timestamp": [datetime(2025, 1, 1)] * 3, "x": [1, 1, 1]}
        )
        result = aggregate_data("smn", "x", order_by="x_mean", sort="desc")
        assert [row["station_abbr"] for row in result.items] == ["BER", "GVE", "ZUR"]

    @patch("foehn.mcp_server.foehn.load")
    def test_station_means_the_point_pair_for_forecast_local(self, mock_load):
        """Point IDs are reused across point types, so one ID is not one point."""
        mock_load.return_value = pl.DataFrame(
            {
                "point_id": [1, 1, 1],
                "point_type_id": [1, 2, 1],
                "reference_timestamp": [datetime(2026, 11, 12, h) for h in (1, 1, 2)],
                "tre200h0": [5.0, 9.0, 7.0],
            }
        )
        result = aggregate_data("forecast_local", "tre200h0", stats="max")
        assert [(r["point_id"], r["point_type_id"], r["tre200h0_max"]) for r in result.items] == [
            (1, 1, 7.0),
            (1, 2, 9.0),
        ]

    @patch("foehn.mcp_server.foehn.load")
    def test_a_column_group_key_is_loaded_alongside_the_metrics(self, mock_load):
        mock_load.return_value = pl.DataFrame(
            {
                "station_abbr": ["BER", "BER"],
                "gwl": ["1.5", "3.0"],
                "date": ["0001-01-01"] * 2,
                "SMHI-RCA-ECEARTH": [2.0, 3.0],
            }
        )
        result = aggregate_data("climate_scenarios", "SMHI-RCA-ECEARTH", group_by=["gwl"])
        assert mock_load.call_args.kwargs["columns"] == ["SMHI-RCA-ECEARTH", "gwl"]
        assert [(r["gwl"], r["SMHI-RCA-ECEARTH_mean"]) for r in result.items] == [("1.5", 2.0), ("3.0", 3.0)]

    @patch("foehn.mcp_server.foehn.load")
    def test_filters_and_budget_are_passed_to_load(self, mock_load):
        mock_load.return_value = _daily_smn()
        aggregate_data(
            "smn",
            ["temp"],
            station="BER",
            frequency="d",
            time_slice="historical",
            year=[2024],
            month=7,
            date_from="2024-01-01",
            date_to="2024-12-31",
        )
        mock_load.assert_called_once_with(
            "smn",
            station="BER",
            frequency="d",
            time_slice="historical",
            year=[2024],
            month=7,
            date_from="2024-01-01",
            date_to="2024-12-31",
            columns=["temp"],
            max_bytes=_LOAD_BUDGET_BYTES,
        )

    @patch("foehn.mcp_server.foehn.load")
    def test_float_results_are_rounded(self, mock_load):
        mock_load.return_value = pl.DataFrame({"station_abbr": ["BER"] * 3, "x": [1.0, 1.0, 2.0]})
        assert aggregate_data("smn", "x").items[0]["x_mean"] == 1.333

    @patch("foehn.mcp_server.foehn.load")
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"columns": []}, "at least one column"),
            ({"columns": "temp", "stats": "mode"}, "Invalid stats"),
            ({"columns": "temp", "sort": "up"}, "Invalid sort"),
            ({"columns": "station_abbr"}, "numeric"),
            ({"columns": "temp", "group_by": ["season"]}, "Invalid group_by 'season'"),
            ({"columns": "temp", "order_by": "temp_max"}, "Invalid order_by"),
        ],
    )
    def test_bad_arguments_are_refused_with_the_valid_options(self, mock_load, kwargs, message):
        mock_load.return_value = _daily_smn()
        with pytest.raises(ValueError, match=message):
            aggregate_data("smn", **kwargs)

    @patch("foehn.mcp_server.foehn.load")
    def test_a_time_group_needs_a_timestamp(self, mock_load):
        mock_load.return_value = pl.DataFrame({"gwl": ["1.5"], "SMHI-RCA-ECEARTH": [2.0]})
        with pytest.raises(ValueError, match="reference_timestamp"):
            aggregate_data("climate_scenarios", "SMHI-RCA-ECEARTH", group_by=["year"])

    @patch("foehn.mcp_server.foehn.load")
    def test_a_dataset_without_a_timestamp_groups_by_its_own_date_column(self, mock_load):
        """climate_scenarios has nominal days in ``date`` and no reference_timestamp."""
        mock_load.return_value = pl.DataFrame({"date": ["0001-01-01", "0001-01-01"], "SMHI-RCA-ECEARTH": [2.0, 4.0]})
        result = aggregate_data("climate_scenarios", "SMHI-RCA-ECEARTH", group_by=["date"])
        assert result.items == [{"date": "0001-01-01", "rows": 2, "SMHI-RCA-ECEARTH_mean": 3.0}]

    def test_an_oversized_aggregation_is_refused(self, fetcher, monkeypatch):
        _oversized_smn(fetcher, monkeypatch)
        with pytest.raises(ValueError, match="station="):
            aggregate_data("smn", "temp", frequency="d")

    def test_dates_in_the_result_reach_protocol_clients(self):
        from mcp import Client

        async def call_tool():
            async with Client(mcp) as client:
                return await client.call_tool(
                    "aggregate_data", {"dataset": "smn", "columns": "rain", "stats": "sum", "group_by": ["date"]}
                )

        with patch("foehn.mcp_server.foehn.load", return_value=_daily_smn()):
            result = asyncio.run(call_tool())
        assert not result.is_error
        assert result.structured_content["items"][0] == {"date": "2024-07-01", "rows": 2, "rain_sum": 5.0}


# ── describe_grid ────────────────────────────────────────────────────────────


class TestDescribeGrid:
    def test_grids_are_cached_in_the_user_cache_not_the_working_directory(self, monkeypatch):
        """A client starts the server wherever it likes — often ``/``, where nothing can be written."""
        from foehn.workspace import DATA_DIR_ENV, user_cache_root

        monkeypatch.delenv(DATA_DIR_ENV, raising=False)
        with (
            patch("foehn.mcp_server.foehn.open_dataset", side_effect=ValueError("stop")) as mock_open,
            pytest.raises(ValueError, match="stop"),
        ):
            describe_grid("surface_derived_grid", match="rhiresd")

        assert mock_open.call_args.kwargs["data_dir"] == user_cache_root()

    def test_foehn_data_dir_still_decides_where_grids_go(self, monkeypatch, tmp_path):
        from foehn.workspace import DATA_DIR_ENV

        monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
        with (
            patch("foehn.mcp_server.foehn.open_dataset", side_effect=ValueError("stop")) as mock_open,
            pytest.raises(ValueError, match="stop"),
        ):
            describe_grid("surface_derived_grid", match="rhiresd")

        # None hands the decision to the Workspace, which reads the variable.
        assert mock_open.call_args.kwargs["data_dir"] is None

    def test_describe_grid_leaves_no_snapshot_behind(self, fetcher, tmp_path, monkeypatch):
        """A long-running server kept one read snapshot per call until it exited."""
        for module in ("xarray", "h5py", "pyproj"):
            pytest.importorskip(module)
        from conftest import write_odim_composite

        monkeypatch.setenv("FOEHN_DATA_DIR", str(tmp_path))
        name = "cpc2613000000_00060.001.h5"
        fetcher.any_items = [{"assets": {"d": {"href": f"https://data.geo.admin.ch/x/{name}"}}}]
        bronze = tmp_path / "bronze" / "radar_precip"
        write_odim_composite(bronze / name)

        summary = describe_grid("radar_precip", match="cpc2613000000")

        assert [variable.name for variable in summary.data_variables] == ["acrr"]
        assert list(bronze.glob(".foehn-snapshot-*")) == []

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            describe_grid("nonexistent")

    def test_csv_dataset_raises(self):
        with pytest.raises(ValueError, match="CSV/tabular"):
            describe_grid("smn")

    def test_grib2_inspectable_via_open_dataset(self):
        """describe_grid now accepts GRIB2; it delegates to open_dataset (here mocked)."""
        xr = pytest.importorskip("xarray")
        import numpy as np

        ds = xr.Dataset({"t2m": (("values",), np.zeros(5, dtype="float32"))})
        with patch("foehn.mcp_server.foehn.open_dataset", return_value=ds) as mock_open:
            result = describe_grid("forecast_icon_ch1", match="202605231500-0-t_2m-ctrl")
        assert isinstance(result, GridSummary)
        assert result.dimensions == {"values": 5}
        mock_open.assert_called_once_with(
            "forecast_icon_ch1", match="202605231500-0-t_2m-ctrl", variables=None, data_dir=ANY
        )

    def test_radar_inspectable_via_open_dataset(self):
        """describe_grid now accepts radar (HDF5) too."""
        xr = pytest.importorskip("xarray")
        import numpy as np

        ds = xr.Dataset(
            {"acrr": (("y", "x"), np.zeros((2, 3), dtype="float32"))},
            coords={"y": [0, 1], "x": [0, 1, 2]},
        )
        with patch("foehn.mcp_server.foehn.open_dataset", return_value=ds):
            result = describe_grid("radar_precip", match="cpc2613000000")
        assert result.dimensions == {"y": 2, "x": 3}
        assert "acrr" in [v.name for v in result.data_variables]

    def test_summary_from_dataset(self):
        xr = pytest.importorskip("xarray")
        import numpy as np

        ds = xr.Dataset(
            {
                "TabsD": (
                    ("time", "y", "x"),
                    np.zeros((2, 3, 4), dtype="float32"),
                    {"long_name": "daily mean temperature", "units": "degC"},
                ),
            },
            coords={"time": [0, 1], "y": [0, 1, 2], "x": [0, 1, 2, 3]},
            attrs={"title": "Test grid", "institution": "MeteoSwiss"},
        )
        with patch("foehn.mcp_server.foehn.open_dataset", return_value=ds) as mock_open:
            result = describe_grid("surface_derived_grid", match="rhiresd")

        assert isinstance(result, GridSummary)
        assert result.dataset == "surface_derived_grid"
        assert result.dimensions == {"time": 2, "y": 3, "x": 4}
        assert "time" in result.coordinates
        var = next(v for v in result.data_variables if v.name == "TabsD")
        assert var.dims == ["time", "y", "x"]
        assert var.shape == [2, 3, 4]
        assert var.dtype == "float32"
        assert var.units == "degC"
        assert var.long_name == "daily mean temperature"
        assert result.attributes["title"] == "Test grid"
        mock_open.assert_called_once_with("surface_derived_grid", match="rhiresd", variables=None, data_dir=ANY)

    def test_missing_var_metadata_is_none(self):
        xr = pytest.importorskip("xarray")
        import numpy as np

        ds = xr.Dataset(
            {"v": (("y", "x"), np.zeros((2, 2), dtype="float32"))},
            coords={"y": [0, 1], "x": [0, 1]},
        )
        with patch("foehn.mcp_server.foehn.open_dataset", return_value=ds):
            result = describe_grid("surface_derived_grid")
        var = result.data_variables[0]
        assert var.units is None
        assert var.long_name is None


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
                "lv95_east": [2601933.0],
                "lv95_north": [1199885.0],
                "lat": [46.9508],
                "lon": [7.4394],
                "data_since": ["01.01.1864"],
            }
        )
        result = get_stations("smn")
        assert len(result) == 1
        assert isinstance(result[0], Station)
        assert result[0].abbr == "BER"
        assert result[0].name == "Bern"
        assert result[0].lv95_east == 2601933.0
        assert result[0].lv95_north == 1199885.0
        mock_stations.assert_called_once_with("smn")

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_stations("nonexistent")


# ── get_points ───────────────────────────────────────────────────────────────


class TestGetPoints:
    @staticmethod
    def _points():
        return pl.DataFrame(
            {
                "point_id": [78, 301100, 641],
                "point_type_id": [1, 2, 3],
                "type": ["Station", "Postal code center", "Point of Interest"],
                "abbr": ["BER", None, None],
                "postal_code": [None, "3011", None],
                "name": ["Bern / Zollikofen", "Bern", "Gurten"],
                "altitude": [553.0, 540.0, 858.0],
                "lv95_east": [2601934.0, 2600550.0, 2600000.0],
                "lv95_north": [1204410.0, 1199625.0, 1197000.0],
                "lat": [46.99, 46.95, 46.92],
                "lon": [7.46, 7.45, 7.44],
            }
        )

    @patch("foehn.mcp_server.foehn.points")
    def test_returns_every_point_without_a_query(self, mock_points):
        mock_points.return_value = self._points()
        result = get_points("forecast_local")
        assert [point.point_id for point in result.items] == [78, 301100, 641]
        assert result.items[1].abbr is None
        assert (result.total, result.has_more, result.next_offset) == (3, False, None)
        mock_points.assert_called_once_with("forecast_local")

    @patch("foehn.mcp_server.foehn.points")
    def test_query_matches_name_abbreviation_or_postal_code(self, mock_points):
        mock_points.return_value = self._points()

        def ids(query):
            return [p.point_id for p in get_points("forecast_local", query=query).items]

        assert ids("bern") == [78, 301100]
        assert ids("ber") == [78, 301100]
        assert ids("3011") == [301100]
        assert ids("zermatt") == []

    @patch("foehn.mcp_server.foehn.points")
    def test_pages_through_points(self, mock_points):
        mock_points.return_value = self._points()

        first = get_points("forecast_local", limit=2)
        assert [p.point_id for p in first.items] == [78, 301100]
        assert (first.total, first.count, first.has_more, first.next_offset) == (3, 2, True, 2)

        second = get_points("forecast_local", limit=2, offset=first.next_offset)
        assert [p.point_id for p in second.items] == [641]
        assert (second.offset, second.has_more, second.next_offset) == (2, False, None)

    @patch("foehn.mcp_server.foehn.points")
    def test_paging_counts_matches_not_all_points(self, mock_points):
        mock_points.return_value = self._points()
        result = get_points("forecast_local", query="bern", limit=1)
        assert (result.total, result.next_offset) == (2, 1)


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
        assert result.total == 1
        assert isinstance(result.items[0], InventoryEntry)
        assert result.items[0].station == "BER"
        assert result.items[0].parameter == "tre200d0"
        mock_inv.assert_called_once_with("smn")

    @staticmethod
    def _inventory():
        return pl.DataFrame(
            {
                "station": ["BER", "BER", "GVE", "ZUR"],
                "parameter": ["tre200d0", "rre150d0", "tre200d0", "tre200d0"],
                "data_since": ["1864-01-01"] * 4,
                "data_till": ["2024-12-31"] * 4,
                "owner": ["MeteoSchweiz"] * 4,
            }
        )

    @patch("foehn.mcp_server.foehn.inventory")
    def test_filters_by_station_and_parameter_case_insensitively(self, mock_inv):
        mock_inv.return_value = self._inventory()

        by_station = get_inventory("smn", station=["ber"])
        assert [(e.station, e.parameter) for e in by_station.items] == [("BER", "tre200d0"), ("BER", "rre150d0")]

        by_parameter = get_inventory("smn", parameter=["TRE200D0"])
        assert [e.station for e in by_parameter.items] == ["BER", "GVE", "ZUR"]

        both = get_inventory("smn", station=["GVE", "ZUR"], parameter=["tre200d0"])
        assert both.total == 2

    @patch("foehn.mcp_server.foehn.inventory")
    def test_pages_through_the_inventory(self, mock_inv):
        mock_inv.return_value = self._inventory()

        first = get_inventory("smn", limit=3)
        assert (first.count, first.total, first.has_more, first.next_offset) == (3, 4, True, 3)

        last = get_inventory("smn", limit=3, offset=3)
        assert [e.station for e in last.items] == ["ZUR"]
        assert (last.has_more, last.next_offset) == (False, None)

        past_the_end = get_inventory("smn", offset=10)
        assert (past_the_end.items, past_the_end.has_more, past_the_end.total) == ([], False, 4)

    @patch("foehn.mcp_server.foehn.inventory")
    def test_page_size_is_capped_and_defaulted(self, mock_inv):
        rows = 600
        mock_inv.return_value = pl.DataFrame(
            {
                "station": ["BER"] * rows,
                "parameter": [f"p{i}" for i in range(rows)],
                "data_since": ["1864-01-01"] * rows,
                "data_till": ["2024-12-31"] * rows,
                "owner": ["MeteoSchweiz"] * rows,
            }
        )
        assert get_inventory("smn").count == 100
        assert get_inventory("smn", limit=10_000).count == 500
        assert get_inventory("smn", limit=0).count == 1

    @patch("foehn.mcp_server.foehn.inventory")
    def test_negative_offset_raises(self, mock_inv):
        mock_inv.return_value = self._inventory()
        with pytest.raises(ValueError, match="Invalid offset"):
            get_inventory("smn", offset=-1)

    @patch("foehn.mcp_server.foehn.inventory")
    def test_preserves_an_open_ended_range(self, mock_inv):
        mock_inv.return_value = pl.DataFrame(
            {
                "station": ["BER"],
                "parameter": ["tre200d0"],
                "data_since": ["1864-01-01"],
                "data_till": [None],
                "owner": ["MeteoSchweiz"],
            },
            schema_overrides={"data_till": pl.String},
        )

        result = get_inventory("smn")

        assert result.items[0].data_till is None

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
        for ds in sorted(registry.grid_datasets()):
            assert f"`{ds}`" in result

    def test_contains_workflow_steps(self):
        result = usage_guide()
        assert "list_datasets()" in result
        assert "get_stations(dataset)" in result
        assert "get_parameters(dataset)" in result
        assert "describe_data(" in result
        assert "load_data(" in result

    def test_contains_describe_grid(self):
        result = usage_guide()
        assert "describe_grid(" in result
        for ds in _INSPECTABLE_GRIDS:
            assert f"`{ds}`" in result

    def test_frequency_docs_come_from_the_vocabulary_load_enforces(self):
        """A token added to GRANULARITIES has to reach the guide, not be retyped into it."""
        from foehn.collections import GRANULARITY_LABELS

        result = usage_guide()
        for token, label in GRANULARITY_LABELS.items():
            assert f"- `{token}` — {label}" in result

    def test_time_slice_docs_come_from_the_vocabulary_load_enforces(self):
        from foehn.collections import TIME_SLICE_LABELS

        result = usage_guide()
        for token, label in TIME_SLICE_LABELS.items():
            assert f"- `{token}` — {label}" in result

    def test_does_not_promise_a_sort_default_that_does_not_exist(self):
        """load() with sort=None does not sort at all; the guide used to say "asc".

        Worth a test rather than just a fix: this seam's whole risk is prose that
        describes another module's contract, and an LLM acts on it.
        """
        result = usage_guide()
        assert "oldest first, default" not in result
        assert "unsorted" in result

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

    @pytest.mark.parametrize(
        ("tool", "arguments", "guidance"),
        [
            ("list_datasets", {"category": "bogus"}, "Valid options"),
            ("load_data", {"dataset": "smn", "frequency": "bogus"}, "Invalid frequency"),
            ("describe_grid", {"dataset": "smn"}, "Use describe_data()"),
            ("get_stations", {"dataset": "bogus"}, "Unknown dataset"),
        ],
    )
    def test_expected_errors_reach_protocol_clients(self, tool, arguments, guidance):
        from mcp import Client

        async def call_tool():
            async with Client(mcp) as client:
                return await client.call_tool(tool, arguments)

        result = asyncio.run(call_tool())
        assert result.is_error
        assert guidance in result.content[0].text

    def test_unexpected_failures_remain_masked(self):
        from mcp import Client

        async def call_tool():
            async with Client(mcp) as client:
                return await client.call_tool("load_data", {"dataset": "smn"})

        with patch("foehn.mcp_server.foehn.load", side_effect=RuntimeError("private implementation detail")):
            result = asyncio.run(call_tool())
        assert result.is_error
        assert "private implementation detail" not in result.content[0].text

    def test_missing_grid_dependencies_explain_the_install_to_protocol_clients(self):
        from mcp import Client

        async def call_tool():
            async with Client(mcp) as client:
                return await client.call_tool("describe_grid", {"dataset": "surface_derived_grid"})

        guidance = 'Reading gridded data requires optional dependencies. Install pip install "foehn[grids]"'
        with patch("foehn.mcp_server.foehn.open_dataset", side_effect=ImportError(guidance)):
            result = asyncio.run(call_tool())

        assert result.is_error
        assert 'pip install "foehn[grids]"' in result.content[0].text

    def test_fetch_errors_keep_guidance_without_url_credentials(self):
        from mcp import Client

        from foehn.fetch import FetchError

        async def call_tool():
            async with Client(mcp) as client:
                return await client.call_tool("load_data", {"dataset": "smn"})

        error = FetchError(
            "Cannot fetch https://user:password@data.geo.admin.ch/x.csv?token=private-token#private-fragment"
        )
        with patch("foehn.mcp_server.foehn.load", side_effect=error):
            result = asyncio.run(call_tool())

        assert result.is_error
        message = result.content[0].text
        assert "Cannot fetch https://data.geo.admin.ch/x.csv" in message
        assert all(private not in message for private in ("password", "private-token", "private-fragment"))

    def test_response_validation_details_remain_masked(self):
        """Pydantic's ValueError subclass reports malformed server output, not bad query arguments."""
        from mcp import Client

        async def call_tool():
            async with Client(mcp) as client:
                return await client.call_tool("get_parameters", {"dataset": "smn"})

        malformed = pl.DataFrame(
            {
                "shortname": ["tre200d0"],
                "description": ["Temperature"],
                "unit": ["C"],
                "type": ["Float"],
                "granularity": ["d"],
                "decimals": [["private server value"]],
                "group": ["Temperature"],
            }
        )
        with patch("foehn.mcp_server.foehn.parameters", return_value=malformed):
            result = asyncio.run(call_tool())

        assert result.is_error
        assert "private server value" not in result.content[0].text
        assert "Error executing tool" in result.content[0].text


# ── Published input schemas ──────────────────────────────────────────────────


def _schemas() -> dict[str, dict]:
    async def list_tools():
        return await mcp.list_tools()

    return {tool.name: tool.input_schema["properties"] for tool in asyncio.run(list_tools())}


def _call(tool: str, arguments: dict):
    from mcp import Client

    async def call_tool():
        async with Client(mcp) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(call_tool())


def _arm(prop: dict, schema_type: str) -> dict:
    """The arm of a (possibly nullable) parameter schema with this JSON type."""
    return next(arm for arm in prop.get("anyOf", [prop]) if arm.get("type") == schema_type)


class TestInputSchemas:
    def test_every_parameter_carries_its_docstring_description(self):
        """Some clients flatten a nullable parameter's anyOf away; the description survives that."""
        missing = [
            f"{tool}.{name}"
            for tool, props in _schemas().items()
            for name, prop in props.items()
            if not prop.get("description")
        ]
        assert missing == []

    def test_descriptions_are_the_docstring_text(self):
        description = _schemas()["load_data"]["station"]["description"]
        assert description.startswith("Station abbreviation(s)")
        assert "get_stations()" in description  # a continuation line, joined

    def test_vocabularies_are_enumerated_from_their_tables(self):
        props = _schemas()
        assert _arm(props["load_data"]["frequency"], "string")["enum"] == list(GRANULARITY_LABELS)
        assert _arm(props["describe_data"]["time_slice"], "string")["enum"] == list(TIME_SLICE_LABELS)
        assert _arm(props["load_data"]["sort"], "string")["enum"] == ["asc", "desc"]
        assert _arm(props["list_datasets"]["category"], "string")["enum"] == sorted(CATEGORIES)

    def test_ranges_are_declared(self):
        props = _schemas()
        for tool in ("load_data", "get_inventory", "get_points"):
            assert (props[tool]["limit"]["minimum"], props[tool]["limit"]["maximum"]) == (1, 500)
            assert props[tool]["offset"]["minimum"] == 0
        month = _arm(props["load_data"]["month"], "integer")
        assert (month["minimum"], month["maximum"]) == (1, 12)

    @pytest.mark.parametrize(
        ("tool", "arguments"),
        [
            ("load_data", {"dataset": "smn", "limit": 9999}),
            ("load_data", {"dataset": "smn", "offset": -1}),
            ("load_data", {"dataset": "smn", "month": [13]}),
            ("get_inventory", {"dataset": "smn", "limit": 0}),
        ],
    )
    def test_out_of_range_values_are_refused_not_clamped(self, tool, arguments):
        with patch("foehn.mcp_server.foehn.load") as mock_load, patch("foehn.mcp_server.foehn.inventory"):
            result = _call(tool, arguments)
        assert result.is_error
        mock_load.assert_not_called()

    @patch("foehn.mcp_server.foehn.load")
    def test_a_lone_station_string_is_accepted_over_the_protocol(self, mock_load):
        mock_load.return_value = _make_df([{"station_abbr": "BER", "x": 1}])
        result = _call("load_data", {"dataset": "smn", "station": "BER", "columns": "x"})
        assert not result.is_error
        assert mock_load.call_args.kwargs["station"] == "BER"
        assert mock_load.call_args.kwargs["columns"] == ["x"]
        assert result.structured_content["total"] == 1


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
