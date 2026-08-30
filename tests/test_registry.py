"""Tests for the dataset-kind registry.

This is where the routing lives now, so this is where it is tested. The
equivalents used to be spread across test_api and test_cli as "does download()
call the grib2 handler?" — one assertion per caller per kind, which is the
duplication the registry removes.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from foehn import registry
from foehn.client import DownloadResult
from foehn.collections import COLLECTIONS, KIND_OF, DatasetKind


def test_every_kind_has_a_spec():
    """A kind with no spec is a dataset that cannot be routed at all."""
    assert set(registry.KINDS) == set(DatasetKind)


def test_every_dataset_resolves_to_a_spec():
    for dataset in COLLECTIONS:
        assert isinstance(registry.spec(dataset), registry.KindSpec)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        registry.spec("nonexistent")


# --- what each kind can do ---


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("smn", DatasetKind.STANDARD_CSV),
        ("climate_scenarios", DatasetKind.PREAMBLE_CSV),
        ("climate_scenarios_indoor", DatasetKind.ARCHIVE_CSV),
        ("forecast_local", DatasetKind.FORECAST_CSV),
        ("surface_derived_grid", DatasetKind.NETCDF_GRID),
        ("forecast_icon_ch1", DatasetKind.GRIB2_GRID),
        ("radar_precip", DatasetKind.RADAR_GRID),
    ],
)
def test_representative_datasets_map_to_their_kind(dataset, expected):
    assert KIND_OF[dataset] is expected


def test_grid_kinds_have_no_parquet_path():
    """Not a branch every caller re-derives — a fact the kind carries."""
    for kind_ in (DatasetKind.NETCDF_GRID, DatasetKind.GRIB2_GRID, DatasetKind.RADAR_GRID):
        assert registry.KINDS[kind_].convert is None
        assert registry.KINDS[kind_].tabular is False


def test_tabular_kinds_all_convert():
    for kind_, spec in registry.KINDS.items():
        if spec.tabular:
            assert spec.convert is not None, f"{kind_} is tabular but has no converter"


def test_only_standard_csv_supports_granularity():
    """The other CSV kinds' filenames carry no granularity segment to filter on."""
    supported = {k for k, spec in registry.KINDS.items() if spec.supports_granularity}
    assert supported == {DatasetKind.STANDARD_CSV}


def test_preamble_csv_rejects_calendar_filters():
    """Its dates are nominal (0001..0030), so a calendar filter matches nothing."""
    assert registry.KINDS[DatasetKind.PREAMBLE_CSV].supports_calendar_filters is False
    assert registry.KINDS[DatasetKind.STANDARD_CSV].supports_calendar_filters is True


def test_key_columns_differ_where_the_schema_does():
    """``columns=`` keeps a different set per kind; it used to be passed in by hand."""
    assert registry.KINDS[DatasetKind.PREAMBLE_CSV].key_columns == ("station_abbr", "variable", "gwl", "date")
    assert "period" in registry.KINDS[DatasetKind.ARCHIVE_CSV].key_columns
    assert registry.KINDS[DatasetKind.STANDARD_CSV].key_columns == ("station_abbr", "reference_timestamp")


# --- dataset listings ---


def test_tabular_and_grid_datasets_partition_the_collections():
    assert set(registry.tabular_datasets()) | set(registry.grid_datasets()) == set(COLLECTIONS)
    assert not set(registry.tabular_datasets()) & set(registry.grid_datasets())


def test_listings_keep_declaration_order():
    """Order is what the CLI prints and what the MCP tool advertises."""
    assert registry.tabular_datasets() == [k for k in COLLECTIONS if k in registry.tabular_datasets()]


# --- dispatch ---


@pytest.mark.parametrize(
    ("dataset", "handler"),
    [
        ("smn", "download_collection"),
        ("forecast_local", "download_collection"),
        ("climate_scenarios", "download_collection"),
        ("climate_scenarios_indoor", "download_climate_scenarios_indoor"),
        ("forecast_icon_ch1", "download_grib2"),
        ("radar_precip", "download_grib2"),
        ("surface_derived_grid", "download_netcdf"),
    ],
)
def test_download_reaches_the_right_handler(dataset, handler, tmp_path):
    with patch(f"foehn.registry.{handler}") as mock, patch("foehn.registry.download_metadata") as mock_meta:
        mock.return_value = DownloadResult()
        mock_meta.return_value = DownloadResult()
        registry.download(dataset, tmp_path, fetcher=object())
    assert mock.called


def test_standard_download_sums_metadata_and_collection(tmp_path):
    """Both callers used to do this pairing themselves, and counted it differently."""
    with (
        patch("foehn.registry.download_metadata") as mock_meta,
        patch("foehn.registry.download_collection") as mock_coll,
    ):
        mock_meta.return_value = DownloadResult(total_assets=1, downloaded=1, filenames=["m.csv"])
        mock_coll.return_value = DownloadResult(total_assets=2, downloaded=2, failed=1, filenames=["a.csv", "b.csv"])

        result = registry.download("smn", tmp_path, fetcher=object())

    assert result.total_assets == 3
    assert result.downloaded == 3
    assert result.failed == 1
    assert result.filenames == ["m.csv", "a.csv", "b.csv"]


def test_download_defaults_to_the_recent_slice(tmp_path):
    with patch("foehn.registry.download_collection") as mock_coll, patch("foehn.registry.download_metadata") as m:
        mock_coll.return_value = m.return_value = DownloadResult()
        registry.download("smn", tmp_path, fetcher=object())
    assert mock_coll.call_args.kwargs["data_types"] == ["recent"]


@pytest.mark.parametrize(
    ("dataset", "converter"),
    [
        ("smn", "convert_to_parquet"),
        ("forecast_local", "convert_to_parquet"),
        ("climate_scenarios", "convert_climate_scenarios_to_parquet"),
        ("climate_scenarios_indoor", "convert_climate_scenarios_indoor_to_parquet"),
    ],
)
def test_convert_reaches_the_right_converter(dataset, converter, tmp_path):
    with patch(f"foehn.registry.{converter}") as mock:
        mock.return_value = 0
        registry.convert(dataset, tmp_path / "bronze", tmp_path / "parquet")
    assert mock.called


@pytest.mark.parametrize("dataset", ["surface_derived_grid", "forecast_icon_ch1", "radar_precip"])
def test_convert_is_a_no_op_for_grids(dataset, tmp_path):
    """Callers used to `continue` past these; now there is nothing to skip."""
    assert registry.convert(dataset, tmp_path / "bronze", tmp_path / "parquet") == 0


def test_convert_passes_the_directories_through(tmp_path):
    with patch("foehn.registry.convert_to_parquet") as mock:
        mock.return_value = 3
        failures = registry.convert("smn", Path("bronze"), Path("parquet"))
    mock.assert_called_once_with("smn", Path("bronze"), Path("parquet"))
    assert failures == 3
