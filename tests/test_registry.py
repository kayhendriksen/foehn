"""Tests for the dataset-kind registry.

This is where the routing lives now, so this is where it is tested. The
equivalents used to be spread across test_api and test_cli as "does download()
call the grib2 handler?" — one assertion per caller per kind, which is the
duplication the registry removes.
"""

import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from foehn import registry
from foehn.collections import COLLECTIONS, KIND_OF, DatasetKind
from foehn.convert import convert_indoor_to_parquet, convert_preamble_to_parquet, convert_to_parquet
from foehn.workspace import Workspace
from tests.fakes import InMemoryFetcher, stac_collection, stac_item


def test_every_kind_has_a_spec():
    """A kind with no spec is a dataset that cannot be routed at all."""
    assert set(registry.KINDS) == set(DatasetKind)


def test_every_dataset_resolves_to_a_spec():
    for dataset in COLLECTIONS:
        assert isinstance(registry.spec(dataset), registry.KindSpec)


def test_unknown_dataset_raises_something_a_caller_can_act_on():
    """It raised a bare KeyError, which is why api wrote this message six times."""
    with pytest.raises(ValueError, match="Unknown dataset"):
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


def test_no_kind_has_two_read_paths():
    """A kind is read as a frame or as a grid, never both.

    ``tabular`` and ``is_grid`` are read off those two adapters rather than set
    beside them, so this is what keeps them meaningful. DIRECT_ZIP has neither:
    it downloads and converts to Parquet but is not reachable from load().
    """
    for kind_, spec in registry.KINDS.items():
        assert not (spec.load is not None and spec.grid is not None), f"{kind_} has two read paths"
        assert spec.tabular is (spec.load is not None)
        assert spec.is_grid is (spec.grid is not None)


def test_direct_zip_is_the_only_kind_with_no_read_path():
    """Guards the exception: a second one would be a kind nobody can get data out of."""
    unreadable = {k for k, spec in registry.KINDS.items() if not spec.tabular and not spec.is_grid}
    assert unreadable == {DatasetKind.DIRECT_ZIP}
    # ...and it still has both write paths, which is why it is a dataset at all.
    assert registry.KINDS[DatasetKind.DIRECT_ZIP].convert is not None


def test_only_the_single_file_kinds_cap_an_open():
    """The cap is what makes match= mandatory, and it differs from the cube's."""
    grib2 = registry.KINDS[DatasetKind.GRIB2_GRID].grid
    radar = registry.KINDS[DatasetKind.RADAR_GRID].grid
    netcdf = registry.KINDS[DatasetKind.NETCDF_GRID].grid

    assert grib2.max_files == 1 and radar.max_files == 1
    assert netcdf.max_files is None  # combines cleanly, so an unfiltered open is fine

    # The cube caps are a different per-kind fact: GRIB2 holds the whole set in
    # memory, radar appends one timestep at a time and wants every file.
    assert grib2.cube_max_files == 1000
    assert radar.cube_max_files is None


def test_only_netcdf_lacks_a_cube_builder():
    """NetCDF needs none — a multi-file match already combines on read."""
    assert registry.KINDS[DatasetKind.NETCDF_GRID].grid.cube is None
    assert registry.KINDS[DatasetKind.GRIB2_GRID].grid.cube is not None
    assert registry.KINDS[DatasetKind.RADAR_GRID].grid.cube is not None


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
    """``columns=`` keeps a different set per kind, and the reader carries it.

    It used to sit on KindSpec, a layer above the reader that produces the frame
    it describes, because the pass that reads it lived in ``api``.
    """
    assert registry.KINDS[DatasetKind.PREAMBLE_CSV].load.key_columns == ("station_abbr", "variable", "gwl", "date")
    assert "period" in registry.KINDS[DatasetKind.ARCHIVE_CSV].load.key_columns
    assert registry.KINDS[DatasetKind.STANDARD_CSV].load.key_columns == ("station_abbr", "reference_timestamp")


def test_only_the_nominal_date_kind_sorts_on_something_else():
    """The preamble kind's dates are 0001..0030 strings, not timestamps."""
    assert registry.KINDS[DatasetKind.PREAMBLE_CSV].load.sort_column == "date"
    assert registry.KINDS[DatasetKind.STANDARD_CSV].load.sort_column == "reference_timestamp"


def test_registry_load_returns_a_finished_frame():
    """sort, columns and limit are applied here; ``api.load`` used to do it after."""
    from foehn.readers import Filters

    body = b"station_abbr;reference_timestamp;tre200d0;other\nBER;2026-01-01;1.0;9\nBER;2026-01-02;2.0;9\n"
    fake = _fetcher([stac_item("BER", "https://data.geo.admin.ch/ogd-smn_ber_d_recent.csv")], body=body)

    df = registry.load("smn", Filters.build(columns=["tre200d0"], sort="desc", limit=1), fetcher=fake)

    assert df.columns == ["station_abbr", "reference_timestamp", "tre200d0"]  # key columns kept
    assert len(df) == 1
    assert df["tre200d0"][0] == 2.0  # newest first


# --- dataset listings ---


def test_tabular_and_grid_datasets_are_disjoint():
    tabular, grid = set(registry.tabular_datasets()), set(registry.grid_datasets())
    assert not tabular & grid
    # climate_normals is in neither: it converts to Parquet but has no reader.
    assert set(COLLECTIONS) - tabular - grid == {"climate_normals"}


def test_non_grid_datasets_is_everything_but_the_grids():
    assert set(registry.non_grid_datasets()) == set(COLLECTIONS) - set(registry.grid_datasets())
    assert "climate_normals" in registry.non_grid_datasets()


def test_listings_keep_declaration_order():
    """Order is what the CLI prints and what the MCP tool advertises."""
    assert registry.tabular_datasets() == [k for k in COLLECTIONS if k in registry.tabular_datasets()]


# --- dispatch ---


def _fetcher(items=(), collection=None, body=b"a;b\n1;2\n"):
    fake = InMemoryFetcher()
    fake.any_items = list(items)
    fake.any_collection = collection if collection is not None else {"assets": {}}
    fake.default_body = body
    return fake


@pytest.mark.parametrize(
    ("dataset", "href", "written"),
    [
        ("smn", "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv", "ogd-smn_tst_d_recent.csv"),
        ("forecast_icon_ch1", "https://data.geo.admin.ch/f.grib2", "f.grib2"),
        ("radar_precip", "https://data.geo.admin.ch/cpc26130000000.h5", "cpc26130000000.h5"),
        ("surface_derived_grid", "https://data.geo.admin.ch/grid.nc", "grid.nc"),
    ],
)
def test_download_uses_each_kinds_own_listing_configuration(dataset, href, written, tmp_path):
    """One engine, configured per row — asserted on what lands, not on who was called."""
    fake = _fetcher([stac_item("i1", href)])
    result = registry.download(dataset, Workspace(tmp_path), fetcher=fake)

    assert (tmp_path / "bronze" / dataset / written).exists()
    assert result.filenames == [written]


@pytest.mark.parametrize(
    ("dataset", "ignored"),
    [
        ("forecast_icon_ch1", "https://data.geo.admin.ch/cpc26130000000.h5"),
        ("radar_precip", "https://data.geo.admin.ch/f.grib2"),
    ],
)
def test_each_binary_kind_downloads_only_its_own_format(dataset, ignored, tmp_path):
    """Radar and GRIB2 shared one suffix list while they shared one handler."""
    fake = _fetcher([stac_item("i1", ignored)])
    result = registry.download(dataset, Workspace(tmp_path), fetcher=fake)

    assert result.downloaded == 0
    assert fake.streams == []


def test_the_csv_kinds_fetch_collection_metadata_in_the_same_pass(tmp_path):
    """Both callers used to do this pairing themselves, and counted it differently."""
    meta = "https://data.geo.admin.ch/ogd-smn_meta_stations.csv"
    data = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fetcher([stac_item("i1", data)], collection=stac_collection("ch.meteoschweiz.ogd-smn", meta))

    result = registry.download("smn", Workspace(tmp_path), fetcher=fake)

    assert result.total_assets == 2
    assert result.downloaded == 2
    assert sorted(result.filenames) == ["ogd-smn_meta_stations.csv", "ogd-smn_tst_d_recent.csv"]


def test_the_grid_kinds_fetch_no_collection_metadata(tmp_path):
    """Only the CSV kinds ship parameter/station/inventory files."""
    fake = _fetcher([stac_item("i1", "https://data.geo.admin.ch/grid.nc")])
    registry.download("surface_derived_grid", Workspace(tmp_path), fetcher=fake)
    assert fake.collection_calls == []


def test_netcdf_download_refreshes_the_same_restated_asset_as_open_dataset(tmp_path):
    path = tmp_path / "bronze" / "surface_derived_grid" / "grid.nc"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"old")
    os.utime(path, (0, 0))
    href = "https://data.geo.admin.ch/grid.nc"
    fake = _fetcher([stac_item("i1", href, updated="2100-01-01T00:00:00Z")], body=b"new")

    registry.download("surface_derived_grid", Workspace(tmp_path), fetcher=fake)

    assert path.read_bytes() == b"new"


def test_download_defaults_to_the_recent_slice(tmp_path):
    base = "https://data.geo.admin.ch"
    fake = _fetcher(
        [
            stac_item("i1", f"{base}/ogd-smn_tst_d_recent.csv"),
            stac_item("i2", f"{base}/ogd-smn_tst_d_historical.csv"),
        ]
    )
    result = registry.download("smn", Workspace(tmp_path), fetcher=fake)
    assert result.filenames == ["ogd-smn_tst_d_recent.csv"]


def test_only_the_forecast_kind_narrows_to_the_newest_run(tmp_path):
    """A forecast item is one day of runs; the newest run bounds the set, not the newest item."""
    base = "https://data.geo.admin.ch"
    fake = _fetcher(
        [
            stac_item("d1", f"{base}/vnut12.lssw.202607210600.dkl010h0.csv"),
            stac_item("d2", f"{base}/vnut12.lssw.202607211200.dkl010h0.csv"),
        ]
    )
    result = registry.download("forecast_local", Workspace(tmp_path), fetcher=fake)
    assert result.filenames == ["vnut12.lssw.202607211200.dkl010h0.csv"]


def test_only_the_ephemeral_kinds_stop_at_the_first_page(tmp_path):
    """Forecast and radar hold thousands of items; walking them all is pure cost."""
    fake = _fetcher([stac_item("i1", "https://data.geo.admin.ch/f.grib2")])
    registry.download("forecast_icon_ch1", Workspace(tmp_path), fetcher=fake)
    assert [max_items for *_, max_items in fake.listings] == [100]

    fake = _fetcher([stac_item("i1", "https://data.geo.admin.ch/grid.nc")])
    registry.download("surface_derived_grid", Workspace(tmp_path), fetcher=fake)
    assert [max_items for *_, max_items in fake.listings] == [None]


@pytest.mark.parametrize(
    ("dataset", "converter"),
    [
        ("smn", convert_to_parquet),
        ("forecast_local", convert_to_parquet),
        ("climate_scenarios", convert_preamble_to_parquet),
        ("climate_scenarios_indoor", convert_indoor_to_parquet),
    ],
)
def test_convert_reaches_the_right_converter(dataset, converter):
    """Which converter a kind uses is its row. The registry stopped wrapping them
    once they all took the same three arguments."""
    assert registry.spec(dataset).convert is converter


@pytest.mark.parametrize("dataset", ["surface_derived_grid", "forecast_icon_ch1", "radar_precip"])
def test_convert_is_a_no_op_for_grids(dataset, tmp_path):
    """Callers used to `continue` past these; now there is nothing to skip."""
    assert registry.convert(dataset, Workspace(tmp_path)) == 0


def test_convert_passes_the_directories_through(tmp_path):
    """The dataset key names both the bronze sub-folder read and the output written."""
    bronze = tmp_path / "bronze" / "smn"
    bronze.mkdir(parents=True)
    (bronze / "ogd-smn_tst_d_recent.csv").write_text("station_abbr;value\nTST;1.0\n", encoding="utf-8")

    failures = registry.convert("smn", Workspace(tmp_path))

    assert failures == 0
    assert (tmp_path / "parquet" / "smn" / "smn_d_recent.parquet").exists()


def test_registry_delegates_the_complete_zarr_recipe_to_the_grid_reader():
    with patch("foehn.grids.GridReader.write_store") as write_store:
        registry.write_zarr(
            "surface_derived_grid",
            Path("unused.zarr"),
            match="rhiresd",
            stack=True,
            workspace=Workspace(Path("unused")),
            fetcher=InMemoryFetcher(),
        )

    assert write_store.call_count == 1


def test_write_zarr_refuses_a_tabular_dataset():
    """The row is what says whether a dataset has a Zarr path at all."""
    with pytest.raises(ValueError, match=re.escape("Use foehn.load")):
        registry.write_zarr(
            "smn",
            Path("unused.zarr"),
            workspace=Workspace(Path("unused")),
            fetcher=InMemoryFetcher(),
        )
