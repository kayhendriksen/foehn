"""Tests for asset parsing and selection.

The rules these cover used to live inline in the download and load paths — most
of them in both, which is why the two could disagree about which forecast run
you got. They belong to one module now, and so do their tests.
"""

import pytest

from foehn.assets import (
    Asset,
    assets_of,
    collection_assets,
    hrefs,
    latest_run_of,
    other_extensions,
    select,
)

BASE = "https://data.geo.admin.ch/x"


def _item(item_id, *names, updated="2026-01-01T00:00:00Z", **asset_extra):
    return {
        "id": item_id,
        "properties": {"updated": updated},
        "assets": {n: {"href": f"{BASE}/{n}", **asset_extra} for n in names},
    }


# --- parsing -------------------------------------------------------------


def test_asset_parses_the_filename_facts():
    (asset,) = assets_of([_item("ber", "ogd-smn_ber_d_recent.csv")])

    assert asset.name == "ogd-smn_ber_d_recent.csv"
    assert asset.time_slice == "recent"
    assert asset.granularity == "d"
    assert asset.forecast_run is None
    assert asset.item_id == "ber"


def test_href_keeps_its_query_string_but_the_name_does_not():
    """A CSV served with a token must still be recognised — and saved unsuffixed.

    ``href.endswith(".csv")`` on the raw href silently dropped these.
    """
    items = [{"id": "ber", "properties": {}, "assets": {"a": {"href": f"{BASE}/ogd-smn_ber_d_recent.csv?token=abc"}}}]

    (asset,) = assets_of(items, suffixes=(".csv",))

    assert asset.href.endswith("?token=abc")  # what gets fetched
    assert asset.name == "ogd-smn_ber_d_recent.csv"  # what gets written
    assert asset.time_slice == "recent"


def test_decade_split_historical_keeps_its_slice():
    """The t/h historical series are split per decade, so the slice sits one
    segment further back. Reading it as "no slice at all" makes every query
    include the full history."""
    (asset,) = assets_of([_item("ber", "ogd-smn_ber_t_historical_2000-2009.csv")])

    assert asset.time_slice == "historical"
    assert asset.granularity == "t"


def test_forecast_filenames_carry_a_run_and_no_slice():
    (asset,) = assets_of([_item("d", "vnut12.lssw.202607210600.dkl010h0.csv")])

    assert asset.forecast_run == "202607210600"
    assert asset.time_slice is None
    assert asset.granularity is None


def test_per_asset_updated_wins_over_the_item():
    """STAC allows either; the binary paths compare it against local mtime."""
    items = [
        {
            "id": "f",
            "properties": {"updated": "2026-01-01T00:00:00Z"},
            "assets": {"a": {"href": f"{BASE}/f.grib2", "updated": "2026-06-01T00:00:00Z"}},
        }
    ]

    (asset,) = assets_of(items)

    assert asset.updated == "2026-06-01T00:00:00Z"


def test_item_updated_is_the_fallback():
    (asset,) = assets_of([_item("f", "f.grib2", updated="2026-01-01T00:00:00Z")])

    assert asset.updated == "2026-01-01T00:00:00Z"


# --- narrowing by name ---------------------------------------------------


def test_suffixes_filter():
    items = [_item("g", "grid.nc", "grid.tif", "readme.pdf")]

    assert {a.name for a in assets_of(items, suffixes=(".nc", ".tif"))} == {"grid.nc", "grid.tif"}


def test_contains_and_excludes():
    items = [_item("c", "ogd-cs_abe_pr_gwl1.5.csv", "ogd-cs_meta_parameters.csv")]

    assert [a.name for a in assets_of(items, excludes="_meta_")] == ["ogd-cs_abe_pr_gwl1.5.csv"]
    assert [a.name for a in assets_of(items, contains="_meta_")] == ["ogd-cs_meta_parameters.csv"]


def test_other_extensions_reports_what_a_collection_does_hold():
    """Feeds the "available asset types" hint when a match finds nothing."""
    assert other_extensions([_item("g", "grid.tif", "bundle.zip")]) == {".tif", ".zip"}


# --- selection -----------------------------------------------------------


def test_time_slice_selection_keeps_unsliced_assets():
    """A file with no slice segment is unsliced data every query includes."""
    items = [_item("ber", "ogd-smn_ber_d_recent.csv", "ogd-smn_ber_d_historical.csv", "ogd-tot_ber.csv")]

    kept = {a.name for a in select(assets_of(items, suffixes=(".csv",)), time_slices=["recent"])}

    assert kept == {"ogd-smn_ber_d_recent.csv", "ogd-tot_ber.csv"}


def test_granularity_selection_drops_assets_without_one():
    """Unlike time slice: a file with no granularity cannot satisfy frequency=."""
    items = [_item("ber", "ogd-smn_ber_d_recent.csv", "ogd-smn_ber_t_recent.csv", "ogd-tot_ber.csv")]

    kept = {a.name for a in select(assets_of(items, suffixes=(".csv",)), granularities=["d"])}

    assert kept == {"ogd-smn_ber_d_recent.csv"}


def test_latest_run_keeps_only_the_newest():
    """One run is ~32 files at ~30 MB; the retained window is ~40 runs."""
    items = [
        _item("20260721-ch", "vnut12.lssw.202607210400.dkl010h0.csv", "vnut12.lssw.202607210400.fu3010h0.csv"),
        _item("20260721-ch", "vnut12.lssw.202607210600.dkl010h0.csv", "vnut12.lssw.202607210600.fu3010h0.csv"),
    ]

    kept = select(assets_of(items, suffixes=(".csv",)), latest_run=True)

    assert {a.forecast_run for a in kept} == {"202607210600"}
    assert len(kept) == 2  # both params of that run, not just one


def test_latest_run_ignores_an_empty_newest_item():
    """The newest day is created at ~04:00 UTC and filled as its runs publish.

    Selecting the newest *item* therefore returned zero CSVs — issue #27. The
    run comes from the filenames, so an empty item contributes nothing.
    """
    items = [
        _item("20260721-ch", "vnut12.lssw.202607210600.dkl010h0.csv"),
        _item("20260722-ch"),  # newest day, not yet populated
    ]

    kept = select(assets_of(items, suffixes=(".csv",)), latest_run=True)

    assert [a.forecast_run for a in kept] == ["202607210600"]


def test_latest_run_is_a_no_op_without_runs():
    """Applied unconditionally to a non-forecast listing, it must change nothing."""
    assets = assets_of([_item("ber", "ogd-smn_ber_d_recent.csv")], suffixes=(".csv",))

    assert select(assets, latest_run=True) == assets


def test_latest_run_of_reports_the_run():
    items = [_item("d", "vnut12.lssw.202607210400.x.csv", "vnut12.lssw.202607210600.x.csv")]

    assert latest_run_of(assets_of(items)) == "202607210600"
    assert latest_run_of(assets_of([_item("ber", "ogd-smn_ber_d_recent.csv")])) is None


def test_selection_composes():
    """What load() does: slice, granularity and run in one pass."""
    items = [_item("ber", "ogd-smn_ber_d_recent.csv", "ogd-smn_ber_t_recent.csv", "ogd-smn_ber_d_historical.csv")]

    kept = select(assets_of(items, suffixes=(".csv",)), time_slices=["recent"], granularities=["d"])

    assert [a.name for a in kept] == ["ogd-smn_ber_d_recent.csv"]


def test_hrefs_returns_what_gets_fetched():
    items = [_item("ber", "ogd-smn_ber_d_recent.csv")]

    assert hrefs(assets_of(items)) == [f"{BASE}/ogd-smn_ber_d_recent.csv"]


# --- collection-level assets ---------------------------------------------


def test_collection_assets_match_by_name():
    collection = {"assets": {"params": {"href": f"{BASE}/ogd-smn_meta_parameters.csv"}}}

    (asset,) = collection_assets(collection, suffixes=(".csv",), contains="_meta_parameters")

    assert asset.name == "ogd-smn_meta_parameters.csv"


def test_collection_assets_match_by_key():
    """The GRIB2 constants file is identified by its key, not its filename."""
    collection = {
        "assets": {
            "horizontal_constants_icon-ch1-eps.grib2": {"href": f"{BASE}/hc.grib2"},
            "params_icon-ch1-eps.csv": {"href": f"{BASE}/params.csv"},
        }
    }

    (asset,) = collection_assets(collection, key_contains="horizontal_constants")

    assert asset.name == "hc.grib2"


def test_collection_with_no_assets_yields_nothing():
    assert collection_assets({}, suffixes=(".csv",)) == []


# --- shape ---------------------------------------------------------------


def test_assets_are_hashable_and_comparable():
    """Frozen, so callers can dedupe or diff listings without care."""
    a, b = assets_of([_item("ber", "x.csv")]), assets_of([_item("ber", "x.csv")])

    assert a == b
    assert len({*a, *b}) == 1


@pytest.mark.parametrize(
    ("name", "expected"),
    [("grid.nc", ".nc"), ("archive.ZIP", ".zip"), ("noext", "")],
)
def test_extension_is_normalised(name, expected):
    (asset,) = assets_of([_item("i", name)])

    assert asset.extension == expected


def test_missing_href_does_not_blow_up():
    """A malformed asset should be inert, not an AttributeError mid-listing."""
    items = [{"id": "i", "properties": {}, "assets": {"broken": {}}}]

    (asset,) = assets_of(items)

    assert asset.href == ""
    assert asset.name == ""


def test_asset_from_stac_defaults():
    asset = Asset.from_stac("k", {"href": f"{BASE}/ogd-smn_ber_d_now.csv"})

    assert asset.key == "k"
    assert asset.item_id == ""
    assert asset.updated == ""
    assert asset.time_slice == "now"
