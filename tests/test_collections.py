"""Tests for collections constants and routing sets."""

from foehn.api import list_datasets
from foehn.collections import (
    COLLECTION_META,
    COLLECTIONS,
    FORECAST_CSV_COLLECTIONS,
    GRIB2_COLLECTIONS,
    NETCDF_COLLECTIONS,
)


def test_collections_keys_are_strings():
    for key, collection_id in COLLECTIONS.items():
        assert isinstance(key, str) and key
        assert isinstance(collection_id, str) and collection_id.startswith("ch.")


def test_routing_sets_are_subsets_of_collections():
    keys = set(COLLECTIONS.keys())
    assert keys >= FORECAST_CSV_COLLECTIONS
    assert keys >= GRIB2_COLLECTIONS
    assert keys >= NETCDF_COLLECTIONS


def test_routing_sets_are_mutually_exclusive():
    """A collection should belong to at most one routing set."""
    all_sets = [FORECAST_CSV_COLLECTIONS, GRIB2_COLLECTIONS, NETCDF_COLLECTIONS]
    for i, a in enumerate(all_sets):
        for b in all_sets[i + 1 :]:
            assert a.isdisjoint(b), f"Overlap between routing sets: {a & b}"


def test_collection_ids_are_unique():
    ids = list(COLLECTIONS.values())
    assert len(ids) == len(set(ids)), "Duplicate STAC collection IDs found"


def test_list_datasets_returns_all():
    rows = list_datasets()
    assert len(rows) == len(COLLECTIONS)


def test_list_datasets_dict_keys():
    expected = {
        "dataset",
        "collection_id",
        "category",
        "subcategory",
        "description",
        "format",
        "frequencies",
        "time_slices",
    }
    for row in list_datasets():
        assert set(row.keys()) == expected


def test_collection_meta_covers_all_keys():
    assert set(COLLECTION_META.keys()) == set(COLLECTIONS.keys())
