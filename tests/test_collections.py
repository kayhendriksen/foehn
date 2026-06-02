"""Tests for collections constants and routing sets."""

from foehn.api import list_datasets
from foehn.collections import (
    COLLECTION_META,
    COLLECTIONS,
    CSV_ZIP_COLLECTIONS,
    FORECAST_CSV_COLLECTIONS,
    GRIB2_COLLECTIONS,
    NETCDF_COLLECTIONS,
    PREAMBLE_CSV_COLLECTIONS,
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
    assert keys >= CSV_ZIP_COLLECTIONS
    assert keys >= PREAMBLE_CSV_COLLECTIONS


def test_routing_sets_are_mutually_exclusive():
    """A collection should belong to at most one routing set."""
    all_sets = [
        FORECAST_CSV_COLLECTIONS,
        GRIB2_COLLECTIONS,
        NETCDF_COLLECTIONS,
        CSV_ZIP_COLLECTIONS,
        PREAMBLE_CSV_COLLECTIONS,
    ]
    for i, a in enumerate(all_sets):
        for b in all_sets[i + 1 :]:
            assert a.isdisjoint(b), f"Overlap between routing sets: {a & b}"


def test_indoor_scenarios_is_csv_zip_not_netcdf():
    assert "climate_scenarios_indoor" in CSV_ZIP_COLLECTIONS
    assert "climate_scenarios_indoor" not in NETCDF_COLLECTIONS
    assert COLLECTION_META["climate_scenarios_indoor"]["format"] == "CSV+ZIP"


def test_climate_scenarios_is_preamble_csv():
    assert "climate_scenarios" in PREAMBLE_CSV_COLLECTIONS
    assert "climate_scenarios" not in NETCDF_COLLECTIONS
    assert "climate_scenarios" not in CSV_ZIP_COLLECTIONS


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


def test_collection_meta_matches_live_stac_granularities():
    expected = {
        "smn": (["t", "h", "d", "m", "y"], ["historical", "recent", "now"]),
        "smn_precip": (["t", "h", "d", "m", "y"], ["historical", "recent", "now"]),
        "smn_tower": (["t", "h", "d", "m", "y"], ["historical", "recent", "now"]),
        "obs": (["d", "m", "y"], ["historical", "recent"]),
        "pollen": (["h", "d", "y"], ["historical", "recent"]),
        "nbcn": (["d", "m", "y"], ["historical", "recent"]),
        "nbcn_precip": (["m", "y"], []),
        "climate_scenarios": (["d"], []),
        "forecast_local": (["h", "d"], []),
    }
    for key, (frequencies, time_slices) in expected.items():
        assert COLLECTION_META[key]["frequencies"] == frequencies
        assert COLLECTION_META[key]["time_slices"] == time_slices


def test_radar_collections_are_hdf5():
    assert COLLECTION_META["radar_precip"]["format"] == "HDF5"
    assert COLLECTION_META["radar_hail"]["format"] == "HDF5"


def test_spatial_climate_analysis_grids_are_netcdf():
    # The OGD spatial climate analyses (hail) and unified normals grid are
    # static NetCDF collections, read via open_dataset/to_zarr like the others.
    for key in ("radar_derived_grid", "climate_normals_grid"):
        assert COLLECTION_META[key]["format"] == "NetCDF"
        assert key in NETCDF_COLLECTIONS
