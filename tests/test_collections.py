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
    forecast_run_from_filename,
    time_slice_from_filename,
)


def test_time_slice_from_filename_detects_trailing_segment():
    assert time_slice_from_filename("ogd-smn_ber_d_recent.csv") == "recent"
    assert time_slice_from_filename("ogd-smn_ber_t_now.csv") == "now"
    assert time_slice_from_filename("ogd-smn_ber_d_historical.csv") == "historical"
    # Works on a full URL too (the filename segment is parsed, not the whole path).
    assert time_slice_from_filename("https://data.geo.admin.ch/x/ogd-smn_ber_d_recent.csv") == "recent"


def test_time_slice_from_filename_detects_decade_split_historical():
    # MeteoSwiss splits the t/h historical series per decade. Parsing these as
    # "no slice" makes every query pull the full history (see #39).
    assert time_slice_from_filename("ogd-smn_ber_t_historical_2000-2009.csv") == "historical"
    assert time_slice_from_filename("ogd-smn_ber_h_historical_1980-1989.csv") == "historical"
    assert time_slice_from_filename("ogd-smn-tower_ban_h_historical_2020-2029.csv") == "historical"
    assert time_slice_from_filename("https://data.geo.admin.ch/x/ogd-smn_ber_t_historical_2010-2019.csv") == (
        "historical"
    )


def test_time_slice_from_filename_only_accepts_a_bare_decade_range():
    # The one-segment-back lookup must not become a general "search anywhere".
    assert time_slice_from_filename("ogd-smn_ber_t_historical_2000.csv") is None
    assert time_slice_from_filename("ogd-smn_ber_t_historical_2000-2009-extra.csv") is None
    assert time_slice_from_filename("ogd-smn_ber_t_notaslice_2000-2009.csv") is None
    # A decade range with no slice in front of it is still unsliced data.
    assert time_slice_from_filename("ogd-smn_ber_t_2000-2009.csv") is None


def test_time_slice_from_filename_returns_none_when_absent():
    assert time_slice_from_filename("ogd-smn_ber_d.csv") is None
    # 'now' as a coincidental substring elsewhere must not be misread as a slice.
    assert time_slice_from_filename("ogd-snow_ber_d.csv") is None
    assert time_slice_from_filename("metadata_meta_parameters.csv") is None


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


def test_collection_meta_declares_expected_frequencies_and_slices():
    # A change-detector over COLLECTION_META, not a check against MeteoSwiss —
    # the old name claimed the latter, which is how pollen's missing "now" slice
    # survived here so long: this table simply repeated the mistake. The real
    # comparison against the live listing is
    # test_declared_time_slices_match_what_meteoswiss_publishes (marked "live").
    expected = {
        "smn": (["t", "h", "d", "m", "y"], ["historical", "recent", "now"]),
        "smn_precip": (["t", "h", "d", "m", "y"], ["historical", "recent", "now"]),
        "smn_tower": (["t", "h", "d", "m", "y"], ["historical", "recent", "now"]),
        "obs": (["d", "m", "y"], ["historical", "recent"]),
        "pollen": (["h", "d", "y"], ["historical", "recent", "now"]),
        "nbcn": (["d", "m", "y"], ["historical", "recent"]),
        "nbcn_precip": (["m", "y"], []),
        # NO_GRANULARITY/CSV_ZIP collections advertise no frequencies: the field
        # doubles as the valid values for load()'s frequency filter, which these
        # datasets don't support (granularity lives in the description instead).
        "climate_scenarios": ([], []),
        "forecast_local": ([], []),
        "climate_scenarios_indoor": ([], []),
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


# --- forecast run timestamps ---


def test_forecast_run_from_filename():
    assert forecast_run_from_filename("vnut12.lssw.202607210600.dkl010h0.csv") == "202607210600"


def test_forecast_run_from_filename_parses_full_url():
    href = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-local-forecasting/20260719-ch/vnut12.lssw.202607191200.fu3q10h1.csv"
    assert forecast_run_from_filename(href) == "202607191200"


def test_forecast_run_from_filename_rejects_non_forecast_names():
    assert forecast_run_from_filename("ogd-smn_ber_d_recent.csv") is None
    assert forecast_run_from_filename("ogd-local-forecasting_meta_point.csv") is None


def test_forecast_runs_sort_lexicographically():
    runs = ["202607210600", "202606300000", "202607192300"]
    assert max(runs) == "202607210600"
