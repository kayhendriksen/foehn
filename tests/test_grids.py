"""Tests for the gridded read path (foehn.open_dataset)."""

from unittest.mock import patch

import pytest

import foehn
from foehn.grids import _ensure_netcdf_files, open_dataset, to_zarr


def test_open_dataset_is_exported():
    assert callable(foehn.open_dataset)


def test_to_zarr_is_exported():
    assert callable(foehn.to_zarr)


def test_open_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        open_dataset("nonexistent")


def test_open_csv_dataset_raises():
    """Tabular collections should redirect callers to load()."""
    with pytest.raises(ValueError, match="tabular"):
        open_dataset("smn")


def test_open_grib2_dataset_raises():
    with pytest.raises(NotImplementedError, match="GRIB2/HDF5"):
        open_dataset("forecast_icon_ch1")


def test_open_radar_dataset_raises():
    """Radar is HDF5/ODIM — not handled in the NetCDF-only phase."""
    with pytest.raises(NotImplementedError, match="GRIB2/HDF5"):
        open_dataset("radar_precip")


@patch("foehn.grids.get_collection_items")
def test_ensure_netcdf_files_raises_when_no_nc_assets(mock_items, tmp_path):
    """A collection that only exposes GeoTIFF/ZIP should raise a clear error."""
    mock_items.return_value = [
        {"assets": {"a": {"href": "https://data.geo.admin.ch/x/grid.tif"}}},
        {"assets": {"b": {"href": "https://data.geo.admin.ch/x/bundle.zip"}}},
    ]
    with pytest.raises(ValueError, match="No NetCDF"):
        _ensure_netcdf_files("hail_hazard_10y", tmp_path / "bronze")


def test_ensure_netcdf_files_match_filters_local_cache(tmp_path):
    """match should keep only local files whose name contains the substring."""
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    keep = out_dir / "x.rhiresd_ch01h.nc"
    drop = out_dir / "x.ranomm9120_ch01r.nc"
    keep.write_bytes(b"x")
    drop.write_bytes(b"x")

    with patch("foehn.grids.get_collection_items") as mock_items:
        result = _ensure_netcdf_files("surface_derived_grid", tmp_path / "bronze", match="rhiresd")
        mock_items.assert_not_called()
    assert result == [keep]


@patch("foehn.grids.get_collection_items")
def test_ensure_netcdf_files_match_no_remote_match_raises(mock_items, tmp_path):
    mock_items.return_value = [
        {"assets": {"a": {"href": "https://data.geo.admin.ch/x/ranomm9120.nc"}}},
    ]
    with pytest.raises(ValueError, match="matching 'rhiresd'"):
        _ensure_netcdf_files("surface_derived_grid", tmp_path / "bronze", match="rhiresd")


def test_ensure_netcdf_files_uses_local_cache(tmp_path):
    """Existing local .nc files should be returned without any network call."""
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    f = out_dir / "data.nc"
    f.write_bytes(b"not really netcdf")

    with patch("foehn.grids.get_collection_items") as mock_items:
        result = _ensure_netcdf_files("surface_derived_grid", tmp_path / "bronze")
        mock_items.assert_not_called()
    assert result == [f]


def test_open_dataset_reads_local_netcdf(tmp_path):
    """End-to-end: open a real NetCDF from the local cache and select a variable."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    import numpy as np

    ds = xr.Dataset(
        {
            "tas": (("y", "x"), np.arange(6, dtype="float32").reshape(2, 3)),
            "pr": (("y", "x"), np.ones((2, 3), dtype="float32")),
        },
        coords={"y": [0, 1], "x": [0, 1, 2]},
    )
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    ds.to_netcdf(out_dir / "grid.nc", engine="h5netcdf")

    opened = open_dataset("surface_derived_grid", data_dir=tmp_path)
    assert "tas" in opened.data_vars
    assert "pr" in opened.data_vars

    only_tas = open_dataset("surface_derived_grid", data_dir=tmp_path, variables="tas")
    assert "tas" in only_tas.data_vars
    assert "pr" not in only_tas.data_vars


def _write_nc(path, xr, np, var="tas"):
    ds = xr.Dataset(
        {var: (("y", "x"), np.arange(6, dtype="float32").reshape(2, 3))},
        coords={"y": [0, 1], "x": [0, 1, 2]},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="h5netcdf")


def test_open_dataset_match_selects_subset(tmp_path):
    """match should restrict a multi-file collection to the matching files."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    import numpy as np

    base = tmp_path / "bronze" / "surface_derived_grid"
    _write_nc(base / "grid.rhiresd_ch01h.nc", xr, np, var="rain")
    _write_nc(base / "grid.ranomm9120_ch01r.nc", xr, np, var="anom")

    ds = open_dataset("surface_derived_grid", data_dir=tmp_path, match="rhiresd")
    assert "rain" in ds.data_vars
    assert "anom" not in ds.data_vars


def test_to_zarr_writes_store(tmp_path):
    """to_zarr should write a readable Zarr store under data_dir/zarr/."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("zarr")
    import numpy as np

    _write_nc(tmp_path / "bronze" / "surface_derived_grid" / "grid.nc", xr, np)

    store = to_zarr("surface_derived_grid", data_dir=tmp_path)
    assert store == tmp_path / "zarr" / "surface_derived_grid.zarr"
    assert store.exists()

    roundtrip = xr.open_zarr(store)
    assert "tas" in roundtrip.data_vars
    assert roundtrip["tas"].shape == (2, 3)


def test_to_zarr_grib2_raises():
    with pytest.raises(NotImplementedError, match="GRIB2/HDF5"):
        to_zarr("forecast_icon_ch1")


def test_sanitize_noncf_time_units_moves_bad_units():
    """'years since ...' units must be renamed so the store can be re-decoded."""
    xr = pytest.importorskip("xarray")
    import numpy as np

    from foehn.grids import _sanitize_noncf_time_units

    ds = xr.Dataset(
        {"v": (("time",), np.array([1.0]))},
        coords={"time": ("time", np.array([0.0]), {"units": "years since 1991-01-01", "calendar": "standard"})},
    )
    out = _sanitize_noncf_time_units(ds)
    assert "units" not in out["time"].attrs
    assert out["time"].attrs["units_noncf"] == "years since 1991-01-01"
    assert out["time"].attrs["calendar_noncf"] == "standard"


def test_sanitize_keeps_valid_cf_time_units():
    """Well-formed CF units (e.g. 'days since ...') must be left untouched."""
    xr = pytest.importorskip("xarray")
    import numpy as np

    from foehn.grids import _sanitize_noncf_time_units

    ds = xr.Dataset(
        {"v": (("time",), np.array([1.0]))},
        coords={"time": ("time", np.array([0.0]), {"units": "days since 2000-01-01"})},
    )
    out = _sanitize_noncf_time_units(ds)
    assert out["time"].attrs["units"] == "days since 2000-01-01"
    assert "units_noncf" not in out["time"].attrs


def test_to_zarr_with_noncf_time_reopens_cleanly(tmp_path):
    """End-to-end: a NetCDF with non-CF time units must produce a re-openable store."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    import numpy as np

    ds = xr.Dataset(
        {"TnormY": (("time", "y", "x"), np.ones((1, 2, 3), dtype="float32"))},
        coords={
            "time": ("time", np.array([0.0]), {"units": "years since 1991-01-01", "calendar": "standard"}),
            "y": [0, 1],
            "x": [0, 1, 2],
        },
    )
    out_dir = tmp_path / "bronze" / "climate_normals_temp_9120"
    out_dir.mkdir(parents=True)
    # netcdf4 engine preserves the non-CF units attribute on the time axis
    ds.to_netcdf(out_dir / "normals_yearly.nc", engine="netcdf4")

    store = to_zarr("climate_normals_temp_9120", data_dir=tmp_path)
    roundtrip = xr.open_zarr(store)  # default CF decode must not throw
    assert "TnormY" in roundtrip.data_vars
