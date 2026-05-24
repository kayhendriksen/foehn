"""Tests for the gridded read path (foehn.open_dataset)."""

from pathlib import Path
from unittest.mock import patch

import pytest

import foehn
from foehn.grids import _ensure_netcdf_files, open_dataset, to_zarr


def _items_for(*filenames):
    """STAC items whose .nc asset hrefs map to the given (already cached) filenames.

    Used to mock get_collection_items so an unfiltered open finds everything it
    expects already on disk and performs no download.
    """
    return [{"assets": {"d": {"href": f"https://data.geo.admin.ch/x/{name}"}}} for name in filenames]


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


def test_ensure_netcdf_files_unfiltered_consults_remote_and_downloads_missing(tmp_path):
    """An unfiltered call must enumerate remote and fetch files missing from a partial cache."""
    base = tmp_path / "bronze" / "surface_derived_grid"
    base.mkdir(parents=True)
    cached = base / "ogd.rhiresd.nc"
    cached.write_bytes(b"x")  # simulate a prior filtered (match="rhiresd") download

    items = [
        {"assets": {"a": {"href": "https://data.geo.admin.ch/x/ogd.rhiresd.nc"}}},
        {"assets": {"b": {"href": "https://data.geo.admin.ch/x/ogd.tabsd.nc"}}},
    ]

    def fake_download(_session, _href, filepath):
        Path(filepath).write_bytes(b"y")

    with (
        patch("foehn.grids.get_collection_items", return_value=items) as mock_items,
        patch("foehn.client._retry_session"),
        patch("foehn.client._download_binary", side_effect=fake_download) as mock_dl,
    ):
        result = _ensure_netcdf_files("surface_derived_grid", tmp_path / "bronze")

    mock_items.assert_called_once()
    assert mock_dl.call_count == 1  # only the missing file is fetched; cache reused
    assert {p.name for p in result} == {"ogd.rhiresd.nc", "ogd.tabsd.nc"}


def test_ensure_netcdf_files_offline_falls_back_to_cache(tmp_path):
    """If the STAC API is unreachable, fall back to the cached files with a warning."""
    import requests

    base = tmp_path / "bronze" / "surface_derived_grid"
    base.mkdir(parents=True)
    (base / "a.nc").write_bytes(b"x")

    with (
        patch("foehn.grids.get_collection_items", side_effect=requests.exceptions.ConnectionError("offline")),
        pytest.warns(UserWarning, match="may be an incomplete subset"),
    ):
        result = _ensure_netcdf_files("surface_derived_grid", tmp_path / "bronze")
    assert [p.name for p in result] == ["a.nc"]


def test_ensure_netcdf_files_offline_no_cache_reraises(tmp_path):
    """Offline with an empty cache must surface the network error, not swallow it."""
    import requests

    with (
        patch("foehn.grids.get_collection_items", side_effect=requests.exceptions.ConnectionError("offline")),
        pytest.raises(requests.exceptions.ConnectionError),
    ):
        _ensure_netcdf_files("surface_derived_grid", tmp_path / "bronze")


@patch("foehn.grids.get_collection_items", return_value=_items_for("grid.nc"))
def test_open_dataset_reads_local_netcdf(_mock_items, tmp_path):
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


def test_open_netcdf_combines_multiple_files_without_dask(tmp_path):
    """Multiple files combine via combine_by_coords — no dask chunk manager required."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    import numpy as np

    from foehn.grids import _open_netcdf

    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    xr.Dataset({"t": (("x",), np.array([1.0, 2.0], "float32"))}, coords={"x": [0, 1]}).to_netcdf(a, engine="h5netcdf")
    xr.Dataset({"t": (("x",), np.array([3.0, 4.0], "float32"))}, coords={"x": [2, 3]}).to_netcdf(b, engine="h5netcdf")

    ds = _open_netcdf(xr, [a, b], engine=None)
    assert ds["t"].shape == (4,)
    assert list(ds["x"].values) == [0, 1, 2, 3]


def test_open_netcdf_combines_despite_conflicting_global_attrs(tmp_path):
    """Differing global attrs (history/source) must not block an otherwise-clean combine."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    import numpy as np

    from foehn.grids import _open_netcdf

    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    xr.Dataset(
        {"t": (("x",), np.array([1.0, 2.0], "float32"))},
        coords={"x": [0, 1]},
        attrs={"title": "rhiresd", "history": "made monday", "source": "A"},
    ).to_netcdf(a, engine="h5netcdf")
    xr.Dataset(
        {"t": (("x",), np.array([3.0, 4.0], "float32"))},
        coords={"x": [2, 3]},
        attrs={"title": "rhiresd", "history": "made tuesday", "source": "B"},
    ).to_netcdf(b, engine="h5netcdf")

    ds = _open_netcdf(xr, [a, b], engine=None)
    assert ds["t"].shape == (4,)
    # Shared attrs survive; conflicting ones are dropped rather than raising.
    assert ds.attrs.get("title") == "rhiresd"
    assert "history" not in ds.attrs
    assert "source" not in ds.attrs


@patch("foehn.grids.get_collection_items", return_value=_items_for("grid.nc"))
def test_to_zarr_writes_store(_mock_items, tmp_path):
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


def test_to_zarr_match_yields_distinct_stores(tmp_path):
    """Different match= filters must write to distinct stores, not clobber each other."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("zarr")
    import numpy as np

    base = tmp_path / "bronze" / "surface_derived_grid"
    _write_nc(base / "g.rhiresd.nc", xr, np, var="rain")
    _write_nc(base / "g.tabsd.nc", xr, np, var="temp")

    s1 = to_zarr("surface_derived_grid", data_dir=tmp_path, match="rhiresd")
    s2 = to_zarr("surface_derived_grid", data_dir=tmp_path, match="tabsd")

    assert s1.name == "surface_derived_grid__rhiresd.zarr"
    assert s2.name == "surface_derived_grid__tabsd.zarr"
    assert s1 != s2
    assert s1.exists() and s2.exists()
    assert "rain" in xr.open_zarr(s1).data_vars
    assert "temp" in xr.open_zarr(s2).data_vars


@patch("foehn.grids.get_collection_items", return_value=_items_for("g.nc"))
def test_to_zarr_explicit_store_path(_mock_items, tmp_path):
    """An explicit store= path overrides the derived data_dir/zarr/<name>.zarr location."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("zarr")
    import numpy as np

    _write_nc(tmp_path / "bronze" / "surface_derived_grid" / "g.nc", xr, np)

    out = tmp_path / "custom" / "mine.zarr"
    store = to_zarr("surface_derived_grid", data_dir=tmp_path, store=out)
    assert store == out
    assert store.exists()


@patch("foehn.grids.get_collection_items", return_value=_items_for("grid.nc"))
def test_to_zarr_does_not_leak_consolidated_warning(_mock_items, tmp_path, recwarn):
    """to_zarr keeps consolidated metadata but must not surface zarr's out-of-spec warning."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("zarr")
    import numpy as np

    _write_nc(tmp_path / "bronze" / "surface_derived_grid" / "grid.nc", xr, np)

    to_zarr("surface_derived_grid", data_dir=tmp_path)

    messages = [str(w.message) for w in recwarn.list]
    assert not any("Consolidated metadata is currently not part" in m for m in messages), messages


def test_to_zarr_grib2_raises():
    with pytest.raises(NotImplementedError, match="GRIB2/HDF5"):
        to_zarr("forecast_icon_ch1")


def test_to_zarr_rechunk_without_dask_raises(tmp_path):
    """rechunk= must raise a helpful ImportError when dask is unavailable."""
    import importlib.util

    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("zarr")
    import numpy as np

    _write_nc(tmp_path / "bronze" / "surface_derived_grid" / "grid.nc", xr, np)

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        # Pretend dask is not installed, regardless of the test environment.
        return None if name == "dask" else real_find_spec(name, *args, **kwargs)

    with (
        patch("foehn.grids.get_collection_items", return_value=_items_for("grid.nc")),
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
        pytest.raises(ImportError, match="requires dask"),
    ):
        to_zarr("surface_derived_grid", data_dir=tmp_path, rechunk={"x": 1})


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


@patch("foehn.grids.get_collection_items", return_value=_items_for("normals_yearly.nc"))
def test_to_zarr_with_noncf_time_reopens_cleanly(_mock_items, tmp_path):
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
