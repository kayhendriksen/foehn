"""Tests for the gridded read path (foehn.open_dataset)."""

import contextlib
from unittest.mock import patch

import pytest
from conftest import write_odim_composite

import foehn
from foehn.api import open_dataset, to_zarr
from foehn.workspace import Workspace
from tests.fakes import InMemoryFetcher


def _items_for(*filenames):
    """STAC items whose .nc asset hrefs map to the given (already cached) filenames.

    Wired into a fetcher so an unfiltered open finds everything it expects
    already on disk and performs no download.
    """
    return [{"assets": {"d": {"href": f"https://data.geo.admin.ch/x/{name}"}}} for name in filenames]


def _fake(items=None, *, body=b"x"):
    """A fetcher listing *items*, serving *body* for every asset."""
    fake = InMemoryFetcher()
    fake.any_items = items if items is not None else []
    fake.default_body = body
    return fake


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


def test_open_grib2_without_match_raises():
    """GRIB2 collections are huge multi-file sets — open_dataset requires match=."""
    with pytest.raises(ValueError, match="match="):
        open_dataset("forecast_icon_ch1")


def test_open_radar_without_match_raises():
    """Radar is per-5-min single-file — open_dataset requires a narrowing match."""
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    with pytest.raises(ValueError, match="match="):
        open_dataset("radar_precip")


def test_open_dataset_reads_local_netcdf(fetcher, tmp_path):
    """End-to-end: open a real NetCDF from the local cache and select a variable."""
    fetcher.any_items = _items_for("grid.nc")
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


def _write_grib2(path, shortname="2t", step=0, datatime=0):
    """Write a tiny 2x3 regular lat/lon GRIB2 message via eccodes (offline fixture).

    ``step`` (lead time, hours) and ``datatime`` (reference HHMM) let a set of
    files differ along the forecast axes so they can be combined into a cube.
    """
    import eccodes
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    gid = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    eccodes.codes_set(gid, "Ni", 3)
    eccodes.codes_set(gid, "Nj", 2)
    eccodes.codes_set(gid, "latitudeOfFirstGridPointInDegrees", 47.0)
    eccodes.codes_set(gid, "longitudeOfFirstGridPointInDegrees", 7.0)
    eccodes.codes_set(gid, "latitudeOfLastGridPointInDegrees", 46.0)
    eccodes.codes_set(gid, "longitudeOfLastGridPointInDegrees", 9.0)
    eccodes.codes_set(gid, "iDirectionIncrementInDegrees", 1.0)
    eccodes.codes_set(gid, "jDirectionIncrementInDegrees", 1.0)
    eccodes.codes_set(gid, "shortName", shortname)
    eccodes.codes_set(gid, "dataTime", datatime)
    eccodes.codes_set(gid, "endStep", step)
    eccodes.codes_set_values(gid, np.arange(6, dtype=float))
    with open(path, "wb") as f:
        eccodes.codes_write(gid, f)
    eccodes.codes_release(gid)


def test_open_dataset_reads_grib2(fetcher, tmp_path):
    """End-to-end: read a real GRIB2 file from the cache via cfgrib (match required)."""
    fetcher.any_items = _items_for("icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2")
    pytest.importorskip("xarray")
    pytest.importorskip("cfgrib")
    pytest.importorskip("eccodes")

    base = tmp_path / "bronze" / "forecast_icon_ch1"
    _write_grib2(base / "icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2")

    # listing confirms the match is unique; the cached file is reused (no .idx sidecar).
    ds = open_dataset("forecast_icon_ch1", data_dir=tmp_path, match="202605231500-0-t_2m-ctrl")
    assert "t2m" in ds.data_vars
    assert ds["t2m"].shape == (2, 3)
    assert not list(base.glob("*.idx"))


def test_open_grib2_overbroad_match_refused_before_download(fetcher, tmp_path):
    """A GRIB2 match resolving to >1 file is refused up front, with no download."""
    items = [
        {"assets": {"a": {"href": "https://data.geo.admin.ch/x/icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2"}}},
        {"assets": {"b": {"href": "https://data.geo.admin.ch/x/icon-ch1-eps-202605231500-1-t_2m-ctrl.grib2"}}},
    ]
    fetcher.any_items = items
    with pytest.raises(ValueError, match="one file at a time"):
        open_dataset("forecast_icon_ch1", data_dir=tmp_path, match="t_2m-ctrl")
    assert fetcher.streams == []


def test_to_zarr_grib2_writes_store(fetcher, tmp_path):
    """to_zarr works for GRIB2 and encodes the (required) match in the store name."""
    fetcher.any_items = _items_for("icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2")
    pytest.importorskip("xarray")
    pytest.importorskip("cfgrib")
    pytest.importorskip("eccodes")
    pytest.importorskip("zarr")
    import xarray as xr

    base = tmp_path / "bronze" / "forecast_icon_ch1"
    _write_grib2(base / "icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2")

    store = to_zarr("forecast_icon_ch1", data_dir=tmp_path, match="t_2m-ctrl")
    assert store.name == "forecast_icon_ch1__t_2m_ctrl.zarr"
    assert store.exists()
    assert "t2m" in xr.open_zarr(store).data_vars


def test_to_zarr_grib2_hypercube(fetcher, tmp_path):
    """stack= promotes the varying forecast axes (time, step) into one cube."""
    pytest.importorskip("xarray")
    pytest.importorskip("cfgrib")
    pytest.importorskip("eccodes")
    pytest.importorskip("zarr")
    fetcher.any_items = _items_for(
        "icon-ch1-eps-202605230000-0-t_2m-ctrl.grib2",
        "icon-ch1-eps-202605230000-6-t_2m-ctrl.grib2",
        "icon-ch1-eps-202605230600-0-t_2m-ctrl.grib2",
        "icon-ch1-eps-202605230600-6-t_2m-ctrl.grib2",
    )
    import xarray as xr

    base = tmp_path / "bronze" / "forecast_icon_ch1"
    for dt in (0, 600):
        for st in (0, 6):
            _write_grib2(base / f"icon-ch1-eps-20260523{dt:04d}-{st}-t_2m-ctrl.grib2", step=st, datatime=dt)

    store = to_zarr("forecast_icon_ch1", data_dir=tmp_path, match="t_2m-ctrl", stack=True)
    cube = xr.open_zarr(store)
    assert "time" in cube.dims and "step" in cube.dims
    assert cube.sizes["time"] == 2 and cube.sizes["step"] == 2
    assert cube["t2m"].dims[-2:] == ("latitude", "longitude")  # spatial dims preserved


def test_to_zarr_grib2_stack_requires_match(tmp_path):
    """A GRIB2 cube needs a match to scope it."""
    pytest.importorskip("cfgrib")
    with pytest.raises(ValueError, match="needs match="):
        to_zarr("forecast_icon_ch1", data_dir=tmp_path, stack=True)


def test_to_zarr_netcdf_stack_writes_combined(fetcher, tmp_path):
    """NetCDF has no cube builder, so stack= writes the already-combined match."""
    fetcher.any_items = _items_for("g.rain.nc", "g.anom.nc")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("zarr")
    import numpy as np

    base = tmp_path / "bronze" / "surface_derived_grid"
    _write_nc(base / "g.rain.nc", xr, np, var="rain")
    _write_nc(base / "g.anom.nc", xr, np, var="anom")

    store = to_zarr("surface_derived_grid", data_dir=tmp_path, match="g.", stack=True)
    ds = xr.open_zarr(store)
    assert "rain" in ds.data_vars
    assert "anom" in ds.data_vars  # both files combined into one store


def test_open_dataset_reads_radar(fetcher, tmp_path):
    """End-to-end: read an ODIM COMP composite — scaling, masking, LV95 coords."""
    fetcher.any_items = _items_for("cpc2613000000_00060.001.h5")
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    import numpy as np

    base = tmp_path / "bronze" / "radar_precip"
    write_odim_composite(base / "cpc2613000000_00060.001.h5")

    ds = open_dataset("radar_precip", data_dir=tmp_path, match="cpc2613000000")
    assert "acrr" in ds.data_vars
    assert ds["acrr"].dims == ("y", "x")
    assert dict(ds.sizes) == {"y": 2, "x": 3}

    a = ds["acrr"].values
    assert np.isnan(a[0, 2])  # nodata -> NaN
    assert a[1, 0] == 0.0  # undetect -> 0
    assert a[0, 1] == 1.5  # scaled value (gain=1, offset=0)

    # Swiss LV95 cell-centre coords, 1 km spacing, north-to-south rows.
    assert float(ds.x[1] - ds.x[0]) == 1000.0
    assert float(ds.y[1] - ds.y[0]) == -1000.0
    assert 2_250_000 < float(ds.x[0]) < 2_300_000
    assert "time" in ds.coords


def test_to_zarr_radar_writes_store(fetcher, tmp_path):
    """to_zarr works for radar and encodes the (required) match in the store name."""
    fetcher.any_items = _items_for("cpc2613000000_00060.001.h5")
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    pytest.importorskip("zarr")
    import xarray as xr

    base = tmp_path / "bronze" / "radar_precip"
    write_odim_composite(base / "cpc2613000000_00060.001.h5")

    store = to_zarr("radar_precip", data_dir=tmp_path, match="cpc2613000000")
    assert store.name == "radar_precip__cpc2613000000.zarr"
    assert store.exists()
    assert "acrr" in xr.open_zarr(store).data_vars


def test_to_zarr_radar_stacked_time_cube(fetcher, tmp_path):
    """stack= assembles radar timesteps into one (time, y, x) cube."""
    fetcher.any_items = _items_for("cpc26130000000.h5", "cpc26130000500.h5", "cpc26130001000.h5")
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    pytest.importorskip("zarr")
    import numpy as np
    import xarray as xr

    base = tmp_path / "bronze" / "radar_precip"
    write_odim_composite(base / "cpc26130000000.h5", time="000000")
    write_odim_composite(base / "cpc26130000500.h5", time="000500")
    write_odim_composite(base / "cpc26130001000.h5", time="001000")

    store = to_zarr("radar_precip", data_dir=tmp_path, match="cpc26130", stack=True)
    ds = xr.open_zarr(store)
    assert ds["acrr"].dims == ("time", "y", "x")
    assert dict(ds.sizes) == {"time": 3, "y": 2, "x": 3}
    # Time axis must be correct + strictly increasing (the append-encoding trap).
    times = ds.time.values
    assert (np.diff(times).astype("int64") > 0).all()
    assert str(times[0])[:16] == "2026-05-10T00:00"
    assert str(times[1])[:16] == "2026-05-10T00:05"


def test_to_zarr_radar_appends_new_timesteps_in_place(fetcher, tmp_path):
    """A later radar batch extends time instead of rewriting or copying the store."""
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    pytest.importorskip("zarr")
    import xarray as xr

    base = tmp_path / "bronze" / "radar_precip"
    first = "cpc26130000000.h5"
    second = "cpc26130000500.h5"
    write_odim_composite(base / first, time="000000")
    fetcher.any_items = _items_for(first)
    store = to_zarr("radar_precip", data_dir=tmp_path, match="cpc26130", stack=True)

    write_odim_composite(base / second, time="000500")
    fetcher.any_items = _items_for(second)
    to_zarr("radar_precip", data_dir=tmp_path, match="cpc26130", stack=True, mode="a")

    appended = xr.open_zarr(store)
    assert appended.sizes["time"] == 2
    assert str(appended.time.values[1])[:16] == "2026-05-10T00:05"


def test_to_zarr_radar_append_does_not_restack_timesteps_already_stored(fetcher, tmp_path):
    """The listing an append sees is cumulative, not just what is new.

    ``match`` scopes the STAC listing, and the listing returns everything
    published under that match — so the second call sees the first timestep
    again. Appending the whole set wrote it twice: [00:00] then [00:00, 00:05]
    produced a cube reading [00:00, 00:00, 00:05]. The sibling test above feeds
    back only the new file, which is the one listing shape that hides this.
    """
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    pytest.importorskip("zarr")
    import numpy as np
    import xarray as xr

    base = tmp_path / "bronze" / "radar_precip"
    first = "cpc26130000000.h5"
    second = "cpc26130000500.h5"

    write_odim_composite(base / first, time="000000")
    fetcher.any_items = _items_for(first)
    store = to_zarr("radar_precip", data_dir=tmp_path, match="cpc26130", stack=True)

    write_odim_composite(base / second, time="000500")
    # What upstream actually returns for this match the second time round.
    fetcher.any_items = _items_for(first, second)
    to_zarr("radar_precip", data_dir=tmp_path, match="cpc26130", stack=True, mode="a")

    appended = xr.open_zarr(store)
    assert appended.sizes["time"] == 2
    times = appended.time.values
    assert len(set(times.tolist())) == 2
    assert (np.diff(times).astype("int64") > 0).all()
    assert str(times[0])[:16] == "2026-05-10T00:00"
    assert str(times[1])[:16] == "2026-05-10T00:05"


def test_to_zarr_radar_append_creates_the_store_when_none_exists(fetcher, tmp_path):
    """mode="a" against a fresh workspace has nothing to extend, so it creates."""
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    pytest.importorskip("zarr")
    import xarray as xr

    base = tmp_path / "bronze" / "radar_precip"
    write_odim_composite(base / "cpc26130000000.h5", time="000000")
    fetcher.any_items = _items_for("cpc26130000000.h5")

    store = to_zarr("radar_precip", data_dir=tmp_path, match="cpc26130", stack=True, mode="a")

    assert xr.open_zarr(store).sizes["time"] == 1


def test_to_zarr_stack_requires_match(tmp_path):
    """A radar cube needs a match to scope the time range."""
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    with pytest.raises(ValueError, match="needs match="):
        to_zarr("radar_precip", data_dir=tmp_path, stack=True)


def test_open_dataset_match_selects_subset(fetcher, tmp_path):
    """match should restrict a multi-file collection to the matching files."""
    fetcher.any_items = _items_for("grid.rhiresd_ch01h.nc", "grid.ranomm9120_ch01r.nc")
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

    from foehn.grids import open_netcdf

    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    xr.Dataset({"t": (("x",), np.array([1.0, 2.0], "float32"))}, coords={"x": [0, 1]}).to_netcdf(a, engine="h5netcdf")
    xr.Dataset({"t": (("x",), np.array([3.0, 4.0], "float32"))}, coords={"x": [2, 3]}).to_netcdf(b, engine="h5netcdf")

    ds = open_netcdf([a, b], dataset="surface_derived_grid")
    assert ds["t"].shape == (4,)
    assert list(ds["x"].values) == [0, 1, 2, 3]


def test_open_netcdf_combines_despite_conflicting_global_attrs(tmp_path):
    """Differing global attrs (history/source) must not block an otherwise-clean combine."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    import numpy as np

    from foehn.grids import open_netcdf

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

    ds = open_netcdf([a, b], dataset="surface_derived_grid")
    assert ds["t"].shape == (4,)
    # Shared attrs survive; conflicting ones are dropped rather than raising.
    assert ds.attrs.get("title") == "rhiresd"
    assert "history" not in ds.attrs
    assert "source" not in ds.attrs


def test_to_zarr_writes_store(fetcher, tmp_path):
    """to_zarr should write a readable Zarr store under data_dir/zarr/."""
    fetcher.any_items = _items_for("grid.nc")
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


def test_to_zarr_match_yields_distinct_stores(fetcher, tmp_path):
    """Different match= filters must write to distinct stores, not clobber each other."""
    fetcher.any_items = _items_for("g.rhiresd.nc", "g.tabsd.nc")
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


def test_to_zarr_explicit_store_path(fetcher, tmp_path):
    """An explicit store= path overrides the derived data_dir/zarr/<name>.zarr location."""
    fetcher.any_items = _items_for("g.nc")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("zarr")
    import numpy as np

    _write_nc(tmp_path / "bronze" / "surface_derived_grid" / "g.nc", xr, np)

    out = tmp_path / "custom" / "mine.zarr"
    store = to_zarr("surface_derived_grid", data_dir=tmp_path, store=out)
    assert store == out
    assert store.exists()


def test_to_zarr_does_not_leak_consolidated_warning(fetcher, tmp_path, recwarn):
    """to_zarr keeps consolidated metadata but must not surface zarr's out-of-spec warning."""
    fetcher.any_items = _items_for("grid.nc")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("zarr")
    import numpy as np

    _write_nc(tmp_path / "bronze" / "surface_derived_grid" / "grid.nc", xr, np)

    to_zarr("surface_derived_grid", data_dir=tmp_path)

    messages = [str(w.message) for w in recwarn.list]
    assert not any("Consolidated metadata is currently not part" in m for m in messages), messages


def test_to_zarr_grib2_without_match_raises():
    """to_zarr inherits open_dataset's GRIB2 match= requirement."""
    with pytest.raises(ValueError, match="match="):
        to_zarr("forecast_icon_ch1")


def test_to_zarr_rechunk_without_dask_raises(fetcher, tmp_path):
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

    fetcher.any_items = _items_for("grid.nc")
    with (
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
        pytest.raises(ImportError, match="requires dask"),
    ):
        to_zarr("surface_derived_grid", data_dir=tmp_path, rechunk={"x": 1})


def test_a_written_store_carries_no_noncf_time_units(tmp_path):
    """'years since ...' throws on every later ``open_zarr``, calendar included.

    Asserted through ``write_zarr`` rather than the renaming step: moving the
    units aside is something a store *has done to it*, not a call a caller makes.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    import numpy as np

    from foehn.grids import write_zarr

    ds = xr.Dataset(
        {"v": (("time",), np.array([1.0]))},
        coords={"time": ("time", np.array([0.0]), {"units": "years since 1991-01-01", "calendar": "standard"})},
    )
    store = tmp_path / "noncf.zarr"
    write_zarr(ds, store)

    written = xr.open_zarr(store)
    assert "units" not in written["time"].attrs and "calendar" not in written["time"].attrs
    assert written["time"].attrs["units_noncf"] == "years since 1991-01-01"
    assert written["time"].attrs["calendar_noncf"] == "standard"


def test_a_written_store_keeps_valid_cf_time_units(tmp_path):
    """Well-formed CF units must survive, decoded, rather than being moved aside too."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    import numpy as np

    from foehn.grids import write_zarr

    ds = xr.Dataset(
        {"v": (("time",), np.array([1.0]))},
        coords={"time": ("time", np.array([0.0]), {"units": "days since 2000-01-01"})},
    )
    store = tmp_path / "cf.zarr"
    write_zarr(ds, store)

    written = xr.open_zarr(store)
    assert "units_noncf" not in written["time"].attrs
    assert written["time"].values[0] == np.datetime64("2000-01-01")


def test_append_writes_in_place_without_copying_the_existing_store(tmp_path):
    from foehn import grids

    store = tmp_path / "existing.zarr"
    store.mkdir()
    (store / "existing").write_text("keep")
    dataset = object()

    with (
        patch.object(grids, "_write_zarr") as implementation,
        patch("foehn.atomicwrite.shutil.copytree", side_effect=AssertionError("copied whole store")),
    ):
        grids.write_zarr(dataset, store, mode="a")

    implementation.assert_called_once_with(dataset, store, "a", rechunk=None, append_dim=None)


def test_grid_reader_append_targets_the_existing_store(fetcher, tmp_path):
    from foehn import grids, registry
    from foehn.workspace import Workspace

    store = tmp_path / "existing.zarr"
    store.mkdir()
    dataset = object()
    reader = registry.spec("surface_derived_grid").grid
    assert reader is not None

    with (
        patch.object(grids.GridReader, "open_dataset", return_value=dataset),
        patch.object(grids, "_write_zarr") as implementation,
        patch("foehn.atomicwrite.shutil.copytree", side_effect=AssertionError("copied whole store")),
    ):
        reader.write_store(
            "surface_derived_grid",
            store,
            match=None,
            variables=None,
            rechunk=None,
            mode="a",
            stack=False,
            workspace=Workspace(tmp_path),
            fetcher=fetcher,
        )

    implementation.assert_called_once_with(dataset, store, "a", rechunk=None)


def test_to_zarr_with_noncf_time_reopens_cleanly(fetcher, tmp_path):
    """End-to-end: a NetCDF with non-CF time units must produce a re-openable store."""
    fetcher.any_items = _items_for("normals_yearly.nc")
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
    out_dir = tmp_path / "bronze" / "climate_normals_grid"
    out_dir.mkdir(parents=True)
    # h5netcdf preserves the non-CF units without netCDF4's deprecated NumPy
    # shape assignment during synthetic fixture creation.
    ds.to_netcdf(out_dir / "normals_yearly.nc", engine="h5netcdf")

    store = to_zarr("climate_normals_grid", data_dir=tmp_path)
    roundtrip = xr.open_zarr(store)  # default CF decode must not throw
    assert "TnormY" in roundtrip.data_vars


# --- What a cube builder refuses ---


def test_grib2_cube_refuses_a_match_whose_files_do_not_differ(fetcher, tmp_path):
    """Two identical fields have no axis to promote — there is no cube to build."""
    pytest.importorskip("xarray")
    pytest.importorskip("cfgrib")
    pytest.importorskip("eccodes")
    fetcher.any_items = _items_for(
        "icon-ch1-eps-202605230000-0-t_2m-ctrl.grib2",
        "icon-ch1-eps-202605230000-0-t_2m-ctrl-copy.grib2",
    )
    base = tmp_path / "bronze" / "forecast_icon_ch1"
    for name in ("icon-ch1-eps-202605230000-0-t_2m-ctrl", "icon-ch1-eps-202605230000-0-t_2m-ctrl-copy"):
        _write_grib2(base / f"{name}.grib2", step=0, datatime=0)

    with pytest.raises(ValueError, match="nothing to assemble into a cube"):
        to_zarr("forecast_icon_ch1", data_dir=tmp_path, match="t_2m-ctrl", stack=True)


def test_radar_cube_refuses_a_composite_with_no_time(fetcher, tmp_path):
    """Without a stamp there is no axis to append along, and appending anyway would misdate it."""
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    pytest.importorskip("zarr")
    fetcher.any_items = _items_for("cpc2613000000_00060.001.h5")
    base = tmp_path / "bronze" / "radar_precip"
    write_odim_composite(base / "cpc2613000000_00060.001.h5", date="", time="")
    store = Workspace(tmp_path).zarr("radar_precip", "cpc26130")
    store.mkdir(parents=True)
    (store / "complete").write_text("previous")

    with pytest.raises(ValueError, match="no time coordinate"):
        to_zarr("radar_precip", data_dir=tmp_path, match="cpc26130", stack=True)

    assert (store / "complete").read_text() == "previous"
    assert not (store / "zarr.json").exists()


def test_radar_cube_keeps_only_the_requested_variables(fetcher, tmp_path):
    """``variables=`` is applied per timestep, before the append."""
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    fetcher.any_items = _items_for("cpc2613000000_00060.001.h5", "cpc2613000500_00060.001.h5")
    base = tmp_path / "bronze" / "radar_precip"
    write_odim_composite(base / "cpc2613000000_00060.001.h5", time="000000")
    write_odim_composite(base / "cpc2613000500_00060.001.h5", time="000500")

    store = to_zarr("radar_precip", data_dir=tmp_path, match="cpc26130", stack=True, variables="acrr")
    cube = xr.open_zarr(store)

    assert list(cube.data_vars) == ["acrr"]
    assert cube.sizes["time"] == 2


# --- What a NetCDF open refuses ---


def test_netcdf_open_points_at_match_when_files_will_not_combine(fetcher, tmp_path):
    """A heterogeneous multi-file set is the common mistake; the error has to name the fix."""
    xr = pytest.importorskip("xarray")
    import numpy as np

    fetcher.any_items = _items_for("a.nc", "b.nc")
    base = tmp_path / "bronze" / "surface_derived_grid"
    base.mkdir(parents=True)
    # Same variable, incompatible shapes on the same dim: combine_by_coords cannot merge them.
    xr.Dataset({"v": ("x", np.arange(3.0))}, coords={"x": [0, 1, 2]}).to_netcdf(base / "a.nc", engine="h5netcdf")
    xr.Dataset({"v": (("x", "y"), np.zeros((3, 2)))}, coords={"x": [0, 1, 2], "y": [0, 1]}).to_netcdf(
        base / "b.nc", engine="h5netcdf"
    )

    with pytest.raises(ValueError, match="narrow to a coherent set with match="):
        open_dataset("surface_derived_grid", data_dir=tmp_path)


# --- The optional-dependency guards ---


@pytest.mark.parametrize(
    ("require", "missing"),
    [
        ("require_netcdf", "xarray"),
        ("require_grib2", "cfgrib"),
        ("require_radar", "h5py"),
        ("require_radar", "pyproj"),
    ],
)
def test_a_missing_grid_dependency_names_the_extra(require, missing):
    """Checked before anything is fetched, so the message has to be the whole fix."""
    import builtins

    import foehn.grids as grids_mod

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == missing:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", refuse), pytest.raises(ImportError, match=r"foehn\[grids\]"):
        getattr(grids_mod, require)()


def test_a_source_that_cannot_be_read_at_all_reports_the_original_failure(tmp_path):
    """The decode_times=False retry is for non-CF units; it must not mask an unrelated error."""
    pytest.importorskip("xarray")
    from foehn.grids import open_netcdf

    corrupt = tmp_path / "broken.nc"
    corrupt.write_bytes(b"not a netcdf file at all")

    with pytest.raises(Exception, match=r"(?i)broken|netcdf|magic|unable"):
        open_netcdf([corrupt], dataset="surface_derived_grid")


def test_a_grib2_cube_recomputes_valid_time_after_combining(fetcher, tmp_path):
    """valid_time conflicts on concat, so it is dropped before the merge and rebuilt after."""
    pytest.importorskip("xarray")
    pytest.importorskip("cfgrib")
    pytest.importorskip("eccodes")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")

    from foehn.grids import cube_grib2

    base = tmp_path / "bronze" / "forecast_icon_ch1"
    files = []
    for step in (0, 6):
        path = base / f"icon-ch1-eps-202605230000-{step}-t_2m-ctrl.grib2"
        _write_grib2(path, step=step, datatime=0)
        files.append(path)

    store = tmp_path / "cube.zarr"
    with patch("foehn.grids.icon.attach_lonlat", side_effect=lambda ds, *a, **k: ds):
        cube_grib2(files, store, dataset="forecast_icon_ch1", workspace=Workspace(tmp_path), fetcher=fetcher)

    cube = xr.open_zarr(store)

    assert "step" in cube.dims
    assert (cube["valid_time"] == cube["time"] + cube["step"]).all()


def test_an_unreadable_netcdf_file_is_reported_as_itself(tmp_path):
    """A corrupt file is not a heterogeneous set, and must not be described as one.

    Every multi-file failure used to be relabelled "this set mixes
    parameters/levels/resolutions — narrow with match=". For an unreadable file
    that diagnosis is wrong and the remedy impossible: no match narrows away a
    corrupt cache entry, and the caller is sent looking for a parameter split
    that does not exist. The underlying error names the file; that is the useful
    thing. A real one looks exactly like this — valid HDF5 magic, nothing behind
    it — which is why the fixture is not simply random bytes.
    """
    xr = pytest.importorskip("xarray")
    import numpy as np

    from foehn.grids import open_netcdf

    good = tmp_path / "a_rhiresd.nc"
    corrupt = tmp_path / "b_rhiresd.nc"
    xr.Dataset({"v": ("x", np.arange(3.0))}, coords={"x": [0, 1, 2]}).to_netcdf(good, engine="h5netcdf")
    corrupt.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 512)

    with pytest.raises(OSError) as caught:
        open_netcdf([good, corrupt], dataset="surface_derived_grid")

    assert "mixes parameters" not in str(caught.value)
    assert corrupt.name in str(caught.value)


def test_open_netcdf_accepts_an_explicit_engine(tmp_path):
    """``engine=`` was a documented v0.4.0 keyword; removing it broke those calls."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    import numpy as np

    from foehn.grids import open_netcdf

    source = tmp_path / "grid.nc"
    xr.Dataset({"v": ("x", np.arange(3.0))}, coords={"x": [0, 1, 2]}).to_netcdf(source, engine="h5netcdf")

    opened = open_netcdf([source], dataset="surface_derived_grid", engine="h5netcdf")

    assert list(opened["v"].values) == [0.0, 1.0, 2.0]


def test_append_dedup_ignores_a_store_it_cannot_read(tmp_path):
    """An unreadable store means "unknown", not "fail" — the append still runs."""
    xr = pytest.importorskip("xarray")

    from foehn import grids as grids_mod

    unreadable = tmp_path / "broken.zarr"
    unreadable.mkdir()
    (unreadable / "zarr.json").write_text("{ not json")

    assert grids_mod._cube_times(xr, unreadable) == frozenset()


def test_append_dedup_ignores_a_store_without_a_time_axis(tmp_path):
    """Nothing to compare against, so nothing is skipped."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    import numpy as np

    from foehn import grids as grids_mod

    store = tmp_path / "no_time.zarr"
    xr.Dataset({"v": ("x", np.arange(3.0))}, coords={"x": [0, 1, 2]}).to_zarr(store, consolidated=False)

    assert grids_mod._cube_times(xr, store) == frozenset()


def _radar_cube(tmp_path, times=("000000", "000500")):
    from foehn.grids import cube_radar

    files = []
    for stamp in times:
        path = tmp_path / f"cpc2613{stamp}.h5"
        write_odim_composite(path, time=stamp)
        files.append(path)
    store = tmp_path / "cube.zarr"
    cube_radar(files, store, mode="w")
    return store, files


@pytest.mark.parametrize("mode", ["w-", "a-", "r+"])
def test_a_non_overwriting_mode_never_truncates_the_store(tmp_path, mode):
    """Only "w" means "replace what is there".

    Coercing every mode that was not "a" into "w" turned "w-" (create, never
    clobber) and "r+" (modify, never truncate) into a silent overwrite of the
    store they were chosen to protect — "r+" reduced a two-timestep cube to one.
    """
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    pytest.importorskip("zarr")
    import xarray as xr

    from foehn.grids import cube_radar

    store, files = _radar_cube(tmp_path)
    assert xr.open_zarr(store).sizes["time"] == 2

    with contextlib.suppress(Exception):
        cube_radar(files[:1], store, mode=mode)

    # Whether the mode refused or was a no-op, the existing cube survives.
    assert xr.open_zarr(store).sizes["time"] == 2


def test_a_restated_observation_is_rewritten_rather_than_skipped(tmp_path):
    """MeteoSwiss republishes a timestamp's values under its original name.

    CombiPrecip reanalysis replaces the original hourly file about eight days
    later, which is why the download path compares STAC ``updated`` rather than
    mere existence. De-duplicating an append on the timestamp alone undid that:
    bronze refreshed and the cube kept the superseded numbers forever.
    """
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    pytest.importorskip("zarr")
    import xarray as xr

    from foehn.grids import cube_radar

    source = tmp_path / "cpc2613000000.h5"
    write_odim_composite(source, time="000000", values=[[0.0, 1.5, 9.0], [9.0, 2.0, 3.0]])
    store = tmp_path / "cube.zarr"
    cube_radar([source], store, mode="w")
    assert float(xr.open_zarr(store).acrr.values.ravel()[1]) == 1.5

    write_odim_composite(source, time="000000", values=[[0.0, 42.0, 9.0], [9.0, 2.0, 3.0]])
    cube_radar([source], store, mode="a")

    revised = xr.open_zarr(store)
    assert float(revised.acrr.values.ravel()[1]) == 42.0
    assert revised.sizes["time"] == 1  # rewritten in place, not appended alongside


def test_an_unchanged_source_is_still_skipped_on_append(tmp_path):
    """The common re-run: same listing, same files, nothing to do."""
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    pytest.importorskip("zarr")
    import xarray as xr

    from foehn.grids import cube_radar

    store, files = _radar_cube(tmp_path)
    cube_radar(files, store, mode="a")

    appended = xr.open_zarr(store)
    assert appended.sizes["time"] == 2
    assert len(set(appended.time.values.tolist())) == 2


def test_source_fingerprints_survive_in_the_store(tmp_path):
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    pytest.importorskip("zarr")

    from foehn.grids import _stored_sources

    store, _ = _radar_cube(tmp_path)

    assert len(_stored_sources(store)) == 2
    assert _stored_sources(tmp_path / "missing.zarr") == {}


def test_grib2_honours_an_explicit_engine(tmp_path):
    """A caller who names a backend and silently gets another cannot tell."""
    pytest.importorskip("xarray")
    import foehn.grids as grids_mod

    with (
        patch.object(grids_mod, "_open_grid") as opened,
        patch.object(grids_mod.icon, "attach_lonlat", lambda ds, *a, **k: ds),
    ):
        grids_mod.open_grib2(
            [tmp_path / "x.grib2"],
            dataset="forecast_icon_ch1",
            workspace=Workspace(tmp_path),
            fetcher=None,
            engine="cfgrib-custom",
        )

    assert opened.call_args.kwargs["engine"] == "cfgrib-custom"


def test_source_fingerprints_tolerate_a_store_they_cannot_read(tmp_path):
    """Bookkeeping must never be the thing that fails a good write."""
    from foehn.grids import _cube_time_index, _record_sources, _stored_sources

    broken = tmp_path / "broken.zarr"
    broken.mkdir()
    (broken / "zarr.json").write_text("{ not json")

    assert _stored_sources(broken) == {}
    assert _cube_time_index(None, broken) == {}
    _record_sources(broken, {"1": "x"})  # logs and moves on


def test_revising_a_timestep_the_cube_no_longer_holds_is_refused(tmp_path):
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    pytest.importorskip("zarr")
    import xarray as xr

    from foehn.grids import _revise_in_place

    store, _files = _radar_cube(tmp_path)
    ds = xr.open_zarr(store).isel(time=[0])

    with pytest.raises(ValueError, match="no longer holds"):
        _revise_in_place(xr, ds, store, frozenset({-1}))
