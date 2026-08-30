"""Tests for the gridded read path (foehn.open_dataset)."""

from pathlib import Path
from unittest.mock import patch

import pytest

import foehn
from foehn.collections import COLLECTIONS
from foehn.fetch import FetchError
from foehn.grids import _ensure_grid_files, open_dataset, to_zarr
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


def test_ensure_netcdf_files_raises_when_no_nc_assets(fetcher, tmp_path):
    """A collection that only exposes GeoTIFF/ZIP should raise a clear error."""
    fetcher.any_items = [
        {"assets": {"a": {"href": "https://data.geo.admin.ch/x/grid.tif"}}},
        {"assets": {"b": {"href": "https://data.geo.admin.ch/x/bundle.zip"}}},
    ]
    with pytest.raises(ValueError, match=r"No \.nc assets"):
        _ensure_grid_files("climate_normals_grid", tmp_path / "bronze", fetcher=fetcher)


def test_ensure_netcdf_files_match_selects_subset_via_remote(tmp_path):
    """A NetCDF match consults the remote listing (to verify completeness) and keeps the subset."""
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    keep = out_dir / "x.rhiresd_ch01h.nc"
    drop = out_dir / "x.ranomm9120_ch01r.nc"
    keep.write_bytes(b"x")
    drop.write_bytes(b"x")

    items = _items_for("x.rhiresd_ch01h.nc", "x.ranomm9120_ch01r.nc")
    fake = _fake(items)
    result = _ensure_grid_files("surface_derived_grid", tmp_path / "bronze", match="rhiresd", fetcher=fake)
    assert len(fake.listings) == 1  # multi-file format always verifies against remote
    assert fake.streams == []  # both already cached
    assert result == [keep]


def test_ensure_netcdf_files_match_downloads_missing_from_partial_cache(tmp_path):
    """A multi-file NetCDF match must not return a partial cache — it fetches what's missing."""
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / "rhiresd_part1.nc").write_bytes(b"x")  # interrupted earlier download left only part 1

    items = _items_for("rhiresd_part1.nc", "rhiresd_part2.nc")

    def fake_download(_href, filepath):
        Path(filepath).write_bytes(b"y")

    fake = _fake(items)
    fake.stream_hook = fake_download
    result = _ensure_grid_files("surface_derived_grid", tmp_path / "bronze", match="rhiresd", fetcher=fake)
    assert len(fake.streams) == 1  # the missing part 2 is fetched
    assert {p.name for p in result} == {"rhiresd_part1.nc", "rhiresd_part2.nc"}


def test_ensure_grid_files_fetches_missing_concurrently(tmp_path):
    """Missing files are fetched in parallel, not one after another."""
    import threading

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    names = [f"rhiresd_part{i}.nc" for i in range(6)]
    items = _items_for(*names)

    active = 0
    peak = 0
    lock = threading.Lock()
    barrier = threading.Barrier(len(names), timeout=5)

    def fake_download(_href, filepath):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        # Blocks until every download is in flight — times out if they are serial.
        barrier.wait()
        with lock:
            active -= 1
        Path(filepath).write_bytes(b"y")

    fake = _fake(items)
    fake.stream_hook = fake_download
    result = _ensure_grid_files("surface_derived_grid", tmp_path / "bronze", match="rhiresd", fetcher=fake)

    assert len(fake.streams) == len(names)
    assert peak > 1  # genuinely overlapping
    assert {p.name for p in result} == set(names)


def test_ensure_grid_files_deduplicates_repeated_asset(tmp_path):
    """One file listed under several asset keys is downloaded once.

    Two workers streaming into the same ``.part`` would corrupt it, so the
    dedupe matters now that these downloads run concurrently.
    """
    items = [
        {
            "assets": {
                "data": {"href": "https://data.geo.admin.ch/x/rhiresd_ch01h.nc"},
                "alternate": {"href": "https://data.geo.admin.ch/x/rhiresd_ch01h.nc"},
            }
        }
    ]

    def fake_download(_href, filepath):
        Path(filepath).write_bytes(b"y")

    fake = _fake(items)
    fake.stream_hook = fake_download
    result = _ensure_grid_files("surface_derived_grid", tmp_path / "bronze", match="rhiresd", fetcher=fake)

    assert len(fake.streams) == 1
    assert [p.name for p in result] == ["rhiresd_ch01h.nc"]


def test_grid_listing_accepts_a_cached_walk(tmp_path):
    """The read path lists on every open, so it opts into a cached listing.

    Whether an entry is reused, and for how long, is the fetcher's business —
    the TTL is derived from what the walk cost, which only it can measure (see
    tests/test_fetch.py). What grids decides is that staleness is acceptable
    here at all; the download paths never ask for it.
    """
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / "x.rhiresd_ch01h.nc").write_bytes(b"x")
    fake = _fake(_items_for("x.rhiresd_ch01h.nc"))

    _ensure_grid_files("surface_derived_grid", tmp_path / "bronze", match="rhiresd", fetcher=fake)

    assert [cache for _, cache, _, _ in fake.listings] == [True]


def test_run_datetime_filter_only_applies_to_grib2():
    """A run stamp narrows GRIB2 listings; other formats must keep the full walk.

    CSV and radar collections set item datetime to a catalog-refresh time, so
    filtering on a real data date there would match nothing.
    """
    from foehn.grids import _run_datetime_filter

    assert _run_datetime_filter("forecast_icon_ch1", "202605231500-0-t_2m-ctrl") == "2026-05-23T15:00:00Z"
    assert _run_datetime_filter("analysis_kenda_ch1", "202605231500-0-t_2m-ctrl") == "2026-05-23T15:00:00Z"
    # radar (HDF5) and NetCDF opt out entirely, even with a 12-digit run in match
    assert _run_datetime_filter("radar_precip", "cpc202605231500") is None
    assert _run_datetime_filter("surface_derived_grid", "202605231500") is None
    # no parseable run stamp, or an impossible one
    assert _run_datetime_filter("forecast_icon_ch1", "t_2m-ctrl") is None
    assert _run_datetime_filter("forecast_icon_ch1", "209913451500") is None
    assert _run_datetime_filter("forecast_icon_ch1", None) is None


def test_ensure_grid_files_narrows_forecast_listing_by_run(tmp_path):
    """A GRIB2 open passes the run through as a server-side datetime filter."""
    out_dir = tmp_path / "bronze" / "forecast_icon_ch1"
    out_dir.mkdir(parents=True)
    name = "icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2"
    (out_dir / name).write_bytes(b"x")

    fake = _fake(_items_for(name))
    _ensure_grid_files(
        "forecast_icon_ch1",
        tmp_path / "bronze",
        suffixes=(".grib2",),
        match="202605231500-0-t_2m-ctrl",
        max_files=1,
        fetcher=fake,
    )

    assert [dt for _, _, dt, _ in fake.listings] == ["2026-05-23T15:00:00Z"]


def test_ensure_grid_files_falls_back_when_filtered_listing_misses(tmp_path):
    """A filtered miss retries unfiltered rather than reporting the file missing."""
    out_dir = tmp_path / "bronze" / "forecast_icon_ch1"
    out_dir.mkdir(parents=True)
    name = "icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2"
    (out_dir / name).write_bytes(b"x")

    class _FilteredMiss(InMemoryFetcher):
        """Server-side filter finds nothing; the full walk does."""

        def items(self, collection_id, *, cache=False, datetime_filter=None, max_items=None):
            found = super().items(collection_id, cache=cache, datetime_filter=datetime_filter, max_items=max_items)
            return [] if datetime_filter else found

    fake = _FilteredMiss()
    fake.any_items = _items_for(name)
    fake.default_body = b"x"
    result = _ensure_grid_files(
        "forecast_icon_ch1",
        tmp_path / "bronze",
        suffixes=(".grib2",),
        match="202605231500-0-t_2m-ctrl",
        max_files=1,
        fetcher=fake,
    )
    assert [p.name for p in result] == [name]
    assert len(fake.listings) == 2  # filtered, then the unfiltered retry


def test_ensure_single_file_match_validated_against_remote_not_cache(tmp_path):
    """A single-file match unique in a sparse cache but ambiguous in the collection is rejected."""
    out_dir = tmp_path / "bronze" / "forecast_icon_ch1"
    out_dir.mkdir(parents=True)
    (out_dir / "icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2").write_bytes(b"x")  # only one cached

    # ...but the collection has TWO files matching "t_2m-ctrl".
    items = _items_for(
        "icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2",
        "icon-ch1-eps-202605231800-0-t_2m-ctrl.grib2",
    )
    fake = _fake(items)
    with pytest.raises(ValueError, match="one file at a time"):
        _ensure_grid_files(
            "forecast_icon_ch1",
            tmp_path / "bronze",
            suffixes=(".grib2", ".grib"),
            match="t_2m-ctrl",
            max_files=1,
            fetcher=fake,
        )
    assert fake.streams == []  # rejected before any download


def test_ensure_single_file_serves_cache_when_remote_unique(tmp_path):
    """When the collection has exactly one match, the cached file is reused (no re-download)."""
    out_dir = tmp_path / "bronze" / "forecast_icon_ch1"
    out_dir.mkdir(parents=True)
    f = out_dir / "icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2"
    f.write_bytes(b"x")

    items = _items_for("icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2")
    fake = _fake(items)
    result = _ensure_grid_files(
        "forecast_icon_ch1",
        tmp_path / "bronze",
        suffixes=(".grib2", ".grib"),
        match="t_2m-ctrl",
        max_files=1,
        fetcher=fake,
    )
    assert fake.streams == []
    assert result == [f]


def test_ensure_netcdf_files_match_no_remote_match_raises(tmp_path):
    fake = _fake([{"assets": {"a": {"href": "https://data.geo.admin.ch/x/ranomm9120.nc"}}}])
    with pytest.raises(ValueError, match="matching 'rhiresd'"):
        _ensure_grid_files("surface_derived_grid", tmp_path / "bronze", match="rhiresd", fetcher=fake)


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

    def fake_download(_href, filepath):
        Path(filepath).write_bytes(b"y")

    fake = _fake(items)
    fake.stream_hook = fake_download
    result = _ensure_grid_files("surface_derived_grid", tmp_path / "bronze", fetcher=fake)

    assert len(fake.listings) == 1
    assert len(fake.streams) == 1  # only the missing file is fetched; cache reused
    assert {p.name for p in result} == {"ogd.rhiresd.nc", "ogd.tabsd.nc"}


def test_ensure_netcdf_files_offline_falls_back_to_cache(tmp_path):
    """If the STAC API is unreachable, fall back to the cached files with a warning."""
    base = tmp_path / "bronze" / "surface_derived_grid"
    base.mkdir(parents=True)
    (base / "a.nc").write_bytes(b"x")

    fake = _fake()
    fake.fail(COLLECTIONS["surface_derived_grid"], FetchError("offline"))
    with pytest.warns(UserWarning, match="without checking the collection"):
        result = _ensure_grid_files("surface_derived_grid", tmp_path / "bronze", fetcher=fake)
    assert [p.name for p in result] == ["a.nc"]


def test_ensure_netcdf_files_offline_no_cache_reraises(tmp_path):
    """Offline with an empty cache must surface the network error, not swallow it."""
    fake = _fake()
    fake.fail(COLLECTIONS["surface_derived_grid"], FetchError("offline"))
    with pytest.raises(FetchError):
        _ensure_grid_files("surface_derived_grid", tmp_path / "bronze", fetcher=fake)


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


# ── GRIB2 lat/lon join (ICON unstructured grid → constants file) ──────────────


def test_ensure_constants_file_uses_cache(tmp_path):
    """A cached horizontal-constants file is returned without a metadata call."""
    from foehn.grids import _ensure_constants_file

    out = tmp_path / "bronze" / "forecast_icon_ch1"
    out.mkdir(parents=True)
    f = out / "horizontal_constants_icon-ch1-eps.grib2"
    f.write_bytes(b"x")

    fake = _fake()
    result = _ensure_constants_file("forecast_icon_ch1", tmp_path / "bronze", fetcher=fake)

    assert fake.collection_calls == []  # the cached file short-circuits the lookup
    assert result == f


def test_ensure_constants_file_none_when_absent(tmp_path):
    """A collection without a horizontal-constants asset yields None (no coords to join)."""
    from foehn.grids import _ensure_constants_file

    fake = _fake()
    fake.any_collection = {"assets": {"params.csv": {"href": "https://data.geo.admin.ch/x/params.csv"}}}

    assert _ensure_constants_file("forecast_icon_ch1", tmp_path / "bronze", fetcher=fake) is None


def test_ensure_constants_file_downloads_when_missing(tmp_path):
    """When absent locally, the constants file is downloaded once and returned."""
    from foehn.grids import _ensure_constants_file

    def fake_download(_href, filepath):
        Path(filepath).write_bytes(b"x")

    fake = _fake()
    fake.any_collection = {
        "assets": {"horizontal_constants_icon-ch1-eps.grib2": {"href": "https://data.geo.admin.ch/x/hc.grib2"}}
    }
    fake.stream_hook = fake_download
    path = _ensure_constants_file("forecast_icon_ch1", tmp_path / "bronze", fetcher=fake)

    assert len(fake.streams) == 1
    assert path.name == "hc.grib2"
    assert path.exists()


def test_icon_unstructured_lonlat_reads_and_caches(tmp_path):
    """tlat/tlon are extracted from the constants GRIB and cached per collection."""
    pytest.importorskip("xarray")
    cfgrib = pytest.importorskip("cfgrib")
    import numpy as np
    import xarray as xr

    from foehn.grids import _ICON_COORDS_CACHE, _icon_unstructured_lonlat

    _ICON_COORDS_CACHE.pop(("forecast_icon_ch1", str(tmp_path / "bronze")), None)
    const = xr.Dataset({"tlat": ("values", np.array([46.0, 47.0])), "tlon": ("values", np.array([7.0, 8.0]))})
    fake_path = tmp_path / "hc.grib2"
    fake_path.write_bytes(b"x")

    with (
        patch("foehn.grids._ensure_constants_file", return_value=fake_path),
        patch.object(cfgrib, "open_datasets", return_value=[const]) as mock_open,
    ):
        lat, lon = _icon_unstructured_lonlat("forecast_icon_ch1", tmp_path / "bronze", fetcher=_fake())
        _icon_unstructured_lonlat("forecast_icon_ch1", tmp_path / "bronze", fetcher=_fake())  # cached → no re-parse

    assert list(lat) == [46.0, 47.0]
    assert list(lon) == [7.0, 8.0]
    mock_open.assert_called_once()
    _ICON_COORDS_CACHE.pop(("forecast_icon_ch1", str(tmp_path / "bronze")), None)


def test_icon_unstructured_lonlat_none_when_no_constants(tmp_path):
    """No constants file → (None, None); the caller leaves the grid un-georeferenced."""
    from foehn.grids import _ICON_COORDS_CACHE, _icon_unstructured_lonlat

    _ICON_COORDS_CACHE.pop(("forecast_icon_ch1", str(tmp_path / "bronze")), None)
    with patch("foehn.grids._ensure_constants_file", return_value=None):
        lat, lon = _icon_unstructured_lonlat("forecast_icon_ch1", tmp_path / "bronze", fetcher=_fake())
    assert lat is None and lon is None
    _ICON_COORDS_CACHE.pop(("forecast_icon_ch1", str(tmp_path / "bronze")), None)


def test_icon_unstructured_lonlat_cache_is_keyed_on_data_dir(tmp_path):
    """A second data_dir must not be served the first one's coordinates.

    The constants file is resolved *under* bronze_dir, so a cache keyed on the
    collection alone silently georeferences one data_dir's grids with another's.
    """
    pytest.importorskip("xarray")
    cfgrib = pytest.importorskip("cfgrib")
    import numpy as np
    import xarray as xr

    from foehn.grids import _ICON_COORDS_CACHE, _icon_unstructured_lonlat

    dir_a, dir_b = tmp_path / "a" / "bronze", tmp_path / "b" / "bronze"
    for d in (dir_a, dir_b):
        _ICON_COORDS_CACHE.pop(("forecast_icon_ch1", str(d)), None)

    def const(lat_val):
        return xr.Dataset({"tlat": ("values", np.array([lat_val])), "tlon": ("values", np.array([7.0]))})

    fake_path = tmp_path / "hc.grib2"
    fake_path.write_bytes(b"x")

    with patch("foehn.grids._ensure_constants_file", return_value=fake_path):
        with patch.object(cfgrib, "open_datasets", return_value=[const(46.0)]):
            lat_a, _ = _icon_unstructured_lonlat("forecast_icon_ch1", dir_a, fetcher=_fake())
        with patch.object(cfgrib, "open_datasets", return_value=[const(99.0)]) as mock_b:
            lat_b, _ = _icon_unstructured_lonlat("forecast_icon_ch1", dir_b, fetcher=_fake())

    assert list(lat_a) == [46.0]
    assert list(lat_b) == [99.0]  # not dir_a's cached value
    mock_b.assert_called_once()  # the second dir really did parse its own file
    for d in (dir_a, dir_b):
        _ICON_COORDS_CACHE.pop(("forecast_icon_ch1", str(d)), None)


def test_icon_unstructured_lonlat_does_not_cache_failure(tmp_path):
    """A transient miss must not poison every later open in the process."""
    pytest.importorskip("xarray")
    cfgrib = pytest.importorskip("cfgrib")
    import numpy as np
    import xarray as xr

    from foehn.grids import _ICON_COORDS_CACHE, _icon_unstructured_lonlat

    bronze = tmp_path / "bronze"
    _ICON_COORDS_CACHE.pop(("forecast_icon_ch1", str(bronze)), None)

    # First call: constants unreachable (offline) → (None, None), not memoised.
    with patch("foehn.grids._ensure_constants_file", return_value=None):
        assert _icon_unstructured_lonlat("forecast_icon_ch1", bronze, fetcher=_fake()) == (None, None)

    const = xr.Dataset({"tlat": ("values", np.array([46.0])), "tlon": ("values", np.array([7.0]))})
    fake_path = tmp_path / "hc.grib2"
    fake_path.write_bytes(b"x")
    with (
        patch("foehn.grids._ensure_constants_file", return_value=fake_path),
        patch.object(cfgrib, "open_datasets", return_value=[const]),
    ):
        lat, lon = _icon_unstructured_lonlat("forecast_icon_ch1", bronze, fetcher=_fake())

    assert list(lat) == [46.0] and list(lon) == [7.0]
    _ICON_COORDS_CACHE.pop(("forecast_icon_ch1", str(bronze)), None)


# ── GRIB2 hypercube (stack="auto") ────────────────────────────────────────────


def test_to_zarr_grib2_hypercube(fetcher, tmp_path):
    """stack='auto' promotes the varying forecast axes (time, step) into one cube."""
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

    store = to_zarr("forecast_icon_ch1", data_dir=tmp_path, match="t_2m-ctrl", stack="auto")
    cube = xr.open_zarr(store)
    assert "time" in cube.dims and "step" in cube.dims
    assert cube.sizes["time"] == 2 and cube.sizes["step"] == 2
    assert cube["t2m"].dims[-2:] == ("latitude", "longitude")  # spatial dims preserved


def test_to_zarr_auto_requires_match(tmp_path):
    """stack='auto' needs a match to scope the cube."""
    pytest.importorskip("cfgrib")
    with pytest.raises(ValueError, match="needs match="):
        to_zarr("forecast_icon_ch1", data_dir=tmp_path, stack="auto")


def test_to_zarr_invalid_stack_value(tmp_path):
    with pytest.raises(ValueError, match="stack="):
        to_zarr("forecast_icon_ch1", data_dir=tmp_path, match="x", stack="member")


def test_to_zarr_netcdf_auto_writes_combined(fetcher, tmp_path):
    """stack='auto' on NetCDF just writes the already-combined multi-file match (no special path)."""
    fetcher.any_items = _items_for("g.rain.nc", "g.anom.nc")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("zarr")
    import numpy as np

    base = tmp_path / "bronze" / "surface_derived_grid"
    _write_nc(base / "g.rain.nc", xr, np, var="rain")
    _write_nc(base / "g.anom.nc", xr, np, var="anom")

    store = to_zarr("surface_derived_grid", data_dir=tmp_path, match="g.", stack="auto")
    ds = xr.open_zarr(store)
    assert "rain" in ds.data_vars
    assert "anom" in ds.data_vars  # both files combined into one store


_SWISS_PROJ = (
    "+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +k_0=1 "
    "+x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs"
)


def _write_odim_composite(
    path, quantity="ACRR", nodata=float("nan"), undetect=float("inf"), date="20260510", time="000000"
):
    """Write a tiny ODIM-H5 Cartesian COMP composite (offline radar fixture)."""
    import h5py
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.array([[0.0, 1.5, nodata], [undetect, 2.0, 3.0]], dtype="float64")
    with h5py.File(path, "w") as f:
        f.attrs["Conventions"] = "ODIM_H5/V2_4"
        what = f.create_group("what")
        what.attrs["object"] = "COMP"
        what.attrs["date"] = date
        what.attrs["time"] = time
        where = f.create_group("where")
        where.attrs.update({"xsize": 3, "ysize": 2, "xscale": 1000.0, "yscale": 1000.0, "projdef": _SWISS_PROJ})
        where.attrs.update({"UL_lon": 2.68942, "UL_lat": 49.3744})
        d1 = f.create_group("dataset1").create_group("data1")
        d1.create_dataset("data", data=data)
        dwhat = d1.create_group("what")
        dwhat.attrs.update({"gain": 1.0, "offset": 0.0, "nodata": nodata, "undetect": undetect, "quantity": quantity})


def test_open_dataset_reads_radar(fetcher, tmp_path):
    """End-to-end: read an ODIM COMP composite — scaling, masking, LV95 coords."""
    fetcher.any_items = _items_for("cpc2613000000_00060.001.h5")
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    import numpy as np

    base = tmp_path / "bronze" / "radar_precip"
    _write_odim_composite(base / "cpc2613000000_00060.001.h5")

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
    _write_odim_composite(base / "cpc2613000000_00060.001.h5")

    store = to_zarr("radar_precip", data_dir=tmp_path, match="cpc2613000000")
    assert store.name == "radar_precip__cpc2613000000.zarr"
    assert store.exists()
    assert "acrr" in xr.open_zarr(store).data_vars


@pytest.mark.parametrize("stack", ["time", "auto"])
def test_to_zarr_radar_stacked_time_cube(fetcher, tmp_path, stack):
    """Both stack='time' and stack='auto' assemble radar timesteps into one (time, y, x) cube."""
    fetcher.any_items = _items_for("cpc26130000000.h5", "cpc26130000500.h5", "cpc26130001000.h5")
    pytest.importorskip("xarray")
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    pytest.importorskip("zarr")
    import numpy as np
    import xarray as xr

    base = tmp_path / "bronze" / "radar_precip"
    _write_odim_composite(base / "cpc26130000000.h5", time="000000")
    _write_odim_composite(base / "cpc26130000500.h5", time="000500")
    _write_odim_composite(base / "cpc26130001000.h5", time="001000")

    store = to_zarr("radar_precip", data_dir=tmp_path, match="cpc26130", stack=stack)
    ds = xr.open_zarr(store)
    assert ds["acrr"].dims == ("time", "y", "x")
    assert dict(ds.sizes) == {"time": 3, "y": 2, "x": 3}
    # Time axis must be correct + strictly increasing (the append-encoding trap).
    times = ds.time.values
    assert (np.diff(times).astype("int64") > 0).all()
    assert str(times[0])[:16] == "2026-05-10T00:00"
    assert str(times[1])[:16] == "2026-05-10T00:05"


def test_to_zarr_stack_requires_match(tmp_path):
    """stack='time' needs a match to scope the time range."""
    pytest.importorskip("h5py")
    pytest.importorskip("pyproj")
    with pytest.raises(ValueError, match="needs match="):
        to_zarr("radar_precip", data_dir=tmp_path, stack="time")


def test_to_zarr_stack_rejected_for_netcdf(tmp_path):
    """stack= is radar-only; NetCDF matches already combine."""
    with pytest.raises(ValueError, match="only supported for radar"):
        to_zarr("surface_derived_grid", data_dir=tmp_path, match="rhiresd", stack="time")


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

    from foehn.grids import _open_grid

    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    xr.Dataset({"t": (("x",), np.array([1.0, 2.0], "float32"))}, coords={"x": [0, 1]}).to_netcdf(a, engine="h5netcdf")
    xr.Dataset({"t": (("x",), np.array([3.0, 4.0], "float32"))}, coords={"x": [2, 3]}).to_netcdf(b, engine="h5netcdf")

    ds = _open_grid(xr, [a, b], engine=None)
    assert ds["t"].shape == (4,)
    assert list(ds["x"].values) == [0, 1, 2, 3]


def test_open_netcdf_combines_despite_conflicting_global_attrs(tmp_path):
    """Differing global attrs (history/source) must not block an otherwise-clean combine."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    import numpy as np

    from foehn.grids import _open_grid

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

    ds = _open_grid(xr, [a, b], engine=None)
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
    # netcdf4 engine preserves the non-CF units attribute on the time axis
    ds.to_netcdf(out_dir / "normals_yearly.nc", engine="netcdf4")

    store = to_zarr("climate_normals_grid", data_dir=tmp_path)
    roundtrip = xr.open_zarr(store)  # default CF decode must not throw
    assert "TnormY" in roundtrip.data_vars
