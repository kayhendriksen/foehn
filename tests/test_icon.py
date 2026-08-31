"""Tests for ICON's unstructured grid coordinates (foehn.icon).

The cache is this module's own internal seam, so its tests read it directly.
They used to read it from ``test_grids``, across a module they did not own.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from foehn.fetch import FetchError
from foehn.icon import _COORDS_CACHE, constants_file, unstructured_lonlat
from foehn.workspace import Workspace
from tests.fakes import InMemoryFetcher


def _fake(items=None, *, body=b"x"):
    """A fetcher listing *items*, serving *body* for every asset."""
    fake = InMemoryFetcher()
    fake.any_items = items if items is not None else []
    fake.default_body = body
    return fake


def test_ensure_constants_file_uses_cache(tmp_path):
    """A cached horizontal-constants file is returned without a metadata call."""
    out = tmp_path / "bronze" / "forecast_icon_ch1"
    out.mkdir(parents=True)
    f = out / "horizontal_constants_icon-ch1-eps.grib2"
    f.write_bytes(b"x")

    fake = _fake()
    result = constants_file("forecast_icon_ch1", Workspace(tmp_path), fetcher=fake)

    assert fake.collection_calls == []  # the cached file short-circuits the lookup
    assert result == f


def test_ensure_constants_file_none_when_absent(tmp_path):
    """A collection without a horizontal-constants asset yields None (no coords to join)."""
    fake = _fake()
    fake.any_collection = {"assets": {"params.csv": {"href": "https://data.geo.admin.ch/x/params.csv"}}}

    assert constants_file("forecast_icon_ch1", Workspace(tmp_path), fetcher=fake) is None


def test_ensure_constants_file_downloads_when_missing(tmp_path):
    """When absent locally, the constants file is downloaded once and returned."""

    def fake_download(_href, filepath):
        Path(filepath).write_bytes(b"x")

    fake = _fake()
    fake.any_collection = {
        "assets": {"horizontal_constants_icon-ch1-eps.grib2": {"href": "https://data.geo.admin.ch/x/hc.grib2"}}
    }
    fake.stream_hook = fake_download
    path = constants_file("forecast_icon_ch1", Workspace(tmp_path), fetcher=fake)

    assert len(fake.streams) == 1
    assert path.name == "hc.grib2"
    assert path.exists()


def test_icon_unstructured_lonlat_reads_and_caches(tmp_path):
    """tlat/tlon are extracted from the constants GRIB and cached per collection."""
    pytest.importorskip("xarray")
    cfgrib = pytest.importorskip("cfgrib")
    import numpy as np
    import xarray as xr

    _COORDS_CACHE.pop(("forecast_icon_ch1", str(tmp_path / "bronze")), None)
    const = xr.Dataset({"tlat": ("values", np.array([46.0, 47.0])), "tlon": ("values", np.array([7.0, 8.0]))})
    fake_path = tmp_path / "hc.grib2"
    fake_path.write_bytes(b"x")

    with (
        patch("foehn.icon.constants_file", return_value=fake_path),
        patch.object(cfgrib, "open_datasets", return_value=[const]) as mock_open,
    ):
        lat, lon = unstructured_lonlat("forecast_icon_ch1", Workspace(tmp_path), fetcher=_fake())
        unstructured_lonlat("forecast_icon_ch1", Workspace(tmp_path), fetcher=_fake())  # cached → no re-parse

    assert list(lat) == [46.0, 47.0]
    assert list(lon) == [7.0, 8.0]
    mock_open.assert_called_once()
    _COORDS_CACHE.pop(("forecast_icon_ch1", str(tmp_path / "bronze")), None)


def test_icon_unstructured_lonlat_none_when_no_constants(tmp_path):
    """No constants file → (None, None); the caller leaves the grid un-georeferenced."""
    _COORDS_CACHE.pop(("forecast_icon_ch1", str(tmp_path / "bronze")), None)
    with patch("foehn.icon.constants_file", return_value=None):
        lat, lon = unstructured_lonlat("forecast_icon_ch1", Workspace(tmp_path), fetcher=_fake())
    assert lat is None and lon is None
    _COORDS_CACHE.pop(("forecast_icon_ch1", str(tmp_path / "bronze")), None)


def test_icon_unstructured_lonlat_cache_is_keyed_on_data_dir(tmp_path):
    """A second data_dir must not be served the first one's coordinates.

    The constants file is resolved *under* bronze_dir, so a cache keyed on the
    collection alone silently georeferences one data_dir's grids with another's.
    """
    pytest.importorskip("xarray")
    cfgrib = pytest.importorskip("cfgrib")
    import numpy as np
    import xarray as xr

    dir_a, dir_b = tmp_path / "a" / "bronze", tmp_path / "b" / "bronze"
    for d in (dir_a, dir_b):
        _COORDS_CACHE.pop(("forecast_icon_ch1", str(d)), None)

    def const(lat_val):
        return xr.Dataset({"tlat": ("values", np.array([lat_val])), "tlon": ("values", np.array([7.0]))})

    fake_path = tmp_path / "hc.grib2"
    fake_path.write_bytes(b"x")

    with patch("foehn.icon.constants_file", return_value=fake_path):
        with patch.object(cfgrib, "open_datasets", return_value=[const(46.0)]):
            lat_a, _ = unstructured_lonlat("forecast_icon_ch1", Workspace(dir_a), fetcher=_fake())
        with patch.object(cfgrib, "open_datasets", return_value=[const(99.0)]) as mock_b:
            lat_b, _ = unstructured_lonlat("forecast_icon_ch1", Workspace(dir_b), fetcher=_fake())

    assert list(lat_a) == [46.0]
    assert list(lat_b) == [99.0]  # not dir_a's cached value
    mock_b.assert_called_once()  # the second dir really did parse its own file
    for d in (dir_a, dir_b):
        _COORDS_CACHE.pop(("forecast_icon_ch1", str(d)), None)


def test_icon_unstructured_lonlat_does_not_cache_failure(tmp_path):
    """A transient miss must not poison every later open in the process."""
    pytest.importorskip("xarray")
    cfgrib = pytest.importorskip("cfgrib")
    import numpy as np
    import xarray as xr

    workspace = Workspace(tmp_path)
    _COORDS_CACHE.pop(("forecast_icon_ch1", str(workspace.root)), None)

    # First call: constants unreachable (offline) → (None, None), not memoised.
    with patch("foehn.icon.constants_file", return_value=None):
        assert unstructured_lonlat("forecast_icon_ch1", workspace, fetcher=_fake()) == (None, None)

    const = xr.Dataset({"tlat": ("values", np.array([46.0])), "tlon": ("values", np.array([7.0]))})
    fake_path = tmp_path / "hc.grib2"
    fake_path.write_bytes(b"x")
    with (
        patch("foehn.icon.constants_file", return_value=fake_path),
        patch.object(cfgrib, "open_datasets", return_value=[const]),
    ):
        lat, lon = unstructured_lonlat("forecast_icon_ch1", workspace, fetcher=_fake())

    assert list(lat) == [46.0] and list(lon) == [7.0]
    _COORDS_CACHE.pop(("forecast_icon_ch1", str(workspace.root)), None)


# --- Attaching the coordinates to a field ---


def _unstructured(size=4):
    """A bare ICON-shaped Dataset: one variable on a 1-D ``values`` dim, no coordinates."""
    xr = pytest.importorskip("xarray")
    import numpy as np

    return xr.Dataset({"t2m": ("values", np.zeros(size))})


def test_lonlat_is_attached_to_an_unstructured_field(tmp_path):
    import numpy as np

    from foehn.icon import attach_lonlat

    lat, lon = np.arange(4.0), np.arange(4.0) + 10
    with patch("foehn.icon.unstructured_lonlat", return_value=(lat, lon)):
        ds = attach_lonlat(_unstructured(), "forecast_icon_ch1", Workspace(tmp_path), fetcher=_fake(), what="x")

    assert ds["lat"].dims == ("values",)
    assert ds["lat"].attrs["standard_name"] == "latitude"
    assert ds["lon"].values.tolist() == [10, 11, 12, 13]


def test_a_grid_that_is_not_unstructured_is_returned_untouched(tmp_path):
    """NetCDF and radar already carry their coordinates; there is nothing to join."""
    xr = pytest.importorskip("xarray")
    import numpy as np

    from foehn.icon import attach_lonlat

    gridded = xr.Dataset({"v": (("y", "x"), np.zeros((2, 2)))})
    with patch("foehn.icon.unstructured_lonlat") as looked_up:
        result = attach_lonlat(gridded, "forecast_icon_ch1", Workspace(tmp_path), fetcher=_fake(), what="x")

    assert looked_up.call_count == 0
    assert result.identical(gridded)


def test_a_wrong_sized_constants_grid_is_ignored_rather_than_broadcast(tmp_path):
    """Coordinates from a different grid would silently mislabel every cell."""
    import numpy as np

    from foehn.icon import attach_lonlat

    mismatched = (np.arange(9.0), np.arange(9.0))
    with patch("foehn.icon.unstructured_lonlat", return_value=mismatched):
        ds = attach_lonlat(_unstructured(4), "forecast_icon_ch1", Workspace(tmp_path), fetcher=_fake(), what="x")

    assert "lat" not in ds.coords


def test_an_unreachable_constants_file_warns_and_returns_the_bare_grid(tmp_path):
    """Offline must not fail an otherwise-good read — the field is still usable."""
    from foehn.icon import attach_lonlat

    with (
        patch("foehn.icon.unstructured_lonlat", side_effect=FetchError("offline")),
        pytest.warns(UserWarning, match="Could not attach ICON lat/lon"),
    ):
        ds = attach_lonlat(
            _unstructured(), "forecast_icon_ch1", Workspace(tmp_path), fetcher=_fake(), what="no coordinates."
        )

    assert "lat" not in ds.coords
    assert "t2m" in ds.data_vars


def test_coordinates_split_across_two_messages_are_both_picked_up(tmp_path):
    """cfgrib splits a GRIB into one Dataset per message shape — tlat and tlon can land apart."""
    pytest.importorskip("xarray")
    cfgrib = pytest.importorskip("cfgrib")
    import numpy as np
    import xarray as xr

    lat_msg = xr.Dataset({"tlat": ("values", np.array([46.0, 47.0]))})
    lon_msg = xr.Dataset({"tlon": ("values", np.array([7.0, 8.0]))})
    path = tmp_path / "hc.grib2"
    path.write_bytes(b"x")

    with (
        patch("foehn.icon.constants_file", return_value=path),
        patch.object(cfgrib, "open_datasets", return_value=[lat_msg, lon_msg]),
    ):
        lat, lon = unstructured_lonlat("forecast_icon_ch1", Workspace(tmp_path), fetcher=_fake())

    assert list(lat) == [46.0, 47.0]
    assert list(lon) == [7.0, 8.0]


def test_a_constants_file_already_on_disk_under_its_real_name_is_not_refetched(tmp_path):
    """The download writes it under the asset's own name; the next call must find it there."""
    out_dir = tmp_path / "bronze" / "forecast_icon_ch1"
    out_dir.mkdir(parents=True)
    (out_dir / "horizontal_constants_icon-ch1-eps.grib2").write_bytes(b"x")

    fake = _fake()
    fake.any_collection = {
        "assets": {"horizontal_constants_icon-ch1-eps.grib2": {"href": "https://data.geo.admin.ch/x/hc.grib2"}}
    }

    path = constants_file("forecast_icon_ch1", Workspace(tmp_path), fetcher=fake)

    assert fake.streams == []
    assert path.name == "horizontal_constants_icon-ch1-eps.grib2"
