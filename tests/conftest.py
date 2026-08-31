"""Shared fixtures for foehn tests."""

import io
import zipfile
from pathlib import Path

import pytest

import foehn.fetch
from tests.fakes import InMemoryFetcher

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# A CH2025 climate-scenario CSV: the KEY;VALUE preamble, then the real DATE;
# header on nominal 30-year dates. Shared because three test modules parse one —
# the conventions, the convert stage, and the public load path.
CLIMATE_SCENARIOS_CSV = (
    "TITLE;Climate CH2025\n"
    "VARIABLE;Daily precipitation sum\n"
    "STATION_ABBR;ABE\n"
    "GWL;GWL1.5\n"
    "\n"
    "DATE;MODEL_A;MODEL_B\n"
    "0001-01-01;0;24.2\n"
    "0001-01-02;1.5;0\n"
)


@pytest.fixture
def fetcher(monkeypatch):
    """Install an :class:`InMemoryFetcher` as the process-wide default.

    The public functions (``foehn.load``, ``download``, ``open_dataset`` …) build
    their fetcher from ``foehn.fetch.default_fetcher()`` rather than taking one,
    so this is the single place the suite substitutes the network. Internal
    functions take a fetcher explicitly — pass this one straight in.
    """
    fake = InMemoryFetcher()
    monkeypatch.setattr(foehn.fetch, "_default_fetcher", fake)
    return fake


@pytest.fixture(autouse=True)
def _clear_process_caches():
    """Reset foehn's process-level memoisation between tests.

    ``icon`` memoises parsed cell coordinates for the life of the process. Left
    alone, one test's cached value is served to the next, which makes results
    depend on test order. (Listings are memoised on the fetcher itself, so a
    fresh fetcher per test already starts cold.)
    """
    from foehn import icon

    icon.clear_cache()
    yield
    icon.clear_cache()


def make_zip(files: dict[str, bytes]) -> bytes:
    """An in-memory ZIP of *files*. Shared by the archive guards and the ZIP download paths."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


_SWISS_PROJ = (
    "+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +k_0=1 "
    "+x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs"
)


def write_odim_composite(
    path,
    quantity="ACRR",
    nodata=float("nan"),
    undetect=float("inf"),
    date="20260510",
    time="000000",
    obj="COMP",
    long_name=None,
):
    """Write a tiny ODIM-H5 Cartesian COMP composite (offline radar fixture).

    Shared because three test modules build one: the format itself, the grid
    reader that opens it, and the cube builder that stacks several.
    """
    import h5py
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.array([[0.0, 1.5, nodata], [undetect, 2.0, 3.0]], dtype="float64")
    with h5py.File(path, "w") as f:
        f.attrs["Conventions"] = "ODIM_H5/V2_4"
        what = f.create_group("what")
        what.attrs["object"] = obj
        what.attrs["date"] = date
        what.attrs["time"] = time
        where = f.create_group("where")
        where.attrs.update({"xsize": 3, "ysize": 2, "xscale": 1000.0, "yscale": 1000.0, "projdef": _SWISS_PROJ})
        where.attrs.update({"UL_lon": 2.68942, "UL_lat": 49.3744})
        d1 = f.create_group("dataset1").create_group("data1")
        d1.create_dataset("data", data=data)
        dwhat = d1.create_group("what")
        dwhat.attrs.update({"gain": 1.0, "offset": 0.0, "nodata": nodata, "undetect": undetect, "quantity": quantity})
        if long_name is not None:
            f.create_group("how").create_group("MeteoSwiss").attrs["long_name"] = long_name
