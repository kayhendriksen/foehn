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
