"""Shared fixtures for foehn tests."""

from pathlib import Path

import pytest

import foehn.fetch
from tests.fakes import InMemoryFetcher

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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

    ``grids`` memoises parsed ICON coordinates for the life of the process. Left
    alone, one test's cached value is served to the next, which makes results
    depend on test order. (Listings are memoised on the fetcher itself, so a
    fresh fetcher per test already starts cold.)
    """
    from foehn.grids import _ICON_COORDS_CACHE

    _ICON_COORDS_CACHE.clear()
    yield
    _ICON_COORDS_CACHE.clear()
