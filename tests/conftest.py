"""Shared fixtures for foehn tests."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _clear_process_caches():
    """Reset foehn's process-level memoisation between tests.

    ``grids`` memoises STAC listings and parsed ICON coordinates for the life of
    the process. Left alone, one test's cached listing is served to the next,
    which makes results depend on test order.
    """
    from foehn.grids import _ICON_COORDS_CACHE, _LISTING_CACHE

    _LISTING_CACHE.clear()
    _ICON_COORDS_CACHE.clear()
    yield
    _LISTING_CACHE.clear()
    _ICON_COORDS_CACHE.clear()
