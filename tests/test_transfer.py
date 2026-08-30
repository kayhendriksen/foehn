"""Tests at :mod:`foehn.transfer`'s interface.

The five download paths that used to carry their own copy of this loop are still
covered through their own interfaces (``test_client``, ``test_grids``). These
test the engine itself: the counting invariant, the two failure policies, ETag
bookkeeping, and destination collisions.
"""

from __future__ import annotations

import pytest

from foehn.assets import Asset
from foehn.fetch import FetchError
from foehn.transfer import DownloadResult, csv_to_disk, fetch_all, stream_to_disk
from tests.fakes import InMemoryFetcher


def asset(name: str, href: str | None = None, *, updated: str = "") -> Asset:
    return Asset.from_stac(name, {"href": href or f"https://data.geo.admin.ch/x/{name}", "updated": updated})


# --- The counting invariant ---


def test_counts_sum_to_total(tmp_path):
    fetcher = InMemoryFetcher()
    fetcher.default_body = b"payload"
    assets = [asset("a.nc"), asset("b.nc"), asset("c.nc")]
    (tmp_path / "c.nc").parent.mkdir(parents=True, exist_ok=True)

    result = fetch_all(assets, tmp_path, fetcher=fetcher)

    assert result.total_assets == 3
    assert result.downloaded + result.skipped + result.failed == result.total_assets
    assert sorted(result.filenames) == ["a.nc", "b.nc", "c.nc"]


def test_skip_rule_counts_as_skipped_not_missing(tmp_path):
    fetcher = InMemoryFetcher()
    fetcher.default_body = b"payload"
    (tmp_path / "cached.nc").write_bytes(b"already here")

    result = fetch_all(
        [asset("cached.nc"), asset("fresh.nc")],
        tmp_path,
        fetcher=fetcher,
        skip=lambda _a, path: path.exists(),
    )

    assert (result.total_assets, result.downloaded, result.skipped, result.failed) == (2, 1, 1, 0)
    assert result.filenames == ["fresh.nc"]
    # The skipped file was never fetched.
    assert fetcher.streams == ["https://data.geo.admin.ch/x/fresh.nc"]


def test_destination_collision_is_kept_once_and_counted(tmp_path):
    """Two hrefs resolving to one filename must not have two workers on one .part."""
    fetcher = InMemoryFetcher()
    fetcher.default_body = b"payload"
    duplicates = [
        asset("same.nc", "https://data.geo.admin.ch/one/same.nc"),
        asset("same.nc", "https://data.geo.admin.ch/two/same.nc"),
    ]

    result = fetch_all(duplicates, tmp_path, fetcher=fetcher)

    assert (result.total_assets, result.downloaded, result.skipped) == (2, 1, 1)
    assert len(fetcher.streams) == 1


# --- Failure policy ---


def test_on_error_count_isolates_one_failure(tmp_path):
    fetcher = InMemoryFetcher()
    fetcher.default_body = b"payload"
    fetcher.fail("https://data.geo.admin.ch/x/bad.nc")

    result = fetch_all([asset("good.nc"), asset("bad.nc")], tmp_path, fetcher=fetcher)

    assert (result.downloaded, result.failed) == (1, 1)
    assert result.filenames == ["good.nc"]
    assert (tmp_path / "good.nc").exists()


def test_on_error_raise_surfaces_the_first_failure(tmp_path):
    """A grid read cannot proceed on a partial set, so its first failure is fatal."""
    fetcher = InMemoryFetcher()
    fetcher.default_body = b"payload"
    fetcher.fail("https://data.geo.admin.ch/x/bad.nc")

    with pytest.raises(FetchError):
        fetch_all([asset("bad.nc")], tmp_path, fetcher=fetcher, on_error="raise")


# --- ETag bookkeeping ---


def test_etag_304_counts_as_skipped_and_keeps_the_file(tmp_path):
    fetcher = InMemoryFetcher()
    fetcher.add_body("https://data.geo.admin.ch/x/s.csv", b"col\n1\n", etag="v1")
    (tmp_path / "s.csv").write_bytes(b"col\n1\n")
    etags = {"https://data.geo.admin.ch/x/s.csv": "v1"}

    result = fetch_all([asset("s.csv")], tmp_path, fetcher=fetcher, write=csv_to_disk, etags=etags)

    assert (result.downloaded, result.skipped) == (0, 1)
    assert etags == {"https://data.geo.admin.ch/x/s.csv": "v1"}


def test_stored_etag_is_ignored_when_the_local_file_is_gone(tmp_path):
    """A 304 must never report "skipped" over a file that no longer exists."""
    fetcher = InMemoryFetcher()
    fetcher.add_body("https://data.geo.admin.ch/x/s.csv", b"col\n1\n", etag="v1")
    etags = {"https://data.geo.admin.ch/x/s.csv": "v1"}

    result = fetch_all([asset("s.csv")], tmp_path, fetcher=fetcher, write=csv_to_disk, etags=etags)

    assert result.downloaded == 1
    assert (tmp_path / "s.csv").read_bytes() == b"col\n1\n"


def test_a_200_without_an_etag_drops_the_stored_one(tmp_path):
    """Keeping it would re-send If-None-Match forever, so the asset would never skip."""
    fetcher = InMemoryFetcher()
    fetcher.add_body("https://data.geo.admin.ch/x/s.csv", b"col\n1\n")  # no ETag
    (tmp_path / "s.csv").write_bytes(b"stale")
    etags = {"https://data.geo.admin.ch/x/s.csv": "v-old"}

    fetch_all([asset("s.csv")], tmp_path, fetcher=fetcher, write=csv_to_disk, etags=etags)

    assert etags == {}


def test_csv_writer_normalises_windows_1252_to_utf8(tmp_path):
    fetcher = InMemoryFetcher()
    fetcher.add_body("https://data.geo.admin.ch/x/s.csv", "name\nZ\xfcrich\n".encode("windows-1252"))

    fetch_all([asset("s.csv")], tmp_path, fetcher=fetcher, write=csv_to_disk)

    assert (tmp_path / "s.csv").read_bytes().decode("utf-8") == "name\nZürich\n"


# --- Result composition ---


def test_results_add(tmp_path):
    """The standard CSV path reports its metadata and data passes as one result."""
    meta = DownloadResult(total_assets=1, downloaded=1, filenames=["m.csv"])
    data = DownloadResult(total_assets=2, downloaded=1, skipped=1, failed=1, filenames=["a.csv"])

    combined = meta + data

    assert (combined.total_assets, combined.downloaded, combined.skipped, combined.failed) == (3, 2, 1, 1)
    assert combined.filenames == ["m.csv", "a.csv"]


def test_empty_input_is_a_noop(tmp_path):
    result = fetch_all([], tmp_path, fetcher=InMemoryFetcher(), write=stream_to_disk)
    assert result == DownloadResult()
