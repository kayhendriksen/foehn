"""Tests for state management and the download functions.

These cross the network port with an :class:`InMemoryFetcher` and assert on what
lands on disk and in ``DownloadResult`` — not on which HTTP calls were made. The
few that do check traffic are the ones where traffic *is* the behaviour: an ETag
skip, a ``since`` filter, first-page-only listing. Session lifetime and retry
policy are the adapter's, and are tested in tests/test_fetch.py.
"""

import io
import zipfile
from datetime import UTC, datetime

import pytest

from foehn import registry
from foehn.client import (
    DownloadResult,
    _safe_extract_zip,
    download_metadata,
    download_normals_zip,
    load_etags,
    load_last_run,
    save_etags,
    save_last_run,
)
from foehn.fetch import FetchError
from tests.fakes import InMemoryFetcher

# --- Helpers ---


def _fake(items=None, *, collection=None, body=b"a;b\n1;2\n"):
    """A fetcher listing *items* (or serving *collection*), with a default body.

    The CSV kinds fetch collection-level metadata in the same pass as their item
    assets, so an empty collection payload is the default — a test that cares
    about the metadata pass passes its own.
    """
    fake = InMemoryFetcher()
    if items is not None:
        fake.any_items = items
    fake.any_collection = collection if collection is not None else {"assets": {}}
    fake.default_body = body
    return fake


def _serve(fake, content=b"a;b\n1;2\n", etag=None, status_code=200):
    """Serve *content* for every URL, optionally with an ETag.

    ``status_code=304`` in the old mocks meant "the server says unchanged"; here
    that is expressed properly — the fetcher answers 304 when the caller sends a
    matching ETag, so the test seeds one.
    """
    fake.default_body = content
    if etag is not None:
        fake.etags["*"] = etag
        fake.default_etag = etag


def _stac_item(asset_url, updated="2026-01-01T00:00:00Z"):
    return {"id": "item-1", "assets": {"data": {"href": asset_url}}, "properties": {"updated": updated}}


# --- State management ---


def test_load_etags_missing_file_returns_empty(tmp_path):
    assert load_etags(tmp_path) == {}


def test_save_and_load_etags_roundtrip(tmp_path):
    etags = {"https://data.geo.admin.ch/file.csv": '"abc123"'}
    save_etags(tmp_path, etags)
    assert load_etags(tmp_path) == etags


def test_save_etags_creates_parent_dirs(tmp_path):
    nested = tmp_path / "a" / "b"
    save_etags(nested, {"k": "v"})
    assert (nested / "_etags.json").exists()


def test_save_etags_overwrites_existing(tmp_path):
    save_etags(tmp_path, {"k": "old"})
    save_etags(tmp_path, {"k": "new"})
    assert load_etags(tmp_path) == {"k": "new"}


def test_load_etags_corrupt_file_returns_empty(tmp_path):
    """A torn write must not brick subsequent runs — treat as empty state."""
    (tmp_path / "_etags.json").write_text('{"truncated": ')
    assert load_etags(tmp_path) == {}


def test_load_last_run_corrupt_file_returns_none(tmp_path):
    (tmp_path / "_last_run.json").write_text("not json")
    assert load_last_run(tmp_path) is None


def test_load_last_run_missing_file_returns_none(tmp_path):
    assert load_last_run(tmp_path) is None


def test_save_and_load_last_run_roundtrip(tmp_path):
    save_last_run(tmp_path)
    timestamp = load_last_run(tmp_path)
    assert timestamp is not None
    dt = datetime.fromisoformat(timestamp)
    assert dt.tzinfo is not None


def test_save_last_run_is_recent(tmp_path):
    before = datetime.now(UTC)
    save_last_run(tmp_path)
    after = datetime.now(UTC)

    saved = datetime.fromisoformat(load_last_run(tmp_path))
    assert before <= saved <= after


# --- the CSV listing path (registry: STANDARD_CSV / PREAMBLE_CSV) ---


def test_csv_download_saves_csv(tmp_path):
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url)])
    _serve(fake, b"station_abbr;value\nTST;1.0\n")

    result = registry.download("smn", tmp_path / "bronze", fetcher=fake)

    assert (tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv").exists()
    assert isinstance(result, DownloadResult)
    assert result.downloaded == 1
    assert result.skipped == 0
    assert result.filenames == ["ogd-smn_tst_d_recent.csv"]


def test_csv_download_detects_csv_with_query_string(tmp_path):
    # Regression: a CSV href carrying a query string (e.g. ?token=...) must not be
    # skipped by the asset filter, which once gated on raw href.endswith(".csv").
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv?token=abc123"
    fake = _fake([_stac_item(url)])
    _serve(fake, b"station_abbr;value\nTST;1.0\n")

    result = registry.download("smn", tmp_path / "bronze", fetcher=fake)

    # Detected, downloaded, and saved under the query-stripped filename.
    assert (tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv").exists()
    assert result.downloaded == 1
    assert result.filenames == ["ogd-smn_tst_d_recent.csv"]
    # The "recent" time slice is still parsed from the query-string URL.
    assert fake.gets[-1] == url


def test_csv_download_re_encodes_to_utf8(tmp_path):
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url)])
    # Windows-1252 encoded content (ä = 0xe4)
    _serve(fake, b"col\n\xe4\n")

    registry.download("smn", tmp_path / "bronze", fetcher=fake)

    content = (tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv").read_text(encoding="utf-8")
    assert "ä" in content


def test_csv_download_saves_etag(tmp_path):
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url)])
    _serve(fake, etag='"v1"')

    registry.download("smn", tmp_path / "bronze", fetcher=fake)

    etags = load_etags(tmp_path)
    assert etags.get(url) == '"v1"'


def test_csv_download_skips_unchanged_asset(tmp_path):
    """A stored ETag plus a file on disk means a conditional request and a skip."""
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url)])
    _serve(fake, etag='"v1"')
    save_etags(tmp_path, {url: '"v1"'})
    out = tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("existing")

    result = registry.download("smn", tmp_path / "bronze", fetcher=fake)

    assert result.skipped == 1
    assert result.downloaded == 0
    assert out.read_text() == "existing"  # untouched


def test_csv_download_ignores_stored_etag_when_file_is_gone(tmp_path):
    """Otherwise a 304 would report 'skipped' over a file that no longer exists."""
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url)])
    _serve(fake, etag='"v1"')
    save_etags(tmp_path, {url: '"v1"'})  # ETag remembered, file deleted

    result = registry.download("smn", tmp_path / "bronze", fetcher=fake)

    assert result.downloaded == 1
    assert (tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv").exists()


def test_csv_download_drops_etag_when_server_stops_sending_one(tmp_path):
    """A 200 with no ETag header must clear the stored one.

    Keeping it would re-send the stale value as If-None-Match on every later
    run, so the asset would download every time and never once be skipped.
    """
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url)])
    save_etags(tmp_path, {url: '"stale-etag"'})

    _serve(fake, b"station_abbr;value\nTST;1.0\n", etag=None)
    registry.download("smn", tmp_path / "bronze", fetcher=fake)

    assert url not in load_etags(tmp_path)


def test_csv_download_deduplicates_same_destination(tmp_path):
    """Two hrefs resolving to one filename are fetched once, not concurrently.

    Both workers would otherwise stream into the same ``.part`` file.
    """
    base = "https://data.geo.admin.ch"
    fake = _fake(
        [
            _stac_item(f"{base}/a/ogd-smn_tst_d_recent.csv"),
            _stac_item(f"{base}/b/ogd-smn_tst_d_recent.csv"),
        ]
    )
    _serve(fake, b"station_abbr;value\nTST;1.0\n")

    result = registry.download("smn", tmp_path / "bronze", fetcher=fake)

    assert result.downloaded == 1
    assert len(fake.gets) == 1


def test_csv_download_prunes_stale_etags(tmp_path):
    """A clean full run drops ETags for assets gone upstream — scoped to this collection."""
    base = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/tst"
    current = f"{base}/ogd-smn_tst_d_recent.csv"
    historical = f"{base}/ogd-smn_tst_d_historical.csv"  # listed, filtered out by data_types
    stale = f"{base}/ogd-smn_tst_d_now.csv"  # no longer listed upstream
    other = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/tst/ogd-nime_tst_d_recent.csv"
    save_etags(tmp_path, {stale: '"gone"', historical: '"hist"', other: '"keep"'})

    item = {
        "id": "tst",
        "assets": {"a": {"href": current}, "b": {"href": historical}},
        "properties": {"updated": "2026-01-01T00:00:00Z"},
    }
    fake = _fake([item])
    _serve(fake, etag='"v1"')

    registry.download("smn", tmp_path / "bronze", time_slice=["recent"], fetcher=fake)

    etags = load_etags(tmp_path)
    assert stale not in etags  # gone upstream → pruned
    assert etags[historical] == '"hist"'  # still listed (other slice) → kept
    assert etags[other] == '"keep"'  # different collection → untouched
    assert etags[current] == '"v1"'


def test_csv_download_incremental_run_does_not_prune(tmp_path):
    """With ``since`` the item list is partial — never prune from it."""
    stale = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/tst/ogd-smn_tst_d_now.csv"
    save_etags(tmp_path, {stale: '"keep"'})
    url = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/tst/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url, updated="2026-02-01T00:00:00Z")])
    _serve(fake, etag='"v1"')

    registry.download("smn", tmp_path / "bronze", since="2026-01-01T00:00:00Z", fetcher=fake)

    assert load_etags(tmp_path)[stale] == '"keep"'


def test_csv_download_since_filter(tmp_path):
    """Items older than `since` should be skipped without any HTTP call."""
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url, updated="2025-06-01T00:00:00Z")])
    _serve(
        fake,
    )

    result = registry.download("smn", tmp_path / "bronze", since="2026-01-01T00:00:00Z", fetcher=fake)

    assert fake.gets == [] and fake.streams == []
    assert result.downloaded == 0
    assert result.filenames == []


def test_csv_download_resilient_to_single_failure(tmp_path):
    """One failing asset must not abort the batch or discard the others' ETags."""
    good = "https://data.geo.admin.ch/ogd-smn_aaa_d_recent.csv"
    bad = "https://data.geo.admin.ch/ogd-smn_bbb_d_recent.csv"
    fake = _fake(
        [
            _stac_item(good),
            {"id": "item-2", "assets": {"data": {"href": bad}}, "properties": {"updated": "2026-01-01T00:00:00Z"}},
        ]
    )

    _serve(fake, b"station_abbr;value\nAAA;1.0\n", etag="etag-aaa")
    fake.fail(bad, FetchError("boom"))

    result = registry.download("smn", tmp_path / "bronze", fetcher=fake)

    # The good asset still downloaded; the bad one is counted, not raised.
    assert result.downloaded == 1
    assert result.failed == 1
    assert result.filenames == ["ogd-smn_aaa_d_recent.csv"]
    assert (tmp_path / "bronze" / "smn" / "ogd-smn_aaa_d_recent.csv").exists()
    # ETags for the successful asset are persisted despite the sibling failure.
    etags = load_etags(tmp_path)
    assert etags.get(good) == "etag-aaa"
    assert bad not in etags


# --- download_metadata ---


def test_download_metadata_saves_csv(tmp_path):
    fake = _fake(collection={"assets": {"stations": {"href": "https://data.geo.admin.ch/stations.csv"}}})
    _serve(fake, b"id;name\nTST;Test Station\n")

    result = download_metadata("smn", tmp_path / "bronze", fetcher=fake)

    assert (tmp_path / "bronze" / "smn" / "stations.csv").exists()
    assert result.downloaded == 1
    assert result.filenames == ["stations.csv"]


def test_download_metadata_skips_non_csv_assets(tmp_path):
    fake = _fake(collection={"assets": {"readme": {"href": "https://data.geo.admin.ch/README.pdf"}}})

    download_metadata("smn", tmp_path / "bronze", fetcher=fake)

    assert fake.gets == [] and fake.streams == []


# --- download_climate_normals_zip ---


def _make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_normals_zip_extracts_files(tmp_path):
    zip_bytes = _make_zip({"sample.txt": b"data"})
    fake = _fake(body=zip_bytes)

    download_normals_zip("climate_normals", tmp_path / "bronze", fetcher=fake)

    assert (tmp_path / "bronze" / "climate_normals" / "normwerte.zip").exists()
    assert (tmp_path / "bronze" / "climate_normals" / "sample.txt").exists()


def test_normals_zip_skips_if_extracted(tmp_path):
    fake = _fake(body=_make_zip({"sample.txt": b"data"}))
    out_dir = tmp_path / "bronze" / "climate_normals"
    out_dir.mkdir(parents=True)
    (out_dir / "sample.txt").write_bytes(b"extracted")

    download_normals_zip("climate_normals", tmp_path / "bronze", fetcher=fake)

    assert fake.gets == [] and fake.streams == []


def test_normals_zip_redownloads_if_not_extracted(tmp_path):
    """A ZIP left by a run that died before extraction must not be mistaken for done."""
    out_dir = tmp_path / "bronze" / "climate_normals"
    out_dir.mkdir(parents=True)
    (out_dir / "normwerte.zip").write_bytes(b"partial-or-unextracted")

    zip_bytes = _make_zip({"sample.txt": b"data"})
    fake = _fake(body=zip_bytes)

    download_normals_zip("climate_normals", tmp_path / "bronze", fetcher=fake)

    assert len(fake.streams) == 1
    assert (out_dir / "sample.txt").exists()


def test_normals_zip_rejects_decompression_bomb(tmp_path, monkeypatch):
    """An archive declaring more decompressed bytes than the cap must not extract."""
    monkeypatch.setattr("foehn.client._MAX_ZIP_EXTRACT_BYTES", 10)
    zip_bytes = _make_zip({"sample.txt": b"x" * 1024})
    fake = _fake(body=zip_bytes)

    with pytest.raises(ValueError, match="decompressed"):
        download_normals_zip("climate_normals", tmp_path / "bronze", fetcher=fake)

    assert not (tmp_path / "bronze" / "climate_normals" / "sample.txt").exists()


def test_safe_extract_zip_accepts_nested_members(tmp_path):
    """Legitimate nested members must extract.

    The guard used to compare strings against ``str(out_dir) + "/"``, which no
    resolved path matches on Windows (separator is "\\") — so every member of
    every archive was rejected there, including the C6 climate normals that
    bare ``foehn download`` always fetches.
    """
    zip_path = tmp_path / "ok.zip"
    zip_path.write_bytes(_make_zip({"nested/dir/sample.txt": b"data"}))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assert _safe_extract_zip(zip_path, out_dir) == 1
    assert (out_dir / "nested" / "dir" / "sample.txt").read_bytes() == b"data"


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    zip_path = tmp_path / "evil.zip"
    zip_path.write_bytes(_make_zip({"../evil.txt": b"x"}))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="Unsafe path"):
        _safe_extract_zip(zip_path, out_dir)

    assert not (tmp_path / "evil.txt").exists()


def test_normals_zip_force_redownloads(tmp_path):
    out_dir = tmp_path / "bronze" / "climate_normals"
    out_dir.mkdir(parents=True)
    (out_dir / "old.txt").write_bytes(b"stale")

    zip_bytes = _make_zip({"new.txt": b"fresh"})
    fake = _fake(body=zip_bytes)

    download_normals_zip("climate_normals", tmp_path / "bronze", force=True, fetcher=fake)

    assert len(fake.streams) == 1
    assert (out_dir / "new.txt").exists()


# --- the binary listing path (registry: GRIB2_GRID / RADAR_GRID) ---


def test_binary_download_saves_binary(tmp_path):
    href = "https://data.geo.admin.ch/forecast.grib2"
    fake = _fake([_stac_item(href)], body=b"GRIBdata")

    result = registry.download("forecast_icon_ch1", tmp_path / "bronze", fetcher=fake)

    assert (tmp_path / "bronze" / "forecast_icon_ch1" / "forecast.grib2").read_bytes() == b"GRIBdata"
    assert result.downloaded == 1
    assert result.filenames == ["forecast.grib2"]


def test_binary_download_skips_existing_file(tmp_path):
    href = "https://data.geo.admin.ch/forecast.grib2"
    fake = _fake([_stac_item(href)], body=b"GRIBdata")
    out_dir = tmp_path / "bronze" / "forecast_icon_ch1"
    out_dir.mkdir(parents=True)
    (out_dir / "forecast.grib2").write_bytes(b"existing")

    registry.download("forecast_icon_ch1", tmp_path / "bronze", fetcher=fake)

    assert fake.streams == []
    assert (out_dir / "forecast.grib2").read_bytes() == b"existing"


def test_binary_download_reads_only_the_newest_page(tmp_path):
    """These collections hold thousands of ephemeral items; walking them all is pure cost."""
    fake = _fake([_stac_item("https://data.geo.admin.ch/forecast.grib2")], body=b"GRIBdata")

    registry.download("forecast_icon_ch1", tmp_path / "bronze", fetcher=fake)

    assert [max_items for *_, max_items in fake.listings] == [100]


# --- the static listing path (registry: NETCDF_GRID) ---


def test_static_download_saves_nc_file(tmp_path):
    fake = _fake([{"id": "g1", "assets": {"data": {"href": "https://data.geo.admin.ch/grid.nc"}}, "properties": {}}])
    fake.default_body = b"\x89HDF"

    result = registry.download("surface_derived_grid", tmp_path / "bronze", fetcher=fake)

    assert (tmp_path / "bronze" / "surface_derived_grid" / "grid.nc").exists()
    assert result.downloaded == 1
    assert result.filenames == ["grid.nc"]


def test_static_download_skips_existing_file(tmp_path):
    fake = _fake([{"id": "g1", "assets": {"data": {"href": "https://data.geo.admin.ch/grid.nc"}}, "properties": {}}])

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / "grid.nc").write_bytes(b"existing")

    result = registry.download("surface_derived_grid", tmp_path / "bronze", fetcher=fake)

    assert fake.gets == [] and fake.streams == []
    assert result.downloaded == 0
    assert result.skipped == 1


def test_static_download_since_filter(tmp_path):
    """Items older than `since` should be skipped without any HTTP call."""
    fake = _fake(
        [
            {
                "id": "g1",
                "assets": {"data": {"href": "https://data.geo.admin.ch/grid.nc"}},
                "properties": {"updated": "2025-06-01T00:00:00Z"},
            }
        ]
    )

    result = registry.download("surface_derived_grid", tmp_path / "bronze", since="2026-01-01T00:00:00Z", fetcher=fake)

    assert fake.gets == [] and fake.streams == []
    assert result.downloaded == 0
    assert result.filenames == []
