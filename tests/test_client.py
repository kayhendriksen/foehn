"""Tests for state management and HTTP download functions."""

import io
import threading
import zipfile
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from foehn.client import (
    DownloadResult,
    _safe_extract_zip,
    _thread_local_session,
    download_climate_normals_zip,
    download_collection,
    download_grib2,
    download_metadata,
    download_netcdf,
    load_etags,
    load_last_run,
    save_etags,
    save_last_run,
)

# --- Helpers ---


def _csv_response(content=b"a;b\n1;2\n", etag=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.headers = {"ETag": etag} if etag else {}
    resp.raise_for_status = MagicMock()
    return resp


def _stream_response(chunks=(b"data",)):
    """Mock for `with requests.get(..., stream=True) as resp:` pattern."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_content.return_value = iter(chunks)
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _stac_item(asset_url, updated="2026-01-01T00:00:00Z"):
    return {"id": "item-1", "assets": {"data": {"href": asset_url}}, "properties": {"updated": updated}}


def _as_session_cm(mock_retry):
    """Make a patched ``_retry_session`` mock behave like a real Session.

    ``requests.Session.__enter__`` returns ``self``, so code using
    ``with _retry_session() as s:`` calls ``s.get`` on the session itself. Wire
    the mock's ``__enter__`` to return the session mock so ``.get`` assertions
    work whether the call site uses the context manager or the session directly.
    """
    mock_retry.return_value.__enter__.return_value = mock_retry.return_value
    return mock_retry.return_value.get


# --- _thread_local_session ---


def test_thread_local_session_same_thread_returns_same_session():
    get = _thread_local_session()
    assert get() is get()


def test_thread_local_session_different_threads_get_different_sessions():
    get = _thread_local_session()
    n = 4
    barrier = threading.Barrier(n)
    sessions: list[requests.Session] = []
    lock = threading.Lock()

    def grab():
        # Hold all threads here until they're all alive — keeps thread IDs distinct.
        barrier.wait()
        s = get()
        with lock:
            sessions.append(s)

    threads = [threading.Thread(target=grab) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each thread got a Session, and the per-thread instances are distinct.
    assert len(sessions) == n
    assert len({id(s) for s in sessions}) == n


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


# --- download_collection ---


@patch("foehn.client.get_collection_items")
@patch("foehn.client._retry_session")
def test_download_collection_saves_csv(mock_retry, mock_items, tmp_path):
    mock_get = mock_retry.return_value.get
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    mock_items.return_value = [_stac_item(url)]
    mock_get.return_value = _csv_response(b"station_abbr;value\nTST;1.0\n")

    result = download_collection("smn", tmp_path / "bronze")

    assert (tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv").exists()
    assert isinstance(result, DownloadResult)
    assert result.downloaded == 1
    assert result.skipped == 0
    assert result.filenames == ["ogd-smn_tst_d_recent.csv"]


@patch("foehn.client.get_collection_items")
@patch("foehn.client._retry_session")
def test_download_collection_detects_csv_with_query_string(mock_retry, mock_items, tmp_path):
    # Regression: a CSV href carrying a query string (e.g. ?token=...) must not be
    # skipped by the asset filter, which once gated on raw href.endswith(".csv").
    mock_get = mock_retry.return_value.get
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv?token=abc123"
    mock_items.return_value = [_stac_item(url)]
    mock_get.return_value = _csv_response(b"station_abbr;value\nTST;1.0\n")

    result = download_collection("smn", tmp_path / "bronze")

    # Detected, downloaded, and saved under the query-stripped filename.
    assert (tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv").exists()
    assert result.downloaded == 1
    assert result.filenames == ["ogd-smn_tst_d_recent.csv"]
    # The "recent" time slice is still parsed from the query-string URL.
    assert mock_get.call_args[0][0] == url


@patch("foehn.client.get_collection_items")
@patch("foehn.client._retry_session")
def test_download_collection_re_encodes_to_utf8(mock_retry, mock_items, tmp_path):
    mock_get = mock_retry.return_value.get
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    mock_items.return_value = [_stac_item(url)]
    # Windows-1252 encoded content (ä = 0xe4)
    mock_get.return_value = _csv_response(b"col\n\xe4\n")

    download_collection("smn", tmp_path / "bronze")

    content = (tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv").read_text(encoding="utf-8")
    assert "ä" in content


@patch("foehn.client.get_collection_items")
@patch("foehn.client._retry_session")
def test_download_collection_saves_etag(mock_retry, mock_items, tmp_path):
    mock_get = mock_retry.return_value.get
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    mock_items.return_value = [_stac_item(url)]
    mock_get.return_value = _csv_response(etag='"v1"')

    download_collection("smn", tmp_path / "bronze")

    etags = load_etags(tmp_path)
    assert etags.get(url) == '"v1"'


@patch("foehn.client.get_collection_items")
@patch("foehn.client._retry_session")
def test_download_collection_sends_if_none_match(mock_retry, mock_items, tmp_path):
    mock_get = mock_retry.return_value.get
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    mock_items.return_value = [_stac_item(url)]

    # Pre-seed an ETag and create the file so the cache path is taken
    save_etags(tmp_path, {url: '"old"'})
    out = tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("existing")

    mock_get.return_value = _csv_response(status_code=304)

    download_collection("smn", tmp_path / "bronze")

    _, kwargs = mock_get.call_args
    assert kwargs["headers"].get("If-None-Match") == '"old"'


@patch("foehn.client.get_collection_items")
@patch("foehn.client._retry_session")
def test_download_collection_skips_304(mock_retry, mock_items, tmp_path):
    mock_get = mock_retry.return_value.get
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    mock_items.return_value = [_stac_item(url)]
    mock_get.return_value = _csv_response(status_code=304)

    # File does not get created when server returns 304
    download_collection("smn", tmp_path / "bronze")

    assert not (tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv").exists()


@patch("foehn.client.get_collection_items")
@patch("foehn.client._retry_session")
def test_download_collection_prunes_stale_etags(mock_retry, mock_items, tmp_path):
    """A clean full run drops ETags for assets gone upstream — scoped to this collection."""
    mock_get = mock_retry.return_value.get
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
    mock_items.return_value = [item]
    mock_get.return_value = _csv_response(etag='"v1"')

    download_collection("smn", tmp_path / "bronze", data_types=["recent"])

    etags = load_etags(tmp_path)
    assert stale not in etags  # gone upstream → pruned
    assert etags[historical] == '"hist"'  # still listed (other slice) → kept
    assert etags[other] == '"keep"'  # different collection → untouched
    assert etags[current] == '"v1"'


@patch("foehn.client.get_collection_items")
@patch("foehn.client._retry_session")
def test_download_collection_incremental_run_does_not_prune(mock_retry, mock_items, tmp_path):
    """With ``since`` the item list is partial — never prune from it."""
    mock_get = mock_retry.return_value.get
    stale = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/tst/ogd-smn_tst_d_now.csv"
    save_etags(tmp_path, {stale: '"keep"'})
    url = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/tst/ogd-smn_tst_d_recent.csv"
    mock_items.return_value = [_stac_item(url, updated="2026-02-01T00:00:00Z")]
    mock_get.return_value = _csv_response(etag='"v1"')

    download_collection("smn", tmp_path / "bronze", since="2026-01-01T00:00:00Z")

    assert load_etags(tmp_path)[stale] == '"keep"'


@patch("foehn.client.get_collection_items")
@patch("foehn.client._retry_session")
def test_download_collection_since_filter(mock_retry, mock_items, tmp_path):
    """Items older than `since` should be skipped without any HTTP call."""
    mock_get = mock_retry.return_value.get
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    mock_items.return_value = [_stac_item(url, updated="2025-06-01T00:00:00Z")]
    mock_get.return_value = _csv_response()

    result = download_collection("smn", tmp_path / "bronze", since="2026-01-01T00:00:00Z")

    mock_get.assert_not_called()
    assert result.downloaded == 0
    assert result.filenames == []


@patch("foehn.client.get_collection_items")
@patch("foehn.client._retry_session")
def test_download_collection_resilient_to_single_failure(mock_retry, mock_items, tmp_path):
    """One failing asset must not abort the batch or discard the others' ETags."""
    good = "https://data.geo.admin.ch/ogd-smn_aaa_d_recent.csv"
    bad = "https://data.geo.admin.ch/ogd-smn_bbb_d_recent.csv"
    mock_items.return_value = [
        _stac_item(good),
        {"id": "item-2", "assets": {"data": {"href": bad}}, "properties": {"updated": "2026-01-01T00:00:00Z"}},
    ]

    def fake_get(url, **kwargs):
        if "bbb" in url:
            raise requests.exceptions.ConnectionError("boom")
        return _csv_response(b"station_abbr;value\nAAA;1.0\n", etag="etag-aaa")

    mock_retry.return_value.get.side_effect = fake_get

    result = download_collection("smn", tmp_path / "bronze")

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


@patch("foehn.client.get_collection_metadata")
@patch("foehn.client._retry_session")
def test_download_metadata_saves_csv(mock_retry, mock_meta, tmp_path):
    mock_get = mock_retry.return_value.get
    mock_meta.return_value = {"assets": {"stations": {"href": "https://data.geo.admin.ch/stations.csv"}}}
    mock_get.return_value = _csv_response(b"id;name\nTST;Test Station\n")

    result = download_metadata("smn", tmp_path / "bronze")

    assert (tmp_path / "bronze" / "smn" / "stations.csv").exists()
    assert result.downloaded == 1
    assert result.filenames == ["stations.csv"]


@patch("foehn.client.get_collection_metadata")
@patch("foehn.client._retry_session")
def test_download_metadata_skips_non_csv_assets(mock_retry, mock_meta, tmp_path):
    mock_get = mock_retry.return_value.get
    mock_meta.return_value = {"assets": {"readme": {"href": "https://data.geo.admin.ch/README.pdf"}}}

    download_metadata("smn", tmp_path / "bronze")

    mock_get.assert_not_called()


# --- download_climate_normals_zip ---


def _make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


@patch("foehn.client._retry_session")
def test_download_climate_normals_zip_extracts_files(mock_retry, tmp_path):
    mock_get = _as_session_cm(mock_retry)
    zip_bytes = _make_zip({"sample.txt": b"data"})
    mock_get.return_value = _stream_response(chunks=(zip_bytes,))

    download_climate_normals_zip(tmp_path / "bronze")

    assert (tmp_path / "bronze" / "climate_normals" / "normwerte.zip").exists()
    assert (tmp_path / "bronze" / "climate_normals" / "sample.txt").exists()


@patch("foehn.client._retry_session")
def test_download_climate_normals_zip_skips_if_extracted(mock_retry, tmp_path):
    mock_get = mock_retry.return_value.get
    out_dir = tmp_path / "bronze" / "climate_normals"
    out_dir.mkdir(parents=True)
    (out_dir / "sample.txt").write_bytes(b"extracted")

    download_climate_normals_zip(tmp_path / "bronze")

    mock_get.assert_not_called()


@patch("foehn.client._retry_session")
def test_download_climate_normals_zip_redownloads_if_not_extracted(mock_retry, tmp_path):
    """A ZIP left by a run that died before extraction must not be mistaken for done."""
    mock_get = _as_session_cm(mock_retry)
    out_dir = tmp_path / "bronze" / "climate_normals"
    out_dir.mkdir(parents=True)
    (out_dir / "normwerte.zip").write_bytes(b"partial-or-unextracted")

    zip_bytes = _make_zip({"sample.txt": b"data"})
    mock_get.return_value = _stream_response(chunks=(zip_bytes,))

    download_climate_normals_zip(tmp_path / "bronze")

    mock_get.assert_called_once()
    assert (out_dir / "sample.txt").exists()


@patch("foehn.client._retry_session")
def test_download_climate_normals_zip_rejects_decompression_bomb(mock_retry, tmp_path, monkeypatch):
    """An archive declaring more decompressed bytes than the cap must not extract."""
    monkeypatch.setattr("foehn.client._MAX_ZIP_EXTRACT_BYTES", 10)
    mock_get = _as_session_cm(mock_retry)
    zip_bytes = _make_zip({"sample.txt": b"x" * 1024})
    mock_get.return_value = _stream_response(chunks=(zip_bytes,))

    with pytest.raises(ValueError, match="decompressed"):
        download_climate_normals_zip(tmp_path / "bronze")

    assert not (tmp_path / "bronze" / "climate_normals" / "sample.txt").exists()


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    zip_path = tmp_path / "evil.zip"
    zip_path.write_bytes(_make_zip({"../evil.txt": b"x"}))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="Unsafe path"):
        _safe_extract_zip(zip_path, out_dir)

    assert not (tmp_path / "evil.txt").exists()


@patch("foehn.client._retry_session")
def test_download_climate_normals_zip_force_redownloads(mock_retry, tmp_path):
    mock_get = _as_session_cm(mock_retry)
    out_dir = tmp_path / "bronze" / "climate_normals"
    out_dir.mkdir(parents=True)
    (out_dir / "old.txt").write_bytes(b"stale")

    zip_bytes = _make_zip({"new.txt": b"fresh"})
    mock_get.return_value = _stream_response(chunks=(zip_bytes,))

    download_climate_normals_zip(tmp_path / "bronze", force=True)

    mock_get.assert_called_once()
    assert (out_dir / "new.txt").exists()


# --- download_grib2 ---


@patch("foehn.client._retry_session")
def test_download_grib2_saves_binary(mock_retry, tmp_path):
    mock_get = _as_session_cm(mock_retry)
    items_resp = MagicMock()
    items_resp.raise_for_status = MagicMock()
    items_resp.json.return_value = {
        "features": [
            {"id": "f1", "assets": {"data": {"href": "https://data.geo.admin.ch/forecast.grib2"}}, "properties": {}}
        ]
    }
    file_resp = _stream_response(chunks=(b"GRIB", b"data"))

    mock_get.side_effect = [items_resp, file_resp]

    result = download_grib2("forecast_icon_ch1", tmp_path / "bronze")

    assert (tmp_path / "bronze" / "forecast_icon_ch1" / "forecast.grib2").exists()
    assert result.downloaded == 1
    assert result.filenames == ["forecast.grib2"]


@patch("foehn.client._retry_session")
def test_download_grib2_skips_existing_file(mock_retry, tmp_path):
    mock_get = _as_session_cm(mock_retry)
    items_resp = MagicMock()
    items_resp.raise_for_status = MagicMock()
    items_resp.json.return_value = {
        "features": [
            {"id": "f1", "assets": {"data": {"href": "https://data.geo.admin.ch/forecast.grib2"}}, "properties": {}}
        ]
    }
    mock_get.return_value = items_resp

    out_dir = tmp_path / "bronze" / "forecast_icon_ch1"
    out_dir.mkdir(parents=True)
    (out_dir / "forecast.grib2").write_bytes(b"existing")

    download_grib2("forecast_icon_ch1", tmp_path / "bronze")

    # Only 1 call for the STAC items page, none for the file
    assert mock_get.call_count == 1


# --- download_netcdf ---


@patch("foehn.client.get_collection_items")
@patch("foehn.client._retry_session")
def test_download_netcdf_saves_nc_file(mock_retry, mock_items, tmp_path):
    mock_get = mock_retry.return_value.get
    mock_items.return_value = [
        {"id": "g1", "assets": {"data": {"href": "https://data.geo.admin.ch/grid.nc"}}, "properties": {}}
    ]
    mock_get.return_value = _stream_response(chunks=(b"\x89HDF",))

    result = download_netcdf("surface_derived_grid", tmp_path / "bronze")

    assert (tmp_path / "bronze" / "surface_derived_grid" / "grid.nc").exists()
    assert result.downloaded == 1
    assert result.filenames == ["grid.nc"]


@patch("foehn.client.get_collection_items")
@patch("foehn.client._retry_session")
def test_download_netcdf_skips_existing_file(mock_retry, mock_items, tmp_path):
    mock_get = mock_retry.return_value.get
    mock_items.return_value = [
        {"id": "g1", "assets": {"data": {"href": "https://data.geo.admin.ch/grid.nc"}}, "properties": {}}
    ]

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / "grid.nc").write_bytes(b"existing")

    result = download_netcdf("surface_derived_grid", tmp_path / "bronze")

    mock_get.assert_not_called()
    assert result.downloaded == 0
    assert result.skipped == 1


@patch("foehn.client.get_collection_items")
@patch("foehn.client._retry_session")
def test_download_netcdf_since_filter(mock_retry, mock_items, tmp_path):
    """Items older than `since` should be skipped without any HTTP call."""
    mock_get = mock_retry.return_value.get
    mock_items.return_value = [
        {
            "id": "g1",
            "assets": {"data": {"href": "https://data.geo.admin.ch/grid.nc"}},
            "properties": {"updated": "2025-06-01T00:00:00Z"},
        }
    ]

    result = download_netcdf("surface_derived_grid", tmp_path / "bronze", since="2026-01-01T00:00:00Z")

    mock_get.assert_not_called()
    assert result.downloaded == 0
    assert result.filenames == []
