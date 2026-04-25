"""Tests for state management and HTTP download functions."""

import io
import threading
import zipfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import requests

from foehn.client import (
    DownloadResult,
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


def test_load_last_run_missing_file_returns_none(tmp_path):
    assert load_last_run(tmp_path) is None


def test_save_and_load_last_run_roundtrip(tmp_path):
    save_last_run(tmp_path)
    timestamp = load_last_run(tmp_path)
    assert timestamp is not None
    dt = datetime.fromisoformat(timestamp)
    assert dt.tzinfo is not None


def test_save_last_run_is_recent(tmp_path):
    before = datetime.now(timezone.utc)
    save_last_run(tmp_path)
    after = datetime.now(timezone.utc)

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
    mock_get = mock_retry.return_value.get
    zip_bytes = _make_zip({"sample.txt": b"data"})
    mock_get.return_value = _csv_response(content=zip_bytes)

    download_climate_normals_zip(tmp_path / "bronze")

    assert (tmp_path / "bronze" / "climate_normals" / "normwerte.zip").exists()
    assert (tmp_path / "bronze" / "climate_normals" / "sample.txt").exists()


@patch("foehn.client._retry_session")
def test_download_climate_normals_zip_skips_if_exists(mock_retry, tmp_path):
    mock_get = mock_retry.return_value.get
    out_dir = tmp_path / "bronze" / "climate_normals"
    out_dir.mkdir(parents=True)
    (out_dir / "normwerte.zip").write_bytes(b"existing")

    download_climate_normals_zip(tmp_path / "bronze")

    mock_get.assert_not_called()


@patch("foehn.client._retry_session")
def test_download_climate_normals_zip_force_redownloads(mock_retry, tmp_path):
    mock_get = mock_retry.return_value.get
    out_dir = tmp_path / "bronze" / "climate_normals"
    out_dir.mkdir(parents=True)
    (out_dir / "normwerte.zip").write_bytes(b"old")

    zip_bytes = _make_zip({"new.txt": b"fresh"})
    mock_get.return_value = _csv_response(content=zip_bytes)

    download_climate_normals_zip(tmp_path / "bronze", force=True)

    mock_get.assert_called_once()
    assert (out_dir / "new.txt").exists()


# --- download_grib2 ---


@patch("foehn.client._retry_session")
def test_download_grib2_saves_binary(mock_retry, tmp_path):
    mock_get = mock_retry.return_value.get
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
    mock_get = mock_retry.return_value.get
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
