"""Tests for state management and HTTP download functions."""

import io
import zipfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from foehn.download import (
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


# --- State management ---


def test_load_etags_missing_file_returns_empty(tmp_path):
    assert load_etags(tmp_path) == {}


def test_save_and_load_etags_roundtrip(tmp_path):
    etags = {"https://example.com/file.csv": '"abc123"'}
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


@patch("foehn.download.get_collection_items")
@patch("foehn.download.requests.get")
def test_download_collection_saves_csv(mock_get, mock_items, tmp_path):
    url = "https://example.com/ogd-smn_tst_d_recent.csv"
    mock_items.return_value = [_stac_item(url)]
    mock_get.return_value = _csv_response(b"station_abbr;value\nTST;1.0\n")

    download_collection("smn", tmp_path / "raw")

    assert (tmp_path / "raw" / "smn" / "ogd-smn_tst_d_recent.csv").exists()


@patch("foehn.download.get_collection_items")
@patch("foehn.download.requests.get")
def test_download_collection_re_encodes_to_utf8(mock_get, mock_items, tmp_path):
    url = "https://example.com/ogd-smn_tst_d_recent.csv"
    mock_items.return_value = [_stac_item(url)]
    # Windows-1252 encoded content (ä = 0xe4)
    mock_get.return_value = _csv_response(b"col\n\xe4\n")

    download_collection("smn", tmp_path / "raw")

    content = (tmp_path / "raw" / "smn" / "ogd-smn_tst_d_recent.csv").read_text(encoding="utf-8")
    assert "ä" in content


@patch("foehn.download.get_collection_items")
@patch("foehn.download.requests.get")
def test_download_collection_saves_etag(mock_get, mock_items, tmp_path):
    url = "https://example.com/ogd-smn_tst_d_recent.csv"
    mock_items.return_value = [_stac_item(url)]
    mock_get.return_value = _csv_response(etag='"v1"')

    download_collection("smn", tmp_path / "raw")

    etags = load_etags(tmp_path)
    assert etags.get(url) == '"v1"'


@patch("foehn.download.get_collection_items")
@patch("foehn.download.requests.get")
def test_download_collection_sends_if_none_match(mock_get, mock_items, tmp_path):
    url = "https://example.com/ogd-smn_tst_d_recent.csv"
    mock_items.return_value = [_stac_item(url)]

    # Pre-seed an ETag and create the file so the cache path is taken
    save_etags(tmp_path, {url: '"old"'})
    out = tmp_path / "raw" / "smn" / "ogd-smn_tst_d_recent.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("existing")

    mock_get.return_value = _csv_response(status_code=304)

    download_collection("smn", tmp_path / "raw")

    _, kwargs = mock_get.call_args
    assert kwargs["headers"].get("If-None-Match") == '"old"'


@patch("foehn.download.get_collection_items")
@patch("foehn.download.requests.get")
def test_download_collection_skips_304(mock_get, mock_items, tmp_path):
    url = "https://example.com/ogd-smn_tst_d_recent.csv"
    mock_items.return_value = [_stac_item(url)]
    mock_get.return_value = _csv_response(status_code=304)

    # File does not get created when server returns 304
    download_collection("smn", tmp_path / "raw")

    assert not (tmp_path / "raw" / "smn" / "ogd-smn_tst_d_recent.csv").exists()


@patch("foehn.download.get_collection_items")
@patch("foehn.download.requests.get")
def test_download_collection_since_filter(mock_get, mock_items, tmp_path):
    """Items older than `since` should be skipped without any HTTP call."""
    url = "https://example.com/ogd-smn_tst_d_recent.csv"
    mock_items.return_value = [_stac_item(url, updated="2025-06-01T00:00:00Z")]
    mock_get.return_value = _csv_response()

    download_collection("smn", tmp_path / "raw", since="2026-01-01T00:00:00Z")

    mock_get.assert_not_called()


# --- download_metadata ---


@patch("foehn.download.get_collection_metadata")
@patch("foehn.download.requests.get")
def test_download_metadata_saves_csv(mock_get, mock_meta, tmp_path):
    mock_meta.return_value = {"assets": {"stations": {"href": "https://example.com/stations.csv"}}}
    mock_get.return_value = _csv_response(b"id;name\nTST;Test Station\n")

    download_metadata("smn", tmp_path / "raw")

    assert (tmp_path / "raw" / "smn" / "stations.csv").exists()


@patch("foehn.download.get_collection_metadata")
@patch("foehn.download.requests.get")
def test_download_metadata_skips_non_csv_assets(mock_get, mock_meta, tmp_path):
    mock_meta.return_value = {"assets": {"readme": {"href": "https://example.com/README.pdf"}}}

    download_metadata("smn", tmp_path / "raw")

    mock_get.assert_not_called()


# --- download_climate_normals_zip ---


def _make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


@patch("foehn.download.requests.get")
def test_download_climate_normals_zip_extracts_files(mock_get, tmp_path):
    zip_bytes = _make_zip({"sample.txt": b"data"})
    mock_get.return_value = _csv_response(content=zip_bytes)

    download_climate_normals_zip(tmp_path / "raw")

    assert (tmp_path / "raw" / "climate_normals" / "normwerte.zip").exists()
    assert (tmp_path / "raw" / "climate_normals" / "sample.txt").exists()


@patch("foehn.download.requests.get")
def test_download_climate_normals_zip_skips_if_exists(mock_get, tmp_path):
    out_dir = tmp_path / "raw" / "climate_normals"
    out_dir.mkdir(parents=True)
    (out_dir / "normwerte.zip").write_bytes(b"existing")

    download_climate_normals_zip(tmp_path / "raw")

    mock_get.assert_not_called()


@patch("foehn.download.requests.get")
def test_download_climate_normals_zip_force_redownloads(mock_get, tmp_path):
    out_dir = tmp_path / "raw" / "climate_normals"
    out_dir.mkdir(parents=True)
    (out_dir / "normwerte.zip").write_bytes(b"old")

    zip_bytes = _make_zip({"new.txt": b"fresh"})
    mock_get.return_value = _csv_response(content=zip_bytes)

    download_climate_normals_zip(tmp_path / "raw", force=True)

    mock_get.assert_called_once()
    assert (out_dir / "new.txt").exists()


# --- download_grib2 ---


@patch("foehn.download.requests.get")
def test_download_grib2_saves_binary(mock_get, tmp_path):
    items_resp = MagicMock()
    items_resp.raise_for_status = MagicMock()
    items_resp.json.return_value = {
        "features": [{"id": "f1", "assets": {"data": {"href": "https://example.com/forecast.grib2"}}, "properties": {}}]
    }
    file_resp = _stream_response(chunks=(b"GRIB", b"data"))

    mock_get.side_effect = [items_resp, file_resp]

    download_grib2("forecast_icon_ch1", tmp_path / "raw")

    assert (tmp_path / "raw" / "forecast_icon_ch1" / "forecast.grib2").exists()


@patch("foehn.download.requests.get")
def test_download_grib2_skips_existing_file(mock_get, tmp_path):
    items_resp = MagicMock()
    items_resp.raise_for_status = MagicMock()
    items_resp.json.return_value = {
        "features": [{"id": "f1", "assets": {"data": {"href": "https://example.com/forecast.grib2"}}, "properties": {}}]
    }
    mock_get.return_value = items_resp

    out_dir = tmp_path / "raw" / "forecast_icon_ch1"
    out_dir.mkdir(parents=True)
    (out_dir / "forecast.grib2").write_bytes(b"existing")

    download_grib2("forecast_icon_ch1", tmp_path / "raw")

    # Only 1 call for the STAC items page, none for the file
    assert mock_get.call_count == 1


# --- download_netcdf ---


@patch("foehn.download.get_collection_items")
@patch("foehn.download.requests.get")
def test_download_netcdf_saves_nc_file(mock_get, mock_items, tmp_path):
    mock_items.return_value = [
        {"id": "g1", "assets": {"data": {"href": "https://example.com/grid.nc"}}, "properties": {}}
    ]
    mock_get.return_value = _stream_response(chunks=(b"\x89HDF",))

    download_netcdf("surface_derived_grid", tmp_path / "raw")

    assert (tmp_path / "raw" / "surface_derived_grid" / "grid.nc").exists()


@patch("foehn.download.get_collection_items")
@patch("foehn.download.requests.get")
def test_download_netcdf_skips_existing_file(mock_get, mock_items, tmp_path):
    mock_items.return_value = [
        {"id": "g1", "assets": {"data": {"href": "https://example.com/grid.nc"}}, "properties": {}}
    ]

    out_dir = tmp_path / "raw" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / "grid.nc").write_bytes(b"existing")

    download_netcdf("surface_derived_grid", tmp_path / "raw")

    mock_get.assert_not_called()
