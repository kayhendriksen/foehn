"""Tests for the ZIP guards: the decompression-bomb cap and the zip-slip check."""

import pytest
from conftest import make_zip

from foehn.archives import safe_extract
from foehn.downloads import download_normals_zip
from foehn.workspace import Workspace
from tests.fakes import InMemoryFetcher


def _fake(*, body: bytes) -> InMemoryFetcher:
    fake = InMemoryFetcher()
    fake.any_collection = {"assets": {}}
    fake.default_body = body
    return fake


def test_normals_zip_rejects_decompression_bomb(tmp_path, monkeypatch):
    """An archive declaring more decompressed bytes than the cap must not extract."""
    monkeypatch.setattr("foehn.archives._MAX_ZIP_EXTRACT_BYTES", 10)
    zip_bytes = make_zip({"sample.txt": b"x" * 1024})
    fake = _fake(body=zip_bytes)

    with pytest.raises(ValueError, match="decompressed"):
        download_normals_zip("climate_normals", Workspace(tmp_path), fetcher=fake)

    assert not (tmp_path / "bronze" / "climate_normals" / "sample.txt").exists()


def test_safe_extract_zip_accepts_nested_members(tmp_path):
    """Legitimate nested members must extract.

    The guard used to compare strings against ``str(out_dir) + "/"``, which no
    resolved path matches on Windows (separator is "\\") — so every member of
    every archive was rejected there, including the C6 climate normals that
    bare ``foehn download`` always fetches.
    """
    zip_path = tmp_path / "ok.zip"
    zip_path.write_bytes(make_zip({"nested/dir/sample.txt": b"data"}))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assert safe_extract(zip_path, out_dir) == 1
    assert (out_dir / "nested" / "dir" / "sample.txt").read_bytes() == b"data"


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    zip_path = tmp_path / "evil.zip"
    zip_path.write_bytes(make_zip({"../evil.txt": b"x"}))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract(zip_path, out_dir)

    assert not (tmp_path / "evil.txt").exists()
