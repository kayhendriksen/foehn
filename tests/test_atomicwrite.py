"""Tests for the staging rule every write to the Workspace goes through.

A leaf module, so these need no fetcher, no workspace and no dataset. What they
assert is the property the four callers were each reasoning their way to: after
a failed write there is nothing at the target and nothing beside it.
"""

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from foehn.atomicwrite import staged, staged_directory, write_bytes, write_text


def test_bytes_land_at_the_target(tmp_path):
    target = tmp_path / "asset.csv"
    write_bytes(target, b"payload")

    assert target.read_bytes() == b"payload"
    assert list(tmp_path.iterdir()) == [target]  # nothing staged left over


def test_text_is_written_as_utf8(tmp_path):
    target = tmp_path / "state.json"
    write_text(target, '{"station": "Zürich"}')

    assert target.read_bytes() == '{"station": "Zürich"}'.encode()


def test_a_new_published_file_honours_the_process_umask(tmp_path):
    target = tmp_path / "shared.csv"
    previous = os.umask(0o022)
    try:
        write_bytes(target, b"payload")
    finally:
        os.umask(previous)

    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_replacing_a_file_preserves_its_existing_mode(tmp_path):
    target = tmp_path / "shared.csv"
    target.write_bytes(b"old")
    target.chmod(0o640)

    write_bytes(target, b"new")

    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_a_failed_write_leaves_neither_the_target_nor_the_staged_file(tmp_path):
    """The whole point: a truncated file with a fresh mtime reads as "already done"."""
    target = tmp_path / "asset.csv"

    with patch.object(Path, "replace", side_effect=OSError("disk full")), pytest.raises(OSError):
        write_bytes(target, b"payload")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_failed_write_does_not_disturb_the_file_already_there(tmp_path):
    """The replace is the commit point, so the previous version survives a failure."""
    target = tmp_path / "asset.csv"
    target.write_bytes(b"the good copy")

    with patch.object(Path, "replace", side_effect=OSError("disk full")), pytest.raises(OSError):
        write_bytes(target, b"the truncated one")

    assert target.read_bytes() == b"the good copy"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["asset.csv"]


def test_an_interrupt_mid_write_is_cleaned_up_like_any_other_failure(tmp_path):
    """BaseException, not Exception — Ctrl-C leaves a partial file exactly as disk-full does."""
    target = tmp_path / "asset.csv"

    with pytest.raises(KeyboardInterrupt), staged(target) as tmp:
        tmp.write_bytes(b"half a fi")
        raise KeyboardInterrupt

    assert list(tmp_path.iterdir()) == []


def test_the_staged_file_sits_beside_the_target_under_the_given_suffix(tmp_path):
    """The suffix is visible in the directory while the write is in flight."""
    target = tmp_path / "grid.grib2"
    seen = []

    with staged(target, suffix=".part") as tmp:
        tmp.write_bytes(b"x")
        seen = sorted(p.name for p in tmp_path.iterdir())

    assert len(seen) == 1
    assert seen[0].startswith(".foehn-stage-")
    assert seen[0].endswith(".part")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["grid.grib2"]


def test_staged_directory_replaces_the_complete_member_set(tmp_path):
    target = tmp_path / "cube.zarr"
    target.mkdir()
    (target / "obsolete").write_text("old")

    with staged_directory(target) as staged_dir:
        (staged_dir / "fresh").write_text("new")
        assert (target / "obsolete").exists()

    assert not (target / "obsolete").exists()
    assert (target / "fresh").read_text() == "new"


def test_staged_directory_failure_preserves_the_previous_version(tmp_path):
    target = tmp_path / "cube.zarr"
    target.mkdir()
    (target / "complete").write_text("old")

    with pytest.raises(OSError, match="disk full"), staged_directory(target) as staged_dir:
        (staged_dir / "partial").write_text("new")
        raise OSError("disk full")

    assert (target / "complete").read_text() == "old"
    assert not (target / "partial").exists()


def test_overlapping_directory_builds_never_share_staging(tmp_path):
    target = tmp_path / "cube.zarr"

    with staged_directory(target) as first:
        (first / "first").write_text("complete")
        with staged_directory(target) as second:
            (second / "second").write_text("complete")
            assert first != second
        assert (target / "second").exists()

    assert (target / "first").exists()
    # Windows uses one stable directory-publication lock file; POSIX locks the
    # directory handle itself. Neither platform leaves a random staging path.
    remaining = {path.name for path in tmp_path.iterdir()} - {".foehn-publish.lock"}
    assert remaining == {"cube.zarr"}


def test_stale_namespaced_staging_artifacts_are_reaped(tmp_path):
    stale_file = tmp_path / ".foehn-stage-dead-file"
    stale_file.write_bytes(b"partial")
    stale_dir = tmp_path / ".foehn-stage-dead-directory"
    stale_dir.mkdir()
    (stale_dir / "partial").write_text("x")
    for path in (stale_file, stale_dir):
        os.utime(path, (0, 0))

    write_bytes(tmp_path / "fresh", b"complete")

    assert not stale_file.exists()
    assert not stale_dir.exists()


def test_stale_legacy_staging_artifacts_are_reaped_during_upgrade(tmp_path):
    target = tmp_path / "asset.csv"
    stale_file = tmp_path / ".asset.csv.dead.transfer"
    stale_file.write_bytes(b"partial")
    stale_dir = tmp_path / ".asset.csv.staging-dead"
    stale_dir.mkdir()
    for path in (stale_file, stale_dir):
        os.utime(path, (0, 0))

    write_bytes(target, b"complete")

    assert not stale_file.exists()
    assert not stale_dir.exists()
