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


def test_a_new_published_file_is_shared_readable_under_the_usual_umask(tmp_path):
    target = tmp_path / "shared.csv"
    previous = os.umask(0o022)
    try:
        write_bytes(target, b"payload")
    finally:
        os.umask(previous)

    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_an_unpublished_stage_is_private_even_with_a_permissive_umask(tmp_path):
    target = tmp_path / "shared.csv"
    previous = os.umask(0o000)
    try:
        with staged(target) as staged_path:
            staged_path.write_bytes(b"payload")
            assert stat.S_IMODE(staged_path.stat().st_mode) == 0o600
    finally:
        os.umask(previous)

    # Private in flight, then whatever the umask allows at publication —
    # 0o666 here, exactly what open() would have produced under umask 0.
    assert stat.S_IMODE(target.stat().st_mode) == 0o666


def test_a_published_file_honours_a_restrictive_umask(tmp_path):
    """A new file must not be more permissive than the caller's umask asks for.

    Publication chmods explicitly, so it has to apply the umask itself. Forcing
    0o644 here published world-readable data for a caller who had asked, through
    umask 0o077, for none of it — and the run state and ETag store, which carry
    full asset URLs, go through this same path.
    """
    target = tmp_path / "private.csv"
    reference = tmp_path / "reference.csv"
    previous = os.umask(0o077)
    try:
        write_bytes(target, b"payload")
        reference.write_bytes(b"payload")
    finally:
        os.umask(previous)

    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)


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


def test_an_unpublished_directory_is_private_even_with_a_permissive_umask(tmp_path):
    target = tmp_path / "cube.zarr"
    previous = os.umask(0o000)
    try:
        with staged_directory(target) as staged_dir:
            (staged_dir / "payload").write_text("complete")
            assert stat.S_IMODE(staged_dir.stat().st_mode) == 0o700
    finally:
        os.umask(previous)

    assert stat.S_IMODE(target.stat().st_mode) == 0o777


def test_a_published_directory_honours_a_restrictive_umask(tmp_path):
    target = tmp_path / "cube.zarr"
    reference = tmp_path / "reference"
    previous = os.umask(0o077)
    try:
        with staged_directory(target) as staged_dir:
            (staged_dir / "payload").write_text("complete")
        reference.mkdir()
    finally:
        os.umask(previous)

    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)


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


def test_publishing_does_not_touch_a_user_directory_named_previous(tmp_path):
    """Publication must not assume it owns ``<target>.previous``.

    The backup lived at that exact sibling name, and publication moves and
    deletes it freely. A user who kept their own ``cube.zarr.previous`` beside
    the store lost it the next time the store was rewritten.
    """
    target = tmp_path / "cube.zarr"
    target.mkdir()
    (target / "old").write_text("old")

    bystander = tmp_path / "cube.zarr.previous"
    bystander.mkdir()
    (bystander / "keepsake").write_text("mine")

    with staged_directory(target) as staged_dir:
        (staged_dir / "new").write_text("new")

    assert (target / "new").read_text() == "new"
    assert (bystander / "keepsake").read_text() == "mine"


def test_a_directory_is_reaped_once_per_process_not_once_per_write(tmp_path):
    """Reaping scans the whole parent, so per-write made bulk writes quadratic."""
    from foehn import atomicwrite

    atomicwrite._reaped.discard(tmp_path)
    real_glob = Path.glob
    scans = []

    def counting_glob(self, pattern, *args, **kwargs):
        scans.append(pattern)
        return real_glob(self, pattern, *args, **kwargs)

    with patch.object(Path, "glob", counting_glob):
        for index in range(25):
            write_bytes(tmp_path / f"asset_{index}.csv", b"payload")

    # One reap pass (its handful of patterns), not one per file.
    assert len(scans) < 25
    assert all((tmp_path / f"asset_{index}.csv").exists() for index in range(25))


def test_reaping_tolerates_a_stage_vanishing_mid_scan(tmp_path):
    """The race the handler exists for: gone between the glob and the stat.

    Two processes reap the same workspace, and the other one wins. Covered
    explicitly because leaving it to chance made the whole suite's coverage
    oscillate across the gate — it fired in roughly one run in five.
    """
    from foehn import atomicwrite

    stale = tmp_path / ".foehn-stage-vanishing"
    stale.write_bytes(b"partial")
    atomicwrite._reaped.discard(tmp_path)

    real_stat = Path.stat

    def stat_that_loses_the_race(self, *args, **kwargs):
        if self.name == ".foehn-stage-vanishing":
            self.unlink(missing_ok=True)
            raise FileNotFoundError(self)
        return real_stat(self, *args, **kwargs)

    with patch.object(Path, "stat", stat_that_loses_the_race):
        write_bytes(tmp_path / "asset.csv", b"payload")

    assert (tmp_path / "asset.csv").read_bytes() == b"payload"
    assert not stale.exists()


def test_reading_the_umask_never_widens_another_thread_s_files(tmp_path):
    """Deriving the mode must not change it for anyone else.

    ``os.umask(0)`` then putting it back is the obvious way to read the umask,
    and it is process-wide: anything creating a file inside those two syscalls
    gets no umask at all. foehn's downloads run on a thread pool, so that window
    is reachable — a concurrent probe caught an unrelated file at 0666 under an
    intended umask of 0077.
    """
    import threading

    leaked = []
    stop = threading.Event()

    def bystander():
        index = 0
        while not stop.is_set():
            probe = tmp_path / f"bystander_{index}"
            index += 1
            try:
                # touch() applies the umask to its own default, so the test does
                # not write a world-writable mask either.
                probe.touch(exist_ok=False)
                if stat.S_IMODE(probe.stat().st_mode) & 0o077:
                    leaked.append(probe.name)
                probe.unlink()
            except OSError:
                pass

    previous = os.umask(0o077)
    try:
        watcher = threading.Thread(target=bystander)
        watcher.start()
        for index in range(200):
            write_bytes(tmp_path / f"asset_{index}.csv", b"payload")
        stop.set()
        watcher.join()
    finally:
        os.umask(previous)

    assert leaked == []


def test_private_content_is_private_whatever_the_umask_allows(tmp_path):
    """Some content's sensitivity is a property of the content.

    The ETag store's entries are full asset URLs and can carry query tokens, so
    it is not for the umask to decide — and preserving an existing target's mode
    left a store already at 0644 from an older foehn readable forever.
    """
    from foehn.atomicwrite import PRIVATE_FILE_MODE

    target = tmp_path / "_etags.json"
    target.write_text("{}")
    target.chmod(0o644)

    previous = os.umask(0o022)
    try:
        write_text(target, '{"https://example.test/a.csv?token=x": "\\"v1\\""}', mode=PRIVATE_FILE_MODE)
    finally:
        os.umask(previous)

    assert stat.S_IMODE(target.stat().st_mode) == PRIVATE_FILE_MODE


def test_an_unprobeable_directory_publishes_privately(tmp_path):
    """Being wrong about permissions should mean too strict, not too loose."""
    from foehn.atomicwrite import _umask_applied

    with patch.object(Path, "mkdir", side_effect=OSError("read-only")):
        assert _umask_applied(directory=True, near=tmp_path) == 0o700
    with patch.object(Path, "touch", side_effect=OSError("read-only")):
        assert _umask_applied(directory=False, near=tmp_path) == 0o600


def test_write_bytes_accepts_an_explicit_mode(tmp_path):
    from foehn.atomicwrite import PRIVATE_FILE_MODE

    target = tmp_path / "state.bin"
    write_bytes(target, b"secret", mode=PRIVATE_FILE_MODE)

    assert stat.S_IMODE(target.stat().st_mode) == PRIVATE_FILE_MODE
