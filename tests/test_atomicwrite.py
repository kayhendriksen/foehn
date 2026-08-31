"""Tests for the staging rule every write to the Workspace goes through.

A leaf module, so these need no fetcher, no workspace and no dataset. What they
assert is the property the four callers were each reasoning their way to: after
a failed write there is nothing at the target and nothing beside it.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from foehn.atomicwrite import staged, write_bytes, write_text


def test_bytes_land_at_the_target(tmp_path):
    target = tmp_path / "asset.csv"
    write_bytes(target, b"payload")

    assert target.read_bytes() == b"payload"
    assert list(tmp_path.iterdir()) == [target]  # nothing staged left over


def test_text_is_written_as_utf8(tmp_path):
    target = tmp_path / "state.json"
    write_text(target, '{"station": "Zürich"}')

    assert target.read_bytes() == '{"station": "Zürich"}'.encode()


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

    assert seen == ["grid.grib2.part"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["grid.grib2"]
