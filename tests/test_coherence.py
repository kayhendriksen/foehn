"""Tests for the protocol that says whether grid files are one generation.

Its own module because more than one writer speaks it: the read path's refresh,
``foehn.download()``, and the reader taking its snapshot all touch the same
directory, and the marker is what carries "some of these are half-replaced"
across a process boundary.
"""

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from foehn import coherence


@pytest.mark.parametrize(
    "content",
    ['{"pending": "not-a-list"}', "[]", '"just a string"', "{ truncated"],
)
def test_a_marker_that_cannot_be_understood_blocks(tmp_path, content):
    """Every unreadable shape has to fail closed, not just invalid JSON."""

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / coherence.MARKER).write_text(content)

    assert coherence.read_pending(out_dir) is None


def test_clearing_leaves_an_unreadable_marker_alone(tmp_path):
    """Replacing it with a guess would release whatever it was protecting."""

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    marker = out_dir / coherence.MARKER
    marker.write_text("{ truncated")

    coherence.clear(out_dir, ["a_rhiresd.nc"])

    assert marker.read_text() == "{ truncated"


def test_an_unreadable_marker_file_blocks_like_a_corrupt_one(tmp_path):
    """A marker we cannot even open is state we do not know."""

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / coherence.MARKER).write_text('{"pending": ["a.nc"]}')

    with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
        assert coherence.read_pending(out_dir) is None


def test_marking_does_not_overwrite_a_marker_it_cannot_read(tmp_path):
    """Merging into an unreadable marker turns "unknown" into a tidy empty set.

    An unrelated successful refresh then deletes that set, releasing the files
    the unreadable marker was protecting.
    """

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    marker = out_dir / coherence.MARKER
    marker.write_text("{ truncated")

    with coherence.refresh_lock(out_dir):
        coherence.mark(out_dir, ["a_rhiresd.nc"])

    assert marker.read_text() == "{ truncated"


def test_concurrent_marker_updates_do_not_lose_entries(tmp_path):
    """Two failed refreshes for different matches each record their own names."""

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    start = threading.Barrier(2)

    def record(names):
        start.wait()
        for _ in range(20):
            with coherence.refresh_lock(out_dir):
                coherence.mark(out_dir, names)

    threads = [
        threading.Thread(target=record, args=(["a_rhiresd.nc", "b_rhiresd.nc"],)),
        threading.Thread(target=record, args=(["c_tabsd.nc", "d_tabsd.nc"],)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert coherence.read_pending(out_dir) == {"a_rhiresd.nc", "b_rhiresd.nc", "c_tabsd.nc", "d_tabsd.nc"}
