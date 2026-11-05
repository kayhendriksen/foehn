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


def test_a_marker_that_is_not_utf8_blocks(tmp_path):
    """A torn write can leave bytes that do not decode; that blocked by raising instead."""
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / coherence.MARKER).write_bytes(b'{"pending": ["a_rhiresd.nc"]}\xff\xfe')

    assert coherence.read_pending(out_dir) is None
    assert coherence.blocked(out_dir, ["a_rhiresd.nc"]) == {"a_rhiresd.nc"}


def test_clearing_an_unreadable_marker_keeps_every_other_file_pending(tmp_path):
    """Left alone, it blocked every set for good and every refresh fetched its whole set again.

    Releasing everything would be a guess; releasing only what this refresh
    made whole, and spelling out the rest, still fails closed and lets each
    other set be released by its own next refresh.
    """
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    for name in ("a_rhiresd.nc", "b_tabsd.nc"):
        (out_dir / name).write_bytes(b"x")
    (out_dir / coherence.MARKER).write_text("{ truncated")

    coherence.clear(out_dir, ["a_rhiresd.nc"])

    assert coherence.read_pending(out_dir) == {"b_tabsd.nc"}


def test_an_unreadable_marker_file_blocks_like_a_corrupt_one(tmp_path):
    """A marker we cannot even open is state we do not know."""

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / coherence.MARKER).write_text('{"pending": ["a.nc"]}')

    with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
        assert coherence.read_pending(out_dir) is None


def test_marking_an_unreadable_marker_spells_out_every_file(tmp_path):
    """Merging into an unreadable marker turned "unknown" into a tidy empty set.

    An unrelated successful refresh then deleted that set, releasing the files
    the unreadable marker was protecting. Every file here stays pending instead.
    """

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / "b_tabsd.nc").write_bytes(b"x")
    (out_dir / coherence.MARKER).write_text("{ truncated")

    with coherence.refresh_lock(out_dir):
        coherence.mark(out_dir, ["a_rhiresd.nc"])

    assert coherence.read_pending(out_dir) == {"a_rhiresd.nc", "b_tabsd.nc"}


def test_the_lock_lives_beside_the_directory_it_guards(tmp_path):
    """The ZIP kinds publish by replacing the directory; Windows will not rename one with a file open in it."""
    out_dir = tmp_path / "bronze" / "climate_normals"

    with coherence.refresh_lock(out_dir):
        assert not (out_dir / coherence.LOCK).exists()
        assert (out_dir.parent / f".climate_normals{coherence.LOCK}").exists()


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
