"""Whether the grid files on disk are one generation, or several.

MeteoSwiss restates a grid file under its original name, so a refresh replaces
files rather than adding them. Replacing several one at a time has a middle:
some new, some old, every one of them present. A Dataset assembled from that
middle is wrong in a way nothing downstream can detect, so the middle is
recorded while it exists and the recording outlives the process that made it.

Two things live here because more than one module needs them and those modules
sit on opposite sides of a seam:

* the lock a writer holds for its whole refresh, and a reader holds while it
  takes its snapshot;
* the marker naming the files whose generation is unknown.

``gridfiles`` refreshes for a read, ``downloads`` refreshes for ``foehn.download()``,
and ``grids`` reads. All three write to or read from the same directory, so all
three speak this protocol. When only the read path did, an ordinary
``foehn.download()`` running beside it produced exactly the mixed set this is
for, with nothing on disk to say so.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from foehn._locking import reentrant_lock
from foehn.atomicwrite import write_text

logger = logging.getLogger(__name__)

MARKER = ".foehn-incoherent.json"
LOCK = ".foehn-refresh.lock"


@contextmanager
def refresh_lock(out_dir: Path) -> Iterator[None]:
    """Hold one grid dataset's directory for a whole refresh or snapshot.

    Not merely the marker's read-modify-write. Two refreshes of the same set
    each checked coherence, each downloaded, and each published into the same
    directory, interleaving into a set neither produced. A reader could likewise
    pass the coherence check and then take its files after a refresh had started
    replacing them.

    Reentrant: a reader holds this across acquiring files and snapshotting them,
    and the acquisition takes it again inside that scope. A file lock belongs to
    the descriptor, so nesting it any other way deadlocks a thread against
    itself.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with reentrant_lock(out_dir / LOCK):
        yield


def read_pending(out_dir: Path) -> set[str] | None:
    """Which files a previous run left at an unknown generation.

    None means the marker exists but cannot be understood. That is not the same
    as "nothing pending": a marker truncated by the crash it was recording would
    otherwise fail open and release the very files it was written to protect.
    """
    try:
        raw = (out_dir / MARKER).read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    except OSError:
        return None
    try:
        recorded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(recorded, dict):
        return None
    pending = recorded.get("pending")
    if not isinstance(pending, list):
        return None
    return {str(name) for name in pending}


def _write(out_dir: Path, names: set[str]) -> None:
    marker = out_dir / MARKER
    if not names:
        marker.unlink(missing_ok=True)
        return
    write_text(marker, json.dumps({"pending": sorted(names)}, indent=2))


def mark(out_dir: Path, names: list[str]) -> None:
    """Record that these files may be at different generations from each other.

    Written *before* the fetch, not after a failure. A process killed outright
    between publishing one file and the next never reaches an exception handler,
    and the set it leaves behind is exactly the one this is for.

    Callers hold :func:`refresh_lock`.
    """
    known = read_pending(out_dir)
    if known is None:
        # Unreadable, and an unreadable marker already blocks everything it
        # could name. Merging into it would turn "state unknown" into a tidy
        # empty set that the next successful refresh of any set then deletes.
        return
    _write(out_dir, known | set(names))


def clear(out_dir: Path, names: list[str]) -> None:
    """Release only the files a refresh actually brought to one generation.

    Callers hold :func:`refresh_lock`, and reach here only after a refresh that
    completed. A failed one never clears: whether it left the set mixed is
    exactly what it cannot answer.
    """
    pending = read_pending(out_dir)
    if pending is None:
        return  # unreadable: leave it rather than replace it with a guess
    if pending & set(names):
        _write(out_dir, pending - set(names))


def blocked(out_dir: Path, names: list[str]) -> set[str]:
    """The subset of *names* recorded as being at an unknown generation."""
    pending = read_pending(out_dir)
    if pending is None:
        return set(names)  # an unreadable marker blocks everything it could name
    return pending & set(names)


__all__ = ["LOCK", "MARKER", "blocked", "clear", "mark", "read_pending", "refresh_lock"]
