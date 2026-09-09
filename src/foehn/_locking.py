"""Small cross-platform file-locking primitive.

Platform lock modules are imported only when a lock is acquired.  Keeping those
imports off the package import path lets Windows import :mod:`foehn` even though
``fcntl`` is unavailable there.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_reentrant = threading.local()


@contextmanager
def reentrant_lock(path: Path) -> Iterator[None]:
    """:func:`exclusive_lock`, but a thread may take it again while holding it.

    A file lock is held by the *file descriptor*, so a thread that opens the
    same lock file twice deadlocks against itself. That matters wherever one
    scope has to cover two operations that each want the lock — acquiring a
    grid dataset's files and then opening them, say. Nesting is counted here so
    only the outermost scope touches the file, and other threads and processes
    still block as they should.
    """
    key = str(path)
    depths = getattr(_reentrant, "depths", None)
    if depths is None:
        depths = _reentrant.depths = {}
    if depths.get(key, 0) > 0:
        depths[key] += 1
        try:
            yield
        finally:
            depths[key] -= 1
        return
    with exclusive_lock(path):
        depths[key] = 1
        try:
            yield
        finally:
            depths[key] = 0


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock for the duration of the context."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            msvcrt = importlib.import_module("msvcrt")
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def take_hold(path: Path):
    """Lock *path* and keep it locked until the handle returned is closed.

    A lock whose lifetime is a scope cannot say "this is still in use" to a
    process that comes along later. Returns the open handle: hold it for as long
    as the thing it guards is live, and close it to release.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            msvcrt = importlib.import_module("msvcrt")
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise
    return handle


def is_held(path: Path) -> bool:
    """Whether anyone still holds *path* — including this process.

    Asked before deleting something on age alone. A snapshot older than its
    lifetime is not necessarily abandoned: a process can outlive it and still be
    reading through it, and deleting that out from under it is worse than
    leaving a stale directory around.
    """
    if not path.exists():
        return False
    try:
        handle = take_hold(path)
    except OSError:
        return True
    handle.close()
    return False


@contextmanager
def exclusive_directory_lock(path: Path) -> Iterator[None]:
    """Lock a directory directly on POSIX and through one stable file on Windows."""
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # Windows cannot open a directory as a lock handle. One stable file per
        # publication directory avoids the split-lock race caused by deleting
        # and recreating a lock file while another process still holds it.
        with exclusive_lock(path / ".foehn-publish.lock"):
            yield
        return

    descriptor = os.open(path, os.O_RDONLY)
    fcntl = importlib.import_module("fcntl")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


__all__ = ["exclusive_directory_lock", "exclusive_lock", "is_held", "reentrant_lock", "take_hold"]
