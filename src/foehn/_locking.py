"""Small cross-platform file-locking primitive.

Platform lock modules are imported only when a lock is acquired.  Keeping those
imports off the package import path lets Windows import :mod:`foehn` even though
``fcntl`` is unavailable there.
"""

from __future__ import annotations

import importlib
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


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


__all__ = ["exclusive_directory_lock", "exclusive_lock"]
