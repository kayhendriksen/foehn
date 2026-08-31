"""Never leave a torn file at its final path.

Every write foehn makes to the **Workspace** stages into a sibling temp file and
``Path.replace``s it into place, so a write that dies part-way leaves nothing
rather than a truncated file with a fresh mtime — which the skip rules above it
read as "already done" and never retry.

One rule, stated once. It used to be written out four times: the download
engine's byte writer, the streaming fetcher, the Parquet converter, and the run
state, which reached into the download engine for it. Three suffixes and three
docstrings, all reasoning their way to the same thing.

A leaf: this knows about the filesystem and nothing about foehn.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def staged(path: Path, *, suffix: str = ".tmp") -> Iterator[Path]:
    """Yield a sibling path to write to, then move it onto *path*.

    The move happens when the block completes; the temp file is removed when it
    raises. ``BaseException``, not ``Exception``: a KeyboardInterrupt mid-write
    leaves a partial file exactly as a disk-full error does.

    The suffix is visible in the directory while the write is in flight, so it
    is worth naming for what it is — the fetcher stages ``.part``. The middle
    token is unique, so concurrent processes never share a partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


@contextlib.contextmanager
def _directory_lock(path: Path) -> Iterator[None]:
    """Serialize only the final sibling-directory exchange, without a lock file."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _recover_directory(path: Path, backup_path: Path) -> None:
    if backup_path.exists() and not path.exists():
        backup_path.replace(path)
    elif backup_path.exists():
        _remove_tree(backup_path)


@contextlib.contextmanager
def staged_directory(path: Path, *, copy_existing: bool = False) -> Iterator[Path]:
    """Build a complete sibling directory, then publish it at ``path``.

    An existing directory remains untouched while the replacement is written.
    The two final renames have a small missing-path window on filesystems without
    directory exchange, but readers can never observe a partial or mixed-version
    materialization. A backup left by process termination is restored on entry.
    """
    backup_path = path.with_name(path.name + ".previous")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _directory_lock(path.parent):
        _recover_directory(path, backup_path)
    staged_path = Path(tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent))
    if copy_existing and path.exists():
        shutil.copytree(path, staged_path, dirs_exist_ok=True)

    try:
        yield staged_path
        with _directory_lock(path.parent):
            _recover_directory(path, backup_path)
            if path.exists():
                path.replace(backup_path)
            try:
                staged_path.replace(path)
            except BaseException:
                if backup_path.exists() and not path.exists():
                    backup_path.replace(path)
                raise
            _remove_tree(backup_path)
    except BaseException:
        _remove_tree(staged_path)
        raise


def write_bytes(path: Path, data: bytes | memoryview) -> None:
    """Write bytes to *path* so readers never see a torn write."""
    with staged(path) as tmp:
        tmp.write_bytes(data)


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to *path* so readers never see a torn write."""
    write_bytes(path, text.encode("utf-8"))


__all__ = ["staged", "staged_directory", "write_bytes", "write_text"]
