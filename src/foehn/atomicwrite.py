"""Never leave a torn file at its final path.

Every replacement write foehn makes to the **Workspace** stages into a sibling
temp file and ``Path.replace``s it into place, so a write that dies part-way
leaves nothing rather than a truncated file with a fresh mtime — which the skip
rules above it read as "already done" and never retry. Incremental Zarr appends
are explicitly in-place and do not use this replacement primitive.

One rule, stated once. It used to be written out four times: the download
engine's byte writer, the streaming fetcher, the Parquet converter, and the run
state, which reached into the download engine for it. Three suffixes and three
docstrings, all reasoning their way to the same thing.

This knows only about the filesystem and the portable locking primitive below
it. Staging names share one namespace so abandoned artifacts can be recognised
and reaped without touching unrelated dotfiles.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from foehn._locking import exclusive_directory_lock

_STAGE_PREFIX = ".foehn-stage-"
_STALE_AFTER_SECONDS = 24 * 60 * 60
_DEFAULT_PUBLISHED_FILE_MODE = 0o644
_DEFAULT_PUBLISHED_DIRECTORY_MODE = 0o755


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _reap_stale_stages(path: Path) -> None:
    """Remove abandoned stages old enough that no live write owns them.

    Target-specific legacy patterns migrate workspaces created before the
    shared namespace was introduced. Their age gate avoids touching a live
    writer from an older foehn process during a rolling upgrade.
    """
    cutoff = time.time() - _STALE_AFTER_SECONDS
    patterns = (
        f"{_STAGE_PREFIX}*",
        f".{path.name}.*.tmp",
        f".{path.name}.*.part",
        f".{path.name}.*.transfer",
        f".{path.name}.staging-*",
    )
    candidates = {candidate for pattern in patterns for candidate in path.parent.glob(pattern)}
    for candidate in candidates:
        try:
            if candidate.stat().st_mtime < cutoff:
                _remove_tree(candidate)
        except FileNotFoundError:
            pass


def _stage_name(path: Path, suffix: str) -> Path:
    return path.parent / f"{_STAGE_PREFIX}{uuid.uuid4().hex}-{path.name}{suffix}"


def _new_stage_file(path: Path, suffix: str) -> Path:
    while True:
        candidate = _stage_name(path, suffix)
        try:
            # Partial content is private regardless of the process umask. The
            # intended final mode is applied only at the publication boundary.
            descriptor = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            continue
        os.close(descriptor)
        return candidate


def _published_file_mode(path: Path) -> int:
    """Preserve an existing target's permissions; use the legacy default for a new file."""
    try:
        return stat.S_IMODE(path.stat().st_mode) & 0o777
    except FileNotFoundError:
        return _DEFAULT_PUBLISHED_FILE_MODE


def _new_stage_directory(path: Path) -> Path:
    while True:
        candidate = _stage_name(path, "")
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate


def _published_directory_mode(path: Path) -> int:
    """Preserve an existing directory mode; use the normal shared-readable default."""
    try:
        return stat.S_IMODE(path.stat().st_mode) & 0o777
    except FileNotFoundError:
        return _DEFAULT_PUBLISHED_DIRECTORY_MODE


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
    _reap_stale_stages(path)
    tmp = _new_stage_file(path, suffix)
    try:
        yield tmp
        tmp.chmod(_published_file_mode(path))
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@contextlib.contextmanager
def _directory_lock(path: Path) -> Iterator[None]:
    """Serialize the short directory exchange with a cross-platform lock."""
    with exclusive_directory_lock(path):
        yield


def _recover_directory(path: Path, backup_path: Path) -> None:
    if backup_path.exists() and not path.exists():
        backup_path.replace(path)
    elif backup_path.exists():
        _remove_tree(backup_path)


@contextlib.contextmanager
def staged_directory(path: Path) -> Iterator[Path]:
    """Build a complete sibling directory, then publish it at ``path``.

    An existing directory remains untouched while the replacement is written.
    The two final renames have a small missing-path window on filesystems without
    directory exchange, but readers can never observe a partial or mixed-version
    materialization. A backup left by process termination is restored on entry.
    """
    backup_path = path.with_name(path.name + ".previous")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reap_stale_stages(path)
    with _directory_lock(path.parent):
        _recover_directory(path, backup_path)
    staged_path = _new_stage_directory(path)

    try:
        yield staged_path
        staged_path.chmod(_published_directory_mode(path))
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
