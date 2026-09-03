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
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from foehn._locking import exclusive_directory_lock

_STAGE_PREFIX = ".foehn-stage-"
# The backup a directory publication parks its previous generation under. It
# lives in foehn's own namespace because the publication deletes and moves it:
# a plain ``<target>.previous`` sibling is a name a user can hold, and holding
# it meant losing it.
_BACKUP_PREFIX = ".foehn-previous-"
_STALE_AFTER_SECONDS = 24 * 60 * 60
# Publication applies the process umask itself, because it chmods explicitly:
# a published file that ignored the umask was more permissive than the caller
# asked for. What the umask allows is measured, not written down — see
# _umask_applied.
# For content that should not be shared whatever the umask allows: the run state
# and the ETag store, whose entries are full asset URLs and can carry query
# tokens. Passed explicitly, and applied to an existing target too — a store
# already sitting at 0644 from an older foehn stays readable otherwise.
PRIVATE_FILE_MODE = 0o600

# Reaping scans the whole parent directory, so doing it per write made a bulk
# download quadratic in the number of files landing in one directory. Abandoned
# stages are by definition older than a day and left by an earlier process, so
# once per directory per process finds everything a per-write scan would.
_reaped: set[Path] = set()
_reaped_lock = threading.Lock()


def _umask_applied(*, directory: bool, near: Path) -> int:
    """What the OS would grant a new file or directory created here.

    Derived by creating a throwaway beside the target and reading the mode the
    kernel actually gave it. The obvious way to get this is ``os.umask(0)``
    followed by putting it back — but that is a *process-wide* setting, and
    foehn's downloads run on a thread pool: anything else creating a file inside
    those two syscalls gets no umask at all. A concurrent probe caught an
    unrelated file at 0666 under an intended umask of 0077.

    ``touch()`` and ``mkdir()`` rather than ``os.open``/``os.mkdir`` with an
    explicit 0o666/0o777 mask. The result is identical — those are the defaults
    the kernel then masks — but the permissive mask is no longer written here,
    which is both what a reader should see and what the "overly permissive file
    permissions" analysis is looking for.

    Falls back to the conservative private mode if the probe cannot be created,
    which is the right way to be wrong about permissions.
    """
    probe = near / f"{_STAGE_PREFIX}umask-{uuid.uuid4().hex}"
    try:
        if directory:
            probe.mkdir()
        else:
            probe.touch(exist_ok=False)
        granted = stat.S_IMODE(probe.stat().st_mode)
    except OSError:
        return 0o700 if directory else 0o600
    finally:
        with contextlib.suppress(OSError):
            probe.rmdir() if directory else probe.unlink()
    return granted


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _reap_stale_stages(path: Path) -> None:
    """Remove abandoned stages old enough that no live write owns them.

    Runs at most once per directory per process: the scan is over the whole
    parent, so repeating it for every file written into that parent made a bulk
    download quadratic. What it looks for — stages left by a process that died
    at least a day ago — cannot appear while this process runs, so the first
    scan of a directory finds everything a per-write scan would.

    Target-specific legacy patterns migrate workspaces created before the
    shared namespace was introduced. Their age gate avoids touching a live
    writer from an older foehn process during a rolling upgrade.
    """
    parent = path.parent
    with _reaped_lock:
        if parent in _reaped:
            return
        _reaped.add(parent)

    cutoff = time.time() - _STALE_AFTER_SECONDS
    patterns = (
        f"{_STAGE_PREFIX}*",
        f".{path.name}.*.tmp",
        f".{path.name}.*.part",
        f".{path.name}.*.transfer",
        f".{path.name}.staging-*",
    )
    candidates = {candidate for pattern in patterns for candidate in parent.glob(pattern)}
    for candidate in candidates:
        try:
            if candidate.stat().st_mtime < cutoff:
                _remove_tree(candidate)
        except FileNotFoundError:
            # The stage was claimed or cleaned between the glob and the stat.
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
    """Preserve an existing target's permissions; apply the umask to a new file."""
    try:
        return stat.S_IMODE(path.stat().st_mode) & 0o777
    except FileNotFoundError:
        return _umask_applied(directory=False, near=path.parent)


def _new_stage_directory(path: Path) -> Path:
    while True:
        candidate = _stage_name(path, "")
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate


def _published_directory_mode(path: Path) -> int:
    """Preserve an existing directory mode; apply the umask to a new one."""
    try:
        return stat.S_IMODE(path.stat().st_mode) & 0o777
    except FileNotFoundError:
        return _umask_applied(directory=True, near=path.parent)


@contextlib.contextmanager
def staged(path: Path, *, suffix: str = ".tmp", mode: int | None = None) -> Iterator[Path]:
    """Yield a sibling path to write to, then move it onto *path*.

    The move happens when the block completes; the temp file is removed when it
    raises. ``BaseException``, not ``Exception``: a KeyboardInterrupt mid-write
    leaves a partial file exactly as a disk-full error does.

    The suffix is visible in the directory while the write is in flight, so it
    is worth naming for what it is — the fetcher stages ``.part``. The middle
    token is unique, so concurrent processes never share a partial file.

    ``mode`` forces the published permissions instead of preserving an existing
    target's or deriving them from the umask. Use it for content whose
    sensitivity is a property of the content, not of the caller's environment.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _reap_stale_stages(path)
    tmp = _new_stage_file(path, suffix)
    try:
        yield tmp
        tmp.chmod(_published_file_mode(path) if mode is None else mode)
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

    The previous generation is parked under ``_BACKUP_PREFIX``, inside foehn's
    own namespace. It used to be a plain ``<target>.previous`` sibling, which
    publication is free to move and delete — so a user directory that happened
    to carry that name was destroyed by publishing next to it.
    """
    backup_path = path.with_name(_BACKUP_PREFIX + path.name)
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


def write_bytes(path: Path, data: bytes | memoryview, *, mode: int | None = None) -> None:
    """Write bytes to *path* so readers never see a torn write."""
    with staged(path, mode=mode) as tmp:
        tmp.write_bytes(data)


def write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    """Write UTF-8 text to *path* so readers never see a torn write."""
    write_bytes(path, text.encode("utf-8"), mode=mode)


__all__ = ["PRIVATE_FILE_MODE", "staged", "staged_directory", "write_bytes", "write_text"]
