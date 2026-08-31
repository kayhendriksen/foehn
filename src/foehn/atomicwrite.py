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
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def staged(path: Path, *, suffix: str = ".tmp") -> Iterator[Path]:
    """Yield a sibling path to write to, then move it onto *path*.

    The move happens when the block completes; the temp file is removed when it
    raises. ``BaseException``, not ``Exception``: a KeyboardInterrupt mid-write
    leaves a partial file exactly as a disk-full error does.

    The suffix is visible in the directory while the write is in flight, so it
    is worth naming for what it is — the fetcher stages ``.part``.
    """
    tmp = path.with_name(path.name + suffix)
    try:
        yield tmp
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_bytes(path: Path, data: bytes | memoryview) -> None:
    """Write bytes to *path* so readers never see a torn write."""
    with staged(path) as tmp:
        tmp.write_bytes(data)


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to *path* so readers never see a torn write."""
    write_bytes(path, text.encode("utf-8"))


__all__ = ["staged", "write_bytes", "write_text"]
