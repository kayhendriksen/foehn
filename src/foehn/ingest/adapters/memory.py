"""In-memory recording adapters for tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from foehn.ingest.ports import WriteMode


@dataclass
class WriteCall:
    table: str
    mode: WriteMode
    frame: pl.DataFrame


@dataclass
class RecordingDeltaSink:
    """``DeltaSink`` that captures every call in memory.

    ``tables`` holds the durable state per table after applying every write
    in order (overwrite replaces; append diagonal-concats). ``calls``
    preserves the full history.
    """

    calls: list[WriteCall] = field(default_factory=list)
    tables: dict[str, pl.DataFrame] = field(default_factory=dict)
    comments: dict[str, dict[str, str]] = field(default_factory=dict)

    def write(self, frame: pl.DataFrame, table: str, mode: WriteMode = "overwrite") -> None:
        self.calls.append(WriteCall(table=table, mode=mode, frame=frame))
        if mode == "overwrite" or table not in self.tables:
            self.tables[table] = frame
        else:
            self.tables[table] = pl.concat([self.tables[table], frame], how="diagonal_relaxed")

    def apply_comments(self, table: str, comments: Mapping[str, str]) -> None:
        self.comments.setdefault(table, {}).update(comments)


@dataclass
class RecordingBinaryFileIndex:
    """``BinaryFileIndex`` that records merge calls and returns the live file count."""

    merges: list[tuple[Path, str]] = field(default_factory=list)

    def merge_index(self, directory: Path, table: str) -> int:
        self.merges.append((directory, table))
        return sum(1 for _ in directory.glob("*.h5"))
