"""Ports for foehn ingestion.

Two seams cross into Spark/Delta. Production adapters live in
``foehn.ingest.adapters.spark_delta``; in-memory recording adapters
for tests live in ``foehn.ingest.adapters.memory``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol

import polars as pl

WriteMode = Literal["overwrite", "append"]


class DeltaSink(Protocol):
    """Port for tabular Delta writes.

    Invariants:
      - ``write`` is total: if it returns, the rows are durable at ``table``.
      - ``mode="overwrite"`` replaces the table; ``mode="append"`` requires
        a schema-compatible existing table.
      - ``apply_comments`` is best-effort: missing columns are skipped and
        per-column failures must not raise.
      - ``table`` is a fully-qualified, already-quoted identifier.
    """

    def write(self, frame: pl.DataFrame, table: str, mode: WriteMode = "overwrite") -> None: ...

    def apply_comments(self, table: str, comments: Mapping[str, str]) -> None: ...


class BinaryFileIndex(Protocol):
    """Port for radar HDF5 indexing — index-only, payload stays in place.

    Invariants:
      - Idempotent: repeated calls with the same directory contents are
        a no-op against existing rows.
      - Upserts by ``path``; updates rows whose ``modification_time`` has
        advanced (catches reanalysis overwrites).
      - Returns the count of files visible in ``directory`` after the merge.
    """

    def merge_index(self, directory: Path, table: str) -> int: ...
