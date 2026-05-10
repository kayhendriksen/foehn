"""Pure grouping and identifier helpers for Delta ingestion."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from foehn.collections import COLLECTIONS, NO_GRANULARITY_COLLECTIONS

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_identifier(value: str, label: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid {label} {value!r} — only alphanumerics, underscores, and hyphens are allowed")
    return f"`{value}`"


def _group_csv_files(csv_dir: Path, collection_key: str) -> dict[tuple[str, ...], list[Path]]:
    """Group CSV files by (frequency, time_slice), same logic as convert_to_parquet."""
    prefix = COLLECTIONS[collection_key].rsplit(".", 1)[-1]
    no_granularity = collection_key in NO_GRANULARITY_COLLECTIONS

    csv_files = sorted(csv_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if "_meta_" not in f.name]

    groups: dict[tuple[str, ...], list[Path]] = {}
    for csv_path in csv_files:
        suffix_part = csv_path.stem[len(prefix) + 1 :]
        parts = suffix_part.split("_")
        if no_granularity:
            group_key: tuple[str, ...] = ()
        elif len(parts) > 2:
            group_key = (parts[1], parts[2])
        else:
            group_key = (parts[1],) if len(parts) > 1 else ()
        groups.setdefault(group_key, []).append(csv_path)

    return groups


def _table_suffix(group_key: tuple[str, ...]) -> str:
    if group_key:
        return f"_{'_'.join(group_key)}"
    return ""


def _build_schema_overrides(files: list[Path], metadata_types: dict[str, pl.DataType]) -> dict[str, pl.DataType] | None:
    """Match CSV column headers to metadata types for schema overrides."""
    if not metadata_types:
        return None
    try:
        header = pl.read_csv(files[0], separator=";", n_rows=0, infer_schema_length=0).columns
        overrides = {col: metadata_types[col] for col in header if col in metadata_types}
        return overrides or None
    except Exception:
        return None
