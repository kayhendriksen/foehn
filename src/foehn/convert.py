"""Convert downloaded CSVs and TXTs to Parquet using Polars."""

from __future__ import annotations

import io
import re
from pathlib import Path

import polars as pl

_COL_RE = re.compile(r"at column '([^']+)'")

_DTYPE_MAP: dict[str, pl.DataType] = {
    "float": pl.Float64,
    "integer": pl.Int64,
}


def _parse_metadata_types(content: bytes | str) -> dict[str, pl.DataType]:
    """Build a parameter→Polars dtype mapping from metadata CSV content.

    Works with both raw bytes (in-memory) and string content.
    Returns an empty dict if the expected columns are missing.
    """
    try:
        if isinstance(content, str):
            content = content.encode("utf-8")
        meta = pl.read_csv(io.BytesIO(content), separator=";", infer_schema_length=0)
    except Exception:
        return {}

    if "parameter_shortname" not in meta.columns or "parameter_datatype" not in meta.columns:
        return {}

    type_map: dict[str, pl.DataType] = {}
    for row in meta.select("parameter_shortname", "parameter_datatype").iter_rows():
        shortname, datatype = row
        if shortname and datatype:
            dtype = _DTYPE_MAP.get(datatype.strip().lower())
            if dtype is not None:
                type_map[shortname] = dtype
    return type_map


def _load_metadata_types(csv_dir: Path) -> dict[str, pl.DataType]:
    """Build a parameter→Polars dtype mapping from a *_meta_parameters.csv file.

    Returns an empty dict if no metadata file is found or if the expected
    columns (``parameter_shortname``, ``parameter_datatype``) are missing.
    """
    meta_files = list(csv_dir.glob("*_meta_parameters.csv"))
    if not meta_files:
        return {}

    meta_path = meta_files[0]
    try:
        return _parse_metadata_types(meta_path.read_bytes())
    except Exception:
        return {}


def parse_csv_bytes(
    content: bytes,
    metadata_types: dict[str, pl.DataType] | None = None,
    _fallback_overrides: dict[str, pl.DataType] | None = None,
) -> pl.DataFrame:
    """Parse CSV bytes into a Polars DataFrame, applying metadata type overrides.

    Args:
        content: Raw CSV bytes (UTF-8 encoded).
        metadata_types: Optional parameter→dtype mapping from metadata.
        _fallback_overrides: If provided, any Float64 fallback overrides applied
            during error recovery will be written into this dict (for diagnostics).

    Returns:
        Parsed Polars DataFrame.
    """
    buf = io.BytesIO(content)

    # Build per-file overrides by matching CSV columns to metadata types.
    overrides: dict[str, pl.DataType] = {}
    if metadata_types:
        try:
            header = pl.read_csv(io.BytesIO(content), separator=";", n_rows=0, infer_schema_length=0).columns
            for col in header:
                if col in metadata_types:
                    overrides[col] = metadata_types[col]
        except Exception:  # nosec B110
            pass

    try:
        return pl.read_csv(
            buf,
            separator=";",
            infer_schema_length=100,
            try_parse_dates=True,
            schema_overrides=overrides or None,
        )
    except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
        # Fallback: accumulate Float64 overrides for problematic columns.
        last_err = e
        while True:
            m = _COL_RE.search(str(last_err))
            if not m:
                break
            overrides[m.group(1)] = pl.Float64
            if _fallback_overrides is not None:
                _fallback_overrides[m.group(1)] = pl.Float64
            try:
                return pl.read_csv(
                    io.BytesIO(content),
                    separator=";",
                    infer_schema_length=100,
                    try_parse_dates=True,
                    schema_overrides=overrides,
                )
            except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e2:
                last_err = e2
            except Exception:
                raise
        raise last_err from None


def convert_to_parquet(collection_key: str, raw_dir: Path, parquet_dir: Path):
    """Convert all CSVs in a collection's raw folder to combined Parquet files.

    Per-station CSVs are grouped by frequency and time slice, then
    concatenated into a single Parquet file per group.  For example,
    all ``ogd-smn_*_d_recent.csv`` files become ``smn_d_recent.parquet``.

    Args:
        collection_key: Key from COLLECTIONS (e.g. "smn").
        raw_dir: Root raw download directory (CSVs in raw_dir/<key>/).
        parquet_dir: Root parquet directory (output to parquet_dir/<key>/).
    """
    from foehn.collections import COLLECTIONS, NO_GRANULARITY_COLLECTIONS

    csv_dir = raw_dir / collection_key
    out_dir = parquet_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if "_meta_" not in f.name]
    if not csv_files:
        return

    # Load parameter type info from metadata once for the whole collection.
    metadata_types = _load_metadata_types(csv_dir)

    # Derive the filename prefix from the collection ID (e.g. "ogd-smn").
    prefix = COLLECTIONS[collection_key].rsplit(".", 1)[-1]
    no_granularity = collection_key in NO_GRANULARITY_COLLECTIONS

    # Group CSVs by (frequency, time_slice).
    groups: dict[tuple[str, ...], list[Path]] = {}
    for csv_path in csv_files:
        suffix_part = csv_path.stem[len(prefix) + 1 :]  # e.g. "ber_d_recent"
        parts = suffix_part.split("_")
        if no_granularity:
            group_key: tuple[str, ...] = ()
        elif len(parts) > 2:
            group_key = (parts[1], parts[2])  # (frequency, time_slice)
        else:
            group_key = (parts[1],) if len(parts) > 1 else ()  # (frequency,)
        groups.setdefault(group_key, []).append(csv_path)

    print(f"Converting {collection_key} to Parquet:", flush=True)
    converted = 0
    skipped = 0
    for group_key, files in sorted(groups.items()):
        # Output name: smn_d_recent.parquet, smn_d.parquet, or smn.parquet
        if group_key:
            out_name = f"{collection_key}_{'_'.join(group_key)}.parquet"
        else:
            out_name = f"{collection_key}.parquet"
        parquet_path = out_dir / out_name

        # Skip if parquet is already newer than all CSVs in the group.
        if parquet_path.exists():
            parquet_mtime = parquet_path.stat().st_mtime
            if all(f.stat().st_mtime <= parquet_mtime for f in files):
                skipped += 1
                continue

        print(f"  {out_name} ({len(files)} files)...", end="", flush=True)

        try:
            frames: list[pl.DataFrame] = []
            overrides: dict[str, pl.DataType] = {}
            for csv_path in files:
                df = parse_csv_bytes(csv_path.read_bytes(), metadata_types, _fallback_overrides=overrides)
                frames.append(df)
            combined = pl.concat(frames, how="diagonal_relaxed")
            combined.write_parquet(parquet_path, compression="zstd")
            converted += 1
            if overrides:
                fixed = ", ".join(f"{c}→float" for c in overrides)
                print(f" OK ({fixed})", flush=True)
            else:
                print(" OK", flush=True)
        except Exception as e:
            print(f" FAIL: {e}", flush=True)

    print(
        f"  Done: {converted} converted, {skipped} skipped (up-to-date)",
        flush=True,
    )


def convert_climate_normals_to_parquet(raw_dir: Path, parquet_dir: Path):
    """Convert C6 climate normals TXT files to Parquet.

    These files use tab separators, latin1 encoding, and have 7 header rows
    to skip before the actual data begins.
    """
    txt_dir = raw_dir / "climate_normals"
    out_dir = parquet_dir / "climate_normals"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        return

    print("Converting climate_normals to Parquet:", flush=True)
    converted = 0
    skipped = 0
    total = len(txt_files)
    for i, txt_path in enumerate(txt_files, 1):
        parquet_path = out_dir / txt_path.with_suffix(".parquet").name

        if parquet_path.exists() and parquet_path.stat().st_mtime >= txt_path.stat().st_mtime:
            skipped += 1
            continue

        print(f"  [{i}/{total}] {txt_path.name}...", end="", flush=True)
        try:
            df = pl.read_csv(
                txt_path,
                separator="\t",
                skip_rows=8,
                encoding="latin1",
                infer_schema_length=None,
                try_parse_dates=True,
                truncate_ragged_lines=True,
            )
            df.write_parquet(parquet_path, compression="snappy")
            converted += 1
            print(" Converted", flush=True)
        except Exception as e:
            print(f" FAIL: {e}", flush=True)

    print(
        f"  Done: {converted} converted, {skipped} skipped (up-to-date)",
        flush=True,
    )
