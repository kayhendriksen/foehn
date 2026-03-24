"""Convert downloaded CSVs and TXTs to Parquet using Polars."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

_COL_RE = re.compile(r"at column '([^']+)'")

_DTYPE_MAP: dict[str, pl.DataType] = {
    "float": pl.Float64,
    "integer": pl.Int64,
}


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
        meta = pl.read_csv(meta_path, separator=";", infer_schema_length=0)
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


def convert_to_parquet(collection_key: str, raw_dir: Path, parquet_dir: Path):
    """Convert all CSVs in a collection's raw folder to Parquet.

    Args:
        collection_key: Key from COLLECTIONS (e.g. "smn").
        raw_dir: Root raw download directory (CSVs in raw_dir/<key>/).
        parquet_dir: Root parquet directory (output to parquet_dir/<key>/).
    """
    csv_dir = raw_dir / collection_key
    out_dir = parquet_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        return

    # Load parameter type info from metadata once for the whole collection.
    metadata_types = _load_metadata_types(csv_dir)

    print(f"Converting {collection_key} to Parquet:", flush=True)
    converted = 0
    skipped = 0
    total = len(csv_files)
    for i, csv_path in enumerate(csv_files, 1):
        # Skip metadata files – they are not data CSVs.
        if "_meta_" in csv_path.name:
            total -= 1
            continue

        parquet_path = out_dir / csv_path.with_suffix(".parquet").name

        # Skip if parquet is already newer than csv
        if parquet_path.exists() and parquet_path.stat().st_mtime >= csv_path.stat().st_mtime:
            skipped += 1
            continue

        print(f"  [{i}/{total}] {csv_path.name}...", end="", flush=True)

        # Build per-file overrides by matching CSV columns to metadata types.
        overrides: dict[str, pl.DataType] = {}
        if metadata_types:
            try:
                header = pl.read_csv(csv_path, separator=";", n_rows=0, infer_schema_length=0).columns
                for col in header:
                    if col in metadata_types:
                        overrides[col] = metadata_types[col]
            except Exception:
                pass

        try:
            df = pl.read_csv(
                csv_path,
                separator=";",
                infer_schema_length=100,
                try_parse_dates=True,
                schema_overrides=overrides or None,
            )
            df.write_parquet(parquet_path, compression="zstd")
            converted += 1
            print(" OK", flush=True)
        except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
            # Fallback: accumulate Float64 overrides and retry until no new
            # columns fail (e.g. a type not covered by metadata).
            last_err = e
            while True:
                m = _COL_RE.search(str(last_err))
                if not m:
                    break
                overrides[m.group(1)] = pl.Float64
                try:
                    df = pl.read_csv(
                        csv_path,
                        separator=";",
                        infer_schema_length=100,
                        try_parse_dates=True,
                        schema_overrides=overrides,
                    )
                    df.write_parquet(parquet_path, compression="zstd")
                    converted += 1
                    fixed = ", ".join(f"{c}→float" for c in overrides)
                    print(f" OK ({fixed})", flush=True)
                    last_err = None
                    break
                except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e2:
                    last_err = e2
                except Exception as e2:
                    last_err = e2
                    break
            if last_err is not None:
                print(f" FAIL: {last_err}", flush=True)
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
