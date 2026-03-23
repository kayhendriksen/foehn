"""Convert downloaded CSVs and TXTs to Parquet using Polars."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

_COL_RE = re.compile(r"at column '([^']+)'")


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

    print(f"Converting {collection_key} to Parquet:", flush=True)
    converted = 0
    skipped = 0
    total = len(csv_files)
    for i, csv_path in enumerate(csv_files, 1):
        parquet_path = out_dir / csv_path.with_suffix(".parquet").name

        # Skip if parquet is already newer than csv
        if parquet_path.exists() and parquet_path.stat().st_mtime >= csv_path.stat().st_mtime:
            skipped += 1
            continue

        print(f"  [{i}/{total}] {csv_path.name}...", end="", flush=True)
        try:
            df = pl.read_csv(
                csv_path,
                separator=";",
                infer_schema_length=100,
                try_parse_dates=True,
            )
            df.write_parquet(parquet_path, compression="zstd")
            converted += 1
            print(" OK", flush=True)
        except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
            # A column was inferred as Int but has float values beyond the
            # inference window (e.g. rre150z0 = 0,0,…,0.2 at row 1096).
            # Promote only that specific column to Float64 and retry.
            m = _COL_RE.search(str(e))
            if not m:
                print(f" FAIL: {e}", flush=True)
                continue
            col_name = m.group(1)
            try:
                df = pl.read_csv(
                    csv_path,
                    separator=";",
                    infer_schema_length=100,
                    try_parse_dates=True,
                    schema_overrides={col_name: pl.Float64},
                )
                df.write_parquet(parquet_path, compression="zstd")
                converted += 1
                print(f" OK ({col_name}→float)", flush=True)
            except Exception as e2:
                print(f" FAIL: {e2}", flush=True)
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
