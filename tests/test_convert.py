"""Tests for CSV/TXT → Parquet conversion."""

import shutil
from pathlib import Path

import polars as pl
import pytest

from foehn.convert import (
    _load_metadata_types,
    _parse_metadata_types,
    convert_climate_normals_to_parquet,
    convert_to_parquet,
    parse_csv_bytes,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def smn_raw_dir(tmp_path):
    """Raw dir with SMN CSVs and metadata in a 'smn' sub-folder."""
    smn_dir = tmp_path / "smn"
    smn_dir.mkdir()
    # Two station files in the same group (d_recent) to test combining.
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_dir / "ogd-smn_abo_d_recent.csv")
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_dir / "ogd-smn_ber_d_recent.csv")
    shutil.copy(
        FIXTURES_DIR / "smn_meta_parameters.csv",
        smn_dir / "ogd-smn_meta_parameters.csv",
    )
    return tmp_path


@pytest.fixture()
def climate_normals_raw_dir(tmp_path):
    """Raw dir with a single climate normals TXT in a 'climate_normals' sub-folder."""
    cn_dir = tmp_path / "climate_normals"
    cn_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "climate_normals_sample.txt", cn_dir / "sample.txt")
    return tmp_path


# --- convert_to_parquet ---


def test_convert_to_parquet_creates_combined_file(smn_raw_dir, tmp_path):
    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", smn_raw_dir, parquet_dir)

    out = parquet_dir / "smn" / "smn_d_recent.parquet"
    assert out.exists()


def test_convert_to_parquet_combines_stations(smn_raw_dir, tmp_path):
    """Two per-station CSVs in the same group should be combined into one Parquet."""
    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", smn_raw_dir, parquet_dir)

    df = pl.read_parquet(parquet_dir / "smn" / "smn_d_recent.parquet")
    assert "station_abbr" in df.columns
    # 2 files × 3 rows each = 6 rows
    assert len(df) == 6


def test_convert_to_parquet_skips_up_to_date(smn_raw_dir, tmp_path):
    """If Parquet is already newer than all CSVs, the group should not be re-written."""
    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", smn_raw_dir, parquet_dir)

    out = parquet_dir / "smn" / "smn_d_recent.parquet"
    mtime_before = out.stat().st_mtime

    convert_to_parquet("smn", smn_raw_dir, parquet_dir)
    assert out.stat().st_mtime == mtime_before


def test_convert_to_parquet_no_csv_is_noop(tmp_path):
    """Empty raw dir should not raise and should produce no output."""
    raw_dir = tmp_path / "raw"
    (raw_dir / "smn").mkdir(parents=True)
    parquet_dir = tmp_path / "parquet"

    convert_to_parquet("smn", raw_dir, parquet_dir)

    assert not (parquet_dir / "smn").exists() or not list((parquet_dir / "smn").iterdir())


# --- _load_metadata_types ---


def test_load_metadata_types(smn_raw_dir):
    """Metadata file should produce a mapping of parameter names to Polars dtypes."""
    types = _load_metadata_types(smn_raw_dir / "smn")
    assert types["tre200d0"] == pl.Float64
    assert types["rre150d0"] == pl.Float64
    assert types["ure200d0"] == pl.Int64
    assert types["sre000d0"] == pl.Int64
    assert "station_abbr" not in types


def test_load_metadata_types_no_file(tmp_path):
    """Returns empty dict when no metadata file exists."""
    assert _load_metadata_types(tmp_path) == {}


def test_convert_applies_metadata_types(smn_raw_dir, tmp_path):
    """Columns listed as Float in metadata should be Float64 in the Parquet output."""
    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", smn_raw_dir, parquet_dir)

    df = pl.read_parquet(parquet_dir / "smn" / "smn_d_recent.parquet")
    assert df["tre200d0"].dtype == pl.Float64
    assert df["rre150d0"].dtype == pl.Float64


def test_convert_skips_meta_files(smn_raw_dir, tmp_path):
    """Metadata CSVs should not be converted to Parquet."""
    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", smn_raw_dir, parquet_dir)

    out_dir = parquet_dir / "smn"
    parquet_files = list(out_dir.glob("*.parquet"))
    assert all("_meta_" not in f.name for f in parquet_files)


# --- convert_climate_normals_to_parquet ---


def test_convert_climate_normals_creates_file(climate_normals_raw_dir, tmp_path):
    parquet_dir = tmp_path / "parquet"
    convert_climate_normals_to_parquet(climate_normals_raw_dir, parquet_dir)

    out = parquet_dir / "climate_normals" / "sample.parquet"
    assert out.exists()


def test_convert_climate_normals_readable(climate_normals_raw_dir, tmp_path):
    parquet_dir = tmp_path / "parquet"
    convert_climate_normals_to_parquet(climate_normals_raw_dir, parquet_dir)

    df = pl.read_parquet(parquet_dir / "climate_normals" / "sample.parquet")
    assert "Station" in df.columns
    assert "Jan" in df.columns
    assert len(df) == 2


# --- _parse_metadata_types edge cases ---


def test_parse_metadata_types_invalid_content():
    """Unparseable content returns empty dict."""
    assert _parse_metadata_types(b"not;valid;csv\x00\xff") == {}


def test_parse_metadata_types_missing_columns():
    """CSV without expected columns returns empty dict."""
    assert _parse_metadata_types(b"col_a;col_b\nfoo;bar\n") == {}


# --- parse_csv_bytes edge cases ---


def test_parse_csv_bytes_header_read_failure():
    """When metadata_types are given but header reading fails, parsing still works."""
    content = b"a;b\n1;2\n3;4\n"
    # Pass metadata types but with a bad separator scenario — should still parse
    df = parse_csv_bytes(content, metadata_types={"a": pl.Float64})
    assert len(df) == 2


def test_parse_csv_bytes_conversion_error_without_column_match():
    """When error message doesn't contain a column name, the error is raised."""
    # Create CSV that will cause an error Polars can't recover from
    content = b""  # empty content
    with pytest.raises(pl.exceptions.NoDataError):
        parse_csv_bytes(content)


# --- convert_to_parquet error handling ---


def test_convert_to_parquet_handles_bad_csv(tmp_path, capsys):
    """A corrupt CSV should print FAIL but not crash the whole conversion."""
    raw_dir = tmp_path / "raw"
    csv_dir = raw_dir / "smn"
    csv_dir.mkdir(parents=True)
    (csv_dir / "ogd-smn_ber_d_recent.csv").write_text("a;b\n1;2\n")
    (csv_dir / "ogd-smn_zur_d_recent.csv").write_bytes(b"")

    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", raw_dir, parquet_dir)

    out = capsys.readouterr().out
    assert "FAIL" in out


# --- convert_climate_normals edge cases ---


def test_convert_climate_normals_no_txt_files(tmp_path):
    """Empty climate_normals dir should not raise."""
    raw_dir = tmp_path / "raw"
    (raw_dir / "climate_normals").mkdir(parents=True)
    parquet_dir = tmp_path / "parquet"
    convert_climate_normals_to_parquet(raw_dir, parquet_dir)
    # Dir may be created but should contain no parquet files
    cn_dir = parquet_dir / "climate_normals"
    assert not cn_dir.exists() or not list(cn_dir.glob("*.parquet"))


def test_convert_climate_normals_skips_up_to_date(climate_normals_raw_dir, tmp_path):
    """Already-converted files should be skipped on second run."""
    parquet_dir = tmp_path / "parquet"
    convert_climate_normals_to_parquet(climate_normals_raw_dir, parquet_dir)

    out_file = parquet_dir / "climate_normals" / "sample.parquet"
    mtime_before = out_file.stat().st_mtime

    convert_climate_normals_to_parquet(climate_normals_raw_dir, parquet_dir)
    assert out_file.stat().st_mtime == mtime_before


def test_convert_climate_normals_handles_bad_file(tmp_path, capsys):
    """A corrupt TXT should print FAIL but not crash."""
    raw_dir = tmp_path
    cn_dir = raw_dir / "climate_normals"
    cn_dir.mkdir()
    (cn_dir / "bad.txt").write_bytes(b"\x00\xff")

    parquet_dir = tmp_path / "parquet"
    convert_climate_normals_to_parquet(raw_dir, parquet_dir)

    out = capsys.readouterr().out
    assert "FAIL" in out
