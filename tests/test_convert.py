"""Tests for CSV/TXT → Parquet conversion."""

import shutil
from pathlib import Path

import polars as pl
import pytest

from foehn.convert import (
    _load_metadata_types,
    _parse_metadata_types,
    convert_climate_normals_to_parquet,
    convert_climate_scenarios_indoor_to_parquet,
    convert_to_parquet,
    parse_csv_bytes,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def smn_bronze_dir(tmp_path):
    """Bronze dir with SMN CSVs and metadata in a 'smn' sub-folder."""
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
def climate_normals_bronze_dir(tmp_path):
    """Bronze dir with a single climate normals TXT in a 'climate_normals' sub-folder."""
    cn_dir = tmp_path / "climate_normals"
    cn_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "climate_normals_sample.txt", cn_dir / "sample.txt")
    return tmp_path


# --- convert_to_parquet ---


def test_convert_to_parquet_creates_combined_file(smn_bronze_dir, tmp_path):
    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", smn_bronze_dir, parquet_dir)

    out = parquet_dir / "smn" / "smn_d_recent.parquet"
    assert out.exists()


def test_convert_to_parquet_combines_stations(smn_bronze_dir, tmp_path):
    """Two per-station CSVs in the same group should be combined into one Parquet."""
    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", smn_bronze_dir, parquet_dir)

    df = pl.read_parquet(parquet_dir / "smn" / "smn_d_recent.parquet")
    assert "station_abbr" in df.columns
    # 2 files × 3 rows each = 6 rows
    assert len(df) == 6


def test_convert_to_parquet_skips_up_to_date(smn_bronze_dir, tmp_path):
    """If Parquet is already newer than all CSVs, the group should not be re-written."""
    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", smn_bronze_dir, parquet_dir)

    out = parquet_dir / "smn" / "smn_d_recent.parquet"
    mtime_before = out.stat().st_mtime

    convert_to_parquet("smn", smn_bronze_dir, parquet_dir)
    assert out.stat().st_mtime == mtime_before


def test_convert_to_parquet_no_csv_is_noop(tmp_path):
    """Empty bronze dir should not raise and should produce no output."""
    bronze_dir = tmp_path / "bronze"
    (bronze_dir / "smn").mkdir(parents=True)
    parquet_dir = tmp_path / "parquet"

    convert_to_parquet("smn", bronze_dir, parquet_dir)

    assert not (parquet_dir / "smn").exists() or not list((parquet_dir / "smn").iterdir())


# --- _load_metadata_types ---


def test_load_metadata_types(smn_bronze_dir):
    """Metadata file should produce a mapping of parameter names to Polars dtypes."""
    types = _load_metadata_types(smn_bronze_dir / "smn")
    assert types["tre200d0"] == pl.Float64
    assert types["rre150d0"] == pl.Float64
    assert types["ure200d0"] == pl.Int64
    assert types["sre000d0"] == pl.Int64
    assert "station_abbr" not in types


def test_load_metadata_types_no_file(tmp_path):
    """Returns empty dict when no metadata file exists."""
    assert _load_metadata_types(tmp_path) == {}


def test_convert_applies_metadata_types(smn_bronze_dir, tmp_path):
    """Columns listed as Float in metadata should be Float64 in the Parquet output."""
    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", smn_bronze_dir, parquet_dir)

    df = pl.read_parquet(parquet_dir / "smn" / "smn_d_recent.parquet")
    assert df["tre200d0"].dtype == pl.Float64
    assert df["rre150d0"].dtype == pl.Float64


def test_convert_skips_meta_files(smn_bronze_dir, tmp_path):
    """Metadata CSVs should not be converted to Parquet."""
    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", smn_bronze_dir, parquet_dir)

    out_dir = parquet_dir / "smn"
    parquet_files = list(out_dir.glob("*.parquet"))
    assert all("_meta_" not in f.name for f in parquet_files)


# --- convert_climate_normals_to_parquet ---


def test_convert_climate_normals_creates_file(climate_normals_bronze_dir, tmp_path):
    parquet_dir = tmp_path / "parquet"
    convert_climate_normals_to_parquet(climate_normals_bronze_dir, parquet_dir)

    out = parquet_dir / "climate_normals" / "sample.parquet"
    assert out.exists()


def test_convert_climate_normals_readable(climate_normals_bronze_dir, tmp_path):
    parquet_dir = tmp_path / "parquet"
    convert_climate_normals_to_parquet(climate_normals_bronze_dir, parquet_dir)

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
    bronze_dir = tmp_path / "bronze"
    csv_dir = bronze_dir / "smn"
    csv_dir.mkdir(parents=True)
    (csv_dir / "ogd-smn_ber_d_recent.csv").write_text("a;b\n1;2\n")
    (csv_dir / "ogd-smn_zur_d_recent.csv").write_bytes(b"")

    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", bronze_dir, parquet_dir)

    out = capsys.readouterr().out
    assert "FAIL" in out


# --- convert_climate_normals edge cases ---


def test_convert_climate_normals_no_txt_files(tmp_path):
    """Empty climate_normals dir should not raise."""
    bronze_dir = tmp_path / "bronze"
    (bronze_dir / "climate_normals").mkdir(parents=True)
    parquet_dir = tmp_path / "parquet"
    convert_climate_normals_to_parquet(bronze_dir, parquet_dir)
    # Dir may be created but should contain no parquet files
    cn_dir = parquet_dir / "climate_normals"
    assert not cn_dir.exists() or not list(cn_dir.glob("*.parquet"))


def test_convert_climate_normals_skips_up_to_date(climate_normals_bronze_dir, tmp_path):
    """Already-converted files should be skipped on second run."""
    parquet_dir = tmp_path / "parquet"
    convert_climate_normals_to_parquet(climate_normals_bronze_dir, parquet_dir)

    out_file = parquet_dir / "climate_normals" / "sample.parquet"
    mtime_before = out_file.stat().st_mtime

    convert_climate_normals_to_parquet(climate_normals_bronze_dir, parquet_dir)
    assert out_file.stat().st_mtime == mtime_before


def test_convert_climate_normals_handles_bad_file(tmp_path, capsys):
    """A corrupt TXT should print FAIL but not crash."""
    bronze_dir = tmp_path
    cn_dir = bronze_dir / "climate_normals"
    cn_dir.mkdir()
    (cn_dir / "bad.txt").write_bytes(b"\x00\xff")

    parquet_dir = tmp_path / "parquet"
    convert_climate_normals_to_parquet(bronze_dir, parquet_dir)

    out = capsys.readouterr().out
    assert "FAIL" in out


# --- indoor climate scenarios (zipped multi-CSV) ---

_INDOOR_CSV = (
    "time.yy,time.mm,time.dd,time.hh,tre200h0,ure200h0,skycover\n"
    "2035,1,1,0,0.3,94.5,36\n"
    "2035,1,1,1,-0.2,94.6,26\n"
    "2035,1,1,2,-0.5,94.3,20\n"
)


def test_convert_indoor_scenarios_parses_filename_and_timestamp(tmp_path):
    bronze_dir = tmp_path / "bronze"
    indoor_dir = bronze_dir / "climate_scenarios_indoor"
    indoor_dir.mkdir(parents=True)
    (indoor_dir / "ABO_2035_RCP85_1in10-warmsummer.csv").write_text(_INDOOR_CSV)
    (indoor_dir / "AIG_2060_RCP26_DRY.csv").write_text(_INDOOR_CSV)

    parquet_dir = tmp_path / "parquet"
    failed = convert_climate_scenarios_indoor_to_parquet(bronze_dir, parquet_dir)
    assert failed == 0

    out = parquet_dir / "climate_scenarios_indoor" / "climate_scenarios_indoor.parquet"
    assert out.exists()

    df = pl.read_parquet(out)
    # 3 rows per file × 2 files
    assert len(df) == 6
    # filename-derived columns + synthesised timestamp + dropped time.* cols
    assert {"station_abbr", "period", "scenario", "variant", "reference_timestamp"} <= set(df.columns)
    assert not any(c.startswith("time.") for c in df.columns)
    assert set(df["station_abbr"].unique()) == {"ABO", "AIG"}
    assert set(df["scenario"].unique()) == {"RCP85", "RCP26"}
    assert "1in10-warmsummer" in set(df["variant"].unique())
    assert df["reference_timestamp"].dtype == pl.Datetime


def test_convert_indoor_scenarios_no_files_returns_zero(tmp_path):
    assert convert_climate_scenarios_indoor_to_parquet(tmp_path / "bronze", tmp_path / "parquet") == 0


def test_convert_indoor_scenarios_skips_metadata_file(tmp_path):
    """The archive's non-data metadata CSV must be skipped, not counted as a failure."""
    bronze_dir = tmp_path / "bronze"
    indoor_dir = bronze_dir / "climate_scenarios_indoor"
    indoor_dir.mkdir(parents=True)
    (indoor_dir / "ABO_2035_RCP85_DRY.csv").write_text(_INDOOR_CSV)
    (indoor_dir / "Klimaszenarien-Raumklima_Metadata.csv").write_text("foo,bar\n1,2\n")

    parquet_dir = tmp_path / "parquet"
    failed = convert_climate_scenarios_indoor_to_parquet(bronze_dir, parquet_dir)
    assert failed == 0

    df = pl.read_parquet(parquet_dir / "climate_scenarios_indoor" / "climate_scenarios_indoor.parquet")
    assert set(df["station_abbr"].unique()) == {"ABO"}
