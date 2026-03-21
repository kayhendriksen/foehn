"""Tests for CSV/TXT → Parquet conversion."""

import shutil
from pathlib import Path

import polars as pl
import pytest

from foehn.convert import convert_climate_normals_to_parquet, convert_to_parquet

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def smn_raw_dir(tmp_path):
    """Raw dir with a single SMN CSV in a 'smn' sub-folder."""
    smn_dir = tmp_path / "smn"
    smn_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_dir / "ogd-smn_abo_d_recent.csv")
    return tmp_path


@pytest.fixture()
def climate_normals_raw_dir(tmp_path):
    """Raw dir with a single climate normals TXT in a 'climate_normals' sub-folder."""
    cn_dir = tmp_path / "climate_normals"
    cn_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "climate_normals_sample.txt", cn_dir / "sample.txt")
    return tmp_path


# --- convert_to_parquet ---


def test_convert_to_parquet_creates_file(smn_raw_dir, tmp_path):
    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", smn_raw_dir, parquet_dir)

    out = parquet_dir / "smn" / "ogd-smn_abo_d_recent.parquet"
    assert out.exists()


def test_convert_to_parquet_readable(smn_raw_dir, tmp_path):
    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", smn_raw_dir, parquet_dir)

    df = pl.read_parquet(parquet_dir / "smn" / "ogd-smn_abo_d_recent.parquet")
    assert "station_abbr" in df.columns
    assert len(df) == 3


def test_convert_to_parquet_skips_up_to_date(smn_raw_dir, tmp_path):
    """If Parquet is already newer than CSV, the file should not be re-written."""
    parquet_dir = tmp_path / "parquet"
    convert_to_parquet("smn", smn_raw_dir, parquet_dir)

    out = parquet_dir / "smn" / "ogd-smn_abo_d_recent.parquet"
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
