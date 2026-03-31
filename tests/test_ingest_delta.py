"""Tests for the Polars-based Delta ingestion script."""

import shutil
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from tests.conftest import FIXTURES_DIR

# We can't import the script directly (it has a top-level pyspark import),
# so we patch pyspark before importing.
pyspark_mock = MagicMock()
with patch.dict("sys.modules", {"pyspark": pyspark_mock, "pyspark.sql": pyspark_mock.sql}):
    from scripts.ingest_delta import (
        TABULAR_COLLECTIONS,
        _apply_column_comments,
        _build_schema_overrides,
        _group_csv_files,
        _ingest_climate_normals,
        _ingest_collection,
        _scan_and_collect,
        _table_suffix,
        _validate_identifier,
        _write_to_delta,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def smn_raw_dir(tmp_path):
    """Raw dir with SMN CSVs and metadata in a 'smn' sub-folder."""
    smn_dir = tmp_path / "smn"
    smn_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_dir / "ogd-smn_abo_d_recent.csv")
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_dir / "ogd-smn_ber_d_recent.csv")
    shutil.copy(FIXTURES_DIR / "smn_meta_parameters.csv", smn_dir / "ogd-smn_meta_parameters.csv")
    return tmp_path


@pytest.fixture()
def climate_normals_raw_dir(tmp_path):
    """Raw dir with climate normals TXT files."""
    cn_dir = tmp_path / "climate_normals"
    cn_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "climate_normals_sample.txt", cn_dir / "sample.txt")
    return tmp_path


@pytest.fixture()
def mock_spark():
    """A mock SparkSession that tracks calls."""
    spark = MagicMock()
    # spark.createDataFrame returns a mock with .write.mode().option().saveAsTable()
    mock_writer = MagicMock()
    mock_writer.mode.return_value = mock_writer
    mock_writer.option.return_value = mock_writer
    spark.createDataFrame.return_value.write = mock_writer
    # spark.table().schema.fields returns empty (no columns to comment on)
    spark.table.return_value.schema.fields = []
    return spark


# ── _validate_identifier ─────────────────────────────────────────────────────


def test_validate_identifier_valid():
    assert _validate_identifier("main", "catalog") == "`main`"
    assert _validate_identifier("my-schema", "schema") == "`my-schema`"
    assert _validate_identifier("landing_01", "volume") == "`landing_01`"


def test_validate_identifier_rejects_injection():
    with pytest.raises(ValueError, match="Invalid"):
        _validate_identifier("main; DROP TABLE", "catalog")


# ── _group_csv_files ─────────────────────────────────────────────────────────


def test_group_csv_files(smn_raw_dir):
    groups = _group_csv_files(smn_raw_dir / "smn", "smn")
    assert ("d", "recent") in groups
    assert len(groups[("d", "recent")]) == 2


def test_group_csv_files_excludes_meta(smn_raw_dir):
    groups = _group_csv_files(smn_raw_dir / "smn", "smn")
    all_files = [f for files in groups.values() for f in files]
    assert all("_meta_" not in f.name for f in all_files)


def test_group_csv_files_empty(tmp_path):
    empty_dir = tmp_path / "smn"
    empty_dir.mkdir()
    groups = _group_csv_files(empty_dir, "smn")
    assert groups == {}


# ── _table_suffix ────────────────────────────────────────────────────────────


def test_table_suffix():
    assert _table_suffix(("d", "recent")) == "_d_recent"
    assert _table_suffix(("d",)) == "_d"
    assert _table_suffix(()) == ""


# ── _build_schema_overrides ──────────────────────────────────────────────────


def test_build_schema_overrides(smn_raw_dir):
    files = sorted((smn_raw_dir / "smn").glob("ogd-smn_*_d_recent.csv"))
    metadata_types = {"tre200d0": pl.Float64, "ure200d0": pl.Int64, "nonexistent": pl.Float64}
    overrides = _build_schema_overrides(files, metadata_types)
    assert overrides is not None
    assert overrides["tre200d0"] == pl.Float64
    assert overrides["ure200d0"] == pl.Int64
    assert "nonexistent" not in overrides


def test_build_schema_overrides_no_metadata():
    assert _build_schema_overrides([], {}) is None


# ── _scan_and_collect ────────────────────────────────────────────────────────


def test_scan_and_collect(smn_raw_dir):
    from foehn.convert import _load_metadata_types

    csv_dir = smn_raw_dir / "smn"
    files = sorted(csv_dir.glob("ogd-smn_*_d_recent.csv"))
    metadata_types = _load_metadata_types(csv_dir)

    df = _scan_and_collect(files, metadata_types)

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 6  # 2 files × 3 rows
    assert "station_abbr" in df.columns
    assert "reference_timestamp" in df.columns
    assert df["tre200d0"].dtype == pl.Float64


def test_scan_and_collect_parses_timestamps(smn_raw_dir):
    from foehn.convert import _load_metadata_types

    csv_dir = smn_raw_dir / "smn"
    files = sorted(csv_dir.glob("ogd-smn_*_d_recent.csv"))
    metadata_types = _load_metadata_types(csv_dir)

    df = _scan_and_collect(files, metadata_types)
    assert df["reference_timestamp"].dtype == pl.Datetime


def test_scan_and_collect_single_file(smn_raw_dir):
    from foehn.convert import _load_metadata_types

    csv_dir = smn_raw_dir / "smn"
    files = [sorted(csv_dir.glob("ogd-smn_*_d_recent.csv"))[0]]
    metadata_types = _load_metadata_types(csv_dir)

    df = _scan_and_collect(files, metadata_types)
    assert len(df) == 3


# ── _write_to_delta ──────────────────────────────────────────────────────────


def test_write_to_delta(mock_spark):
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    _write_to_delta(mock_spark, df, "`cat`.`sch`.`tbl`")

    mock_spark.createDataFrame.assert_called_once()
    writer = mock_spark.createDataFrame.return_value.write
    writer.mode.assert_called_with("overwrite")
    writer.mode.return_value.option.assert_called_with("mergeSchema", "true")


def test_write_to_delta_append_mode(mock_spark):
    df = pl.DataFrame({"a": [1]})
    _write_to_delta(mock_spark, df, "`cat`.`sch`.`tbl`", mode="append")

    writer = mock_spark.createDataFrame.return_value.write
    writer.mode.assert_called_with("append")


# ── _apply_column_comments ───────────────────────────────────────────────────


def test_apply_column_comments(smn_raw_dir):
    spark = MagicMock()
    # Simulate table with tre200d0 and ure200d0 columns.
    field1 = MagicMock()
    field1.name = "tre200d0"
    field2 = MagicMock()
    field2.name = "ure200d0"
    spark.table.return_value.schema.fields = [field1, field2]

    _apply_column_comments(spark, "`cat`.`sch`.`smn_d_recent`", smn_raw_dir / "smn")

    # Should have issued ALTER TABLE for matching columns.
    sql_calls = [c.args[0] for c in spark.sql.call_args_list]
    alter_calls = [s for s in sql_calls if "ALTER TABLE" in s]
    assert len(alter_calls) >= 2
    # Check that English descriptions and units are in the comments.
    assert any("Param A en" in c and "[°C]" in c for c in alter_calls)
    assert any("Param D en" in c and "[%]" in c for c in alter_calls)


def test_apply_column_comments_no_metadata(tmp_path):
    """No meta file should be a no-op."""
    spark = MagicMock()
    _apply_column_comments(spark, "`cat`.`sch`.`tbl`", tmp_path)
    spark.sql.assert_not_called()


# ── _ingest_collection ───────────────────────────────────────────────────────


def test_ingest_collection(smn_raw_dir, mock_spark):
    ok, skip = _ingest_collection(
        mock_spark,
        "smn",
        smn_raw_dir / "smn",
        "`main`",
        "`meteoswiss`",
    )
    assert ok == 1  # one group (d_recent)
    assert skip == 0
    mock_spark.createDataFrame.assert_called_once()


def test_ingest_collection_chunked(smn_raw_dir, mock_spark):
    """Chunked mode with chunk_size=1 should produce 2 Spark writes."""
    ok, skip = _ingest_collection(
        mock_spark,
        "smn",
        smn_raw_dir / "smn",
        "`main`",
        "`meteoswiss`",
        chunked=True,
        chunk_size=1,
    )
    assert ok == 1
    assert skip == 0
    # 2 files, chunk_size=1 → 2 createDataFrame calls (overwrite + append)
    assert mock_spark.createDataFrame.call_count == 2


def test_ingest_collection_empty(tmp_path, mock_spark):
    empty_dir = tmp_path / "smn"
    empty_dir.mkdir()
    ok, skip = _ingest_collection(mock_spark, "smn", empty_dir, "`main`", "`meteoswiss`")
    assert ok == 0
    assert skip == 1


# ── _ingest_climate_normals ──────────────────────────────────────────────────


def test_ingest_climate_normals(climate_normals_raw_dir, mock_spark):
    ok, skip = _ingest_climate_normals(mock_spark, climate_normals_raw_dir, "`main`", "`meteoswiss`")
    assert ok == 1
    assert skip == 0
    mock_spark.createDataFrame.assert_called_once()


def test_ingest_climate_normals_no_dir(tmp_path, mock_spark):
    ok, skip = _ingest_climate_normals(mock_spark, tmp_path, "`main`", "`meteoswiss`")
    assert ok == 0
    assert skip == 1


def test_ingest_climate_normals_empty_dir(tmp_path, mock_spark):
    (tmp_path / "climate_normals").mkdir()
    ok, skip = _ingest_climate_normals(mock_spark, tmp_path, "`main`", "`meteoswiss`")
    assert ok == 0
    assert skip == 1


# ── TABULAR_COLLECTIONS ─────────────────────────────────────────────────────


def test_tabular_collections_excludes_binary():
    from foehn.collections import GRIB2_COLLECTIONS, NETCDF_COLLECTIONS

    for key in TABULAR_COLLECTIONS:
        assert key not in GRIB2_COLLECTIONS
        assert key not in NETCDF_COLLECTIONS


def test_tabular_collections_includes_climate_normals():
    assert "climate_normals" in TABULAR_COLLECTIONS
