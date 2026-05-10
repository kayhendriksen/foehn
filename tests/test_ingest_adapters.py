"""Tests for foehn.ingest adapter implementations."""

from __future__ import annotations

from unittest.mock import MagicMock

import polars as pl
import pytest

from foehn.ingest.adapters.memory import RecordingBinaryFileIndex, RecordingDeltaSink
from foehn.ingest.adapters.spark_delta import SparkBinaryFileIndex, SparkDeltaSink

# ── RecordingDeltaSink ───────────────────────────────────────────────────────


def test_recording_sink_overwrite_replaces_table():
    sink = RecordingDeltaSink()
    sink.write(pl.DataFrame({"a": [1, 2]}), "t")
    sink.write(pl.DataFrame({"a": [9]}), "t")  # overwrite (default mode)

    assert sink.tables["t"].to_dict(as_series=False) == {"a": [9]}
    assert [c.mode for c in sink.calls] == ["overwrite", "overwrite"]


def test_recording_sink_append_concats():
    sink = RecordingDeltaSink()
    sink.write(pl.DataFrame({"a": [1]}), "t")
    sink.write(pl.DataFrame({"a": [2]}), "t", mode="append")
    sink.write(pl.DataFrame({"a": [3]}), "t", mode="append")

    assert sink.tables["t"].to_dict(as_series=False) == {"a": [1, 2, 3]}
    assert [c.mode for c in sink.calls] == ["overwrite", "append", "append"]


def test_recording_sink_apply_comments_accumulates():
    sink = RecordingDeltaSink()
    sink.apply_comments("t", {"a": "first"})
    sink.apply_comments("t", {"a": "second", "b": "new"})

    assert sink.comments["t"] == {"a": "second", "b": "new"}


# ── RecordingBinaryFileIndex ─────────────────────────────────────────────────


def test_recording_index_returns_h5_count(tmp_path):
    (tmp_path / "a.h5").write_bytes(b"")
    (tmp_path / "b.h5").write_bytes(b"")
    (tmp_path / "ignore.txt").write_text("")

    index = RecordingBinaryFileIndex()
    count = index.merge_index(tmp_path, "`cat`.`sch`.`radar_precip`")

    assert count == 2
    assert index.merges == [(tmp_path, "`cat`.`sch`.`radar_precip`")]


# ── SparkDeltaSink ───────────────────────────────────────────────────────────


@pytest.fixture()
def mock_spark():
    spark = MagicMock()
    writer = MagicMock()
    writer.mode.return_value = writer
    writer.option.return_value = writer
    spark.createDataFrame.return_value.write = writer
    spark.table.return_value.schema.fields = []
    return spark


def test_spark_sink_write_invokes_save(mock_spark):
    sink = SparkDeltaSink(mock_spark)
    sink.write(pl.DataFrame({"a": [1]}), "`cat`.`sch`.`tbl`")

    mock_spark.createDataFrame.assert_called_once()
    writer = mock_spark.createDataFrame.return_value.write
    writer.mode.assert_called_with("overwrite")
    writer.mode.return_value.option.assert_called_with("mergeSchema", "true")
    writer.mode.return_value.option.return_value.saveAsTable.assert_called_with("`cat`.`sch`.`tbl`")


def test_spark_sink_write_append_mode(mock_spark):
    sink = SparkDeltaSink(mock_spark)
    sink.write(pl.DataFrame({"a": [1]}), "`cat`.`sch`.`tbl`", mode="append")

    writer = mock_spark.createDataFrame.return_value.write
    writer.mode.assert_called_with("append")


def test_spark_sink_apply_comments_filters_missing_columns(mock_spark):
    field_a = MagicMock()
    field_a.name = "a"
    mock_spark.table.return_value.schema.fields = [field_a]

    sink = SparkDeltaSink(mock_spark)
    sink.apply_comments("`cat`.`sch`.`tbl`", {"a": "exists", "b": "missing"})

    sql_calls = [c.args[0] for c in mock_spark.sql.call_args_list]
    assert any("ALTER TABLE" in s and "`a`" in s and "exists" in s for s in sql_calls)
    assert not any("`b`" in s for s in sql_calls)


def test_spark_sink_apply_comments_swallows_table_lookup_failure(mock_spark):
    mock_spark.table.side_effect = RuntimeError("table not found")

    sink = SparkDeltaSink(mock_spark)
    # Must not raise.
    sink.apply_comments("`cat`.`sch`.`tbl`", {"a": "x"})

    # No ALTER TABLE issued.
    sql_calls = [c.args[0] for c in mock_spark.sql.call_args_list]
    assert not any("ALTER TABLE" in s for s in sql_calls)


def test_spark_sink_apply_comments_escapes_single_quote(mock_spark):
    field_a = MagicMock()
    field_a.name = "a"
    mock_spark.table.return_value.schema.fields = [field_a]

    sink = SparkDeltaSink(mock_spark)
    sink.apply_comments("`cat`.`sch`.`tbl`", {"a": "it's tricky"})

    sql_calls = [c.args[0] for c in mock_spark.sql.call_args_list]
    alter = next(s for s in sql_calls if "ALTER TABLE" in s)
    assert "it\\'s tricky" in alter


# ── SparkBinaryFileIndex ─────────────────────────────────────────────────────


def test_spark_index_merge_calls_create_then_merge(tmp_path, mock_spark):
    new_df = MagicMock()
    new_df.count.return_value = 3
    (mock_spark.read.format.return_value.option.return_value.load.return_value.selectExpr.return_value) = new_df

    h5_dir = tmp_path / "radar_precip"
    h5_dir.mkdir()

    index = SparkBinaryFileIndex(mock_spark)
    count = index.merge_index(h5_dir, "`cat`.`sch`.`radar_precip`")

    assert count == 3
    sql_calls = [c.args[0] for c in mock_spark.sql.call_args_list]
    assert any("CREATE TABLE IF NOT EXISTS" in s and "`cat`.`sch`.`radar_precip`" in s for s in sql_calls)
    assert any("MERGE INTO" in s for s in sql_calls)
    new_df.createOrReplaceTempView.assert_called_with("_radar_precip_new")
