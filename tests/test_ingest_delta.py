"""Tests for the Polars-based Delta ingestion script."""

import re
import shutil
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from conftest import FIXTURES_DIR

from foehn import registry
from scripts.ingest_delta import (
    INGESTED_DATASETS,
    RADAR_PRODUCT_PATTERN,
    _apply_column_comments,
    _ingest_climate_normals,
    _ingest_collection,
    _ingest_metadata,
    _ingest_radar,
    _scan_and_collect,
    _table_suffix,
    _validate_identifier,
    _write_to_delta,
    main,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def smn_bronze_dir(tmp_path):
    """Bronze dir with SMN CSVs and metadata in a 'smn' sub-folder."""
    smn_dir = tmp_path / "smn"
    smn_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_dir / "ogd-smn_abo_d_recent.csv")
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_dir / "ogd-smn_ber_d_recent.csv")
    shutil.copy(FIXTURES_DIR / "smn_meta_parameters.csv", smn_dir / "ogd-smn_meta_parameters.csv")
    return tmp_path


@pytest.fixture
def climate_normals_bronze_dir(tmp_path):
    """Bronze dir with climate normals TXT files."""
    cn_dir = tmp_path / "climate_normals"
    cn_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "climate_normals_sample.txt", cn_dir / "sample.txt")
    return tmp_path


@pytest.fixture
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


# ── _table_suffix ────────────────────────────────────────────────────────────


def test_table_suffix():
    assert _table_suffix(("d", "recent")) == "_d_recent"
    assert _table_suffix(("d",)) == "_d"
    assert _table_suffix(()) == ""


# ── _scan_and_collect ────────────────────────────────────────────────────────


def test_scan_and_collect(smn_bronze_dir):
    from foehn.convert import load_metadata_types

    csv_dir = smn_bronze_dir / "smn"
    files = sorted(csv_dir.glob("ogd-smn_*_d_recent.csv"))
    metadata_types = load_metadata_types(csv_dir)

    df = _scan_and_collect("smn", files, metadata_types)

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 6  # 2 files × 3 rows
    assert "station_abbr" in df.columns
    assert "reference_timestamp" in df.columns
    assert df["tre200d0"].dtype == pl.Float64


def test_scan_and_collect_parses_timestamps(smn_bronze_dir):
    from foehn.convert import load_metadata_types

    csv_dir = smn_bronze_dir / "smn"
    files = sorted(csv_dir.glob("ogd-smn_*_d_recent.csv"))
    metadata_types = load_metadata_types(csv_dir)

    df = _scan_and_collect("smn", files, metadata_types)
    assert df["reference_timestamp"].dtype == pl.Datetime


def test_scan_and_collect_single_file(smn_bronze_dir):
    from foehn.convert import load_metadata_types

    csv_dir = smn_bronze_dir / "smn"
    files = [min(csv_dir.glob("ogd-smn_*_d_recent.csv"))]
    metadata_types = load_metadata_types(csv_dir)

    df = _scan_and_collect("smn", files, metadata_types)
    assert len(df) == 3


def test_scan_and_collect_normalizes_preamble_csv(tmp_path):
    from conftest import CLIMATE_SCENARIOS_CSV

    path = tmp_path / "ogd-climate-scenarios-ch2025_abe_pr_gwl1.5.csv"
    path.write_text(CLIMATE_SCENARIOS_CSV)

    df = _scan_and_collect("climate_scenarios", [path], {})

    assert df.columns[:4] == ["station_abbr", "variable", "gwl", "date"]


def test_scan_and_collect_normalizes_forecast_csv(tmp_path):
    path = tmp_path / "vnut12.lssw.202605210000.dkl010h0.csv"
    path.write_text("point_id;point_type_id;Date;dkl010h0\n1;1;202605202100;282\n")

    df = _scan_and_collect("forecast_local", [path], {})

    assert df["reference_timestamp"].dtype == pl.Datetime


def _forecast_run_files(tmp_path):
    tre = tmp_path / "vnut12.lssw.202605210000.tre200h0.csv"
    tre.write_text("point_id;point_type_id;Date;tre200h0\n1;1;202605210000;10.5\n1;2;202605210000;3.5\n")
    rre = tmp_path / "vnut12.lssw.202605210000.rre150h0.csv"
    rre.write_text("point_id;point_type_id;Date;rre150h0\n1;1;202605210000;0.2\n1;2;202605210000;1.4\n")
    return [rre, tre]


def test_scan_and_collect_joins_forecast_files_per_point_and_time(tmp_path):
    """One row per point and time, every parameter a column — not stacked."""
    df = _scan_and_collect("forecast_local", _forecast_run_files(tmp_path), {})

    assert df.height == 2
    assert df.filter(pl.col("point_type_id") == 2).select("tre200h0", "rre150h0").row(0) == (3.5, 1.4)
    assert "reference_timestamp_right" not in df.columns
    assert df["reference_timestamp"].null_count() == 0


def test_scan_and_collect_eager_fallback_also_joins_forecast_files(tmp_path, monkeypatch):
    collect = pl.LazyFrame.collect

    # Only the streaming collect fails: eager DataFrame methods run through
    # LazyFrame.collect on newer Polars, so failing every call broke the fallback too.
    def boom(self, *args, **kwargs):
        if kwargs.get("engine") == "streaming":
            raise pl.exceptions.ComputeError("streaming collect failed")
        return collect(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", boom)

    df = _scan_and_collect("forecast_local", _forecast_run_files(tmp_path), {})

    assert df.height == 2
    assert df.select("tre200h0", "rre150h0").null_count().row(0) == (0, 0)


def test_scan_and_collect_normalizes_archive_csv(tmp_path):
    path = tmp_path / "ABE_2035_RCP85_DRY.csv"
    path.write_text("time.yy,time.mm,time.dd,time.hh,tre200h0\n2035,1,1,0,0.3\n")

    df = _scan_and_collect("climate_scenarios_indoor", [path], {})

    assert df["station_abbr"].to_list() == ["ABE"]
    assert df["reference_timestamp"].dtype == pl.Datetime


# ── _write_to_delta ──────────────────────────────────────────────────────────


# scripts.ingest_delta._write_to_delta hands Spark an Arrow table, so anything
# reaching it needs pyarrow. It has no wheels for 3.15 yet, where this module is
# otherwise importable — skip rather than error, as the grids tests do for xarray.
def _require_arrow():
    pytest.importorskip("pyarrow", reason="pyarrow has no wheels for this interpreter")


def test_write_to_delta(mock_spark):
    _require_arrow()
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    _write_to_delta(mock_spark, df, "`cat`.`sch`.`tbl`")

    mock_spark.createDataFrame.assert_called_once()
    writer = mock_spark.createDataFrame.return_value.write
    writer.mode.assert_called_with("overwrite")
    writer.mode.return_value.option.assert_called_with("overwriteSchema", "true")


def test_write_to_delta_append_mode(mock_spark):
    _require_arrow()
    df = pl.DataFrame({"a": [1]})
    _write_to_delta(mock_spark, df, "`cat`.`sch`.`tbl`", mode="append")

    writer = mock_spark.createDataFrame.return_value.write
    writer.mode.assert_called_with("append")
    writer.mode.return_value.option.assert_called_with("mergeSchema", "true")


# ── _apply_column_comments ───────────────────────────────────────────────────


def test_apply_column_comments(smn_bronze_dir):
    spark = MagicMock()
    # Simulate table with tre200d0 and ure200d0 columns.
    field1 = MagicMock()
    field1.name = "tre200d0"
    field2 = MagicMock()
    field2.name = "ure200d0"
    spark.table.return_value.schema.fields = [field1, field2]

    _apply_column_comments(spark, "`cat`.`sch`.`smn_d_recent`", smn_bronze_dir / "smn")

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


def test_apply_column_comments_escapes_backslashes_and_quotes(tmp_path):
    """A description ending in ``\\'`` must not close the SQL string literal."""
    (tmp_path / "x_meta_parameters.csv").write_text(
        "parameter_shortname;parameter_description_en;parameter_unit\ntre200d0;it's 5\\;°C\n",
        encoding="utf-8",
    )
    spark = MagicMock()
    field = MagicMock()
    field.name = "tre200d0"
    spark.table.return_value.schema.fields = [field]

    _apply_column_comments(spark, "`cat`.`sch`.`tbl`", tmp_path)

    sql_calls = [c.args[0] for c in spark.sql.call_args_list]
    # Backslash doubled first, then the quote escaped: it's 5\ [°C] → it\'s 5\\ [°C]
    assert any("COMMENT 'it\\'s 5\\\\ [°C]'" in s for s in sql_calls)


# ── _ingest_collection ───────────────────────────────────────────────────────


def test_ingest_collection(smn_bronze_dir, mock_spark):
    pytest.importorskip("pyarrow")  # tables reach Spark through Polars' to_arrow()
    ok, skip, failed = _ingest_collection(
        mock_spark,
        "smn",
        smn_bronze_dir / "smn",
        "`main`",
        "`meteoswiss`",
    )
    # 1 data group (d_recent) + 1 meta table (parameters)
    assert ok == 2
    assert skip == 0
    assert failed == 0
    assert mock_spark.createDataFrame.call_count == 2


def test_ingest_collection_chunked(smn_bronze_dir, mock_spark):
    """Chunked mode with chunk_size=1 should produce 2 data writes + 1 meta write."""
    pytest.importorskip("pyarrow")  # tables reach Spark through Polars' to_arrow()
    ok, skip, failed = _ingest_collection(
        mock_spark,
        "smn",
        smn_bronze_dir / "smn",
        "`main`",
        "`meteoswiss`",
        chunked=True,
        chunk_size=1,
    )
    # 1 data group + 1 meta table
    assert (ok, skip, failed) == (2, 0, 0)
    # 2 files × chunk_size=1 (overwrite + append) + 1 meta write = 3
    assert mock_spark.createDataFrame.call_count == 3
    # Both chunks went to this run's staging table, which then replaced the table once.
    read_back = [call.args[0] for call in mock_spark.table.call_args_list]
    assert any(_STAGING_RE.fullmatch(name) for name in read_back)
    published = mock_spark.table.return_value.write.mode.return_value.option.return_value.saveAsTable
    published.assert_called_once_with("`main`.`meteoswiss`.`smn_d_recent`")


_STAGING_RE = re.compile(r"`main`\.`meteoswiss`\.`smn_d_recent__foehn_staging_[0-9a-f]{12}`")


class _TableStore:
    """Just enough of a SparkSession to keep tables, so overlapping runs can be observed."""

    def __init__(self):
        self.tables: dict[str, pl.DataFrame] = {}

    def createDataFrame(self, arrow):  # noqa: N802 - Spark's name
        return _Frame(self, pl.from_arrow(arrow))

    def table(self, name):
        return _Frame(self, self.tables[name])

    def sql(self, statement):
        if statement.startswith("DROP TABLE IF EXISTS "):
            self.tables.pop(statement.removeprefix("DROP TABLE IF EXISTS "), None)


class _Frame:
    def __init__(self, store, frame):
        self.store, self.frame, self._mode = store, frame, "error"

    @property
    def write(self):
        return self

    def mode(self, mode):
        self._mode = mode
        return self

    def option(self, *_):
        return self

    def saveAsTable(self, name):  # noqa: N802 - Spark's name
        existing = self.store.tables.get(name)
        appending = self._mode == "append" and existing is not None
        self.store.tables[name] = pl.concat([existing, self.frame]) if appending else self.frame


def test_overlapping_chunked_ingests_each_publish_a_complete_table(smn_bronze_dir):
    """Two runs shared one staging table, and each overwrote, appended to and dropped the other's chunks.

    The second run happens entirely between the first run's two chunks here, which
    is the interleaving that published a table made of one chunk.
    """
    pytest.importorskip("pyarrow")  # tables reach Spark through Polars' to_arrow()
    store = _TableStore()
    state = {"scans": 0, "inner": False}

    def scan(key, files, metadata_types):
        state["scans"] += 1
        if state["scans"] == 2:  # the outer run's second chunk: run a whole other ingest first
            state["inner"] = True
            inner = _ingest_collection(
                store, "smn", smn_bronze_dir / "smn", "`main`", "`meteoswiss`", chunked=True, chunk_size=1
            )
            state["inner"] = False
            assert tuple(inner)[2] == 0
        return pl.DataFrame({"run": ["inner" if state["inner"] else "outer"], "chunk": [state["scans"]]})

    with patch("scripts.ingest_delta._scan_and_collect", side_effect=scan):
        outer = _ingest_collection(
            store, "smn", smn_bronze_dir / "smn", "`main`", "`meteoswiss`", chunked=True, chunk_size=1
        )

    assert tuple(outer)[2] == 0
    published = store.tables["`main`.`meteoswiss`.`smn_d_recent`"]
    assert published["run"].to_list() == ["outer", "outer"]
    assert not [name for name in store.tables if "__foehn_staging" in name]


def _written_tables(spark):
    return {c.args[0] for c in spark.createDataFrame.return_value.write.saveAsTable.call_args_list}


def test_daily_ingest_leaves_the_historical_tables_alone(smn_bronze_dir, mock_spark):
    """A backfill's historical CSVs stay in Bronze; the daily pass used to rebuild them every day."""
    pytest.importorskip("pyarrow")  # tables reach Spark through Polars' to_arrow()
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_bronze_dir / "smn" / "ogd-smn_ber_d_historical.csv")

    ok, skip, failed = _ingest_collection(mock_spark, "smn", smn_bronze_dir / "smn", "`main`", "`meteoswiss`")

    assert (ok, skip, failed) == (2, 1, 0)
    assert "`main`.`meteoswiss`.`smn_d_historical`" not in _written_tables(mock_spark)
    assert "`main`.`meteoswiss`.`smn_d_recent`" in _written_tables(mock_spark)


def test_historical_ingest_writes_the_historical_tables(smn_bronze_dir, mock_spark):
    pytest.importorskip("pyarrow")  # tables reach Spark through Polars' to_arrow()
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_bronze_dir / "smn" / "ogd-smn_ber_d_historical.csv")

    ok, skip, failed = _ingest_collection(
        mock_spark, "smn", smn_bronze_dir / "smn", "`main`", "`meteoswiss`", historical=True
    )

    assert (ok, skip, failed) == (3, 0, 0)
    assert "`main`.`meteoswiss`.`smn_d_historical`" in _written_tables(mock_spark)


def test_ingest_collection_empty(tmp_path, mock_spark):
    empty_dir = tmp_path / "smn"
    empty_dir.mkdir()
    ok, skip, failed = _ingest_collection(mock_spark, "smn", empty_dir, "`main`", "`meteoswiss`")
    assert ok == 0
    assert skip == 1
    assert failed == 0


# ── _ingest_metadata ─────────────────────────────────────────────────────────


def test_ingest_metadata(smn_bronze_dir, mock_spark):
    pytest.importorskip("pyarrow")  # tables reach Spark through Polars' to_arrow()
    ok, skip, failed = _ingest_metadata(
        mock_spark,
        "smn",
        smn_bronze_dir / "smn",
        "`main`",
        "`meteoswiss`",
    )
    assert ok == 1  # ogd-smn_meta_parameters.csv
    assert skip == 0
    assert failed == 0
    mock_spark.createDataFrame.assert_called_once()


def test_ingest_metadata_multiple_files(smn_bronze_dir, mock_spark):
    """Stations + datainventory + parameters should each become their own table."""
    pytest.importorskip("pyarrow")  # tables reach Spark through Polars' to_arrow()
    smn_dir = smn_bronze_dir / "smn"
    (smn_dir / "ogd-smn_meta_stations.csv").write_text("station_abbr;station_name\nABO;Adelboden\nBER;Bern\n")
    (smn_dir / "ogd-smn_meta_datainventory.csv").write_text(
        "station_abbr;parameter_shortname;data_since;data_till\nABO;tre200d0;1900-01-01;2026-01-01\n"
    )

    ok, skip, failed = _ingest_metadata(mock_spark, "smn", smn_dir, "`main`", "`meteoswiss`")
    assert ok == 3
    assert skip == 0
    assert failed == 0

    # Each meta file should land in its own suffixed table.
    save_as = mock_spark.createDataFrame.return_value.write.mode.return_value.option.return_value.saveAsTable
    tables_written = [c.args[0] for c in save_as.call_args_list]
    assert any(t.endswith("`smn_meta_stations`") for t in tables_written)
    assert any(t.endswith("`smn_meta_parameters`") for t in tables_written)
    assert any(t.endswith("`smn_meta_datainventory`") for t in tables_written)


def test_ingest_metadata_no_files(tmp_path, mock_spark):
    empty_dir = tmp_path / "smn"
    empty_dir.mkdir()
    ok, skip, failed = _ingest_metadata(mock_spark, "smn", empty_dir, "`main`", "`meteoswiss`")
    assert ok == 0
    assert skip == 0
    assert failed == 0
    mock_spark.createDataFrame.assert_not_called()


# ── _ingest_climate_normals ──────────────────────────────────────────────────


def test_ingest_climate_normals(climate_normals_bronze_dir, mock_spark):
    _require_arrow()
    ok, skip, failed = _ingest_climate_normals(mock_spark, climate_normals_bronze_dir, "`main`", "`meteoswiss`")
    assert ok == 1
    assert skip == 0
    assert failed == 0
    mock_spark.createDataFrame.assert_called_once()


def test_ingest_climate_normals_no_dir(tmp_path, mock_spark):
    ok, skip, failed = _ingest_climate_normals(mock_spark, tmp_path, "`main`", "`meteoswiss`")
    assert ok == 0
    assert skip == 1
    assert failed == 0


def test_ingest_climate_normals_empty_dir(tmp_path, mock_spark):
    (tmp_path / "climate_normals").mkdir()
    ok, skip, failed = _ingest_climate_normals(mock_spark, tmp_path, "`main`", "`meteoswiss`")
    assert ok == 0
    assert skip == 1
    assert failed == 0


# ── _ingest_radar ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "product"),
    [("cpc2630112000_00060.001.h5", "CPC"), ("rzc263011200rl.001.h5", "RZC"), ("mzc263021145pl.850.h5", "MZC")],
)
def test_radar_product_reads_meteoswiss_lowercase_names(name, product):
    """MeteoSwiss names radar files in lower case; an uppercase-only pattern left every product empty."""
    found = re.match(RADAR_PRODUCT_PATTERN, name)
    assert found is not None
    assert found.group(0).upper() == product


def test_radar_merge_backfills_products_an_earlier_run_left_empty(tmp_path, mock_spark):
    h5_dir = tmp_path / "radar_precip"
    h5_dir.mkdir()
    (h5_dir / "cpc2630112000_00060.001.h5").write_bytes(b"x")

    _ingest_radar(mock_spark, tmp_path, "`main`", "`meteoswiss`", ("radar_precip",))

    select = mock_spark.read.format.return_value.option.return_value.load.return_value.selectExpr
    assert any(arg.startswith("upper(regexp_extract(") for arg in select.call_args.args)
    merge = next(c.args[0] for c in mock_spark.sql.call_args_list if c.args[0].startswith("MERGE"))
    assert "t.product IS DISTINCT FROM s.product" in merge
    assert "product = s.product" in merge


# ── INGESTED_COLLECTIONS ─────────────────────────────────────────────────────


def test_tabular_collections_excludes_binary():
    from foehn import registry

    for key in INGESTED_DATASETS:
        # climate_normals is a ZIP from opendata.swiss, not a STAC collection,
        # so it has no kind — everything else must be a tabular one.
        if key != "climate_normals":
            assert registry.spec(key).tabular


def test_ingested_collections_includes_climate_normals():
    """It used to be appended by hand; it is a row in the registry now."""
    assert "climate_normals" in INGESTED_DATASETS
    assert registry.non_grid_datasets() == INGESTED_DATASETS


# ── Failures are not skips ──────────────────────────────────────────────────


def test_a_failed_later_chunk_leaves_the_published_table_alone(smn_bronze_dir, mock_spark):
    """Overwriting with chunk 1 and appending the rest left a partial table on failure."""
    written: list[str] = []

    def write(spark, frame, table, mode="overwrite"):
        if written:
            raise RuntimeError("disk quota exceeded")
        written.append(table)

    with patch("scripts.ingest_delta._write_to_delta", side_effect=write):
        _, _, failed = _ingest_collection(
            mock_spark, "smn", smn_bronze_dir / "smn", "`main`", "`meteoswiss`", chunked=True, chunk_size=1
        )

    assert failed == 1  # the data group; the meta table was written before it
    assert written == ["`main`.`meteoswiss`.`smn_meta_parameters`"]
    mock_spark.table.return_value.write.mode.return_value.option.return_value.saveAsTable.assert_not_called()
    dropped = [call.args[0].removeprefix("DROP TABLE IF EXISTS ") for call in mock_spark.sql.call_args_list]
    assert any(_STAGING_RE.fullmatch(name) for name in dropped)


@pytest.mark.parametrize("size", ["0", "-1"])
def test_a_chunk_size_below_one_is_refused_before_spark(size, capsys):
    """Zero stopped range() mid-run; a negative size published a staging table it never wrote."""
    with (
        patch("sys.argv", ["ingest_delta.py", "--historical", "--chunk-size", size]),
        pytest.raises(SystemExit) as exited,
    ):
        main()

    assert exited.value.code == 2
    assert "must be at least 1" in capsys.readouterr().err


def test_a_write_failure_is_reported_with_its_reason(smn_bronze_dir, mock_spark, capsys):
    with patch("scripts.ingest_delta._write_to_delta", side_effect=RuntimeError("permission denied on table")):
        ok, skip, failed = _ingest_collection(mock_spark, "smn", smn_bronze_dir / "smn", "`main`", "`meteoswiss`")

    assert (ok, skip, failed) == (0, 0, 2)
    assert "RuntimeError: permission denied on table" in capsys.readouterr().out


def test_an_unreadable_normals_file_fails_the_table_instead_of_shrinking_it(climate_normals_bronze_dir, mock_spark):
    (climate_normals_bronze_dir / "climate_normals" / "broken.txt").write_bytes(b"\x00")

    with patch("scripts.ingest_delta.read_normals_txt", side_effect=[pl.DataFrame({"a": [1]}), ValueError("bad")]):
        ok, skip, failed = _ingest_climate_normals(mock_spark, climate_normals_bronze_dir, "`main`", "`meteoswiss`")

    assert (ok, skip, failed) == (0, 0, 1)
    mock_spark.createDataFrame.assert_not_called()


def test_main_exits_non_zero_when_a_table_fails(smn_bronze_dir, capsys):
    """The scheduled job reported success over stale tables: failures were "skipped"."""
    pytest.importorskip("pyspark")  # main() imports SparkSession once its arguments parse
    from foehn.workspace import Workspace

    workspace = Workspace(smn_bronze_dir / "volume")
    shutil.copytree(smn_bronze_dir / "smn", workspace.bronze("smn"))
    with (
        patch("pyspark.sql.SparkSession"),
        patch("scripts.ingest_delta.Workspace", return_value=workspace),
        patch("scripts.ingest_delta.INGESTED_DATASETS", ["smn"]),
        patch("scripts.ingest_delta._write_to_delta", side_effect=RuntimeError("write failed")),
        patch("sys.argv", ["ingest_delta.py"]),
        pytest.raises(SystemExit) as exited,
    ):
        main()

    assert exited.value.code == 1
    assert "0 tables written, 0 skipped, 2 failed" in capsys.readouterr().out


def test_a_polars_panic_fails_the_table_not_the_job(smn_bronze_dir, mock_spark):
    with patch("scripts.ingest_delta._scan_and_collect", side_effect=pl.exceptions.PanicException("boom")):
        _, _, failed = _ingest_collection(mock_spark, "smn", smn_bronze_dir / "smn", "`main`", "`meteoswiss`")

    assert failed == 1
