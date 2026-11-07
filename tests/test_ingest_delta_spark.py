"""The Delta ingest against a real Spark session, not a mock.

Every other test of ``scripts/ingest_delta.py`` mocks ``SparkSession``, so the SQL
it builds by hand — the radar ``MERGE``, ``ALTER COLUMN … COMMENT``, the staged
publish of a chunked group — and the Arrow handoff from Polars had only ever run
on a Databricks cluster. The handoff is the one that bit: ``createDataFrame``
takes a ``pyarrow.Table`` only from PySpark 4.0, while the ``databricks`` extra
allowed 3.5.

CI runs this module in its own job, with Java and the ``databricks`` extra
installed. Anywhere PySpark is missing, it skips.
"""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("pyspark", reason="pyspark not installed (the databricks extra)")
pytest.importorskip("delta", reason="delta-spark not installed (the databricks extra)")

# Imported only once pyspark is known to be there.
from foehn.meteocsv import read_normals_txt
from foehn.retention import RETIREMENT_GRACE, record_listing
from scripts.ingest_delta import (
    _ingest_climate_normals,
    _ingest_collection,
    _ingest_radar,
    _write_to_delta,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CATALOG = "`spark_catalog`"
SCHEMA = "`meteoswiss`"


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.master("local[1]")
        .appName("foehn-tests")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.warehouse.dir", str(tmp_path_factory.mktemp("warehouse")))
        # Databricks writes Delta unless told otherwise; a local session writes Parquet.
        .config("spark.sql.sources.default", "delta")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
    )
    session = configure_spark_with_delta_pip(builder).getOrCreate()
    session.sql(f"CREATE DATABASE IF NOT EXISTS {SCHEMA}")
    yield session
    session.stop()


def _table(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.`{name}`"


def _smn_bronze(tmp_path: Path, slice_name: str) -> Path:
    csv_dir = tmp_path / "smn"
    csv_dir.mkdir()
    for station in ("abo", "ber"):
        shutil.copy(FIXTURES_DIR / "smn_sample.csv", csv_dir / f"ogd-smn_{station}_d_{slice_name}.csv")
    shutil.copy(FIXTURES_DIR / "smn_meta_parameters.csv", csv_dir / "ogd-smn_meta_parameters.csv")
    return csv_dir


def test_a_collection_lands_as_a_commented_delta_table(spark, tmp_path):
    csv_dir = _smn_bronze(tmp_path, "recent")

    written, _, failed = _ingest_collection(spark, "smn", csv_dir, CATALOG, SCHEMA)

    assert (written, failed) == (2, 0)  # the data group and the metadata table
    table = _table("smn_d_recent")
    assert spark.table(table).count() == 6  # three rows from each of two station files
    assert spark.sql(f"DESCRIBE DETAIL {table}").first()["format"] == "delta"
    comments = {row["col_name"]: row["comment"] for row in spark.sql(f"DESCRIBE TABLE {table}").collect()}
    assert comments["tre200d0"] == "Param A en [°C]"
    assert spark.table(_table("smn_meta_parameters")).count() == 6  # one row per parameter in the fixture


def test_a_chunked_group_is_published_whole_and_its_staging_table_dropped(spark, tmp_path):
    csv_dir = _smn_bronze(tmp_path, "historical")

    written, _, failed = _ingest_collection(
        spark, "smn", csv_dir, CATALOG, SCHEMA, historical=True, chunked=True, chunk_size=1
    )

    assert failed == 0
    assert written == 2
    assert spark.table(_table("smn_d_historical")).count() == 6
    tables = [row["tableName"] for row in spark.sql(f"SHOW TABLES IN {SCHEMA}").collect()]
    assert not [name for name in tables if "foehn_staging" in name]


def test_retired_rolling_group_is_emptied_without_losing_schema_or_historical_data(spark, tmp_path):
    csv_dir = _smn_bronze(tmp_path, "recent")
    for name in ("ogd-smn_ber_h_recent.csv", "ogd-smn_ber_d_historical.csv"):
        shutil.copy(FIXTURES_DIR / "smn_sample.csv", csv_dir / name)
    initial = _ingest_collection(spark, "smn", csv_dir, CATALOG, SCHEMA, historical=True)
    assert initial.failed == 0
    retired_table = _table("smn_d_recent")
    before = spark.table(retired_table).schema
    assert spark.table(retired_table).count() == 6

    for path in csv_dir.glob("*_d_recent.csv"):
        path.unlink()
    # Simulate the authority recorded by successful refreshes a grace period apart.
    names = [path.name for path in csv_dir.glob("*.csv")]
    start = datetime(2026, 1, 1, tzinfo=UTC)
    record_listing(csv_dir, names, now=start)
    record_listing(csv_dir, names, now=start + RETIREMENT_GRACE)

    for _ in range(2):
        refreshed = _ingest_collection(spark, "smn", csv_dir, CATALOG, SCHEMA)
        assert refreshed.failed == 0
        assert spark.table(retired_table).count() == 0
        assert spark.table(retired_table).schema == before  # includes column comments
        assert spark.table(_table("smn_h_recent")).count() == 3
        assert spark.table(_table("smn_d_historical")).count() == 3


def test_radar_files_are_merged_by_path_and_reindexing_adds_nothing(spark, tmp_path):
    radar = tmp_path / "radar_precip"
    radar.mkdir()
    for name in ("cpc2630112000_00060.001.h5", "cpc2630112500_00060.001.h5"):
        (radar / name).write_bytes(b"\x89HDF\r\n\x1a\n")

    for _ in range(2):
        written, _, failed = _ingest_radar(spark, tmp_path, CATALOG, SCHEMA, ("radar_precip",))
        assert (written, failed) == (1, 0)

    rows = spark.table(_table("radar_precip")).collect()
    assert len(rows) == 2
    assert {row["product"] for row in rows} == {"CPC"}


def test_climate_normals_rebuild_preserves_identity_without_translation_duplicates(spark, tmp_path):
    txt_dir = tmp_path / "climate_normals"
    txt_dir.mkdir()
    wind = txt_dir / "climate-reports-normtables_fkl010m0_1991-2020_en.txt"
    shutil.copy(FIXTURES_DIR / "climate_normals_sample.txt", wind)
    temperature = wind.read_text(encoding="latin1").replace(
        "Wind speed scalar / meter/second", "Air temperature / degree Celsius"
    )
    (txt_dir / "climate-reports-normtables_tre200m0_1991-2020_en.txt").write_text(temperature, encoding="latin1")
    shutil.copy(wind, txt_dir / "climate-reports-normtables_fkl010m0_1991-2020_it.txt")
    # A pre-fix table lacks identity; an overwrite must migrate that schema too.
    _write_to_delta(spark, read_normals_txt(wind), _table("climate_normals"))

    written, _, failed = _ingest_climate_normals(spark, tmp_path, CATALOG, SCHEMA)

    assert (written, failed) == (1, 0)
    table = spark.table(_table("climate_normals"))
    assert table.count() == 4
    assert {row["parameter"] for row in table.select("parameter").distinct().collect()} == {"fkl010m0", "tre200m0"}
    assert table.select("Station", "parameter", "normal_period").distinct().count() == 4
    assert table.filter("Station IS NULL OR unit IS NULL OR source_file IS NULL").count() == 0
    # The names the Parquet path publishes, kept — not renamed to suit Delta.
    assert "Altitude [m asl]" in table.columns
