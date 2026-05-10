"""Tests for the foehn.ingest pipeline."""

import shutil

import polars as pl
import pytest
from conftest import FIXTURES_DIR

from foehn.ingest._grouping import (
    _build_schema_overrides,
    _group_csv_files,
    _table_suffix,
    _validate_identifier,
)
from foehn.ingest.adapters.memory import RecordingBinaryFileIndex, RecordingDeltaSink
from foehn.ingest.pipeline import (
    TABULAR_COLLECTIONS,
    _apply_column_comments,
    _ingest_climate_normals,
    _ingest_collection,
    _ingest_metadata,
    _scan_and_collect,
    run_radar,
    run_tabular,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def smn_bronze_dir(tmp_path):
    """Bronze dir with SMN CSVs and metadata in a 'smn' sub-folder."""
    smn_dir = tmp_path / "smn"
    smn_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_dir / "ogd-smn_abo_d_recent.csv")
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_dir / "ogd-smn_ber_d_recent.csv")
    shutil.copy(FIXTURES_DIR / "smn_meta_parameters.csv", smn_dir / "ogd-smn_meta_parameters.csv")
    return tmp_path


@pytest.fixture()
def climate_normals_bronze_dir(tmp_path):
    """Bronze dir with climate normals TXT files."""
    cn_dir = tmp_path / "climate_normals"
    cn_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "climate_normals_sample.txt", cn_dir / "sample.txt")
    return tmp_path


# ── _validate_identifier ─────────────────────────────────────────────────────


def test_validate_identifier_valid():
    assert _validate_identifier("main", "catalog") == "`main`"
    assert _validate_identifier("my-schema", "schema") == "`my-schema`"
    assert _validate_identifier("landing_01", "volume") == "`landing_01`"


def test_validate_identifier_rejects_injection():
    with pytest.raises(ValueError, match="Invalid"):
        _validate_identifier("main; DROP TABLE", "catalog")


# ── _group_csv_files ─────────────────────────────────────────────────────────


def test_group_csv_files(smn_bronze_dir):
    groups = _group_csv_files(smn_bronze_dir / "smn", "smn")
    assert ("d", "recent") in groups
    assert len(groups[("d", "recent")]) == 2


def test_group_csv_files_excludes_meta(smn_bronze_dir):
    groups = _group_csv_files(smn_bronze_dir / "smn", "smn")
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


def test_build_schema_overrides(smn_bronze_dir):
    files = sorted((smn_bronze_dir / "smn").glob("ogd-smn_*_d_recent.csv"))
    metadata_types = {"tre200d0": pl.Float64, "ure200d0": pl.Float64, "irrelevant": pl.Int64}
    overrides = _build_schema_overrides(files, metadata_types)
    assert overrides is not None
    assert "tre200d0" in overrides
    assert "irrelevant" not in overrides


def test_build_schema_overrides_no_metadata():
    assert _build_schema_overrides([], {}) is None


# ── _scan_and_collect ────────────────────────────────────────────────────────


def test_scan_and_collect(smn_bronze_dir):
    from foehn.convert import _load_metadata_types

    csv_dir = smn_bronze_dir / "smn"
    files = sorted(csv_dir.glob("ogd-smn_*_d_recent.csv"))
    metadata_types = _load_metadata_types(csv_dir)

    df = _scan_and_collect(files, metadata_types)

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 6  # 2 files × 3 rows
    assert "station_abbr" in df.columns
    assert "reference_timestamp" in df.columns
    assert df["tre200d0"].dtype == pl.Float64


def test_scan_and_collect_parses_timestamps(smn_bronze_dir):
    from foehn.convert import _load_metadata_types

    csv_dir = smn_bronze_dir / "smn"
    files = sorted(csv_dir.glob("ogd-smn_*_d_recent.csv"))
    metadata_types = _load_metadata_types(csv_dir)

    df = _scan_and_collect(files, metadata_types)
    assert df["reference_timestamp"].dtype == pl.Datetime


def test_scan_and_collect_single_file(smn_bronze_dir):
    from foehn.convert import _load_metadata_types

    csv_dir = smn_bronze_dir / "smn"
    files = [sorted(csv_dir.glob("ogd-smn_*_d_recent.csv"))[0]]
    metadata_types = _load_metadata_types(csv_dir)

    df = _scan_and_collect(files, metadata_types)
    assert len(df) == 3


# ── _apply_column_comments ───────────────────────────────────────────────────


def test_apply_column_comments_forwards_dict(smn_bronze_dir):
    sink = RecordingDeltaSink()
    _apply_column_comments(sink, "`cat`.`sch`.`smn_d_recent`", smn_bronze_dir / "smn")

    table_comments = sink.comments["`cat`.`sch`.`smn_d_recent`"]
    assert any("Param A en" in c and "[°C]" in c for c in table_comments.values())
    assert any("Param D en" in c and "[%]" in c for c in table_comments.values())


def test_apply_column_comments_no_metadata_is_noop(tmp_path):
    sink = RecordingDeltaSink()
    _apply_column_comments(sink, "`cat`.`sch`.`tbl`", tmp_path)
    assert sink.comments == {}


# ── _ingest_collection ───────────────────────────────────────────────────────


def test_ingest_collection_writes_data_and_meta(smn_bronze_dir):
    sink = RecordingDeltaSink()
    ok, skip = _ingest_collection(sink, "smn", smn_bronze_dir / "smn", "`main`", "`meteoswiss`")

    assert ok == 2  # 1 data group (d_recent) + 1 meta table (parameters)
    assert skip == 0
    assert "`main`.`meteoswiss`.`smn_d_recent`" in sink.tables
    assert "`main`.`meteoswiss`.`smn_meta_parameters`" in sink.tables
    data = sink.tables["`main`.`meteoswiss`.`smn_d_recent`"]
    assert data.height == 6  # 2 files × 3 rows
    assert "tre200d0" in sink.comments["`main`.`meteoswiss`.`smn_d_recent`"]


def test_ingest_collection_chunked_writes_overwrite_then_append(smn_bronze_dir):
    sink = RecordingDeltaSink()
    ok, skip = _ingest_collection(
        sink,
        "smn",
        smn_bronze_dir / "smn",
        "`main`",
        "`meteoswiss`",
        chunked=True,
        chunk_size=1,
    )
    assert ok == 2
    assert skip == 0

    data_calls = [c for c in sink.calls if c.table == "`main`.`meteoswiss`.`smn_d_recent`"]
    assert [c.mode for c in data_calls] == ["overwrite", "append"]
    # Final state has both files concatenated.
    assert sink.tables["`main`.`meteoswiss`.`smn_d_recent`"].height == 6


def test_ingest_collection_empty_dir_is_skipped(tmp_path):
    sink = RecordingDeltaSink()
    empty_dir = tmp_path / "smn"
    empty_dir.mkdir()
    ok, skip = _ingest_collection(sink, "smn", empty_dir, "`main`", "`meteoswiss`")
    assert ok == 0
    assert skip == 1
    assert sink.calls == []


# ── _ingest_metadata ─────────────────────────────────────────────────────────


def test_ingest_metadata_single_file(smn_bronze_dir):
    sink = RecordingDeltaSink()
    ok, skip = _ingest_metadata(sink, "smn", smn_bronze_dir / "smn", "`main`", "`meteoswiss`")

    assert ok == 1
    assert skip == 0
    assert "`main`.`meteoswiss`.`smn_meta_parameters`" in sink.tables


def test_ingest_metadata_multiple_files(smn_bronze_dir):
    smn_dir = smn_bronze_dir / "smn"
    (smn_dir / "ogd-smn_meta_stations.csv").write_text("station_abbr;station_name\nABO;Adelboden\nBER;Bern\n")
    (smn_dir / "ogd-smn_meta_datainventory.csv").write_text(
        "station_abbr;parameter_shortname;data_since;data_till\nABO;tre200d0;1900-01-01;2026-01-01\n"
    )

    sink = RecordingDeltaSink()
    ok, skip = _ingest_metadata(sink, "smn", smn_dir, "`main`", "`meteoswiss`")

    assert ok == 3
    assert skip == 0
    assert "`main`.`meteoswiss`.`smn_meta_stations`" in sink.tables
    assert "`main`.`meteoswiss`.`smn_meta_parameters`" in sink.tables
    assert "`main`.`meteoswiss`.`smn_meta_datainventory`" in sink.tables


def test_ingest_metadata_no_files(tmp_path):
    sink = RecordingDeltaSink()
    empty_dir = tmp_path / "smn"
    empty_dir.mkdir()
    ok, skip = _ingest_metadata(sink, "smn", empty_dir, "`main`", "`meteoswiss`")
    assert ok == 0
    assert skip == 0
    assert sink.calls == []


# ── _ingest_climate_normals ──────────────────────────────────────────────────


def test_ingest_climate_normals_writes_table(climate_normals_bronze_dir):
    sink = RecordingDeltaSink()
    ok, skip = _ingest_climate_normals(sink, climate_normals_bronze_dir, "`main`", "`meteoswiss`")
    assert ok == 1
    assert skip == 0
    assert "`main`.`meteoswiss`.`climate_normals`" in sink.tables


def test_ingest_climate_normals_no_dir_is_skipped(tmp_path):
    sink = RecordingDeltaSink()
    ok, skip = _ingest_climate_normals(sink, tmp_path, "`main`", "`meteoswiss`")
    assert ok == 0
    assert skip == 1
    assert sink.calls == []


def test_ingest_climate_normals_empty_dir_is_skipped(tmp_path):
    sink = RecordingDeltaSink()
    (tmp_path / "climate_normals").mkdir()
    ok, skip = _ingest_climate_normals(sink, tmp_path, "`main`", "`meteoswiss`")
    assert ok == 0
    assert skip == 1
    assert sink.calls == []


# ── run_tabular ──────────────────────────────────────────────────────────────


def test_run_tabular_single_dataset(smn_bronze_dir):
    sink = RecordingDeltaSink()
    ok, skip = run_tabular(smn_bronze_dir, "`main`", "`meteoswiss`", sink, keys=["smn"])

    assert ok == 2  # data + meta
    assert skip == 0
    assert "`main`.`meteoswiss`.`smn_d_recent`" in sink.tables


def test_run_tabular_climate_normals(climate_normals_bronze_dir):
    sink = RecordingDeltaSink()
    ok, skip = run_tabular(climate_normals_bronze_dir, "`main`", "`meteoswiss`", sink, keys=["climate_normals"])
    assert ok == 1
    assert skip == 0
    assert "`main`.`meteoswiss`.`climate_normals`" in sink.tables


def test_run_tabular_missing_dataset_dir_is_skipped(tmp_path):
    sink = RecordingDeltaSink()
    ok, skip = run_tabular(tmp_path, "`main`", "`meteoswiss`", sink, keys=["smn"])
    assert ok == 0
    assert skip == 1
    assert sink.calls == []


# ── run_radar ────────────────────────────────────────────────────────────────


def test_run_radar_indexes_present_datasets(tmp_path):
    radar_dir = tmp_path / "radar_precip"
    radar_dir.mkdir()
    (radar_dir / "RZC0001.h5").write_bytes(b"")
    (radar_dir / "RZC0002.h5").write_bytes(b"")

    index = RecordingBinaryFileIndex()
    ok, skip = run_radar(tmp_path, "`main`", "`meteoswiss`", index, keys=["radar_precip", "radar_hail"])

    assert ok == 1
    assert skip == 1  # radar_hail is missing
    assert index.merges == [(radar_dir, "`main`.`meteoswiss`.`radar_precip`")]


def test_run_radar_all_missing_is_all_skipped(tmp_path):
    index = RecordingBinaryFileIndex()
    ok, skip = run_radar(tmp_path, "`main`", "`meteoswiss`", index)
    assert ok == 0
    assert skip == 2
    assert index.merges == []


# ── TABULAR_COLLECTIONS ──────────────────────────────────────────────────────


def test_tabular_collections_excludes_binary():
    from foehn.collections import GRIB2_COLLECTIONS, NETCDF_COLLECTIONS

    for key in TABULAR_COLLECTIONS:
        assert key not in GRIB2_COLLECTIONS
        assert key not in NETCDF_COLLECTIONS


def test_tabular_collections_includes_climate_normals():
    assert "climate_normals" in TABULAR_COLLECTIONS
