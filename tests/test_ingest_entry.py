"""Tests for foehn.ingest.ingest — the public entry point with auto-derivation."""

from __future__ import annotations

import shutil

import pytest
from conftest import FIXTURES_DIR

from foehn.ingest import ingest
from foehn.ingest.adapters.memory import RecordingBinaryFileIndex, RecordingDeltaSink
from foehn.ingest.entry import _detect_historical, _discover_datasets

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def smn_bronze_dir(tmp_path):
    smn = tmp_path / "smn"
    smn.mkdir()
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn / "ogd-smn_abo_d_recent.csv")
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn / "ogd-smn_ber_d_recent.csv")
    shutil.copy(FIXTURES_DIR / "smn_meta_parameters.csv", smn / "ogd-smn_meta_parameters.csv")
    return tmp_path


@pytest.fixture()
def radar_bronze_dir(tmp_path):
    radar = tmp_path / "radar_precip"
    radar.mkdir()
    (radar / "RZC0001.h5").write_bytes(b"")
    (radar / "RZC0002.h5").write_bytes(b"")
    return tmp_path


@pytest.fixture()
def mixed_bronze_dir(smn_bronze_dir, radar_bronze_dir):
    """smn + radar_precip under the same bronze base (tmp_path is shared)."""
    assert smn_bronze_dir == radar_bronze_dir
    return smn_bronze_dir


# ── _discover_datasets ───────────────────────────────────────────────────────


def test_discover_datasets_lists_tabular_only(tmp_path):
    (tmp_path / "smn").mkdir()
    (tmp_path / "radar_precip").mkdir()  # excluded — radar is opt-in
    (tmp_path / "junk").mkdir()  # not a known dataset
    (tmp_path / "stray.csv").write_text("")  # not a directory

    assert _discover_datasets(tmp_path) == ["smn"]


def test_discover_datasets_missing_base_returns_empty(tmp_path):
    assert _discover_datasets(tmp_path / "nope") == []


# ── _detect_historical ───────────────────────────────────────────────────────


def test_detect_historical_true_for_historical_files(tmp_path):
    smn = tmp_path / "smn"
    smn.mkdir()
    (smn / "ogd-smn_abo_h_historical.csv").write_text("")
    assert _detect_historical(tmp_path, ["smn"]) is True


def test_detect_historical_false_for_recent_only(tmp_path):
    smn = tmp_path / "smn"
    smn.mkdir()
    (smn / "ogd-smn_abo_d_recent.csv").write_text("")
    assert _detect_historical(tmp_path, ["smn"]) is False


def test_detect_historical_false_for_missing_dataset(tmp_path):
    assert _detect_historical(tmp_path, ["smn"]) is False


# ── ingest — basics ──────────────────────────────────────────────────────────


def test_ingest_dispatches_tabular_through_sink(smn_bronze_dir):
    sink = RecordingDeltaSink()
    index = RecordingBinaryFileIndex()
    ok, skip = ingest(bronze=smn_bronze_dir, datasets=["smn"], sink=sink, index=index)

    assert ok == 2  # data + meta
    assert skip == 0
    assert "`main`.`meteoswiss`.`smn_d_recent`" in sink.tables
    assert index.merges == []


def test_ingest_dispatches_radar_through_index(radar_bronze_dir):
    sink = RecordingDeltaSink()
    index = RecordingBinaryFileIndex()
    ok, skip = ingest(bronze=radar_bronze_dir, datasets=["radar_precip"], sink=sink, index=index)

    assert ok == 1
    assert skip == 0
    assert sink.calls == []
    assert index.merges == [(radar_bronze_dir / "radar_precip", "`main`.`meteoswiss`.`radar_precip`")]


def test_ingest_routes_mixed_keys_separately(mixed_bronze_dir):
    sink = RecordingDeltaSink()
    index = RecordingBinaryFileIndex()
    ok, skip = ingest(
        bronze=mixed_bronze_dir,
        datasets=["smn", "radar_precip"],
        sink=sink,
        index=index,
    )

    assert ok == 3  # smn data + smn meta + radar_precip
    assert skip == 0
    assert "`main`.`meteoswiss`.`smn_d_recent`" in sink.tables
    assert any(t == "`main`.`meteoswiss`.`radar_precip`" for _, t in index.merges)


# ── ingest — auto-discovery ─────────────────────────────────────────────────


def test_ingest_auto_discovers_tabular_only(mixed_bronze_dir):
    """Auto-discovery skips radar — daily ingest stays cheap. Radar must be explicit."""
    sink = RecordingDeltaSink()
    index = RecordingBinaryFileIndex()
    ok, skip = ingest(bronze=mixed_bronze_dir, sink=sink, index=index)

    assert ok == 2  # smn data + smn meta only
    assert skip == 0
    assert "`main`.`meteoswiss`.`smn_d_recent`" in sink.tables
    assert index.merges == []


def test_ingest_empty_bronze_is_noop(tmp_path):
    sink = RecordingDeltaSink()
    index = RecordingBinaryFileIndex()
    ok, skip = ingest(bronze=tmp_path, sink=sink, index=index)

    assert (ok, skip) == (0, 0)
    assert sink.calls == []
    assert index.merges == []


# ── ingest — auto-historical ─────────────────────────────────────────────────


def test_ingest_auto_detects_historical_from_disk(tmp_path, monkeypatch):
    """When *_historical*.csv files are present, run_tabular receives chunked=True."""
    smn = tmp_path / "smn"
    smn.mkdir()
    (smn / "ogd-smn_abo_h_historical.csv").write_text("a;b\n1;2\n")

    received: list[bool] = []

    def fake_run_tabular(bronze_base, cat, sch, sink_, *, keys, chunked, chunk_size):
        received.append(chunked)
        return 0, 0

    monkeypatch.setattr("foehn.ingest.entry.run_tabular", fake_run_tabular)

    sink = RecordingDeltaSink()
    index = RecordingBinaryFileIndex()
    ingest(bronze=tmp_path, datasets=["smn"], sink=sink, index=index)

    assert received == [True]


def test_ingest_recent_only_does_not_trigger_chunking(smn_bronze_dir, monkeypatch):
    received: list[bool] = []

    def fake_run_tabular(bronze_base, cat, sch, sink_, *, keys, chunked, chunk_size):
        received.append(chunked)
        return 0, 0

    monkeypatch.setattr("foehn.ingest.entry.run_tabular", fake_run_tabular)

    sink = RecordingDeltaSink()
    index = RecordingBinaryFileIndex()
    ingest(bronze=smn_bronze_dir, datasets=["smn"], sink=sink, index=index)

    assert received == [False]


def test_ingest_explicit_historical_overrides_auto_detection(tmp_path, monkeypatch):
    """historical=False wins even if historical files exist on disk."""
    smn = tmp_path / "smn"
    smn.mkdir()
    (smn / "ogd-smn_abo_h_historical.csv").write_text("a;b\n1;2\n")

    received: list[bool] = []

    def fake_run_tabular(bronze_base, cat, sch, sink_, *, keys, chunked, chunk_size):
        received.append(chunked)
        return 0, 0

    monkeypatch.setattr("foehn.ingest.entry.run_tabular", fake_run_tabular)

    ingest(
        bronze=tmp_path,
        datasets=["smn"],
        historical=False,
        sink=RecordingDeltaSink(),
        index=RecordingBinaryFileIndex(),
    )

    assert received == [False]


# ── ingest — validation ─────────────────────────────────────────────────────


def test_ingest_unknown_dataset_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        ingest(
            bronze=tmp_path,
            datasets=["bogus"],
            sink=RecordingDeltaSink(),
            index=RecordingBinaryFileIndex(),
        )


def test_ingest_invalid_catalog_identifier_raises(tmp_path):
    with pytest.raises(ValueError, match="Invalid catalog"):
        ingest(
            bronze=tmp_path,
            catalog="evil; DROP TABLE",
            sink=RecordingDeltaSink(),
            index=RecordingBinaryFileIndex(),
        )


# ── ingest — SMN historical scale ────────────────────────────────────────────


def test_smn_historical_auto_chunks_via_byte_threshold(tmp_path, monkeypatch):
    """End-to-end mirror of the foehn_historical job for SMN.

    Many station files in the ``(t, historical)`` group, ``*_historical*.csv``
    naming, no explicit ``historical=`` kwarg — auto-detection sets
    ``chunked=True`` and the byte-size threshold gates the chunked write.
    Without chunking the production job would do a single ~20 GB
    Arrow→Spark transfer and OOM the worker.
    """
    smn = tmp_path / "smn"
    smn.mkdir()
    shutil.copy(FIXTURES_DIR / "smn_meta_parameters.csv", smn / "ogd-smn_meta_parameters.csv")

    # 4 station files in the (t, historical) group — analogue of SMN's ~250
    # station files at 10-min cadence. Reuses the existing 3-row fixture.
    stations = ["abo", "ber", "luz", "zur"]
    for sta in stations:
        shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn / f"ogd-smn_{sta}_t_historical.csv")

    # Drop the threshold so the small test fixture triggers chunking.
    monkeypatch.setattr("foehn.ingest.pipeline.LARGE_THRESHOLD_BYTES", 1)

    sink = RecordingDeltaSink()
    index = RecordingBinaryFileIndex()

    ok, skip = ingest(
        bronze=tmp_path,
        chunk_size=2,  # 4 station files / 2 per chunk = 2 chunks → overwrite + append
        sink=sink,
        index=index,
    )

    # 1 data table (t, historical) + 1 meta table (parameters).
    assert ok == 2
    assert skip == 0

    # Auto-detected historical=True via *_historical*.csv glob, threshold met,
    # 4 > chunk_size → chunked write fires.
    table = "`main`.`meteoswiss`.`smn_t_historical`"
    data_calls = [c for c in sink.calls if c.table == table]
    assert [c.mode for c in data_calls] == ["overwrite", "append"]

    # Final durable state: 4 stations × 3 rows per fixture = 12 rows.
    assert sink.tables[table].height == 12


def test_ingest_default_bronze_path_uses_volume_layout(tmp_path, monkeypatch):
    """When bronze is None, the default points at /Volumes/{catalog}/{schema}/{volume}/bronze."""
    seen: list = []

    def fake_run_tabular(bronze_base, cat, sch, sink_, *, keys, chunked, chunk_size):
        seen.append(bronze_base)
        return 0, 0

    def fake_run_radar(bronze_base, cat, sch, index_, *, keys):
        seen.append(bronze_base)
        return 0, 0

    monkeypatch.setattr("foehn.ingest.entry.run_tabular", fake_run_tabular)
    monkeypatch.setattr("foehn.ingest.entry.run_radar", fake_run_radar)

    ingest(
        catalog="cat1",
        schema="sch1",
        volume="vol1",
        datasets=["smn"],
        sink=RecordingDeltaSink(),
        index=RecordingBinaryFileIndex(),
    )

    from pathlib import Path

    assert seen == [Path("/Volumes/cat1/sch1/vol1/bronze")]
