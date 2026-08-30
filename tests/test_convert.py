"""Tests for CSV/TXT → Parquet conversion."""

import shutil
from pathlib import Path

import polars as pl
import pytest

from foehn.convert import (
    add_forecast_local_timestamp,
    convert_indoor_to_parquet,
    convert_normals_to_parquet,
    convert_preamble_to_parquet,
    convert_to_parquet,
    decode_meteoswiss_csv,
    group_csv_files,
    load_metadata_types,
    parse_climate_scenarios_csv,
    parse_csv_bytes,
    parse_metadata_types,
    scan_climate_scenarios_csv,
)
from foehn.workspace import Workspace


def test_decode_meteoswiss_csv_handles_utf8_and_windows1252():
    assert decode_meteoswiss_csv("café".encode()) == "café"
    assert decode_meteoswiss_csv("café".encode("utf-8-sig")) == "café"  # BOM stripped
    assert decode_meteoswiss_csv(b"caf\xe9") == "café"  # Windows-1252 fallback


def test_decode_meteoswiss_csv_is_total():
    # 0x81 is invalid UTF-8 *and* unmapped in cp1252 — must replace, not raise.
    decoded = decode_meteoswiss_csv(b"col\n\x81\n")
    assert "col" in decoded


def test_parse_climate_scenarios_csv_rejects_short_filename():
    content = b"DATE;MODEL_A\n0001-01-01;1.0\n"
    with pytest.raises(ValueError, match="Unexpected climate-scenario filename"):
        parse_climate_scenarios_csv(content, "weird.csv")


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def smn_workspace(tmp_path):
    """A smn_workspace whose bronze holds SMN CSVs and metadata."""
    smn_dir = Workspace(tmp_path).bronze("smn")
    smn_dir.mkdir(parents=True)
    # Two station files in the same group (d_recent) to test combining.
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_dir / "ogd-smn_abo_d_recent.csv")
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_dir / "ogd-smn_ber_d_recent.csv")
    shutil.copy(
        FIXTURES_DIR / "smn_meta_parameters.csv",
        smn_dir / "ogd-smn_meta_parameters.csv",
    )
    return Workspace(tmp_path)


@pytest.fixture()
def climate_normals_workspace(tmp_path):
    """A climate_normals_workspace whose bronze holds a single climate normals TXT."""
    cn_dir = Workspace(tmp_path).bronze("climate_normals")
    cn_dir.mkdir(parents=True)
    shutil.copy(FIXTURES_DIR / "climate_normals_sample.txt", cn_dir / "sample.txt")
    return Workspace(tmp_path)


# --- group_csv_files ---


def test_group_csv_files(smn_workspace):
    groups = group_csv_files(smn_workspace.bronze("smn"), "smn")
    assert ("d", "recent") in groups
    assert len(groups[("d", "recent")]) == 2


def test_group_csv_files_excludes_meta(smn_workspace):
    groups = group_csv_files(smn_workspace.bronze("smn"), "smn")
    all_files = [f for files in groups.values() for f in files]
    assert all("_meta_" not in f.name for f in all_files)


def test_group_csv_files_empty(tmp_path):
    empty_dir = tmp_path / "smn"
    empty_dir.mkdir()
    assert group_csv_files(empty_dir, "smn") == {}


def test_group_csv_files_decade_split_historical_share_one_group(tmp_path):
    """Per-decade historical chunks belong to the same (frequency, slice) group."""
    d = tmp_path / "smn"
    d.mkdir()
    for name in ("ogd-smn_ber_t_historical_2000-2009.csv", "ogd-smn_ber_t_historical_2010-2019.csv"):
        (d / name).write_text("a;b\n1;2\n")

    groups = group_csv_files(d, "smn")
    assert list(groups) == [("t", "historical")]
    assert len(groups[("t", "historical")]) == 2


def test_group_csv_files_no_granularity_collection_is_one_group(tmp_path):
    """forecast_local filenames don't carry the collection prefix at all.

    They are vnut12.lssw.<run>.<param>.csv, so the prefix-stripping path would
    slice arbitrary characters off the stem — the collection collapses to a
    single unkeyed group instead.
    """
    d = tmp_path / "forecast_local"
    d.mkdir()
    for name in ("vnut12.lssw.202607210600.dkl010h0.csv", "vnut12.lssw.202607210600.tre200h0.csv"):
        (d / name).write_text("a;b\n1;2\n")

    groups = group_csv_files(d, "forecast_local")
    assert list(groups) == [()]
    assert len(groups[()]) == 2


# --- convert_to_parquet ---


def test_convert_to_parquet_creates_combined_file(smn_workspace, tmp_path):
    parquet_dir = smn_workspace.parquet()
    convert_to_parquet("smn", smn_workspace)

    out = parquet_dir / "smn" / "smn_d_recent.parquet"
    assert out.exists()


def test_convert_to_parquet_combines_stations(smn_workspace, tmp_path):
    """Two per-station CSVs in the same group should be combined into one Parquet."""
    parquet_dir = smn_workspace.parquet()
    convert_to_parquet("smn", smn_workspace)

    df = pl.read_parquet(parquet_dir / "smn" / "smn_d_recent.parquet")
    assert "station_abbr" in df.columns
    # 2 files × 3 rows each = 6 rows
    assert len(df) == 6


def test_convert_to_parquet_skips_up_to_date(smn_workspace, tmp_path):
    """If Parquet is already newer than all CSVs, the group should not be re-written."""
    parquet_dir = smn_workspace.parquet()
    convert_to_parquet("smn", smn_workspace)

    out = parquet_dir / "smn" / "smn_d_recent.parquet"
    mtime_before = out.stat().st_mtime

    convert_to_parquet("smn", smn_workspace)
    assert out.stat().st_mtime == mtime_before


def test_convert_to_parquet_failed_write_leaves_no_output(smn_workspace, tmp_path, monkeypatch):
    """A sink that dies part-way must leave nothing at the final path.

    sink_parquet creates the file before it can fail (disk full, source read
    error). Left at the final path, that truncated file carries a fresh mtime —
    so the up-to-date check would skip it forever and the next run would report
    success over a corrupt Parquet.
    """
    parquet_dir = smn_workspace.parquet()

    def flaky_sink(self, path, **kwargs):
        Path(path).write_bytes(b"PAR1\x00\x00truncated")
        raise OSError("No space left on device")

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", flaky_sink)
    assert convert_to_parquet("smn", smn_workspace) == 1

    out_dir = parquet_dir / "smn"
    assert not (out_dir / "smn_d_recent.parquet").exists()
    assert list(out_dir.glob("*.tmp")) == []


def test_convert_to_parquet_retries_after_failed_write(smn_workspace, tmp_path, monkeypatch):
    """The group is retried on the next run rather than skipped as up-to-date."""
    parquet_dir = smn_workspace.parquet()

    def flaky_sink(self, path, **kwargs):
        Path(path).write_bytes(b"PAR1\x00\x00truncated")
        raise OSError("No space left on device")

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", flaky_sink)
    assert convert_to_parquet("smn", smn_workspace) == 1

    monkeypatch.undo()
    assert convert_to_parquet("smn", smn_workspace) == 0
    assert len(pl.read_parquet(parquet_dir / "smn" / "smn_d_recent.parquet")) == 6


def test_convert_to_parquet_no_csv_is_noop(tmp_path):
    """Empty bronze dir should not raise and should produce no output."""
    workspace = Workspace(tmp_path)
    bronze_dir = workspace.bronze()
    (bronze_dir / "smn").mkdir(parents=True)
    parquet_dir = workspace.parquet()

    convert_to_parquet("smn", workspace)

    assert not (parquet_dir / "smn").exists() or not list((parquet_dir / "smn").iterdir())


# --- load_metadata_types ---


def test_load_metadata_types(smn_workspace):
    """Metadata file should produce a mapping of parameter names to Polars dtypes."""
    types = load_metadata_types(smn_workspace.bronze("smn"))
    assert types["tre200d0"] == pl.Float64
    assert types["rre150d0"] == pl.Float64
    assert types["ure200d0"] == pl.Int64
    assert types["sre000d0"] == pl.Int64
    assert "station_abbr" not in types


def test_load_metadata_types_no_file(tmp_path):
    """Returns empty dict when no metadata file exists."""
    assert load_metadata_types(tmp_path) == {}


def test_convert_applies_metadata_types(smn_workspace, tmp_path):
    """Columns listed as Float in metadata should be Float64 in the Parquet output."""
    parquet_dir = smn_workspace.parquet()
    convert_to_parquet("smn", smn_workspace)

    df = pl.read_parquet(parquet_dir / "smn" / "smn_d_recent.parquet")
    assert df["tre200d0"].dtype == pl.Float64
    assert df["rre150d0"].dtype == pl.Float64


def test_convert_skips_meta_files(smn_workspace, tmp_path):
    """Metadata CSVs should not be converted to Parquet."""
    parquet_dir = smn_workspace.parquet()
    convert_to_parquet("smn", smn_workspace)

    out_dir = parquet_dir / "smn"
    parquet_files = list(out_dir.glob("*.parquet"))
    assert all("_meta_" not in f.name for f in parquet_files)


# --- convert_normals_to_parquet ---


def test_convert_normals_creates_file(climate_normals_workspace, tmp_path):
    parquet_dir = climate_normals_workspace.parquet()
    convert_normals_to_parquet("climate_normals", climate_normals_workspace)

    out = parquet_dir / "climate_normals" / "sample.parquet"
    assert out.exists()


def test_convert_normals_readable(climate_normals_workspace, tmp_path):
    parquet_dir = climate_normals_workspace.parquet()
    convert_normals_to_parquet("climate_normals", climate_normals_workspace)

    df = pl.read_parquet(parquet_dir / "climate_normals" / "sample.parquet")
    assert "Station" in df.columns
    assert "Jan" in df.columns
    assert len(df) == 2


# --- parse_metadata_types edge cases ---


def test_parse_metadata_types_invalid_content():
    """Unparseable content returns empty dict."""
    assert parse_metadata_types(b"not;valid;csv\x00\xff") == {}


def test_parse_metadata_types_missing_columns():
    """CSV without expected columns returns empty dict."""
    assert parse_metadata_types(b"col_a;col_b\nfoo;bar\n") == {}


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


def test_parse_csv_bytes_unwidenable_column_raises_instead_of_looping():
    """Regression: a column that fails even as Float64 must raise, not retry forever.

    The value sits past the 100-row inference window, so the column is inferred
    Int64, the parse fails, the Float64 override is applied — and fails again.
    Without the already-Float64 guard this re-parsed the same bytes infinitely.
    """
    rows = ["station_abbr;val"] + [f"BER;{i}" for i in range(150)]
    rows[120] = "BER;abc"
    content = "\n".join(rows).encode()

    with pytest.raises(pl.exceptions.ComputeError):
        parse_csv_bytes(content)


# --- convert_to_parquet error handling ---


def test_convert_to_parquet_handles_bad_csv(tmp_path):
    """A corrupt CSV should be reported as a failure, not crash the whole conversion."""
    workspace = Workspace(tmp_path)
    bronze_dir = workspace.bronze()
    csv_dir = bronze_dir / "smn"
    csv_dir.mkdir(parents=True)
    (csv_dir / "ogd-smn_ber_d_recent.csv").write_text("a;b\n1;2\n")
    (csv_dir / "ogd-smn_zur_d_recent.csv").write_bytes(b"")

    failed = convert_to_parquet("smn", workspace)

    # The bad group is counted as a failure; the call returns rather than raising.
    assert failed >= 1


# --- convert_normals edge cases ---


def test_convert_normals_no_txt_files(tmp_path):
    """Empty climate_normals dir should not raise."""
    workspace = Workspace(tmp_path)
    bronze_dir = workspace.bronze()
    (bronze_dir / "climate_normals").mkdir(parents=True)
    parquet_dir = workspace.parquet()
    convert_normals_to_parquet("climate_normals", workspace)
    # Dir may be created but should contain no parquet files
    cn_dir = parquet_dir / "climate_normals"
    assert not cn_dir.exists() or not list(cn_dir.glob("*.parquet"))


def test_convert_normals_skips_up_to_date(climate_normals_workspace, tmp_path):
    """Already-converted files should be skipped on second run."""
    parquet_dir = climate_normals_workspace.parquet()
    convert_normals_to_parquet("climate_normals", climate_normals_workspace)

    out_file = parquet_dir / "climate_normals" / "sample.parquet"
    mtime_before = out_file.stat().st_mtime

    convert_normals_to_parquet("climate_normals", climate_normals_workspace)
    assert out_file.stat().st_mtime == mtime_before


def test_convert_normals_handles_bad_file(tmp_path):
    """A corrupt TXT should be reported as a failure, not crash."""
    workspace = Workspace(tmp_path)
    cn_dir = workspace.bronze("climate_normals")
    cn_dir.mkdir(parents=True)
    (cn_dir / "bad.txt").write_bytes(b"\x00\xff")

    failed = convert_normals_to_parquet("climate_normals", workspace)

    assert failed >= 1


# --- indoor climate scenarios (zipped multi-CSV) ---

_INDOOR_CSV = (
    "time.yy,time.mm,time.dd,time.hh,tre200h0,ure200h0,skycover\n"
    "2035,1,1,0,0.3,94.5,36\n"
    "2035,1,1,1,-0.2,94.6,26\n"
    "2035,1,1,2,-0.5,94.3,20\n"
)


def test_convert_indoor_scenarios_parses_filename_and_timestamp(tmp_path):
    workspace = Workspace(tmp_path)
    bronze_dir = workspace.bronze()
    indoor_dir = bronze_dir / "climate_scenarios_indoor"
    indoor_dir.mkdir(parents=True)
    (indoor_dir / "ABO_2035_RCP85_1in10-warmsummer.csv").write_text(_INDOOR_CSV)
    (indoor_dir / "AIG_2060_RCP26_DRY.csv").write_text(_INDOOR_CSV)

    parquet_dir = workspace.parquet()
    failed = convert_indoor_to_parquet("climate_scenarios_indoor", workspace)
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
    assert convert_indoor_to_parquet("climate_scenarios_indoor", Workspace(tmp_path)) == 0


def test_convert_indoor_scenarios_skips_metadata_file(tmp_path):
    """The archive's non-data metadata CSV must be skipped, not counted as a failure."""
    workspace = Workspace(tmp_path)
    bronze_dir = workspace.bronze()
    indoor_dir = bronze_dir / "climate_scenarios_indoor"
    indoor_dir.mkdir(parents=True)
    (indoor_dir / "ABO_2035_RCP85_DRY.csv").write_text(_INDOOR_CSV)
    (indoor_dir / "Klimaszenarien-Raumklima_Metadata.csv").write_text("foo,bar\n1,2\n")

    parquet_dir = workspace.parquet()
    failed = convert_indoor_to_parquet("climate_scenarios_indoor", workspace)
    assert failed == 0

    df = pl.read_parquet(parquet_dir / "climate_scenarios_indoor" / "climate_scenarios_indoor.parquet")
    assert set(df["station_abbr"].unique()) == {"ABO"}


# --- climate_scenarios C8 (metadata preamble + wide model table) ---

_CS_CSV = (
    "TITLE;Climate CH2025\n"
    "VARIABLE;Daily precipitation sum\n"
    "STATION_ABBR;ABE\n"
    "GWL;GWL1.5\n"
    "\n"
    "DATE;MODEL_A;MODEL_B\n"
    "0001-01-01;0;24.2\n"
    "0001-01-02;1.5;0\n"
)


def test_parse_climate_scenarios_csv_skips_preamble():
    df = parse_climate_scenarios_csv(_CS_CSV.encode("utf-8"), "ogd-climate-scenarios-ch2025_abe_pr_gwl1.5.csv")
    assert df.columns[:4] == ["station_abbr", "variable", "gwl", "date"]
    assert {"MODEL_A", "MODEL_B"} <= set(df.columns)
    assert df["station_abbr"][0] == "abe"
    assert df["variable"][0] == "pr"
    assert df["gwl"][0] == "gwl1.5"
    assert df["date"].to_list() == ["0001-01-01", "0001-01-02"]
    assert df["MODEL_A"].dtype == pl.Float64
    assert len(df) == 2


def test_convert_preamble_to_parquet(tmp_path):
    workspace = Workspace(tmp_path)
    bronze_dir = workspace.bronze()
    cs_dir = bronze_dir / "climate_scenarios"
    cs_dir.mkdir(parents=True)
    (cs_dir / "ogd-climate-scenarios-ch2025_abe_pr_gwl1.5.csv").write_text(_CS_CSV)
    (cs_dir / "ogd-climate-scenarios-ch2025_ber_tas_gwl2.0.csv").write_text(_CS_CSV)

    parquet_dir = workspace.parquet()
    failed = convert_preamble_to_parquet("climate_scenarios", workspace)
    assert failed == 0

    df = pl.read_parquet(parquet_dir / "climate_scenarios" / "climate_scenarios.parquet")
    assert set(df["station_abbr"].unique()) == {"abe", "ber"}
    assert set(df["variable"].unique()) == {"pr", "tas"}
    assert set(df["gwl"].unique()) == {"gwl1.5", "gwl2.0"}
    assert len(df) == 4


def test_scan_climate_scenarios_csv_is_lazy_and_matches_eager(tmp_path):
    path = tmp_path / "ogd-climate-scenarios-ch2025_abe_pr_gwl1.5.csv"
    path.write_text(_CS_CSV)

    lf = scan_climate_scenarios_csv(path)
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().equals(parse_climate_scenarios_csv(_CS_CSV.encode("utf-8"), path.name))


def test_scan_climate_scenarios_csv_survives_quote_in_preamble(tmp_path):
    """A stray quote in a metadata value must not shift the header offset."""
    path = tmp_path / "ogd-climate-scenarios-ch2025_abe_pr_gwl1.5.csv"
    path.write_text(_CS_CSV.replace("Daily precipitation sum", 'Daily precipitation ("mm")'))

    df = scan_climate_scenarios_csv(path).collect()
    assert df.columns[:4] == ["station_abbr", "variable", "gwl", "date"]
    assert len(df) == 2


def test_convert_preamble_to_parquet_counts_bad_file(tmp_path):
    workspace = Workspace(tmp_path)
    bronze_dir = workspace.bronze()
    cs_dir = bronze_dir / "climate_scenarios"
    cs_dir.mkdir(parents=True)
    (cs_dir / "ogd-climate-scenarios-ch2025_abe_pr_gwl1.5.csv").write_text(_CS_CSV)
    # No 'DATE;' header — must be skipped and counted, not abort the whole run.
    (cs_dir / "ogd-climate-scenarios-ch2025_ber_tas_gwl2.0.csv").write_text("TITLE;nope\n")

    parquet_dir = workspace.parquet()
    assert convert_preamble_to_parquet("climate_scenarios", workspace) == 1

    df = pl.read_parquet(parquet_dir / "climate_scenarios" / "climate_scenarios.parquet")
    assert set(df["station_abbr"].unique()) == {"abe"}


# --- forecast_local Date -> reference_timestamp ---


def test_add_forecast_local_timestamp():
    df = pl.DataFrame({"point_id": [1], "Date": [202605202100], "dkl010h0": [282]})
    out = add_forecast_local_timestamp(df)
    assert "reference_timestamp" in out.columns
    assert out["reference_timestamp"].dtype == pl.Datetime
    ts = out["reference_timestamp"][0]
    assert (ts.year, ts.month, ts.day, ts.hour) == (2026, 5, 20, 21)


def test_add_forecast_local_timestamp_noop_without_date():
    df = pl.DataFrame({"a": [1]})
    assert add_forecast_local_timestamp(df).columns == ["a"]
