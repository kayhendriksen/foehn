"""Tests for MeteoSwiss's CSV conventions — decoding, typing, filenames, preamble.

Split out of test_convert alongside the module: these describe how upstream
writes a file, and hold whether or not foehn ever produces a Parquet one.
"""

import shutil
from pathlib import Path

import polars as pl
import pytest
from conftest import CLIMATE_SCENARIOS_CSV

from foehn.meteocsv import (
    add_forecast_local_timestamp,
    decode_meteoswiss_csv,
    derive_timestamp,
    group_csv_files,
    load_metadata_types,
    parse_climate_scenarios_csv,
    parse_csv_bytes,
    parse_metadata_types,
    scan_climate_scenarios_csv,
    source_columns_for,
)
from foehn.workspace import Workspace

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def smn_workspace(tmp_path):
    """A workspace whose bronze holds SMN CSVs and metadata."""
    smn_dir = Workspace(tmp_path).bronze("smn")
    smn_dir.mkdir(parents=True)
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_dir / "ogd-smn_abo_d_recent.csv")
    shutil.copy(FIXTURES_DIR / "smn_sample.csv", smn_dir / "ogd-smn_ber_d_recent.csv")
    shutil.copy(FIXTURES_DIR / "smn_meta_parameters.csv", smn_dir / "ogd-smn_meta_parameters.csv")
    return Workspace(tmp_path)


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


def test_parse_climate_scenarios_csv_skips_preamble():
    df = parse_climate_scenarios_csv(
        CLIMATE_SCENARIOS_CSV.encode("utf-8"), "ogd-climate-scenarios-ch2025_abe_pr_gwl1.5.csv"
    )
    assert df.columns[:4] == ["station_abbr", "variable", "gwl", "date"]
    assert {"MODEL_A", "MODEL_B"} <= set(df.columns)
    assert df["station_abbr"][0] == "abe"
    assert df["variable"][0] == "pr"
    assert df["gwl"][0] == "gwl1.5"
    assert df["date"].to_list() == ["0001-01-01", "0001-01-02"]
    assert df["MODEL_A"].dtype == pl.Float64
    assert len(df) == 2


def test_scan_climate_scenarios_csv_is_lazy_and_matches_eager(tmp_path):
    path = tmp_path / "ogd-climate-scenarios-ch2025_abe_pr_gwl1.5.csv"
    path.write_text(CLIMATE_SCENARIOS_CSV)

    lf = scan_climate_scenarios_csv(path)
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().equals(parse_climate_scenarios_csv(CLIMATE_SCENARIOS_CSV.encode("utf-8"), path.name))


def test_scan_climate_scenarios_csv_survives_quote_in_preamble(tmp_path):
    """A stray quote in a metadata value must not shift the header offset."""
    path = tmp_path / "ogd-climate-scenarios-ch2025_abe_pr_gwl1.5.csv"
    path.write_text(CLIMATE_SCENARIOS_CSV.replace("Daily precipitation sum", 'Daily precipitation ("mm")'))

    df = scan_climate_scenarios_csv(path).collect()
    assert df.columns[:4] == ["station_abbr", "variable", "gwl", "date"]
    assert len(df) == 2


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


# --- which kinds derive a timestamp, and from what ---


def test_derive_timestamp_is_a_no_op_for_kinds_that_ship_one():
    """Callers derive unconditionally; the kind decides whether anything happens."""
    frame = pl.DataFrame({"Date": [202605202100], "v": [1]})
    assert derive_timestamp(frame, "smn").equals(frame)
    assert "reference_timestamp" in derive_timestamp(frame, "forecast_local").columns


def test_derive_timestamp_leaves_an_existing_column_alone():
    frame = pl.DataFrame({"Date": [202605202100], "reference_timestamp": [None]})
    assert derive_timestamp(frame, "forecast_local").equals(frame)


def test_only_the_deriving_kinds_need_extra_source_columns():
    """The two facts agree: a kind needs a source column exactly when it derives."""
    from foehn import registry
    from foehn.collections import DERIVED_TIMESTAMP_KINDS, kind

    for dataset in registry.tabular_datasets():
        needs = source_columns_for(dataset)
        assert bool(needs) is (kind(dataset) in DERIVED_TIMESTAMP_KINDS), dataset
    assert source_columns_for("forecast_local") == {"Date"}
