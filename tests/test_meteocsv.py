"""Tests for MeteoSwiss's CSV conventions — decoding, typing, filenames, preamble.

Split out of test_convert alongside the module: these describe how upstream
writes a file, and hold whether or not foehn ever produces a Parquet one.
"""

import re
import shutil
from pathlib import Path
from unittest.mock import patch

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
    utf8_meteoswiss_csv,
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


# --- What upstream can send that must not raise ---


def test_a_utf8_bom_is_skipped_without_copying_the_file():
    """MeteoSwiss ships some CSVs BOM-first; polars would read it into the first column name."""
    content = b"\xef\xbb\xbfstation_abbr;tre200d0\nBER;12.5\n"

    out = utf8_meteoswiss_csv(content)

    assert bytes(out).startswith(b"station_abbr")
    assert pl.read_csv(bytes(out), separator=";").columns == ["station_abbr", "tre200d0"]


def test_unreadable_metadata_yields_no_dtypes_rather_than_raising():
    """A truncated _meta_parameters.csv must cost inferred dtypes, not the whole load."""
    assert parse_metadata_types(b"\x00\x01\x02 not a csv") == {}


def test_metadata_without_the_expected_columns_yields_no_dtypes():
    assert parse_metadata_types(b"something;else\n1;2\n") == {}


def test_an_unreadable_metadata_file_yields_no_dtypes(tmp_path, monkeypatch):
    """The file-based counterpart: an I/O failure is the same non-event."""
    (tmp_path / "ogd-smn_meta_parameters.csv").write_bytes(b"x")

    def boom(_self, *args, **kwargs):
        raise OSError("device error")

    monkeypatch.setattr(Path, "read_bytes", boom)
    assert load_metadata_types(tmp_path) == {}


def test_no_metadata_file_yields_no_dtypes(tmp_path):
    assert load_metadata_types(tmp_path) == {}


def test_an_unreadable_header_falls_back_to_parsing_every_column():
    """The header probe is an optimisation; losing it must not lose the data."""
    data = b"station_abbr;tre200d0\nBER;12.5\n"
    with patch.object(pl, "read_csv", side_effect=[ValueError("no header for you"), pl.DataFrame({"a": [1]})]):
        df = parse_csv_bytes(data, metadata_types={"tre200d0": pl.Float64})
    assert df.columns == ["a"]


def test_a_scenarios_csv_with_no_data_header_names_the_file():
    """The preamble runs until 'DATE;'; without it there is nothing to parse."""
    with pytest.raises(ValueError, match=re.escape("ch2025_abe_pr_gwl1.5.csv")):
        parse_climate_scenarios_csv("TITLE;Climate\nVARIABLE;pr\n", "ch2025_abe_pr_gwl1.5.csv")


def test_a_filename_without_a_time_slice_groups_by_granularity_alone(tmp_path):
    """A name carrying a granularity and no slice groups on the granularity alone."""
    for name in ("ogd-nbcn_ber_d.csv", "ogd-nbcn_zur_d.csv"):
        (tmp_path / name).write_text("station_abbr;x\nBER;1\n")

    groups = group_csv_files(tmp_path, "nbcn")

    assert list(groups) == [("d",)]
    assert len(groups[("d",)]) == 2


def test_empty_metadata_content_yields_no_dtypes():
    """A zero-byte _meta_parameters.csv is what a truncated download leaves behind."""
    assert parse_metadata_types(b"") == {}


def test_a_dtype_error_outside_the_projection_is_not_retried():
    """Widening a column the projection excluded is rejected by polars — re-raise instead."""
    data = b"station_abbr;wanted;other\nBER;1;2\n"
    err = pl.exceptions.ComputeError("could not parse at column 'other'")

    with (
        patch.object(pl, "read_csv", side_effect=[pl.DataFrame({"wanted": ["x"]}), err]),
        pytest.raises(pl.exceptions.ComputeError, match="at column 'other'"),
    ):
        parse_csv_bytes(data, metadata_types={"wanted": pl.Int64}, wanted_columns={"station_abbr", "wanted"})


def test_a_non_dtype_failure_on_retry_is_raised_as_itself():
    """Only dtype drift is recoverable; anything else must surface unchanged."""
    data = b"station_abbr;tre200d0\nBER;12.5\n"
    header = pl.DataFrame({"station_abbr": ["x"], "tre200d0": ["y"]})
    attempts = [
        header,
        pl.exceptions.ComputeError("could not parse at column 'tre200d0'"),
        MemoryError("out of memory"),
    ]

    with patch.object(pl, "read_csv", side_effect=attempts), pytest.raises(MemoryError):
        parse_csv_bytes(data, metadata_types={"tre200d0": pl.Int64})


# --- The indoor archive's members ---


def test_the_archives_metadata_csv_has_no_station():
    """The one member that is not data, and the question both pipelines ask of a name."""
    from foehn.meteocsv import indoor_station

    assert indoor_station("ABO_2035_RCP85_DRY.csv") == "ABO"
    assert indoor_station("indoor/AIG_2060_RCP26_DRY_v2.csv") == "AIG"
    assert indoor_station("metadata.csv") is None


def test_reading_a_non_data_member_says_to_ask_first():
    """Whether a member is data is ``indoor_station``'s answer, asked before the read.

    The readers take a member the caller has already accepted, so reaching one
    of them with the metadata CSV is a caller error rather than a row to skip.
    """
    from foehn.meteocsv import parse_indoor_csv

    with pytest.raises(ValueError, match="indoor_station"):
        parse_indoor_csv(b"a,b\n1,2\n", "metadata.csv")
