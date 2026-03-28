"""Tests for the CLI argument handling."""

from unittest.mock import patch

import polars as pl
import pytest

from foehn.cli import main

_PATCHES = [
    "foehn.cli.download_collection",
    "foehn.cli.download_metadata",
    "foehn.cli.download_grib2",
    "foehn.cli.download_netcdf",
    "foehn.cli.download_climate_normals_zip",
    "foehn.cli.convert_to_parquet",
    "foehn.cli.convert_climate_normals_to_parquet",
    "foehn.cli.save_last_run",
    "foehn.cli.load_last_run",
]


def _run(subcommand, args, tmp_path):
    """Invoke main() with patched sys.argv and all I/O mocked."""
    mocks = {}
    patchers = [patch(t) for t in _PATCHES]
    started = [p.start() for p in patchers]
    for name, mock in zip(_PATCHES, started, strict=True):
        mocks[name.split(".")[-1]] = mock
    mocks["load_last_run"].return_value = None

    try:
        with patch("sys.argv", ["foehn", subcommand, "--data-dir", str(tmp_path), *args]):
            main()
    finally:
        for p in patchers:
            p.stop()

    return mocks


def _run_without_data_dir(subcommand, args, tmp_path):
    """Like _run but without --data-dir, so env var can take effect."""
    mocks = {}
    patchers = [patch(t) for t in _PATCHES]
    started = [p.start() for p in patchers]
    for name, mock in zip(_PATCHES, started, strict=True):
        mocks[name.split(".")[-1]] = mock
    mocks["load_last_run"].return_value = None

    try:
        with patch("sys.argv", ["foehn", subcommand, *args]):
            main()
    finally:
        for p in patchers:
            p.stop()

    return mocks


# --- time_slice assembly ---


def test_default_uses_recent_only(tmp_path):
    mocks = _run("download", [], tmp_path)
    calls = mocks["download_collection"].call_args_list
    assert calls, "download_collection should be called"
    time_slices = calls[0][1]["data_types"]
    assert time_slices == ["recent"]


def test_now_flag_adds_now(tmp_path):
    mocks = _run("download", ["--now"], tmp_path)
    calls = mocks["download_collection"].call_args_list
    time_slices = calls[0][1]["data_types"]
    assert "now" in time_slices
    assert "recent" in time_slices


def test_historical_flag_prepends_historical(tmp_path):
    mocks = _run("download", ["--historical"], tmp_path)
    calls = mocks["download_collection"].call_args_list
    time_slices = calls[0][1]["data_types"]
    assert time_slices[0] == "historical"
    assert "recent" in time_slices


def test_all_time_slices(tmp_path):
    mocks = _run("download", ["--now", "--historical"], tmp_path)
    calls = mocks["download_collection"].call_args_list
    time_slices = calls[0][1]["data_types"]
    assert set(time_slices) == {"historical", "recent", "now"}


# --- to-parquet subcommand ---


def test_to_parquet_skips_downloads(tmp_path):
    mocks = _run("to-parquet", [], tmp_path)
    mocks["download_collection"].assert_not_called()
    mocks["download_metadata"].assert_not_called()
    mocks["convert_to_parquet"].assert_called()


# --- no-parquet ---


def test_no_parquet_skips_conversion(tmp_path):
    mocks = _run("download", ["--no-parquet"], tmp_path)
    mocks["convert_to_parquet"].assert_not_called()
    mocks["convert_climate_normals_to_parquet"].assert_not_called()


def test_default_runs_conversion(tmp_path):
    mocks = _run("download", [], tmp_path)
    mocks["convert_to_parquet"].assert_called()
    mocks["convert_climate_normals_to_parquet"].assert_called()


# --- full-refresh ---


def test_full_refresh_ignores_last_run(tmp_path):
    mocks = _run("download", ["--full-refresh"], tmp_path)
    mocks["load_last_run"].assert_not_called()


def test_incremental_passes_since_to_download(tmp_path):
    mocks = _run("download", [], tmp_path)
    mocks["load_last_run"].assert_called_once()


# --- grids ---


def test_grids_flag_enables_grib2_and_netcdf(tmp_path):
    mocks = _run("download", ["--grids"], tmp_path)
    mocks["download_grib2"].assert_called()
    mocks["download_netcdf"].assert_called()


def test_default_skips_grids(tmp_path):
    mocks = _run("download", [], tmp_path)
    mocks["download_grib2"].assert_not_called()
    mocks["download_netcdf"].assert_not_called()


# --- list subcommand ---


def test_list_prints_collections(tmp_path, capsys):
    mocks = _run("list", [], tmp_path)
    out = capsys.readouterr().out
    assert "smn" in out
    assert "Automatic weather stations" in out
    mocks["download_collection"].assert_not_called()


# --- env vars ---


def test_env_data_dir_used_when_no_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("FOEHN_DATA_DIR", str(tmp_path / "env-dir"))
    mocks = _run_without_data_dir("download", [], tmp_path)
    calls = mocks["download_collection"].call_args_list
    assert calls
    raw_dir = calls[0][0][1]
    assert str(tmp_path / "env-dir") in str(raw_dir)


def test_cli_data_dir_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FOEHN_DATA_DIR", str(tmp_path / "env-dir"))
    mocks = _run("download", [], tmp_path)
    calls = mocks["download_collection"].call_args_list
    assert calls
    raw_dir = calls[0][0][1]
    assert str(tmp_path) in str(raw_dir)
    assert "env-dir" not in str(raw_dir)


def test_env_full_refresh_truthy(tmp_path, monkeypatch):
    monkeypatch.setenv("FOEHN_FULL_REFRESH", "1")
    mocks = _run("download", [], tmp_path)
    mocks["load_last_run"].assert_not_called()


# --- unknown key ---


def test_unknown_dataset_exits(tmp_path):
    with pytest.raises(SystemExit):
        _run("download", ["nonexistent_dataset"], tmp_path)


# --- list filters ---


def test_list_category_filter(tmp_path, capsys):
    _run("list", ["--category", "A"], tmp_path)
    out = capsys.readouterr().out
    assert "Ground-based measurements" in out
    assert "Forecast data" not in out


def test_list_format_filter(tmp_path, capsys):
    _run("list", ["--format", "GRIB2"], tmp_path)
    out = capsys.readouterr().out
    assert "GRIB2" in out


def test_list_no_matches(tmp_path, capsys):
    _run("list", ["--category", "Z"], tmp_path)
    out = capsys.readouterr().out
    assert "No datasets match" in out


# --- incremental update ---


def test_incremental_prints_since(tmp_path, capsys):
    mocks = {}
    patchers = [patch(t) for t in _PATCHES]
    started = [p.start() for p in patchers]
    for name, mock in zip(_PATCHES, started, strict=True):
        mocks[name.split(".")[-1]] = mock
    mocks["load_last_run"].return_value = "2025-01-01T00:00:00"

    try:
        with patch("sys.argv", ["foehn", "download", "--data-dir", str(tmp_path)]):
            main()
    finally:
        for p in patchers:
            p.stop()

    out = capsys.readouterr().out
    assert "Incremental update" in out


# --- to-parquet skips grids ---


def test_to_parquet_skips_grid_collections(tmp_path):
    mocks = _run("to-parquet", [], tmp_path)
    # convert_to_parquet should only be called for CSV collections, not grid ones
    for call in mocks["convert_to_parquet"].call_args_list:
        key = call[0][0]
        from foehn.collections import GRIB2_COLLECTIONS, NETCDF_COLLECTIONS

        assert key not in GRIB2_COLLECTIONS
        assert key not in NETCDF_COLLECTIONS


# --- load subcommand ---


def test_load_prints_dataframe(capsys):
    fake_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with patch("foehn.api.load", return_value=fake_df) as mock_load, patch("sys.argv", ["foehn", "load", "smn"]):
        main()
    mock_load.assert_called_once_with("smn")
    out = capsys.readouterr().out
    assert "3 rows x 2 columns" in out


def test_load_with_filters():
    fake_df = pl.DataFrame({"a": [1]})
    argv = ["foehn", "load", "smn", "--station", "BER", "--frequency", "d", "--time-slice", "recent", "-n", "5"]
    with patch("foehn.api.load", return_value=fake_df) as mock_load, patch("sys.argv", argv):
        main()
    mock_load.assert_called_once_with("smn", station=["BER"], frequency=["d"], time_slice=["recent"])


# --- metadata subcommand ---


def test_metadata_parameters(capsys):
    fake_df = pl.DataFrame({"shortname": ["tre200d0"], "description": ["Air temp"], "unit": ["°C"]})
    with (
        patch("foehn.cli.parameters", return_value=fake_df),
        patch("sys.argv", ["foehn", "metadata", "parameters", "smn"]),
    ):
        main()
    out = capsys.readouterr().out
    assert "tre200d0" in out
    assert "Air temp" in out
    assert "1 rows" in out


def test_metadata_stations(capsys):
    fake_df = pl.DataFrame({"abbr": ["BER"], "name": ["Bern"], "canton": ["BE"]})
    with (
        patch("foehn.cli.stations", return_value=fake_df),
        patch("sys.argv", ["foehn", "metadata", "stations", "smn"]),
    ):
        main()
    out = capsys.readouterr().out
    assert "BER" in out
    assert "Bern" in out


def test_metadata_inventory(capsys):
    fake_df = pl.DataFrame({"station": ["BER"], "parameter": ["tre200d0"]})
    with (
        patch("foehn.cli.inventory", return_value=fake_df),
        patch("sys.argv", ["foehn", "metadata", "inventory", "smn"]),
    ):
        main()
    out = capsys.readouterr().out
    assert "BER" in out
    assert "tre200d0" in out


def test_metadata_empty(capsys):
    fake_df = pl.DataFrame({"shortname": [], "description": [], "unit": []})
    with (
        patch("foehn.cli.parameters", return_value=fake_df),
        patch("sys.argv", ["foehn", "metadata", "parameters", "smn"]),
    ):
        main()
    out = capsys.readouterr().out
    assert "No parameters metadata found" in out
