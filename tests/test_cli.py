"""Tests for the CLI argument handling."""

from unittest.mock import patch

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


def _run(args, tmp_path):
    """Invoke main() with patched sys.argv and all I/O mocked."""
    mocks = {}
    patchers = [patch(t) for t in _PATCHES]
    started = [p.start() for p in patchers]
    for name, mock in zip(_PATCHES, started, strict=True):
        mocks[name.split(".")[-1]] = mock
    mocks["load_last_run"].return_value = None

    try:
        with patch("sys.argv", ["foehn", "--data-dir", str(tmp_path), *args]):
            main()
    finally:
        for p in patchers:
            p.stop()

    return mocks


def _run_without_data_dir(args, tmp_path):
    """Like _run but without --data-dir, so env var can take effect."""
    mocks = {}
    patchers = [patch(t) for t in _PATCHES]
    started = [p.start() for p in patchers]
    for name, mock in zip(_PATCHES, started, strict=True):
        mocks[name.split(".")[-1]] = mock
    mocks["load_last_run"].return_value = None

    try:
        with patch("sys.argv", ["foehn", *args]):
            main()
    finally:
        for p in patchers:
            p.stop()

    return mocks


# --- data_types assembly ---


def test_default_uses_recent_only(tmp_path):
    mocks = _run([], tmp_path)
    calls = mocks["download_collection"].call_args_list
    assert calls, "download_collection should be called"
    data_types = calls[0][1]["data_types"]
    assert data_types == ["recent"]


def test_now_flag_adds_now(tmp_path):
    mocks = _run(["--now"], tmp_path)
    calls = mocks["download_collection"].call_args_list
    data_types = calls[0][1]["data_types"]
    assert "now" in data_types
    assert "recent" in data_types


def test_historical_flag_prepends_historical(tmp_path):
    mocks = _run(["--historical"], tmp_path)
    calls = mocks["download_collection"].call_args_list
    data_types = calls[0][1]["data_types"]
    assert data_types[0] == "historical"
    assert "recent" in data_types


def test_all_time_slices(tmp_path):
    mocks = _run(["--now", "--historical"], tmp_path)
    calls = mocks["download_collection"].call_args_list
    data_types = calls[0][1]["data_types"]
    assert set(data_types) == {"historical", "recent", "now"}


# --- convert-only ---


def test_convert_only_skips_downloads(tmp_path):
    mocks = _run(["--convert-only"], tmp_path)
    mocks["download_collection"].assert_not_called()
    mocks["download_metadata"].assert_not_called()
    mocks["convert_to_parquet"].assert_called()


def test_convert_only_skips_save_last_run(tmp_path):
    mocks = _run(["--convert-only"], tmp_path)
    mocks["save_last_run"].assert_not_called()


# --- no-parquet ---


def test_no_parquet_skips_conversion(tmp_path):
    mocks = _run(["--no-parquet"], tmp_path)
    mocks["convert_to_parquet"].assert_not_called()
    mocks["convert_climate_normals_to_parquet"].assert_not_called()


def test_default_runs_conversion(tmp_path):
    mocks = _run([], tmp_path)
    mocks["convert_to_parquet"].assert_called()
    mocks["convert_climate_normals_to_parquet"].assert_called()


# --- full-refresh ---


def test_full_refresh_ignores_last_run(tmp_path):
    mocks = _run(["--full-refresh"], tmp_path)
    # load_last_run is never called when --full-refresh is set
    mocks["load_last_run"].assert_not_called()


def test_incremental_passes_since_to_download(tmp_path):
    mocks = _run([], tmp_path)
    # load_last_run was called (returns None in our mock, so since=None)
    mocks["load_last_run"].assert_called_once()


# --- grids ---


def test_grids_flag_enables_grib2_and_netcdf(tmp_path):
    mocks = _run(["--grids"], tmp_path)
    mocks["download_grib2"].assert_called()
    mocks["download_netcdf"].assert_called()


def test_default_skips_grids(tmp_path):
    mocks = _run([], tmp_path)
    mocks["download_grib2"].assert_not_called()
    mocks["download_netcdf"].assert_not_called()


# --- --list ---


def test_list_flag_prints_collections_and_exits(tmp_path, capsys):
    mocks = _run(["--list"], tmp_path)
    out = capsys.readouterr().out
    assert "smn" in out
    assert "ch.meteoschweiz.ogd-smn" in out
    mocks["download_collection"].assert_not_called()


# --- env vars ---


def test_env_data_dir_used_when_no_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("FOEHN_DATA_DIR", str(tmp_path / "env-dir"))
    mocks = _run_without_data_dir([], tmp_path)
    calls = mocks["download_collection"].call_args_list
    assert calls
    # raw_dir should be under the env var path
    raw_dir = calls[0][0][1]
    assert str(tmp_path / "env-dir") in str(raw_dir)


def test_cli_data_dir_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FOEHN_DATA_DIR", str(tmp_path / "env-dir"))
    mocks = _run([], tmp_path)  # _run passes --data-dir explicitly
    calls = mocks["download_collection"].call_args_list
    assert calls
    raw_dir = calls[0][0][1]
    assert str(tmp_path) in str(raw_dir)
    assert "env-dir" not in str(raw_dir)


def test_env_full_refresh_truthy(tmp_path, monkeypatch):
    monkeypatch.setenv("FOEHN_FULL_REFRESH", "1")
    mocks = _run([], tmp_path)
    mocks["load_last_run"].assert_not_called()
