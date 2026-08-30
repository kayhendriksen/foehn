"""Tests for the CLI argument handling."""

from unittest.mock import patch

import polars as pl
import pytest

from foehn import registry
from foehn.cli import main

# The CLI no longer knows which handler serves which dataset — the registry does.
# What is left to stub is the registry's two entry points plus the C6 climate
# normals, which are not a STAC collection and so have no kind.
_PATCHES = [
    "foehn.registry.download",
    "foehn.registry.convert",
    "foehn.cli.download_climate_normals_zip",
    "foehn.cli.convert_climate_normals_to_parquet",
    "foehn.cli.save_last_run",
    "foehn.cli.load_last_run",
]


def _start_mocks():
    """Start patches, set conversion mocks to return 0 (no failures), return mock dict."""
    mocks = {}
    patchers = [patch(t) for t in _PATCHES]
    started = [p.start() for p in patchers]
    for name, mock in zip(_PATCHES, started, strict=True):
        mocks[name.split(".")[-1]] = mock
    mocks["load_last_run"].return_value = None
    mocks["convert"].return_value = 0
    mocks["convert_climate_normals_to_parquet"].return_value = 0
    # cmd_download sums result.failed from each download call to gate _last_run;
    # give the download mock a clean (0-failure) DownloadResult by default.
    mocks["download"].return_value.failed = 0
    return mocks, patchers


def _downloaded(mocks):
    """Datasets passed to registry.download, in call order."""
    return [call.args[0] for call in mocks["download"].call_args_list]


def _converted(mocks):
    """Datasets passed to registry.convert, in call order."""
    return [call.args[0] for call in mocks["convert"].call_args_list]


def _run(subcommand, args, tmp_path):
    """Invoke main() with patched sys.argv and all I/O mocked."""
    mocks, patchers = _start_mocks()
    try:
        with patch("sys.argv", ["foehn", subcommand, "--data-dir", str(tmp_path), *args]):
            main()
    finally:
        for p in patchers:
            p.stop()

    return mocks


def _run_without_data_dir(subcommand, args, tmp_path):
    """Like _run but without --data-dir, so env var can take effect."""
    mocks, patchers = _start_mocks()
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
    calls = mocks["download"].call_args_list
    assert calls, "registry.download should be called"
    time_slices = calls[0].kwargs["time_slice"]
    assert time_slices == ["recent"]


def test_now_flag_adds_now(tmp_path):
    mocks = _run("download", ["--now"], tmp_path)
    calls = mocks["download"].call_args_list
    time_slices = calls[0].kwargs["time_slice"]
    assert "now" in time_slices
    assert "recent" in time_slices


def test_historical_flag_prepends_historical(tmp_path):
    mocks = _run("download", ["--historical"], tmp_path)
    calls = mocks["download"].call_args_list
    time_slices = calls[0].kwargs["time_slice"]
    assert time_slices[0] == "historical"
    assert "recent" in time_slices


def test_all_time_slices(tmp_path):
    mocks = _run("download", ["--now", "--historical"], tmp_path)
    calls = mocks["download"].call_args_list
    time_slices = calls[0].kwargs["time_slice"]
    assert set(time_slices) == {"historical", "recent", "now"}


# --- to-parquet subcommand ---


def test_to_parquet_skips_downloads(tmp_path):
    mocks = _run("to-parquet", [], tmp_path)
    mocks["download"].assert_not_called()
    mocks["convert"].assert_called()


# --- no-parquet ---


def test_no_parquet_skips_conversion(tmp_path):
    mocks = _run("download", ["--no-parquet"], tmp_path)
    mocks["convert"].assert_not_called()
    mocks["convert_climate_normals_to_parquet"].assert_not_called()


def test_default_runs_conversion(tmp_path):
    mocks = _run("download", [], tmp_path)
    mocks["convert"].assert_called()
    mocks["convert_climate_normals_to_parquet"].assert_called()


def test_default_run_covers_every_tabular_dataset(tmp_path):
    """Which handler each one needs is the registry's business, not the CLI's."""
    mocks = _run("download", [], tmp_path)

    assert _downloaded(mocks) == registry.tabular_datasets()
    assert _converted(mocks) == registry.tabular_datasets()


# --- full-refresh ---


def test_full_refresh_ignores_last_run(tmp_path):
    mocks = _run("download", ["--full-refresh"], tmp_path)
    mocks["load_last_run"].assert_not_called()


def test_incremental_passes_since_to_download(tmp_path):
    mocks = _run("download", [], tmp_path)
    mocks["load_last_run"].assert_called_once()


# --- grids ---


def test_grids_flag_enables_grid_datasets(tmp_path):
    mocks = _run("download", ["--grids"], tmp_path)
    assert set(_downloaded(mocks)) >= set(registry.grid_datasets())


def test_default_skips_grids(tmp_path):
    mocks = _run("download", [], tmp_path)
    assert not set(_downloaded(mocks)) & set(registry.grid_datasets())


# --- list subcommand ---


def test_list_prints_collections(tmp_path, capsys):
    mocks = _run("list", [], tmp_path)
    out = capsys.readouterr().out
    assert "smn" in out
    assert "Automatic weather stations" in out
    mocks["download"].assert_not_called()


# --- env vars ---


def test_env_data_dir_used_when_no_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("FOEHN_DATA_DIR", str(tmp_path / "env-dir"))
    mocks = _run_without_data_dir("download", [], tmp_path)
    calls = mocks["download"].call_args_list
    assert calls
    bronze_dir = calls[0].args[1]
    assert str(tmp_path / "env-dir") in str(bronze_dir)


def test_cli_data_dir_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FOEHN_DATA_DIR", str(tmp_path / "env-dir"))
    mocks = _run("download", [], tmp_path)
    calls = mocks["download"].call_args_list
    assert calls
    bronze_dir = calls[0].args[1]
    assert str(tmp_path) in str(bronze_dir)
    assert "env-dir" not in str(bronze_dir)


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
    mocks, patchers = _start_mocks()
    mocks["load_last_run"].return_value = "2025-01-01T00:00:00"

    try:
        with patch("sys.argv", ["foehn", "download", "--data-dir", str(tmp_path)]):
            main()
    finally:
        for p in patchers:
            p.stop()

    out = capsys.readouterr().out
    assert "Incremental update" in out


# --- failure gating ---


def test_save_last_run_skipped_on_convert_failure(tmp_path, capsys):
    """If a conversion reports failures, _last_run.json must NOT be saved."""
    mocks, patchers = _start_mocks()
    mocks["convert"].return_value = 2  # 2 groups failed
    try:
        with patch("sys.argv", ["foehn", "download", "--data-dir", str(tmp_path)]), pytest.raises(SystemExit) as exc:
            main()
    finally:
        for p in patchers:
            p.stop()

    assert exc.value.code == 1
    mocks["save_last_run"].assert_not_called()
    err = capsys.readouterr().err
    assert "not advancing _last_run.json" in err


def test_save_last_run_skipped_on_download_failure(tmp_path, capsys):
    """If a download reports failures, _last_run.json must NOT be saved."""
    mocks, patchers = _start_mocks()
    mocks["download"].return_value.failed = 1
    try:
        with patch("sys.argv", ["foehn", "download", "--data-dir", str(tmp_path)]), pytest.raises(SystemExit) as exc:
            main()
    finally:
        for p in patchers:
            p.stop()

    assert exc.value.code == 1
    mocks["save_last_run"].assert_not_called()
    err = capsys.readouterr().err
    assert "not advancing _last_run.json" in err


def test_save_last_run_called_on_clean_run(tmp_path):
    """When all conversions return 0 failures, _last_run.json is saved as before."""
    mocks = _run("download", [], tmp_path)
    mocks["save_last_run"].assert_called_once()


def test_to_parquet_exits_nonzero_on_failure(tmp_path):
    mocks, patchers = _start_mocks()
    mocks["convert"].return_value = 1
    try:
        with patch("sys.argv", ["foehn", "to-parquet", "--data-dir", str(tmp_path)]), pytest.raises(SystemExit) as exc:
            main()
    finally:
        for p in patchers:
            p.stop()
    assert exc.value.code == 1


# --- to-parquet skips grids ---


def test_to_parquet_skips_grid_collections(tmp_path):
    mocks = _run("to-parquet", [], tmp_path)
    assert all(registry.spec(key).tabular for key in _converted(mocks))


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


def test_load_with_post_filters():
    fake_df = pl.DataFrame({"a": [1]})
    argv = [
        "foehn",
        "load",
        "smn",
        "--station",
        "BER",
        "--frequency",
        "d",
        "--year",
        "2025",
        "--month",
        "6",
        "7",
        "--date-from",
        "2025-06-01",
        "--date-to",
        "2025-08-31",
        "--columns",
        "temp",
        "precip",
        "--drop-null",
        "temp",
        "--sort",
        "desc",
    ]
    with patch("foehn.api.load", return_value=fake_df) as mock_load, patch("sys.argv", argv):
        main()
    mock_load.assert_called_once_with(
        "smn",
        station=["BER"],
        frequency=["d"],
        year=[2025],
        month=[6, 7],
        date_from="2025-06-01",
        date_to="2025-08-31",
        columns=["temp", "precip"],
        drop_null="temp",
        sort="desc",
    )


def test_load_forwards_limit_and_workers():
    """--limit and --workers must reach foehn.load (not just bound the preview)."""
    fake_df = pl.DataFrame({"a": [1]})
    argv = ["foehn", "load", "smn", "--limit", "5", "--workers", "3"]
    with patch("foehn.api.load", return_value=fake_df) as mock_load, patch("sys.argv", argv):
        main()
    mock_load.assert_called_once_with("smn", limit=5, workers=3)


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
