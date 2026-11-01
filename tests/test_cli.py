"""Tests for the CLI argument handling."""

from datetime import datetime
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
    "foehn.cli.advance_cursor",
    "foehn.cli.load_cursor",
]


def _start_mocks():
    """Start patches, set conversion mocks to return 0 (no failures), return mock dict."""
    mocks = {}
    patchers = [patch(t) for t in _PATCHES]
    started = [p.start() for p in patchers]
    for name, mock in zip(_PATCHES, started, strict=True):
        mocks[name.split(".")[-1]] = mock
    mocks["load_cursor"].return_value = None
    mocks["convert"].return_value = 0
    # cmd_download reads result.failed from each download call to gate its cursor;
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


def test_default_runs_conversion(tmp_path):
    mocks = _run("download", [], tmp_path)
    mocks["convert"].assert_called()


def test_default_run_covers_every_non_grid_dataset(tmp_path):
    """Which handler each one needs is the registry's business, not the CLI's.

    climate_normals is in this list rather than bolted on after it: the CLI used
    to loop over the tabular datasets and then download and convert that one by
    hand, in both this command and to-parquet.
    """
    mocks = _run("download", [], tmp_path)

    assert _downloaded(mocks) == registry.non_grid_datasets()
    assert _converted(mocks) == registry.non_grid_datasets()
    assert "climate_normals" in _downloaded(mocks)


# --- full-refresh ---


def test_full_refresh_ignores_the_cursors(tmp_path):
    mocks = _run("download", ["--full-refresh"], tmp_path)
    mocks["load_cursor"].assert_not_called()


def test_incremental_reads_each_datasets_own_cursor(tmp_path):
    mocks = _run("download", ["smn", "pollen"], tmp_path)
    assert [call.args[1:] for call in mocks["load_cursor"].call_args_list] == [
        ("smn", ["recent"]),
        ("pollen", ["recent"]),
    ]


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
    mocks["load_cursor"].assert_not_called()


# --- unknown key ---


def test_unknown_dataset_exits(tmp_path):
    with pytest.raises(SystemExit):
        _run("download", ["nonexistent_dataset"], tmp_path)


def test_unknown_dataset_is_refused_before_the_workspace_is_created(tmp_path):
    workspace = tmp_path / "ws"
    with pytest.raises(SystemExit):
        _run("download", ["nonexistent_dataset", "--data-dir", str(workspace)], tmp_path)

    assert not workspace.exists()


# --- errors and version ---


def test_version_prints_the_package_version(capsys):
    import foehn

    with patch("sys.argv", ["foehn", "--version"]), pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"foehn {foehn.__version__}"


def test_a_user_error_is_one_line_on_stderr_not_a_traceback(capsys):
    with patch("sys.argv", ["foehn", "load", "bogus"]), pytest.raises(SystemExit) as exc:
        main()

    err = capsys.readouterr().err
    assert exc.value.code == 1
    assert err.startswith("foehn: error: Unknown dataset: 'bogus'")
    assert "Traceback" not in err


def test_a_network_failure_is_reported_without_a_traceback(capsys):
    from foehn.fetch import FetchError

    with (
        patch("foehn.api.load", side_effect=FetchError("ConnectionError fetching https://data.geo.admin.ch/x")),
        patch("sys.argv", ["foehn", "load", "smn"]),
        pytest.raises(SystemExit) as exc,
    ):
        main()

    assert exc.value.code == 1
    assert "foehn: error: ConnectionError fetching" in capsys.readouterr().err


def test_mcp_without_its_extra_names_the_install_command(capsys, monkeypatch):
    import sys

    # Unimportable ``mcp``, and a fresh import of the server module that needs it.
    for name in [m for m in sys.modules if m.startswith("mcp.")] + ["foehn.mcp_server"]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "mcp", None)

    with patch("sys.argv", ["foehn", "mcp"]), pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    assert 'pip install "foehn[mcp]"' in capsys.readouterr().err


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
    captured = capsys.readouterr()
    # A diagnostic, not output: stdout stays empty so `foehn list ... | wc -l` reads 0.
    assert "No datasets match" in captured.err
    assert captured.out == ""


# --- incremental update ---


def test_incremental_prints_since(tmp_path, capsys):
    mocks, patchers = _start_mocks()
    mocks["load_cursor"].return_value = "2025-01-01T00:00:00+00:00"

    try:
        with patch("sys.argv", ["foehn", "download", "--data-dir", str(tmp_path)]):
            main()
    finally:
        for p in patchers:
            p.stop()

    err = capsys.readouterr().err
    assert "Incremental update" in err
    assert "changes since 2025-01-01T00:00:00+00:00" in err


# --- failure gating ---


def test_cursor_held_back_on_convert_failure(tmp_path, capsys):
    """If a conversion reports failures, that Dataset's cursor must NOT advance."""
    mocks, patchers = _start_mocks()
    mocks["convert"].return_value = 2  # 2 groups failed
    try:
        with patch("sys.argv", ["foehn", "download", "--data-dir", str(tmp_path)]), pytest.raises(SystemExit) as exc:
            main()
    finally:
        for p in patchers:
            p.stop()

    assert exc.value.code == 1
    mocks["advance_cursor"].assert_not_called()
    err = capsys.readouterr().err
    assert "not advancing the _last_run.json cursor" in err


def test_cursor_held_back_on_download_failure(tmp_path, capsys):
    """If a download reports failures, that Dataset's cursor must NOT advance."""
    mocks, patchers = _start_mocks()
    mocks["download"].return_value.failed = 1
    try:
        with patch("sys.argv", ["foehn", "download", "--data-dir", str(tmp_path)]), pytest.raises(SystemExit) as exc:
            main()
    finally:
        for p in patchers:
            p.stop()

    assert exc.value.code == 1
    mocks["advance_cursor"].assert_not_called()
    err = capsys.readouterr().err
    assert "not advancing the _last_run.json cursor" in err


def test_each_clean_dataset_advances_its_own_cursor(tmp_path):
    """When a Dataset downloads and converts cleanly, its cursor advances — and only its."""
    mocks = _run("download", ["smn", "pollen"], tmp_path)
    assert [call.args[1:3] for call in mocks["advance_cursor"].call_args_list] == [
        ("smn", ["recent"]),
        ("pollen", ["recent"]),
    ]


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
    assert not any(registry.spec(key).is_grid for key in _converted(mocks))


# --- load subcommand ---


def test_load_prints_dataframe(capsys):
    fake_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with patch("foehn.api.load", return_value=fake_df) as mock_load, patch("sys.argv", ["foehn", "load", "smn"]):
        main()
    mock_load.assert_called_once()
    assert mock_load.call_args.args == ("smn",)
    # Nothing was asked for, so every filter goes across as "no filter".
    assert all(v is None for k, v in mock_load.call_args.kwargs.items() if k != "workers")
    out = capsys.readouterr().out
    assert "3 rows x 2 columns" in out


def test_load_with_filters():
    fake_df = pl.DataFrame({"a": [1]})
    argv = ["foehn", "load", "smn", "--station", "BER", "--frequency", "d", "--time-slice", "recent", "-n", "5"]
    with patch("foehn.api.load", return_value=fake_df) as mock_load, patch("sys.argv", argv):
        main()
    mock_load.assert_called_once()
    assert mock_load.call_args.args == ("smn",)
    assert mock_load.call_args.kwargs | {"station": ["BER"], "frequency": ["d"], "time_slice": ["recent"]} == (
        mock_load.call_args.kwargs
    )


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
    mock_load.assert_called_once()
    assert mock_load.call_args.args == ("smn",)
    forwarded = mock_load.call_args.kwargs
    assert (
        forwarded
        | {
            "station": ["BER"],
            "frequency": ["d"],
            "year": [2025],
            "month": [6, 7],
            "date_from": "2025-06-01",
            "date_to": "2025-08-31",
            "columns": ["temp", "precip"],
            "drop_null": "temp",
            "sort": "desc",
        }
        == forwarded
    )


def test_load_forwards_limit_and_workers():
    """--limit and --workers must reach foehn.load (not just bound the preview)."""
    fake_df = pl.DataFrame({"a": [1]})
    argv = ["foehn", "load", "smn", "--limit", "5", "--workers", "3"]
    with patch("foehn.api.load", return_value=fake_df) as mock_load, patch("sys.argv", argv):
        main()
    mock_load.assert_called_once()
    assert mock_load.call_args.kwargs["limit"] == 5
    assert mock_load.call_args.kwargs["workers"] == 3


# --- metadata subcommand ---


def test_metadata_parameters(capsys):
    fake_df = pl.DataFrame({"shortname": ["tre200d0"], "description": ["Air temp"], "unit": ["°C"]})
    with (
        patch("foehn.cli.metadata", return_value=fake_df),
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
        patch("foehn.cli.metadata", return_value=fake_df),
        patch("sys.argv", ["foehn", "metadata", "stations", "smn"]),
    ):
        main()
    out = capsys.readouterr().out
    assert "BER" in out
    assert "Bern" in out


def test_metadata_inventory(capsys):
    fake_df = pl.DataFrame({"station": ["BER"], "parameter": ["tre200d0"]})
    with (
        patch("foehn.cli.metadata", return_value=fake_df),
        patch("sys.argv", ["foehn", "metadata", "inventory", "smn"]),
    ):
        main()
    out = capsys.readouterr().out
    assert "BER" in out
    assert "tre200d0" in out


def test_metadata_column_width_matches_what_is_printed(capsys):
    """Width is measured from the same strings that get printed.

    Measuring in Polars while printing Python's ``str()`` looked equivalent but
    is not for Datetime: the cast renders microseconds (26 chars) where str()
    gives 19, so the column was padded wider than anything in it. Latent while
    the metadata frames are all strings — this pins it shut.
    """
    fake_df = pl.DataFrame({"when": [datetime(2026, 1, 1), None], "name": ["Bern", "Zurich"]})
    with (
        patch("foehn.cli.metadata", return_value=fake_df),
        patch("sys.argv", ["foehn", "metadata", "stations", "smn"]),
    ):
        main()

    body = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    rule = next(line for line in body if line.startswith("─"))
    rows = [line for line in body if line and not line.startswith(("─", "when", "["))]
    when_width = len(rule.split("  ")[0])

    assert when_width == max(len(row.split("  ")[0].rstrip()) for row in rows)
    assert "—" in "".join(rows)  # nulls still render as a dash


def test_metadata_empty(capsys):
    fake_df = pl.DataFrame({"shortname": [], "description": [], "unit": []})
    with (
        patch("foehn.cli.metadata", return_value=fake_df),
        patch("sys.argv", ["foehn", "metadata", "parameters", "smn"]),
    ):
        main()
    captured = capsys.readouterr()
    assert "No parameters metadata found" in captured.err
    assert captured.out == ""


# --- The grid subcommands ---


def test_open_prints_the_dataset_summary(tmp_path, capsys):
    """``foehn open`` is a thin front for open_dataset — it prints what xarray reprs."""
    with (
        patch("foehn.api.open_dataset", return_value="<xarray summary>") as opened,
        patch("sys.argv", ["foehn", "open", "surface_derived_grid", "--match", "rhiresd", "--data-dir", str(tmp_path)]),
    ):
        main()

    assert opened.call_args.kwargs["match"] == "rhiresd"
    assert opened.call_args.kwargs["variables"] is None
    assert "<xarray summary>" in capsys.readouterr().out


def test_to_zarr_prints_the_store_path(tmp_path, capsys):
    store = tmp_path / "zarr" / "surface_derived_grid__rhiresd.zarr"
    with (
        patch("foehn.api.to_zarr", return_value=store) as written,
        patch(
            "sys.argv",
            [
                "foehn",
                "to-zarr",
                "surface_derived_grid",
                "--match",
                "rhiresd",
                "--variables",
                "rhiresd",
                "--stack",
                "--data-dir",
                str(tmp_path),
            ],
        ),
    ):
        main()

    assert written.call_args.kwargs["stack"] is True
    assert written.call_args.kwargs["variables"] == ["rhiresd"]
    assert str(store) in capsys.readouterr().out


@pytest.mark.parametrize("transport", ["stdio", "streamable-http", "sse"])
def test_mcp_starts_the_server_on_the_requested_transport(transport):
    with (
        patch("foehn.mcp_server.run") as run,
        patch("sys.argv", ["foehn", "mcp", "--transport", transport]),
    ):
        main()

    run.assert_called_once_with(transport=transport)


def test_named_datasets_are_used_as_given(tmp_path):
    """An explicit list skips the default resolution — including the grid opt-out."""
    mocks = _run("to-parquet", ["smn", "radar_precip"], tmp_path)

    assert _converted(mocks) == ["smn", "radar_precip"]


@pytest.mark.parametrize(
    ("argv_stack", "expected"),
    [
        (["--stack"], True),
        (["--stack", "time"], True),
        (["--stack", "auto"], True),
        ([], False),
    ],
)
def test_stack_still_accepts_the_v0_4_spellings(tmp_path, argv_stack, expected):
    """v0.4.0 documented ``--stack time`` and ``--stack auto``.

    The kind decides how to cube now, so the value carries nothing — but turning
    the option into a bare flag made every documented command line fail outright
    with "unrecognized arguments: time" rather than ignoring a redundant word.
    """
    store = tmp_path / "zarr" / "radar_precip__cpc26130.zarr"
    with (
        patch("foehn.api.to_zarr", return_value=store) as written,
        patch(
            "sys.argv",
            ["foehn", "to-zarr", "radar_precip", "--match", "cpc26130", *argv_stack, "--data-dir", str(tmp_path)],
        ),
    ):
        main()

    assert written.call_args.kwargs["stack"] is expected


# --- Successive commands against one Workspace (no mocks below the CLI) ---


def test_successive_downloads_each_list_what_they_have_never_fetched(tmp_path, capsys):
    """A run of smn advanced one shared cursor, and pollen's first run then skipped it all."""
    from foehn.collections import COLLECTIONS
    from foehn.workspace import Workspace
    from tests.fakes import InMemoryFetcher, stac_item

    smn_recent = f"https://data.geo.admin.ch/{COLLECTIONS['smn']}/ogd-smn_ber_d_recent.csv"
    smn_historical = f"https://data.geo.admin.ch/{COLLECTIONS['smn']}/ogd-smn_ber_d_historical.csv"
    pollen_recent = f"https://data.geo.admin.ch/{COLLECTIONS['pollen']}/ogd-pollen_pbe_d_recent.csv"
    fake = InMemoryFetcher(
        items={
            COLLECTIONS["smn"]: [stac_item("ber", smn_recent, smn_historical, updated="2026-01-01T00:00:00Z")],
            COLLECTIONS["pollen"]: [stac_item("pbe", pollen_recent, updated="2026-01-01T00:00:00Z")],
        }
    )
    fake.any_collection = {"assets": {}}
    fake.default_body = b"station_abbr;reference_timestamp;x\nBER;01.01.2026 00:00;1\n"
    workspace = Workspace(tmp_path)

    def download(*args):
        argv = ["foehn", "download", "--no-parquet", "--data-dir", str(tmp_path), *args]
        with patch("foehn.cli.default_fetcher", return_value=fake), patch("sys.argv", argv):
            main()

    download("smn")
    download("pollen")
    download("--historical", "smn")

    assert (workspace.bronze("pollen") / "ogd-pollen_pbe_d_recent.csv").exists()
    assert (workspace.bronze("smn") / "ogd-smn_ber_d_historical.csv").exists()


# --- Output conventions: data on stdout, diagnostics on stderr ---


def _main(argv, capsys):
    """Run ``foehn <argv>`` and return (exit status, stdout, stderr)."""
    code = 0
    try:
        main(argv)
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_load_preview_shows_every_requested_row(capsys):
    """Polars cuts any frame to ten rows by default, so -n 30 printed ten and an ellipsis."""
    fake_df = pl.DataFrame({"a": range(100)})
    with patch("foehn.api.load", return_value=fake_df):
        code, out, _ = _main(["load", "smn", "-n", "30"], capsys)

    assert code == 0
    assert "…" not in out
    assert "│ 29  │" in out
    assert "│ 30  │" not in out


def test_load_preview_of_zero_rows_shows_none(capsys):
    """``-n 0`` meant 20: the default was applied with ``or``, which 0 fails."""
    fake_df = pl.DataFrame({"a": [101, 102, 103]})
    with patch("foehn.api.load", return_value=fake_df):
        code, out, _ = _main(["load", "smn", "-n", "0"], capsys)

    assert code == 0
    assert "101" not in out
    assert "[3 rows x 1 columns]" in out


@pytest.mark.parametrize(
    ("output_format", "expected"),
    [
        ("csv", "a,b\n1,x\n2,y\n"),
        ("ndjson", '{"a":1,"b":"x"}\n{"a":2,"b":"y"}\n'),
    ],
)
def test_load_writes_every_row_in_a_machine_format(capsys, output_format, expected):
    fake_df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    with patch("foehn.api.load", return_value=fake_df):
        code, out, _ = _main(["load", "smn", "--output-format", output_format], capsys)

    assert code == 0
    assert out == expected


def test_load_json_is_one_array_of_rows(capsys):
    import json

    fake_df = pl.DataFrame({"a": [1, 2]})
    with patch("foehn.api.load", return_value=fake_df):
        _, out, _ = _main(["load", "smn", "--output-format", "json"], capsys)

    assert json.loads(out) == [{"a": 1}, {"a": 2}]


@pytest.mark.parametrize("suffix", [".parquet", ".csv", ".json", ".ndjson", ".jsonl"])
def test_load_output_writes_the_format_its_suffix_names(tmp_path, capsys, suffix):
    fake_df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    target = tmp_path / "nested" / f"out{suffix}"
    with patch("foehn.api.load", return_value=fake_df):
        code, out, err = _main(["load", "smn", "-o", str(target)], capsys)

    assert code == 0
    assert out == ""
    assert "Wrote 2 rows x 2 columns" in err
    readers = {
        ".parquet": pl.read_parquet,
        ".csv": pl.read_csv,
        ".json": pl.read_json,
        ".ndjson": pl.read_ndjson,
        ".jsonl": pl.read_ndjson,
    }
    assert readers[suffix](target).equals(fake_df)


def test_load_output_with_an_unknown_suffix_names_the_ones_it_knows(tmp_path, capsys):
    with patch("foehn.api.load", return_value=pl.DataFrame({"a": [1]})):
        code, _, err = _main(["load", "smn", "-o", str(tmp_path / "out.xlsx")], capsys)

    assert code == 1
    assert "foehn: error: cannot tell which format" in err
    assert "hint: end the path in one of .parquet" in err
    assert not (tmp_path / "out.xlsx").exists()


@pytest.mark.parametrize("output_format", ["csv", "json", "ndjson"])
def test_metadata_writes_a_machine_format(capsys, output_format):
    fake_df = pl.DataFrame({"shortname": ["tre200d0"], "unit": ["°C"]})
    with patch("foehn.cli.metadata", return_value=fake_df):
        code, out, _ = _main(["metadata", "parameters", "smn", "--output-format", output_format], capsys)

    assert code == 0
    assert "tre200d0" in out
    assert "°C" in out
    assert "─" not in out  # no table rule


def test_list_json_is_the_list_datasets_rows(capsys):
    import json

    import foehn

    code, out, _ = _main(["list", "--output-format", "json"], capsys)

    assert code == 0
    assert json.loads(out) == foehn.list_datasets()


def test_list_csv_flattens_the_list_columns(capsys):
    code, out, _ = _main(["list", "--category", "A", "--output-format", "csv"], capsys)

    assert code == 0
    header, *rows = out.splitlines()
    assert header.startswith("dataset,collection_id,")
    smn = next(row for row in rows if row.startswith("smn,"))
    assert '"t,h,d,m,y"' in smn


def test_the_librarys_log_goes_to_stderr_never_stdout(tmp_path, capsys):
    """stdout is the command's output — and, under `foehn mcp`, the JSON-RPC stream.

    The CLI's handler wrote the library's log to stdout, so any download behind
    an MCP tool call (describe_grid opens a grid, which fetches its file) put a
    progress line into the protocol channel.
    """
    import logging

    def download(*_args, **_kwargs):
        logging.getLogger("foehn.transfer").info("  [1/1] Downloaded: x.csv")
        logging.getLogger("foehn.transfer").warning("  Failed: y.csv")
        return type("Result", (), {"failed": 0})()

    with patch("foehn.registry.download", side_effect=download), patch("foehn.registry.convert", return_value=0):
        code, out, err = _main(["download", "smn", "--data-dir", str(tmp_path)], capsys)

    assert code == 0
    assert out == ""
    assert "[1/1] Downloaded: x.csv" in err
    assert "warning:   Failed: y.csv" in err


def test_a_logged_error_is_labelled_and_survives_double_quiet(tmp_path, capsys):
    import logging

    def download(*_args, **_kwargs):
        logging.getLogger("foehn.transfer").error("disk full")
        return type("Result", (), {"failed": 0})()

    with patch("foehn.registry.download", side_effect=download), patch("foehn.registry.convert", return_value=0):
        _, out, err = _main(["download", "smn", "-qq", "--data-dir", str(tmp_path)], capsys)

    assert out == ""
    assert err == "error: disk full\n"


def test_mcp_logs_to_stderr(capsys):
    import logging

    def serve(transport):
        logging.getLogger("foehn.grids").info("downloading a grid file")

    with patch("foehn.mcp_server.run", side_effect=serve):
        code, out, err = _main(["mcp"], capsys)

    assert code == 0
    assert out == ""
    assert "downloading a grid file" in err


@pytest.mark.parametrize("argv", [["-q", "download", "smn"], ["download", "smn", "--quiet"]])
def test_quiet_silences_progress_before_or_after_the_subcommand(tmp_path, capsys, argv):
    import logging

    def download(*_args, **_kwargs):
        logging.getLogger("foehn.transfer").info("  [1/1] Downloaded: x.csv")
        return type("Result", (), {"failed": 0})()

    with patch("foehn.registry.download", side_effect=download), patch("foehn.registry.convert", return_value=0):
        code, out, err = _main([*argv, "--data-dir", str(tmp_path)], capsys)

    assert code == 0
    assert out == err == ""


def test_double_quiet_silences_warnings(tmp_path, capsys):
    mocks, patchers = _start_mocks()
    mocks["download"].return_value.failed = 1
    try:
        code, _, err = _main(["download", "smn", "-qq", "--data-dir", str(tmp_path)], capsys)
    finally:
        for p in patchers:
            p.stop()

    assert code == 1
    assert err == ""


def test_verbose_shows_the_librarys_debug_log(tmp_path, capsys):
    import logging

    def download(*_args, **_kwargs):
        logging.getLogger("foehn.fetch").debug("GET https://example")
        return type("Result", (), {"failed": 0})()

    with patch("foehn.registry.download", side_effect=download), patch("foehn.registry.convert", return_value=0):
        _, _, quiet_err = _main(["download", "smn", "--data-dir", str(tmp_path)], capsys)
        _, _, verbose_err = _main(["-v", "download", "smn", "--data-dir", str(tmp_path)], capsys)

    assert "GET https://example" not in quiet_err
    assert "GET https://example" in verbose_err


# --- Errors ---


@pytest.mark.parametrize("flag", ["--workers", "-n", "--limit"])
def test_a_bad_count_is_a_usage_error_before_anything_runs(tmp_path, capsys, flag):
    """--workers 0 used to print a header, create the Workspace, then fail deep in the run."""
    value = "0" if flag == "--workers" else "-1"
    workspace = tmp_path / "ws"
    argv = (
        ["download", "smn", "--workers", value, "--data-dir", str(workspace)]
        if flag == "--workers"
        else ["load", "smn", flag, value]
    )
    with patch("foehn.api.load") as load:
        code, out, err = _main(argv, capsys)

    assert code == 2
    assert out == ""
    assert flag in err
    load.assert_not_called()
    assert not workspace.exists()


def test_an_unknown_dataset_suggests_the_closest_one(tmp_path, capsys):
    code, _, err = _main(["download", "smm", "--data-dir", str(tmp_path / "ws")], capsys)

    assert code == 1
    assert err.splitlines() == ["foehn: error: unknown dataset 'smm'", "  hint: did you mean 'smn'?"]
    assert not (tmp_path / "ws").exists()


def test_an_unknown_dataset_with_nothing_close_points_at_foehn_list(tmp_path, capsys):
    _, _, err = _main(["to-parquet", "bogus", "--data-dir", str(tmp_path)], capsys)

    assert "hint: run 'foehn list'" in err


def test_a_closed_stdout_ends_quietly(capsys):
    """`foehn metadata inventory smn | head` died with a BrokenPipeError traceback."""
    with patch("foehn.cli.metadata", side_effect=BrokenPipeError):
        code, _, err = _main(["metadata", "inventory", "smn"], capsys)

    assert code == 141
    assert err == ""


def test_interrupt_exits_130(capsys):
    with patch("foehn.cli.metadata", side_effect=KeyboardInterrupt):
        code, _, _ = _main(["metadata", "inventory", "smn"], capsys)

    assert code == 130


@pytest.mark.parametrize(
    ("env", "argv", "colored"),
    [
        ({}, [], False),  # captured stderr is not a terminal
        ({"FORCE_COLOR": "1"}, [], True),
        ({"FORCE_COLOR": "1", "NO_COLOR": "1"}, [], False),
        ({"NO_COLOR": "1"}, ["--color", "always"], True),
        ({"FORCE_COLOR": "1"}, ["--color", "never"], False),
    ],
)
def test_error_colour_follows_the_flag_then_the_environment(capsys, monkeypatch, env, argv, colored):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    _, _, err = _main([*argv, "load", "bogus"], capsys)

    assert ("\033[" in err) is colored


# --- version ---


def test_version_subcommand_prints_the_report(capsys):
    import foehn

    code, out, _ = _main(["version"], capsys)

    assert code == 0
    assert f"foehn:       {foehn.__version__}" in out
    assert "polars:" in out


def test_version_subcommand_json(capsys):
    import json

    import foehn

    code, out, _ = _main(["version", "--output-format", "json"], capsys)

    report = json.loads(out)
    assert code == 0
    assert report["foehn"] == foehn.__version__
    assert report["dependencies"]["polars"] == pl.__version__


def test_short_version_flag(capsys):
    import foehn

    code, out, _ = _main(["-V"], capsys)

    assert code == 0
    assert out.strip() == f"foehn {foehn.__version__}"


def test_python_dash_m_runs_the_cli():
    import subprocess
    import sys

    import foehn

    # S603: a fixed argument list running this interpreter, not input.
    result = subprocess.run([sys.executable, "-m", "foehn", "--version"], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == f"foehn {foehn.__version__}"
