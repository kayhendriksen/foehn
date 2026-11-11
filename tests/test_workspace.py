"""Tests for the workspace: one root, one resolution rule, one layout.

The bug this module exists to remove was not that the paths were wrong but that
there were several of them: ``Path.cwd() / "data" / "meteoswiss"`` at seven call
sites, only one of which read ``$FOEHN_DATA_DIR``.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import foehn
from foehn.workspace import DATA_DIR_ENV, Workspace


def test_explicit_data_dir_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, "/env/dir")
    assert Workspace.resolve(tmp_path).root == tmp_path


def test_environment_is_used_when_no_argument(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert Workspace.resolve().root == tmp_path


def test_empty_environment_falls_back_to_the_default(tmp_path, monkeypatch):
    """An unset variable and an empty one mean the same thing."""
    monkeypatch.setenv(DATA_DIR_ENV, "")
    monkeypatch.chdir(tmp_path)
    assert Workspace.resolve().root == tmp_path / "data" / "meteoswiss"


def test_default_is_relative_to_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert Workspace.resolve().root == tmp_path / "data" / "meteoswiss"


def test_a_string_data_dir_is_accepted():
    assert Workspace.resolve("/some/where").root == Path("/some/where")


def test_a_leading_tilde_means_the_home_directory(monkeypatch):
    """--data-dir ~/... and $FOEHN_DATA_DIR must not create a literal '~' directory."""
    monkeypatch.setenv(DATA_DIR_ENV, "~/somedata-xyz")
    assert Workspace.resolve().root == Path.home() / "somedata-xyz"
    assert Workspace.resolve("~/somedata-xyz").root == Path.home() / "somedata-xyz"


def test_equivalent_spellings_resolve_to_the_same_root(tmp_path):
    """Two names for one directory must produce one workspace.

    Every ownership and cache lock foehn takes is filed under a path derived
    from this root; two spellings that stayed distinct here would carry that
    all the way down to two locks on the one real directory.
    """
    spelled_differently = tmp_path / "sub" / ".."
    assert Workspace.resolve(spelled_differently).root == Workspace.resolve(tmp_path).root


def test_layout(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.bronze() == tmp_path / "bronze"
    assert ws.bronze("smn") == tmp_path / "bronze" / "smn"
    assert ws.parquet() == tmp_path / "parquet"
    assert ws.parquet("smn") == tmp_path / "parquet" / "smn"
    assert ws.zarr("smn__d") == tmp_path / "zarr" / "smn__d.zarr"
    assert ws.etags == tmp_path / "_etags.json"
    assert ws.last_run == tmp_path / "_last_run.json"


def test_workspaces_compare_by_root(tmp_path):
    """Frozen and comparable, so a caller can assert on the one it passed."""
    assert Workspace(tmp_path) == Workspace(tmp_path)
    assert Workspace(tmp_path) != Workspace(tmp_path / "other")


# --- the seam: one rule for every entry point ---


@pytest.mark.parametrize(
    ("call", "target"),
    [
        (lambda: foehn.download("smn"), "foehn.registry.download"),
        (lambda: foehn.to_parquet("smn"), "foehn.registry.convert"),
    ],
)
def test_python_api_honours_the_environment_variable(call, target, tmp_path, monkeypatch):
    """The CLI used to be the only entry point that read it.

    With the variable set, ``foehn download`` wrote to it and ``foehn.download()``
    wrote to ./data/meteoswiss — the same environment, two directories.
    """
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    with patch(target) as mock:
        mock.return_value = 0
        call()
    assert mock.call_args.args[1] == Workspace(tmp_path)


def test_open_dataset_honours_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    with patch("foehn.registry.open_grid") as mock:
        foehn.open_dataset("surface_derived_grid")
    assert mock.call_args.kwargs["workspace"] == Workspace(tmp_path)


def test_to_zarr_store_lands_in_the_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    with patch("foehn.registry.write_zarr") as mock:
        store = foehn.to_zarr("surface_derived_grid", match="rhiresd")

    assert store == Workspace(tmp_path).zarr("surface_derived_grid", "rhiresd")
    assert mock.call_args.args[1] == store
    assert mock.call_args.kwargs["workspace"] == Workspace(tmp_path)


def test_an_explicit_store_expands_a_leading_tilde(tmp_path, monkeypatch):
    """``store="~/x.zarr"`` from Python never passes through a shell that would expand it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "data"))
    with patch("foehn.registry.write_zarr") as mock:
        store = foehn.to_zarr("surface_derived_grid", match="rhiresd", store="~/mine.zarr")

    assert store == tmp_path / "mine.zarr"
    assert mock.call_args.args[1] == store


def test_a_match_narrows_the_store_name(tmp_path):
    """Two filtered slices of one collection must not overwrite each other.

    ``api`` derived this name and handed the result to :meth:`Workspace.zarr` —
    the one path foehn spelled out above the seam rather than here, and the
    reason that method used to document the encoding as its caller's job.
    """
    workspace = Workspace(tmp_path)
    assert workspace.zarr("radar_precip") == tmp_path / "zarr" / "radar_precip.zarr"
    filtered = workspace.zarr("radar_precip", "cpc2613/00")
    assert filtered.parent == tmp_path / "zarr"
    assert filtered.name.startswith("radar_precip__cpc2613_00__")
    assert filtered.suffix == ".zarr"
    # A match of nothing but separators still has to name a directory.
    assert workspace.zarr("radar_precip", "//").name.startswith("radar_precip__match__")


def test_store_names_keep_the_exact_match_identity(tmp_path):
    workspace = Workspace(tmp_path)
    matches = ["tas", "_tas_", "tas-", "tas.", "TAS", "//", "__", ""]
    paths = [workspace.zarr("climate_scenarios_grid", match) for match in matches]
    assert len(set(paths)) == len(matches)
    assert len({path.name.lower() for path in paths}) == len(matches)
    assert workspace.zarr("climate_scenarios_grid", "tas") == paths[0]
    assert all(path.parent == tmp_path / "zarr" for path in paths)


def test_a_long_match_still_produces_a_usable_store_name(tmp_path):
    path = Workspace(tmp_path).zarr("climate_scenarios_grid", "tas" * 200)
    path.parent.mkdir()
    path.mkdir()
    assert path.is_dir()


@pytest.mark.parametrize(
    ("platform", "env", "expected"),
    [
        ("darwin", {}, Path("~/Library/Caches/foehn")),
        ("linux", {}, Path("~/.cache/foehn")),
        ("linux", {"XDG_CACHE_HOME": "/xdg"}, Path("/xdg/foehn")),
        ("win32", {"LOCALAPPDATA": "C:/Users/me/AppData/Local"}, Path("C:/Users/me/AppData/Local/foehn/Cache")),
    ],
)
def test_the_user_cache_root_follows_each_platforms_convention(monkeypatch, platform, env, expected):
    from foehn.workspace import user_cache_root

    monkeypatch.setattr("foehn.workspace.sys.platform", platform)
    for name in ("XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert user_cache_root() == expected.expanduser()
