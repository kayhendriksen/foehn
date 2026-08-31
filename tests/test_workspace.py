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

    assert store == Workspace(tmp_path).zarr("surface_derived_grid__rhiresd")
    assert mock.call_args.args[1] == store
    assert mock.call_args.kwargs["workspace"] == Workspace(tmp_path)
