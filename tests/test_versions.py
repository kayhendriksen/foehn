"""The environment report behind ``foehn.show_versions()`` and ``foehn version``."""

import json
import re
import tomllib
from pathlib import Path

import polars as pl

import foehn
from foehn.versions import DEPENDENCIES, format_versions, versions

ROOT = Path(__file__).resolve().parent.parent


def _distribution(requirement: str) -> str:
    return re.split(r"[\s\[<>=!~;]", requirement, maxsplit=1)[0]


def test_every_declared_dependency_is_reported():
    """A dependency added to pyproject.toml and not here would be missing from every bug report."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = {_distribution(r) for r in project["dependencies"]}
    for extra in project["optional-dependencies"].values():
        declared |= {_distribution(r) for r in extra}

    assert declared <= set(DEPENDENCIES)


def test_versions_reports_installed_and_missing_dependencies():
    info = versions()

    assert info["foehn"] == foehn.__version__
    assert info["dependencies"]["polars"] == pl.__version__
    # dask is optional and not part of the test environment.
    assert info["dependencies"]["dask"] is None
    json.dumps(info)  # serialisable as-is, for `foehn version --output-format json`


def test_the_data_dir_reports_where_it_came_from(tmp_path, monkeypatch):
    monkeypatch.setenv("FOEHN_DATA_DIR", str(tmp_path))

    info = versions()

    assert info["data_dir"] == str(tmp_path.resolve())
    assert info["data_dir_source"] == "$FOEHN_DATA_DIR"


def test_show_versions_prints_the_report(capsys):
    foehn.show_versions()

    out = capsys.readouterr().out
    assert out == format_versions() + "\n"
    assert "dask:" in out
    assert "<not installed>" in out
