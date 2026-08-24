"""Tests that the declared version is consistent across the repo."""

import json
import tomllib
from pathlib import Path

import foehn

ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def test_dunder_version_matches_pyproject():
    """__version__ is not checked by the publish workflow, so it can drift silently."""
    assert foehn.__version__ == _pyproject_version()


def test_server_json_versions_match_pyproject():
    """The publish workflow rewrites server.json from the tag; keep the committed
    copy in sync so a local read never reports a stale version."""
    server = json.loads((ROOT / "server.json").read_text())
    expected = _pyproject_version()
    assert server["version"] == expected
    assert [p["version"] for p in server["packages"]] == [expected]
