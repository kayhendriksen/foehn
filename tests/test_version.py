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


def test_server_json_launches_the_mcp_subcommand_with_its_extra():
    """What a registry client runs: ``uvx --with=foehn[mcp]==<v> foehn mcp``.

    ``mcp`` is an argument to foehn, not to uvx, so it belongs in
    ``packageArguments`` with a ``value`` (a ``valueHint`` alone is only a label
    and never reaches the command line). And a bare ``uvx foehn`` installs no
    MCP SDK, so the extra is pulled in at the same version.
    """
    (package,) = json.loads((ROOT / "server.json").read_text())["packages"]

    assert package["runtimeHint"] == "uvx"
    assert package["packageArguments"] == [{"type": "positional", "value": "mcp"}]
    with_extra = [arg for arg in package["runtimeArguments"] if arg.get("name") == "--with"]
    assert [arg["value"] for arg in with_extra] == [f"foehn[mcp]=={_pyproject_version()}"]
