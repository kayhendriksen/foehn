"""The pinned dev toolchain is declared once and read from there.

CI resolves ruff and zizmor out of .pre-commit-config.yaml via
scripts/hook_version.sh, so if that parse breaks the lint and zizmor jobs break
with it. These tests guard the contract from the repo side, where a broken
config shows up before it reaches CI.
"""

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PRE_COMMIT = ROOT / ".pre-commit-config.yaml"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _hook_version(repo: str) -> str:
    # S603: the command and its argument are repo-local literals, not input.
    result = subprocess.run(  # noqa: S603
        [str(ROOT / "scripts" / "hook_version.sh"), repo],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    key, _, value = result.stdout.strip().partition("=")
    assert key == "version"
    return value


@pytest.mark.parametrize("repo", ["ruff-pre-commit", "zizmor-pre-commit"])
def test_hook_version_resolves_a_version(repo):
    assert re.fullmatch(r"\d+\.\d+\.\d+", _hook_version(repo)), f"unparseable rev for {repo}"


@pytest.mark.parametrize("repo", ["ruff-pre-commit", "zizmor-pre-commit"])
def test_hook_version_matches_the_config(repo):
    """The script must report the rev actually pinned, not a stale or nearby one."""
    config = PRE_COMMIT.read_text()
    block = config.split(f"/{repo}\n", 1)[1]
    expected = re.search(r"rev:\s*v?(\S+)", block).group(1)
    assert _hook_version(repo) == expected


def test_hook_version_fails_on_an_unknown_repo():
    """A silent empty version would make CI run `uvx ruff@` and fail obscurely."""
    result = subprocess.run(  # noqa: S603
        [str(ROOT / "scripts" / "hook_version.sh"), "not-a-repo"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode != 0
    assert "no rev found" in result.stderr


def test_workflow_hardcodes_no_shared_tool_version():
    """ruff, ty and zizmor are declared where Dependabot bumps them, not here.

    Scoped to the three that are pinned in two places — .pre-commit-config.yaml
    (ruff, zizmor) and the pyproject dev extra (ty) — because a second copy in
    ci.yml is what silently drifts out of step. The workflow's other pins
    (pip-audit, build, twine, bandit) live only here and so cannot disagree with
    anything; they are stale-prone, not drift-prone.

    Checks for a literal rather than for particular variable names: the names are
    fine (the jobs pass the resolved version through ``env:`` to keep it out of
    the run block), it is a hardcoded value that goes stale.
    """
    workflow = WORKFLOW.read_text()
    for tool in ("RUFF", "TY", "ZIZMOR"):
        hardcoded = re.findall(rf"^\s*{tool}_\w*:\s*\"?\d+\.\d+\.\d+", workflow, re.MULTILINE)
        assert not hardcoded, f"ci.yml hardcodes {tool}; it will drift from its source of truth"


def test_ty_is_pinned_exactly():
    """A floor would let a pre-1.0 bump land on an unrelated PR."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    ty = next(d for d in dev if d.startswith("ty"))
    assert ty.startswith("ty=="), f"expected an exact pin, got {ty!r}"
