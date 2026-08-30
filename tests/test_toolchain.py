"""Every pinned tool is declared once, and the workflows resolve it from there.

The versions live in .pre-commit-config.yaml (tools that also run as local
hooks), requirements-ci.txt (tools only CI runs) and the pyproject dev extra
(ty) — three files Dependabot bumps. The workflows read them back with
scripts/tool_version.sh instead of repeating them. These tests guard that
contract from the repo side, where a break shows up before it reaches CI.
"""

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PRE_COMMIT = ROOT / ".pre-commit-config.yaml"
REQUIREMENTS = ROOT / "requirements-ci.txt"
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

HOOK_TOOLS = ["ruff-pre-commit", "zizmor-pre-commit"]
CI_TOOLS = ["bandit", "build", "cyclonedx-bom", "pip-audit", "twine"]


def _tool_version(name: str) -> str:
    # S603: the command and its argument are repo-local literals, not input.
    result = subprocess.run(  # noqa: S603
        [str(ROOT / "scripts" / "tool_version.sh"), name],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    key, _, value = result.stdout.strip().partition("=")
    assert key == "version"
    return value


@pytest.mark.parametrize("name", HOOK_TOOLS + CI_TOOLS)
def test_resolves_a_version(name):
    assert re.fullmatch(r"\d+\.\d+\.\d+", _tool_version(name)), f"unparseable pin for {name}"


@pytest.mark.parametrize("name", HOOK_TOOLS)
def test_hook_version_matches_the_pre_commit_config(name):
    """Must report the rev actually pinned, not a stale or neighbouring one."""
    block = PRE_COMMIT.read_text().split(f"/{name}\n", 1)[1]
    expected = re.search(r"rev:\s*v?(\S+)", block).group(1)

    assert _tool_version(name) == expected


@pytest.mark.parametrize("name", CI_TOOLS)
def test_ci_tool_version_matches_requirements(name):
    expected = re.search(rf"^{re.escape(name)}==(\S+)$", REQUIREMENTS.read_text(), re.MULTILINE).group(1)

    assert _tool_version(name) == expected


def test_unknown_tool_fails_loudly():
    """A silent empty version would make CI run `uvx ruff@` and fail obscurely."""
    result = subprocess.run(  # noqa: S603
        [str(ROOT / "scripts" / "tool_version.sh"), "not-a-tool"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode != 0
    assert "no pin found" in result.stderr


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_hardcodes_no_pinned_tool_version(workflow):
    """A second copy of a version in a workflow is what silently goes stale.

    Matches the tool name against a literal version, in either the `tool@1.2.3`
    or `tool==1.2.3` form, so it catches a reintroduced pin however it is
    spelled. Names are fine — the jobs pass resolved versions through `env:` to
    keep them out of the run block — it is a hardcoded value that drifts.
    """
    text = workflow.read_text()
    for name in HOOK_TOOLS + CI_TOOLS:
        tool = name.removesuffix("-pre-commit")
        hardcoded = re.findall(rf"{re.escape(tool)}[@=]=?\d+\.\d+\.\d+", text)
        assert not hardcoded, f"{workflow.name} hardcodes {hardcoded}; it will drift from its pin"


def test_every_ci_pin_is_actually_used():
    """A pin nothing resolves is one Dependabot keeps bumping for no reason."""
    workflows = "\n".join(w.read_text() for w in WORKFLOWS)
    unused = [name for name in CI_TOOLS if f"tool_version.sh {name}" not in workflows]

    assert not unused, f"requirements-ci.txt pins {unused}, which no workflow resolves"


def test_the_two_sources_do_not_overlap():
    """A tool pinned in both files could be resolved from either, and drift."""
    hook_names = {n.removesuffix("-pre-commit") for n in HOOK_TOOLS}

    assert not hook_names & set(CI_TOOLS)


def test_ty_is_pinned_exactly():
    """A floor would let a pre-1.0 bump land on an unrelated PR."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev = pyproject["project"]["optional-dependencies"]["dev"]
    ty = next(d for d in dev if d.startswith("ty"))

    assert ty.startswith("ty=="), f"expected an exact pin, got {ty!r}"


def test_workflow_yaml_structure_survives_edits():
    """Regression: removing entries from ci.yml's `env:` once took the key with
    them, reparenting the rest under `concurrency:`. That is valid YAML and
    passes a schema check, so both slipped through — and GitHub refused to start
    the workflow. actionlint (a pre-commit hook) catches the general case.
    """
    yaml = pytest.importorskip("ruamel.yaml", reason="strict YAML parser not installed")

    for workflow in WORKFLOWS:
        loaded = yaml.YAML(typ="safe").load(workflow.read_text())
        assert "jobs" in loaded, f"{workflow.name} has no jobs — a key was probably swallowed"
        if "concurrency" in loaded:
            assert set(loaded["concurrency"]) <= {"group", "cancel-in-progress"}
