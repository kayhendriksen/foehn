"""docs/cli.md documents exactly the command line ``foehn.cli.build_parser`` builds.

The reference is hand-written, for its prose, so nothing regenerates it when a
flag is added — which is how ``foehn load --limit`` and ``--workers`` went
undocumented. uv checks its generated CLI reference against clap in CI; this is
the same contract for a hand-written page: every subcommand has a section, and
every flag it accepts appears in that section.
"""

import argparse
import re
from pathlib import Path

import pytest

from foehn.cli import FULL_REFRESH_ENV, build_parser
from foehn.workspace import DATA_DIR_ENV

DOCS = Path(__file__).resolve().parent.parent / "docs" / "cli.md"

# Accepted by every subcommand, and documented once under "Global options".
GLOBAL_FLAGS = {"-h", "--help", "-q", "--quiet", "-v", "--verbose", "--color"}


def _subcommands() -> dict[str, argparse.ArgumentParser]:
    parser = build_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return dict(action.choices)


def _sections() -> dict[str, str]:
    """Each ``## `foehn <command> ...` `` heading's command, mapped to its body."""
    text = DOCS.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    for match in re.finditer(r"^## `foehn ([a-z-]+)[^`]*`\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL):
        sections[match.group(1)] = match.group(2)
    return sections


def test_every_subcommand_has_a_section():
    assert set(_sections()) == set(_subcommands())


@pytest.mark.parametrize("command", sorted(_subcommands()))
def test_every_flag_is_documented_in_its_section(command):
    section = _sections().get(command, "")
    flags = {
        option
        for action in _subcommands()[command]._actions
        for option in action.option_strings
        if option not in GLOBAL_FLAGS
    }
    missing = sorted(flag for flag in flags if f"`{flag}" not in section)

    assert not missing, f"docs/cli.md's `foehn {command}` section does not mention {missing}"


def test_the_global_options_are_documented():
    text = DOCS.read_text(encoding="utf-8")
    for flag in GLOBAL_FLAGS - {"-h", "--help"}:
        assert f"`{flag}" in text


@pytest.mark.parametrize("variable", [DATA_DIR_ENV, FULL_REFRESH_ENV, "NO_COLOR", "FORCE_COLOR"])
def test_every_environment_variable_is_documented(variable):
    assert f"`{variable}`" in DOCS.read_text(encoding="utf-8")
