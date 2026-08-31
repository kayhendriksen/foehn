"""That a public docstring never restates a vocabulary the code already holds.

``help(foehn.load)`` and an MCP tool's description are what a caller — human or
model — actually reads, so they are interfaces. A token list typed into one can
only agree with the table that defines it or drift from it, and prose cannot be
checked against a set. Both had drifted before: the MCP guide claimed ``sort``
defaults to "asc" when an omitted ``sort`` does not sort at all, and
``load_data`` named twelve loadable datasets when the registry had thirteen.

These tests are what makes the rendering load-bearing: add a granularity to
``GRANULARITY_LABELS`` and every surface that offers one has to show it.
"""

from __future__ import annotations

import inspect

import pytest

import foehn
from foehn.collections import (
    CATEGORY_LABELS,
    DEFAULT_TIME_SLICE,
    GRANULARITY_LABELS,
    TIME_SLICE_LABELS,
    options,
)
from foehn.fetch import DEFAULT_WORKERS

mcp_server = pytest.importorskip("foehn.mcp_server", reason="mcp not installed")


def _doc(fn) -> str:
    doc = inspect.getdoc(fn)
    assert doc, f"{fn.__name__} has no docstring to render into"
    return doc


# The surfaces that offer a filter, and which vocabularies each one offers.
_OFFERS = [
    (foehn.load, (GRANULARITY_LABELS, TIME_SLICE_LABELS)),
    (foehn.download, (TIME_SLICE_LABELS,)),
    (mcp_server.load_data, (GRANULARITY_LABELS, TIME_SLICE_LABELS)),
    (mcp_server.describe_data, (GRANULARITY_LABELS, TIME_SLICE_LABELS)),
    (mcp_server.list_datasets, (CATEGORY_LABELS,)),
]


@pytest.mark.parametrize(("fn", "tables"), _OFFERS, ids=lambda v: getattr(v, "__name__", ""))
def test_every_offered_token_reaches_the_docstring(fn, tables):
    """A token added to a table has to show up wherever that filter is offered."""
    doc = _doc(fn)
    for table in tables:
        assert options(table) in doc, f"{fn.__name__} does not render {sorted(table)}"


@pytest.mark.parametrize("fn", [pair[0] for pair in _OFFERS], ids=lambda v: v.__name__)
def test_no_placeholder_survives_into_a_public_docstring(fn):
    """An unfilled ``$name`` is a fragment key that was renamed or misspelled."""
    assert "$" not in _doc(fn)


@pytest.mark.parametrize("fn", [foehn.load, foehn.download, mcp_server.load_data], ids=lambda v: v.__name__)
def test_the_documented_time_slice_default_is_the_real_one(fn):
    assert DEFAULT_TIME_SLICE in _doc(fn)


@pytest.mark.parametrize("fn", [foehn.load, foehn.download], ids=lambda v: v.__name__)
def test_the_documented_worker_count_is_the_signature_default(fn):
    """The one number a caller plans around, and it was typed out twice."""
    assert inspect.signature(fn).parameters["workers"].default == DEFAULT_WORKERS
    assert f"default {DEFAULT_WORKERS}" in _doc(fn)
