"""Filling a public docstring's placeholders from the tables it would otherwise retype.

A docstring is an interface: ``help(foehn.load)`` and an MCP tool's description
are what a caller — human or model — actually reads. So a vocabulary stated in
code and restated there can only agree with it or drift from it, and prose
cannot be checked against a set.

The MCP layer worked this out first and rendered its guide from the tables. This
is the same decorator, moved below both front ends so ``api`` can use it too:
``load()``'s docstring named the granularity and time-slice tokens as prose, and
so did two of the tool docstrings sitting under the comment saying they would
not.

Sits under everything and imports nothing of foehn's — it knows about text, not
about datasets.
"""

from __future__ import annotations

from string import Template
from typing import TypeVar

F = TypeVar("F")


def renders(**fragments: str):
    """Fill a docstring's ``$name`` placeholders before anything reads it.

    Spelled ``$name`` rather than ``{name}`` so a docstring stays free to contain
    a brace. Where a registration decorator is involved this must sit *below* it,
    so the finished text is what registers.
    """

    def apply(fn: F) -> F:
        if fn.__doc__:  # python -OO strips docstrings
            fn.__doc__ = Template(fn.__doc__).safe_substitute(**fragments)
        return fn

    return apply


__all__ = ["renders"]
