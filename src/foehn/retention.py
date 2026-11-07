"""Evidence that a rolling CSV output has disappeared upstream.

The full listing is recorded only after a successful refresh, before pruning
Bronze. Conversion and Delta ingestion can then reconcile a completely absent
group, including one pruned by an older foehn. Missing local files alone never
authorize removing an output. Historical and unsliced groups are never retired.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from foehn.atomicwrite import write_text
from foehn.collections import GRANULARITY_LABELS, ROLLING_TIME_SLICES
from foehn.meteocsv import group_csv_files, group_csv_paths

_LISTING = ".foehn-rolling-listing.json"


def record_listing(bronze: Path, names: list[str]) -> None:
    """Record a successful, nonempty full listing while holding the refresh lock."""
    if names:
        write_text(bronze / _LISTING, json.dumps({"version": 1, "files": sorted(names)}), durable=True)


def retired_groups(bronze: Path, dataset: str, candidates: Iterable[tuple[str, ...]]) -> set[tuple[str, ...]]:
    """Rolling candidates absent both upstream and locally, or none without evidence."""
    try:
        listing = json.loads((bronze / _LISTING).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return set()
    if not isinstance(listing, dict) or listing.get("version") != 1:
        return set()
    names = listing.get("files")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and Path(n).name == n for n in names):
        return set()
    active = set(group_csv_paths([Path(name) for name in names], dataset)) | set(group_csv_files(bronze, dataset))
    return {
        group
        for group in candidates
        if len(group) == 2
        and group[0] in GRANULARITY_LABELS
        and group[1] in ROLLING_TIME_SLICES
        and group not in active
    }
