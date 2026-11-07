"""Evidence that a rolling CSV output has disappeared upstream.

Each successful refresh records its full listing before pruning Bronze, and notes
when a file it stopped listing first went missing. Conversion and Delta ingestion
retire a group only when it is gone locally and a later listing shows it has
stayed unlisted for :data:`RETIREMENT_GRACE`, so an incomplete listing cannot
delete an output. A group an older foehn pruned has no such note, and waits the
same period from the first recorded listing. Missing local files alone never
authorize removing an output. Historical and unsliced groups are never retired.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from foehn.atomicwrite import write_text
from foehn.collections import GRANULARITY_LABELS, ROLLING_TIME_SLICES
from foehn.meteocsv import group_csv_files, group_csv_paths
from foehn.state import parse_instant

_LISTING = ".foehn-rolling-listing.json"

RETIREMENT_GRACE = timedelta(days=7)
"""How long a group must stay unlisted before its outputs are retired.

Time rather than a count of listings: the near-real-time job lists every five
minutes, so any small count fits inside one upstream incident. Waiting costs a
stale table for a week; retiring early empties one that is still published.
"""


@dataclass(frozen=True)
class _Evidence:
    files: list[str]
    recorded_at: datetime
    listed_since: datetime
    absent_since: dict[str, datetime]


def _read(bronze: Path) -> _Evidence | None:
    """The recorded evidence, or None for anything short of a complete record."""
    try:
        recorded = json.loads((bronze / _LISTING).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(recorded, dict) or recorded.get("version") != 1:
        return None
    files = recorded.get("files")
    absent = recorded.get("absent_since")
    if not isinstance(files, list) or not files or not isinstance(absent, dict):
        return None
    if not all(isinstance(name, str) and Path(name).name == name for name in [*files, *absent]):
        return None
    recorded_at = parse_instant(recorded.get("recorded_at"))
    listed_since = parse_instant(recorded.get("listed_since"))
    stamps = {name: stamp for name, value in absent.items() if (stamp := parse_instant(value)) is not None}
    if recorded_at is None or listed_since is None or len(stamps) != len(absent):
        return None
    return _Evidence(files, recorded_at, listed_since, stamps)


def record_listing(bronze: Path, names: list[str], *, now: datetime | None = None) -> None:
    """Record a successful, nonempty full listing while holding the refresh lock.

    Unreadable earlier evidence restarts every wait rather than shortening one.
    """
    if not names:
        return
    now = now or datetime.now(UTC)
    listed = set(names)
    earlier = _read(bronze)
    absent: dict[str, datetime] = {}
    if earlier is not None:
        absent = {name: now for name in earlier.files if name not in listed}
        absent |= {name: stamp for name, stamp in earlier.absent_since.items() if name not in listed}
    record = {
        "version": 1,
        "recorded_at": now.isoformat(),
        "listed_since": (now if earlier is None else earlier.listed_since).isoformat(),
        "files": sorted(listed),
        "absent_since": {name: stamp.isoformat() for name, stamp in sorted(absent.items())},
    }
    write_text(bronze / _LISTING, json.dumps(record), durable=True)


def retired_groups(bronze: Path, dataset: str, candidates: Iterable[tuple[str, ...]]) -> set[tuple[str, ...]]:
    """Rolling candidates gone locally and unlisted for the grace period, or none without evidence."""
    evidence = _read(bronze)
    if evidence is None:
        return set()
    active = set(group_csv_paths([Path(name) for name in evidence.files], dataset))
    active |= set(group_csv_files(bronze, dataset))
    unlisted = group_csv_paths([Path(name) for name in evidence.absent_since], dataset)
    # A group's newest note is when its last file went. No note means it was
    # already gone when the first listing was recorded here.
    gone = {group: max(evidence.absent_since[path.name] for path in paths) for group, paths in unlisted.items()}
    return {
        group
        for group in candidates
        if len(group) == 2
        and group[0] in GRANULARITY_LABELS
        and group[1] in ROLLING_TIME_SLICES
        and group not in active
        and evidence.recorded_at - gone.get(group, evidence.listed_since) >= RETIREMENT_GRACE
    }
