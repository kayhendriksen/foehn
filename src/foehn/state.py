"""What foehn remembers between runs: the ETag store and the incremental cursors.

Two small JSON files whose *location* is the :class:`~foehn.workspace.Workspace`'s
and whose *contents* are the download path's. They lived in the download module,
which is why its docstring described two of the four things it held.

Both reads are total: a corrupt or unreadable file is reported and treated as
absent, because a lost cursor costs one redundant download and a raised
exception costs the whole run.

The cursor is *scoped*: one instant per Dataset and Time slice, under keys like
``smn/recent`` in ``_last_run.json``. A single Workspace-wide instant used to
gate every Dataset, so downloading ``smn`` and then ``pollen`` filtered pollen's
listing against smn's run and skipped Items that had never been fetched; adding
``--historical`` skipped the historical Assets the same way. The pre-scoping
``timestamp`` key is still read and written by :func:`load_last_run` and
:func:`save_last_run` for compatibility, and gates nothing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from foehn._locking import exclusive_lock
from foehn.atomicwrite import PRIVATE_FILE_MODE, write_text
from foehn.workspace import Workspace

logger = logging.getLogger(__name__)


def _read_mapping(path: Path, label: str) -> dict[str, str]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            logger.warning("Could not read %s %s (%s) — treating as absent", label, path, exc)
            return {}
        if not isinstance(data, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in data.items()
        ):
            logger.warning("Could not read %s %s (invalid shape) — treating as absent", label, path)
            return {}
        return data
    return {}


def load_etags(workspace: Workspace) -> dict[str, str]:
    return _read_mapping(workspace.etags, "ETag store")


def _save_etags_unlocked(workspace: Workspace, etags: dict[str, str]) -> None:
    path = workspace.etags
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(path, json.dumps(etags, indent=2), mode=PRIVATE_FILE_MODE)


def save_etags(workspace: Workspace, etags: dict[str, str]) -> None:
    """Replace the ETag store under the Run state lock.

    Download paths should use :class:`EtagRun`, which merges their transition
    with updates from other processes. This function remains for compatibility
    and deliberate administrative replacement.
    """
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in etags.items()):
        raise ValueError("ETags must map string Asset hrefs to string validators.")
    with exclusive_lock(workspace.state_lock):
        _save_etags_unlocked(workspace, etags)


@dataclass
class EtagRun:
    """One Dataset download's isolated ETag transition."""

    workspace: Workspace
    original: dict[str, str]
    values: dict[str, str]

    @classmethod
    def begin(cls, workspace: Workspace) -> EtagRun:
        original = load_etags(workspace)
        return cls(workspace, original, dict(original))

    def commit(self, collection_id: str, *, listed: set[str] | None = None) -> None:
        """Merge this run's changes and optionally prune a complete Collection listing."""
        changed = {key: value for key, value in self.values.items() if self.original.get(key) != value}
        removed = self.original.keys() - self.values.keys()
        with exclusive_lock(self.workspace.state_lock):
            current = load_etags(self.workspace)
            current.update(changed)
            for key in removed:
                current.pop(key, None)
            if listed is not None:
                prefix = f"/{collection_id}/"
                # Prune only entries this run actually observed at begin and that
                # no concurrent run changed. A stale full listing must not erase
                # a newly published Asset another process learned about meanwhile.
                stale = [
                    key
                    for key, original_value in self.original.items()
                    if prefix in key and key not in listed and current.get(key) == original_value
                ]
                for key in stale:
                    del current[key]
                if stale:
                    logger.info("  Pruned %d stale ETag entries", len(stale))
            _save_etags_unlocked(self.workspace, current)


def parse_instant(value: str) -> datetime | None:
    """A timezone-aware ISO 8601 instant, or None if *value* is not one.

    Compared as instants, never as strings: ``2026-10-25T10:30:00Z`` is later than
    ``2026-10-25T11:00:00+01:00``, and a string comparison says otherwise.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return parsed if parsed.tzinfo is not None else None


def _later(a: str | None, b: str) -> str:
    """Whichever of two stored instants is later, keeping a valid one over junk."""
    if a is None:
        return b
    pa, pb = parse_instant(a), parse_instant(b)
    if pa is None:
        return b
    if pb is None:
        return a
    return a if pa > pb else b


def _cursor_key(dataset: str, time_slice: str) -> str:
    return f"{dataset}/{time_slice}"


def load_cursor(workspace: Workspace, dataset: str, time_slices: list[str]) -> str | None:
    """The instant after which nothing upstream changed for *dataset*'s requested slices.

    None — list everything — unless every requested slice has a cursor of its
    own: a slice requested for the first time has nothing to be incremental
    against. With several slices the earliest cursor wins, so each of them is
    covered. Files deleted from Bronze are not this function's concern: the
    download step fetches any wanted file that is missing, whatever the cursor.
    """
    if not time_slices:
        return None
    stored = _read_mapping(workspace.last_run, "last-run cursor")
    cursors: list[tuple[datetime, str]] = []
    for time_slice in time_slices:
        value = stored.get(_cursor_key(dataset, time_slice))
        parsed = parse_instant(value) if value is not None else None
        if value is None or parsed is None:
            return None
        cursors.append((parsed, value))
    return min(cursors)[1]


def advance_cursor(workspace: Workspace, dataset: str, time_slices: list[str], watermark: str) -> None:
    """Record that *dataset*'s *time_slices* are complete up to *watermark*.

    Only ever forward: a run that started earlier than one already recorded
    leaves the later cursor in place. Other Datasets' cursors are untouched.
    """
    if parse_instant(watermark) is None:
        raise ValueError("Run watermark must be an ISO 8601 timestamp with a timezone.")
    path = workspace.last_run
    path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(workspace.state_lock):
        current = _read_mapping(path, "last-run cursor")
        for time_slice in time_slices:
            key = _cursor_key(dataset, time_slice)
            current[key] = _later(current.get(key), watermark)
        write_text(path, json.dumps(current, indent=2, sort_keys=True), mode=PRIVATE_FILE_MODE)


def load_last_run(workspace: Workspace) -> str | None:
    """Return the pre-scoping, Workspace-wide ``timestamp``, or None.

    Kept for callers of the v0.4.0 API. The download command reads
    :func:`load_cursor` instead.
    """
    timestamp = _read_mapping(workspace.last_run, "last-run cursor").get("timestamp")
    if timestamp is None:
        return None
    if parse_instant(timestamp) is None:
        logger.warning("Could not read last-run cursor %s (invalid timestamp) — treating as absent", workspace.last_run)
        return None
    return timestamp


def run_watermark() -> str:
    """Timestamp captured before Collection listings begin."""
    return datetime.now(UTC).isoformat()


def save_last_run(workspace: Workspace, timestamp: str | None = None) -> None:
    """Advance the pre-scoping ``timestamp``, keeping the scoped cursors beside it."""
    path = workspace.last_run
    path.parent.mkdir(parents=True, exist_ok=True)
    value = timestamp or run_watermark()
    if parse_instant(value) is None:
        raise ValueError("Run watermark must include a timezone.")
    with exclusive_lock(workspace.state_lock):
        current = _read_mapping(path, "last-run cursor")
        current["timestamp"] = _later(current.get("timestamp"), value)
        write_text(path, json.dumps(current, indent=2, sort_keys=True), mode=PRIVATE_FILE_MODE)


__all__ = [
    "EtagRun",
    "advance_cursor",
    "load_cursor",
    "load_etags",
    "load_last_run",
    "parse_instant",
    "run_watermark",
    "save_etags",
    "save_last_run",
]
