"""What foehn remembers between runs: the ETag store and the last-run cursor.

Two small JSON files whose *location* is the :class:`~foehn.workspace.Workspace`'s
and whose *contents* are the download path's. They lived in the download module,
which is why its docstring described two of the four things it held.

Both reads are total: a corrupt or unreadable file is reported and treated as
absent, because a lost cursor costs one redundant download and a raised
exception costs the whole run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from foehn._locking import exclusive_lock
from foehn.atomicwrite import write_text
from foehn.workspace import Workspace

logger = logging.getLogger(__name__)


def _read_mapping(path: Path, label: str) -> dict[str, str]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
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
    write_text(path, json.dumps(etags, indent=2))


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


def load_last_run(workspace: Workspace) -> str | None:
    """Return ISO timestamp of last successful run, or None."""
    timestamp = _read_mapping(workspace.last_run, "last-run cursor").get("timestamp")
    if timestamp is None:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("Could not read last-run cursor %s (invalid timestamp) — treating as absent", workspace.last_run)
        return None
    if parsed.tzinfo is None:
        logger.warning("Could not read last-run cursor %s (timezone missing) — treating as absent", workspace.last_run)
        return None
    return timestamp


def run_watermark() -> str:
    """Timestamp captured before Collection listings begin."""
    return datetime.now(UTC).isoformat()


def save_last_run(workspace: Workspace, timestamp: str | None = None) -> None:
    path = workspace.last_run
    path.parent.mkdir(parents=True, exist_ok=True)
    value = timestamp or run_watermark()
    # Use the same structural and timezone rule the reader applies.
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Run watermark must include a timezone.")
    with exclusive_lock(workspace.state_lock):
        current = _read_mapping(path, "last-run cursor").get("timestamp")
        if current is not None:
            try:
                current_parsed = datetime.fromisoformat(current.replace("Z", "+00:00"))
            except ValueError:
                current_parsed = None
            if current_parsed is not None and current_parsed.tzinfo is not None and current_parsed > parsed:
                value = current
        write_text(path, json.dumps({"timestamp": value}))


__all__ = ["EtagRun", "load_etags", "load_last_run", "run_watermark", "save_etags", "save_last_run"]
