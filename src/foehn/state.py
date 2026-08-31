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
from datetime import UTC, datetime

from foehn.atomicwrite import write_text
from foehn.workspace import Workspace

logger = logging.getLogger(__name__)


def load_etags(workspace: Workspace) -> dict:
    path = workspace.etags
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s (%s) — treating as empty", path, exc)
    return {}


def save_etags(workspace: Workspace, etags: dict):
    path = workspace.etags
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(path, json.dumps(etags, indent=2))


def load_last_run(workspace: Workspace) -> str | None:
    """Return ISO timestamp of last successful run, or None."""
    path = workspace.last_run
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s (%s) — treating as no previous run", path, exc)
            return None
        return data.get("timestamp")
    return None


def save_last_run(workspace: Workspace):
    path = workspace.last_run
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text(path, json.dumps({"timestamp": datetime.now(UTC).isoformat()}))
