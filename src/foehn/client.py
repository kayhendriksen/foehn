"""Deprecated compatibility shim for the pre-0.5 ``foehn.client`` module.

``client`` was one module doing the work that ``downloads``, ``transfer``,
``fetch``, ``state`` and ``archives`` now do separately, and splitting it up
deleted the import path along with it. Most of what lived here was internal
machinery with no caller outside foehn, but the run state and ETag helpers were
importable, documented and took a plain ``Path`` — so code written against
v0.4.0 stopped importing at all rather than stopping at one changed call.

Everything here forwards to :mod:`foehn.state` and warns. The signatures are
v0.4.0's: a ``data_dir`` path, not a :class:`~foehn.workspace.Workspace`. New
code should use :mod:`foehn.state` with a Workspace, or the public
:func:`foehn.download` / :func:`foehn.load` entry points.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from foehn import state
from foehn.transfer import DownloadResult
from foehn.workspace import Workspace

_REPLACEMENTS = {
    "download_collection": "foehn.download(dataset, ...)",
    "download_metadata": "foehn.download(dataset, ...)",
    "download_grib2": "foehn.download(dataset, ...)",
    "download_netcdf": "foehn.download(dataset, ...)",
    "download_climate_normals_zip": 'foehn.download("climate_normals", ...)',
    "download_climate_scenarios_indoor": 'foehn.download("climate_scenarios_indoor", ...)',
}


def _deprecated(name: str, replacement: str) -> None:
    warnings.warn(
        f"foehn.client.{name}() is deprecated and will be removed in a future release; use {replacement}.",
        DeprecationWarning,
        stacklevel=3,
    )


def load_etags(data_dir: Path) -> dict[str, str]:
    """Deprecated: use ``foehn.state.load_etags(workspace)``."""
    _deprecated("load_etags", "foehn.state.load_etags(Workspace.resolve(data_dir))")
    return state.load_etags(Workspace.resolve(data_dir))


def save_etags(data_dir: Path, etags: dict[str, str]) -> None:
    """Deprecated: use ``foehn.state.save_etags(workspace, etags)``."""
    _deprecated("save_etags", "foehn.state.save_etags(Workspace.resolve(data_dir), etags)")
    state.save_etags(Workspace.resolve(data_dir), etags)


def load_last_run(data_dir: Path) -> str | None:
    """Deprecated: use ``foehn.state.load_last_run(workspace)``."""
    _deprecated("load_last_run", "foehn.state.load_last_run(Workspace.resolve(data_dir))")
    return state.load_last_run(Workspace.resolve(data_dir))


def save_last_run(data_dir: Path, timestamp: str | None = None) -> None:
    """Deprecated: use ``foehn.state.save_last_run(workspace)``."""
    _deprecated("save_last_run", "foehn.state.save_last_run(Workspace.resolve(data_dir))")
    state.save_last_run(Workspace.resolve(data_dir), timestamp)


def __getattr__(name: str):
    """Name the replacement for the download helpers rather than 404 on them.

    These moved rather than disappeared, and an AttributeError on a module that
    exists is a worse signpost than the one line saying where the work went.
    """
    if name in _REPLACEMENTS:
        raise AttributeError(
            f"foehn.client.{name}() was removed in 0.5; the download paths now live behind "
            f"the public API — use {_REPLACEMENTS[name]}."
        )
    raise AttributeError(f"module 'foehn.client' has no attribute {name!r}")


__all__ = ["DownloadResult", "load_etags", "load_last_run", "save_etags", "save_last_run"]
