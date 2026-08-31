"""Where foehn's files live.

One module owns the layout — **Bronze**, Parquet, Zarr and the two state files —
and one rule resolves its root: the caller's argument, else ``$FOEHN_DATA_DIR``,
else ``./data/meteoswiss``.

Before this, ``Path.cwd() / "data" / "meteoswiss"`` was written at seven call
sites and only one of them — the CLI — read the environment variable, so the
same environment sent ``foehn download`` and ``foehn.download()`` to different
directories. ``data_dir / "bronze"`` was written at more sites still, and the
ETag store was placed at whatever ``output_dir.parent`` a caller happened to
pass: give the download path an arbitrary output directory and its state landed
a level above that, rather than in the workspace it belonged to.

Sits below everything and imports nothing of foehn's. The public functions keep
taking ``data_dir=`` and build one of these; everything inside foehn takes the
:class:`Workspace` itself, so no module below the seam derives a path from
another path.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR_ENV = "FOEHN_DATA_DIR"
"""Environment variable naming the workspace root."""

DEFAULT_ROOT = ("data", "meteoswiss")
"""Relative to the current working directory, when nothing else says otherwise."""


def _slug(match: str) -> str:
    """A ``match`` filter as a filesystem-safe fragment of a store name."""
    return re.sub(r"[^0-9A-Za-z]+", "_", match).strip("_") or "match"


@dataclass(frozen=True)
class Workspace:
    """One data directory, and every path foehn derives from it."""

    root: Path

    @classmethod
    def resolve(cls, data_dir: Path | str | None = None) -> Workspace:
        """Build a workspace: explicit argument wins, then the environment, then the default.

        The one rule. A caller that passes ``data_dir`` is answered exactly; a
        caller that passes nothing gets ``$FOEHN_DATA_DIR`` whether it reached
        foehn through the CLI or through the Python API.
        """
        if data_dir is not None:
            return cls(Path(data_dir))
        env_dir = os.environ.get(DATA_DIR_ENV)
        if env_dir:
            return cls(Path(env_dir))
        return cls(Path.cwd().joinpath(*DEFAULT_ROOT))

    def bronze(self, dataset: str | None = None) -> Path:
        """The raw download cache, or one dataset's folder inside it."""
        base = self.root / "bronze"
        return base if dataset is None else base / dataset

    def parquet(self, dataset: str | None = None) -> Path:
        """The Parquet output tree, or one dataset's folder inside it."""
        base = self.root / "parquet"
        return base if dataset is None else base / dataset

    def zarr(self, dataset: str, match: str | None = None) -> Path:
        """The store path for *dataset*, narrowed by *match* where there is one.

        The name encodes the match so two filtered slices of one collection do
        not overwrite each other. Deriving it was ``api``'s — the one path rule
        foehn stated above this seam, and the reason this method used to ask its
        caller for a name that "already encodes any ``match`` filter" rather
        than for the two facts the name is made of.
        """
        name = dataset if match is None else f"{dataset}__{_slug(match)}"
        return self.root / "zarr" / f"{name}.zarr"

    @property
    def etags(self) -> Path:
        """The ETag store. A property of the workspace, not of a download's output directory."""
        return self.root / "_etags.json"

    @property
    def last_run(self) -> Path:
        """The incremental cursor the CLI advances only after a fully clean run."""
        return self.root / "_last_run.json"


__all__ = ["DATA_DIR_ENV", "DEFAULT_ROOT", "Workspace"]
