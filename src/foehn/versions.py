"""What a bug report needs to know about the environment foehn runs in.

``polars.show_versions()``'s idea: one call that prints foehn's version, the
interpreter, the platform, where the **Workspace** resolves to, and the version
of every dependency foehn declares — required and optional alike — so a report
says which of them was missing instead of leaving a maintainer to ask.
``foehn version`` prints the same report, and ``--output-format json`` the same
facts as data.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
from typing import TypedDict

from foehn import __version__
from foehn.workspace import DATA_DIR_ENV, Workspace

DEPENDENCIES: tuple[str, ...] = (
    # Required.
    "polars",
    "requests",
    "urllib3",
    # grids
    "xarray",
    "numpy",
    "netCDF4",
    "h5netcdf",
    "h5py",
    "zarr",
    "cfgrib",
    "eccodes",
    "eccodeslib",
    "pyproj",
    # mcp
    "mcp",
    "pydantic",
    # databricks
    "pyspark",
    "delta-spark",
    # Used when installed, never required.
    "pyarrow",
    "dask",
)
"""Every distribution foehn declares, by extra, plus the two it uses opportunistically."""


class VersionInfo(TypedDict):
    """What :func:`versions` reports."""

    foehn: str
    python: str
    implementation: str
    platform: str
    data_dir: str
    data_dir_source: str
    dependencies: dict[str, str | None]


def _installed(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions() -> VersionInfo:
    """The facts :func:`show_versions` prints, as a JSON-serialisable mapping.

    Dependencies that are not installed map to None rather than being left out,
    so the report shows what is missing as plainly as what is present.
    """
    return {
        "foehn": __version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "data_dir": str(Workspace.resolve().root),
        "data_dir_source": f"${DATA_DIR_ENV}" if os.environ.get(DATA_DIR_ENV) else "default",
        "dependencies": {name: _installed(name) for name in DEPENDENCIES},
    }


def format_versions(info: VersionInfo | None = None) -> str:
    """Render :func:`versions` as the plain-text report."""
    info = versions() if info is None else info
    dependencies = info["dependencies"]
    width = max(len(name) for name in dependencies) + 2
    lines = [
        "--------Version info---------",
        f"{'foehn:':<{width}}{info['foehn']}",
        f"{'python:':<{width}}{info['python']} ({info['implementation']})",
        f"{'platform:':<{width}}{info['platform']}",
        f"{'data dir:':<{width}}{info['data_dir']} ({info['data_dir_source']})",
        "",
        "--------Dependencies---------",
    ]
    lines += [f"{name + ':':<{width}}{version or '<not installed>'}" for name, version in dependencies.items()]
    return "\n".join(lines)


def show_versions() -> None:
    """Print foehn's version, the platform and every dependency's version.

    Paste the output into a bug report::

        import foehn

        foehn.show_versions()

    The command line equivalent is ``foehn version``.
    """
    print(format_versions())  # noqa: T201 - printing the report is this function's whole job


__all__ = ["DEPENDENCIES", "VersionInfo", "format_versions", "show_versions", "versions"]
