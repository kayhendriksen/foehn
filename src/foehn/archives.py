"""Reading a ZIP MeteoSwiss published, without trusting what is inside it.

A decompression-bomb cap and a zip-slip guard. Both the download paths, which
extract to :term:`Bronze`, and the load path, which reads an archive's members
in memory, apply the same cap — one rule, and neither of them is the other's
module.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


# Cap on declared total decompressed size. Generous — the largest legitimate
# archive (indoor scenarios) is well under this — but stops a decompression
# bomb from a compromised upstream filling the disk (or RAM, for in-memory
# reads). Python's zipfile enforces each member's declared size on read, so
# checking the headers is sufficient.
_MAX_ZIP_EXTRACT_BYTES = 10 * 1024**3  # 10 GiB


def check_zip_size(zf: zipfile.ZipFile, source: str) -> None:
    """Raise ValueError if the archive declares more decompressed bytes than the cap.

    Separate from :func:`safe_extract` because the archive *load* path reads its
    ZIP in memory rather than extracting it, and has to apply the same cap. One
    rule, one place — the reader crossing this seam is part of the interface.
    """
    total = sum(m.file_size for m in zf.infolist())
    if total > _MAX_ZIP_EXTRACT_BYTES:
        raise ValueError(
            f"ZIP {source!r} declares {total / 1e9:.1f} GB decompressed "
            f"(cap {_MAX_ZIP_EXTRACT_BYTES / 1e9:.0f} GB) — refusing to extract."
        )


def safe_extract(zip_path: Path, out_dir: Path) -> int:
    """Extract a ZIP after validating total size and member paths. Returns member count."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        check_zip_size(zf, zip_path.name)
        resolved_out_dir = out_dir.resolve()
        for member in zf.infolist():
            target = (resolved_out_dir / member.filename).resolve()
            # Path comparison, not string prefixing: a hardcoded "/" separator
            # rejects every legitimate member on Windows, where the resolved
            # target is separated by "\" and nothing ever matches the prefix.
            if target == resolved_out_dir or not target.is_relative_to(resolved_out_dir):
                raise ValueError(f"Unsafe path in ZIP: {member.filename!r}")
        zf.extractall(out_dir)
        return len(zf.namelist())
