"""Fuzz the ZIP guards with archives whose member names are attacker-chosen.

The property that matters is not "no exception" but "nothing lands outside the
target directory": a member named ``../x`` or ``/etc/x`` is how a malicious
archive overwrites files (zip-slip). The input is split into members, one per
line, as ``name|content``; the guard may refuse the archive with ValueError and
the filesystem may refuse a name with OSError, but nothing may escape.
"""

import contextlib
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

try:
    import atheris
except ImportError:  # tests/test_fuzz_targets.py runs these without Atheris
    atheris = None

with atheris.instrument_imports(include=["foehn"]) if atheris else contextlib.nullcontext():
    from foehn import archives

MAX_MEMBERS = 8

# Repeated member names are fair input for a ZIP guard; zipfile warns about each.
warnings.filterwarnings("ignore", message="Duplicate name", category=UserWarning)


def _exists(path: Path) -> bool:
    """``Path.exists``, with a name the filesystem refuses counted as absent.

    Python 3.11 raises OSError (ENAMETOOLONG) from ``exists()`` for a member name
    longer than the filesystem allows, where 3.12 returns False. Such a name
    cannot have been written, so it cannot have escaped either.
    """
    try:
        return path.exists()
    except OSError:
        return False


def _members(data: bytes) -> list[tuple[str, bytes]]:
    members = []
    for line in data.split(b"\n")[:MAX_MEMBERS]:
        raw, _, content = line.partition(b"|")
        # zipfile cuts a name at its first NUL, and a name left empty makes
        # writestr raise IndexError while building the archive. Cut it here
        # instead, so what is tested is the name that ends up in the archive.
        name = raw.decode("utf-8", errors="replace").split("\x00", 1)[0] or "_"
        members.append((name, content))
    return members


def TestOneInput(data: bytes) -> None:  # noqa: N802 - the name libFuzzer and Atheris call
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = root / "a" / "b" / "c" / "out"
        out.mkdir(parents=True)
        inside = out.resolve()
        source = root / "in.zip"
        with zipfile.ZipFile(source, "w") as zf:
            for name, content in _members(data):
                with contextlib.suppress(ValueError):  # zipfile refuses some names outright
                    zf.writestr(name, content)
            names = zf.namelist()

        # Where each member would land if written as named. One that resolves
        # outside the target and appears there is an escape, however many `..`
        # it took — a scan of the temporary directory alone cannot see past it.
        targets = {(out / name).resolve() for name in names}
        outside = {target for target in targets if not target.is_relative_to(inside)}
        present_before = {target for target in outside if _exists(target)}

        with contextlib.suppress(ValueError, OSError):
            archives.safe_extract(source, out)

        escaped = sorted(str(t) for t in outside - present_before if _exists(t))
        escaped += [
            str(p)
            for p in root.rglob("*")
            if p != source and not p.resolve().is_relative_to(inside) and not inside.is_relative_to(p.resolve())
        ]
        assert not escaped, f"extracted outside the target: {escaped}"


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
