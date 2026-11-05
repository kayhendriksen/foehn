"""Fuzz the run-state reads with arbitrary file contents.

foehn's state files (the ETag store, the incremental cursors) promise that a
corrupt or unreadable file is treated as absent: a lost cursor costs one
redundant download, an exception costs the whole run. So every read must
return, whatever bytes are on disk.
"""

import contextlib
import logging
import sys
import tempfile
from pathlib import Path

try:
    import atheris
except ImportError:  # tests/test_fuzz_targets.py runs these without Atheris
    atheris = None

with atheris.instrument_imports(include=["foehn"]) if atheris else contextlib.nullcontext():
    from foehn import state
    from foehn.workspace import Workspace

# Every corrupt file is logged as a warning, which would flood the fuzzer's output.
logging.getLogger("foehn.state").setLevel(logging.ERROR)


def TestOneInput(data: bytes) -> None:  # noqa: N802 - the name libFuzzer and Atheris call
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Workspace(Path(tmp))
        workspace.bronze("smn").mkdir(parents=True)
        workspace.etags.write_bytes(data)
        workspace.last_run.write_bytes(data)

        etags = state.load_etags(workspace)
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in etags.items())
        last_run = state.load_last_run(workspace)
        assert last_run is None or state.parse_instant(last_run) is not None
        cursor = state.load_cursor(workspace, "smn", ["recent"])
        assert cursor is None or state.parse_instant(cursor) is not None


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
