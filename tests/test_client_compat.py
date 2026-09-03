"""Tests for the deprecated ``foehn.client`` shim.

Splitting ``client`` into ``downloads``/``transfer``/``fetch``/``state``/
``archives`` removed the import path along with the module, so v0.4.0 code
calling ``foehn.client.load_last_run(data_dir)`` stopped importing rather than
stopping at one changed call. These assert the old spelling still works, still
reads and writes the same files, and says so is deprecated.
"""

import json
import warnings

import pytest

from foehn.workspace import Workspace


def test_the_run_cursor_round_trips_through_the_old_path_signature(tmp_path):
    """v0.4.0 passed a data_dir Path; state.py takes a Workspace."""
    from foehn import client, state

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        client.save_last_run(tmp_path, "2026-09-02T12:00:00+00:00")

        assert client.load_last_run(tmp_path) == "2026-09-02T12:00:00+00:00"

    # Same file the current API reads, not a parallel store.
    assert state.load_last_run(Workspace(tmp_path)) == "2026-09-02T12:00:00+00:00"
    assert json.loads((tmp_path / "_last_run.json").read_text())["timestamp"].startswith("2026-09-02")


def test_the_etag_store_round_trips_through_the_old_path_signature(tmp_path):
    from foehn import client, state

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        client.save_etags(tmp_path, {"https://example.test/a.csv": '"v1"'})

        assert client.load_etags(tmp_path) == {"https://example.test/a.csv": '"v1"'}

    assert state.load_etags(Workspace(tmp_path)) == {"https://example.test/a.csv": '"v1"'}


@pytest.mark.parametrize("call", ["load_last_run", "load_etags"])
def test_the_old_entry_points_warn(tmp_path, call):
    from foehn import client

    with pytest.warns(DeprecationWarning, match=f"foehn.client.{call}"):
        getattr(client, call)(tmp_path)


def test_download_result_is_still_importable_from_the_old_module():
    """``from foehn.client import DownloadResult`` was the documented spelling."""
    from foehn.client import DownloadResult
    from foehn.transfer import DownloadResult as Current

    assert DownloadResult is Current


def test_a_relocated_download_helper_names_its_replacement():
    """An AttributeError on a module that exists is a poor signpost."""
    import foehn.client as client

    with pytest.raises(AttributeError, match=r"foehn\.download\(dataset"):
        _ = client.download_collection

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = client.never_existed


def test_the_documented_spelling_works_after_a_plain_import(tmp_path):
    """v0.4.0 documented ``import foehn`` then ``foehn.client.load_last_run(...)``.

    A submodule is not an attribute of its package until something imports it.
    v0.4.0's __init__ imported client for its own reasons, so the documented
    call resolved there; here nothing did, and it raised AttributeError. Run in
    a subprocess because the rest of this module imports foehn.client directly,
    which would bind the attribute and hide the bug.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(f"""
        import warnings
        warnings.simplefilter("ignore", DeprecationWarning)
        import foehn
        foehn.client.save_last_run({str(tmp_path)!r}, "2026-09-03T00:00:00+00:00")
        print(foehn.client.load_last_run({str(tmp_path)!r}))
    """)
    done = subprocess.run(  # noqa: S603 - our own interpreter running a literal script
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "2026-09-03T00:00:00+00:00"
