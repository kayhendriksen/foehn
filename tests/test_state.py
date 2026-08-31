"""Tests for the ETag store and the last-run cursor.

Both reads are total: a corrupt file is treated as absent, because a lost cursor
costs one redundant download and a raised exception costs the whole run.
"""

from datetime import UTC, datetime

from foehn.state import load_etags, load_last_run, save_etags, save_last_run
from foehn.workspace import Workspace

# --- State management ---


def test_load_etags_missing_file_returns_empty(tmp_path):
    assert load_etags(Workspace(tmp_path)) == {}


def test_save_and_load_etags_roundtrip(tmp_path):
    etags = {"https://data.geo.admin.ch/file.csv": '"abc123"'}
    save_etags(Workspace(tmp_path), etags)
    assert load_etags(Workspace(tmp_path)) == etags


def test_save_etags_creates_parent_dirs(tmp_path):
    nested = Workspace(tmp_path / "a" / "b")
    save_etags(nested, {"k": "v"})
    assert nested.etags.exists()


def test_save_etags_overwrites_existing(tmp_path):
    save_etags(Workspace(tmp_path), {"k": "old"})
    save_etags(Workspace(tmp_path), {"k": "new"})
    assert load_etags(Workspace(tmp_path)) == {"k": "new"}


def test_load_etags_corrupt_file_returns_empty(tmp_path):
    """A torn write must not brick subsequent runs — treat as empty state."""
    (tmp_path / "_etags.json").write_text('{"truncated": ')
    assert load_etags(Workspace(tmp_path)) == {}


def test_load_last_run_corrupt_file_returns_none(tmp_path):
    (tmp_path / "_last_run.json").write_text("not json")
    assert load_last_run(Workspace(tmp_path)) is None


def test_load_last_run_missing_file_returns_none(tmp_path):
    assert load_last_run(Workspace(tmp_path)) is None


def test_save_and_load_last_run_roundtrip(tmp_path):
    save_last_run(Workspace(tmp_path))
    timestamp = load_last_run(Workspace(tmp_path))
    assert timestamp is not None
    dt = datetime.fromisoformat(timestamp)
    assert dt.tzinfo is not None


def test_save_last_run_is_recent(tmp_path):
    before = datetime.now(UTC)
    save_last_run(Workspace(tmp_path))
    after = datetime.now(UTC)

    saved = datetime.fromisoformat(load_last_run(Workspace(tmp_path)))
    assert before <= saved <= after
