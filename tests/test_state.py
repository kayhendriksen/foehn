"""Tests for the ETag store and the last-run cursor.

Both reads are total: a corrupt file is treated as absent, because a lost cursor
costs one redundant download and a raised exception costs the whole run.
"""

from datetime import UTC, datetime

import pytest

from foehn.state import EtagRun, load_etags, load_last_run, save_etags, save_last_run
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


def test_load_etags_valid_json_with_wrong_shape_returns_empty(tmp_path):
    (tmp_path / "_etags.json").write_text("[]")
    assert load_etags(Workspace(tmp_path)) == {}


def test_load_etags_rejects_non_string_entries(tmp_path):
    (tmp_path / "_etags.json").write_text('{"asset": 42}')
    assert load_etags(Workspace(tmp_path)) == {}


def test_overlapping_dataset_etag_runs_merge_their_changes(tmp_path):
    workspace = Workspace(tmp_path)
    first = EtagRun.begin(workspace)
    second = EtagRun.begin(workspace)
    first.values["https://data.geo.admin.ch/a/file.csv"] = '"a"'
    second.values["https://data.geo.admin.ch/b/file.csv"] = '"b"'

    first.commit("a")
    second.commit("b")

    assert load_etags(workspace) == {
        "https://data.geo.admin.ch/a/file.csv": '"a"',
        "https://data.geo.admin.ch/b/file.csv": '"b"',
    }


def test_a_stale_listing_does_not_prune_a_concurrently_discovered_asset(tmp_path):
    workspace = Workspace(tmp_path)
    old = "https://data.geo.admin.ch/ch.example.collection/old.csv"
    new = "https://data.geo.admin.ch/ch.example.collection/new.csv"
    save_etags(workspace, {old: '"old"'})
    stale_listing = EtagRun.begin(workspace)
    concurrent = EtagRun.begin(workspace)
    concurrent.values[new] = '"new"'

    concurrent.commit("ch.example.collection")
    stale_listing.commit("ch.example.collection", listed=set())

    assert load_etags(workspace) == {new: '"new"'}


def test_load_last_run_corrupt_file_returns_none(tmp_path):
    (tmp_path / "_last_run.json").write_text("not json")
    assert load_last_run(Workspace(tmp_path)) is None


def test_load_last_run_valid_json_with_wrong_shape_returns_none(tmp_path):
    (tmp_path / "_last_run.json").write_text("[]")
    assert load_last_run(Workspace(tmp_path)) is None


@pytest.mark.parametrize("payload", ['{"timestamp": 42}', '{"timestamp": "not-a-date"}', '{"timestamp": "2026-01-01"}'])
def test_load_last_run_rejects_invalid_timestamps(tmp_path, payload):
    (tmp_path / "_last_run.json").write_text(payload)
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


def test_save_last_run_accepts_the_pre_listing_watermark(tmp_path):
    watermark = "2026-08-31T10:00:00+00:00"
    save_last_run(Workspace(tmp_path), watermark)
    assert load_last_run(Workspace(tmp_path)) == watermark


def test_an_older_concurrent_run_cannot_move_the_cursor_backwards(tmp_path):
    workspace = Workspace(tmp_path)
    save_last_run(workspace, "2026-08-31T10:05:00+00:00")

    save_last_run(workspace, "2026-08-31T10:00:00+00:00")

    assert load_last_run(workspace) == "2026-08-31T10:05:00+00:00"
