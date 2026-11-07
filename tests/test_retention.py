"""Authoritative retirement must preserve data whenever the evidence is incomplete."""

import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from foehn.convert import convert_to_parquet
from foehn.retention import RETIREMENT_GRACE, record_listing
from foehn.workspace import Workspace

START = datetime(2026, 1, 1, tzinfo=UTC)
WITH_GROUP = ["ogd-smn_aaa_d_recent.csv", "ogd-smn_bbb_h_recent.csv"]
WITHOUT_GROUP = ["ogd-smn_bbb_h_recent.csv"]


def _evidence(**changes) -> bytes:
    """A record that retires smn_d_recent, but for *changes*."""
    record = {
        "version": 1,
        "recorded_at": (START + RETIREMENT_GRACE).isoformat(),
        "listed_since": START.isoformat(),
        "files": WITHOUT_GROUP,
        "absent_since": {},
    }
    return json.dumps(record | changes).encode()


def _unlisted_throughout_grace(bronze, names):
    record_listing(bronze, names, now=START)
    record_listing(bronze, names, now=START + RETIREMENT_GRACE)


@pytest.fixture
def absent_sources(tmp_path):
    workspace = Workspace(tmp_path)
    bronze = workspace.bronze("smn")
    bronze.mkdir(parents=True)
    for suffix in ("d_recent", "d_historical", "m"):
        (bronze / f"ogd-smn_aaa_{suffix}.csv").write_text(
            "station_abbr;reference_timestamp;tre200d0\nAAA;01.01.2026 00:00;1.5\n"
        )
    assert convert_to_parquet("smn", workspace) == 0
    for source in bronze.glob("*.csv"):
        source.unlink()
    return workspace


def test_complete_evidence_retires_the_group(absent_sources):
    """What keeps the invalid payloads below from passing for an unrelated reason."""
    workspace = absent_sources
    (workspace.bronze("smn") / ".foehn-rolling-listing.json").write_bytes(_evidence())

    assert convert_to_parquet("smn", workspace) == 0
    assert not (workspace.parquet("smn") / "smn_d_recent.parquet").exists()


@pytest.mark.parametrize(
    "payload",
    [
        b"{truncated",
        b"\xff",
        b"[]",
        _evidence(version=2),
        _evidence(files=[]),
        _evidence(files=["../ogd-smn_bbb_h_recent.csv"]),
        _evidence(files=42),
        _evidence(recorded_at="yesterday"),
        _evidence(recorded_at="2026-01-08T00:00:00"),  # no timezone
        _evidence(listed_since=None),
        _evidence(absent_since=[]),
        _evidence(absent_since={"../ogd-smn_aaa_d_recent.csv": START.isoformat()}),
        _evidence(absent_since={"ogd-smn_aaa_d_recent.csv": "soon"}),
    ],
)
def test_invalid_retirement_evidence_preserves_existing_outputs(absent_sources, payload):
    workspace = absent_sources
    before = {path: path.read_bytes() for path in workspace.parquet("smn").glob("*.parquet")}
    (workspace.bronze("smn") / ".foehn-rolling-listing.json").write_bytes(payload)

    assert convert_to_parquet("smn", workspace) == 0
    assert {path: path.read_bytes() for path in before} == before


def test_retirement_repairs_old_outputs_but_preserves_archives_and_foreign_files(absent_sources):
    """The last source can already be gone before this version's first successful listing."""
    workspace = absent_sources
    outputs = workspace.parquet("smn")
    foreign = outputs / "smn_y_now.parquet"
    pl.DataFrame({"unrelated": [7]}).write_parquet(foreign)
    preserved = [outputs / "smn_d_historical.parquet", outputs / "smn_m.parquet", foreign]
    before = {path: path.read_bytes() for path in preserved}
    _unlisted_throughout_grace(workspace.bronze("smn"), WITHOUT_GROUP)

    assert convert_to_parquet("smn", workspace) == 0
    assert not (outputs / "smn_d_recent.parquet").exists()
    assert {path: path.read_bytes() for path in preserved} == before


def test_an_upstream_group_is_not_retired_when_its_local_files_are_missing(absent_sources):
    workspace = absent_sources
    output = workspace.parquet("smn") / "smn_d_recent.parquet"
    before = output.read_bytes()
    _unlisted_throughout_grace(workspace.bronze("smn"), ["ogd-smn_aaa_d_recent.csv"])

    assert convert_to_parquet("smn", workspace) == 0
    assert output.read_bytes() == before


def test_a_locally_reintroduced_group_survives_an_older_listing(absent_sources):
    workspace = absent_sources
    bronze = workspace.bronze("smn")
    _unlisted_throughout_grace(bronze, WITHOUT_GROUP)
    (bronze / "ogd-smn_ccc_d_recent.csv").write_text(
        "station_abbr;reference_timestamp;tre200d0\nCCC;01.01.2026 00:00;42\n"
    )

    assert convert_to_parquet("smn", workspace) == 0
    result = pl.read_parquet(workspace.parquet("smn") / "smn_d_recent.parquet")
    assert result["station_abbr"].to_list() == ["CCC"]
    assert result["tre200d0"].to_list() == [42]


def test_a_first_listing_alone_retires_nothing(absent_sources):
    """It may be the incomplete one: nothing yet says the group was ever expected to return."""
    workspace = absent_sources
    record_listing(workspace.bronze("smn"), WITHOUT_GROUP, now=START)

    assert convert_to_parquet("smn", workspace) == 0
    assert (workspace.parquet("smn") / "smn_d_recent.parquet").exists()


def test_frequent_listings_cannot_outrun_the_grace(absent_sources):
    """The near-real-time job lists every five minutes, so one incident spans many listings."""
    workspace = absent_sources
    bronze = workspace.bronze("smn")
    record_listing(bronze, WITH_GROUP, now=START)
    for minutes in range(5, 65, 5):
        record_listing(bronze, WITHOUT_GROUP, now=START + timedelta(minutes=minutes))

    assert convert_to_parquet("smn", workspace) == 0
    assert (workspace.parquet("smn") / "smn_d_recent.parquet").exists()


def test_a_group_is_retired_once_its_absence_outlasts_the_grace(absent_sources):
    workspace = absent_sources
    bronze = workspace.bronze("smn")
    output = workspace.parquet("smn") / "smn_d_recent.parquet"
    dropped = START + timedelta(days=1)
    record_listing(bronze, WITH_GROUP, now=START)
    record_listing(bronze, WITHOUT_GROUP, now=dropped)
    record_listing(bronze, WITHOUT_GROUP, now=dropped + RETIREMENT_GRACE - timedelta(minutes=1))

    assert convert_to_parquet("smn", workspace) == 0
    assert output.exists()

    record_listing(bronze, WITHOUT_GROUP, now=dropped + RETIREMENT_GRACE)

    assert convert_to_parquet("smn", workspace) == 0
    assert not output.exists()


def test_a_single_infrequent_listing_cannot_retire_a_group(absent_sources):
    """A long gap since the group was last listed is not a long absence: only one listing has missed it."""
    workspace = absent_sources
    bronze = workspace.bronze("smn")
    record_listing(bronze, WITH_GROUP, now=START)
    record_listing(bronze, WITHOUT_GROUP, now=START + 10 * RETIREMENT_GRACE)

    assert convert_to_parquet("smn", workspace) == 0
    assert (workspace.parquet("smn") / "smn_d_recent.parquet").exists()


def test_a_group_listed_again_restarts_its_grace(absent_sources):
    workspace = absent_sources
    bronze = workspace.bronze("smn")
    first_dropped = START + timedelta(days=1)
    record_listing(bronze, WITH_GROUP, now=START)
    record_listing(bronze, WITHOUT_GROUP, now=first_dropped)
    record_listing(bronze, WITH_GROUP, now=START + timedelta(days=2))
    record_listing(bronze, WITHOUT_GROUP, now=START + timedelta(days=3))
    record_listing(bronze, WITHOUT_GROUP, now=first_dropped + RETIREMENT_GRACE)

    assert convert_to_parquet("smn", workspace) == 0
    assert (workspace.parquet("smn") / "smn_d_recent.parquet").exists()


def test_a_group_waits_for_its_last_file_to_go(absent_sources):
    """One station leaving long ago says nothing about the group that outlived it."""
    workspace = absent_sources
    bronze = workspace.bronze("smn")
    last_dropped = START + timedelta(days=30)
    record_listing(bronze, ["ogd-smn_zzz_d_recent.csv", *WITH_GROUP], now=START)
    record_listing(bronze, WITH_GROUP, now=START + timedelta(days=1))
    record_listing(bronze, WITHOUT_GROUP, now=last_dropped)
    record_listing(bronze, WITHOUT_GROUP, now=last_dropped + timedelta(days=1))

    assert convert_to_parquet("smn", workspace) == 0
    assert (workspace.parquet("smn") / "smn_d_recent.parquet").exists()


def test_unreadable_evidence_restarts_the_wait(absent_sources):
    workspace = absent_sources
    bronze = workspace.bronze("smn")
    _unlisted_throughout_grace(bronze, WITHOUT_GROUP)
    (bronze / ".foehn-rolling-listing.json").write_bytes(b"{truncated")
    record_listing(bronze, WITHOUT_GROUP, now=START + 2 * RETIREMENT_GRACE)

    assert convert_to_parquet("smn", workspace) == 0
    assert (workspace.parquet("smn") / "smn_d_recent.parquet").exists()
