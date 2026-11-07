"""Authoritative retirement must preserve data whenever the evidence is incomplete."""

import polars as pl
import pytest

from foehn.convert import convert_to_parquet
from foehn.retention import record_listing
from foehn.workspace import Workspace


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


@pytest.mark.parametrize(
    "payload",
    [
        b"{truncated",
        b"\xff",
        b"[]",
        b'{"version": 2, "files": ["ogd-smn_bbb_h_recent.csv"]}',
        b'{"version": 1, "files": []}',
        b'{"version": 1, "files": ["../ogd-smn_bbb_h_recent.csv"]}',
        b'{"version": 1, "files": 42}',
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
    record_listing(workspace.bronze("smn"), ["ogd-smn_bbb_h_recent.csv"])

    assert convert_to_parquet("smn", workspace) == 0
    assert not (outputs / "smn_d_recent.parquet").exists()
    assert {path: path.read_bytes() for path in preserved} == before


def test_an_upstream_group_is_not_retired_when_its_local_files_are_missing(absent_sources):
    workspace = absent_sources
    output = workspace.parquet("smn") / "smn_d_recent.parquet"
    before = output.read_bytes()
    record_listing(workspace.bronze("smn"), ["ogd-smn_aaa_d_recent.csv"])

    assert convert_to_parquet("smn", workspace) == 0
    assert output.read_bytes() == before


def test_a_locally_reintroduced_group_survives_an_older_listing(absent_sources):
    workspace = absent_sources
    bronze = workspace.bronze("smn")
    record_listing(bronze, ["ogd-smn_bbb_h_recent.csv"])
    (bronze / "ogd-smn_ccc_d_recent.csv").write_text(
        "station_abbr;reference_timestamp;tre200d0\nCCC;01.01.2026 00:00;42\n"
    )

    assert convert_to_parquet("smn", workspace) == 0
    result = pl.read_parquet(workspace.parquet("smn") / "smn_d_recent.parquet")
    assert result["station_abbr"].to_list() == ["CCC"]
    assert result["tre200d0"].to_list() == [42]
