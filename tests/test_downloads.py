"""Tests for the download adapters — the STAC engine and the two ZIP paths.

These cross the network port with an :class:`InMemoryFetcher` and assert on what
lands on disk and in ``DownloadResult`` — not on which HTTP calls were made. The
few that do check traffic are the ones where traffic *is* the behaviour: an ETag
skip, a ``since`` filter, first-page-only listing.
"""

import pytest
from conftest import make_zip

from foehn import registry
from foehn.downloads import download_indoor_zip, download_metadata, download_normals_zip
from foehn.fetch import FetchError
from foehn.state import load_etags, save_etags
from foehn.transfer import DownloadResult
from foehn.workspace import Workspace
from tests.fakes import InMemoryFetcher, stac_item

# --- Helpers ---


def _fake(items=None, *, collection=None, body=b"a;b\n1;2\n"):
    """A fetcher listing *items* (or serving *collection*), with a default body.

    The CSV kinds fetch collection-level metadata in the same pass as their item
    assets, so an empty collection payload is the default — a test that cares
    about the metadata pass passes its own.
    """
    fake = InMemoryFetcher()
    if items is not None:
        fake.any_items = items
    fake.any_collection = collection if collection is not None else {"assets": {}}
    fake.default_body = body
    return fake


def _serve(fake, content=b"a;b\n1;2\n", etag=None, status_code=200):
    """Serve *content* for every URL, optionally with an ETag.

    ``status_code=304`` in the old mocks meant "the server says unchanged"; here
    that is expressed properly — the fetcher answers 304 when the caller sends a
    matching ETag, so the test seeds one.
    """
    fake.default_body = content
    if etag is not None:
        fake.etags["*"] = etag
        fake.default_etag = etag


def _stac_item(asset_url, updated="2026-01-01T00:00:00Z"):
    return {"id": "item-1", "assets": {"data": {"href": asset_url}}, "properties": {"updated": updated}}


# --- the CSV listing path (registry: STANDARD_CSV / PREAMBLE_CSV) ---


def test_csv_download_saves_csv(tmp_path):
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url)])
    _serve(fake, b"station_abbr;value\nTST;1.0\n")

    result = registry.download("smn", Workspace(tmp_path), fetcher=fake)

    assert (tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv").exists()
    assert isinstance(result, DownloadResult)
    assert result.downloaded == 1
    assert result.skipped == 0
    assert result.filenames == ["ogd-smn_tst_d_recent.csv"]


def test_csv_download_detects_csv_with_query_string(tmp_path):
    # Regression: a CSV href carrying a query string (e.g. ?token=...) must not be
    # skipped by the asset filter, which once gated on raw href.endswith(".csv").
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv?token=abc123"
    fake = _fake([_stac_item(url)])
    _serve(fake, b"station_abbr;value\nTST;1.0\n")

    result = registry.download("smn", Workspace(tmp_path), fetcher=fake)

    # Detected, downloaded, and saved under the query-stripped filename.
    assert (tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv").exists()
    assert result.downloaded == 1
    assert result.filenames == ["ogd-smn_tst_d_recent.csv"]
    # The "recent" time slice is still parsed from the query-string URL.
    assert fake.gets[-1] == url


def test_csv_download_re_encodes_to_utf8(tmp_path):
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url)])
    # Windows-1252 encoded content (ä = 0xe4)
    _serve(fake, b"col\n\xe4\n")

    registry.download("smn", Workspace(tmp_path), fetcher=fake)

    content = (tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv").read_text(encoding="utf-8")
    assert "ä" in content


def test_csv_download_saves_etag(tmp_path):
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url)])
    _serve(fake, etag='"v1"')

    registry.download("smn", Workspace(tmp_path), fetcher=fake)

    etags = load_etags(Workspace(tmp_path))
    assert etags.get(url) == '"v1"'


def test_csv_download_skips_unchanged_asset(tmp_path):
    """A stored ETag plus a file on disk means a conditional request and a skip."""
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url)])
    _serve(fake, etag='"v1"')
    save_etags(Workspace(tmp_path), {url: '"v1"'})
    out = tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("existing")

    result = registry.download("smn", Workspace(tmp_path), fetcher=fake)

    assert result.skipped == 1
    assert result.downloaded == 0
    assert out.read_text() == "existing"  # untouched


def test_csv_download_ignores_stored_etag_when_file_is_gone(tmp_path):
    """Otherwise a 304 would report 'skipped' over a file that no longer exists."""
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url)])
    _serve(fake, etag='"v1"')
    save_etags(Workspace(tmp_path), {url: '"v1"'})  # ETag remembered, file deleted

    result = registry.download("smn", Workspace(tmp_path), fetcher=fake)

    assert result.downloaded == 1
    assert (tmp_path / "bronze" / "smn" / "ogd-smn_tst_d_recent.csv").exists()


def test_csv_download_drops_etag_when_server_stops_sending_one(tmp_path):
    """A 200 with no ETag header must clear the stored one.

    Keeping it would re-send the stale value as If-None-Match on every later
    run, so the asset would download every time and never once be skipped.
    """
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url)])
    save_etags(Workspace(tmp_path), {url: '"stale-etag"'})

    _serve(fake, b"station_abbr;value\nTST;1.0\n", etag=None)
    registry.download("smn", Workspace(tmp_path), fetcher=fake)

    assert url not in load_etags(Workspace(tmp_path))


def test_csv_download_deduplicates_same_destination(tmp_path):
    """Two hrefs resolving to one filename are fetched once, not concurrently.

    Both workers would otherwise stream into the same ``.part`` file.
    """
    base = "https://data.geo.admin.ch"
    fake = _fake(
        [
            _stac_item(f"{base}/a/ogd-smn_tst_d_recent.csv"),
            _stac_item(f"{base}/b/ogd-smn_tst_d_recent.csv"),
        ]
    )
    _serve(fake, b"station_abbr;value\nTST;1.0\n")

    result = registry.download("smn", Workspace(tmp_path), fetcher=fake)

    assert result.downloaded == 1
    assert len(fake.gets) == 1


def test_csv_download_prunes_stale_etags(tmp_path):
    """A clean full run drops ETags for assets gone upstream — scoped to this collection."""
    base = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/tst"
    current = f"{base}/ogd-smn_tst_d_recent.csv"
    historical = f"{base}/ogd-smn_tst_d_historical.csv"  # listed, filtered out by data_types
    stale = f"{base}/ogd-smn_tst_d_now.csv"  # no longer listed upstream
    other = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/tst/ogd-nime_tst_d_recent.csv"
    save_etags(Workspace(tmp_path), {stale: '"gone"', historical: '"hist"', other: '"keep"'})

    item = {
        "id": "tst",
        "assets": {"a": {"href": current}, "b": {"href": historical}},
        "properties": {"updated": "2026-01-01T00:00:00Z"},
    }
    fake = _fake([item])
    _serve(fake, etag='"v1"')

    registry.download("smn", Workspace(tmp_path), time_slice=["recent"], fetcher=fake)

    etags = load_etags(Workspace(tmp_path))
    assert stale not in etags  # gone upstream → pruned
    assert etags[historical] == '"hist"'  # still listed (other slice) → kept
    assert etags[other] == '"keep"'  # different collection → untouched
    assert etags[current] == '"v1"'


def test_csv_download_incremental_run_does_not_prune(tmp_path):
    """With ``since`` the item list is partial — never prune from it."""
    stale = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/tst/ogd-smn_tst_d_now.csv"
    save_etags(Workspace(tmp_path), {stale: '"keep"'})
    url = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/tst/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url, updated="2026-02-01T00:00:00Z")])
    _serve(fake, etag='"v1"')

    registry.download("smn", Workspace(tmp_path), since="2026-01-01T00:00:00Z", fetcher=fake)

    assert load_etags(Workspace(tmp_path))[stale] == '"keep"'


def test_csv_download_since_filter(tmp_path):
    """Items older than `since` should be skipped without any HTTP call."""
    url = "https://data.geo.admin.ch/ogd-smn_tst_d_recent.csv"
    fake = _fake([_stac_item(url, updated="2025-06-01T00:00:00Z")])
    _serve(
        fake,
    )

    result = registry.download("smn", Workspace(tmp_path), since="2026-01-01T00:00:00Z", fetcher=fake)

    assert fake.gets == [] and fake.streams == []
    assert result.downloaded == 0
    assert result.filenames == []


def test_csv_download_resilient_to_single_failure(tmp_path):
    """One failing asset must not abort the batch or discard the others' ETags."""
    good = "https://data.geo.admin.ch/ogd-smn_aaa_d_recent.csv"
    bad = "https://data.geo.admin.ch/ogd-smn_bbb_d_recent.csv"
    fake = _fake(
        [
            _stac_item(good),
            {"id": "item-2", "assets": {"data": {"href": bad}}, "properties": {"updated": "2026-01-01T00:00:00Z"}},
        ]
    )

    _serve(fake, b"station_abbr;value\nAAA;1.0\n", etag="etag-aaa")
    fake.fail(bad, FetchError("boom"))

    result = registry.download("smn", Workspace(tmp_path), fetcher=fake)

    # The good asset still downloaded; the bad one is counted, not raised.
    assert result.downloaded == 1
    assert result.failed == 1
    assert result.filenames == ["ogd-smn_aaa_d_recent.csv"]
    assert (tmp_path / "bronze" / "smn" / "ogd-smn_aaa_d_recent.csv").exists()
    # ETags for the successful asset are persisted despite the sibling failure.
    etags = load_etags(Workspace(tmp_path))
    assert etags.get(good) == "etag-aaa"
    assert bad not in etags


# --- download_metadata ---


def test_download_metadata_saves_csv(tmp_path):
    fake = _fake(collection={"assets": {"stations": {"href": "https://data.geo.admin.ch/stations.csv"}}})
    _serve(fake, b"id;name\nTST;Test Station\n")

    result = download_metadata("smn", Workspace(tmp_path), fetcher=fake)

    assert (tmp_path / "bronze" / "smn" / "stations.csv").exists()
    assert result.downloaded == 1
    assert result.filenames == ["stations.csv"]


def test_download_metadata_skips_non_csv_assets(tmp_path):
    fake = _fake(collection={"assets": {"readme": {"href": "https://data.geo.admin.ch/README.pdf"}}})

    download_metadata("smn", Workspace(tmp_path), fetcher=fake)

    assert fake.gets == [] and fake.streams == []


# --- download_climate_normals_zip ---


def test_normals_zip_extracts_files(tmp_path):
    zip_bytes = make_zip({"sample.txt": b"data"})
    fake = _fake(body=zip_bytes)

    download_normals_zip("climate_normals", Workspace(tmp_path), fetcher=fake)

    assert (tmp_path / "bronze" / "climate_normals" / "normwerte.zip").exists()
    assert (tmp_path / "bronze" / "climate_normals" / "sample.txt").exists()


def test_normals_zip_skips_a_completed_materialization(tmp_path):
    first = _fake(body=make_zip({"sample.txt": b"data"}))
    download_normals_zip("climate_normals", Workspace(tmp_path), fetcher=first)

    second = _fake(body=make_zip({"sample.txt": b"changed"}))
    download_normals_zip("climate_normals", Workspace(tmp_path), fetcher=second)

    assert second.gets == [] and second.streams == []


def test_normals_zip_redownloads_if_not_extracted(tmp_path):
    """A ZIP left by a run that died before extraction must not be mistaken for done."""
    out_dir = tmp_path / "bronze" / "climate_normals"
    out_dir.mkdir(parents=True)
    (out_dir / "normwerte.zip").write_bytes(b"partial-or-unextracted")

    zip_bytes = make_zip({"sample.txt": b"data"})
    fake = _fake(body=zip_bytes)

    download_normals_zip("climate_normals", Workspace(tmp_path), fetcher=fake)

    assert len(fake.streams) == 1
    assert (out_dir / "sample.txt").exists()


def test_normals_zip_force_redownloads(tmp_path):
    out_dir = tmp_path / "bronze" / "climate_normals"
    out_dir.mkdir(parents=True)
    (out_dir / "old.txt").write_bytes(b"stale")

    zip_bytes = make_zip({"new.txt": b"fresh"})
    fake = _fake(body=zip_bytes)

    download_normals_zip("climate_normals", Workspace(tmp_path), force=True, fetcher=fake)

    assert len(fake.streams) == 1
    assert (out_dir / "new.txt").exists()
    assert not (out_dir / "old.txt").exists()


def test_normals_refresh_keeps_the_previous_complete_directory_on_failure(tmp_path, monkeypatch):
    out_dir = tmp_path / "bronze" / "climate_normals"
    first = _fake(body=make_zip({"old.txt": b"complete"}))
    download_normals_zip("climate_normals", Workspace(tmp_path), fetcher=first)

    def fail_after_one_member(_zip_path, staged_dir):
        (staged_dir / "new.txt").write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr("foehn.transfer.safe_extract", fail_after_one_member)
    second = _fake(body=make_zip({"new.txt": b"fresh"}))
    with pytest.raises(OSError, match="disk full"):
        download_normals_zip("climate_normals", Workspace(tmp_path), force=True, fetcher=second)

    assert (out_dir / "old.txt").read_bytes() == b"complete"
    assert not (out_dir / "new.txt").exists()


# --- the binary listing path (registry: GRIB2_GRID / RADAR_GRID) ---


def test_binary_download_saves_binary(tmp_path):
    href = "https://data.geo.admin.ch/forecast.grib2"
    fake = _fake([_stac_item(href)], body=b"GRIBdata")

    result = registry.download("forecast_icon_ch1", Workspace(tmp_path), fetcher=fake)

    assert (tmp_path / "bronze" / "forecast_icon_ch1" / "forecast.grib2").read_bytes() == b"GRIBdata"
    assert result.downloaded == 1
    assert result.filenames == ["forecast.grib2"]


def test_binary_download_skips_existing_file(tmp_path):
    href = "https://data.geo.admin.ch/forecast.grib2"
    fake = _fake([_stac_item(href)], body=b"GRIBdata")
    out_dir = tmp_path / "bronze" / "forecast_icon_ch1"
    out_dir.mkdir(parents=True)
    (out_dir / "forecast.grib2").write_bytes(b"existing")

    registry.download("forecast_icon_ch1", Workspace(tmp_path), fetcher=fake)

    assert fake.streams == []
    assert (out_dir / "forecast.grib2").read_bytes() == b"existing"


def test_binary_download_reads_only_the_newest_page(tmp_path):
    """These collections hold thousands of ephemeral items; walking them all is pure cost."""
    fake = _fake([_stac_item("https://data.geo.admin.ch/forecast.grib2")], body=b"GRIBdata")

    registry.download("forecast_icon_ch1", Workspace(tmp_path), fetcher=fake)

    assert [max_items for *_, max_items in fake.listings] == [100]


# --- the static listing path (registry: NETCDF_GRID) ---


def test_static_download_saves_nc_file(tmp_path):
    fake = _fake([{"id": "g1", "assets": {"data": {"href": "https://data.geo.admin.ch/grid.nc"}}, "properties": {}}])
    fake.default_body = b"\x89HDF"

    result = registry.download("surface_derived_grid", Workspace(tmp_path), fetcher=fake)

    assert (tmp_path / "bronze" / "surface_derived_grid" / "grid.nc").exists()
    assert result.downloaded == 1
    assert result.filenames == ["grid.nc"]


def test_static_download_skips_existing_file(tmp_path):
    fake = _fake([{"id": "g1", "assets": {"data": {"href": "https://data.geo.admin.ch/grid.nc"}}, "properties": {}}])

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / "grid.nc").write_bytes(b"existing")

    result = registry.download("surface_derived_grid", Workspace(tmp_path), fetcher=fake)

    assert fake.gets == [] and fake.streams == []
    assert result.downloaded == 0
    assert result.skipped == 1


def test_static_download_since_filter(tmp_path):
    """Items older than `since` should be skipped without any HTTP call."""
    fake = _fake(
        [
            {
                "id": "g1",
                "assets": {"data": {"href": "https://data.geo.admin.ch/grid.nc"}},
                "properties": {"updated": "2025-06-01T00:00:00Z"},
            }
        ]
    )

    result = registry.download("surface_derived_grid", Workspace(tmp_path), since="2026-01-01T00:00:00Z", fetcher=fake)

    assert fake.gets == [] and fake.streams == []
    assert result.downloaded == 0
    assert result.filenames == []


# --- The indoor scenarios ZIP ---


def test_indoor_zip_downloads_and_extracts(tmp_path):
    """The collection ships one .csv.zip rather than per-station assets."""
    zip_bytes = make_zip({"ABE_2020_RCP26_a.csv": b"time.yy,x\n2020,1\n"})
    fake = _fake(items=[stac_item("i", "https://data.geo.admin.ch/x/indoor.zip")], body=zip_bytes)

    result = download_indoor_zip("climate_scenarios_indoor", Workspace(tmp_path), fetcher=fake)

    out_dir = tmp_path / "bronze" / "climate_scenarios_indoor"
    assert (out_dir / "indoor.zip").exists()
    assert (out_dir / "ABE_2020_RCP26_a.csv").exists()
    assert (result.total_assets, result.downloaded, result.skipped) == (1, 1, 0)
    assert result.filenames == ["indoor.zip"]


def test_indoor_zip_skips_when_the_materialization_is_current(tmp_path):
    """The listing is checked for freshness, but a current Archive Asset is not fetched."""
    item = stac_item("i", "https://data.geo.admin.ch/x/indoor.zip")
    first = _fake(items=[item], body=make_zip({"ABE_2020_RCP26_a.csv": b"extracted"}))
    download_indoor_zip("climate_scenarios_indoor", Workspace(tmp_path), fetcher=first)

    second = _fake(items=[item])
    result = download_indoor_zip("climate_scenarios_indoor", Workspace(tmp_path), fetcher=second)

    assert second.streams == [] and len(second.listings) == 1
    assert (result.total_assets, result.downloaded, result.skipped) == (1, 0, 1)


def test_indoor_zip_refreshes_when_the_archive_asset_is_newer(tmp_path):
    href = "https://data.geo.admin.ch/x/indoor.zip"
    first = _fake(
        items=[stac_item("i", href, updated="2026-01-01T00:00:00Z")],
        body=make_zip({"ABE_2020_RCP26_a.csv": b"old"}),
    )
    download_indoor_zip("climate_scenarios_indoor", Workspace(tmp_path), fetcher=first)

    second = _fake(
        items=[stac_item("i", href, updated="2026-02-01T00:00:00Z")],
        body=make_zip({"ABE_2020_RCP26_a.csv": b"new"}),
    )
    result = download_indoor_zip("climate_scenarios_indoor", Workspace(tmp_path), fetcher=second)

    out = tmp_path / "bronze" / "climate_scenarios_indoor" / "ABE_2020_RCP26_a.csv"
    assert result.downloaded == 1
    assert out.read_bytes() == b"new"


def test_indoor_zip_force_redownloads_over_an_extracted_directory(tmp_path):
    """``force`` is what the CLI's --full-refresh reaches for; it must beat the skip."""
    out_dir = tmp_path / "bronze" / "climate_scenarios_indoor"
    out_dir.mkdir(parents=True)
    (out_dir / "ABE_2020_RCP26_a.csv").write_bytes(b"stale")
    zip_bytes = make_zip({"ABE_2020_RCP26_a.csv": b"fresh"})
    fake = _fake(items=[stac_item("i", "https://data.geo.admin.ch/x/indoor.zip")], body=zip_bytes)

    result = download_indoor_zip("climate_scenarios_indoor", Workspace(tmp_path), force=True, fetcher=fake)

    assert result.downloaded == 1
    assert (out_dir / "ABE_2020_RCP26_a.csv").read_bytes() == b"fresh"


def test_indoor_zip_reports_nothing_when_the_collection_has_no_archive(tmp_path):
    """An empty result rather than an exception: the run reports it and carries on."""
    fake = _fake(items=[stac_item("i", "https://data.geo.admin.ch/x/notes.txt")])

    result = download_indoor_zip("climate_scenarios_indoor", Workspace(tmp_path), fetcher=fake)

    assert result == DownloadResult()
    assert fake.streams == []
