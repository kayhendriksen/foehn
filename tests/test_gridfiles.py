"""Tests for getting a grid kind's matched files onto disk (foehn.gridfiles).

Split out of ``test_grids`` with the module: what these assert is which assets a
match selects, what the caps refuse and what is fetched — never what a Dataset
comes back looking like.
"""

import os
import time
from pathlib import Path

import pytest

from foehn.collections import COLLECTIONS
from foehn.fetch import FetchError
from foehn.gridfiles import _run_datetime_filter, ensure_grid_files
from foehn.workspace import Workspace
from tests.fakes import InMemoryFetcher


def _items_for(*filenames):
    """STAC items whose .nc asset hrefs map to the given (already cached) filenames."""
    return [{"assets": {"d": {"href": f"https://data.geo.admin.ch/x/{name}"}}} for name in filenames]


def _fake(items=None, *, body=b"x"):
    """A fetcher listing *items*, serving *body* for every asset."""
    fake = InMemoryFetcher()
    fake.any_items = items if items is not None else []
    fake.default_body = body
    return fake


def test_ensure_netcdf_files_raises_when_no_nc_assets(fetcher, tmp_path):
    """A collection that only exposes GeoTIFF/ZIP should raise a clear error."""
    fetcher.any_items = [
        {"assets": {"a": {"href": "https://data.geo.admin.ch/x/grid.tif"}}},
        {"assets": {"b": {"href": "https://data.geo.admin.ch/x/bundle.zip"}}},
    ]
    with pytest.raises(ValueError, match=r"No \.nc assets"):
        ensure_grid_files("climate_normals_grid", Workspace(tmp_path), fetcher=fetcher)


def test_ensure_netcdf_files_match_selects_subset_via_remote(tmp_path):
    """A NetCDF match consults the remote listing (to verify completeness) and keeps the subset."""
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    keep = out_dir / "x.rhiresd_ch01h.nc"
    drop = out_dir / "x.ranomm9120_ch01r.nc"
    keep.write_bytes(b"x")
    drop.write_bytes(b"x")

    items = _items_for("x.rhiresd_ch01h.nc", "x.ranomm9120_ch01r.nc")
    fake = _fake(items)
    result = ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)
    assert len(fake.listings) == 1  # multi-file format always verifies against remote
    assert fake.streams == []  # both already cached
    assert result == [keep]


def test_ensure_grid_files_refreshes_a_restated_asset(tmp_path):
    out_dir = tmp_path / "bronze" / "radar_precip"
    out_dir.mkdir(parents=True)
    cached = out_dir / "cpc2613000000.h5"
    cached.write_bytes(b"old")
    cached.touch()
    item = {
        "assets": {"d": {"href": "https://data.geo.admin.ch/x/cpc2613000000.h5"}},
        "properties": {"updated": "2100-01-01T00:00:00+00:00"},
    }
    fake = _fake([item], body=b"new")

    result = ensure_grid_files(
        "radar_precip", Workspace(tmp_path), suffixes=(".h5",), match="cpc2613000000", fetcher=fake
    )

    assert result == [cached]
    assert cached.read_bytes() == b"new"
    assert len(fake.streams) == 1


def test_a_failed_refresh_falls_back_to_a_complete_grid_cache(tmp_path):
    out_dir = tmp_path / "bronze" / "radar_precip"
    out_dir.mkdir(parents=True)
    cached = out_dir / "cpc2613000000.h5"
    cached.write_bytes(b"old but complete")
    href = f"https://data.geo.admin.ch/x/{cached.name}"
    item = {"assets": {"d": {"href": href}}, "properties": {"updated": "2100-01-01T00:00:00+00:00"}}
    fake = _fake([item])
    fake.fail(href)

    with pytest.warns(UserWarning, match="using the complete local cache"):
        result = ensure_grid_files(
            "radar_precip", Workspace(tmp_path), suffixes=(".h5",), match="cpc2613000000", fetcher=fake
        )

    assert result == [cached]
    assert cached.read_bytes() == b"old but complete"


def test_ensure_netcdf_files_match_downloads_missing_from_partial_cache(tmp_path):
    """A multi-file NetCDF match must not return a partial cache — it fetches what's missing."""
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / "rhiresd_part1.nc").write_bytes(b"x")  # interrupted earlier download left only part 1

    items = _items_for("rhiresd_part1.nc", "rhiresd_part2.nc")

    def fake_download(_href, filepath):
        Path(filepath).write_bytes(b"y")

    fake = _fake(items)
    fake.stream_hook = fake_download
    result = ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)
    assert len(fake.streams) == 1  # the missing part 2 is fetched
    assert {p.name for p in result} == {"rhiresd_part1.nc", "rhiresd_part2.nc"}


def test_ensure_grid_files_fetches_missing_concurrently(tmp_path):
    """Missing files are fetched in parallel, not one after another."""
    import threading

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    names = [f"rhiresd_part{i}.nc" for i in range(6)]
    items = _items_for(*names)

    active = 0
    peak = 0
    lock = threading.Lock()
    barrier = threading.Barrier(len(names), timeout=5)

    def fake_download(_href, filepath):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        # Blocks until every download is in flight — times out if they are serial.
        barrier.wait()
        with lock:
            active -= 1
        Path(filepath).write_bytes(b"y")

    fake = _fake(items)
    fake.stream_hook = fake_download
    result = ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)

    assert len(fake.streams) == len(names)
    assert peak > 1  # genuinely overlapping
    assert {p.name for p in result} == set(names)


def test_ensure_grid_files_deduplicates_repeated_asset(tmp_path):
    """One file listed under several asset keys is downloaded once.

    Two workers streaming into the same ``.part`` would corrupt it, so the
    dedupe matters now that these downloads run concurrently.
    """
    items = [
        {
            "assets": {
                "data": {"href": "https://data.geo.admin.ch/x/rhiresd_ch01h.nc"},
                "alternate": {"href": "https://data.geo.admin.ch/x/rhiresd_ch01h.nc"},
            }
        }
    ]

    def fake_download(_href, filepath):
        Path(filepath).write_bytes(b"y")

    fake = _fake(items)
    fake.stream_hook = fake_download
    result = ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)

    assert len(fake.streams) == 1
    assert [p.name for p in result] == ["rhiresd_ch01h.nc"]


def test_grid_listing_accepts_a_cached_walk(tmp_path):
    """The read path lists on every open, so it opts into a cached listing.

    Whether an entry is reused, and for how long, is the fetcher's business —
    the TTL is derived from what the walk cost, which only it can measure (see
    tests/test_fetch.py). What grids decides is that staleness is acceptable
    here at all; the download paths never ask for it.
    """
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / "x.rhiresd_ch01h.nc").write_bytes(b"x")
    fake = _fake(_items_for("x.rhiresd_ch01h.nc"))

    ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)

    assert [cache for _, cache, _, _ in fake.listings] == [True]


def test_run_datetime_filter_reads_the_run_stamp():
    """A match naming a run becomes a STAC datetime; anything else stays unfiltered."""
    assert _run_datetime_filter("202605231500-0-t_2m-ctrl") == "2026-05-23T15:00:00Z"
    # no parseable run stamp, an impossible one, or no match at all
    assert _run_datetime_filter("t_2m-ctrl") is None
    assert _run_datetime_filter("cpc26130") is None
    assert _run_datetime_filter("209913451500") is None
    assert _run_datetime_filter(None) is None


def test_only_grib2_rows_narrow_by_run():
    """Whether an item's datetime *is* the run is a row, not a test inside the filter.

    CSV and radar collections set item datetime to a catalog-refresh time, so
    filtering on a real data date there would match nothing — those rows opt out,
    which is why a 12-digit stamp in a radar match is never sent as a filter.
    """
    from foehn import registry

    assert registry.spec("forecast_icon_ch1").grid.run_datetime is True
    assert registry.spec("analysis_kenda_ch1").grid.run_datetime is True
    assert registry.spec("radar_precip").grid.run_datetime is False
    assert registry.spec("surface_derived_grid").grid.run_datetime is False


def test_ensure_grid_files_narrows_forecast_listing_by_run(tmp_path):
    """A GRIB2 open passes the run through as a server-side datetime filter."""
    out_dir = tmp_path / "bronze" / "forecast_icon_ch1"
    out_dir.mkdir(parents=True)
    name = "icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2"
    (out_dir / name).write_bytes(b"x")

    fake = _fake(_items_for(name))
    ensure_grid_files(
        "forecast_icon_ch1",
        Workspace(tmp_path),
        suffixes=(".grib2",),
        match="202605231500-0-t_2m-ctrl",
        max_files=1,
        run_datetime=True,
        fetcher=fake,
    )

    assert [dt for _, _, dt, _ in fake.listings] == ["2026-05-23T15:00:00Z"]


def test_ensure_grid_files_falls_back_when_filtered_listing_misses(tmp_path):
    """A filtered miss retries unfiltered rather than reporting the file missing."""
    out_dir = tmp_path / "bronze" / "forecast_icon_ch1"
    out_dir.mkdir(parents=True)
    name = "icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2"
    (out_dir / name).write_bytes(b"x")

    class _FilteredMiss(InMemoryFetcher):
        """Server-side filter finds nothing; the full walk does."""

        def items(self, collection_id, *, cache=False, datetime_filter=None, max_items=None):
            found = super().items(collection_id, cache=cache, datetime_filter=datetime_filter, max_items=max_items)
            return [] if datetime_filter else found

    fake = _FilteredMiss()
    fake.any_items = _items_for(name)
    fake.default_body = b"x"
    result = ensure_grid_files(
        "forecast_icon_ch1",
        Workspace(tmp_path),
        suffixes=(".grib2",),
        match="202605231500-0-t_2m-ctrl",
        max_files=1,
        run_datetime=True,
        fetcher=fake,
    )
    assert [p.name for p in result] == [name]
    assert len(fake.listings) == 2  # filtered, then the unfiltered retry


def test_ensure_single_file_match_validated_against_remote_not_cache(tmp_path):
    """A single-file match unique in a sparse cache but ambiguous in the collection is rejected."""
    out_dir = tmp_path / "bronze" / "forecast_icon_ch1"
    out_dir.mkdir(parents=True)
    (out_dir / "icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2").write_bytes(b"x")  # only one cached

    # ...but the collection has TWO files matching "t_2m-ctrl".
    items = _items_for(
        "icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2",
        "icon-ch1-eps-202605231800-0-t_2m-ctrl.grib2",
    )
    fake = _fake(items)
    with pytest.raises(ValueError, match="one file at a time"):
        ensure_grid_files(
            "forecast_icon_ch1",
            Workspace(tmp_path),
            suffixes=(".grib2", ".grib"),
            match="t_2m-ctrl",
            max_files=1,
            fetcher=fake,
        )
    assert fake.streams == []  # rejected before any download


def test_ensure_single_file_serves_cache_when_remote_unique(tmp_path):
    """When the collection has exactly one match, the cached file is reused (no re-download)."""
    out_dir = tmp_path / "bronze" / "forecast_icon_ch1"
    out_dir.mkdir(parents=True)
    f = out_dir / "icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2"
    f.write_bytes(b"x")

    items = _items_for("icon-ch1-eps-202605231500-0-t_2m-ctrl.grib2")
    fake = _fake(items)
    result = ensure_grid_files(
        "forecast_icon_ch1",
        Workspace(tmp_path),
        suffixes=(".grib2", ".grib"),
        match="t_2m-ctrl",
        max_files=1,
        fetcher=fake,
    )
    assert fake.streams == []
    assert result == [f]


def test_ensure_netcdf_files_match_no_remote_match_raises(tmp_path):
    fake = _fake([{"assets": {"a": {"href": "https://data.geo.admin.ch/x/ranomm9120.nc"}}}])
    with pytest.raises(ValueError, match="matching 'rhiresd'"):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)


def test_ensure_netcdf_files_unfiltered_consults_remote_and_downloads_missing(tmp_path):
    """An unfiltered call must enumerate remote and fetch files missing from a partial cache."""
    base = tmp_path / "bronze" / "surface_derived_grid"
    base.mkdir(parents=True)
    cached = base / "ogd.rhiresd.nc"
    cached.write_bytes(b"x")  # simulate a prior filtered (match="rhiresd") download

    items = [
        {"assets": {"a": {"href": "https://data.geo.admin.ch/x/ogd.rhiresd.nc"}}},
        {"assets": {"b": {"href": "https://data.geo.admin.ch/x/ogd.tabsd.nc"}}},
    ]

    def fake_download(_href, filepath):
        Path(filepath).write_bytes(b"y")

    fake = _fake(items)
    fake.stream_hook = fake_download
    result = ensure_grid_files("surface_derived_grid", Workspace(tmp_path), fetcher=fake)

    assert len(fake.listings) == 1
    assert len(fake.streams) == 1  # only the missing file is fetched; cache reused
    assert {p.name for p in result} == {"ogd.rhiresd.nc", "ogd.tabsd.nc"}


def test_ensure_netcdf_files_offline_falls_back_to_cache(tmp_path):
    """If the STAC API is unreachable, fall back to the cached files with a warning."""
    base = tmp_path / "bronze" / "surface_derived_grid"
    base.mkdir(parents=True)
    (base / "a.nc").write_bytes(b"x")

    fake = _fake()
    fake.fail(COLLECTIONS["surface_derived_grid"], FetchError("offline"))
    with pytest.warns(UserWarning, match="without checking the collection"):
        result = ensure_grid_files("surface_derived_grid", Workspace(tmp_path), fetcher=fake)
    assert [p.name for p in result] == ["a.nc"]


def test_ensure_netcdf_files_offline_no_cache_reraises(tmp_path):
    """Offline with an empty cache must surface the network error, not swallow it."""
    fake = _fake()
    fake.fail(COLLECTIONS["surface_derived_grid"], FetchError("offline"))
    with pytest.raises(FetchError):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), fetcher=fake)


def test_the_cube_cap_explains_itself_differently_from_the_single_file_cap():
    """One-file kinds say "pick a single file"; the cube cap says why the set is bounded."""
    fake = _fake(_items_for(*[f"icon-{i}.grib2" for i in range(6)]))

    with pytest.raises(ValueError, match="exceeds the 3-file cap"):
        ensure_grid_files(
            "forecast_icon_ch1",
            Workspace(Path("unused")),
            suffixes=(".grib2",),
            match="icon-",
            max_files=3,
            fetcher=fake,
        )


def test_a_refresh_that_fails_after_replacing_a_file_refuses_the_mixed_cache(tmp_path):
    """A part-done refresh must not be handed back as "the complete local cache".

    ``fetch_all`` replaces one asset at a time. When the second fails, the first
    is already at the new generation and the second is still at the old one —
    and because both files exist, the existence check called that complete and
    returned it. A cube built from that set silently mixes generations.
    """
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    first, second = out_dir / "a_rhiresd.nc", out_dir / "b_rhiresd.nc"
    first.write_bytes(b"generation one")
    second.write_bytes(b"generation one")

    hrefs = [f"https://data.geo.admin.ch/x/{path.name}" for path in (first, second)]
    items = [
        {"assets": {"d": {"href": href}}, "properties": {"updated": "2100-01-01T00:00:00+00:00"}} for href in hrefs
    ]
    fake = _fake(items, body=b"generation two")
    fake.fail(hrefs[1])  # the first refresh lands, the second does not

    with pytest.raises(FetchError, match="mix file generations"):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)

    # The half-done state is reported, not silently read.
    assert first.read_bytes() == b"generation two"
    assert second.read_bytes() == b"generation one"


def test_a_failed_refresh_with_files_still_missing_raises(tmp_path):
    """No usable cache to fall back to, so the fetch error stands."""
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    hrefs = [f"https://data.geo.admin.ch/x/{name}_rhiresd.nc" for name in ("a", "b")]
    fake = _fake(_items_for("a_rhiresd.nc", "b_rhiresd.nc"))
    fake.fail(hrefs[0])

    with pytest.raises(FetchError):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)


def _mixed_cache(tmp_path):
    """Leave a cache where one file is refreshed and the other is not."""
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    first, second = out_dir / "a_rhiresd.nc", out_dir / "b_rhiresd.nc"
    first.write_bytes(b"generation one")
    second.write_bytes(b"generation one")
    hrefs = [f"https://data.geo.admin.ch/x/{p.name}" for p in (first, second)]
    items = [{"assets": {"d": {"href": h}}, "properties": {"updated": "2100-01-01T00:00:00+00:00"}} for h in hrefs]
    fake = _fake(items, body=b"generation two")
    fake.fail(hrefs[1])
    with pytest.raises(FetchError, match=r"mix file generations"):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)
    return out_dir, items, hrefs


def test_a_mixed_cache_stays_refused_on_the_next_attempt(tmp_path):
    """The next run must not re-baseline against the mix it inherited.

    Judging coherence by what *this* run replaced only catches the run that
    causes the mix. The following attempt measures a cache nothing has touched
    since it started, finds it unchanged, and hands the mix back as complete.
    """
    out_dir, items, hrefs = _mixed_cache(tmp_path)
    assert (out_dir / "a_rhiresd.nc").read_bytes() == b"generation two"
    assert (out_dir / "b_rhiresd.nc").read_bytes() == b"generation one"

    again = _fake(items, body=b"generation two")
    again.fail(hrefs[0])
    again.fail(hrefs[1])
    # Either refusal is right: the marker's, or the retry's own incomplete run.
    with pytest.raises(FetchError, match="generations"):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=again)


def test_a_mixed_cache_stays_refused_when_the_collection_is_unreachable(tmp_path):
    """Offline there is no Asset metadata to judge coherence with, only the marker."""
    _mixed_cache(tmp_path)

    offline = _fake([])
    offline.fail(COLLECTIONS["surface_derived_grid"], FetchError("offline"))
    with pytest.raises(FetchError, match="mixes file generations"):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=offline)


def test_a_completed_refresh_clears_the_mixed_cache_marker(tmp_path):
    """The set is coherent again once every file is at the same generation."""
    out_dir, items, _ = _mixed_cache(tmp_path)

    healed = _fake(items, body=b"generation two")
    result = ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=healed)

    assert len(result) == 2
    assert all(p.read_bytes() == b"generation two" for p in result)
    assert not (out_dir / ".foehn-incoherent.json").exists()


def test_an_unrelated_match_does_not_release_the_files_still_mixed(tmp_path):
    """One collection holds many independent parameter sets.

    A completed ``tabs`` refresh says nothing about whether ``rhiresd`` was ever
    finished, so clearing a dataset-wide flag on it handed back exactly the
    files still known to be mixed.
    """
    out_dir, _items, _hrefs = _mixed_cache(tmp_path)

    # A different parameter, refreshed cleanly and completely.
    tabs = out_dir / "c_tabsd.nc"
    tabs.write_bytes(b"generation one")
    tabs_href = f"https://data.geo.admin.ch/x/{tabs.name}"
    tabs_items = [{"assets": {"d": {"href": tabs_href}}, "properties": {"updated": "2000-01-01T00:00:00+00:00"}}]
    ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="tabsd", fetcher=_fake(tabs_items))

    # rhiresd is still half-refreshed, and still refused.
    offline = _fake([])
    offline.fail(COLLECTIONS["surface_derived_grid"], FetchError("offline"))
    with pytest.raises(FetchError, match="mixes file generations"):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=offline)


def test_an_interrupted_refresh_is_recorded_like_a_failed_one(tmp_path):
    """Ctrl-C leaves the same half-done set a network error does."""
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    first, second = out_dir / "a_rhiresd.nc", out_dir / "b_rhiresd.nc"
    first.write_bytes(b"generation one")
    second.write_bytes(b"generation one")
    hrefs = [f"https://data.geo.admin.ch/x/{p.name}" for p in (first, second)]
    items = [{"assets": {"d": {"href": h}}, "properties": {"updated": "2100-01-01T00:00:00+00:00"}} for h in hrefs]
    fake = _fake(items, body=b"generation two")
    fake.fail(hrefs[1], KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)

    offline = _fake([])
    offline.fail(COLLECTIONS["surface_derived_grid"], FetchError("offline"))
    with pytest.raises(FetchError, match="mixes file generations"):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=offline)


def test_the_marker_is_on_disk_before_any_file_is_published(tmp_path):
    """A killed process never reaches an exception handler.

    Recording the mix only when the fetch raised meant a hard exit between
    publishing one file and the next left [new, old] with nothing to say so, and
    the next offline read handed both back.
    """
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    first, second = out_dir / "a_rhiresd.nc", out_dir / "b_rhiresd.nc"
    first.write_bytes(b"generation one")
    second.write_bytes(b"generation one")
    hrefs = [f"https://data.geo.admin.ch/x/{p.name}" for p in (first, second)]
    items = [{"assets": {"d": {"href": h}}, "properties": {"updated": "2100-01-01T00:00:00+00:00"}} for h in hrefs]

    seen = []
    fake = _fake(items, body=b"generation two")

    def note_marker(url, path):
        seen.append((out_dir / ".foehn-incoherent.json").exists())
        path.write_bytes(b"generation two")

    fake.stream_hook = note_marker
    ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)

    assert seen and all(seen), "marker must exist before the first file is published"
    # A completed refresh takes it away again.
    assert not (out_dir / ".foehn-incoherent.json").exists()


def test_a_marked_file_is_refetched_even_when_it_looks_current(tmp_path):
    """A Collection with no usable ``updated`` makes every local file look fresh.

    The refresh then skips everything, succeeds, and clearing the marker on that
    released the mix untouched. Marked files have to be fetched regardless.
    """
    out_dir, _items, hrefs = _mixed_cache(tmp_path)

    # No "updated" at all: already_current() says every file is fine.
    undated = [{"assets": {"d": {"href": h}}} for h in hrefs]
    fake = _fake(undated, body=b"generation two")
    result = ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)

    assert all(p.read_bytes() == b"generation two" for p in result)
    assert not (out_dir / ".foehn-incoherent.json").exists()


def test_a_corrupt_marker_blocks_rather_than_fails_open(tmp_path):
    """A marker truncated by the crash it was recording must not read as "nothing pending"."""
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    cached = out_dir / "a_rhiresd.nc"
    cached.write_bytes(b"generation one")
    (out_dir / ".foehn-incoherent.json").write_text("{ truncated")

    offline = _fake([])
    offline.fail(COLLECTIONS["surface_derived_grid"], FetchError("offline"))
    with pytest.raises(FetchError, match="mixes file generations"):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=offline)


def test_concurrent_marker_updates_do_not_lose_entries(tmp_path):
    """Two failed refreshes for different matches each record their own names."""
    import threading

    from foehn.gridfiles import _mark_incoherent, _read_incoherent, _refresh_lock

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    start = threading.Barrier(2)

    def record(names):
        start.wait()
        for _ in range(20):
            with _refresh_lock(out_dir):
                _mark_incoherent(out_dir, names)

    threads = [
        threading.Thread(target=record, args=(["a_rhiresd.nc", "b_rhiresd.nc"],)),
        threading.Thread(target=record, args=(["c_tabsd.nc", "d_tabsd.nc"],)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert _read_incoherent(out_dir) == {"a_rhiresd.nc", "b_rhiresd.nc", "c_tabsd.nc", "d_tabsd.nc"}


@pytest.mark.parametrize(
    "content",
    ['{"pending": "not-a-list"}', "[]", '"just a string"', "{ truncated"],
)
def test_a_marker_that_cannot_be_understood_blocks(tmp_path, content):
    """Every unreadable shape has to fail closed, not just invalid JSON."""
    from foehn.gridfiles import _read_incoherent

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / ".foehn-incoherent.json").write_text(content)

    assert _read_incoherent(out_dir) is None


def test_clearing_leaves_an_unreadable_marker_alone(tmp_path):
    """Replacing it with a guess would release whatever it was protecting."""
    from foehn.gridfiles import _clear_incoherent

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    marker = out_dir / ".foehn-incoherent.json"
    marker.write_text("{ truncated")

    _clear_incoherent(out_dir, ["a_rhiresd.nc"])

    assert marker.read_text() == "{ truncated"


def test_a_failed_refresh_keeps_its_mark_rather_than_guessing(tmp_path):
    """A refresh that did not finish cannot say whether it left the set mixed.

    Releasing the mark whenever "nothing was replaced" looked like a safe
    exception and was not: a file can be rewritten with different bytes at the
    same length and its mtime put back, and both size and mtime then say nothing
    happened. Only a completed refresh clears.
    """
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    first, second = out_dir / "a_rhiresd.nc", out_dir / "b_rhiresd.nc"
    first.write_bytes(b"generation one")
    second.write_bytes(b"generation one")
    hrefs = [f"https://data.geo.admin.ch/x/{p.name}" for p in (first, second)]
    items = [{"assets": {"d": {"href": h}}, "properties": {"updated": "2100-01-01T00:00:00+00:00"}} for h in hrefs]

    fake = _fake(items, body=b"generation two")

    def same_stat_rewrite(url, path):
        """Publish different bytes at the same length, then put the mtime back."""
        if path.name == first.name:
            original = path.stat()
            path.write_bytes(b"generation TWO")  # same length as "generation one"
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
            return
        raise FetchError("boom")

    fake.stream_hook = same_stat_rewrite

    with pytest.raises(FetchError, match="mix file generations"):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)

    assert (out_dir / ".foehn-incoherent.json").exists()

    offline = _fake([])
    offline.fail(COLLECTIONS["surface_derived_grid"], FetchError("offline"))
    with pytest.raises(FetchError, match="mixes file generations"):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=offline)


def test_overlapping_refreshes_of_one_match_never_interleave(tmp_path):
    """Two refreshes of the same match must not publish into each other.

    Each checked coherence, each downloaded, and each published into the same
    directory — producing a set neither one would have produced on its own, and
    then both clearing the marker on the way out.
    """
    import threading

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    first, second = out_dir / "a_rhiresd.nc", out_dir / "b_rhiresd.nc"
    first.write_bytes(b"AAAA")
    second.write_bytes(b"AAAA")
    hrefs = [f"https://data.geo.admin.ch/x/{p.name}" for p in (first, second)]
    items = [{"assets": {"d": {"href": h}}, "properties": {"updated": "2100-01-01T00:00:00+00:00"}} for h in hrefs]

    ready = threading.Barrier(2)

    def writer(payload):
        fake = _fake(items, body=payload)

        def slow(url, path):
            # Wide open window for the other refresh to interleave with this one.
            time.sleep(0.02)
            path.write_bytes(payload)

        fake.stream_hook = slow
        ready.wait()
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)

    threads = [threading.Thread(target=writer, args=(p,)) for p in (b"BBBB", b"CCCC")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Whichever refresh finished last, both files are from the same one.
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() in (b"BBBB", b"CCCC")
    assert not (out_dir / ".foehn-incoherent.json").exists()


def test_an_unreadable_marker_file_blocks_like_a_corrupt_one(tmp_path):
    """A marker we cannot even open is state we do not know."""
    from unittest.mock import patch

    from foehn.gridfiles import _read_incoherent

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / ".foehn-incoherent.json").write_text('{"pending": ["a.nc"]}')

    with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
        assert _read_incoherent(out_dir) is None


def test_marking_does_not_overwrite_a_marker_it_cannot_read(tmp_path):
    """Merging into an unreadable marker turns "unknown" into a tidy empty set.

    An unrelated successful refresh then deletes that set, releasing the files
    the unreadable marker was protecting.
    """
    from foehn.gridfiles import _mark_incoherent, _refresh_lock

    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    marker = out_dir / ".foehn-incoherent.json"
    marker.write_text("{ truncated")

    with _refresh_lock(out_dir):
        _mark_incoherent(out_dir, ["a_rhiresd.nc"])

    assert marker.read_text() == "{ truncated"


def test_a_failed_refresh_with_a_file_missing_raises_the_original(tmp_path):
    """No usable cache to fall back on, and nothing needed refreshing."""
    out_dir = tmp_path / "bronze" / "surface_derived_grid"
    out_dir.mkdir(parents=True)
    (out_dir / "a_rhiresd.nc").write_bytes(b"only one of two")
    hrefs = [f"https://data.geo.admin.ch/x/{n}" for n in ("a_rhiresd.nc", "b_rhiresd.nc")]
    # No "updated", so the present file is current and only the missing one is fetched.
    items = [{"assets": {"d": {"href": h}}} for h in hrefs]
    fake = _fake(items)
    fake.fail(hrefs[1])

    with pytest.raises(FetchError):
        ensure_grid_files("surface_derived_grid", Workspace(tmp_path), match="rhiresd", fetcher=fake)
