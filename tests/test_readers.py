"""Tests at :mod:`foehn.readers`' interface.

The three read paths themselves are covered through ``foehn.load`` in
``test_api``, which is where callers meet them. These cover the query object the
readers take — the normalisation that used to be written out once per reader.
"""

from __future__ import annotations

import polars as pl
import pytest
from conftest import CLIMATE_SCENARIOS_CSV, make_zip

from foehn.fetch import DEFAULT_WORKERS
from foehn.readers import Filters, apply_time_filters, read_archive, read_preamble
from tests.fakes import InMemoryFetcher, stac_item

# --- Normalisation ---


def test_station_and_frequency_are_lowercased_sets():
    filters = Filters.build(station=["BER", "Zur"], frequency="D")
    assert filters.stations == frozenset({"ber", "zur"})
    assert filters.granularities == frozenset({"d"})


def test_a_single_string_becomes_a_one_element_set():
    assert Filters.build(station="BER").stations == frozenset({"ber"})


def test_empty_list_means_no_filter_not_match_nothing():
    """All three readers now agree; the archive and preamble ones used to error."""
    filters = Filters.build(station=[], frequency=[])
    assert filters.stations is None
    assert filters.granularities is None


def test_time_slice_defaults_to_recent():
    assert Filters.build().time_slices == ("recent",)
    assert Filters.build(time_slice="now").time_slices == ("now",)
    assert Filters.build(time_slice=["historical", "recent"]).time_slices == ("historical", "recent")


def test_scalar_year_and_month_become_tuples():
    filters = Filters.build(year=2025, month=[6, 7])
    assert filters.year == (2025,)
    assert filters.month == (6, 7)


def test_workers_defaults_to_the_shared_concurrency():
    assert Filters.build().workers == DEFAULT_WORKERS


def test_has_calendar_filter_reports_any_of_the_four():
    assert not Filters.build().has_calendar_filter
    assert Filters.build(year=2025).has_calendar_filter
    assert Filters.build(month=7).has_calendar_filter
    assert Filters.build(date_from="2025-01-01").has_calendar_filter
    assert Filters.build(date_to="2025-01-01").has_calendar_filter


def test_filters_are_hashable_and_frozen():
    """A query is a value: readers can hold one without copying it defensively."""
    assert hash(Filters.build(station="BER", year=[2025, 2026])) is not None


@pytest.mark.parametrize("month", [0, 13, [1, 12, 99]])
def test_month_must_be_in_calendar_range(month):
    with pytest.raises(ValueError, match="Invalid month"):
        Filters.build(month=month)


@pytest.mark.parametrize("value", ["sideways", "ASC"])
def test_sort_is_validated_at_the_filters_interface(value):
    with pytest.raises(ValueError, match="Invalid sort"):
        Filters.build(sort=value)


def test_zero_limit_is_intentional_but_negative_is_invalid():
    assert Filters.build(limit=0).limit == 0
    with pytest.raises(ValueError, match="zero or greater"):
        Filters.build(limit=-1)


def test_workers_must_be_positive():
    with pytest.raises(ValueError, match="greater than zero"):
        Filters.build(workers=0)


@pytest.mark.parametrize("label", ["date_from", "date_to"])
def test_date_bounds_must_be_iso(label):
    with pytest.raises(ValueError, match=label):
        Filters.build(**{label: "not-a-date"})


def test_date_bounds_must_not_be_reversed():
    with pytest.raises(ValueError, match="date_from"):
        Filters.build(date_from="2025-02-01", date_to="2025-01-01")


def test_timezone_bounds_are_normalized_before_polars_compares_them():
    filters = Filters.build(date_from="2025-01-01T00:00:00+01:00")

    assert filters.date_from == "2024-12-31T23:00:00"
    assert apply_time_filters(_frame(["2024-12-31 22:00", "2024-12-31 23:00"]), filters).height == 1


# --- The shared time predicates ---


def _frame(timestamps: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"reference_timestamp": timestamps}).with_columns(
        pl.col("reference_timestamp").str.to_datetime()
    )


def test_no_filters_is_identity():
    df = _frame(["2025-06-01 00:00", "2025-07-01 00:00"])
    assert len(apply_time_filters(df, Filters.build())) == 2


def test_year_and_month_combine():
    df = _frame(["2025-06-01 00:00", "2025-07-01 00:00", "2026-07-01 00:00"])
    out = apply_time_filters(df, Filters.build(year=2025, month=7))
    assert len(out) == 1


# --- What a reader refuses ---


def _fetcher(items, *, body=b""):
    fake = InMemoryFetcher()
    fake.any_items = items
    fake.default_body = body
    return fake


def test_preamble_reader_narrows_items_to_the_requested_stations():
    """The station filter runs at the item level, so an unmatched station is never fetched."""
    fake = _fetcher(
        [
            stac_item("ABE", "https://data.geo.admin.ch/x/CH2025_ABE_pr_GWL1.5.csv"),
            stac_item("BER", "https://data.geo.admin.ch/x/CH2025_BER_pr_GWL1.5.csv"),
        ],
        body=CLIMATE_SCENARIOS_CSV.encode(),
    )

    df = read_preamble("climate_scenarios", Filters.build(station="abe"), fetcher=fake)

    assert fake.gets == ["https://data.geo.admin.ch/x/CH2025_ABE_pr_GWL1.5.csv"]
    assert df["station_abbr"].unique().to_list() == ["ABE"]


def test_preamble_reader_names_the_station_when_nothing_matches():
    """ "No CSVs found" without the station is a message that cannot be acted on."""
    fake = _fetcher([stac_item("ABE", "https://data.geo.admin.ch/x/CH2025_ABE_pr_GWL1.5.csv")])

    with pytest.raises(ValueError, match=r"station=\['zzz'\]"):
        read_preamble("climate_scenarios", Filters.build(station="ZZZ"), fetcher=fake)


def test_archive_reader_refuses_a_collection_with_no_zip():
    fake = _fetcher([stac_item("i", "https://data.geo.admin.ch/x/readme.txt")])

    with pytest.raises(ValueError, match=r"No \.zip asset"):
        read_archive("climate_scenarios_indoor", Filters.build(), fetcher=fake)


def test_archive_reader_ignores_non_csv_members():
    """A ZIP carries a licence and a readme alongside the data."""
    zip_bytes = make_zip(
        {
            "README.txt": b"not data",
            "ABE_2020_RCP26_a.csv": b"time.yy,time.mm,time.dd,time.hh,top\n2020,1,1,0,21.5\n",
        }
    )
    fake = _fetcher([stac_item("i", "https://data.geo.admin.ch/x/indoor.zip")], body=zip_bytes)

    df = read_archive("climate_scenarios_indoor", Filters.build(), fetcher=fake)

    assert df["station_abbr"].to_list() == ["ABE"]


def test_archive_reader_names_the_station_when_nothing_matches():
    zip_bytes = make_zip({"ABE_2020_RCP26_a.csv": b"time.yy,top\n2020,21.5\n"})
    fake = _fetcher([stac_item("i", "https://data.geo.admin.ch/x/indoor.zip")], body=zip_bytes)

    with pytest.raises(ValueError, match=r"station=\['zzz'\]"):
        read_archive("climate_scenarios_indoor", Filters.build(station="ZZZ"), fetcher=fake)
