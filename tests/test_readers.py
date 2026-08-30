"""Tests at :mod:`foehn.readers`' interface.

The three read paths themselves are covered through ``foehn.load`` in
``test_api``, which is where callers meet them. These cover the query object the
readers take — the normalisation that used to be written out once per reader.
"""

from __future__ import annotations

import polars as pl

from foehn.fetch import DEFAULT_WORKERS
from foehn.readers import Filters, apply_time_filters

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
