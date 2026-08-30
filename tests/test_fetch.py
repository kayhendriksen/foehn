"""Tests for the network port and its production adapter.

This is the one module that mocks ``requests``: it is testing the adapter that
wraps it. Everything else in the suite crosses the port with an
``InMemoryFetcher`` instead.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest
import requests

from foehn._urls import validate_stac_url
from foehn.fetch import Fetched, FetchError, HttpFetcher, default_fetcher

CSV_URL = "https://data.geo.admin.ch/data.csv"


def _json_response(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


def _body_response(content=b"a;b\n1;2\n", *, etag=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.headers = {"ETag": etag} if etag else {}
    resp.raise_for_status = MagicMock()
    return resp


def _stream_response(chunks=(b"data",)):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_content.return_value = iter(chunks)
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _page(features, next_url=None):
    links = [{"rel": "next", "href": next_url}] if next_url else []
    return {"features": features, "links": links}


def _item(item_id="item-1"):
    return {"id": item_id, "assets": {}, "properties": {}}


def _fetcher_with(*responses):
    """An HttpFetcher whose sessions are a mock cycling through *responses*."""
    session = MagicMock()
    if responses:
        session.get.side_effect = list(responses)
    fetcher = HttpFetcher()
    fetcher._build_session = lambda: session
    return fetcher, session


# --- items ---------------------------------------------------------------


def test_items_single_page():
    fetcher, session = _fetcher_with(_json_response(_page([_item()])))

    result = fetcher.items("ch.test.collection")

    assert [i["id"] for i in result] == ["item-1"]
    assert session.get.call_count == 1


def test_items_follows_pagination():
    fetcher, session = _fetcher_with(
        _json_response(_page([_item("a")], next_url="https://data.geo.admin.ch/api/stac/v1/page2")),
        _json_response(_page([_item("b")])),
    )

    result = fetcher.items("ch.test.collection")

    assert [i["id"] for i in result] == ["a", "b"]
    assert session.get.call_count == 2


def test_items_empty_collection():
    fetcher, _ = _fetcher_with(_json_response(_page([])))

    assert fetcher.items("ch.test.collection") == []


def test_items_rejects_untrusted_next_link():
    fetcher, _ = _fetcher_with(_json_response(_page([_item()], next_url="https://evil.test/page2")))

    with pytest.raises(ValueError, match="Untrusted STAC URL"):
        fetcher.items("ch.test.collection")


def test_items_max_items_stops_before_following_next():
    """The ephemeral collections only want the newest page — don't walk thousands."""
    features = [_item(f"i{n}") for n in range(3)]
    fetcher, session = _fetcher_with(
        _json_response(_page(features, next_url="https://data.geo.admin.ch/api/stac/v1/page2"))
    )

    result = fetcher.items("ch.test.collection", max_items=2)

    assert [i["id"] for i in result] == ["i0", "i1"]
    assert session.get.call_count == 1  # did not follow next


def test_items_cache_serves_second_call_from_cache():
    fetcher, session = _fetcher_with(_json_response(_page([_item()])))

    first = fetcher.items("ch.test.collection", cache=True)
    second = fetcher.items("ch.test.collection", cache=True)

    assert first == second
    assert session.get.call_count == 1


def test_items_without_cache_always_refetches():
    """The download paths must see upstream changes, so they never read a cache."""
    fetcher, session = _fetcher_with(
        _json_response(_page([_item("old")])),
        _json_response(_page([_item("new")])),
    )

    fetcher.items("ch.test.collection")
    second = fetcher.items("ch.test.collection")

    assert [i["id"] for i in second] == ["new"]
    assert session.get.call_count == 2


def test_items_max_items_result_is_not_cached():
    """A truncated listing must not be served later as if it were complete."""
    fetcher, session = _fetcher_with(
        _json_response(_page([_item("a"), _item("b")])),
        _json_response(_page([_item("a"), _item("b")])),
    )

    fetcher.items("ch.test.collection", cache=True, max_items=1)
    full = fetcher.items("ch.test.collection", cache=True)

    assert len(full) == 2
    assert session.get.call_count == 2


def test_items_datetime_filter_is_sent_to_the_api():
    """A run-narrowed listing is what turns a 571-page forecast walk into one page."""
    fetcher, session = _fetcher_with(_json_response(_page([_item()])))

    fetcher.items("ch.test.collection", datetime_filter="2026-05-23T15:00:00Z")

    url = session.get.call_args.args[0]
    assert "datetime=2026-05-23T15%3A00%3A00Z" in url


def test_items_cache_is_keyed_on_the_datetime_filter():
    """A run-narrowed listing is not a substitute for the full one."""
    fetcher, session = _fetcher_with(
        _json_response(_page([_item("narrow")])),
        _json_response(_page([_item("a"), _item("b")])),
    )

    narrowed = fetcher.items("ch.test.collection", cache=True, datetime_filter="2026-05-23T15:00:00Z")
    full = fetcher.items("ch.test.collection", cache=True)

    assert [i["id"] for i in narrowed] == ["narrow"]
    assert [i["id"] for i in full] == ["a", "b"]
    assert session.get.call_count == 2


def test_cached_listing_ttl_scales_with_what_the_walk_cost(monkeypatch):
    """An entry is held ~10x the time it took, so it is never cheaper to re-fetch.

    A fixed TTL got this backwards: the 171s forecast walk expired before any
    follow-up call could reuse it, so the one collection that needed caching
    never got it.
    """
    import foehn.fetch as fetch_mod

    clock = iter([0.0, 30.0, 30.0])  # walk start, walk end (30s elapsed), stored-at
    monkeypatch.setattr(fetch_mod.time, "monotonic", lambda: next(clock))
    fetcher, _ = _fetcher_with(_json_response(_page([_item()])))

    fetcher.items("ch.test.collection", cache=True)

    ((_, (_, ttl, _)),) = fetcher._listing_cache.items()
    assert ttl == 300.0  # 30s walk x _LISTING_TTL_FACTOR


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (0.01, 120.0),  # a cheap walk still stays cached long enough to be worth having
        (500.0, 1800.0),  # a very slow one is capped, so a new run is not hidden for hours
    ],
)
def test_cached_listing_ttl_is_floored_and_capped(monkeypatch, elapsed, expected):
    import foehn.fetch as fetch_mod

    clock = iter([0.0, elapsed, elapsed])
    monkeypatch.setattr(fetch_mod.time, "monotonic", lambda: next(clock))
    fetcher, _ = _fetcher_with(_json_response(_page([_item()])))

    fetcher.items("ch.test.collection", cache=True)

    ((_, (_, ttl, _)),) = fetcher._listing_cache.items()
    assert ttl == expected


def test_clear_cache_forces_a_refetch():
    fetcher, session = _fetcher_with(_json_response(_page([_item()])), _json_response(_page([_item()])))

    fetcher.items("ch.test.collection", cache=True)
    fetcher.clear_cache()
    fetcher.items("ch.test.collection", cache=True)

    assert session.get.call_count == 2


# --- collection ----------------------------------------------------------


def test_collection_returns_payload():
    fetcher, _ = _fetcher_with(_json_response({"id": "ch.test.collection", "title": "Test", "assets": {}}))

    result = fetcher.collection("ch.test.collection")

    assert result["title"] == "Test"


# --- get -----------------------------------------------------------------


def test_get_returns_body_and_etag():
    fetcher, _ = _fetcher_with(_body_response(b"payload", etag='W/"abc"'))

    fetched = fetcher.get(CSV_URL)

    assert fetched == Fetched(b"payload", 'W/"abc"')
    assert fetched.not_modified is False


def test_get_304_reports_not_modified():
    fetcher, session = _fetcher_with(_body_response(status_code=304))

    fetched = fetcher.get(CSV_URL, etag='W/"abc"')

    assert fetched.not_modified is True
    assert fetched.body == b""
    assert session.get.call_args.kwargs["headers"] == {"If-None-Match": 'W/"abc"'}


def test_get_without_etag_sends_no_conditional_header():
    fetcher, session = _fetcher_with(_body_response())

    fetcher.get(CSV_URL)

    assert session.get.call_args.kwargs["headers"] == {}


def test_get_rejects_untrusted_host():
    fetcher, session = _fetcher_with()

    with pytest.raises(ValueError, match="Untrusted download URL"):
        fetcher.get("https://evil.test/data.csv")
    assert session.get.call_count == 0


# --- stream --------------------------------------------------------------


def test_stream_writes_the_file(tmp_path):
    fetcher, _ = _fetcher_with(_stream_response((b"chunk1", b"chunk2")))
    target = tmp_path / "out.grib2"

    fetcher.stream(CSV_URL, target)

    assert target.read_bytes() == b"chunk1chunk2"


def test_stream_leaves_nothing_behind_when_it_fails(tmp_path):
    """A truncated download must not look like a complete one to the next run."""
    resp = _stream_response()
    resp.iter_content.return_value = iter([b"partial"])
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
    fetcher, _ = _fetcher_with(resp)
    target = tmp_path / "out.grib2"

    with pytest.raises(FetchError):
        fetcher.stream(CSV_URL, target)

    assert not target.exists()
    assert not (tmp_path / "out.grib2.part").exists()


def test_stream_rejects_untrusted_host(tmp_path):
    fetcher, _ = _fetcher_with()

    with pytest.raises(ValueError, match="Untrusted download URL"):
        fetcher.stream("https://evil.test/x.grib2", tmp_path / "x.grib2")


# --- errors --------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.ConnectionError("offline"),
        requests.exceptions.Timeout("slow"),
        requests.exceptions.HTTPError("500"),
    ],
)
def test_requests_failures_surface_as_fetch_error(exc):
    """Callers react to an unreachable API without knowing the adapter is requests."""
    fetcher, session = _fetcher_with()
    session.get.side_effect = exc

    with pytest.raises(FetchError):
        fetcher.get(CSV_URL)
    with pytest.raises(FetchError):
        fetcher.items("ch.test.collection")
    with pytest.raises(FetchError):
        fetcher.collection("ch.test.collection")


def test_fetch_error_keeps_the_cause():
    cause = requests.exceptions.ConnectionError("offline")
    fetcher, session = _fetcher_with()
    session.get.side_effect = cause

    with pytest.raises(FetchError) as excinfo:
        fetcher.get(CSV_URL)

    assert excinfo.value.__cause__ is cause


# --- sessions ------------------------------------------------------------


def test_each_thread_gets_its_own_session():
    """requests.Session is not fully thread-safe, so the pool must not share one."""
    fetcher = HttpFetcher()
    n = 4
    barrier = threading.Barrier(n)
    seen: list[requests.Session] = []
    lock = threading.Lock()

    def grab():
        barrier.wait()  # keep all four alive at once so thread ids stay distinct
        session = fetcher._session()
        with lock:
            seen.append(session)

    threads = [threading.Thread(target=grab) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({id(s) for s in seen}) == n


def test_same_thread_reuses_its_session():
    fetcher = HttpFetcher()

    assert fetcher._session() is fetcher._session()


def test_session_retries_transient_statuses():
    adapter = HttpFetcher()._session().get_adapter("https://data.geo.admin.ch")
    retry = adapter.max_retries

    assert retry.total == 3
    assert 429 in retry.status_forcelist
    assert 503 in retry.status_forcelist


# --- the process-wide default --------------------------------------------


def test_default_fetcher_is_built_once():
    with patch.object(__import__("foehn.fetch", fromlist=["fetch"]), "_default_fetcher", None):
        first = default_fetcher()
        second = default_fetcher()
    assert first is second
    assert isinstance(first, HttpFetcher)


def test_validate_stac_url_rejects_untrusted_urls():
    url = "https://data.geo.admin.ch/api/stac/v1/page2"
    assert validate_stac_url(url) == url

    for bad in ("https://example.test/api/stac/v1/page2", "http://data.geo.admin.ch/api/stac/v1/page2"):
        with pytest.raises(ValueError, match="Untrusted STAC URL"):
            validate_stac_url(bad)
