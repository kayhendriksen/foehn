"""Tests for STAC API client (all HTTP calls mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from foehn._urls import validate_stac_url
from foehn.stac import get_collection_items, get_collection_metadata


def _page(features, next_url=None):
    """Build a minimal STAC FeatureCollection response dict."""
    links = [{"rel": "next", "href": next_url}] if next_url else []
    return {"features": features, "links": links}


def _item(csv_url):
    return {"id": "item-1", "assets": {"data": {"href": csv_url}}, "properties": {}}


def _session_get(mock_retry, *pages):
    """Wire a patched ``_retry_session`` for ``with _retry_session() as s: s.get(...)``.

    Returns the session's ``get`` mock, configured to cycle through the given
    page dicts as JSON responses.
    """
    session = mock_retry.return_value
    session.__enter__.return_value = session
    responses = []
    for page in pages:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = page
        responses.append(resp)
    if responses:
        session.get.side_effect = responses
    return session.get


# --- get_collection_items ---


@patch("foehn.client._retry_session")
def test_get_collection_items_single_page(mock_retry):
    items = [_item("https://data.geo.admin.ch/data.csv")]
    mock_get = _session_get(mock_retry, _page(items))

    result = get_collection_items("ch.test.collection", verbose=False)

    assert len(result) == 1
    assert result[0]["id"] == "item-1"
    assert mock_get.call_count == 1


@patch("foehn.client._retry_session")
def test_get_collection_items_pagination(mock_retry):
    page1 = _page([_item("https://data.geo.admin.ch/a.csv")], next_url="https://data.geo.admin.ch/page2")
    page2 = _page([_item("https://data.geo.admin.ch/b.csv")])
    mock_get = _session_get(mock_retry, page1, page2)

    result = get_collection_items("ch.test.collection", verbose=False)

    assert len(result) == 2
    assert mock_get.call_count == 2


@patch("foehn.client._retry_session")
def test_get_collection_items_stops_early_when_no_csv(mock_retry):
    """require_csv=True should stop after first page when no .csv assets found."""
    item_no_csv = {"id": "item-nc", "assets": {"data": {"href": "https://data.geo.admin.ch/grid.nc"}}, "properties": {}}
    page1 = _page([item_no_csv], next_url="https://data.geo.admin.ch/page2")
    mock_get = _session_get(mock_retry, page1)

    result = get_collection_items("ch.test.collection", require_csv=True, verbose=False)

    assert mock_get.call_count == 1  # did not follow next link
    assert len(result) == 1


@patch("foehn.client._retry_session")
def test_get_collection_items_require_csv_false_follows_next(mock_retry):
    item_nc = {"id": "item-nc", "assets": {"data": {"href": "https://data.geo.admin.ch/grid.nc"}}, "properties": {}}
    page1 = _page([item_nc], next_url="https://data.geo.admin.ch/page2")
    page2 = _page([item_nc])
    mock_get = _session_get(mock_retry, page1, page2)

    result = get_collection_items("ch.test.collection", require_csv=False, verbose=False)

    assert mock_get.call_count == 2
    assert len(result) == 2


@patch("foehn.client._retry_session")
def test_get_collection_items_empty_collection(mock_retry):
    _session_get(mock_retry, _page([]))

    result = get_collection_items("ch.test.collection", verbose=False)

    assert result == []


def test_validate_stac_url_rejects_untrusted_urls():
    url = "https://data.geo.admin.ch/api/stac/v1/page2"
    assert validate_stac_url(url) == url

    for url in ("https://example.test/api/stac/v1/page2", "http://data.geo.admin.ch/api/stac/v1/page2"):
        with pytest.raises(ValueError, match="Untrusted STAC URL"):
            validate_stac_url(url)


# --- get_collection_metadata ---


@patch("foehn.client._retry_session")
def test_get_collection_metadata_returns_dict(mock_retry):
    payload = {"id": "ch.test.collection", "title": "Test", "assets": {}}
    _session_get(mock_retry, payload)

    result = get_collection_metadata("ch.test.collection")

    assert result["id"] == "ch.test.collection"
    assert result["title"] == "Test"
