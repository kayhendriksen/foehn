"""Tests for STAC API client (all HTTP calls mocked)."""

from unittest.mock import MagicMock, patch

from foehn.stac import get_collection_items, get_collection_metadata


def _page(features, next_url=None):
    """Build a minimal STAC FeatureCollection response dict."""
    links = [{"rel": "next", "href": next_url}] if next_url else []
    return {"features": features, "links": links}


def _item(csv_url):
    return {"id": "item-1", "assets": {"data": {"href": csv_url}}, "properties": {}}


def _mock_get(*pages):
    """Return a mock for requests.get that cycles through the given page dicts."""
    responses = []
    for page in pages:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = page
        responses.append(resp)
    mock = MagicMock(side_effect=responses)
    return mock


# --- get_collection_items ---


@patch("foehn.stac.requests.get")
def test_get_collection_items_single_page(mock_get):
    items = [_item("https://example.com/data.csv")]
    mock_get.side_effect = _mock_get(_page(items)).side_effect

    result = get_collection_items("ch.test.collection", verbose=False)

    assert len(result) == 1
    assert result[0]["id"] == "item-1"


@patch("foehn.stac.requests.get")
def test_get_collection_items_pagination(mock_get):
    page1 = _page([_item("https://example.com/a.csv")], next_url="https://example.com/page2")
    page2 = _page([_item("https://example.com/b.csv")])
    mock_get.side_effect = _mock_get(page1, page2).side_effect

    result = get_collection_items("ch.test.collection", verbose=False)

    assert len(result) == 2
    assert mock_get.call_count == 2


@patch("foehn.stac.requests.get")
def test_get_collection_items_stops_early_when_no_csv(mock_get):
    """require_csv=True should stop after first page when no .csv assets found."""
    item_no_csv = {"id": "item-nc", "assets": {"data": {"href": "https://example.com/grid.nc"}}, "properties": {}}
    page1 = _page([item_no_csv], next_url="https://example.com/page2")
    mock_get.side_effect = _mock_get(page1).side_effect

    result = get_collection_items("ch.test.collection", require_csv=True, verbose=False)

    assert mock_get.call_count == 1  # did not follow next link
    assert len(result) == 1


@patch("foehn.stac.requests.get")
def test_get_collection_items_require_csv_false_follows_next(mock_get):
    item_nc = {"id": "item-nc", "assets": {"data": {"href": "https://example.com/grid.nc"}}, "properties": {}}
    page1 = _page([item_nc], next_url="https://example.com/page2")
    page2 = _page([item_nc])
    mock_get.side_effect = _mock_get(page1, page2).side_effect

    result = get_collection_items("ch.test.collection", require_csv=False, verbose=False)

    assert mock_get.call_count == 2
    assert len(result) == 2


@patch("foehn.stac.requests.get")
def test_get_collection_items_empty_collection(mock_get):
    mock_get.side_effect = _mock_get(_page([])).side_effect

    result = get_collection_items("ch.test.collection", verbose=False)

    assert result == []


# --- get_collection_metadata ---


@patch("foehn.stac.requests.get")
def test_get_collection_metadata_returns_dict(mock_get):
    payload = {"id": "ch.test.collection", "title": "Test", "assets": {}}
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload
    mock_get.return_value = resp

    result = get_collection_metadata("ch.test.collection")

    assert result["id"] == "ch.test.collection"
    assert result["title"] == "Test"
