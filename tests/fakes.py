"""The in-memory adapter at foehn's network port.

The second adapter that makes the seam in ``foehn.fetch`` real: production uses
``HttpFetcher``, the suite uses this. Tests wire up the STAC listings and bodies
a call should see, pass it in (or install it as the default via the ``fetcher``
fixture), and assert on outcomes rather than on which HTTP calls were made.
"""

from __future__ import annotations

from pathlib import Path

from foehn.fetch import Fetched, FetchError


class InMemoryFetcher:
    """A Fetcher backed by dicts. Serves what it was given, records what it was asked."""

    def __init__(
        self,
        *,
        items: dict[str, list[dict]] | None = None,
        collections: dict[str, dict] | None = None,
        bodies: dict[str, bytes] | None = None,
        etags: dict[str, str] | None = None,
    ) -> None:
        self.items_by_collection: dict[str, list[dict]] = dict(items or {})
        self.collections: dict[str, dict] = dict(collections or {})
        self.bodies: dict[str, bytes] = dict(bodies or {})
        self.etags: dict[str, str] = dict(etags or {})
        self.errors: dict[str, Exception] = {}

        # Fallbacks for the common case where a test cares about one collection
        # and every asset in it serves the same payload. Patching the old
        # module-level functions had exactly these semantics: whatever you set,
        # every call saw it.
        self.any_items: list[dict] | None = None
        self.any_collection: dict | None = None
        self.default_body: bytes | str | None = None
        self.default_etag: str | None = None
        # Called instead of writing default_body, for tests that need to observe
        # or delay the write (concurrency, partial-cache repair).
        self.stream_hook = None

        # What was asked for, in order. For the handful of tests that genuinely
        # care about traffic (ETag skips, the listing cache, first-page-only).
        self.gets: list[str] = []
        self.streams: list[str] = []
        self.listings: list[tuple[str, bool, str | None, int | None]] = []
        self.collection_calls: list[str] = []

    # --- wiring ---------------------------------------------------------

    def add_body(self, url: str, body: bytes | str, *, etag: str | None = None) -> str:
        self.bodies[url] = body
        if etag is not None:
            self.etags[url] = etag
        return url

    def add_items(self, collection_id: str, items: list[dict]) -> None:
        self.items_by_collection[collection_id] = items

    def add_collection(self, collection_id: str, payload: dict) -> None:
        self.collections[collection_id] = payload

    def fail(self, key: str, exc: Exception | None = None) -> None:
        """Make the next access to *key* (a URL or collection id) raise."""
        self.errors[key] = exc or FetchError(f"boom fetching {key}")

    def _check(self, key: str) -> None:
        exc = self.errors.get(key)
        if exc is not None:
            raise exc

    # --- the port -------------------------------------------------------

    def _body_for(self, url: str) -> bytes:
        body = self.bodies.get(url, self.default_body)
        if body is None:
            raise FetchError(f"no body registered for {url}")
        return body.encode("utf-8") if isinstance(body, str) else body

    def get(self, url: str, *, etag: str | None = None, timeout: int = 60) -> Fetched:
        self.gets.append(url)
        self._check(url)
        body = self._body_for(url)
        stored = self.etags.get(url, self.default_etag)
        if etag is not None and stored is not None and etag == stored:
            return Fetched(b"", None, not_modified=True)
        return Fetched(body, stored)

    def stream(self, url: str, path: Path, *, timeout: int = 120) -> None:
        self.streams.append(url)
        self._check(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.stream_hook is not None:
            self.stream_hook(url, path)
            return
        path.write_bytes(self._body_for(url))

    def items(
        self,
        collection_id: str,
        *,
        cache: bool = False,
        datetime_filter: str | None = None,
        max_items: int | None = None,
    ) -> list[dict]:
        self.listings.append((collection_id, cache, datetime_filter, max_items))
        self._check(collection_id)
        found = self.items_by_collection.get(collection_id)
        if found is None:
            found = self.any_items if self.any_items is not None else []
        found = list(found)
        return found[:max_items] if max_items is not None else found

    def collection(self, collection_id: str, *, timeout: int = 30) -> dict:
        self.collection_calls.append(collection_id)
        self._check(collection_id)
        payload = self.collections.get(collection_id, self.any_collection)
        if payload is None:
            raise FetchError(f"no collection registered for {collection_id}")
        return payload


# --- STAC shape builders ------------------------------------------------


def stac_asset(href: str, **extra) -> dict:
    return {"href": href, **extra}


def stac_item(item_id: str = "item-1", *hrefs: str, updated: str = "2026-01-01T00:00:00Z", **props) -> dict:
    """A STAC item carrying one asset per href, keyed ``asset-0``, ``asset-1``, …"""
    assets = {f"asset-{i}": stac_asset(href) for i, href in enumerate(hrefs)}
    return {"id": item_id, "assets": assets, "properties": {"updated": updated, **props}}


def stac_collection(collection_id: str = "ch.test.collection", *hrefs: str, **extra) -> dict:
    """A collection-level payload whose assets are keyed by their filename."""
    assets = {href.split("?")[0].split("/")[-1]: stac_asset(href) for href in hrefs}
    return {"id": collection_id, "title": "Test", "assets": assets, **extra}
