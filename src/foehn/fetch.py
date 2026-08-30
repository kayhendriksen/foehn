"""Every HTTP call foehn makes to MeteoSwiss.

One port, four operations. Retry policy, per-thread sessions, URL validation,
STAC pagination and atomic streaming to disk all live behind it, so no other
module builds a ``requests.Session`` or knows what a STAC ``next`` link is.

Two adapters satisfy it: :class:`HttpFetcher` in production, and an in-memory
one in the test suite. Callers inside foehn take a ``Fetcher`` explicitly; the
public functions construct the default and thread it down, so ``foehn.load()``
and friends never mention it.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from foehn._urls import validate_download_href, validate_stac_url
from foehn.collections import STAC_API_BASE

logger = logging.getLogger(__name__)

# Default concurrency for per-asset downloads, and the size of each session's
# connection pool. Kept modest to stay polite on the MeteoSwiss/CSCS CDNs — they
# handle bursts fine but we don't need to hammer them.
DEFAULT_WORKERS = 8

# How long a cached listing stays valid, derived from what the walk cost so an
# entry is never cheaper to re-fetch than to keep: held for ~10x the time it took
# to build, floored and capped. Cost varies enormously — smn is 2 pages and
# radar_precip 1, but forecast_icon_ch1 is 571 pages (~170s), because the forecast
# collections carry one item per file. A fixed TTL got this backwards: at 120s the
# 171s forecast walk expired before any follow-up call could reuse it, so the one
# collection that truly needed caching never got it. The floor keeps a cheap walk
# cached long enough to be worth having; the cap keeps a new forecast run or radar
# timestep from being hidden for too long.
_LISTING_TTL_FACTOR = 10.0
_LISTING_TTL_MIN_SECONDS = 120.0
_LISTING_TTL_MAX_SECONDS = 1800.0


class FetchError(Exception):
    """Any network-level failure crossing the port.

    The port's own error type on purpose: callers that need to react to an
    unreachable API (the gridded read path falls back to its local cache) should
    not have to know the adapter is built on ``requests``.
    """


@dataclass(frozen=True)
class Fetched:
    """One fetched body, plus the cache metadata the caller needs to store.

    ``not_modified`` is the 304 case: ``body`` is empty and ``etag`` is None,
    because the server sent neither. Returning this rather than ``None`` keeps
    the caller's happy path from having to guard an Optional it will forget.
    """

    body: bytes
    etag: str | None = None
    not_modified: bool = False


class Fetcher(Protocol):
    """The port. Anything foehn needs from the network is one of these four."""

    def get(self, url: str, *, etag: str | None = None, timeout: int = 60) -> Fetched:
        """Fetch a download URL's body, conditionally when ``etag`` is given."""
        ...

    def stream(self, url: str, path: Path, *, timeout: int = 120) -> None:
        """Stream a download URL to ``path``, atomically."""
        ...

    def items(
        self,
        collection_id: str,
        *,
        cache: bool = False,
        datetime_filter: str | None = None,
        max_items: int | None = None,
    ) -> list[dict]:
        """Return STAC items in a collection, following pagination.

        ``cache`` permits serving a recent listing instead of re-walking; how
        long one stays valid is the fetcher's decision, not the caller's.
        ``datetime_filter`` narrows the listing server-side. ``max_items`` stops
        paginating early, newest first.
        """
        ...

    def collection(self, collection_id: str, *, timeout: int = 30) -> dict:
        """Return a STAC collection's own metadata (title, description, assets)."""
        ...


@contextlib.contextmanager
def _as_fetch_error(url: str):
    """Re-raise ``requests`` failures as :class:`FetchError`, keeping the cause."""
    try:
        yield
    except requests.exceptions.RequestException as exc:
        raise FetchError(f"{type(exc).__name__} fetching {url}: {exc}") from exc


class HttpFetcher:
    """The production adapter: ``requests`` against the swisstopo STAC API and CDNs."""

    def __init__(
        self,
        *,
        retries: int = 3,
        backoff_factor: float = 1.0,
        status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
        pool_maxsize: int = DEFAULT_WORKERS,
    ) -> None:
        self._retries = retries
        self._backoff_factor = backoff_factor
        self._status_forcelist = status_forcelist
        self._pool_maxsize = pool_maxsize
        # requests.Session is not fully thread-safe (cookie jar, headers), so
        # every worker thread gets its own. They live as long as this fetcher.
        self._local = threading.local()
        # (collection, datetime filter) -> (stored_at, ttl, items). Keyed on the
        # filter too: a run-narrowed listing is not a substitute for the full one.
        self._listing_cache: dict[tuple[str, str | None], tuple[float, float, list[dict]]] = {}

    # --- session management -------------------------------------------------

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self._retries,
            backoff_factor=self._backoff_factor,
            status_forcelist=self._status_forcelist,
            allowed_methods=["GET"],
            raise_on_status=False,
            connect=self._retries,
            read=self._retries,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=self._pool_maxsize,
            pool_maxsize=self._pool_maxsize,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._build_session()
            self._local.session = session
        return session

    # --- the port -----------------------------------------------------------

    def get(self, url: str, *, etag: str | None = None, timeout: int = 60) -> Fetched:
        validate_download_href(url)
        headers = {"If-None-Match": etag} if etag else {}
        with _as_fetch_error(url):
            resp = self._session().get(url, headers=headers, timeout=timeout)
            if resp.status_code == 304:
                return Fetched(b"", None, not_modified=True)
            resp.raise_for_status()
            return Fetched(resp.content, resp.headers.get("ETag"))

    def stream(self, url: str, path: Path, *, timeout: int = 120) -> None:
        """Stream to a sibling ``.part`` file, then ``Path.replace`` onto ``path``.

        A timeout or connection drop mid-stream leaves nothing at ``path``, so
        the next run's existence/mtime check can't mistake a truncated download
        for a complete one.
        """
        validate_download_href(url)
        tmp = path.with_name(path.name + ".part")
        try:
            with _as_fetch_error(url), self._session().get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        fh.write(chunk)
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def items(
        self,
        collection_id: str,
        *,
        cache: bool = False,
        datetime_filter: str | None = None,
        max_items: int | None = None,
    ) -> list[dict]:
        """Paginate through a collection's items.

        ``cache`` is for the gridded read path, which lists a collection on every
        open — including repeat opens of an already-cached file — because it
        verifies ``match`` against the collection rather than the local cache. The
        download paths leave it off: noticing what changed upstream is their whole
        job, and a cache there would suppress exactly the updates they look for.

        How long an entry stays valid is derived from what the walk cost, so it is
        never cheaper to re-fetch than to keep (see ``_LISTING_TTL_FACTOR``). Only
        the fetcher can measure that, which is why callers say whether a stale
        listing is acceptable and not for how long.

        ``max_items`` stops after the first page(s) covering that many items. The
        ephemeral collections only ever want the newest, so walking the whole
        listing is pure cost. Truncated listings are never cached, so a partial
        result is never served later as if it were complete.
        """
        cacheable = cache and max_items is None
        key = (collection_id, datetime_filter)
        if cacheable:
            hit = self._listing_cache.get(key)
            if hit is not None and time.monotonic() - hit[0] < hit[1]:
                return hit[2]

        collected: list[dict] = []
        url: str | None = validate_stac_url(f"{STAC_API_BASE}/collections/{collection_id}/items?limit=100")
        if datetime_filter:
            url += f"&datetime={quote(datetime_filter, safe='')}"
        session = self._session()
        started = time.monotonic()
        while url:
            with _as_fetch_error(url):
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            collected.extend(data.get("features", []))
            if max_items is not None and len(collected) >= max_items:
                return collected[:max_items]
            next_href = next(
                (link["href"] for link in data.get("links", []) if link.get("rel") == "next"),
                None,
            )
            url = validate_stac_url(next_href) if next_href else None

        if cacheable:
            elapsed = time.monotonic() - started
            ttl = min(max(elapsed * _LISTING_TTL_FACTOR, _LISTING_TTL_MIN_SECONDS), _LISTING_TTL_MAX_SECONDS)
            self._listing_cache[key] = (time.monotonic(), ttl, collected)
        return collected

    def collection(self, collection_id: str, *, timeout: int = 30) -> dict:
        url = validate_stac_url(f"{STAC_API_BASE}/collections/{collection_id}")
        with _as_fetch_error(url):
            resp = self._session().get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()

    def clear_cache(self) -> None:
        """Drop any memoised listings. Used between tests."""
        self._listing_cache.clear()


# One fetcher for the process. Built lazily so importing foehn opens no sockets,
# and shared so a long-lived caller (the MCP server makes many small STAC calls)
# reuses connections instead of building a session per call.
_default_fetcher: Fetcher | None = None
_default_lock = threading.Lock()


def default_fetcher() -> Fetcher:
    """Return the process-wide fetcher, building it on first use."""
    global _default_fetcher
    if _default_fetcher is None:
        with _default_lock:
            if _default_fetcher is None:
                _default_fetcher = HttpFetcher()
    return _default_fetcher
