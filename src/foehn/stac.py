"""Interact with the swisstopo STAC API used by MeteoSwiss."""

from __future__ import annotations

import logging

from foehn._urls import clean_href, validate_stac_url
from foehn.collections import STAC_API_BASE

logger = logging.getLogger(__name__)


def get_collection_items(
    collection_id: str,
    require_csv: bool = True,
    *,
    verbose: bool = True,
) -> list[dict]:
    """Paginate through all items in a STAC collection.

    Args:
        collection_id: The STAC collection ID.
        require_csv: If True and the first page has no CSV assets, stop early.
        verbose: Log progress at INFO level.
    """
    # Lazy import to avoid a circular module-level import (client imports stac).
    from foehn.client import _retry_session

    items: list[dict] = []
    url: str | None = f"{STAC_API_BASE}/collections/{collection_id}/items?limit=100"
    page = 0

    with _retry_session() as session:
        while url:
            page += 1
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            items.extend(features)

            # After first page, check if any item has CSV assets — if not, stop
            if require_csv and page == 1 and features:
                has_csv = any(
                    clean_href(href).endswith(".csv")
                    for feat in features
                    for href in (a.get("href", "") for a in feat.get("assets", {}).values())
                )
                if not has_csv:
                    if verbose:
                        logger.info("  No CSV assets found on first page — skipping remaining pages")
                    return items

            next_href = next(
                (link["href"] for link in data.get("links", []) if link.get("rel") == "next"),
                None,
            )
            url = validate_stac_url(next_href) if next_href else None

    return items


def get_collection_metadata(collection_id: str) -> dict:
    """Fetch collection-level metadata (title, description, assets)."""
    from foehn.client import _retry_session

    with _retry_session() as session:
        resp = session.get(f"{STAC_API_BASE}/collections/{collection_id}", timeout=30)
        resp.raise_for_status()
        return resp.json()
