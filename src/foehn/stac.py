"""Interact with the swisstopo STAC API used by MeteoSwiss."""

from __future__ import annotations

import requests

from foehn.collections import STAC_API_BASE


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
        verbose: Print progress to stdout.
    """
    items: list[dict] = []
    url: str | None = f"{STAC_API_BASE}/collections/{collection_id}/items?limit=100"
    page = 0

    while url:
        page += 1
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        items.extend(features)

        # After first page, check if any item has CSV assets — if not, stop
        if require_csv and page == 1 and features:
            has_csv = any(
                href.endswith(".csv")
                for feat in features
                for href in (a.get("href", "") for a in feat.get("assets", {}).values())
            )
            if not has_csv:
                if verbose:
                    print(
                        "  No CSV assets found on first page — skipping remaining pages",
                        flush=True,
                    )
                return items

        url = next(
            (link["href"] for link in data.get("links", []) if link.get("rel") == "next"),
            None,
        )

    return items


def get_collection_metadata(collection_id: str) -> dict:
    """Fetch collection-level metadata (title, description, assets)."""
    resp = requests.get(f"{STAC_API_BASE}/collections/{collection_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()
