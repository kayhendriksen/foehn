"""Trusted URL validation helpers for MeteoSwiss/STAC endpoints."""

from __future__ import annotations

from urllib.parse import urlparse

STAC_DOMAINS = frozenset({"data.geo.admin.ch"})
DOWNLOAD_DOMAINS = frozenset({"data.geo.admin.ch", "opendata.swiss", "rgw.cscs.ch"})


def _validate_https_url(url: str, allowed_domains: frozenset[str], label: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in allowed_domains:
        raise ValueError(f"Untrusted {label} URL: {url!r}")
    return url


def validate_stac_url(url: str) -> str:
    """Raise ValueError if *url* is not a trusted STAC API URL."""
    return _validate_https_url(url, STAC_DOMAINS, "STAC")


def validate_download_href(href: str) -> str:
    """Raise ValueError if *href* is not a trusted download URL."""
    return _validate_https_url(href, DOWNLOAD_DOMAINS, "download")
