"""Trusted URL validation helpers for MeteoSwiss/STAC endpoints."""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import urlparse, urlsplit, urlunsplit

STAC_DOMAINS = frozenset({"data.geo.admin.ch"})
DOWNLOAD_DOMAINS = frozenset({"data.geo.admin.ch", "opendata.swiss", "rgw.cscs.ch"})


def safe_url(url: str) -> str:
    """The diagnostic form of a URL, without userinfo, query or fragment."""
    try:
        parsed = urlsplit(url)
        return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
    except ValueError:
        return "<invalid URL>"


def redact_urls(message: str) -> str:
    """Redact full URLs repeated inside a downstream diagnostic message.

    Transport exceptions are handled separately: requests/urllib3 can also
    print a credential-bearing *relative* request target, so their raw messages
    must never be used as public FetchError text.
    """
    return re.sub(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>\"']+", lambda match: safe_url(match.group()), message)


def _validate_https_url(url: str, allowed_domains: frozenset[str], label: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        raise ValueError(f"Untrusted {label} URL: {safe_url(url)!r}") from None
    if parsed.scheme != "https" or parsed.hostname not in allowed_domains:
        raise ValueError(f"Untrusted {label} URL: {safe_url(url)!r}")
    return url


def validate_stac_url(url: str) -> str:
    """Raise ValueError if *url* is not a trusted STAC API URL."""
    return _validate_https_url(url, STAC_DOMAINS, "STAC")


def validate_download_href(href: str) -> str:
    """Raise ValueError if *href* is not a trusted download URL."""
    return _validate_https_url(href, DOWNLOAD_DOMAINS, "download")


def _clean_href(href: str) -> str:
    """Return *href* without its query string or fragment.

    STAC asset hrefs may carry a query (``?token=...``), so every suffix or
    filename check has to strip it first — an ``endswith(".csv")`` on the raw
    href silently drops those assets.
    """
    return href.split("#", 1)[0].split("?", 1)[0]


def asset_filename(href: str) -> str:
    """Return just the filename of an asset href, ignoring any query string."""
    return _clean_href(href).rsplit("/", 1)[-1]


def is_plain_filename(name: str) -> bool:
    r"""Whether *name* is one ordinary file name, safe to join onto a directory on any OS.

    An asset's name is the last segment of an href upstream chose, and it is
    joined straight onto Bronze. A backslash is a separator on Windows, so
    ``..\..\evil.csv`` — a single URL segment — named a file three directories
    up there; ``C:evil.csv`` is drive-relative. A leading dot is refused too: it
    could name one of foehn's own markers beside the data, and MeteoSwiss
    publishes no hidden files.
    """
    if not name or name.startswith(".") or "\0" in name:
        return False
    return PurePosixPath(name).name == name and PureWindowsPath(name).name == name
