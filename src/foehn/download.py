"""Download MeteoSwiss data from the STAC API and opendata.swiss."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

from foehn.collections import (
    CLIMATE_NORMALS_ZIP_URL,
    COLLECTIONS,
    FORECAST_CSV_COLLECTIONS,
    STAC_API_BASE,
)
from foehn.stac import get_collection_items, get_collection_metadata

# --- State files (ETags + last-run timestamp) ---


def load_etags(data_dir: Path) -> dict:
    path = data_dir / "_etags.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_etags(data_dir: Path, etags: dict):
    path = data_dir / "_etags.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(etags, indent=2))


def load_last_run(data_dir: Path) -> str | None:
    """Return ISO timestamp of last successful run, or None."""
    path = data_dir / "_last_run.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data.get("timestamp")
    return None


def save_last_run(data_dir: Path):
    path = data_dir / "_last_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat()}))


# --- CSV downloads ---


def download_collection(
    collection_key: str,
    output_dir: Path,
    data_types: list[str] | None = None,
    since: str | None = None,
):
    """Download CSVs for a collection.

    Args:
        collection_key: Key from COLLECTIONS (e.g. "smn").
        output_dir: Root directory for raw downloads (files go to output_dir/<key>/).
        data_types: List of "historical", "recent", "now". Defaults to ["recent"].
        since: ISO timestamp — only process items updated after this time.
    """
    if data_types is None:
        data_types = ["recent"]

    collection_id = COLLECTIONS[collection_key]
    out_dir = output_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    etags = load_etags(output_dir.parent)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Collection: {collection_id}", flush=True)
    print(f"Data types: {data_types}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print(f"{'=' * 60}", flush=True)

    items = get_collection_items(collection_id)
    print(f"  Found {len(items)} items", flush=True)

    # Filter to items updated since last run
    if since:
        items = [item for item in items if item.get("properties", {}).get("updated", "") > since]
        print(f"  {len(items)} items updated since last run", flush=True)
        if not items:
            print("  Nothing changed — skipping", flush=True)
            return

    # For forecast collections, only keep the latest item (newest forecast run)
    if collection_key in FORECAST_CSV_COLLECTIONS and items:
        items.sort(
            key=lambda x: x.get("properties", {}).get("datetime", x.get("id", "")),
            reverse=True,
        )
        items = items[:1]
        print(f"  Using latest forecast: {items[0].get('id', '?')}", flush=True)

    # Collect matching CSV assets
    skip_data_type_filter = collection_key in FORECAST_CSV_COLLECTIONS
    csv_assets = []
    for item in items:
        assets = item.get("assets", {})
        for _asset_key, asset_info in assets.items():
            href = asset_info.get("href", "")
            if not href.endswith(".csv"):
                continue
            if not skip_data_type_filter:
                has_time_slice = any(ts in href for ts in ("historical", "recent", "now"))
                if has_time_slice and not any(dt in href for dt in data_types):
                    continue
            csv_assets.append((href, asset_info))

    print(f"  {len(csv_assets)} CSV files to process", flush=True)

    downloaded = 0
    skipped = 0
    for i, (href, _asset_info) in enumerate(csv_assets, 1):
        filename = href.split("/")[-1]
        filepath = out_dir / filename

        # Use ETag to skip files that haven't changed
        headers = {}
        old_etag = etags.get(href)
        if old_etag and filepath.exists():
            headers["If-None-Match"] = old_etag

        resp = requests.get(href, headers=headers, timeout=60)
        if resp.status_code == 304:
            skipped += 1
            continue
        resp.raise_for_status()

        # MeteoSwiss CSVs are Windows-1252; re-encode to UTF-8
        try:
            content = resp.content.decode("windows-1252")
        except UnicodeDecodeError:
            content = resp.content.decode("utf-8")
        filepath.write_text(content, encoding="utf-8")

        new_etag = resp.headers.get("ETag")
        if new_etag:
            etags[href] = new_etag

        downloaded += 1
        print(f"  [{i}/{len(csv_assets)}] Downloaded: {filename}", flush=True)

    save_etags(output_dir.parent, etags)
    if skipped:
        print(f"  Skipped {skipped} unchanged files", flush=True)
    print(f"  Done — {downloaded} files downloaded", flush=True)


# --- Metadata downloads ---


def download_metadata(collection_key: str, output_dir: Path):
    """Download collection-level metadata files (stations, parameters, inventory)."""
    collection_id = COLLECTIONS[collection_key]
    out_dir = output_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    coll = get_collection_metadata(collection_id)
    assets = coll.get("assets", {})
    if not assets:
        return

    downloaded = 0
    for _key, asset_info in assets.items():
        href = asset_info.get("href", "")
        if not href.endswith(".csv"):
            continue
        filename = href.split("/")[-1]
        filepath = out_dir / filename

        resp = requests.get(href, timeout=60)
        resp.raise_for_status()
        try:
            content = resp.content.decode("windows-1252")
        except UnicodeDecodeError:
            content = resp.content.decode("utf-8")
        filepath.write_text(content, encoding="utf-8")
        downloaded += 1
        print(f"  Metadata: {filename}", flush=True)

    if downloaded:
        print(f"  {downloaded} metadata files downloaded", flush=True)


# --- GRIB2 / HDF5 downloads ---


def download_grib2(
    collection_key: str,
    output_dir: Path,
    since: str | None = None,
):
    """Download GRIB2/HDF5 binary files (latest page only)."""
    collection_id = COLLECTIONS[collection_key]
    out_dir = output_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"GRIB2 Collection: {collection_id}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Only fetch first page — forecast/radar data is ephemeral
    url = f"{STAC_API_BASE}/collections/{collection_id}/items?limit=100"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("features", [])
    print(f"  Found {len(items)} items (latest page)", flush=True)

    if since:
        items = [item for item in items if item.get("properties", {}).get("updated", "") > since]
        print(f"  {len(items)} items updated since last run", flush=True)
        if not items:
            print("  Nothing changed — skipping", flush=True)
            return

    binary_assets = []
    for item in items:
        assets = item.get("assets", {})
        for _asset_key, asset_info in assets.items():
            href = asset_info.get("href", "")
            clean = href.split("?")[0]
            # Accept grib2, h5, and other binary formats
            if any(clean.endswith(ext) for ext in (".grib2", ".h5", ".hdf5")):
                binary_assets.append((href, clean.split("/")[-1]))

    print(f"  {len(binary_assets)} binary files to download", flush=True)

    downloaded = 0
    for i, (href, filename) in enumerate(binary_assets, 1):
        filepath = out_dir / filename
        if filepath.exists():
            continue

        with requests.get(href, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with filepath.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
        downloaded += 1
        print(f"  [{i}/{len(binary_assets)}] Downloaded: {filename}", flush=True)

    print(f"\n  Done — {downloaded} binary files downloaded", flush=True)


# --- NetCDF / GeoTIFF / ZIP downloads ---


def download_netcdf(collection_key: str, output_dir: Path):
    """Download NetCDF, GeoTIFF, and ZIP files for spatial/static collections."""
    collection_id = COLLECTIONS[collection_key]
    out_dir = output_dir / collection_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"NetCDF Collection: {collection_id}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print(f"{'=' * 60}", flush=True)

    items = get_collection_items(collection_id, require_csv=False)
    print(f"  Found {len(items)} items", flush=True)

    downloaded = 0
    for item in items:
        assets = item.get("assets", {})
        for _asset_key, asset_info in assets.items():
            href = asset_info.get("href", "")
            clean = href.split("?")[0]
            if not (clean.endswith(".nc") or clean.endswith(".tif") or clean.endswith(".zip")):
                continue
            filename = clean.split("/")[-1]
            filepath = out_dir / filename
            if filepath.exists():
                continue
            with requests.get(href, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with filepath.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
            downloaded += 1
            print(f"  Downloaded: {filename}", flush=True)

    print(f"  Done — {downloaded} files downloaded", flush=True)


# --- C6 climate normals ZIP ---


def download_climate_normals_zip(output_dir: Path, force: bool = False):
    """Download C6 climate normals ZIP from opendata.swiss and extract."""
    out_dir = output_dir / "climate_normals"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / "normwerte.zip"

    if filepath.exists() and not force:
        print("\n  Climate normals ZIP already downloaded — skipping", flush=True)
        return

    print(f"\n{'=' * 60}", flush=True)
    print("Climate normals (C6): downloading from opendata.swiss", flush=True)
    print(f"{'=' * 60}", flush=True)

    resp = requests.get(CLIMATE_NORMALS_ZIP_URL, timeout=120)
    resp.raise_for_status()
    filepath.write_bytes(resp.content)
    print(f"  Downloaded: normwerte.zip ({len(resp.content) / 1024:.0f} KB)", flush=True)

    with zipfile.ZipFile(filepath, "r") as zf:
        zf.extractall(out_dir)
        print(f"  Extracted {len(zf.namelist())} files", flush=True)
