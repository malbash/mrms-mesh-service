from __future__ import annotations

import gzip
import hashlib
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
import pygrib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

APP_NAME = "mrms-mesh-service"
APP_VERSION = "1.4.0"

AWS_BUCKET_BASE = os.getenv("MRMS_AWS_BASE", "https://noaa-mrms-pds.s3.amazonaws.com")
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/mrms-cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "45"))
DOWNLOAD_TIMEOUT_SECONDS = float(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "120"))

FILENAME_PATTERNS = [
    re.compile(r"MRMS_MESH_Max_1440min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"),
    re.compile(r"MRMS_MESHMax1440min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"),
]

MM_PER_INCH = 25.4
AWS_ARCHIVE_START = date_cls(2020, 10, 1)

# Cache discovered product prefixes in memory so we do not re-list the bucket every request
MESH_PREFIX_CACHE: list[str] | None = None


class MeshResponse(BaseModel):
    meshIn: float = Field(..., description="Nearest-grid MESH value in inches")
    source: str = Field(..., description="Upstream source name")
    timestamp: str = Field(..., description="Resolved MRMS file timestamp in UTC")
    note: str = Field(..., description="Description of returned value")
    file: str = Field(..., description="Resolved MRMS filename")
    url: str = Field(..., description="Resolved MRMS source URL")


class HealthResponse(BaseModel):
    ok: bool
    service: str
    version: str


@dataclass
class ResolvedFile:
    key: str
    url: str
    filename: str
    timestamp: datetime


app = FastAPI(title=APP_NAME, version=APP_VERSION)


@app.get("/")
async def root():
    return {
        "ok": True,
        "service": APP_NAME,
        "version": APP_VERSION,
        "message": "MRMS MESH service is running",
        "routes": [
            "/healthz",
            "/docs",
            "/mesh?lat=33.80&lon=-84.31&date=2025-06-26",
        ],
    }


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(ok=True, service=APP_NAME, version=APP_VERSION)


@app.get("/mesh", response_model=MeshResponse)
async def get_mesh(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    date: str = Query(..., description="UTC date in YYYY-MM-DD format"),
) -> MeshResponse:
    requested_date = parse_iso_date(date)

    if requested_date < AWS_ARCHIVE_START:
        raise HTTPException(
            status_code=422,
            detail="This service currently supports MRMS AWS archive dates from 2020-10-01 onward.",
        )

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(
            connect=HTTP_TIMEOUT_SECONDS,
            read=DOWNLOAD_TIMEOUT_SECONDS,
            write=HTTP_TIMEOUT_SECONDS,
            pool=HTTP_TIMEOUT_SECONDS,
        ),
        headers={"User-Agent": f"{APP_NAME}/{APP_VERSION}"},
    ) as client:
        resolved = await resolve_daily_mesh_file(client, requested_date)
        grib_path = await download_and_cache_grib(client, resolved.url)

    mesh_mm = extract_mesh_mm_at_point(grib_path, lat=lat, lon=lon)
    mesh_in = round(mesh_mm / MM_PER_INCH, 2)

    return MeshResponse(
        meshIn=mesh_in,
        source="NOAA MRMS MESH",
        timestamp=resolved.timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        note="Maximum Estimated Size of Hail 1440-minute swath at nearest MRMS grid point",
        file=resolved.filename,
        url=resolved.url,
    )


def parse_iso_date(value: str) -> date_cls:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD") from exc


def normalize_lon_scalar(lon: float) -> float:
    if lon > 180:
        return lon - 360
    if lon < -180:
        return lon + 360
    return lon


def normalize_lon_grid(lons: np.ndarray) -> np.ndarray:
    return np.where(lons > 180.0, lons - 360.0, lons)


def to_360_lon(lon: float) -> float:
    return lon if lon >= 0 else lon + 360.0


def parse_filename_timestamp(filename: str) -> datetime:
    for pattern in FILENAME_PATTERNS:
        match = pattern.search(filename)
        if match:
            dt = datetime.strptime(
                f"{match.group('ymd')}{match.group('hms')}",
                "%Y%m%d%H%M%S",
            )
            return dt.replace(tzinfo=timezone.utc)

    raise HTTPException(status_code=500, detail=f"Unrecognized MRMS filename: {filename}")


def matches_mesh_filename(filename: str, ymd: str) -> bool:
    for pattern in FILENAME_PATTERNS:
        match = pattern.search(filename)
        if match and match.group("ymd") == ymd:
            return True
    return False


async def resolve_daily_mesh_file(client: httpx.AsyncClient, requested_date: date_cls) -> ResolvedFile:
    ymd = requested_date.strftime("%Y%m%d")
    mesh_prefixes = await discover_mesh_product_prefixes(client)

    if not mesh_prefixes:
        raise HTTPException(
            status_code=500,
            detail="No MESH-related CONUS prefixes found in AWS MRMS bucket.",
        )

    candidates: list[ResolvedFile] = []

    for product_prefix in mesh_prefixes:
        day_prefix = f"{product_prefix}{ymd}/"
        keys = await list_s3_keys(client, day_prefix)

        for key in keys:
            filename = key.rsplit("/", 1)[-1]
            if matches_mesh_filename(filename, ymd):
                candidates.append(
                    ResolvedFile(
                        key=key,
                        url=f"{AWS_BUCKET_BASE}/{quote(key)}",
                        filename=filename,
                        timestamp=parse_filename_timestamp(filename),
                    )
                )

    deduped = {item.key: item for item in candidates}
    candidates = list(deduped.values())

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No MRMS MESH 1440-minute files found in AWS archive for {requested_date.isoformat()}. "
                f"Tried prefixes: {mesh_prefixes}"
            ),
        )

    candidates.sort(key=lambda item: item.timestamp)
    return candidates[-1]


async def discover_mesh_product_prefixes(client: httpx.AsyncClient) -> list[str]:
    global MESH_PREFIX_CACHE

    if MESH_PREFIX_CACHE is not None:
        return MESH_PREFIX_CACHE

    prefixes = await list_s3_common_prefixes(client, "CONUS/")

    # Keep only product folders relevant to daily MESH swath products
    filtered = []
    for prefix in prefixes:
        upper = prefix.upper()
        if "MESH" in upper and "1440" in upper:
            filtered.append(prefix)

    MESH_PREFIX_CACHE = sorted(filtered)
    return MESH_PREFIX_CACHE


async def list_s3_common_prefixes(client: httpx.AsyncClient, prefix: str) -> list[str]:
    prefixes: list[str] = []
    continuation_token: str | None = None

    while True:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "delimiter": "/",
            "max-keys": "1000",
        }
        if continuation_token:
            params["continuation-token"] = continuation_token

        response = await client.get(AWS_BUCKET_BASE + "/", params=params)
        response.raise_for_status()

        root = ET.fromstring(response.text)

        for elem in root.findall(".//{*}CommonPrefixes/{*}Prefix"):
            if elem.text:
                prefixes.append(elem.text)

        is_truncated = root.findtext(".//{*}IsTruncated", default="false").lower() == "true"
        if not is_truncated:
            break

        continuation_token = root.findtext(".//{*}NextContinuationToken")
        if not continuation_token:
            break

    return list(dict.fromkeys(prefixes))


async def list_s3_keys(client: httpx.AsyncClient, prefix: str) -> list[str]:
    keys: list[str] = []
    continuation_token: str | None = None

    while True:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": "1000",
        }
        if continuation_token:
            params["continuation-token"] = continuation_token

        response = await client.get(AWS_BUCKET_BASE + "/", params=params)
        response.raise_for_status()

        root = ET.fromstring(response.text)

        for elem in root.findall(".//{*}Contents/{*}Key"):
            if elem.text:
                keys.append(elem.text)

        is_truncated = root.findtext(".//{*}IsTruncated", default="false").lower() == "true"
        if not is_truncated:
            break

        continuation_token = root.findtext(".//{*}NextContinuationToken")
        if not continuation_token:
            break

    return keys


async def download_and_cache_grib(client: httpx.AsyncClient, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    grib_path = CACHE_DIR / f"{digest}.grib2"

    if grib_path.exists() and grib_path.stat().st_size > 0:
        return grib_path

    response = await client.get(url)
    response.raise_for_status()
    payload = response.content

    if url.endswith(".gz"):
        try:
            raw_bytes = gzip.decompress(payload)
        except OSError as exc:
            raise HTTPException(status_code=502, detail="Failed to gunzip MRMS payload") from exc
    else:
        raw_bytes = payload

    tmp_path = grib_path.with_suffix(".tmp")
    tmp_path.write_bytes(raw_bytes)
    tmp_path.replace(grib_path)

    return grib_path


def extract_mesh_mm_at_point(grib_path: Path, lat: float, lon: float) -> float:
    """
    Memory-safe extraction:
    Instead of loading the full CONUS grid with grb.values + grb.latlons(),
    read only a small bounding box around the requested point.
    """
    norm_lon = normalize_lon_scalar(lon)
    lon_360 = to_360_lon(norm_lon)

    try:
        grbs = pygrib.open(str(grib_path))
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Unable to open GRIB2 file") from exc

    try:
        grb = grbs.message(1)

        # Try progressively larger local boxes around the target point.
        for pad in (0.25, 0.50, 1.0, 2.0):
            lat1 = max(-90.0, lat - pad)
            lat2 = min(90.0, lat + pad)

            for center_lon in (lon_360, norm_lon):
                lon1 = center_lon - pad
                lon2 = center_lon + pad

                try:
                    values, lats, lons = grb.data(lat1=lat1, lat2=lat2, lon1=lon1, lon2=lon2)
                except Exception:
                    continue

                arr = np.ma.filled(values, np.nan).astype(float)
                if arr.size == 0 or not np.isfinite(arr).any():
                    continue

                lats_arr = np.asarray(lats, dtype=float)
                lons_arr = normalize_lon_grid(np.asarray(lons, dtype=float))

                lat_scale = np.cos(np.deg2rad(lat))
                dist2 = (lats_arr - lat) ** 2 + ((lons_arr - norm_lon) * lat_scale) ** 2
                dist2 = np.where(np.isfinite(arr), dist2, np.inf)

                idx = np.argmin(dist2)
                if np.isfinite(dist2.flat[idx]):
                    return float(arr.flat[idx])

        raise HTTPException(status_code=404, detail="No finite MESH value found near requested point")
    finally:
        grbs.close()
