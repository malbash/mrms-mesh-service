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
APP_VERSION = "1.1.0"

AWS_BUCKET_BASE = os.getenv("MRMS_AWS_BASE", "https://noaa-mrms-pds.s3.amazonaws.com")
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/mrms-cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "45"))
DOWNLOAD_TIMEOUT_SECONDS = float(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "120"))

# Support the documented current naming, plus a fallback without underscores
FILENAME_PATTERNS = [
    re.compile(
        r"MRMS_MESH_Max_1440min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"
    ),
    re.compile(
        r"MRMS_MESHMax1440min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"
    ),
]

# Prefix candidates to search in AWS
PREFIX_PATTERNS = [
    "CONUS/MESH_Max_1440min/{ymd}/",
    "CONUS/MESHMax1440min/{ymd}/",
]

MM_PER_INCH = 25.4
AWS_ARCHIVE_START = date_cls(2020, 10, 1)


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
    candidates: list[ResolvedFile] = []

    for prefix_template in PREFIX_PATTERNS:
        prefix = prefix_template.format(ymd=ymd)
        keys = await list_s3_keys(client, prefix)

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

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No MRMS MESH 1440-minute files found in AWS archive for {requested_date.isoformat()}",
        )

    candidates.sort(key=lambda item: item.timestamp)
    return candidates[-1]


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
    norm_lon = normalize_lon_scalar(lon)

    try:
        grbs = pygrib.open(str(grib_path))
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Unable to open GRIB2 file") from exc

    try:
        grb = grbs.message(1)
        values = np.ma.filled(grb.values, np.nan).astype(float)
        lats, lons = grb.latlons()
        lons = normalize_lon_grid(lons)

        lat_scale = np.cos(np.deg2rad(lat))
        dist2 = (lats - lat) ** 2 + ((lons - norm_lon) * lat_scale) ** 2

        flat_order = np.argsort(dist2, axis=None)

        for flat_idx in flat_order[:500]:
            i, j = np.unravel_index(flat_idx, values.shape)
            value = values[i, j]
            if np.isfinite(value):
                return float(value)

        raise HTTPException(status_code=404, detail="No finite MESH value found near requested point")
    finally:
        grbs.close()
