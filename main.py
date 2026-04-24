from __future__ import annotations

import gzip
import hashlib
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional, Pattern
from urllib.parse import quote, unquote

import httpx
import numpy as np
import pygrib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

APP_NAME = "mrms-mesh-service"
APP_VERSION = "local-render-5.0.0"

AWS_BUCKET_BASE = os.getenv("MRMS_AWS_BASE", "https://noaa-mrms-pds.s3.amazonaws.com")
MTARCHIVE_BASE = os.getenv("MTARCHIVE_BASE", "https://mtarchive.geol.iastate.edu")

CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/mrms-cache" if os.name != "nt" else "./mrms-cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "45"))
DOWNLOAD_TIMEOUT_SECONDS = float(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "180"))

MM_PER_INCH = 25.4
EARTH_RADIUS_MILES = 3958.7613

AWS_ARCHIVE_START = date_cls(2020, 10, 1)
MTARCHIVE_START = date_cls(2014, 10, 1)

HTML_GRIB_FILENAME_RE = re.compile(r'([^/"<>]+\.grib2\.gz)')


@dataclass(frozen=True)
class ProductConfig:
    product_key: str
    product_label: str
    grib_name: str
    grib_short_name: str
    grib_units: str
    filename_patterns: list[Pattern[str]]
    known_aws_prefixes: list[str]
    mtarchive_product_dirs: list[str]
    discover_markers: list[str]


DAILY_PRODUCT = ProductConfig(
    product_key="mesh_1440min",
    product_label="MESH_Max_1440min",
    grib_name="MESH_Max_1440min",
    grib_short_name="MESHMax1440min",
    grib_units="mm",
    filename_patterns=[
        re.compile(r"MRMS_MESH_Max_1440min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"),
        re.compile(r"MESH_Max_1440min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"),
        re.compile(r"MRMS_Max_1440min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"),
        re.compile(r"MRMS_MESHMax1440min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"),
        re.compile(r"MESHMax1440min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"),
    ],
    known_aws_prefixes=[
        "CONUS/MESH_Max_1440min_00.50/",
        "CONUS/MESH_Max_1440min/",
        "CONUS/MESHMax1440min_00.50/",
        "CONUS/MESHMax1440min/",
    ],
    mtarchive_product_dirs=[
        "MESH_Max_1440min",
        "MESH_Max_1440min_00.50",
        "MESHMax1440min",
        "MESHMax1440min_00.50",
    ],
    discover_markers=["MESH_MAX_1440MIN", "MESHMAX1440MIN"],
)

HOURLY_PRODUCT = ProductConfig(
    product_key="mesh_60min",
    product_label="MESH_Max_60min",
    grib_name="MESH_Max_60min",
    grib_short_name="MESHMax60min",
    grib_units="mm",
    filename_patterns=[
        re.compile(r"MRMS_MESH_Max_60min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"),
        re.compile(r"MESH_Max_60min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"),
        re.compile(r"MRMS_Max_60min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"),
        re.compile(r"MRMS_MESHMax60min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"),
        re.compile(r"MESHMax60min_00\.50_(?P<ymd>\d{8})-(?P<hms>\d{6})\.grib2\.gz$"),
    ],
    known_aws_prefixes=[
        "CONUS/MESH_Max_60min_00.50/",
        "CONUS/MESH_Max_60min/",
        "CONUS/MESHMax60min_00.50/",
        "CONUS/MESHMax60min/",
    ],
    mtarchive_product_dirs=[
        "MESH_Max_60min",
        "MESH_Max_60min_00.50",
        "MESHMax60min",
        "MESHMax60min_00.50",
    ],
    discover_markers=["MESH_MAX_60MIN", "MESHMAX60MIN"],
)

PREFIX_CACHE: dict[str, list[str]] = {}


class MeshValue(BaseModel):
    meshIn: Optional[float] = None
    meshMm: Optional[float] = None
    rawMm: Optional[float] = None
    status: str
    distanceMiles: Optional[float] = None
    gridLat: Optional[float] = None
    gridLon: Optional[float] = None
    note: str


class MeshResponse(BaseModel):
    meshIn: Optional[float] = Field(None, description="Selected reportable MESH value in inches")
    meshMm: Optional[float] = Field(None, description="Selected reportable MESH value in millimeters")
    status: str = Field(..., description="hail_detected, no_hail_detected, or no_data")
    radiusMiles: float
    boundaryHours: int
    searchMode: str
    selected: dict
    nearest: MeshValue
    radiusMax: MeshValue
    daily: dict
    boundary: dict
    source: str
    timestamp: str
    note: str
    diagnostics: dict


class HealthResponse(BaseModel):
    ok: bool
    service: str
    version: str


class ResolveResponse(BaseModel):
    date: str
    product: str
    gribName: str
    gribShortName: str
    gribUnits: str
    archive: str
    file: str
    timestamp: str
    key: str
    url: str


@dataclass
class ResolvedFile:
    key: str
    url: str
    filename: str
    timestamp: datetime
    archive: str
    product_key: str
    product_label: str
    grib_name: str
    grib_short_name: str
    grib_units: str


app = FastAPI(title=APP_NAME, version=APP_VERSION)


@app.get("/")
async def root():
    return {
        "ok": True,
        "service": APP_NAME,
        "version": APP_VERSION,
        "message": "MRMS MESH service is running",
        "searchConcept": "Daily MESH_Max_1440min + previous-day last boundary hours + following-day first boundary hours using MESH_Max_60min.",
        "units": {
            "sourceGribUnits": "mm",
            "meshMm": "millimeters",
            "meshIn": "inches",
            "negativeValues": "negative MRMS values are treated as no_data sentinel values and are not returned as reportable size values",
        },
        "routes": [
            "/healthz",
            "/docs",
            "/debug/prefixes?product=daily",
            "/debug/prefixes?product=hourly",
            "/debug/resolve?date=2025-08-20&product=daily",
            "/debug/window?lat=34.960468&lon=-81.880455&date=2025-08-20&product=daily&pad=0.1",
            "/mesh?lat=34.960468&lon=-81.880455&date=2025-08-20&radiusMiles=5&boundaryHours=3",
        ],
    }


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(ok=True, service=APP_NAME, version=APP_VERSION)


@app.get("/debug/prefixes")
async def debug_prefixes(product: str = Query("daily", description="daily or hourly")):
    config = get_product_config(product)
    async with make_client() as client:
        prefixes = await discover_product_prefixes(client, config)
    return {
        "product": config.product_label,
        "gribName": config.grib_name,
        "gribShortName": config.grib_short_name,
        "gribUnits": config.grib_units,
        "count": len(prefixes),
        "prefixes": prefixes,
    }


@app.get("/debug/resolve", response_model=ResolveResponse)
async def debug_resolve(
    date: str = Query(..., description="UTC date in YYYY-MM-DD format"),
    product: str = Query("daily", description="daily or hourly"),
) -> ResolveResponse:
    requested_date = parse_iso_date(date)
    config = get_product_config(product)
    async with make_client() as client:
        resolved = await resolve_latest_product_file(client, requested_date, config)
    return ResolveResponse(
        date=date,
        product=resolved.product_label,
        gribName=resolved.grib_name,
        gribShortName=resolved.grib_short_name,
        gribUnits=resolved.grib_units,
        archive=resolved.archive,
        file=resolved.filename,
        timestamp=resolved.timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        key=resolved.key,
        url=resolved.url,
    )


@app.get("/debug/window")
async def debug_window(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    date: str = Query(..., description="UTC date in YYYY-MM-DD format"),
    product: str = Query("daily", description="daily or hourly"),
    pad: float = Query(0.25, ge=0.01, le=5.0, description="Search half-window in degrees"),
):
    requested_date = parse_iso_date(date)
    config = get_product_config(product)
    async with make_client() as client:
        resolved = await resolve_latest_product_file(client, requested_date, config)
        grib_path = await download_and_cache_grib(client, resolved.url)
    stats = extract_mesh_window_debug(grib_path, lat=lat, lon=lon, pad=pad, resolved=resolved)
    return {
        "date": date,
        "lat": lat,
        "lon": lon,
        "padDegrees": pad,
        "archive": resolved.archive,
        "product": resolved.product_label,
        "gribName": resolved.grib_name,
        "gribShortName": resolved.grib_short_name,
        "gribUnits": resolved.grib_units,
        "file": resolved.filename,
        "timestamp": resolved.timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "url": resolved.url,
        **stats,
    }


@app.get("/mesh", response_model=MeshResponse)
async def get_mesh(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    date: str = Query(..., description="UTC date in YYYY-MM-DD format"),
    radiusMiles: float = Query(5.0, ge=0.1, le=50.0, description="Search radius around property in miles"),
    boundaryHours: int = Query(3, ge=0, le=12, description="Extra boundary hours before and after the selected UTC date"),
) -> MeshResponse:
    requested_date = parse_iso_date(date)
    if requested_date < MTARCHIVE_START:
        raise HTTPException(
            status_code=422,
            detail="This service currently supports dates from 2014-10-01 onward, subject to archive/product availability.",
        )

    async with make_client() as client:
        daily_resolved = await resolve_latest_product_file(client, requested_date, DAILY_PRODUCT)
        daily_component = await analyze_resolved_file(
            client=client,
            resolved=daily_resolved,
            lat=lat,
            lon=lon,
            radius_miles=radiusMiles,
            coverage="selected_date_daily_1440min",
        )

        previous_components: list[dict] = []
        following_components: list[dict] = []

        if boundaryHours > 0:
            previous_files, following_files = await resolve_boundary_hourly_files(
                client=client,
                requested_date=requested_date,
                boundary_hours=boundaryHours,
            )

            for resolved in previous_files:
                previous_components.append(
                    await analyze_resolved_file(
                        client=client,
                        resolved=resolved,
                        lat=lat,
                        lon=lon,
                        radius_miles=radiusMiles,
                        coverage=f"previous_day_last_{boundaryHours}_hours_hourly_60min",
                    )
                )

            for resolved in following_files:
                following_components.append(
                    await analyze_resolved_file(
                        client=client,
                        resolved=resolved,
                        lat=lat,
                        lon=lon,
                        radius_miles=radiusMiles,
                        coverage=f"following_day_first_{boundaryHours}_hours_hourly_60min",
                    )
                )

    all_components = [daily_component] + previous_components + following_components
    selected_component = select_best_component(all_components)

    selected_radius_max = MeshValue(**selected_component["analysis"]["radiusMax"])
    daily_nearest = MeshValue(**daily_component["analysis"]["nearest"])

    mesh_in = selected_radius_max.meshIn
    mesh_mm = selected_radius_max.meshMm
    overall_status = selected_radius_max.status

    if overall_status == "hail_detected":
        note = (
            f"Positive MRMS MESH was detected within {radiusMiles:g} miles during the expanded search window. "
            "The selected value is the maximum positive MESH value found across the daily product and boundary-hour checks."
        )
    elif overall_status == "no_hail_detected":
        note = f"No positive MRMS MESH was detected within {radiusMiles:g} miles during the expanded search window."
    else:
        note = (
            f"No positive MRMS MESH was detected within {radiusMiles:g} miles during the expanded search window. "
            "Returned values were missing, negative/sentinel, or unavailable."
        )

    previous_hail_components = [
        compact_component(component, include_nearest=True, include_diagnostics=False)
        for component in previous_components
        if component_has_hail(component)
    ]
    following_hail_components = [
        compact_component(component, include_nearest=True, include_diagnostics=False)
        for component in following_components
        if component_has_hail(component)
    ]

    suppressed_previous_count = len(previous_components) - len(previous_hail_components)
    suppressed_following_count = len(following_components) - len(following_hail_components)

    daily_summary = compact_component(daily_component, include_nearest=True, include_diagnostics=True)
    selected_summary = compact_component(selected_component, include_nearest=True, include_diagnostics=True)

    return MeshResponse(
        meshIn=mesh_in,
        meshMm=mesh_mm,
        status=overall_status,
        radiusMiles=radiusMiles,
        boundaryHours=boundaryHours,
        searchMode="daily_1440min_plus_boundary_60min",
        selected=selected_summary,
        nearest=daily_nearest,
        radiusMax=selected_radius_max,
        daily=daily_summary,
        boundary={
            "enabled": boundaryHours > 0,
            "hourlyProduct": HOURLY_PRODUCT.product_label,
            "hourlyGribName": HOURLY_PRODUCT.grib_name,
            "hourlyGribShortName": HOURLY_PRODUCT.grib_short_name,
            "hourlyGribUnits": HOURLY_PRODUCT.grib_units,
            "previousDayLastHoursWithHail": previous_hail_components,
            "followingDayFirstHoursWithHail": following_hail_components,
            "previousFileCountChecked": len(previous_components),
            "followingFileCountChecked": len(following_components),
            "previousNoHailFilesHidden": suppressed_previous_count,
            "followingNoHailFilesHidden": suppressed_following_count,
            "hourlyHailFileCount": len(previous_hail_components) + len(following_hail_components),
            "note": (
                "Hourly boundary files are only shown when positive MESH is detected. "
                "No-hail or no-data hourly files are hidden to keep the response concise."
            ),
        },
        source="NOAA MRMS MESH",
        timestamp=selected_component["timestamp"],
        note=note,
        diagnostics={
            "dailyProduct": DAILY_PRODUCT.product_label,
            "dailyGribName": DAILY_PRODUCT.grib_name,
            "dailyGribShortName": DAILY_PRODUCT.grib_short_name,
            "dailyGribUnits": DAILY_PRODUCT.grib_units,
            "boundaryProduct": HOURLY_PRODUCT.product_label,
            "boundaryGribName": HOURLY_PRODUCT.grib_name,
            "boundaryGribShortName": HOURLY_PRODUCT.grib_short_name,
            "boundaryGribUnits": HOURLY_PRODUCT.grib_units,
            "componentCountSearched": len(all_components),
            "previousBoundaryFilesSearched": len(previous_components),
            "followingBoundaryFilesSearched": len(following_components),
            "hourlyBoundaryFilesWithHail": len(previous_hail_components) + len(following_hail_components),
            "hourlyBoundaryFilesHiddenNoHail": suppressed_previous_count + suppressed_following_count,
            "negativeValuesPolicy": (
                "Negative MRMS values are treated as no_data sentinel values and are not returned "
                "as reportable meshMm, meshIn, or rawMm."
            ),
            "boundarySampling": "hourly 60-min files near UTC hour marks",
        },
    )


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(
            connect=HTTP_TIMEOUT_SECONDS,
            read=DOWNLOAD_TIMEOUT_SECONDS,
            write=HTTP_TIMEOUT_SECONDS,
            pool=HTTP_TIMEOUT_SECONDS,
        ),
        headers={"User-Agent": f"{APP_NAME}/{APP_VERSION}"},
    )


def get_product_config(product: str) -> ProductConfig:
    product_lower = product.lower().strip()
    if product_lower in {"daily", "1440", "1440min", "mesh_1440min", "mesh_max_1440min"}:
        return DAILY_PRODUCT
    if product_lower in {"hourly", "60", "60min", "mesh_60min", "mesh_max_60min"}:
        return HOURLY_PRODUCT
    raise HTTPException(status_code=422, detail="product must be daily or hourly")


def parse_iso_date(value: str) -> date_cls:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD") from exc


def sanitize_nonnegative_value(value: Optional[float]) -> Optional[float]:
    if value is None or not np.isfinite(value) or value < 0:
        return None
    return float(value)


def clean_mesh_value(mesh_mm: Optional[float]) -> tuple[Optional[float], Optional[float], str, str]:
    clean_mm = sanitize_nonnegative_value(mesh_mm)
    if clean_mm is None:
        return (None, None, "no_data", "MRMS returned a missing, non-finite, or negative/sentinel value; value treated as no_data.")
    if clean_mm == 0:
        return (0.0, 0.0, "no_hail_detected", "MRMS MESH returned 0 mm.")
    mesh_in = round(clean_mm / MM_PER_INCH, 2)
    return (mesh_in, round(clean_mm, 2), "hail_detected", "Positive MRMS MESH value detected.")


def build_mesh_value(raw_mm: Optional[float], distance_miles: Optional[float], grid_lat: Optional[float], grid_lon: Optional[float], extra_note: str = "") -> dict:
    mesh_in, mesh_mm, status, note = clean_mesh_value(raw_mm)
    clean_raw = sanitize_nonnegative_value(raw_mm)
    if extra_note:
        note = f"{note} {extra_note}"
    return {
        "meshIn": mesh_in,
        "meshMm": mesh_mm,
        "rawMm": None if clean_raw is None else round(clean_raw, 3),
        "status": status,
        "distanceMiles": None if distance_miles is None else round(float(distance_miles), 3),
        "gridLat": None if grid_lat is None else round(float(grid_lat), 6),
        "gridLon": None if grid_lon is None else round(float(grid_lon), 6),
        "note": note,
    }


def sanitized_min(values: np.ndarray) -> Optional[float]:
    clean = values[np.isfinite(values) & (values >= 0)]
    if clean.size == 0:
        return None
    return float(np.min(clean))


def sanitized_max(values: np.ndarray) -> Optional[float]:
    clean = values[np.isfinite(values) & (values >= 0)]
    if clean.size == 0:
        return None
    return float(np.max(clean))


def component_has_hail(component: dict) -> bool:
    return component.get("analysis", {}).get("radiusMax", {}).get("status") == "hail_detected"


def compact_component(component: dict, include_nearest: bool = True, include_diagnostics: bool = False) -> dict:
    analysis = component.get("analysis", {})
    output = {
        "coverage": component.get("coverage"),
        "product": component.get("product"),
        "gribName": component.get("gribName"),
        "gribShortName": component.get("gribShortName"),
        "gribUnits": component.get("gribUnits"),
        "archive": component.get("archive"),
        "timestamp": component.get("timestamp"),
        "file": component.get("file"),
        "url": component.get("url"),
        "radiusMax": analysis.get("radiusMax"),
    }
    if include_nearest:
        output["nearest"] = analysis.get("nearest")
    if include_diagnostics:
        output["diagnostics"] = analysis.get("diagnostics")
    return output


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


def haversine_miles(lat1: float, lon1: float, lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = np.deg2rad(lats2)
    lon2_rad = np.deg2rad(lons2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return EARTH_RADIUS_MILES * c


def parse_filename_timestamp(filename: str, config: ProductConfig) -> datetime:
    for pattern in config.filename_patterns:
        match = pattern.search(filename)
        if match:
            dt = datetime.strptime(f"{match.group('ymd')}{match.group('hms')}", "%Y%m%d%H%M%S")
            return dt.replace(tzinfo=timezone.utc)
    raise HTTPException(status_code=500, detail=f"Unrecognized MRMS filename for {config.product_label}: {filename}")


def matches_product_filename(filename: str, ymd: str, config: ProductConfig) -> bool:
    for pattern in config.filename_patterns:
        match = pattern.search(filename)
        if match and match.group("ymd") == ymd:
            return True
    return False


async def resolve_latest_product_file(client: httpx.AsyncClient, requested_date: date_cls, config: ProductConfig) -> ResolvedFile:
    candidates = await resolve_product_candidates(client, requested_date, config)
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No {config.product_label} file found for {requested_date.isoformat()}")
    candidates.sort(key=lambda item: item.timestamp)
    return candidates[-1]


async def resolve_product_candidates(client: httpx.AsyncClient, requested_date: date_cls, config: ProductConfig) -> list[ResolvedFile]:
    errors: list[str] = []
    if requested_date >= AWS_ARCHIVE_START:
        try:
            aws_candidates = await resolve_candidates_from_aws(client, requested_date, config)
            if aws_candidates:
                return aws_candidates
            errors.append("AWS: no candidates")
        except HTTPException as exc:
            errors.append(f"AWS: {exc.detail}")
    try:
        mt_candidates = await resolve_candidates_from_mtarchive(client, requested_date, config)
        if mt_candidates:
            return mt_candidates
        errors.append("MTArchive: no candidates")
    except HTTPException as exc:
        errors.append(f"MTArchive: {exc.detail}")
    raise HTTPException(status_code=404, detail={"message": f"No {config.product_label} files found for {requested_date.isoformat()}", "attempts": errors})


async def resolve_candidates_from_aws(client: httpx.AsyncClient, requested_date: date_cls, config: ProductConfig) -> list[ResolvedFile]:
    ymd = requested_date.strftime("%Y%m%d")
    prefixes = await discover_product_prefixes(client, config)
    candidates: list[ResolvedFile] = []
    for product_prefix in prefixes:
        day_prefix = f"{product_prefix}{ymd}/"
        keys = await list_s3_keys(client, day_prefix)
        for key in keys:
            filename = key.rsplit("/", 1)[-1]
            if matches_product_filename(filename, ymd, config):
                candidates.append(
                    ResolvedFile(
                        key=key,
                        url=f"{AWS_BUCKET_BASE}/{quote(key)}",
                        filename=filename,
                        timestamp=parse_filename_timestamp(filename, config),
                        archive="NOAA MRMS AWS Open Data",
                        product_key=config.product_key,
                        product_label=config.product_label,
                        grib_name=config.grib_name,
                        grib_short_name=config.grib_short_name,
                        grib_units=config.grib_units,
                    )
                )
    return list({item.key: item for item in candidates}.values())


async def resolve_candidates_from_mtarchive(client: httpx.AsyncClient, requested_date: date_cls, config: ProductConfig) -> list[ResolvedFile]:
    ymd = requested_date.strftime("%Y%m%d")
    yyyy = requested_date.strftime("%Y")
    mm = requested_date.strftime("%m")
    dd = requested_date.strftime("%d")
    candidates: list[ResolvedFile] = []
    for product_dir in config.mtarchive_product_dirs:
        base_dir = f"{MTARCHIVE_BASE}/{yyyy}/{mm}/{dd}/mrms/ncep/{product_dir}/"
        try:
            response = await client.get(base_dir)
            response.raise_for_status()
        except httpx.HTTPError:
            continue
        filenames = extract_matching_filenames_from_html(response.text, ymd, config)
        for filename in filenames:
            candidates.append(
                ResolvedFile(
                    key=f"{yyyy}/{mm}/{dd}/mrms/ncep/{product_dir}/{filename}",
                    url=f"{base_dir}{quote(filename)}",
                    filename=filename,
                    timestamp=parse_filename_timestamp(filename, config),
                    archive="Iowa State MTArchive",
                    product_key=config.product_key,
                    product_label=config.product_label,
                    grib_name=config.grib_name,
                    grib_short_name=config.grib_short_name,
                    grib_units=config.grib_units,
                )
            )
    return list({item.key: item for item in candidates}.values())


async def resolve_boundary_hourly_files(client: httpx.AsyncClient, requested_date: date_cls, boundary_hours: int) -> tuple[list[ResolvedFile], list[ResolvedFile]]:
    previous_date = requested_date - timedelta(days=1)
    following_date = requested_date + timedelta(days=1)
    previous_candidates = await safe_resolve_candidates(client, previous_date, HOURLY_PRODUCT)
    following_candidates = await safe_resolve_candidates(client, following_date, HOURLY_PRODUCT)

    previous_targets = [
        datetime.combine(previous_date, time(hour=h, minute=0, second=0), tzinfo=timezone.utc)
        for h in range(24 - boundary_hours, 24)
    ]
    following_targets = [
        datetime.combine(following_date, time(hour=h, minute=0, second=0), tzinfo=timezone.utc)
        for h in range(0, boundary_hours + 1)
    ]
    previous_files = select_closest_files_to_targets(previous_candidates, previous_targets)
    following_files = select_closest_files_to_targets(following_candidates, following_targets)
    return previous_files, following_files


async def safe_resolve_candidates(client: httpx.AsyncClient, requested_date: date_cls, config: ProductConfig) -> list[ResolvedFile]:
    try:
        return await resolve_product_candidates(client, requested_date, config)
    except HTTPException:
        return []


def select_closest_files_to_targets(candidates: list[ResolvedFile], target_times: list[datetime], tolerance_minutes: int = 10) -> list[ResolvedFile]:
    selected: list[ResolvedFile] = []
    for target in target_times:
        if not candidates:
            continue
        closest = min(candidates, key=lambda item: abs((item.timestamp - target).total_seconds()))
        diff_minutes = abs((closest.timestamp - target).total_seconds()) / 60.0
        if diff_minutes <= tolerance_minutes:
            selected.append(closest)
    return list({item.key: item for item in selected}.values())


def extract_matching_filenames_from_html(html: str, ymd: str, config: ProductConfig) -> list[str]:
    filenames: list[str] = []
    for raw_match in HTML_GRIB_FILENAME_RE.findall(html):
        filename = unquote(raw_match.strip()).rsplit("/", 1)[-1]
        if matches_product_filename(filename, ymd, config):
            filenames.append(filename)
    return list(dict.fromkeys(filenames))


async def discover_product_prefixes(client: httpx.AsyncClient, config: ProductConfig) -> list[str]:
    if config.product_key in PREFIX_CACHE:
        return PREFIX_CACHE[config.product_key]
    prefixes = await list_s3_common_prefixes(client, "CONUS/")
    filtered = []
    for prefix in prefixes:
        upper = prefix.upper()
        if any(marker in upper for marker in config.discover_markers):
            filtered.append(prefix)
    merged = list(dict.fromkeys(config.known_aws_prefixes + filtered))
    PREFIX_CACHE[config.product_key] = sorted(merged)
    return PREFIX_CACHE[config.product_key]


async def list_s3_common_prefixes(client: httpx.AsyncClient, prefix: str) -> list[str]:
    prefixes: list[str] = []
    continuation_token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": "1000"}
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
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
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

    gz_tmp_path = CACHE_DIR / f"{digest}.download.gz"
    raw_tmp_path = CACHE_DIR / f"{digest}.download.tmp"
    if gz_tmp_path.exists():
        gz_tmp_path.unlink()
    if raw_tmp_path.exists():
        raw_tmp_path.unlink()

    async with client.stream("GET", url) as response:
        response.raise_for_status()
        if url.endswith(".gz"):
            with open(gz_tmp_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
        else:
            with open(raw_tmp_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)

    if url.endswith(".gz"):
        try:
            with gzip.open(gz_tmp_path, "rb") as src, open(raw_tmp_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
        except OSError as exc:
            raise HTTPException(status_code=502, detail="Failed to gunzip MRMS payload") from exc
        finally:
            if gz_tmp_path.exists():
                gz_tmp_path.unlink()

    raw_tmp_path.replace(grib_path)
    return grib_path


async def analyze_resolved_file(client: httpx.AsyncClient, resolved: ResolvedFile, lat: float, lon: float, radius_miles: float, coverage: str) -> dict:
    grib_path = await download_and_cache_grib(client, resolved.url)
    analysis = extract_mesh_radius_analysis(grib_path, lat=lat, lon=lon, radius_miles=radius_miles, resolved=resolved)
    return {
        "coverage": coverage,
        "product": resolved.product_label,
        "gribName": resolved.grib_name,
        "gribShortName": resolved.grib_short_name,
        "gribUnits": resolved.grib_units,
        "archive": resolved.archive,
        "timestamp": resolved.timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "file": resolved.filename,
        "url": resolved.url,
        "nearest": analysis["nearest"],
        "radiusMax": analysis["radiusMax"],
        "diagnostics": analysis["diagnostics"],
        "analysis": analysis,
    }


def select_best_component(components: list[dict]) -> dict:
    hail_components = [
        component
        for component in components
        if component["analysis"]["radiusMax"]["status"] == "hail_detected"
        and component["analysis"]["radiusMax"]["meshMm"] is not None
    ]
    if hail_components:
        return max(hail_components, key=lambda component: component["analysis"]["radiusMax"]["meshMm"])
    no_hail_components = [component for component in components if component["analysis"]["radiusMax"]["status"] == "no_hail_detected"]
    if no_hail_components:
        return no_hail_components[0]
    return components[0]


def extract_mesh_radius_analysis(grib_path: Path, lat: float, lon: float, radius_miles: float, resolved: ResolvedFile) -> dict:
    norm_lon = normalize_lon_scalar(lon)
    lon_360 = to_360_lon(norm_lon)
    pad_degrees = max(0.05, radius_miles / 69.0 + 0.03)

    try:
        grbs = pygrib.open(str(grib_path))
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Unable to open GRIB2 file") from exc

    try:
        grb = grbs.message(1)
        last_error = None
        for center_lon in (lon_360, norm_lon):
            lat1 = max(-90.0, lat - pad_degrees)
            lat2 = min(90.0, lat + pad_degrees)
            lon1 = center_lon - pad_degrees
            lon2 = center_lon + pad_degrees
            try:
                values, lats, lons = grb.data(lat1=lat1, lat2=lat2, lon1=lon1, lon2=lon2)
            except Exception as exc:
                last_error = str(exc)
                continue

            arr = np.ma.filled(values, np.nan).astype(float)
            if arr.size == 0:
                continue

            lats_arr = np.asarray(lats, dtype=float)
            lons_arr = normalize_lon_grid(np.asarray(lons, dtype=float))
            finite_mask = np.isfinite(arr)
            if not np.any(finite_mask):
                continue

            distances = haversine_miles(lat, norm_lon, lats_arr, lons_arr)
            within_radius = finite_mask & (distances <= radius_miles)

            nearest_dist = np.where(finite_mask, distances, np.inf)
            nearest_idx = int(np.argmin(nearest_dist))
            nearest_raw = float(arr.flat[nearest_idx]) if np.isfinite(nearest_dist.flat[nearest_idx]) else None
            nearest_distance = float(nearest_dist.flat[nearest_idx]) if np.isfinite(nearest_dist.flat[nearest_idx]) else None
            nearest_lat = float(lats_arr.flat[nearest_idx]) if nearest_raw is not None else None
            nearest_lon = float(lons_arr.flat[nearest_idx]) if nearest_raw is not None else None

            positive_within_radius = within_radius & (arr > 0)
            zero_within_radius = within_radius & (arr == 0)
            negative_within_radius = within_radius & (arr < 0)

            if np.any(positive_within_radius):
                positive_values = np.where(positive_within_radius, arr, -np.inf)
                max_idx = int(np.argmax(positive_values))
                radius_max = build_mesh_value(
                    raw_mm=float(arr.flat[max_idx]),
                    distance_miles=float(distances.flat[max_idx]),
                    grid_lat=float(lats_arr.flat[max_idx]),
                    grid_lon=float(lons_arr.flat[max_idx]),
                    extra_note=f"This is the maximum positive MESH value within {radius_miles:g} miles.",
                )
            elif np.any(zero_within_radius):
                zero_dist = np.where(zero_within_radius, distances, np.inf)
                zero_idx = int(np.argmin(zero_dist))
                radius_max = build_mesh_value(
                    raw_mm=0.0,
                    distance_miles=float(zero_dist.flat[zero_idx]),
                    grid_lat=float(lats_arr.flat[zero_idx]),
                    grid_lon=float(lons_arr.flat[zero_idx]),
                    extra_note=f"No positive MESH values were detected within {radius_miles:g} miles.",
                )
            else:
                radius_max = build_mesh_value(
                    raw_mm=None,
                    distance_miles=None,
                    grid_lat=None,
                    grid_lon=None,
                    extra_note=f"No valid positive or zero MESH cells were detected within {radius_miles:g} miles.",
                )

            nonnegative_values = arr[np.isfinite(arr) & (arr >= 0)]
            diagnostics = {
                "gribName": resolved.grib_name,
                "gribShortName": resolved.grib_short_name,
                "gribUnits": resolved.grib_units,
                "padDegreesUsed": round(pad_degrees, 5),
                "finiteCellCountInBox": int(np.sum(finite_mask)),
                "cellCountWithinRadius": int(np.sum(within_radius)),
                "positiveCellCountWithinRadius": int(np.sum(positive_within_radius)),
                "zeroCellCountWithinRadius": int(np.sum(zero_within_radius)),
                "negativeSentinelCellCountWithinRadius": int(np.sum(negative_within_radius)),
                "nonnegativeCellCountInBox": int(nonnegative_values.size),
                "boxMinNonnegativeRaw": sanitized_min(arr),
                "boxMaxNonnegativeRaw": sanitized_max(arr),
            }
            return {
                "nearest": build_mesh_value(
                    raw_mm=nearest_raw,
                    distance_miles=nearest_distance,
                    grid_lat=nearest_lat,
                    grid_lon=nearest_lon,
                    extra_note="This is the nearest finite MRMS grid cell to the property.",
                ),
                "radiusMax": radius_max,
                "diagnostics": diagnostics,
            }

        raise HTTPException(status_code=404, detail=f"Could not extract MESH values around point. Last error: {last_error}")
    finally:
        grbs.close()


def extract_mesh_window_debug(grib_path: Path, lat: float, lon: float, pad: float, resolved: ResolvedFile) -> dict:
    norm_lon = normalize_lon_scalar(lon)
    lon_360 = to_360_lon(norm_lon)
    try:
        grbs = pygrib.open(str(grib_path))
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Unable to open GRIB2 file") from exc

    try:
        grb = grbs.message(1)
        debug = {
            "gribName": resolved.grib_name,
            "gribShortName": resolved.grib_short_name,
            "gribUnits": resolved.grib_units,
            "rawNearestMm": None,
            "cleanedNearestStatus": None,
            "cleanedNearestMeshMm": None,
            "cleanedNearestMeshIn": None,
            "maxPositiveMmInWindow": None,
            "maxPositiveInInWindow": None,
            "validPositiveCellCount": 0,
            "finiteCellCount": 0,
            "negativeSentinelCellCount": 0,
            "zeroCellCount": 0,
            "windowMinNonnegativeRaw": None,
            "windowMaxNonnegativeRaw": None,
        }
        lat1 = max(-90.0, lat - pad)
        lat2 = min(90.0, lat + pad)
        last_error = None
        for center_lon in (lon_360, norm_lon):
            lon1 = center_lon - pad
            lon2 = center_lon + pad
            try:
                values, lats, lons = grb.data(lat1=lat1, lat2=lat2, lon1=lon1, lon2=lon2)
            except Exception as exc:
                last_error = str(exc)
                continue

            arr = np.ma.filled(values, np.nan).astype(float)
            if arr.size == 0:
                continue
            lats_arr = np.asarray(lats, dtype=float)
            lons_arr = normalize_lon_grid(np.asarray(lons, dtype=float))
            finite_mask = np.isfinite(arr)
            positive_mask = finite_mask & (arr > 0)
            negative_mask = finite_mask & (arr < 0)
            zero_mask = finite_mask & (arr == 0)
            if not np.any(finite_mask):
                continue

            distances = haversine_miles(lat, norm_lon, lats_arr, lons_arr)
            nearest_dist = np.where(finite_mask, distances, np.inf)
            idx = int(np.argmin(nearest_dist))
            raw_nearest = float(arr.flat[idx]) if np.isfinite(nearest_dist.flat[idx]) else None
            mesh_in, mesh_mm, status, _note = clean_mesh_value(raw_nearest)
            clean_raw_nearest = sanitize_nonnegative_value(raw_nearest)
            debug["rawNearestMm"] = None if clean_raw_nearest is None else round(clean_raw_nearest, 3)
            debug["cleanedNearestStatus"] = status
            debug["cleanedNearestMeshMm"] = mesh_mm
            debug["cleanedNearestMeshIn"] = mesh_in
            debug["finiteCellCount"] = int(np.sum(finite_mask))
            debug["validPositiveCellCount"] = int(np.sum(positive_mask))
            debug["negativeSentinelCellCount"] = int(np.sum(negative_mask))
            debug["zeroCellCount"] = int(np.sum(zero_mask))
            debug["windowMinNonnegativeRaw"] = sanitized_min(arr)
            debug["windowMaxNonnegativeRaw"] = sanitized_max(arr)
            if np.any(positive_mask):
                max_positive = float(np.nanmax(np.where(positive_mask, arr, np.nan)))
                debug["maxPositiveMmInWindow"] = round(max_positive, 2)
                debug["maxPositiveInInWindow"] = round(max_positive / MM_PER_INCH, 2)
            return debug
        raise HTTPException(status_code=404, detail=f"Could not extract debug window around point. Last error: {last_error}")
    finally:
        grbs.close()
