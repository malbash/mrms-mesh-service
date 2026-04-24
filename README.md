# MRMS MESH Service

FastAPI microservice for NOAA MRMS MESH hail-size lookup.

## Main endpoint

```text
GET /mesh?lat=34.960468&lon=-81.880455&date=2025-08-20&radiusMiles=5&boundaryHours=3
```

The endpoint returns:
- selected maximum MESH within the radius and expanded search window
- daily `MESH_Max_1440min` result
- hourly boundary `MESH_Max_60min` results only when hail is detected
- hidden no-hail/no-data hourly counts to keep response concise

## Health check

```text
GET /healthz
```

## Render deployment

Use Docker Web Service. Health check path: `/healthz`. Recommended instance: 2 GB RAM or higher.
