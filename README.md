# MRMS MESH Service

Small Python FastAPI microservice for returning NOAA MRMS MESH hail estimates.

## Endpoint

`GET /mesh?lat=33.80&lon=-84.31&date=2025-06-26`

Example response:

```json
{
  "meshIn": 1.42,
  "source": "NOAA MRMS MESH",
  "timestamp": "2025-06-26T23:30:00Z",
  "note": "Maximum Estimated Size of Hail 1440-minute swath at nearest MRMS grid point"
}
