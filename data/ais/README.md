# Offline AIS Transponder Telemetry Guide

This directory holds local Automatic Identification System (AIS) transponder logs for spatial correlation against satellite radar returns.

## 1. Schema Specifications
AIS CSV dumps adhere to standard coastal station formats:
- `mmsi`: Maritime Mobile Service Identity (9-digit unique vessel identifier)
- `vessel_name`: Name registered with ITU / national maritime authority
- `ship_type`: Vessel classification (e.g., Container Ship, Tanker, Passenger, Pilot)
- `latitude`: WGS84 decimal degrees (North positive)
- `longitude`: WGS84 decimal degrees (East positive)
- `sog`: Speed Over Ground in knots
- `cog`: Course Over Ground in degrees (000.0 - 359.9)
- `timestamp`: ISO 8601 UTC timestamp of broadcast

## 2. Spatial Correlator Logic
- Physical radar returns detected in Sentinel-1 SAR imagery are projected to WGS84 GPS coordinates.
- Using `scipy.spatial.cKDTree` in metric coordinate space, the engine searches for matching AIS transponders within a radius of $r = 500\text{ meters}$.
- If an AIS signal exists within 500m -> Target is flagged as `VERIFIED_VESSEL` (Cooperative).
- If zero AIS signals exist within 500m -> Target is flagged as `DARK_VESSEL` (Non-cooperative, suspected transponder shutoff).
