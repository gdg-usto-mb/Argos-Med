"""
Argos-Med AIS Live Cache Collector
Collects and caches live coastal AIS transponder messages from aisstream.io
into a local SQLite database for offline velocity-aware radar correlation.
"""

import asyncio
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ais", "live_cache.sqlite")

# Target Surveillance Corridors (Bounding Boxes: [[lat_min, lon_min], [lat_max, lon_max]])
BOUNDING_BOXES = [
    # Marseille Approaches & Calanques
    [[43.100, 5.100], [43.350, 5.550]],
    # Oran Sector & Habibas Marine Reserve
    [[35.650, -1.250], [35.900, -0.200]],
]


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Initialize local SQLite AIS cache table and spatial indexes."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ais_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mmsi INTEGER NOT NULL,
            vessel_name TEXT,
            ship_type INTEGER,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            sog REAL,
            cog REAL,
            true_heading INTEGER,
            timestamp_utc TEXT NOT NULL,
            received_at TEXT NOT NULL
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mmsi ON ais_positions(mmsi);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON ais_positions(timestamp_utc);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lat_lon ON ais_positions(latitude, longitude);")
    conn.commit()
    return conn


def save_position_report(conn: sqlite3.Connection, data: dict):
    """Save parsed AIS position report into SQLite cache."""
    now_utc = datetime.now(timezone.utc).isoformat()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO ais_positions (
            mmsi, vessel_name, ship_type, latitude, longitude,
            sog, cog, true_heading, timestamp_utc, received_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("mmsi"),
            data.get("vessel_name", "UNKNOWN"),
            data.get("ship_type", 0),
            data.get("latitude"),
            data.get("longitude"),
            data.get("sog", 0.0),
            data.get("cog", 0.0),
            data.get("true_heading", 511),
            data.get("timestamp_utc", now_utc),
            now_utc,
        ),
    )
    conn.commit()


async def run_collector(api_key: str):
    """Connect to aisstream.io WebSocket stream and cache incoming messages."""
    try:
        import websockets
    except ImportError:
        print("[ERROR] websockets package required. Install with: pip install websockets")
        sys.exit(1)

    url = "wss://stream.aisstream.io/v0/stream"
    sub_message = {
        "APIKey": api_key,
        "BoundingBoxes": BOUNDING_BOXES,
        "FilterMessageTypes": ["PositionReport"],
    }

    conn = init_db()
    print(f"[INFO] Connecting to aisstream.io with bounding boxes: {BOUNDING_BOXES}")
    print(f"[INFO] Caching live positions into: {DB_PATH}")

    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(sub_message))
        async for msg in ws:
            payload = json.loads(msg)
            msg_type = payload.get("MessageType")
            if msg_type == "PositionReport":
                meta = payload.get("MetaData", {})
                report = payload.get("Message", {}).get("PositionReport", {})
                pos_data = {
                    "mmsi": meta.get("MMSI"),
                    "vessel_name": meta.get("ShipName", "").strip(),
                    "latitude": report.get("Latitude"),
                    "longitude": report.get("Longitude"),
                    "sog": report.get("Sog"),
                    "cog": report.get("Cog"),
                    "true_heading": report.get("TrueHeading"),
                    "timestamp_utc": meta.get("time_utc"),
                }
                save_position_report(conn, pos_data)
                print(
                    f"[AIS] MMSI: {pos_data['mmsi']} | {pos_data['vessel_name']} | "
                    f"Pos: ({pos_data['latitude']:.4f}, {pos_data['longitude']:.4f}) | "
                    f"SOG: {pos_data['sog']} kn | COG: {pos_data['cog']}°"
                )


if __name__ == "__main__":
    api_key = os.environ.get("AISSTREAM_API_KEY")
    if not api_key:
        print("[NOTICE] AISSTREAM_API_KEY not set. Initializing local database schema only.")
        init_db()
        print(f"[SUCCESS] Database schema initialized at {DB_PATH}")
    else:
        asyncio.run(run_collector(api_key))
