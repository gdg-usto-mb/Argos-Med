"""
Argos-Med: Edge-Native Maritime Surveillance System
Module: engine.matcher
Description: Correlates georeferenced SAR radar targets against local AIS transponder logs
             using scipy.spatial.cKDTree to detect Dark Vessels (unidentified radar returns)
             and verify compliance with Euro-Med Marine Protected Areas (MPAs).
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree

from .geoprocessor import GeoreferencedVessel


@dataclass
class AISRecord:
    """Standardized representation of a local AIS transponder broadcast."""
    mmsi: int
    vessel_name: str
    ship_type: str
    latitude: float
    longitude: float
    sog_knots: float  # Speed Over Ground
    cog_degrees: float  # Course Over Ground
    timestamp: str


@dataclass
class CorrelatedTarget:
    """Unified maritime target classification after radar-to-AIS correlation."""
    target_id: str
    classification: str  # 'VERIFIED_VESSEL' or 'DARK_VESSEL'
    sar_detection_id: int
    latitude: float
    longitude: float
    confidence: float
    estimated_length_m: float
    estimated_width_m: float
    ais_match: Optional[AISRecord]
    correlation_distance_m: Optional[float]
    nearest_mpa_name: Optional[str]
    distance_to_mpa_m: Optional[float]
    is_inside_mpa: bool


class AISCorrelator:
    """
    Spatial correlation engine utilizing cKDTree spatial indexing in metric coordinate space.
    Performs deterministic, air-gapped identification of non-cooperative dark vessels.
    """

    def __init__(
        self,
        correlation_radius_m: float = 500.0,
        reference_lat: float = 43.250,
        mpa_polygons: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Args:
            correlation_radius_m: Spatial tolerance window for radar-to-AIS pairing (meters).
            reference_lat: Latitude used for local equirectangular metric projection.
            mpa_polygons: Optional list of MPA boundary definitions with coordinates.
        """
        self.correlation_radius_m = correlation_radius_m
        self.reference_lat = reference_lat
        self.lat_scale_m = 111139.0
        self.lon_scale_m = 111139.0 * math.cos(math.radians(reference_lat))

        self.ais_records: List[AISRecord] = []
        self.ais_kdtree: Optional[cKDTree] = None
        self.mpa_zones = mpa_polygons or self._default_med_mpas()

    def _default_med_mpas(self) -> List[Dict[str, Any]]:
        """Pre-cached Euro-Med marine protected zones (Marseille & Oran corridors)."""
        return [
            {
                "name": "Parc National des Calanques",
                "authority": "Région Sud / Parc National",
                "center": (43.200, 5.450),
                "radius_m": 8500.0,
                "restriction": "Strict No-Anchor / Speed Limit 12 knots / Zero Bunkering",
            },
            {
                "name": "Reserve Naturelle Marine des Iles Habibas",
                "authority": "Wilaya d'Oran / Commissariat National du Littoral",
                "center": (35.720, -1.130),
                "radius_m": 6000.0,
                "restriction": "Total Marine Sanctuary / Prohibited Commercial Trawling",
            },
            {
                "name": "Archipel du Frioul Marine Zone",
                "authority": "Prefecture Maritime de la Mediterranee",
                "center": (43.280, 5.305),
                "radius_m": 4200.0,
                "restriction": "Traffic Separation Scheme Buffer",
            },
        ]

    def _gps_to_metric(self, lat: float, lon: float) -> Tuple[float, float]:
        """Projects (lat, lon) to local planar metric coordinates (x_meters, y_meters)."""
        x = lon * self.lon_scale_m
        y = lat * self.lat_scale_m
        return x, y

    def load_ais_csv(self, csv_path: Union[str, Path]) -> int:
        """
        Parses offline AIS transponder log CSV into internal index.
        Expected columns: mmsi, vessel_name, ship_type, latitude, longitude, sog, cog, timestamp.
        """
        csv_path = Path(csv_path)
        self.ais_records.clear()

        if not csv_path.exists():
            # Generate deterministic fallback AIS scenario
            return self._load_fallback_ais()

        coords_metric = []
        with open(csv_path, "r", encoding="utf-8") as f:
            header = [h.strip().lower() for h in f.readline().split(",")]
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue
                try:
                    record = AISRecord(
                        mmsi=int(parts[0]),
                        vessel_name=parts[1],
                        ship_type=parts[2],
                        latitude=float(parts[3]),
                        longitude=float(parts[4]),
                        sog_knots=float(parts[5]),
                        cog_degrees=float(parts[6]),
                        timestamp=parts[7] if len(parts) > 7 else "2026-11-05T08:00:00Z",
                    )
                    self.ais_records.append(record)
                    xm, ym = self._gps_to_metric(record.latitude, record.longitude)
                    coords_metric.append([xm, ym])
                except (ValueError, IndexError):
                    continue

        if coords_metric:
            self.ais_kdtree = cKDTree(np.array(coords_metric))
        return len(self.ais_records)

    def _load_fallback_ais(self) -> int:
        """Loads offline synthetic baseline AIS logs for Marseille and Oran corridors."""
        synthetic = [
            AISRecord(227001230, "CMA CGM ANTOINE", "Container Ship", 43.235, 5.230, 16.4, 142.0, "2026-11-05T08:15:00Z"),
            AISRecord(228394000, "JEAN CHARCOT", "Oceanographic Research", 43.275, 5.340, 8.2, 210.0, "2026-11-05T08:14:30Z"),
            AISRecord(605001140, "TARIQ IBN ZIYAD", "Passenger / Ro-Ro", 35.790, -0.610, 19.1, 045.0, "2026-11-05T08:12:00Z"),
        ]
        self.ais_records = synthetic
        coords = [self._gps_to_metric(r.latitude, r.longitude) for r in synthetic]
        self.ais_kdtree = cKDTree(np.array(coords))
        return len(synthetic)

    def correlate(
        self, sar_vessels: List[GeoreferencedVessel]
    ) -> List[CorrelatedTarget]:
        """
        Executes cKDTree spatial nearest-neighbor search against AIS points.
        Identifies targets lacking active AIS broadcast as high-risk 'DARK_VESSEL'.
        """
        correlated: List[CorrelatedTarget] = []

        if not self.ais_records or self.ais_kdtree is None:
            self._load_fallback_ais()

        for vessel in sar_vessels:
            xm, ym = self._gps_to_metric(vessel.latitude, vessel.longitude)

            # Query KD-Tree for nearest AIS transponder beacon
            dist, idx = self.ais_kdtree.query([xm, ym], distance_upper_bound=self.correlation_radius_m * 4)

            matched_ais: Optional[AISRecord] = None
            classification = "DARK_VESSEL"
            dist_m: Optional[float] = None

            if dist <= self.correlation_radius_m:
                matched_ais = self.ais_records[idx]
                classification = "VERIFIED_VESSEL"
                dist_m = round(float(dist), 1)

            # Check proximity to Marine Protected Areas
            mpa_name, mpa_dist, inside_mpa = self._evaluate_mpa_status(vessel.latitude, vessel.longitude)

            target_code = "VER" if classification == "VERIFIED_VESSEL" else "DRK"
            target_id = f"ARGOS-{target_code}-{vessel.detection_id:03d}"

            correlated.append(
                CorrelatedTarget(
                    target_id=target_id,
                    classification=classification,
                    sar_detection_id=vessel.detection_id,
                    latitude=vessel.latitude,
                    longitude=vessel.longitude,
                    confidence=vessel.confidence,
                    estimated_length_m=vessel.length_meters,
                    estimated_width_m=vessel.width_meters,
                    ais_match=matched_ais,
                    correlation_distance_m=dist_m,
                    nearest_mpa_name=mpa_name,
                    distance_to_mpa_m=mpa_dist,
                    is_inside_mpa=inside_mpa,
                )
            )

        return correlated

    def _evaluate_mpa_status(
        self, lat: float, lon: float
    ) -> Tuple[Optional[str], float, bool]:
        """Evaluates whether coordinates fall inside or adjacent to a sensitive Euro-Med MPA."""
        closest_name = None
        min_dist = float("inf")
        inside = False

        vx, vy = self._gps_to_metric(lat, lon)

        for mpa in self.mpa_zones:
            c_lat, c_lon = mpa["center"]
            cx, cy = self._gps_to_metric(c_lat, c_lon)
            d = math.hypot(vx - cx, vy - cy)

            if d < min_dist:
                min_dist = d
                closest_name = mpa["name"]

            if d <= mpa["radius_m"]:
                inside = True
                closest_name = mpa["name"]
                min_dist = 0.0
                break

        return closest_name, round(min_dist, 1), inside
