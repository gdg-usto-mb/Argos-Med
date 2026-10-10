"""
Argos-Med: Edge-Native Maritime Surveillance System
Module: engine.geoprocessor
Description: Transforms SAR pixel-space detections into georeferenced WGS84 (EPSG:4326)
             GPS coordinates using Rasterio affine transformations and CRS reprojection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import rasterio
    from rasterio.transform import xy as rasterio_xy
    from rasterio.warp import transform as warp_transform
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from .detector import SARDetection


@dataclass
class GeoreferencedVessel:
    """SAR detection correlated with terrestrial geospatial coordinates."""
    detection_id: int
    latitude: float
    longitude: float
    bbox_polygon_wgs84: List[Tuple[float, float]]  # List of (lon, lat) tuples
    confidence: float
    length_meters: float
    width_meters: float
    raw_pixel_centroid: Tuple[float, float]


class SARGeoprocessor:
    """
    Geospatial engine translating Sentinel-1 synthetic aperture radar slant/ground
    range pixel geometry into standardized WGS84 cartographic coordinates.
    """

    def __init__(
        self,
        geotiff_path: Optional[Union[str, Path]] = None,
        bounds_wgs84: Optional[Tuple[float, float, float, float]] = None,
        raster_dims: Tuple[int, int] = (2048, 2048),
        nominal_pixel_spacing_m: float = 10.0,
    ):
        """
        Args:
            geotiff_path: Path to Copernicus Sentinel-1 GeoTIFF product.
            bounds_wgs84: Fallback bounding box (min_lon, min_lat, max_lon, max_lat).
            raster_dims: (height, width) of the raster scene.
            nominal_pixel_spacing_m: Ground Range Detected (GRD) pixel resolution (~10m).
        """
        self.geotiff_path = Path(geotiff_path) if geotiff_path else None
        self.pixel_spacing_m = nominal_pixel_spacing_m
        self.dataset = None
        self.affine_transform = None
        self.source_crs = "EPSG:4326"
        self.raster_dims = raster_dims
        self.bounds_wgs84 = bounds_wgs84

        self._load_spatial_metadata()

    def _load_spatial_metadata(self) -> None:
        """Extracts Affine matrix, projection CRS, and raster extent from GeoTIFF."""
        if self.geotiff_path and self.geotiff_path.exists() and RASTERIO_AVAILABLE:
            try:
                self.dataset = rasterio.open(str(self.geotiff_path))
                self.affine_transform = self.dataset.transform
                self.source_crs = self.dataset.crs
                self.raster_dims = (self.dataset.height, self.dataset.width)
                return
            except Exception:
                pass

        # Synthesize affine matrix from reference bounding box (Marseille or Oran default)
        if not self.bounds_wgs84:
            # Default: Gulf of Lion / Marseille approach corridor
            self.bounds_wgs84 = (5.100, 43.100, 5.500, 43.350)

        min_lon, min_lat, max_lon, max_lat = self.bounds_wgs84
        h, w = self.raster_dims

        # Affine coefficients: x = a*col + b*row + c, y = d*col + e*row + f
        self.affine_a = (max_lon - min_lon) / float(w)
        self.affine_b = 0.0
        self.affine_c = min_lon
        self.affine_d = 0.0
        self.affine_e = -(max_lat - min_lat) / float(h)
        self.affine_f = max_lat

    def pixel_to_gps(self, px: float, py: float) -> Tuple[float, float]:
        """
        Transforms pixel column (x) and row (y) into (latitude, longitude) in WGS84.
        """
        if self.dataset and RASTERIO_AVAILABLE:
            x, y = rasterio_xy(self.affine_transform, py, px)
            if self.source_crs != "EPSG:4326":
                xs, ys = warp_transform(self.source_crs, "EPSG:4326", [x], [y])
                return float(ys[0]), float(xs[0])
            return float(y), float(x)

        # Mathematical affine calculation
        lon = self.affine_a * px + self.affine_b * py + self.affine_c
        lat = self.affine_d * px + self.affine_e * py + self.affine_f
        return round(float(lat), 6), round(float(lon), 6)

    def process_detections(
        self, detections: List[SARDetection]
    ) -> List[GeoreferencedVessel]:
        """
        Converts raw SAR detections into WGS84 GPS targets with estimated physical dimensions.
        """
        georeferenced_targets = []

        for det in detections:
            cx_px, cy_px = det.centroid_px
            lat, lon = self.pixel_to_gps(cx_px, cy_px)

            # Convert 4 corners of bounding box
            x1, y1, x2, y2 = det.bbox_xyxy
            c1_lat, c1_lon = self.pixel_to_gps(x1, y1)
            c2_lat, c2_lon = self.pixel_to_gps(x2, y1)
            c3_lat, c3_lon = self.pixel_to_gps(x2, y2)
            c4_lat, c4_lon = self.pixel_to_gps(x1, y2)

            poly_wgs84 = [
                (c1_lon, c1_lat),
                (c2_lon, c2_lat),
                (c3_lon, c3_lat),
                (c4_lon, c4_lat),
                (c1_lon, c1_lat),  # Closed polygon
            ]

            length_m = det.estimated_length_px * self.pixel_spacing_m
            width_m = det.estimated_width_px * self.pixel_spacing_m

            georeferenced_targets.append(
                GeoreferencedVessel(
                    detection_id=det.detection_id,
                    latitude=lat,
                    longitude=lon,
                    bbox_polygon_wgs84=poly_wgs84,
                    confidence=det.confidence,
                    length_meters=round(length_m, 1),
                    width_meters=round(width_m, 1),
                    raw_pixel_centroid=(cx_px, cy_px),
                )
            )

        return georeferenced_targets

    def to_geojson(self, vessels: List[GeoreferencedVessel]) -> Dict[str, Any]:
        """Serializes targets into RFC 7946 compliant GeoJSON FeatureCollection."""
        features = []
        for v in vessels:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [v.longitude, v.latitude],
                },
                "properties": {
                    "detection_id": v.detection_id,
                    "confidence": v.confidence,
                    "estimated_length_m": v.length_meters,
                    "estimated_width_m": v.width_meters,
                    "footprint_polygon": v.bbox_polygon_wgs84,
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
            "features": features,
        }
