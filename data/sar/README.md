# Copernicus Sentinel-1 SAR Data Guide

This directory holds offline Synthetic Aperture Radar (SAR) rasters for Argos-Med processing.

## 1. Scene Specifications
- **Sensor**: Sentinel-1 C-SAR (C-Band, 5.405 GHz)
- **Product Type**: Level-1 Ground Range Detected (GRD)
- **Swath Mode**: Interferometric Wide (IW)
- **Polarizations**: VV (Vertical-Vertical) and VH (Vertical-Horizontal)
- **Pixel Spacing**: 10m x 10m (High Resolution)
- **Coordinate Reference System**: EPSG:4326 (WGS84)

## 2. Included Synthetic Demo Rasters
To enable full 100% offline development and unit testing without requiring large multi-gigabyte satellite downloads, Argos-Med includes synthetic georeferenced GeoTIFFs:
- `marseille_s1_synthetic.tif`: Simulated radar backscatter covering Golfe du Lion, Frioul archipelago, and Parc National des Calanques. Contains 4 verified vessel returns and 2 dark vessel targets.
- `oran_s1_synthetic.tif`: Simulated radar backscatter covering Baie d'Oran, Arzew channel, and the protected Habibas Islands reserve.

## 3. Sourcing Production Sentinel-1 Scenes
To acquire operational Copernicus scenes for Euro-Med surveillance:
1. Register at the **Copernicus Data Space Ecosystem (CDSE)**: [dataspace.copernicus.eu](https://dataspace.copernicus.eu).
2. Filter by:
   - Collection: `Sentinel-1`
   - Product Type: `GRD`
   - Sensor Mode: `IW`
   - Polarization: `VV+VH`
3. Download the `.SAFE` product zip or extract the calibrated measurement GeoTIFF (`measurement/s1a-iw-grd-*.tiff`).
4. Place the target sector GeoTIFF crop into this directory.
