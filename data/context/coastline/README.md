# Coastline Masking (GSHHG)

This directory holds shoreline vector layers and buffered land masks to eliminate coastal clutter.

- **Source**: Global Self-consistent, Hierarchical, High-resolution Geography Database (GSHHG)
- **Reference**: https://www.soest.hawaii.edu/pwessel/gshhg/
- **Pipeline Role**: Dilated 500m coastal buffer mask applied before YOLOv8 tiling. Vessels within the near-shore surf zone are masked out in v1 to prevent false alarms from breaking surf, breakwaters, and harbor structures.
