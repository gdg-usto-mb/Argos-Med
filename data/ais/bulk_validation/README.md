# Bulk AIS Validation Datasets

This directory provides public high-volume historical AIS data dumps to validate the velocity-gated bipartite matcher (`engine/matcher.py`) at scale before deployment in Euro-Med corridors.

- **Sources**:
  - MarineCadastre (US Coastal Waterways): https://marinecadastre.gov/ais/
  - Danish Maritime Authority (DMA European Waters): https://www.dma.dk
- **Validation Objective**:
  - Benchmark bipartite assignment (`scipy.optimize.linear_sum_assignment`) across thousands of simultaneous tracks in dense traffic anchorages.
  - Evaluate false match rates with varying radial velocity gates and propagation deltas.
