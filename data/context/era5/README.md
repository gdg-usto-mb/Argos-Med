# ERA5 Ocean Wind & Sea-State Context

This directory stores offline ECMWF ERA5 10m ocean surface wind speed layers for satellite acquisition timestamps.

- **Source**: Copernicus Climate Data Store (CDS)
- **Reference**: https://cds.climate.copernicus.eu
- **Operational Role**: Radar detection reliability depends directly on sea state. High wind (> 12 m/s) causes severe capillary wave clutter, degrading small-boat probability of detection (PoD). Low wind (< 3 m/s) can cause dark slick false alarms. ERA5 wind velocity (m/s) is attached to every SITREP as operational reliability context.
