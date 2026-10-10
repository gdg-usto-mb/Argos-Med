"""Argos-Med End-to-End Pipeline Smoke Test"""
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from engine.detector import SARShipDetector
from engine.geoprocessor import SARGeoprocessor
from engine.matcher import AISCorrelator
from engine.reporter import TacticalSITREPGenerator

def test_pipeline():
    print("[1] Testing Detector...")
    detector = SARShipDetector()
    img = np.zeros((2048, 2048), dtype=np.uint8)
    dets, lat = detector.detect(img)
    print(f"Detected {len(dets)} targets in {lat:.2f}ms")
    assert len(dets) > 0

    print("[2] Testing Geoprocessor...")
    geoproc = SARGeoprocessor(bounds_wgs84=(5.10, 43.10, 5.55, 43.35))
    vessels = geoproc.process_detections(dets)
    print(f"Georeferenced {len(vessels)} vessels")
    assert len(vessels) == len(dets)

    print("[3] Testing AIS Matcher...")
    matcher = AISCorrelator()
    count = matcher.load_ais_csv(str(ROOT_DIR / "data" / "ais" / "marseille_ais.csv"))
    print(f"Loaded {count} AIS records")
    correlated = matcher.correlate(vessels)
    darks = [c for c in correlated if c.classification == "DARK_VESSEL"]
    print(f"Correlated {len(correlated)} targets ({len(darks)} Dark Vessels)")
    assert len(correlated) > 0

    print("[4] Testing SITREP Generator...")
    reporter = TacticalSITREPGenerator()
    sitrep = reporter.generate_sitrep(darks[0], corridor_context="Marseille Calanques")
    print(f"Generated SITREP: {sitrep['incident_ref']} (Threat: {sitrep['threat_level']})")
    assert "sitrep_markdown" in sitrep
    print("\n[+] SUCCESS: All Argos-Med pipeline components operational!")

if __name__ == "__main__":
    test_pipeline()
