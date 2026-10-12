"""
Argos-Med Deterministic Severity Rule Table (engine/ruler.py)
Implements published, deterministic rule-based triage for maritime radar contacts.
Assigns severity codes (GREEN / AMBER / RED) and analyst review actions.
Contractually forbidden from assigning military interdiction orders or hallucinating intent.
"""

from typing import Dict, Any, List, Optional
import math


class SeverityRuler:
    """
    Deterministic rule engine that maps correlated radar telemetry
    into standardized severity codes and analyst review recommendations.
    Adheres to EU AI Act Art. 14 Human Oversight (Decision Support System posture).
    """

    SEVERITY_GREEN = "GREEN"    # Routine verified vessel
    SEVERITY_AMBER = "AMBER"    # Unexplained contact outside sensitive zones
    SEVERITY_RED = "RED"        # Unexplained contact inside or proximate to MPA

    ACTION_VERIFY = "VERIFY: Cross-check flag-state registry and VMS via coastal authorities."
    ACTION_INVESTIGATE = "INVESTIGATE: Re-check next satellite pass; if persistent, task patrol assets."
    ACTION_FLAG_ANALYST = "FLAG-FOR-ANALYST: Urgent review required for sensitive zone boundary proximity."
    ACTION_LOG_ROUTINE = "LOG-ROUTINE: Normal cooperative transit logged."

    def __init__(self, mpa_buffer_nm: float = 2.0):
        self.mpa_buffer_nm = mpa_buffer_nm

    def evaluate_contact(
        self,
        status: str,
        radar_conf: float,
        range_size_m: float,
        reception_density: int,
        is_inside_mpa: bool,
        distance_to_mpa_nm: Optional[float] = None,
        mpa_name: Optional[str] = None,
        is_persistent_across_passes: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single radar detection against the published deterministic rule table.

        Classification States:
        - VERIFIED_VESSEL: Valid AIS match within velocity-aware gate.
        - INFRASTRUCTURE: Unmatched but static across >= 2 satellite passes.
        - INCONCLUSIVE: Unmatched but local AIS reception density is zero (coverage gap).
        - UNEXPLAINED_CONTACT: Unmatched despite nominal AIS reception density.
        """

        # Rule 1: Infrastructure filter (static non-AIS contact across passes)
        if is_persistent_across_passes and status != "VERIFIED_VESSEL":
            return {
                "classification": "INFRASTRUCTURE",
                "severity": self.SEVERITY_GREEN,
                "rationale": "Static non-AIS radar return detected across >= 2 passes (platform/aquaculture/fixed gear).",
                "recommended_actions": ["LOG: Mark coordinate as known charted maritime infrastructure."],
                "requires_analyst_review": False
            }

        # Rule 2: Coverage Gap Filter
        if status == "INCONCLUSIVE" or reception_density == 0:
            return {
                "classification": "INCONCLUSIVE",
                "severity": self.SEVERITY_GREEN,
                "rationale": "Zero local AIS transponder messages in 20 km pass window (probable SAT-AIS coverage gap).",
                "recommended_actions": ["AWAIT-PASS: Defer evaluation until coverage restored on subsequent pass."],
                "requires_analyst_review": False
            }

        # Rule 3: Verified Vessel (Cooperative)
        if status == "VERIFIED_VESSEL":
            if is_inside_mpa:
                return {
                    "classification": "VERIFIED_VESSEL",
                    "severity": self.SEVERITY_AMBER,
                    "rationale": f"Cooperative vessel with valid AIS navigating inside {mpa_name or 'MPA boundary'}.",
                    "recommended_actions": [self.ACTION_VERIFY],
                    "requires_analyst_review": True
                }
            return {
                "classification": "VERIFIED_VESSEL",
                "severity": self.SEVERITY_GREEN,
                "rationale": "Radar contact correlated with active AIS transponder within velocity-aware gate.",
                "recommended_actions": [self.ACTION_LOG_ROUTINE],
                "requires_analyst_review": False
            }

        # Rule 4: Unexplained Contact (AIS-Dark) inside or proximate to Marine Protected Area
        dist = distance_to_mpa_nm if distance_to_mpa_nm is not None else 999.0
        if is_inside_mpa or dist <= self.mpa_buffer_nm:
            zone_desc = "INSIDE sanctuary boundary" if is_inside_mpa else f"{dist:.1f} NM from sanctuary boundary"
            return {
                "classification": "UNEXPLAINED_CONTACT",
                "severity": self.SEVERITY_RED,
                "rationale": f"Unexplained radar contact ({zone_desc} of {mpa_name or 'MPA'}) with nominal AIS reception density ({reception_density} neighbors).",
                "recommended_actions": [
                    self.ACTION_FLAG_ANALYST,
                    self.ACTION_INVESTIGATE,
                    self.ACTION_VERIFY
                ],
                "requires_analyst_review": True
            }

        # Rule 5: Unexplained Contact in Open Corridor
        return {
            "classification": "UNEXPLAINED_CONTACT",
            "severity": self.SEVERITY_AMBER,
            "rationale": f"Unexplained radar contact in open corridor with nominal AIS reception density ({reception_density} neighbors in 20 km window).",
            "recommended_actions": [
                self.ACTION_VERIFY,
                self.ACTION_INVESTIGATE
            ],
            "requires_analyst_review": True
        }


def format_deterministic_sitrep(
    ref_id: str,
    dtg: str,
    sector: str,
    lat: float,
    lon: float,
    conf: float,
    size_class: str,
    evaluation: Dict[str, Any],
    reception_count: int,
    pass_history: str = "Single-pass contact",
    mpa_info: str = "Outside marine sanctuaries",
    sea_state: str = "ERA5 10m wind 6.1 m/s (detection class reliable)"
) -> str:
    """Format standard naval SITREP using published deterministic fields (Deep-Dive B)."""
    actions_formatted = "\n   ".join(evaluation.get("recommended_actions", []))
    review_status = "HUMAN DECISION REQUIRED" if evaluation.get("requires_analyst_review") else "ROUTINE LOG"

    return f"""=== MARITIME SITREP -- DECISION SUPPORT OUTPUT (NOT AN INTERCEPT ORDER) ===
REF:     {ref_id:<27} DTG: {dtg}
SECTOR:  {sector}
SOURCE:  Copernicus Sentinel-1 IW GRD // local AIS cache

1. CONTACT
   POS: {lat:.4f} N {abs(lon):.4f} {'W' if lon < 0 else 'E'} (WGS84) | conf {conf:.2f}, VV+VH
   SIZE CLASS: {size_class} (range-dimension estimate; azimuth smear excluded)

2. AIS CORRELATION -- STATUS: {evaluation.get('classification', 'UNEXPLAINED')}
   Gate: velocity-aware ellipse (r0 500 m, azimuth-elongated) -- {evaluation.get('rationale', '')}
   Reception: nominal -- {reception_count} AIS contacts within 20 km in pass window.
   Persistence: {pass_history}

3. CONTEXT
   MPA Proximity: {mpa_info}
   Sea state: {sea_state}

4. RECOMMENDED ACTIONS -- ANALYST REVIEW REQUIRED ({review_status})
   {actions_formatted}

5. SYSTEM NOTES
   Detector: YOLOv8n INT8, xView3-validated.
   Severity: {evaluation.get('severity', 'UNKNOWN')} -- assigned by published rule table, not by language model.
"""


if __name__ == "__main__":
    ruler = SeverityRuler(mpa_buffer_nm=2.0)
    # Test Red Case (Unexplained contact inside MPA)
    eval_res = ruler.evaluate_contact(
        status="UNEXPLAINED_CONTACT",
        radar_conf=0.88,
        range_size_m=48.0,
        reception_density=14,
        is_inside_mpa=True,
        mpa_name="Calanques National Park MPA",
        is_persistent_across_passes=False
    )
    sitrep = format_deterministic_sitrep(
        ref_id="AM-20261105-0842-01",
        dtg="050842Z NOV 26",
        sector="Marseille Approaches",
        lat=43.1250,
        lon=5.3520,
        conf=0.88,
        size_class="Medium (approx 48m LOA)",
        evaluation=eval_res,
        reception_count=14,
        mpa_info="Inside Calanques National Park MPA boundary"
    )
    print("--- EVALUATION RESULT ---")
    print(eval_res)
    print("\n--- SITREP DEMO ---")
    print(sitrep)

