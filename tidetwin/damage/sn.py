"""DNV-RP-C203 S-N curves and Miner accumulation.

The two-slope curve is

.. math::
    \\log_{10} N = \\log_{10} \\bar{a} - m \\log_{10}\\!\\left[
        \\Delta\\sigma \\left(\\frac{t}{t_{ref}}\\right)^{k} \\right]

with ``(log a1, m1)`` below the knee at 1e7 cycles and ``(log a2, m2)`` above it,
and ``k`` the thickness exponent (DNV-RP-C203 Section 2.4).

**The curve constants are not shipped.** DNV-RP-C203 is a paid standard, and
transcribing its Table 2-2 from memory would produce numbers that look
authoritative and cannot be checked - the precise failure mode this application
exists to prevent. Supply them once, from your own copy of the standard, in
``data/sn/dnv_rp_c203_T.json``; every claim that depends on them reports
``UNTESTABLE - DATA MISSING`` until you do.

Only the C6 no-update baseline needs this. C1 through C5, C7 and the nuisance
budget do not touch it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..provenance import Citation, DataUnavailable

__all__ = ["SNCurve", "load_curve", "sn_status", "cycles_to_failure", "miner_damage"]

DNV_C203 = Citation(
    document="DNV-RP-C203, Fatigue Design of Offshore Steel Structures",
    locator="Table 2-2 (S-N curves in seawater with cathodic protection), curve T; Section 2.4",
)

SCHEMA = """{
  "curve": "T",
  "environment": "seawater with cathodic protection",
  "citation": {"document": "DNV-RP-C203", "locator": "Table 2-2", "year": 2016},
  "log_a1": <float>, "m1": <float>,
  "log_a2": <float>, "m2": <float>,
  "knee_cycles": 1.0e7,
  "thickness_exponent": <float>,
  "t_ref_mm": 32.0,
  "stress_units": "MPa"
}"""


@dataclass(frozen=True)
class SNCurve:
    log_a1: float
    m1: float
    log_a2: float
    m2: float
    knee_cycles: float
    thickness_exponent: float
    t_ref_mm: float
    name: str
    environment: str
    citation: Citation


def sn_status(root: Path | None = None) -> tuple[bool, str]:
    d = root or Path(__file__).resolve().parents[2] / "data" / "sn"
    p = d / "dnv_rp_c203_T.json"
    if p.is_file():
        return True, f"S-N curve parameters loaded from {p}."
    return False, (
        "DATA UNAVAILABLE - DNV-RP-C203 curve T constants are not shipped (paid standard). "
        f"Create {p} with the schema:\n{SCHEMA}"
    )


def load_curve(root: Path | None = None) -> SNCurve:
    ok, why = sn_status(root)
    if not ok:
        raise DataUnavailable("DNV-RP-C203 curve T", why, "Transcribe Table 2-2 into data/sn/.")
    d = root or Path(__file__).resolve().parents[2] / "data" / "sn"
    spec = json.loads((d / "dnv_rp_c203_T.json").read_text(encoding="utf-8"))
    c = spec.get("citation", {})
    return SNCurve(
        log_a1=float(spec["log_a1"]),
        m1=float(spec["m1"]),
        log_a2=float(spec["log_a2"]),
        m2=float(spec["m2"]),
        knee_cycles=float(spec.get("knee_cycles", 1.0e7)),
        thickness_exponent=float(spec["thickness_exponent"]),
        t_ref_mm=float(spec.get("t_ref_mm", 32.0)),
        name=str(spec.get("curve", "T")),
        environment=str(spec.get("environment", "")),
        citation=Citation(
            document=c.get("document", DNV_C203.document),
            locator=c.get("locator", DNV_C203.locator),
            year=c.get("year"),
        ),
    )


def cycles_to_failure(
    stress_range_MPa: np.ndarray, curve: SNCurve, thickness_mm: float
) -> np.ndarray:
    """Cycles to failure at a given stress range, with the thickness correction."""
    s = np.asarray(stress_range_MPa, float)
    corr = max(thickness_mm / curve.t_ref_mm, 1.0) ** curve.thickness_exponent
    se = np.maximum(s * corr, 1e-12)
    n1 = 10.0 ** (curve.log_a1 - curve.m1 * np.log10(se))
    n2 = 10.0 ** (curve.log_a2 - curve.m2 * np.log10(se))
    return np.where(n1 <= curve.knee_cycles, n1, n2)


def miner_damage(
    stress_ranges_MPa: np.ndarray,
    counts: np.ndarray,
    curve: SNCurve,
    thickness_mm: float,
) -> float:
    """Palmgren-Miner accumulated damage; failure is conventionally at 1.0."""
    n = np.asarray(counts, float)
    N = cycles_to_failure(stress_ranges_MPa, curve, thickness_mm)
    return float(np.sum(n / np.maximum(N, 1e-300)))
