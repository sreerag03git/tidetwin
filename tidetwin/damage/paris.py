"""BS 7910 Paris-law crack growth.

.. math:: \\frac{da}{dN} = A (\\Delta K)^{m} \\quad\\text{for}\\quad \\Delta K > \\Delta K_{th}

**The constants are not shipped.** BS 7910:2019 is a paid standard, and its
Table 8 values for steels in a marine environment with cathodic protection are
strongly unit-dependent - ``A`` differs by many orders of magnitude between
``N/mm^{3/2}`` and ``MPa m^{1/2}`` conventions. Transcribing them from memory
would be the single easiest way to produce a plausible-looking crack growth
curve that is wrong by a factor of a thousand. Supply them from your own copy in
``data/paris/bs7910_marine_cp.json``.

The integration machinery below is complete and tested; only the constants are
missing, so the moment a user supplies them C6 becomes live.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..provenance import Citation, DataUnavailable
from .newman_raju import sif

__all__ = ["ParisLaw", "load_constants", "paris_status", "growth_rate", "integrate_growth"]

BS7910 = Citation(
    document="BS 7910:2019, Guide to methods for assessing the acceptability of flaws "
    "in metallic structures",
    locator="Table 8 (fatigue crack growth laws for steels in marine environments "
    "with cathodic protection)",
    year=2019,
)

SCHEMA = """{
  "citation": {"document": "BS 7910:2019", "locator": "Table 8", "year": 2019},
  "environment": "marine, cathodic protection",
  "stage": "simplified two-stage or single-stage",
  "A": <float>, "m": <float>,
  "delta_K_threshold": <float>,
  "units": {"delta_K": "MPa.m^0.5", "da_dN": "m/cycle"}
}"""


@dataclass(frozen=True)
class ParisLaw:
    A: float
    m: float
    delta_K_threshold: float
    units_delta_K: str
    units_da_dN: str
    environment: str
    citation: Citation


def paris_status(root: Path | None = None) -> tuple[bool, str]:
    d = root or Path(__file__).resolve().parents[2] / "data" / "paris"
    p = d / "bs7910_marine_cp.json"
    if p.is_file():
        return True, f"Paris constants loaded from {p}."
    return False, (
        "DATA UNAVAILABLE - BS 7910:2019 Table 8 constants are not shipped (paid standard). "
        f"Create {p} with the schema:\n{SCHEMA}\n"
        "Check the units carefully: A differs by orders of magnitude between "
        "N/mm^1.5 and MPa.m^0.5 conventions."
    )


def load_constants(root: Path | None = None) -> ParisLaw:
    ok, why = paris_status(root)
    if not ok:
        raise DataUnavailable("BS 7910 Paris constants", why, "Transcribe Table 8 into data/paris/.")
    d = root or Path(__file__).resolve().parents[2] / "data" / "paris"
    spec = json.loads((d / "bs7910_marine_cp.json").read_text(encoding="utf-8"))
    c = spec.get("citation", {})
    u = spec.get("units", {})
    return ParisLaw(
        A=float(spec["A"]),
        m=float(spec["m"]),
        delta_K_threshold=float(spec.get("delta_K_threshold", 0.0)),
        units_delta_K=str(u.get("delta_K", "MPa.m^0.5")),
        units_da_dN=str(u.get("da_dN", "m/cycle")),
        environment=str(spec.get("environment", "")),
        citation=Citation(
            document=c.get("document", BS7910.document),
            locator=c.get("locator", BS7910.locator),
            year=c.get("year"),
        ),
    )


def growth_rate(delta_K: np.ndarray, law: ParisLaw) -> np.ndarray:
    """``da/dN``, zero below the threshold."""
    dk = np.asarray(delta_K, float)
    return np.where(dk > law.delta_K_threshold, law.A * np.maximum(dk, 0.0) ** law.m, 0.0)


def integrate_growth(
    a0: np.ndarray,
    stress_range_MPa: float,
    aspect_ratio: float,
    thickness_m: float,
    cycles: float,
    law: ParisLaw,
    n_steps: int = 64,
) -> np.ndarray:
    """Advance crack depth by ``cycles`` under a constant stress range.

    Forward Euler in cycles with a fixed aspect ratio, using the Newman-Raju SIF
    at the deepest point. Growth stops at 95 percent through-wall, beyond which
    the surface-crack solution no longer applies and the problem becomes a
    through-wall one.
    """
    a = np.asarray(a0, float).copy()
    dn = cycles / n_steps
    for _ in range(n_steps):
        c = a / max(aspect_ratio, 1e-9)
        valid = (a / thickness_m) < 0.95
        dk = sif(stress_range_MPa * 1e6, 0.0, a, c, thickness_m) / 1e6  # MPa.m^0.5
        a = np.where(valid, a + growth_rate(dk, law) * dn, a)
    return a
