"""Efthymiou stress concentration factors for tubular joints.

The Efthymiou parametric equations (Efthymiou, M., "Development of SCF formulae
and generalised influence functions for use in fatigue analysis", OTC Recent
Developments in Tubular Joint Technology, 1988) are reproduced in
DNV-RP-C203 Appendix B, which also carries the worked examples this module's
tests would be verified against.

**The coefficients are not shipped.** DNV-RP-C203 is a paid standard and the
1988 OTC paper is not open. Transcribing SCF equations from memory would produce
stress concentrations that look authoritative and cannot be checked, and an SCF
error propagates to the fifth power through the S-N curve. Supply them from your
own copy in ``data/scf/efthymiou.json``.

No claim in the ledger currently depends on this module: C1 through C5, C7 and
the nuisance budget work on strain ratios and joint compliance, not on hot-spot
stress. It is here so that the fatigue-life route is a transcription away rather
than a rewrite, and so the gap is visible instead of implicit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..fe.ljf import JointGeometry
from ..provenance import Citation, DataUnavailable

__all__ = ["EFTHYMIOU", "SCFSet", "scf_status", "load_coefficients", "scf"]

EFTHYMIOU = Citation(
    document=(
        "Efthymiou, M., 'Development of SCF formulae and generalised influence functions "
        "for use in fatigue analysis', OTC Recent Developments in Tubular Joint Technology; "
        "reproduced in DNV-RP-C203 Appendix B"
    ),
    locator="T/Y, X and K joint SCF equations, chord saddle and crown, brace saddle and crown",
    year=1988,
)

SCHEMA = """{
  "citation": {"document": "DNV-RP-C203", "locator": "Appendix B, Table B-1", "year": 2016},
  "joint_type": "K",
  "load_case": "balanced axial",
  "positions": {
    "chord_saddle": {"C": <float>, "beta": <float>, "gamma": <float>,
                     "tau": <float>, "sin_theta": <float>, "zeta": <float>},
    "chord_crown":  {...}, "brace_saddle": {...}, "brace_crown": {...}
  },
  "validity": {"beta": [0.2, 1.0], "gamma": [8, 32], "tau": [0.2, 1.0],
               "theta_deg": [30, 90]}
}

Each position evaluates as
    SCF = C * beta^b * gamma^g * tau^t * sin(theta)^s * zeta^z
which is the multiplicative form the Efthymiou equations take for the simple
load cases. Cases with additive terms need the full expression; extend this
loader rather than forcing them into the product form."""


@dataclass(frozen=True)
class SCFSet:
    """Stress concentration factors around a joint, one per hot-spot position."""

    chord_saddle: float
    chord_crown: float
    brace_saddle: float
    brace_crown: float
    joint_type: str
    load_case: str
    citation: Citation
    warnings: tuple[str, ...] = ()

    @property
    def governing(self) -> float:
        """The largest SCF, which governs the fatigue life at the joint."""
        return float(
            max(self.chord_saddle, self.chord_crown, self.brace_saddle, self.brace_crown)
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "chord saddle": self.chord_saddle,
            "chord crown": self.chord_crown,
            "brace saddle": self.brace_saddle,
            "brace crown": self.brace_crown,
        }


def scf_status(root: Path | None = None) -> tuple[bool, str]:
    d = root or Path(__file__).resolve().parents[2] / "data" / "scf"
    p = d / "efthymiou.json"
    if p.is_file():
        return True, f"Efthymiou coefficients loaded from {p}."
    return False, (
        "DATA UNAVAILABLE - Efthymiou SCF coefficients are not shipped. They are in "
        "DNV-RP-C203 Appendix B (paid standard). An SCF error propagates to the fifth "
        f"power through the S-N curve, so these are not reproduced from memory. Create {p} "
        f"with the schema:\n{SCHEMA}"
    )


def load_coefficients(root: Path | None = None) -> dict:
    ok, why = scf_status(root)
    if not ok:
        raise DataUnavailable(
            "Efthymiou SCF coefficients",
            why,
            "Transcribe DNV-RP-C203 Appendix B into data/scf/efthymiou.json.",
        )
    d = root or Path(__file__).resolve().parents[2] / "data" / "scf"
    return json.loads((d / "efthymiou.json").read_text(encoding="utf-8"))


def scf(g: JointGeometry, zeta: float = 0.3, root: Path | None = None) -> SCFSet:
    """Stress concentration factors for a joint geometry.

    ``zeta`` is the gap parameter ``g/D`` for K joints. Raises
    :class:`~tidetwin.provenance.DataUnavailable` until the coefficients are
    supplied.
    """
    spec = load_coefficients(root)
    pos = spec["positions"]

    def evaluate(block: dict) -> float:
        return float(
            block["C"]
            * g.beta ** block.get("beta", 0.0)
            * g.gamma ** block.get("gamma", 0.0)
            * g.tau ** block.get("tau", 0.0)
            * np.sin(g.theta) ** block.get("sin_theta", 0.0)
            * max(zeta, 1e-6) ** block.get("zeta", 0.0)
        )

    warnings = tuple(g.validity())
    c = spec.get("citation", {})
    return SCFSet(
        chord_saddle=evaluate(pos["chord_saddle"]),
        chord_crown=evaluate(pos["chord_crown"]),
        brace_saddle=evaluate(pos["brace_saddle"]),
        brace_crown=evaluate(pos["brace_crown"]),
        joint_type=str(spec.get("joint_type", "?")),
        load_case=str(spec.get("load_case", "?")),
        citation=Citation(
            document=c.get("document", EFTHYMIOU.document),
            locator=c.get("locator", EFTHYMIOU.locator),
            year=c.get("year"),
        ),
        warnings=warnings,
    )
