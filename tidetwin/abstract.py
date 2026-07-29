"""The paper's own parameters, quoted verbatim and used as the test conditions.

Testing a claim against conditions the author did not specify is not a test of
the claim. This module holds every number the abstract states about its own
setup, so the analysis runs on the paper's terms rather than on defaults chosen
here.

Two of these were previously set far more pessimistically than the paper
specifies, which made the analysis unfair to it:

* FBG resolution was 1.0 microstrain against the stated **0.1**;
* FBG drift was 5 microstrain per year against the stated **below 0.05**.

A finding that a signal is "below sensor resolution" means nothing if the
assumed sensor is ten times worse than the one proposed. Both are corrected.

The abstract also supplies a decisive intermediate quantity. It does not merely
claim a crack changes the ratio; it says a 20 percent through-wall crack produces
a **10 percent joint stiffness reduction**, and that this takes the ratio from
1.800 to 2.000. The second step is pure structural mechanics and can be tested
exactly, with no crack model involved - see
:func:`tidetwin.claims.tests.c2_damage.stiffness_reduction_test`. That makes C2
testable without the shell-FE surface, which is the single most valuable thing
the abstract provides.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fe.ljf import JointGeometry
from .provenance import Citation
import numpy as np

__all__ = [
    "PAPER",
    "LOWER_ZAKUM_JOINT",
    "PaperParameters",
    "ABSTRACT_CITATION",
    "CLAIM_TEXT",
]

ABSTRACT_CITATION = Citation(
    document=(
        "Probabilistic Fatigue Digital Twin for Offshore Jackets: EnKF with Tidal "
        "Calibration Signal (abstract under test)"
    ),
    locator="Results, Observations, Conclusions",
)


@dataclass(frozen=True)
class PaperParameters:
    """Every quantity the abstract states about its own configuration."""

    # Joint geometry, Lower Zakum K-joint.
    chord_od_m: float = 0.762
    chord_wall_m: float = 0.025
    brace_angle_deg: float = 45.0
    #: Not stated in the abstract. A brace-to-chord diameter ratio of 0.6 is
    #: mid-range for a jacket K-joint and is flagged ASSUMED wherever it is used.
    brace_od_m: float = 0.457
    brace_wall_m: float = 0.016

    # Loading.
    spring_tidal_current_ms: float = 0.8

    # Sensing, as specified by the paper.
    fbg_resolution_ustrain: float = 0.1
    fbg_drift_ustrain_per_year: float = 0.05

    # Damage.
    crack_through_wall_fraction: float = 0.20
    stiffness_reduction: float = 0.10

    # The claims themselves.
    intact_ratio: float = 1.800
    damaged_ratio: float = 2.000
    damage_signature: float = 0.111
    detection_days: tuple[float, float] = (4.0, 9.0)
    thermal_ustrain: tuple[float, float] = (5.0, 15.0)
    modal_shift_limit: float = 0.005
    enkf_convergence_months: float = 18.0
    enkf_convergence_error: float = 0.08
    ci_initial_years: float = 3.2
    ci_final_years: float = 0.9
    ci_final_month: float = 36.0
    npv_usd: float = 19.9e6
    n_jackets: int = 30

    @property
    def gamma(self) -> float:
        """Chord slenderness D/2T, the parameter LJF depends on most strongly."""
        return self.chord_od_m / (2.0 * self.chord_wall_m)

    def joint(self) -> JointGeometry:
        """The abstract's K-joint as a JointGeometry."""
        return JointGeometry(
            chord_D=self.chord_od_m,
            chord_T=self.chord_wall_m,
            brace_d=self.brace_od_m,
            brace_t=self.brace_wall_m,
            theta=np.radians(self.brace_angle_deg),
        )


PAPER = PaperParameters()
LOWER_ZAKUM_JOINT = PAPER.joint()


#: Verbatim claim text, so the ledger quotes the paper rather than paraphrasing it.
CLAIM_TEXT: dict[str, str] = {
    "C1": "elastic frame analysis gives intact tidal strain ratio = 1.800 at bracketing "
          "sensor pairs",
    "C2": "A 20% through-wall crack producing 10% stiffness reduction ... increases this "
          "ratio to 2.000, an 11.1% change",
    "C3": "the dual-signal architecture is necessary, not optional - the tidal ratio must "
          "remain stable enough under environmental variation for an 11.1% change to be "
          "detectable",
    "C4": "Detection threshold at SNR of 3 is achieved within 4 to 9 days under spring "
          "tides",
    "C5": "During neap periods the diurnal thermal signal at 5 to 15 ustrain maintains "
          "detection coverage",
    "C6": "the log-EnKF converges to within 8% of true damage within 18 months, narrowing "
          "the 90% confidence interval from +/-3.2 to +/-0.9 years by month 36",
    "C7": "global natural frequencies shift by less than 0.5%",
    "C8": "For 30 ADNOC jackets, condition-based scheduling saves an estimated $19.9 "
          "million net over 20 years",
    "C9": "cross-validated localised damage detection invisible to vibration-based modal "
          "analysis",
}
