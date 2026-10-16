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

The paper's observable is the **excess** strain ratio Phi = R - 1 (Eq 8), where R
is the bracketing gauge over a remote reference, so the common unit term that
carries no joint information is removed. Its central analytical result (Eq 11) is
that a local stiffness reduction d changes Phi by exactly d/(1-d) - independent of
geometry, gauge length and load amplitude - so a 10 percent reduction gives
+11.1 percent and a 20 percent reduction +25.0 percent. The raw ratio R moves only
eta = 1 - 1/R_0 times as much (about one sixth for the reference clamp), which is
why earlier testing of the raw ratio made the method look an order of magnitude
weaker than it is. :mod:`tidetwin.paper_model` reproduces every number the paper
prints from its own Tables 2 and 3, so the arithmetic is checkable rather than
quoted. The crack-to-stiffness map that sets d (Eq 9) is the paper's own stated,
unvalidated assumption and its central open question - that is where the real
uncertainty lives, not in the d/(1-d) step.
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

    # Joint geometry, the K-joint archetype of Table 2.
    chord_od_m: float = 0.762
    chord_wall_m: float = 0.025
    brace_angle_deg: float = 45.0
    #: Paper Table 2: brace OD 508 mm, giving the stated diameter ratio beta = 0.667.
    #: (Earlier manuscript versions left this unstated; the current paper fixes it.)
    brace_od_m: float = 0.508
    brace_wall_m: float = 0.016

    # Loading.
    spring_tidal_current_ms: float = 0.8

    # Sensing, as specified by the paper.
    fbg_resolution_ustrain: float = 0.1
    fbg_drift_ustrain_per_year: float = 0.05
    gauge_length_m: float = 0.55

    # Damage.
    crack_through_wall_fraction: float = 0.20
    stiffness_reduction: float = 0.10

    # The claims themselves, as the current paper states them.
    #
    # The observable is now the EXCESS strain ratio Phi = R - 1 (Eq 8), where R is
    # the bracketing gauge over the remote reference. The paper's headline change
    # is in Phi, not in the raw ratio R - an earlier confusion that made the method
    # look thirty times weaker than it is. See :mod:`tidetwin.paper_model`.
    intact_excess_ratio: float = 0.200        # Phi_0 (Eq 8), reference clamp
    intact_raw_ratio: float = 1.200           # R_0 = 1 + Phi_0 (Eq 7)
    isolation_factor: float = 0.167           # eta = 1 - 1/R_0 (Eq 11)
    damage_signature: float = 0.111           # d(Phi)/Phi at a 10% stiffness reduction
    damage_signature_d20: float = 0.250       # d(Phi)/Phi at a 20% stiffness reduction
    detection_days_spring: float = 1.4        # SNR 3 at a/T = 10% under spring tides
    detection_days_thermal: float = 1.5       # thermal channel, tide-independent
    thermal_ustrain: float = 10.0             # restrained diurnal thermal strain
    modal_shift_limit: float = 0.005          # stiffness/modal is a lagging indicator
    enkf_convergence_months: float = 13.0     # median, within 8% of true life
    enkf_convergence_error: float = 0.08
    enkf_coverage: float = 0.94               # 90% interval coverage, matched model
    ci_initial_years: float = 2.15            # 90% interval half-width at commissioning
    ci_final_years: float = 0.36              # half-width by month 36 (83% reduction)
    ci_final_month: float = 36.0
    # Economics is now a break-even THRESHOLD, not a saving: on avoided inspection
    # alone condition-based monitoring is a net cost; it breaks even only if it cuts
    # the annual unplanned-intervention probability by about this much.
    breakeven_prob_points: float = 0.5        # percentage points (2.0% -> 1.5%)
    net_cost_avoided_inspection_usd: float = -11.8e6  # fleet, avoided inspection only
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
    "C1": "An excess strain ratio formed from a bracketing long-gauge FBG pair and a "
          "remote reference gauge isolates local joint compliance from the frame-level "
          "response ... independent of the tidal load amplitude and cancels common-mode "
          "sensor drift to first order; the intact excess ratio is 0.200",
    "C2": "The fractional change in that observable is d/(1-d) for a local stiffness "
          "reduction d, independent of joint geometry, clamp gauge length, and load "
          "amplitude ... cracks producing local stiffness reductions of 10% and 20% raise "
          "it by 11.1% and 25.0% respectively. Using the raw ratio instead would recover "
          "only about one-sixth of this sensitivity",
    "C3": "Because both gauges are driven by the same load, the observable is independent "
          "of the tidal load amplitude and cancels common-mode sensor drift to first order; "
          "the dual-signal architecture is necessary for continuous coverage",
    "C4": "Under spring tides a simulated crack at 10% of the wall reaches an SNR of 3 "
          "within about 1.4 days of coherent averaging",
    "C5": "the diurnal thermal signal is required for continuous detection coverage ... the "
          "thermal signal holds detection near 1.5 days irrespective of tidal state",
    "C6": "the estimator ... recovers the remaining life to within 8% after a simulated "
          "median of 13 months ... the 90% interval half-width falling from +/-2.15 years "
          "at commissioning to +/-0.36 years at month 36, an 83% reduction",
    "C7": "any stiffness-based observable is intrinsically a lagging indicator ... the "
          "reason stiffness- and modal-based monitoring has historically underperformed for "
          "crack detection",
    "C8": "The lifecycle-cost analysis is a scenario study, not a projection of realized "
          "savings ... condition-based monitoring is a net cost ... it reaches break-even "
          "only if it reduces the annual unplanned-intervention probability by about 0.5 "
          "percentage points. The break-even threshold, not any monetary saving, is the "
          "intended output",
    "C9": "a crack that the conventional non-updating S-N estimate leaves undetected ... "
          "invisible to vibration-based modal analysis",
}
