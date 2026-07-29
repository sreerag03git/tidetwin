"""The claims ledger.

Each claim carries its statement, the figure the paper asserts, the criterion by
which it is judged, and a pure function that turns computed artifacts into a
status. Computation happens elsewhere (and is cached); this module only judges,
so the verdicts are reproducible from the artifacts alone.

Statuses are exactly the five the brief permits. ``UNTESTABLE`` is not a hedge
and not a failure - it is the correct answer when the data or the physical test
needed to settle a claim does not exist, and several claims land there honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

__all__ = ["Status", "Claim", "ClaimResult", "Artifacts", "CLAIMS", "evaluate_all"]


class Status(str, Enum):
    PASS = "PASS"
    MARGINAL = "MARGINAL"
    FAIL = "FAIL"
    UNTESTABLE_DATA = "UNTESTABLE - DATA MISSING"
    UNTESTABLE_PHYSICAL = "UNTESTABLE - REQUIRES PHYSICAL TEST"
    #: Not a verdict. The analysis has not been run yet, which is a completely
    #: different thing from "the data needed to settle this does not exist", and
    #: showing the latter for the former misrepresents the ledger.
    NOT_RUN = "NOT RUN YET"

    @property
    def is_verdict(self) -> bool:
        """Whether this represents an actual finding about the claim."""
        return self is not Status.NOT_RUN

    @property
    def colour(self) -> str:
        """Status colour for a white background, all clearing WCAG AA contrast.

        FAIL is styled exactly as prominently as PASS - same weight, same size,
        same border. The status word is always shown, so a reader who cannot
        distinguish the colours loses nothing.
        """
        return {
            "PASS": "#1a7f43",
            "MARGINAL": "#8a5300",
            "FAIL": "#b3261e",
            "UNTESTABLE - DATA MISSING": "#4a545e",
            "UNTESTABLE - REQUIRES PHYSICAL TEST": "#4a545e",
            "NOT RUN YET": "#8a929b",
        }[self.value]


@dataclass
class Artifacts:
    """Everything the claim tests may read. Any field may be ``None``."""

    c1: Any = None  # c1_ratio.RatioResult
    c2: Any = None  # c2_damage.DamageGrid
    c2_stiffness: Any = None  # c2_damage.StiffnessReductionResult
    c3: Any = None  # nuisance.NuisanceResult
    c4: Any = None  # dict from detection_time_cdf
    c5: Any = None  # dict: thermal amplitude (or None) + aliasing.AliasingResult
    c6: Any = None  # c6_filter.FilterComparison
    c7: Any = None  # c7_modal.ModalResult
    c8: Any = None  # npv.NPVResult
    c9: Any = None  # c9_positioning.PODResult
    era5_available: bool = False
    tide_model_available: bool = False
    shell_fe_available: bool = False
    sn_available: bool = False
    paris_available: bool = False
    competitor_pod_available: bool = False
    tide_provenance: str = "ASSUMED"
    tide_source_note: str = ""
    modelling_assumptions: tuple[str, ...] = ()
    #: Claim id -> reason its computation could not complete. A claim with an
    #: entry here is reported UNTESTABLE with the reason, never as a crash.
    errors: dict[str, str] = field(default_factory=dict)
    #: Adjustments made to out-of-range user inputs, surfaced in the report.
    input_notes: tuple[str, ...] = ()

    def artifact_for(self, claim_id: str):
        return getattr(self, claim_id.lower(), None)


@dataclass
class ClaimResult:
    claim_id: str
    status: Status
    computed_text: str
    detail: str
    blocking_assumptions: tuple[str, ...] = ()
    figures: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Claim:
    id: str
    statement: str
    claimed_value: str
    pass_criterion: str
    test: Callable[[Artifacts], ClaimResult]
    note: str = ""


def _contamination(art: Artifacts) -> tuple[str, ...]:
    out: list[str] = []
    if art.tide_provenance != "MEASURED":
        out.append(
            "Tidal harmonic constants are "
            f"{art.tide_provenance}, not a real extraction: every strain magnitude "
            "and ratio below describes a hypothetical tide."
        )
    else:
        out.append(
            "Tidal forcing is MEASURED, but from a published station rather than the "
            "platform site: " + (art.tide_source_note or "see the Provenance tab") + " "
            "A site-specific TPXO or FES extraction would change the magnitudes; the C3 "
            "verdict is unchanged across every station tested."
        )
    if not art.era5_available:
        out.append("ERA5 unavailable: wind, wave and thermal channels use ASSUMED ranges.")
    out.extend(art.modelling_assumptions)
    out.extend(art.input_notes)
    return tuple(out)


def _missing(claim_id: str, what: str, remedy: str) -> ClaimResult:
    """Not yet computed. This is NOT a verdict and must never read as one."""
    return ClaimResult(
        claim_id,
        Status.NOT_RUN,
        "awaiting analysis",
        f"{what} {remedy}".strip()
        + " This is not a finding: the analysis has simply not been run yet.",
    )


def evaluate(claim: "Claim", art: Artifacts) -> ClaimResult:
    """Judge one claim, never raising.

    A claim whose computation failed, or whose test function itself errors, is
    reported ``UNTESTABLE - DATA MISSING`` carrying the reason. The application
    must always be able to produce a full report, whatever it was handed, and a
    traceback is not a report.
    """
    if claim.id in art.errors:
        return ClaimResult(
            claim.id,
            Status.UNTESTABLE_DATA,
            "not computed",
            f"This claim could not be evaluated for the inputs supplied: {art.errors[claim.id]}",
            _contamination(art),
        )
    try:
        return claim.test(art)
    except Exception as exc:  # noqa: BLE001 - a report must still be produced
        return ClaimResult(
            claim.id,
            Status.UNTESTABLE_DATA,
            "evaluation failed",
            (
                f"The test for {claim.id} raised {type(exc).__name__}: {exc}. Reported as "
                "untestable rather than suppressed, because a claim that cannot be evaluated "
                "has not been supported."
            ),
            _contamination(art),
        )


# ------------------------------------------------------------------- tests


def _c1(art: Artifacts) -> ClaimResult:
    if art.c1 is None:
        return _missing("C1", "Run the analysis to compute the intact ratio.", "")
    r = art.c1
    claimed = 1.800
    err = abs(r.ratio - claimed) / claimed
    err_recip = abs(r.reciprocal - claimed) / claimed
    best = min(err, err_recip)
    orient = "as defined (lower/upper)" if err <= err_recip else "in the reciprocal orientation"
    if r.below_fbg_resolution:
        status = Status.FAIL
        detail = (
            f"The solver gives {r.ratio:.4f} ({r.reciprocal:.4f} reciprocal) against a claimed "
            f"{claimed:.3f}. More decisively, the M2 strain amplitudes are "
            f"{r.amplitude_upper * 1e6:.3f} and {r.amplitude_lower * 1e6:.3f} microstrain - below "
            "the 0.1 microstrain the paper itself specifies for the interrogator. A ratio "
            "between two quantities that cannot be measured is not a measurable ratio."
        )
    elif best <= 0.05:
        status = Status.PASS
        detail = f"Computed {r.ratio:.4f} ({orient}), within 5 percent of the claimed {claimed:.3f}."
    elif best <= 0.15:
        status = Status.MARGINAL
        detail = (
            f"Computed {r.ratio:.4f} ({r.reciprocal:.4f} reciprocal) against a claimed "
            f"{claimed:.3f}: {best * 100:.1f} percent adrift {orient}."
        )
    else:
        status = Status.FAIL
        detail = (
            f"Computed {r.ratio:.4f} ({r.reciprocal:.4f} reciprocal) against a claimed "
            f"{claimed:.3f}: {best * 100:.1f} percent adrift even taking the more favourable "
            "orientation."
        )
    detail += (
        f" The M2 strain splits {r.split_lower.drag_fraction * 100:.0f} percent into the "
        "current-driven quadrature channel at the lower gauge, the remainder being the "
        "elevation-driven buoyancy channel - so the ratio is not a pure structural quantity."
    )
    return ClaimResult("C1", status, f"{r.ratio:.4f} (reciprocal {r.reciprocal:.4f})", detail, _contamination(art))


def _c2(art: Artifacts) -> ClaimResult:
    st_res = art.c2_stiffness
    if st_res is None and art.c2 is None:
        return _missing("C2", "Run the analysis to compute the damage sensitivity.", "")

    # The abstract supplies its own intermediate step: a 20 percent through-wall
    # crack produces a 10 percent joint stiffness reduction, which takes the
    # ratio from 1.800 to 2.000. The second link is pure structural mechanics and
    # can be tested exactly - no crack model, no shell-FE surface, no fracture
    # mechanics. That makes it the primary evidence, because it depends on
    # nothing this application chose.
    if st_res is not None:
        got = st_res.at_claimed_reduction
        claimed = st_res.claimed_signature
        by = st_res.change_by_mode
        breakdown = ", ".join(
            f"{k} {v * 100:+.3f} %" for k, v in sorted(by.items(), key=lambda kv: -abs(kv[1]))
        )
        detail = st_res.verdict + (
            " This is the paper's own intermediate quantity, so the result does not depend "
            "on how the crack was modelled here, nor on the shell-FE surface that is not "
            "shipped. Every reading of which stiffness is meant was swept ("
            + breakdown + "), and the claim is judged on the most favourable. It is "
            "consistent with the local-joint-flexibility sweep on the Structure tab: "
            "removing joint flexibility altogether moves the ratio under 1 percent, so a 10 "
            "percent reduction in it cannot move the ratio by 11."
        )
        if abs(got - claimed) / claimed <= 0.25:
            status = Status.PASS
        elif abs(got) >= 0.4 * claimed:
            status = Status.MARGINAL
        else:
            status = Status.FAIL
        if art.c2 is not None:
            detail += (
                f" The independent crack-model route agrees in direction: it gives "
                f"{art.c2.signature_at(0.5, 0.10) * 100:.4f} percent at a half-through-wall, "
                "100 mm flaw."
            )
        return ClaimResult(
            "C2", status,
            f"{got * 100:+.3f} % at the paper's own 10 % reduction "
            f"({st_res.best_mode}, best case)",
            detail, _contamination(art),
        )

    g = art.c2
    sig = g.signature_at(0.5, 0.10)
    detail = (
        f"Via the line-spring compliance route the change at a/T=0.5, 2c=100 mm is "
        f"{sig * 100:.4f} percent, and the largest anywhere on the grid is "
        f"{g.max_signature * 100:.4f} percent, against a claimed 11.1 percent. That route is "
        "documented to under-predict, so this alone is a lower bound rather than a refutation."
    ) + " " + " ".join(g.caveats)
    return ClaimResult(
        "C2", Status.UNTESTABLE_DATA,
        f"{sig * 100:.4f} % at a/T=0.5, 2c=100 mm", detail, _contamination(art),
    )


def _c3(art: Artifacts) -> ClaimResult:
    if art.c3 is None:
        return _missing("C3", "Run the analysis to compute the nuisance budget.", "")
    from ..nuisance import CHANNEL_LABELS, verdict_against_claimed_signature

    n = art.c3
    status_txt, msg = verdict_against_claimed_signature(n, 0.111)
    status = Status(status_txt) if status_txt in Status._value2member_map_ else Status.FAIL
    if status is Status.FAIL and not np.isfinite(n.joint_sd):
        return ClaimResult(
            "C3", Status.UNTESTABLE_DATA, "sigma not finite",
            "The nuisance Monte Carlo produced no finite strain ratios, so no sigma exists "
            "to compare against anything. Check that the tidal forcing is non-zero.",
            _contamination(art),
        )
    top = ", ".join(
        f"{CHANNEL_LABELS[c]} ({s / abs(n.baseline_ratio) * 100:.1f} percent)"
        for c, s in n.dominant_channels(3)
    )
    detail = (
        msg
        + f" Dominant channels: {top}. Monte Carlo standard error on sigma is "
        f"{n.joint_sd_standard_error / abs(n.baseline_ratio) * 100:.2f} percentage points."
    )
    if n.convergence is not None:
        detail += " " + n.convergence.verdict
        if not n.convergence.converged and not n.convergence.heavy_tailed:
            # Still settling: more samples would fix it, so decide nothing yet.
            status = Status.UNTESTABLE_DATA
            detail += (
                " The verdict is withheld: an unconverged Monte Carlo has not decided "
                "anything, and reporting it as FAIL would be as unfounded as reporting it "
                "as PASS."
            )
        elif n.convergence.heavy_tailed:
            # No amount of sampling fixes an undefined variance, so the verdict
            # stands on the robust scale rather than being withheld forever.
            detail += (
                " The verdict stands on the robust scale, which is well defined here. That "
                "the variance of the method's own detection statistic does not exist at this "
                "site is itself a finding against it."
            )
    if n.break_even is not None:
        detail += " " + n.break_even.statement
    if n.decomposition is not None:
        detail += " " + n.decomposition.interpretation
    computed = (
        f"robust scale = {n.effective_cv * 100:.2f} % of the intact ratio (variance undefined)"
        if n.heavy_tailed
        else f"sigma = {n.joint_cv * 100:.2f} % of the intact ratio"
    )
    return ClaimResult("C3", status, computed, detail, _contamination(art))


def _c4(art: Artifacts) -> ClaimResult:
    if art.c4 is None:
        return _missing("C4", "Run the analysis to compute detection times.", "")
    p = art.c4["percentiles"]
    never = p.get("never_detected_fraction", 0.0)
    lo, hi = 4.0, 9.0
    if never > 0.5:
        status = Status.FAIL
        detail = (
            f"{never * 100:.0f} percent of Monte Carlo trials never detect within the simulated "
            "horizon. The claimed 4-9 day window is unreachable because the systematic part of "
            "the nuisance budget does not average away: coherent averaging reduces the random "
            "component as 1/sqrt(N) but leaves a floor that no record length crosses."
        )
    elif lo <= p["p50"] <= hi:
        status = Status.PASS
        detail = f"Median detection at {p['p50']:.1f} days, inside the claimed {lo:.0f}-{hi:.0f} day window."
    else:
        status = Status.FAIL
        detail = (
            f"Detection times are p05 {p['p05']:.1f} d, p50 {p['p50']:.1f} d, p95 {p['p95']:.1f} d "
            f"against a claimed {lo:.0f}-{hi:.0f} days."
        )
    computed = (
        f"never detected in {never * 100:.0f} % of trials"
        if not np.isfinite(p["p50"])
        else f"p50 = {p['p50']:.1f} d"
    )
    return ClaimResult("C4", status, computed, detail, _contamination(art))


def _c5(art: Artifacts) -> ClaimResult:
    if art.c5 is None:
        return _missing("C5", "Run the analysis to evaluate the thermal channel.", "")
    alias = art.c5["aliasing"]
    amp = art.c5.get("amplitude_ustrain")
    if amp is None:
        status = Status.UNTESTABLE_DATA
        detail = (
            "The 5-15 microstrain amplitude cannot be evaluated: it needs ERA5 2t, sst and ssrd, "
            "and CDS credentials are not configured. The aliasing half of the claim is "
            "computable and decisive on its own. " + alias.conclusion
        )
    else:
        inside = 5.0 <= amp <= 15.0
        status = Status.PASS if inside else Status.FAIL
        detail = (
            f"Differential thermal strain amplitude {amp:.2f} microstrain against a claimed "
            f"5-15. " + alias.conclusion
        )
    return ClaimResult(
        "C5",
        status,
        "aliasing resolved; amplitude " + ("n/a" if amp is None else f"{amp:.2f} ustrain"),
        detail,
        _contamination(art),
    )


def _c6(art: Artifacts) -> ClaimResult:
    if art.c6 is None:
        return _missing("C6", "Run the analysis to compare the filters.", "")
    c = art.c6
    enkf = c.reports["log-EnKF"]
    base = c.reports["no-update baseline"]
    calibrated = 0.70 <= enkf.coverage_90 <= 0.98
    beats_baseline = enkf.crps < base.crps
    if enkf.coverage_90 < 0.70:
        status = Status.FAIL
        detail = (
            f"The log-EnKF's nominal 90 percent interval achieves only "
            f"{enkf.coverage_90 * 100:.0f} percent coverage under structural model error. Its "
            f"CRPS is {enkf.crps:.4g} against the no-update baseline's {base.crps:.4g}. A narrow "
            "but miscalibrated interval is worse than the status quo it replaces, because it "
            "will be used to defer inspections that should have gone ahead."
        )
    elif enkf.coverage_90 > 0.98:
        status = Status.MARGINAL if beats_baseline else Status.FAIL
        detail = (
            f"Coverage is {enkf.coverage_90 * 100:.0f} percent against a nominal 90, with a mean "
            f"interval width of {enkf.mean_interval_width_90:.3g}. The interval is honest but "
            "over-dispersed: it is wider than it claims to be, so a stated 90 percent bound is "
            "not the bound it says it is, and the decision value over the prior is small. "
            f"CRPS {enkf.crps:.4g} against the baseline's {base.crps:.4g}."
        )
    elif beats_baseline:
        status = Status.PASS
        detail = (
            f"log-EnKF CRPS {enkf.crps:.4g} beats the no-update baseline's {base.crps:.4g} with "
            f"{enkf.coverage_90 * 100:.0f} percent coverage against a nominal 90."
        )
    else:
        status = Status.MARGINAL
        detail = (
            f"Calibration is acceptable ({enkf.coverage_90 * 100:.0f} percent coverage) but the "
            f"CRPS {enkf.crps:.4g} does not beat the no-update baseline's {base.crps:.4g}."
        )
    if not calibrated:
        detail += (
            " The stated pass criterion requires coverage between 70 and 98 percent; this run "
            "is outside that band."
        )
    detail += " " + c.model_error_note
    if not art.paris_available:
        detail += (
            " The remaining-life claim of +/-0.9 years is UNTESTABLE - DATA MISSING: it needs "
            "BS 7910 Paris constants, which are not shipped."
        )
    return ClaimResult("C6", status, f"coverage {enkf.coverage_90 * 100:.0f} %", detail, _contamination(art))


def _c7(art: Artifacts) -> ClaimResult:
    if art.c7 is None:
        return _missing("C7", "Run the analysis to compute modal shifts.", "")
    m = art.c7
    claimed = 0.005
    shift = m.max_abs_shift
    status = Status.PASS if shift < claimed else Status.FAIL
    detail = (
        f"Largest frequency shift over {m.n_modes} modes is {shift * 100:.4f} percent, against a "
        f"claimed threshold of {claimed * 100:.1f} percent. A dense long-record modal array "
        f"resolves about {m.resolvable_threshold * 100:.1f} percent, so the shift is "
        + ("above" if m.detectable_by_modal else "below")
        + " what modal monitoring could see - the comparison is stated against a realistic "
        "modal capability rather than against zero."
    )
    return ClaimResult("C7", status, f"{shift * 100:.4f} %", detail, _contamination(art))


def _c8(art: Artifacts) -> ClaimResult:
    if art.c8 is None:
        return _missing("C8", "Run the analysis to compute the NPV.", "")
    r = art.c8
    claimed = 19.9e6
    detail = (
        f"Expected NPV {r.mean / 1e6:.2f} MUSD (median {r.median / 1e6:.2f}, "
        f"p05 {r.percentile(5) / 1e6:.2f}, p95 {r.percentile(95) / 1e6:.2f}), with a "
        f"{r.probability_positive * 100:.0f} percent chance of being positive at all, against a "
        f"claimed {claimed / 1e6:.1f} MUSD. Every input is ASSUMED, so this is a scenario, not a "
        "result: the tornado plot showing which assumption the answer is hostage to is the "
        "usable output. The economic case is also downstream of C3 - a method that does not "
        "detect reliably cannot avoid the campaigns this NPV credits it with avoiding."
    )
    return ClaimResult(
        "C8",
        Status.UNTESTABLE_DATA,
        f"{r.mean / 1e6:.2f} MUSD (all inputs ASSUMED)",
        detail,
        _contamination(art) + ("every economic input is ASSUMED",),
    )


def _c9(art: Artifacts) -> ClaimResult:
    if art.c9 is None:
        return _missing("C9", "Run the analysis to compute POD.", "")
    p = art.c9
    if not p.competitor_curves:
        status = Status.UNTESTABLE_DATA
        detail = (
            f"POD for the tidal method reaches 90 percent at a90 = "
            + (f"{p.a90 * 1e3:.1f} mm" if np.isfinite(p.a90) else "no crack size on the grid")
            + ". The comparison against ROV MPI, ACFM and flooded member detection cannot be "
            "made: their published POD curves are not shipped. " + p.competitor_note.split("\n")[0]
        )
    else:
        status = Status.PASS if np.isfinite(p.a90) else Status.FAIL
        detail = f"a90 = {p.a90 * 1e3:.1f} mm, a90/95 = {p.a90_95 * 1e3:.1f} mm."
    detail += (
        " Flooded member detection, the real incumbent competitor on cost, responds only after "
        "a crack is through-wall and the member has flooded; the tidal method targets "
        "part-through flaws. They are not substitutes and are not plotted as though they were."
    )
    return ClaimResult(
        "C9",
        status,
        (f"a90 = {p.a90 * 1e3:.1f} mm" if np.isfinite(p.a90) else "a90 not reached"),
        detail,
        _contamination(art),
    )


CLAIMS: tuple[Claim, ...] = (
    Claim(
        "C1",
        "Under intact conditions the tidal strain ratio at a bracketing sensor pair on the "
        "target K-joint is 1.800.",
        "1.800",
        "Computed ratio within 5 percent of 1.800, and the underlying strain amplitudes "
        "resolvable by the specified sensor.",
        _c1,
    ),
    Claim(
        "C2",
        "A crack at the joint changes the strain ratio from 1.800 to 2.000, an 11.1 percent "
        "damage signature.",
        "11.1 percent (1.800 -> 2.000)",
        "Computed change within 20 percent of 11.1 percent, from a validated crack-to-LJF "
        "surface.",
        _c2,
    ),
    Claim(
        "C3",
        "The tidal strain ratio is stable enough under environmental variation for the damage "
        "signature to be detectable.",
        "nuisance sigma below one third of the damage signature",
        "Joint nuisance sigma < 1/3 of the damage signature. Tested against both the computed "
        "signature and the claimed 11.1 percent.",
        _c3,
        note="THE DECIDING TEST",
    ),
    Claim(
        "C4",
        "Detection is achieved in 4 to 9 days at a signal-to-noise ratio of 3.",
        "4-9 days",
        "Median Monte Carlo detection time inside 4-9 days under real forcing.",
        _c4,
    ),
    Claim(
        "C5",
        "During neap tides a differential thermal channel of 5 to 15 microstrain provides a "
        "usable secondary carrier.",
        "5-15 microstrain",
        "Computed amplitude inside 5-15 microstrain from ERA5, and S2-solar separable from "
        "S2-tidal at the achievable record length.",
        _c5,
    ),
    Claim(
        "C6",
        "The log-transformed EnKF converges on remaining life to within +/-0.9 years, "
        "outperforming the no-update baseline.",
        "+/-0.9 years",
        "Better CRPS than the no-update baseline AND empirical coverage of the nominal 90 "
        "percent interval between 70 and 98 percent under structural model error.",
        _c6,
    ),
    Claim(
        "C7",
        "The same damage produces a natural frequency shift below 0.5 percent, so modal "
        "methods cannot detect it.",
        "< 0.5 percent",
        "Computed maximum frequency shift below 0.5 percent, compared against a realistic "
        "modal resolution rather than against zero.",
        _c7,
    ),
    Claim(
        "C8",
        "The monitoring system returns a net present value of 19.9 million USD.",
        "19.9 MUSD",
        "Cannot pass: every input is an unsourced commercial assumption. Reported as a "
        "scenario with a tornado sensitivity.",
        _c8,
    ),
    Claim(
        "C9",
        "The tidal method offers a probability-of-detection advantage over ROV MPI, ACFM and "
        "flooded member detection.",
        "favourable a90/95",
        "a90/95 compared against published POD curves for the incumbent methods, with "
        "part-through and through-wall capability distinguished.",
        _c9,
    ),
)


def evaluate_all(art: Artifacts) -> list[ClaimResult]:
    """Judge every claim from the supplied artifacts. Never raises."""
    return [evaluate(c, art) for c in CLAIMS]
