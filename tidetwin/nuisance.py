"""C3: the nuisance variance budget.

This is the deciding test. The method under examination reads a crack from a
*change* in the strain ratio between two gauges bracketing a K-joint. That only
works if the ratio is otherwise stable. Everything in the ocean that moves the
ratio without a crack being present is a nuisance, and the question is simply
whether their combined standard deviation is small compared with the change a
crack would produce.

Eight nuisance channels are propagated, one at a time and then jointly:

1. **Rotary current direction.** The tidal current traces an ellipse, so the drag
   loads the frame from a continuously changing bearing. The strain ratio at a
   given joint depends on that bearing.
2. **Spring/neap range.** The S2-to-M2 amplitude ratio modulates the current, and
   the drag nonlinearity ``|u|u`` means the response is not a simple rescaling.
3. **Wind-driven residual current** (ERA5 ``u10``/``v10``). A slowly varying drift
   added to the tidal current *before* the nonlinearity.
4. **Water level.** Surge and setup shift the wetted length and the current
   profile.
5. **Marine growth** accreting over 20 years: increased hydrodynamic diameter and
   drag coefficient.
6. **Wave-induced quasi-static offset** (ERA5 ``swh``/``pp1d``), entering as an
   additional near-surface velocity.
7. **Scour** reducing foundation stiffness under the piles.
8. **Differential FBG drift** between the two gauges of the pair.

Channels 5 and 7 change the structure and so need their own response surface;
they are evaluated on a small grid and interpolated. The rest are load-side and
cost a table lookup each.

Verdict rule, applied without softening: if the joint nuisance dispersion exceeds
one third of the C2 damage signature, C3 is ``FAIL`` and the method as specified
does not achieve reliable detection.

One result deserves flagging here rather than being buried. The statistic under
test is a *ratio* of two strain amplitudes, and at a site with a reversing
(near-rectilinear) tidal current its denominator passes close to zero twice a
cycle, when the current slackens. A ratio with a near-zero denominator is
Cauchy-like: **its variance does not exist**, and the sample standard deviation
grows without bound as the Monte Carlo runs longer rather than converging. That
is not a defect of this Monte Carlo, it is a property of the statistic the method
chose. It is detected (:func:`convergence_trace`), reported, and worked around
with a robust scale (:func:`robust_scale`) so a verdict can still be reached.

The practical consequence: nuisance dispersion tracks the *shape* of the tidal
ellipse, not its size. At a strongly rotary site the ratio is well behaved
(standard deviation and robust scale agree to within a couple of percent); at a
reversing site the standard deviation runs to five times the robust scale and the
statistic is pathological.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .geometry.oc4 import SensorPair, build_jacket, load_tables, sensor_pair
from .loads.morison import HydroConfig
from .loads.tides import TidalConstituents, constituent_frequency
from .provenance import Quantity, assumed, derived
from .response import ResponseSurface, build_response_surface
from .rosette import ROSETTE_ANGLES_DEG, axial_drag_ratio
from .signal.harmonic import fit_harmonics

#: The two sensor layouts C3 can be evaluated under. ``"single"`` is the paper's
#: own two-gauge pair; ``"rosette"`` is the four-gauge, direction-and-amplitude
#: invariant layout that scripts/rosette_experiment.py shows takes the nuisance
#: dispersion from 11.6 percent of the ratio to 1.2 percent. The default is the
#: paper's, so the deciding verdict is always computed on the method as specified
#: unless a reader deliberately asks for the proposed instrumentation.
MEASUREMENT_MODES: tuple[str, ...] = ("single", "rosette")

__all__ = [
    "NuisanceRanges",
    "NuisanceResult",
    "CHANNELS",
    "ratio_from_series",
    "run_nuisance_budget",
    "verdict",
    "verdict_against_claimed_signature",
    "VarianceDecomposition",
    "ConvergenceTrace",
    "BreakEven",
    "convergence_trace",
    "variance_decomposition",
]

#: Channel identifiers, in the order the brief lists them.
CHANNELS: tuple[str, ...] = (
    "current_direction",
    "spring_neap",
    "wind_current",
    "water_level",
    "marine_growth",
    "wave_offset",
    "scour",
    "fbg_drift",
)

#: Channels that vary independently between successive records, so coherent
#: averaging reduces them as 1/sqrt(N). Current direction, spring/neap phase,
#: wind drift, surge and sea state all cycle on hours to days.
RANDOM_CHANNELS: frozenset[str] = frozenset(
    {"current_direction", "spring_neap", "wind_current", "water_level", "wave_offset"}
)

#: Channels that evolve over months to years. Averaging longer does not reduce
#: them - it turns them into a trend, which is exactly what a growing crack also
#: looks like. These set the detection floor.
SYSTEMATIC_CHANNELS: frozenset[str] = frozenset({"marine_growth", "scour", "fbg_drift"})

CHANNEL_LABELS: dict[str, str] = {
    "current_direction": "Rotary tidal current direction",
    "spring_neap": "Spring/neap range variation",
    "wind_current": "Wind-driven residual current",
    "water_level": "Water level (surge and setup)",
    "marine_growth": "Marine growth over 20 years",
    "wave_offset": "Wave-induced quasi-static offset",
    "scour": "Scour-driven foundation stiffness",
    "fbg_drift": "Differential FBG drift",
}


@dataclass(frozen=True)
class NuisanceRanges:
    """Sampling ranges for each nuisance channel.

    Every field here is ASSUMED. They are sidebar-editable, they render red, and
    every result computed from them carries an assumption-contaminated flag.
    They are ranges rather than point values precisely because none of them is
    known for a specific platform without a site survey.
    """

    # Ellipse orientation wander and shape, radians / dimensionless.
    direction_bias_sd_deg: float = 15.0
    ellipse_ratio_sd: float = 0.15
    # Spring/neap: multiplicative scale on the whole current, 1 sd.
    spring_neap_sd: float = 0.25
    # Wind-driven surface current, m/s, as a fraction of wind speed (~3%).
    wind_current_sd_ms: float = 0.08
    # Water level offset beyond the astronomical tide, m.
    water_level_sd_m: float = 0.20
    # Marine growth thickness after 20 years, mm, uniform over the range.
    marine_growth_mm: tuple[float, float] = (0.0, 100.0)
    # Equivalent steady near-surface velocity from wave orbital motion, m/s.
    wave_offset_sd_ms: float = 0.10
    # Foundation stiffness retained after scour, as a fraction of the intact value.
    scour_factor_range: tuple[float, float] = (0.15, 1.0)
    # Differential drift between the two gauges over the record, microstrain.
    # Set to the paper's own stated figure of 0.05 per year, not a pessimistic
    # default: testing a claim against worse hardware than it specifies is not a
    # test of the claim.
    fbg_drift_sd_ustrain: float = 0.05
    # Random measurement noise on each gauge, microstrain rms.
    fbg_noise_ustrain: float = 0.05
    #: Correlation between the three storm-driven channels - wind-driven current,
    #: wave-induced offset and water-level surge. They share a cause, so drawing
    #: them independently understates the joint variance. 0.6 is a moderate
    #: positive coupling; it is ASSUMED like everything else here, and the
    #: break-even analysis reports how much the verdict depends on it.
    storm_correlation: float = 0.6

    def scaled(self, k: float) -> "NuisanceRanges":
        """All ranges narrowed (k < 1) or widened (k > 1) by the same factor.

        Used by the break-even analysis. Scaling every channel together is the
        most generous possible reading of "our assumed ranges are too wide":
        it asks how much better the *whole* environment would have to be, not
        just the one channel that happens to dominate.
        """
        k = max(float(k), 0.0)
        lo_scour = 1.0 - (1.0 - self.scour_factor_range[0]) * k
        return NuisanceRanges(
            direction_bias_sd_deg=self.direction_bias_sd_deg * k,
            ellipse_ratio_sd=self.ellipse_ratio_sd * k,
            spring_neap_sd=self.spring_neap_sd * k,
            wind_current_sd_ms=self.wind_current_sd_ms * k,
            water_level_sd_m=self.water_level_sd_m * k,
            marine_growth_mm=(self.marine_growth_mm[0], self.marine_growth_mm[1] * k),
            wave_offset_sd_ms=self.wave_offset_sd_ms * k,
            scour_factor_range=(min(max(lo_scour, 0.0), 1.0), 1.0),
            fbg_drift_sd_ustrain=self.fbg_drift_sd_ustrain * k,
            fbg_noise_ustrain=self.fbg_noise_ustrain * k,
            storm_correlation=self.storm_correlation,
        )

    def as_quantities(self) -> list[Quantity]:
        return [
            assumed(self.direction_bias_sd_deg, "deg", "current direction bias, 1 sd"),
            assumed(self.ellipse_ratio_sd, "-", "ellipse minor/major perturbation, 1 sd"),
            assumed(self.spring_neap_sd, "-", "spring/neap current scale, 1 sd"),
            assumed(self.wind_current_sd_ms, "m/s", "wind-driven residual current, 1 sd"),
            assumed(self.water_level_sd_m, "m", "residual water level, 1 sd"),
            assumed(np.mean(self.marine_growth_mm), "mm", "marine growth after 20 y"),
            assumed(self.wave_offset_sd_ms, "m/s", "wave-induced velocity offset, 1 sd"),
            assumed(np.mean(self.scour_factor_range), "-", "foundation stiffness retained"),
            assumed(self.fbg_drift_sd_ustrain, "ustrain", "differential FBG drift, 1 sd"),
            assumed(self.fbg_noise_ustrain, "ustrain", "FBG noise, rms"),
        ]


@dataclass(frozen=True)
class VarianceDecomposition:
    """How the joint variance relates to the sum of the individual ones.

    If the channels acted independently on a linear response, the joint variance
    would equal the sum of the per-channel variances. It does not, for two
    reasons that pull in opposite directions: the storm-driven channels are
    correlated (raising the joint variance), while the drag nonlinearity and the
    ratio normalisation make some channels partially cancel one another (lowering
    it). The residual is reported rather than assumed away.
    """

    sum_individual_variance: float
    joint_variance: float
    interaction: float

    @property
    def interaction_fraction(self) -> float:
        return (
            float(self.interaction / self.sum_individual_variance)
            if self.sum_individual_variance > 0
            else float("nan")
        )

    @property
    def interpretation(self) -> str:
        f = self.interaction_fraction
        if not np.isfinite(f):
            return "No individual-channel variance to compare against."
        if f < -0.1:
            return (
                f"The joint variance is {abs(f) * 100:.0f} percent below the sum of the "
                "individual ones, so the channels partially cancel: the strain ratio is a "
                "nonlinear function of the forcing and normalising by the second gauge "
                "rejects part of what each channel does alone. Adding channels does not "
                "simply add variance, and quoting a per-channel budget as if it did would "
                "overstate the total."
            )
        if f > 0.1:
            return (
                f"The joint variance is {f * 100:.0f} percent above the sum of the individual "
                "ones. The storm-driven channels share a cause and reinforce each other, so "
                "a budget built by adding independent channels would understate the total."
            )
        return (
            "The joint variance is within 10 percent of the sum of the individual ones, so "
            "the channels behave close to independently over these ranges."
        )


@dataclass(frozen=True)
class ConvergenceTrace:
    """Running estimate of the joint sigma against sample count.

    Distinguishes two very different reasons a run has not settled:

    *Still settling* - sigma wanders and will converge with more samples.

    *Heavy tailed* - sigma **grows** steadily with sample count, which is what
    happens when the underlying distribution has no finite variance. More samples
    will not help, because there is nothing to converge to. The strain ratio is
    especially prone to this: its denominator is the upper gauge's M2 amplitude,
    which approaches zero when a reversing tidal current goes slack, and a ratio
    with a near-zero denominator is Cauchy-like. That is a property of the
    statistic the method under test chose, not of this Monte Carlo.
    """

    n_samples: np.ndarray
    sigma: np.ndarray
    converged: bool
    relative_drift: float
    heavy_tailed: bool = False
    growth_ratio: float = 1.0
    robust: np.ndarray | None = None
    robust_drift: float = float("nan")
    threshold: float = 0.05

    @property
    def verdict(self) -> str:
        if self.converged:
            return (
                f"Converged: sigma moves by {self.relative_drift * 100:.1f} percent across the "
                f"second half of the run, within the {self.threshold * 100:.1f} percent that "
                "the estimator's own sampling error would produce anyway."
            )
        if self.heavy_tailed:
            return (
                f"Sigma does not settle ({self.relative_drift * 100:.0f} percent drift) while a "
                f"robust scale of the same samples does ({self.robust_drift * 100:.1f} percent). "
                "A robust scale converges for any distribution with a density; the standard "
                "deviation converges only if the variance exists. So this is the signature of "
                "a distribution with no finite variance: the strain ratio's denominator passes "
                "near zero when the tidal current slackens, making the ratio Cauchy-like. More "
                "samples will not help. The verdict uses the robust scale."
            )
        return (
            f"NOT converged: sigma is still moving by {self.relative_drift * 100:.1f} percent "
            f"across the second half of the run, against the {self.threshold * 100:.1f} percent "
            "expected from sampling error alone. Increase the sample count before relying on "
            "the verdict."
        )


@dataclass(frozen=True)
class BreakEven:
    """How much smaller every nuisance would have to be for C3 to pass.

    The nuisance ranges are all ASSUMED, so the obvious objection to a FAIL is
    "your ranges are too wide". This answers it quantitatively: every range is
    scaled by a common factor and the joint sigma recomputed, giving the factor
    at which the verdict would flip. A factor of 0.3 means the whole environment
    would have to be three times quieter than assumed, on every channel at once.
    """

    scales: np.ndarray
    sigmas: np.ndarray
    threshold: float
    factor: float
    signature_fraction: float

    @property
    def achievable(self) -> bool:
        return bool(np.isfinite(self.factor) and self.factor >= 1.0)

    @property
    def statement(self) -> str:
        if self.achievable:
            return (
                f"C3 already passes at the assumed ranges (break-even factor "
                f"{self.factor:.2f}x)."
            )
        if not np.isfinite(self.factor):
            return (
                "C3 does not pass at any scaling tried, down to "
                f"{self.scales.min():.2f}x the assumed ranges. The verdict is not an "
                "artefact of the assumed range widths."
            )
        return (
            f"For C3 to pass, every nuisance range would have to shrink to "
            f"{self.factor:.2f}x its assumed width - simultaneously, on all eight channels. "
            "That is the margin by which the method misses, expressed in a form that does "
            "not depend on trusting any single assumed range."
        )


@dataclass
class NuisanceResult:
    """Per-channel and joint standard deviations of the intact strain ratio."""

    baseline_ratio: float
    per_channel_sd: dict[str, float]
    per_channel_samples: dict[str, np.ndarray]
    joint_sd: float
    joint_samples: np.ndarray
    n_samples: int
    record_days: float
    seed: int
    ranges: NuisanceRanges
    gated_channels: dict[str, str] = field(default_factory=dict)
    n_structural_solves: int = 0
    convergence: ConvergenceTrace | None = None
    decomposition: VarianceDecomposition | None = None
    break_even: BreakEven | None = None
    #: Which sensor layout produced this budget: "single" (the paper's two-gauge
    #: pair) or "rosette" (the proposed four-gauge, direction-and-amplitude
    #: invariant layout). Carried so the verdict states which was measured.
    measurement_mode: str = "single"

    @property
    def joint_cv(self) -> float:
        """Joint sigma as a fraction of the baseline ratio."""
        return float(self.joint_sd / abs(self.baseline_ratio)) if self.baseline_ratio else np.inf

    @property
    def joint_robust_sd(self) -> float:
        """Dispersion from the IQR, which stays finite under heavy tails."""
        return robust_scale(self.joint_samples)

    @property
    def heavy_tailed(self) -> bool:
        return bool(self.convergence is not None and self.convergence.heavy_tailed)

    @property
    def effective_sd(self) -> float:
        """The dispersion the verdict should use.

        The standard deviation normally, the robust scale when the variance has
        been shown not to exist. Using a statistic that diverges would make the
        verdict depend on how long the Monte Carlo happened to run.
        """
        if self.heavy_tailed:
            r = self.joint_robust_sd
            if np.isfinite(r):
                return r
        return self.joint_sd

    @property
    def effective_cv(self) -> float:
        return (
            float(self.effective_sd / abs(self.baseline_ratio))
            if self.baseline_ratio
            else np.inf
        )

    @property
    def dispersion_kind(self) -> str:
        return "robust scale (0.7413 x IQR)" if self.heavy_tailed else "standard deviation"

    def false_alarm_fraction(self, signature: float = 0.111) -> float:
        """Share of nuisance-only draws that already look like the claimed damage.

        The most direct statement of the C3 failure there is, and the one that
        needs no statistics to read: with no crack anywhere in the structure,
        this is how often the sea alone moves the strain ratio by at least as
        much as a crack is claimed to move it. Every one of those would be a
        false alarm from a detector set at the claimed signature.

        Counted two-sided, on the magnitude of the change, because a detector
        watching for an 11.1 percent shift has no way to know which way a real
        crack would push this particular ratio - C2 finds the sign depends on
        which joint spring softens.
        """
        if not self.baseline_ratio or self.joint_samples.size == 0:
            return float("nan")
        rel = np.abs(self.joint_samples - self.baseline_ratio) / abs(self.baseline_ratio)
        return float(np.mean(rel >= signature))

    @property
    def joint_sd_standard_error(self) -> float:
        """Monte Carlo standard error on the joint sigma itself.

        ``se(s) ~ s / sqrt(2(n-1))`` for a normal sample. Reported so a reader
        can tell whether a near-threshold verdict is real or sampling noise.
        """
        n = max(self.n_samples, 2)
        return float(self.joint_sd / np.sqrt(2.0 * (n - 1)))

    def dominant_channels(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.per_channel_sd.items(), key=lambda kv: -kv[1])[:k]

    def split_random_systematic(self) -> tuple[float, float]:
        """Nuisance sigma split into averageable and non-averageable parts.

        Combined in quadrature within each group. The systematic part is the
        detection floor: no record length reduces it, so it is what C4's
        time-to-detection ultimately runs into.
        """
        rnd = np.sqrt(
            sum(v**2 for k, v in self.per_channel_sd.items() if k in RANDOM_CHANNELS)
        )
        sysm = np.sqrt(
            sum(v**2 for k, v in self.per_channel_sd.items() if k in SYSTEMATIC_CHANNELS)
        )
        # Rescale so the two reproduce the measured joint sigma rather than the
        # sum of one-at-a-time runs, which ignores interaction.
        total = float(np.hypot(rnd, sysm))
        if total > 0 and self.joint_sd > 0:
            scale = self.joint_sd / total
            return float(rnd * scale), float(sysm * scale)
        return float(rnd), float(sysm)

    def as_quantity(self) -> Quantity:
        return derived(
            self.joint_sd,
            "-",
            "nuisance sigma of the intact M2 strain ratio",
            self.ranges.as_quantities(),
            (
                f"Monte Carlo, {self.n_samples} samples, {self.record_days:.0f} d record, "
                f"seed {self.seed}; {self.n_structural_solves} structural solves"
            ),
            note=(
                "assumption-contaminated: nuisance ranges are ASSUMED. "
                "Dominant channels: "
                + ", ".join(f"{CHANNEL_LABELS[c]} {s:.4f}" for c, s in self.dominant_channels())
            ),
        )


def ratio_from_series(
    times_s: np.ndarray,
    eps_upper: np.ndarray,
    eps_lower: np.ndarray,
    constituent: str = "M2",
) -> float:
    """Strain ratio, defined as the below-joint gauge over the above-joint gauge.

    Taken from the fitted harmonic amplitude at ``constituent`` rather than from
    raw amplitudes, so that broadband noise, drift and other constituents do not
    leak into it. The orientation (lower over upper) is fixed here once and used
    everywhere, so the number is comparable across runs; the reciprocal is
    reported alongside it in the UI so no reader has to guess the convention.
    """
    om = np.array([float(constituent_frequency(constituent).value)])
    fu = fit_harmonics(times_s, eps_upper, (constituent,), om)
    fl = fit_harmonics(times_s, eps_lower, (constituent,), om)
    au = fu.amplitude_of(constituent)
    if au <= 0:
        return float("nan")
    return float(fl.amplitude_of(constituent) / au)


def _structural_grid(
    pair: SensorPair,
    cfg: HydroConfig,
    ljf_model,
    growth_mm: np.ndarray,
    scour_factors: np.ndarray,
    n_theta: int,
) -> tuple[list[list[ResponseSurface]], int]:
    """Response surfaces over the (marine growth, scour) grid."""
    tables = load_tables()
    K_ref = None
    grid: list[list[ResponseSurface]] = []
    solves = 0
    for g in growth_mm:
        row: list[ResponseSurface] = []
        for sf in scour_factors:
            if K_ref is None:
                probe = build_jacket(ljf_model=ljf_model)
                Kp, _ = probe.model.assemble()
                d = Kp.diagonal()
                K_ref = float(np.mean(d[d > 0]))
            found = None if sf >= 0.999 else K_ref * float(sf)
            b = build_jacket(
                ljf_model=ljf_model,
                marine_growth_mm=float(g),
                foundation_stiffness=found,
                tables=tables,
            )
            c = replace(cfg, marine_growth_mm=float(g))
            s = build_response_surface(b, pair, c, n_theta=n_theta)
            solves += s.n_solves
            row.append(s)
        grid.append(row)
    return grid, solves


def _interp_surface_eval(
    grid: list[list[ResponseSurface]],
    growth_axis: np.ndarray,
    scour_axis: np.ndarray,
    growth: float,
    scour: float,
    speed: np.ndarray,
    direction: np.ndarray,
    level: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the response, bilinearly interpolating between grid surfaces."""
    gi = np.clip(np.searchsorted(growth_axis, growth, "right") - 1, 0, len(growth_axis) - 2)
    si = np.clip(np.searchsorted(scour_axis, scour, "right") - 1, 0, len(scour_axis) - 2)
    tg = (growth - growth_axis[gi]) / (growth_axis[gi + 1] - growth_axis[gi])
    ts = (scour - scour_axis[si]) / (scour_axis[si + 1] - scour_axis[si])
    tg = float(np.clip(tg, 0.0, 1.0))
    ts = float(np.clip(ts, 0.0, 1.0))
    out_u = np.zeros_like(speed)
    out_l = np.zeros_like(speed)
    for dg, wg in ((0, 1 - tg), (1, tg)):
        for ds, ws in ((0, 1 - ts), (1, ts)):
            w = wg * ws
            if w == 0.0:
                continue
            u, l = grid[gi + dg][si + ds].evaluate(speed, direction, level)
            out_u += w * u
            out_l += w * l
    return out_u, out_l


def run_nuisance_budget(
    pair: SensorPair,
    constituents: TidalConstituents,
    cfg: HydroConfig,
    ranges: NuisanceRanges = NuisanceRanges(),
    ljf_model=None,
    n_samples: int = 512,
    record_days: float = 14.0,
    sample_interval_s: float = 1800.0,
    n_theta: int = 24,
    seed: int = 20260728,
    era5_available: bool = False,
    run_break_even: bool = True,
    signature_fraction: float = 0.111,
    threshold_fraction: float = 1.0 / 3.0,
    estimator: str = "single",
    progress: Callable[[float, str], None] | None = None,
) -> NuisanceResult:
    """Propagate every nuisance channel through to the intact M2 strain ratio.

    Runs each channel alone, then all channels together. Returns standard
    deviations of the ratio, not of the strain: the ratio is the quantity the
    method actually keys on, and normalising by the second gauge is exactly the
    step that is supposed to reject common-mode nuisance.

    Three further analyses turn the number into a decision:

    * a **convergence trace**, because an unconverged Monte Carlo has decided
      nothing and the claim is withheld if it has not settled;
    * a **variance decomposition**, because the joint sigma is not the sum of the
      per-channel ones and the difference deserves measuring rather than
      hand-waving;
    * a **break-even sweep** (``run_break_even``), which scales every range by a
      common factor and reports the factor at which the verdict would flip. Since
      every range is ASSUMED, this is what makes the verdict robust to them.

    ``signature_fraction`` is the damage signature the break-even target is set
    against, defaulting to the 11.1 percent the abstract asserts. It is an input
    to a comparison, never a computed result.
    """
    from .fe.ljf import LJFModel

    ljf_model = ljf_model or LJFModel.RIGID
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, record_days * 86400.0, sample_interval_s)

    if estimator not in MEASUREMENT_MODES:
        raise ValueError(
            f"estimator must be one of {MEASUREMENT_MODES}, got {estimator!r}. "
            "'single' is the paper's two-gauge pair; 'rosette' is the four-gauge layout."
        )
    growth_axis = np.linspace(ranges.marine_growth_mm[0], ranges.marine_growth_mm[1], 3)
    scour_axis = np.array([ranges.scour_factor_range[0], 1.0])
    if progress:
        progress(0.05, "building structural response surfaces")

    # The single-pair path is unchanged. The rosette path builds one structural
    # grid per gauge angle, on the same joint and offset, and reduces each draw
    # with the direction-and-amplitude-invariant estimator instead of the raw
    # strain ratio. The draw logic below is byte-identical between the two, so
    # the comparison is paired: only the reduction differs.
    if estimator == "single":
        grid, n_solves = _structural_grid(
            pair, cfg, ljf_model, growth_axis, scour_axis, n_theta
        )

        def reduce_draw(d: dict, rg: np.random.Generator | None) -> float:
            eu, el, _eta = _sample_series(
                grid, growth_axis, scour_axis, constituents, t, d, ranges, rg
            )
            return ratio_from_series(t, eu, el)
    else:
        tables = load_tables()
        pairs = [
            sensor_pair(tables, pair.joint_id, pair.offset_m, np.radians(a))
            for a in ROSETTE_ANGLES_DEG
        ]
        grids, n_solves = [], 0
        for p in pairs:
            g, s = _structural_grid(p, cfg, ljf_model, growth_axis, scour_axis, n_theta)
            grids.append(g)
            n_solves += s

        def reduce_draw(d: dict, rg: np.random.Generator | None) -> float:
            U, L, eta = [], [], None
            for g in grids:
                eu, el, eta = _sample_series(
                    g, growth_axis, scour_axis, constituents, t, d, ranges, rg
                )
                U.append(eu)
                L.append(el)
            return axial_drag_ratio(t, U, L, eta)

    gated: dict[str, str] = {}
    if not era5_available:
        gated["wind_current"] = (
            "ERA5 u10/v10 unavailable: the wind-driven residual is sampled from an ASSUMED "
            "range instead of a measured wind series, so this channel's contribution is "
            "indicative only."
        )
        gated["wave_offset"] = (
            "ERA5 swh/pp1d unavailable: the wave-induced offset is sampled from an ASSUMED "
            "range instead of a measured sea state."
        )

    baseline = reduce_draw(_zero_draw(), None)

    def make_draw(rg: np.random.Generator, rn: NuisanceRanges):
        def draw(active: set[str]) -> dict[str, float]:
            d = _zero_draw()
            # Wind-driven current, wave offset and surge share a cause (a storm),
            # so they are drawn from a correlated Gaussian rather than
            # independently. Independent draws would understate the joint sigma,
            # which would flatter the method under test.
            z = _correlated_storm(rg, rn.storm_correlation)
            if "current_direction" in active:
                d["direction_bias"] = np.radians(rg.normal(0.0, rn.direction_bias_sd_deg))
                d["ellipse_ratio"] = rg.normal(0.0, rn.ellipse_ratio_sd)
            if "spring_neap" in active:
                d["current_scale"] = max(0.05, 1.0 + rg.normal(0.0, rn.spring_neap_sd))
            if "wind_current" in active:
                d["wind_u"] = z[0] * rn.wind_current_sd_ms
                d["wind_v"] = rg.normal(0.0, rn.wind_current_sd_ms)
            if "water_level" in active:
                d["water_level"] = z[1] * rn.water_level_sd_m
            if "marine_growth" in active:
                d["growth"] = rg.uniform(*rn.marine_growth_mm)
            if "wave_offset" in active:
                d["wave_u"] = z[2] * rn.wave_offset_sd_ms
                d["wave_v"] = rg.normal(0.0, rn.wave_offset_sd_ms)
            if "scour" in active:
                d["scour"] = rg.uniform(*rn.scour_factor_range)
            if "fbg_drift" in active:
                d["drift"] = rg.normal(0.0, rn.fbg_drift_sd_ustrain) * 1e-6
            d["noise"] = rn.fbg_noise_ustrain * 1e-6
            return d

        return draw

    draw = make_draw(rng, ranges)

    per_channel_sd: dict[str, float] = {}
    per_channel_samples: dict[str, np.ndarray] = {}
    total = len(CHANNELS) + 1

    for k, ch in enumerate(CHANNELS):
        if progress:
            progress(0.1 + 0.8 * k / total, f"channel: {CHANNEL_LABELS[ch]}")
        vals = np.empty(n_samples)
        for i in range(n_samples):
            d = draw({ch})
            vals[i] = reduce_draw(d, rng)
        per_channel_samples[ch] = vals
        per_channel_sd[ch] = float(np.nanstd(vals, ddof=1))

    if progress:
        progress(0.82, "joint propagation")

    def joint_samples_for(rn: NuisanceRanges, n: int, sub_seed: int) -> np.ndarray:
        """Joint propagation at a given set of ranges, reusing the structural grid."""
        rg = np.random.default_rng(sub_seed)
        dr = make_draw(rg, rn)
        out = np.empty(n)
        for i in range(n):
            d = dr(set(CHANNELS))
            out[i] = reduce_draw(d, rg)
        return out

    joint = joint_samples_for(ranges, n_samples, seed + 101)
    joint_sd = float(np.nanstd(joint, ddof=1))

    conv = convergence_trace(joint)
    decomp = variance_decomposition(per_channel_sd, joint_sd)

    break_even = None
    if run_break_even and abs(baseline) > 0 and signature_fraction > 0:
        if progress:
            progress(0.9, "break-even: how much quieter would the sea have to be?")
        target_cv = threshold_fraction * signature_fraction
        scales = np.array([0.1, 0.25, 0.5, 1.0])
        n_be = max(40, n_samples // 2)
        cvs = []
        for k, s in enumerate(scales):
            sd_k = float(
                np.nanstd(joint_samples_for(ranges.scaled(float(s)), n_be, seed + 500 + k), ddof=1)
            )
            cvs.append(sd_k / abs(baseline))
            if progress:
                progress(0.9 + 0.09 * (k + 1) / len(scales), f"break-even {s:.2f}x")
        cvs = np.array(cvs)
        # sigma rises with the scale factor, so interpolate the inverse.
        if np.all(cvs > target_cv):
            factor = float("nan")
        elif np.all(cvs <= target_cv):
            factor = float(scales[-1])
        else:
            order = np.argsort(cvs)
            factor = float(np.interp(target_cv, cvs[order], scales[order]))
        break_even = BreakEven(
            scales=scales,
            sigmas=cvs,
            threshold=target_cv,
            factor=factor,
            signature_fraction=signature_fraction,
        )

    if progress:
        progress(1.0, "done")
    return NuisanceResult(
        baseline_ratio=float(baseline),
        per_channel_sd=per_channel_sd,
        per_channel_samples=per_channel_samples,
        joint_sd=joint_sd,
        joint_samples=joint,
        n_samples=n_samples,
        record_days=record_days,
        seed=seed,
        ranges=ranges,
        gated_channels=gated,
        n_structural_solves=n_solves,
        convergence=conv,
        decomposition=decomp,
        break_even=break_even,
        measurement_mode=estimator,
    )


def _correlated_storm(rng: np.random.Generator, rho: float) -> np.ndarray:
    """Three unit-variance normals with common correlation ``rho``.

    Built as ``z_i = sqrt(rho) * c + sqrt(1-rho) * e_i`` with a shared factor
    ``c``, which gives exactly equicorrelation ``rho`` and unit variance for
    ``0 <= rho <= 1`` without needing a Cholesky factorisation.
    """
    r = float(np.clip(rho, 0.0, 1.0))
    common = rng.normal()
    idio = rng.normal(size=3)
    return np.sqrt(r) * common + np.sqrt(1.0 - r) * idio


def convergence_trace(
    samples: np.ndarray, n_points: int = 24, tolerance: float = 0.05
) -> ConvergenceTrace:
    """Running standard deviation against sample count.

    A deciding test that has not converged is not a decision. This uses the
    joint samples already drawn, so it costs nothing extra, and flags the run as
    unconverged if the estimate is still drifting by more than ``tolerance``
    across the second half of the run.
    """
    x = np.asarray(samples, float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 8:
        return ConvergenceTrace(np.array([x.size]), np.array([np.nan]), False, float("nan"))
    ns = np.unique(np.linspace(8, x.size, n_points).astype(int))

    def _drift(trace: np.ndarray) -> float:
        half = trace[len(trace) // 2 :]
        final = trace[-1]
        return float(np.max(np.abs(half - final)) / final) if final > 0 else float("inf")

    sd = np.array([np.std(x[:n], ddof=1) for n in ns])
    rob = np.array([robust_scale(x[:n]) for n in ns])
    drift = _drift(sd)
    rob_drift = _drift(rob)

    # Convergence has to be judged against the estimator's OWN sampling error,
    # not a fixed percentage. The relative standard error of a sample standard
    # deviation is 1/sqrt(2(n-1)), which at the start of the second half of a
    # 400-sample run is already 5 percent. A flat 5 percent tolerance therefore
    # demands precision below the noise floor and can essentially never be met -
    # which is why runs kept reporting "not converged" no matter how long they
    # ran. The test is now whether the observed drift is consistent with sampling
    # noise, with an allowance for taking a maximum over several trace points,
    # and a floor so that very long runs still have to be genuinely tight.
    n_half = int(ns[len(ns) // 2]) if len(ns) > 1 else int(ns[-1])
    expected = 2.5 / np.sqrt(2.0 * max(n_half - 1, 1))
    threshold = float(max(tolerance, expected))
    converged = bool(drift <= threshold)

    half = sd[len(sd) // 2 :]
    mid = half[0] if half.size else sd[-1]
    growth = float(sd[-1] / mid) if mid > 0 else 1.0

    # Heavy tails are detected by comparing how the two dispersion measures
    # behave, not by thresholding their ratio. A robust scale converges for any
    # distribution with a density; the standard deviation converges only if the
    # variance exists. So "the robust scale has settled and sigma has not" is the
    # signature of a variance that does not exist - and unlike a threshold on
    # sigma/robust, it does not depend on whether one extreme draw happened to
    # land in this particular sample. The final ratio is required to exceed 1.5
    # as well, purely to avoid labelling a run where the two measures agree.
    ratio = float(sd[-1] / rob[-1]) if rob[-1] > 0 and np.isfinite(rob[-1]) else float("inf")
    heavy = bool(not converged and rob_drift <= threshold and ratio > 1.5)
    return ConvergenceTrace(
        ns, sd, converged, drift, heavy, growth, rob, rob_drift, threshold
    )


def robust_scale(samples: np.ndarray) -> float:
    """Normal-consistent dispersion from the interquartile range.

    ``0.7413 * IQR`` equals the standard deviation for a normal sample and stays
    finite for heavy-tailed ones, where the standard deviation does not exist.
    Used for the C3 verdict whenever the variance fails to converge, so a site
    whose strain ratio is Cauchy-like still gets a decision rather than a shrug.
    """
    x = np.asarray(samples, float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan")
    q75, q25 = np.percentile(x, [75, 25])
    return float(0.7413 * (q75 - q25))


def variance_decomposition(
    per_channel_sd: dict[str, float], joint_sd: float
) -> VarianceDecomposition:
    """Joint variance against the sum of the per-channel variances."""
    total = float(sum(v**2 for v in per_channel_sd.values()))
    joint = float(joint_sd**2)
    return VarianceDecomposition(total, joint, joint - total)


def _zero_draw() -> dict[str, float]:
    return {
        "direction_bias": 0.0,
        "ellipse_ratio": 0.0,
        "current_scale": 1.0,
        "wind_u": 0.0,
        "wind_v": 0.0,
        "water_level": 0.0,
        "growth": 0.0,
        "wave_u": 0.0,
        "wave_v": 0.0,
        "scour": 1.0,
        "drift": 0.0,
        "noise": 0.0,
    }


def _sample_series(
    grid,
    growth_axis: np.ndarray,
    scour_axis: np.ndarray,
    constituents: TidalConstituents,
    t: np.ndarray,
    d: dict[str, float],
    ranges: NuisanceRanges,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One realisation of the two strain series under a nuisance draw.

    Returns ``(upper, lower, elevation)``. The elevation is the same series the
    strains were evaluated against, so a rosette estimator can use it as the
    phase reference for its drag projection without recomputing it and risking a
    mismatch.
    """
    con = constituents
    if d["ellipse_ratio"] != 0.0 or d["direction_bias"] != 0.0:
        con = TidalConstituents(
            names=con.names,
            omega=con.omega,
            elev_amp=con.elev_amp,
            elev_phase=con.elev_phase,
            semi_major=con.semi_major,
            semi_minor=con.semi_minor * (1.0 + d["ellipse_ratio"]),
            inclination=con.inclination + d["direction_bias"],
            current_phase=con.current_phase,
            provenance=con.provenance,
            citation=con.citation,
            latitude=con.latitude,
            longitude=con.longitude,
            source_note=con.source_note,
        )

    uv = con.depth_averaged_current(t) * d["current_scale"]
    uv = uv + np.array([d["wind_u"] + d["wave_u"], d["wind_v"] + d["wave_v"]])[None, :]
    eta = con.elevation(t) + d["water_level"]

    speed = np.hypot(uv[:, 0], uv[:, 1])
    direction = np.arctan2(uv[:, 1], uv[:, 0])
    eu, el = _interp_surface_eval(
        grid, growth_axis, scour_axis, d["growth"], d["scour"], speed, direction, eta
    )

    if d["drift"]:
        # Differential drift: a linear ramp on the upper gauge only.
        eu = eu + d["drift"] * (t - t[0]) / max(t[-1] - t[0], 1.0)
    if d["noise"] and rng is not None:
        eu = eu + rng.normal(0.0, d["noise"], size=eu.shape)
        el = el + rng.normal(0.0, d["noise"], size=el.shape)
    return eu, el, eta


def verdict(
    nuisance_sd: float, damage_signature: float, threshold_fraction: float = 1.0 / 3.0
) -> tuple[str, str]:
    """C3 pass/fail against the damage signature.

    ``FAIL`` if the nuisance sigma exceeds ``threshold_fraction`` of the change a
    crack would produce. The wording of the failure message is deliberately
    plain: a method whose noise floor is comparable to its signal does not detect
    anything, and saying so is the purpose of this application.
    """
    if not np.isfinite(damage_signature) or damage_signature <= 0:
        return (
            "UNTESTABLE - DATA MISSING",
            "No damage signature available from C2, so the nuisance floor cannot be "
            "compared against anything. C3 is unresolved, not passed.",
        )
    frac = nuisance_sd / damage_signature
    if frac > threshold_fraction:
        return (
            "FAIL",
            (
                f"Nuisance sigma is {frac * 100:.1f} percent of the damage signature, above the "
                f"{threshold_fraction * 100:.0f} percent limit. The method as specified does not "
                "achieve reliable detection: environmental variation alone moves the strain "
                "ratio by as much as the crack it is meant to find."
            ),
        )
    if frac > 0.5 * threshold_fraction:
        return (
            "MARGINAL",
            (
                f"Nuisance sigma is {frac * 100:.1f} percent of the damage signature. Detection is "
                "possible but the margin is thin, and it rests on nuisance ranges that are "
                "themselves assumptions."
            ),
        )
    return (
        "PASS",
        (
            f"Nuisance sigma is {frac * 100:.1f} percent of the damage signature, within the "
            f"{threshold_fraction * 100:.0f} percent limit."
        ),
    )


def verdict_against_claimed_signature(
    result: NuisanceResult,
    claimed_signature_fraction: float,
    threshold_fraction: float = 1.0 / 3.0,
) -> tuple[str, str]:
    """C3 tested against the *claimed* damage signature rather than the computed one.

    ``claimed_signature_fraction`` is the hypothesis under test - the fractional
    change in the strain ratio that the abstract asserts a crack produces. It is
    an input to a comparison, never a computed result, and the UI labels it as
    such.

    This is the more robust of the two verdicts, and the reason it exists: the
    computed damage signature depends on our line-spring crack model, which is
    documented to under-predict. Testing the nuisance floor against the paper's
    own, more generous, figure removes that dependency. If the method fails even
    when granted its own claimed signal strength, the conclusion does not rest on
    anything this application modelled.
    """
    if not np.isfinite(claimed_signature_fraction) or claimed_signature_fraction <= 0:
        return "UNTESTABLE - DATA MISSING", "No claimed signature supplied to compare against."
    nuisance_fraction = result.effective_cv
    frac = nuisance_fraction / claimed_signature_fraction
    label = "Nuisance dispersion" if result.heavy_tailed else "Nuisance sigma"
    generous = (
        f"{label} is {nuisance_fraction * 100:.2f} percent of the intact ratio "
        f"({result.dispersion_kind}), against a claimed damage signature of "
        f"{claimed_signature_fraction * 100:.2f} percent - a ratio of "
        f"{frac:.2f}, versus the {threshold_fraction:.2f} limit. "
    )
    if result.heavy_tailed:
        generous += (
            "The variance itself does not converge at this site, so a robust scale is used; "
            "note that a statistic whose variance does not exist is a poor basis for a "
            "detection threshold in the first place. "
        )
    if frac > threshold_fraction:
        return "FAIL", generous + (
            "The method fails on the paper's own claimed signal strength, so this verdict "
            "does not depend on how the crack-to-compliance step was modelled here. "
            "Environmental variation alone moves the strain ratio by more than the crack "
            "is claimed to."
        )
    if frac > 0.5 * threshold_fraction:
        return "MARGINAL", generous + "Detection is possible but the margin is thin."
    return "PASS", generous + "Within the limit."
