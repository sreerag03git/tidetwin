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

Verdict rule, applied without softening: if the joint nuisance sigma exceeds one
third of the C2 damage signature, C3 is ``FAIL`` and the method as specified does
not achieve reliable detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .geometry.oc4 import SensorPair, build_jacket, load_tables
from .loads.morison import HydroConfig
from .loads.tides import TidalConstituents, constituent_frequency
from .provenance import Provenance, Quantity, assumed, derived
from .response import ResponseSurface, build_response_surface
from .signal.harmonic import fit_harmonics

__all__ = [
    "NuisanceRanges",
    "NuisanceResult",
    "CHANNELS",
    "ratio_from_series",
    "run_nuisance_budget",
    "verdict",
    "verdict_against_claimed_signature",
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
    fbg_drift_sd_ustrain: float = 0.5
    # Random measurement noise on each gauge, microstrain rms.
    fbg_noise_ustrain: float = 0.2

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

    @property
    def joint_cv(self) -> float:
        """Joint sigma as a fraction of the baseline ratio."""
        return float(self.joint_sd / abs(self.baseline_ratio)) if self.baseline_ratio else np.inf

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
                f"Dominant channels: "
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
    progress: Callable[[float, str], None] | None = None,
) -> NuisanceResult:
    """Propagate every nuisance channel through to the intact M2 strain ratio.

    Runs each channel alone, then all channels together. Returns standard
    deviations of the ratio, not of the strain: the ratio is the quantity the
    method actually keys on, and normalising by the second gauge is exactly the
    step that is supposed to reject common-mode nuisance.
    """
    from .fe.ljf import LJFModel

    ljf_model = ljf_model or LJFModel.RIGID
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, record_days * 86400.0, sample_interval_s)

    growth_axis = np.linspace(ranges.marine_growth_mm[0], ranges.marine_growth_mm[1], 3)
    scour_axis = np.array([ranges.scour_factor_range[0], 1.0])
    if progress:
        progress(0.05, "building structural response surfaces")
    grid, n_solves = _structural_grid(pair, cfg, ljf_model, growth_axis, scour_axis, n_theta)

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

    base_u, base_l = _sample_series(
        grid, growth_axis, scour_axis, constituents, t, _zero_draw(), ranges
    )
    baseline = ratio_from_series(t, base_u, base_l)

    def draw(active: set[str]) -> dict[str, float]:
        d = _zero_draw()
        if "current_direction" in active:
            d["direction_bias"] = np.radians(rng.normal(0.0, ranges.direction_bias_sd_deg))
            d["ellipse_ratio"] = rng.normal(0.0, ranges.ellipse_ratio_sd)
        if "spring_neap" in active:
            d["current_scale"] = max(0.05, 1.0 + rng.normal(0.0, ranges.spring_neap_sd))
        if "wind_current" in active:
            d["wind_u"] = rng.normal(0.0, ranges.wind_current_sd_ms)
            d["wind_v"] = rng.normal(0.0, ranges.wind_current_sd_ms)
        if "water_level" in active:
            d["water_level"] = rng.normal(0.0, ranges.water_level_sd_m)
        if "marine_growth" in active:
            d["growth"] = rng.uniform(*ranges.marine_growth_mm)
        if "wave_offset" in active:
            d["wave_u"] = rng.normal(0.0, ranges.wave_offset_sd_ms)
            d["wave_v"] = rng.normal(0.0, ranges.wave_offset_sd_ms)
        if "scour" in active:
            d["scour"] = rng.uniform(*ranges.scour_factor_range)
        if "fbg_drift" in active:
            d["drift"] = rng.normal(0.0, ranges.fbg_drift_sd_ustrain) * 1e-6
        d["noise"] = ranges.fbg_noise_ustrain * 1e-6
        return d

    per_channel_sd: dict[str, float] = {}
    per_channel_samples: dict[str, np.ndarray] = {}
    total = len(CHANNELS) + 1

    for k, ch in enumerate(CHANNELS):
        if progress:
            progress(0.1 + 0.8 * k / total, f"channel: {CHANNEL_LABELS[ch]}")
        vals = np.empty(n_samples)
        for i in range(n_samples):
            d = draw({ch})
            eu, el = _sample_series(grid, growth_axis, scour_axis, constituents, t, d, ranges, rng)
            vals[i] = ratio_from_series(t, eu, el)
        per_channel_samples[ch] = vals
        per_channel_sd[ch] = float(np.nanstd(vals, ddof=1))

    if progress:
        progress(0.9, "joint propagation")
    joint = np.empty(n_samples)
    for i in range(n_samples):
        d = draw(set(CHANNELS))
        eu, el = _sample_series(grid, growth_axis, scour_axis, constituents, t, d, ranges, rng)
        joint[i] = ratio_from_series(t, eu, el)

    if progress:
        progress(1.0, "done")
    return NuisanceResult(
        baseline_ratio=float(baseline),
        per_channel_sd=per_channel_sd,
        per_channel_samples=per_channel_samples,
        joint_sd=float(np.nanstd(joint, ddof=1)),
        joint_samples=joint,
        n_samples=n_samples,
        record_days=record_days,
        seed=seed,
        ranges=ranges,
        gated_channels=gated,
        n_structural_solves=n_solves,
    )


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
) -> tuple[np.ndarray, np.ndarray]:
    """One realisation of the two strain series under a nuisance draw."""
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
    return eu, el


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
    nuisance_fraction = result.joint_cv
    frac = nuisance_fraction / claimed_signature_fraction
    generous = (
        f"Nuisance sigma is {nuisance_fraction * 100:.2f} percent of the intact ratio, against a "
        f"claimed damage signature of {claimed_signature_fraction * 100:.2f} percent - a ratio of "
        f"{frac:.2f}, versus the {threshold_fraction:.2f} limit. "
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
