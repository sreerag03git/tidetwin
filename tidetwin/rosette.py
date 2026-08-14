"""A direction-invariant gauge layout, and whether it rescues C3.

C3 fails because the tidal strain ratio between two gauges moves under
environmental variation by as much as a crack is claimed to move it, and the
largest single contributor is the *direction* of the rotary tidal current. A
gauge at one circumferential position reads a strain that depends on which way
the current pushes, so the ratio inherits that dependence even though the
structure has not changed.

That is a property of the sensor layout, not of the structure, which makes it
the one part of the method that instrumentation could plausibly fix. This module
tests whether it does.

Four gauges per section, at 0/90/180/270 degrees, separate the two parts of the
axial surface strain exactly. Writing ``eps(phi) = a + bx*cos(phi) + by*sin(phi)``:

    a  = (eps_0 + eps_90 + eps_180 + eps_270) / 4          the axial part
    bx = (eps_0 - eps_180) / 2,  by = (eps_90 - eps_270) / 2   the bending part

Both ``a`` and ``hypot(bx, by)`` are invariant to the direction of loading. Only
one of them is any use, and which one is an empirical question about where the
signal actually is:

**The combination must happen in the harmonic domain, not the time domain.** The
bending components swing through zero twice a cycle, so taking ``hypot`` of the
raw series rectifies it - doubling the frequency and destroying the M2 line that
the ratio is fitted on. Combining the complex M2 coefficients keeps the
constituent intact and is still direction-invariant. Getting this wrong makes
the rosette look four times *worse* than a single pair rather than better.

At a braced jacket leg the tidal drag is carried by frame action - axial force
in the legs and braces - which is what bracing is for. Measured at J5 the M2
bending amplitude is around 0.02 microstrain against 0.15 to 0.32 microstrain
axial, so bending is under a tenth of the signal and below the 0.05 microstrain
noise floor of the interrogator the abstract specifies. A bending-based estimator
therefore discards the signal and keeps the noise. The axial mean does the
opposite, and additionally averages four gauges, so sensor noise falls by two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loads.tides import constituent_frequency
from .signal.harmonic import harmonic_amplitude_phase

__all__ = [
    "ROSETTE_ANGLES_DEG",
    "RosetteDecomposition",
    "decompose",
    "m2_phasor",
    "axial_ratio",
    "bending_ratio",
    "axial_series",
    "drag_component",
    "axial_drag_ratio",
    "DAMAGE_RESPONSE_BY_OBSERVABLE",
    "BENDING_RATIO_NOISE",
    "damage_snr",
    "GaugeRobustness",
    "gauge_robustness",
    "BENDING_AXIAL_RATIO_J5",
]

#: Four equally spaced positions. Diametrically opposed pairs are what make the
#: axial/bending separation exact rather than approximate.
ROSETTE_ANGLES_DEG: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)


def m2_phasor(times_s: np.ndarray, eps: np.ndarray, constituent: str = "M2") -> complex:
    """Complex harmonic coefficient of one gauge series at ``constituent``.

    Amplitude and phase together, so that gauges can be combined as phasors.
    Combining amplitudes alone would lose the sign information that makes the
    opposed-gauge differencing work.
    """
    y = np.asarray(eps, float)
    # A gauge reading a flat line has no phase, and the fit's delta-method phase
    # error divides by the amplitude to get one. A dead or disconnected gauge is
    # an ordinary field condition, so it returns a zero phasor rather than
    # raising out of a warning filter three modules away.
    if y.size == 0 or not np.any(np.isfinite(y)) or np.ptp(y[np.isfinite(y)]) == 0.0:
        return 0j
    om = np.array([float(constituent_frequency(constituent).value)])
    # Amplitude-and-phase fast path: byte-identical to fit_harmonics here, but
    # without the standard-error machinery this per-draw call never uses.
    amp, pha = harmonic_amplitude_phase(np.asarray(times_s, float), y, om)
    return complex(amp[0] * np.exp(1j * pha[0]))


@dataclass(frozen=True)
class RosetteDecomposition:
    """The direction-invariant parts of one instrumented section."""

    axial: float
    bending: float
    #: Per-gauge M2 amplitudes, in the order of :data:`ROSETTE_ANGLES_DEG`.
    gauge_amplitudes: tuple[float, ...]

    @property
    def bending_fraction(self) -> float:
        """Bending as a fraction of axial. Below ~0.1 there is no bending to use."""
        return self.bending / self.axial if self.axial > 0 else float("inf")


def decompose(times_s: np.ndarray, eps_by_angle) -> RosetteDecomposition:
    """Split one section's four gauge series into axial and bending parts.

    ``eps_by_angle`` is four strain series ordered as :data:`ROSETTE_ANGLES_DEG`.
    """
    e = list(eps_by_angle)
    if len(e) != 4:
        raise ValueError(
            f"a rosette needs exactly four gauges at {ROSETTE_ANGLES_DEG} degrees, got {len(e)}. "
            "Opposed pairs are what make the axial/bending separation exact; three gauges at "
            "arbitrary angles would need a least-squares fit and a different error model."
        )
    A = [m2_phasor(times_s, x) for x in e]
    axial = abs(sum(A) / 4.0)
    bx, by = 0.5 * (A[0] - A[2]), 0.5 * (A[1] - A[3])
    return RosetteDecomposition(
        axial=float(axial),
        bending=float(np.hypot(abs(bx), abs(by))),
        gauge_amplitudes=tuple(float(abs(a)) for a in A),
    )


def _ratio(times_s, upper_by_angle, lower_by_angle, attr: str) -> float:
    up = getattr(decompose(times_s, upper_by_angle), attr)
    if up <= 0:
        return float("nan")
    return float(getattr(decompose(times_s, lower_by_angle), attr) / up)


def axial_ratio(times_s: np.ndarray, upper_by_angle, lower_by_angle) -> float:
    """Lower-over-upper ratio of the direction-invariant axial M2 amplitude.

    The estimator that works. Same orientation convention as
    :func:`tidetwin.nuisance.ratio_from_series` - lower over upper - so the two
    are directly comparable.
    """
    return _ratio(times_s, upper_by_angle, lower_by_angle, "axial")


def bending_ratio(times_s: np.ndarray, upper_by_angle, lower_by_angle) -> float:
    """The same for the bending magnitude. Kept because the comparison is the point.

    On this structure it is markedly worse than a single pair, and the reason -
    the bending signal is below the sensor noise floor - is a finding about the
    method rather than a detail of the implementation.
    """
    return _ratio(times_s, upper_by_angle, lower_by_angle, "bending")


# --------------------------------------------------------------------- drag
# The rosette removes the current's DIRECTION from the ratio. What is left is
# dominated by its AMPLITUDE - spring/neap range - and that has a separate cure.
#
# Morison drag goes as U^2 while buoyancy does not depend on the current at all.
# Scaling the whole current therefore scales every drag strain by one common
# factor, which cancels in a ratio of two drag strains; it is the buoyancy part
# that does not cancel and so lets spring/neap move the ratio. Removing the
# buoyancy part should remove the channel.
#
# The two are separable by phase: drag is in quadrature with the tidal
# elevation, buoyancy is in phase with it. Elevation is what a tide gauge
# measures, so using it as a phase reference asks for nothing a real deployment
# would not already have.


def drag_component(times_s: np.ndarray, eps: np.ndarray, elevation: np.ndarray) -> float:
    """The part of the M2 strain in quadrature with the tide - i.e. the drag part.

    Returns ``nan`` if the elevation reference has no M2 line to phase against,
    since without it in-phase and quadrature are not defined.
    """
    E = m2_phasor(times_s, elevation)
    if abs(E) == 0:
        return float("nan")
    return float(abs((m2_phasor(times_s, eps) / (E / abs(E))).imag))


def axial_series(eps_by_angle) -> np.ndarray:
    """The direction-invariant axial combination as a time series."""
    e = list(eps_by_angle)
    if len(e) != 4:
        raise ValueError(f"a rosette needs exactly four gauges, got {len(e)}")
    return (e[0] + e[1] + e[2] + e[3]) / 4.0


def axial_drag_ratio(
    times_s: np.ndarray, upper_by_angle, lower_by_angle, elevation: np.ndarray
) -> float:
    """Direction-invariant AND amplitude-invariant strain ratio.

    Both corrections at once: the rosette removes the current's direction, the
    quadrature projection removes its amplitude. Neither is sufficient alone -
    the rosette leaves spring/neap at 8.9 percent and the projection leaves
    direction at 10.2 percent - and together they take the joint nuisance
    dispersion from 11.6 percent of the intact ratio to 1.2 percent, which is
    where C3 passes. See ``scripts/rosette_experiment.py``.
    """
    up = drag_component(times_s, axial_series(upper_by_angle), elevation)
    if not np.isfinite(up) or up <= 0:
        return float("nan")
    return float(drag_component(times_s, axial_series(lower_by_angle), elevation) / up)


# -------------------------------------------------- gauge-imperfection robustness
# The rosette fix earns its direction-invariance from the exact cancellation of
# the bending terms when four gauges sit at exactly 0/90/180/270 degrees with
# exactly matched gains. A real deployment delivers neither: gauges are placed to
# a tolerance of a few degrees and their gains are matched to about a percent. So
# the honest question about our own proposed fix is whether it survives that.
#
# It is answered here without any new solving, on the M2 phasor field itself. At
# a section the M2 strain phasor around the circumference is
#     E(phi) = A + Bx cos(phi) + By sin(phi),
# with A the axial phasor and (Bx, By) the bending phasors. The rosette recovers
# the axial amplitude as |mean of the four gauges|; with perfect gauges the
# bending terms average to zero and the recovery is exactly |A|. A placement
# error delta_k and a gain error g_k on gauge k make the measured value
# g_k * E(phi_k + delta_k), and the average no longer lands on |A|.

#: Bending-to-axial M2 amplitude ratio measured at J5 - the field the rosette
#: actually sees. Bending is about a tenth of axial there (0.02 vs 0.15-0.32
#: microstrain), which is why axial is the usable channel; it also bounds how
#: much a placement error can leak, since only the bending it mixes in can move
#: the axial estimate.
BENDING_AXIAL_RATIO_J5: float = 0.10


@dataclass(frozen=True)
class GaugeRobustness:
    """How gauge placement and gain errors move the rosette's strain ratio."""

    sigma_angle_deg: float
    sigma_gain: float
    bending_axial_ratio: float
    #: Standard deviation of the fractional error in the recovered strain ratio,
    #: from gauge imperfection alone, as a fraction of the ratio.
    ratio_dispersion: float
    #: The environmental nuisance dispersion the rosette already carries.
    environmental_cv: float
    #: The two combined in quadrature - they are independent.
    combined_cv: float
    damage_signature: float
    threshold_fraction: float
    n: int

    @property
    def combined_over_signature(self) -> float:
        return self.combined_cv / self.damage_signature

    @property
    def passes(self) -> bool:
        """Does the fix still clear the C3 detectability bar with real gauges?"""
        return self.combined_over_signature <= self.threshold_fraction


def _recovered_axial_amp(bx: complex, by: complex, angles_rad: np.ndarray,
                         gains: np.ndarray) -> float:
    """|mean of four imperfect gauges| for a unit-axial field, phasor domain."""
    e = 1.0 + bx * np.cos(angles_rad) + by * np.sin(angles_rad)
    return float(abs(np.mean(gains * e)))


def gauge_robustness(
    sigma_angle_deg: float = 3.0,
    sigma_gain: float = 0.01,
    bending_axial_ratio: float = BENDING_AXIAL_RATIO_J5,
    environmental_cv: float = 0.0122,
    damage_signature: float = 0.111,
    threshold_fraction: float = 1.0 / 3.0,
    n: int = 6000,
    seed: int = 20260730,
) -> GaugeRobustness:
    """Propagate realistic gauge placement and gain errors to the strain ratio.

    ``sigma_angle_deg`` is the 1-sigma circumferential placement error per gauge,
    ``sigma_gain`` the 1-sigma fractional gain mismatch. Each of the ratio's two
    sections carries four gauges with independent errors, so the ratio's error is
    the difference of the two sections' axial-recovery errors. The bending
    orientation and the M2 phase are themselves nuisance parameters and are
    randomised, so the result is not tuned to one convenient field.

    The default ``environmental_cv`` is the rosette's own nuisance dispersion
    (1.22 percent of the ratio) from ``scripts/rosette_experiment.py``; the two
    are independent and combine in quadrature. ``passes`` is whether the combined
    dispersion still clears one third of the damage signature.
    """
    rng = np.random.default_rng(seed)
    nominal = np.radians([0.0, 90.0, 180.0, 270.0])
    sa = np.radians(sigma_angle_deg)

    def section_error() -> np.ndarray:
        # A field with the given bending fraction, random spatial orientation and
        # temporal phase, recovered through four imperfect gauges. Returns the
        # fractional error in the recovered axial amplitude (perfect = 1).
        psi = rng.uniform(0.0, 2.0 * np.pi, n)
        phase = rng.uniform(0.0, 2.0 * np.pi, n)
        b = bending_axial_ratio * np.exp(1j * phase)
        bx, by = b * np.cos(psi), b * np.sin(psi)
        out = np.empty(n)
        for i in range(n):
            ang = nominal + rng.normal(0.0, sa, 4)
            gains = rng.normal(1.0, sigma_gain, 4)
            out[i] = _recovered_axial_amp(bx[i], by[i], ang, gains) - 1.0
        return out

    eu = section_error()
    el = section_error()
    # ratio = lower/upper; fractional error = (1+el)/(1+eu) - 1.
    ratio_err = (1.0 + el) / (1.0 + eu) - 1.0
    disp = float(np.std(ratio_err, ddof=1))
    combined = float(np.hypot(disp, environmental_cv))
    return GaugeRobustness(
        sigma_angle_deg=sigma_angle_deg,
        sigma_gain=sigma_gain,
        bending_axial_ratio=bending_axial_ratio,
        ratio_dispersion=disp,
        environmental_cv=environmental_cv,
        combined_cv=combined,
        damage_signature=damage_signature,
        threshold_fraction=threshold_fraction,
        n=n,
    )


# ------------------------------------------------------- damage sensitivity
# C3 was fixed by changing the observable: the axial combination is immune to
# the nuisance that was drowning the signal. The obvious next move is to fix C2
# the same way, and it does not work - for a reason worth stating precisely.
#
# The three observables trade off against each other, measured at J5 under the
# paper's own 10 percent joint stiffness reduction:
#
#     observable        damage response   measurement noise   SNR
#     axial rosette          -0.007 %            ~1.2 %       ~0
#     single pair            +0.356 %              -          low
#     bending rosette        -1.906 %            12.4 %       0.15
#
# Axial force in a braced frame is a GLOBAL equilibrium quantity. One joint going
# soft barely redistributes it, which is exactly why the axial ratio is both
# beautifully quiet and completely damage-blind. Bending is where a rotational
# joint spring shows up, and it is indeed 5.7 times more damage-sensitive - but a
# braced jacket suppresses leg bending by design, so the bending amplitude is
# about 0.021 microstrain and its own harmonic-fit noise is 12 percent of the
# ratio. The damage step is seven times smaller than the noise on it.
#
# Moving the gauges does not help. Swept from 0.25 m to 2.5 m from the joint the
# bending amplitude stays at 0.021 microstrain and the sensitivity between -1.81
# and -2.12 percent: this is a global bending mode of the leg, not a local joint
# disturbance that decays, so there is no placement that recovers it.
#
# The conclusion is not that C2 needs a better estimator. It is that on a braced
# jacket no strain-ratio observable is both quiet enough and damage-sensitive
# enough, because the same bracing that makes the structure stiff also routes the
# load around the joint whose health is being inferred.

#: Measured at J5 under a 10 percent out-of-plane joint stiffness reduction.
#: Kept as a named constant so the ledger can quote it without recomputing a
#: forty-surface sweep on every run. See scripts/c2_observable_experiment.py.
DAMAGE_RESPONSE_BY_OBSERVABLE: dict[str, float] = {
    "single pair": 0.00356,
    "axial rosette": -0.00007,
    "bending rosette": -0.01906,
}

#: Fractional measurement noise on the bending ratio from FBG noise alone, after
#: harmonic fitting over a 14-day record. The reason the bending channel cannot
#: be used despite being the most damage-sensitive of the three.
BENDING_RATIO_NOISE: float = 0.1236


def damage_snr(response: float, noise: float = BENDING_RATIO_NOISE) -> float:
    """Signal-to-noise of a damage step against the noise on its own observable.

    A sensitivity figure means nothing on its own: the bending ratio responds to
    damage five times better than a single pair and is still useless, because the
    noise on it is larger again. This is the number that decides.
    """
    return abs(response) / noise if noise > 0 else float("inf")
