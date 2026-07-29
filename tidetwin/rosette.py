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
from .signal.harmonic import fit_harmonics

__all__ = [
    "ROSETTE_ANGLES_DEG",
    "RosetteDecomposition",
    "decompose",
    "m2_phasor",
    "axial_ratio",
    "bending_ratio",
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
    f = fit_harmonics(np.asarray(times_s, float), y, (constituent,), om)
    i = f.index(constituent)
    return complex(f.amplitude[i] * np.exp(1j * f.phase[i]))


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
