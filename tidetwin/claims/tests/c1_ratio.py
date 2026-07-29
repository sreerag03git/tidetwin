"""C1 - intact tidal strain ratio at a bracketing sensor pair.

Solves the OC4 frame under the tidal current and elevation loading, fits the M2
harmonic at each gauge, and reports the ratio. Whatever the solver returns is
what is reported.

The ratio is defined once, here, as **below-joint gauge over above-joint gauge**.
The reciprocal is reported alongside it so that a claim quoted in the other
orientation can still be compared without anyone having to guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...loads.tides import constituent_frequency
from ...response import ResponseSurface, strain_series
from ...signal.harmonic import HarmonicFit, fit_harmonics
from ...signal.quadrature import QuadratureSplit, decompose

__all__ = ["RatioResult", "intact_ratio"]


@dataclass
class RatioResult:
    ratio: float
    reciprocal: float
    amplitude_upper: float
    amplitude_lower: float
    ratio_se: float
    fit_upper: HarmonicFit
    fit_lower: HarmonicFit
    split_upper: QuadratureSplit
    split_lower: QuadratureSplit
    times_s: np.ndarray
    eps_upper: np.ndarray
    eps_lower: np.ndarray
    constituent: str

    @property
    def resolution_ustrain(self) -> float:
        """The sensor the paper proposes, not a generic one.

        The abstract specifies 0.1 microstrain. A typical commercial interrogator
        manages about 1, so judging the signal against 1 would be testing a
        sensor ten times worse than the one under discussion - and a finding that
        the signal is "below resolution" means nothing if the resolution assumed
        is not the resolution claimed.
        """
        from ...abstract import PAPER

        return PAPER.fbg_resolution_ustrain

    @property
    def resolution_margin(self) -> float:
        """Weaker gauge's M2 amplitude as a multiple of the sensor resolution.

        Below 1 the signal cannot be read at all. The ratio is between the two
        gauges, so the weaker one governs: a strong upper reading does not rescue
        a lower reading that is under the floor.
        """
        weaker = min(self.amplitude_upper, self.amplitude_lower)
        return float(weaker * 1e6 / self.resolution_ustrain)

    @property
    def below_fbg_resolution(self) -> bool:
        """True if the M2 signal is under the resolution the paper specifies.

        If the tidal signal itself sits below the floor, no amount of processing
        recovers it, and this flag says so before any ratio is quoted.
        """
        return self.resolution_margin < 1.0


def intact_ratio(
    surface: ResponseSurface,
    constituents,
    record_days: float = 30.0,
    sample_interval_s: float = 600.0,
    constituent: str = "M2",
    all_constituents: tuple[str, ...] = ("M2", "S2", "N2", "K1", "O1"),
) -> RatioResult:
    """Compute the intact strain ratio and its supporting decomposition."""
    t = np.arange(0.0, record_days * 86400.0, sample_interval_s)
    eu, el = strain_series(surface, t, constituents)

    names = tuple(n for n in all_constituents if n in constituents.names)
    om = np.array([float(constituent_frequency(n).value) for n in names])
    fu = fit_harmonics(t, eu, names, om)
    fl = fit_harmonics(t, el, names, om)

    au = fu.amplitude_of(constituent)
    al = fl.amplitude_of(constituent)
    ratio = al / au if au > 0 else float("nan")

    # Delta method on a ratio of two independent amplitude estimates.
    su = fu.amplitude_se[fu.index(constituent)]
    sl = fl.amplitude_se[fl.index(constituent)]
    ratio_se = (
        abs(ratio) * float(np.hypot(sl / al if al else np.inf, su / au if au else np.inf))
        if au > 0 and al > 0
        else float("nan")
    )

    eta = constituents.elevation(t)
    return RatioResult(
        ratio=float(ratio),
        reciprocal=float(1.0 / ratio) if ratio else float("nan"),
        amplitude_upper=float(au),
        amplitude_lower=float(al),
        ratio_se=float(ratio_se),
        fit_upper=fu,
        fit_lower=fl,
        split_upper=decompose(t, eu, eta, constituent),
        split_lower=decompose(t, el, eta, constituent),
        times_s=t,
        eps_upper=eu,
        eps_lower=el,
        constituent=constituent,
    )
