"""In-phase and quadrature decomposition of the M2 strain response.

Two physically distinct channels drive strain at the M2 frequency, and they are
not in phase with one another:

* **Drag**, driven by the tidal *current*. For a progressive tide the current
  leads the elevation by roughly a quarter cycle.
* **Buoyancy and hydrostatics**, driven by the tidal *elevation* through the
  wetted length of members near the surface. This is in phase with elevation.

Projecting the strain onto the elevation phasor splits the two. That matters
because the strain ratio the method keys on is a ratio of *total* M2 amplitudes,
and the two channels have different ratios between the gauges. Changing the
current strength without changing the tide range - which is what a wind event or
a spring/neap swing does - therefore moves the ratio even with the structure
completely unchanged. The C3 budget shows this as the spring/neap and
wind-current channels, and this module is how the effect is diagnosed rather
than merely observed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..loads.tides import constituent_frequency
from .harmonic import fit_harmonics

__all__ = ["QuadratureSplit", "decompose"]


@dataclass(frozen=True)
class QuadratureSplit:
    """M2 strain resolved into elevation-aligned and elevation-quadrature parts."""

    in_phase: float  # strain amplitude aligned with elevation (buoyancy-like)
    quadrature: float  # strain amplitude 90 deg from elevation (drag-like)
    total: float
    phase_lag: float  # radians, strain relative to elevation
    reference_phase: float

    @property
    def drag_fraction(self) -> float:
        """Share of the M2 strain variance carried by the quadrature channel."""
        t2 = self.total**2
        return float(self.quadrature**2 / t2) if t2 > 0 else float("nan")


def decompose(
    times_s: np.ndarray,
    strain: np.ndarray,
    elevation: np.ndarray,
    constituent: str = "M2",
) -> QuadratureSplit:
    """Split the strain at one constituent relative to the elevation phasor."""
    om = np.array([float(constituent_frequency(constituent).value)])
    fs = fit_harmonics(times_s, np.asarray(strain, float), (constituent,), om)
    fe = fit_harmonics(times_s, np.asarray(elevation, float), (constituent,), om)
    i = fs.index(constituent)
    amp = float(fs.amplitude[i])
    lag = float(fs.phase[i] - fe.phase[i])
    return QuadratureSplit(
        in_phase=float(amp * np.cos(lag)),
        quadrature=float(amp * np.sin(lag)),
        total=amp,
        phase_lag=float(np.arctan2(np.sin(lag), np.cos(lag))),
        reference_phase=float(fe.phase[i]),
    )
