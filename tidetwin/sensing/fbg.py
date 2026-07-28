"""Fibre Bragg grating sensing model.

An FBG reads a wavelength shift that responds to both strain and temperature:

.. math:: \\frac{\\Delta\\lambda}{\\lambda} = (1 - p_e)\\,\\varepsilon
                                            + (\\alpha_f + \\xi)\\,\\Delta T

with ``p_e`` the effective photoelastic constant (about 0.22 for germanosilicate
fibre), ``alpha_f`` the fibre expansion coefficient and ``xi`` the thermo-optic
coefficient. The temperature term is not small: for a typical grating the
apparent strain from a 1 K change is around 8-10 microstrain. Since C1's whole
signal at the OC4 K-joint comes out at a few tenths of a microstrain, thermal
cross-sensitivity is not a correction to the measurement - it is larger than the
measurement, and the method stands or falls on rejecting it.

Coefficients here are order-of-magnitude values for standard telecom-grade
gratings and are ASSUMED unless the user substitutes calibration data for the
specific gratings deployed. They are exposed in the sidebar for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..provenance import Quantity, assumed

__all__ = ["FBGSpec", "apparent_strain_from_temperature", "simulate_pair"]


@dataclass(frozen=True)
class FBGSpec:
    """Grating and interrogator characteristics.

    ``resolution_ustrain`` is the interrogator's strain resolution; 1 microstrain
    is typical of a mid-range commercial unit and 0.1 of a high-end one.
    ``drift_ustrain_per_year`` covers grating relaxation and clamp creep, which
    for a bonded or welded offshore installation is the dominant long-term error
    and the one that most resembles a slowly growing crack.
    """

    resolution_ustrain: float = 1.0
    noise_ustrain_rms: float = 0.5
    drift_ustrain_per_year: float = 5.0
    photoelastic_constant: float = 0.22
    thermo_optic_per_K: float = 6.5e-6
    fibre_expansion_per_K: float = 0.55e-6
    steel_expansion_per_K: float = 12.0e-6

    def as_quantities(self) -> list[Quantity]:
        return [
            assumed(self.resolution_ustrain, "ustrain", "FBG interrogator resolution"),
            assumed(self.noise_ustrain_rms, "ustrain", "FBG noise, rms"),
            assumed(self.drift_ustrain_per_year, "ustrain/yr", "FBG drift and clamp creep"),
            assumed(self.photoelastic_constant, "-", "effective photoelastic constant"),
            assumed(self.thermo_optic_per_K, "1/K", "thermo-optic coefficient"),
        ]

    @property
    def temperature_sensitivity_ustrain_per_K(self) -> float:
        """Apparent strain per kelvin for a grating bonded to steel.

        The grating cannot distinguish real strain from the combined thermo-optic
        and differential-expansion response, so a temperature change appears as

        .. math::
            \\varepsilon_{app} = \\frac{\\alpha_f + \\xi}{1 - p_e}
                               + \\alpha_{steel}

        the second term being the real thermal strain of the substrate that the
        grating faithfully follows.
        """
        optical = (self.fibre_expansion_per_K + self.thermo_optic_per_K) / (
            1.0 - self.photoelastic_constant
        )
        return float((optical + self.steel_expansion_per_K) * 1e6)


def apparent_strain_from_temperature(
    delta_T: np.ndarray, spec: FBGSpec = FBGSpec()
) -> np.ndarray:
    """Apparent strain, in strain units, from an uncompensated temperature change."""
    return np.asarray(delta_T, float) * spec.temperature_sensitivity_ustrain_per_K * 1e-6


def simulate_pair(
    strain_upper: np.ndarray,
    strain_lower: np.ndarray,
    times_s: np.ndarray,
    spec: FBGSpec = FBGSpec(),
    delta_T_upper: np.ndarray | None = None,
    delta_T_lower: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add readout noise, quantisation, drift and thermal cross-talk to a pair.

    Drift is applied with opposite sign to the two gauges, which is the worst
    case for a ratio statistic and the one worth designing against: common-mode
    drift divides out, differential drift does not.
    """
    rng = rng or np.random.default_rng(0)
    t = np.asarray(times_s, float)
    years = (t - t[0]) / (365.25 * 86400.0)
    out = []
    for k, eps in enumerate((np.asarray(strain_upper, float), np.asarray(strain_lower, float))):
        y = eps.copy()
        dT = (delta_T_upper, delta_T_lower)[k]
        if dT is not None:
            y = y + apparent_strain_from_temperature(dT, spec)
        y = y + np.sign(0.5 - k) * spec.drift_ustrain_per_year * 1e-6 * years
        y = y + rng.normal(0.0, spec.noise_ustrain_rms * 1e-6, size=y.shape)
        if spec.resolution_ustrain > 0:
            q = spec.resolution_ustrain * 1e-6
            y = np.round(y / q) * q
        out.append(y)
    return out[0], out[1]
