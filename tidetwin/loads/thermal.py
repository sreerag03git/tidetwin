"""Differential thermal strain between a bracketing FBG pair.

The claim under test (C5) is that during neap tides, when the M2 current signal
weakens, a thermal channel of 5-15 microstrain becomes usable. The physics is
straightforward and worth stating plainly, because it is also the reason the
thermal channel is a *liability* rather than an asset:

* Steel expands at ``alpha = 12e-6 /K`` (EN 1993-1-1:2005 Section 3.2.6, and
  DNV-RP-C203 uses the same for offshore steels).
* A gauge above water sits at the sol-air temperature: air temperature plus the
  solar rise ``a_s * I / h_o`` (ASHRAE Handbook - Fundamentals, sol-air
  temperature). A gauge below water sits at sea temperature.
* A free member at uniform temperature ``T`` simply expands; it is the
  *difference* between the two gauge locations, and any restraint against it,
  that shows up as differential strain.

The dominant term is diurnal, at exactly 24 h. The solar semidiurnal harmonic
sits at 12.000 h, which is the S2 tidal period to the digit. That is not a
coincidence to be worked around - it is a hard aliasing problem, quantified in
:mod:`tidetwin.signal.aliasing`, and it is why M2 at 12.4206 h has to be the
carrier.

Air, sea and radiation inputs are ERA5 MEASURED quantities. Without them this
module cannot run and C5's amplitude sub-claim is UNTESTABLE - DATA MISSING.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..provenance import Citation, DataUnavailable, Quantity, derived, published

__all__ = [
    "STEEL_ALPHA",
    "ThermalConfig",
    "sol_air_temperature",
    "differential_thermal_strain",
    "free_expansion_strain",
]

EN1993 = Citation(
    document="EN 1993-1-1:2005 Eurocode 3: Design of steel structures, Part 1-1",
    locator="Section 3.2.6 (coefficient of linear thermal expansion for structural steel)",
    year=2005,
)
ASHRAE = Citation(
    document="ASHRAE Handbook - Fundamentals, Nonresidential Cooling and Heating Load Calculations",
    locator="sol-air temperature, T_sa = T_air + a_s I / h_o",
)


def STEEL_ALPHA() -> Quantity:
    """Coefficient of linear thermal expansion of structural steel, 1/K."""
    return published(12.0e-6, "1/K", "steel thermal expansion coefficient", EN1993)


@dataclass(frozen=True)
class ThermalConfig:
    """Surface heat-transfer settings for the above-water gauge.

    ``solar_absorptivity`` 0.7 is representative of weathered/painted steel and
    ``h_outside`` 20 W/m^2K of a moderately windy marine surface; both are
    ASSUMED unless the user substitutes measured values, and both scale the
    solar term linearly, so the app reports the resulting amplitude as a range
    rather than a point value.
    """

    solar_absorptivity: float = 0.7
    h_outside: float = 20.0
    emissivity: float = 0.9


def sol_air_temperature(
    air_temperature_K: np.ndarray,
    solar_irradiance_W_m2: np.ndarray,
    cfg: ThermalConfig = ThermalConfig(),
) -> np.ndarray:
    """Sol-air temperature of a sunlit steel surface, K.

    ``T_sa = T_air + a_s * I / h_o`` (ASHRAE Fundamentals). The long-wave sky
    correction is omitted, which biases the daytime rise slightly high and the
    night-time drop slightly low; both directions are within the spread the
    absorptivity and film-coefficient uncertainty already produce.
    """
    return np.asarray(air_temperature_K, float) + cfg.solar_absorptivity * np.asarray(
        solar_irradiance_W_m2, float
    ) / max(cfg.h_outside, 1e-6)


def ssrd_to_irradiance(ssrd_J_m2: np.ndarray, seconds: float = 3600.0) -> np.ndarray:
    """Convert ERA5 accumulated ``ssrd`` (J/m^2 per hour) to mean W/m^2."""
    return np.asarray(ssrd_J_m2, float) / seconds


def free_expansion_strain(delta_T: np.ndarray) -> np.ndarray:
    """Unrestrained thermal strain ``alpha * dT``."""
    return float(STEEL_ALPHA().value) * np.asarray(delta_T, float)


def differential_thermal_strain(
    air_temperature_K: np.ndarray | None,
    sea_temperature_K: np.ndarray | None,
    solar_irradiance_W_m2: np.ndarray | None,
    restraint_factor: float = 1.0,
    cfg: ThermalConfig = ThermalConfig(),
) -> Quantity:
    """Differential strain between an above-water and a below-water gauge.

    .. math::
        \\Delta\\varepsilon = R\\,\\alpha\\,(T_{sa} - T_{sea})

    ``restraint_factor`` ``R`` is the fraction of free expansion that the frame
    converts into measured strain: 0 for a fully free member (it expands, the
    gauge follows, nothing is read differentially) and 1 for full restraint.
    A real jacket leg sits between the two, and where exactly is a property of
    the frame, not of the thermal environment - so it is exposed as an input and
    the result inherits its uncertainty.

    Raises
    ------
    DataUnavailable
        If any of the three ERA5 inputs is missing.
    """
    missing = [
        n
        for n, v in (
            ("2t", air_temperature_K),
            ("sst", sea_temperature_K),
            ("ssrd", solar_irradiance_W_m2),
        )
        if v is None
    ]
    if missing:
        raise DataUnavailable(
            "Thermal channel (C5)",
            f"ERA5 variable(s) {', '.join(missing)} not available",
            "Configure CDS credentials and fetch 2t, sst and ssrd for the analysis window.",
        )

    t_sa = sol_air_temperature(air_temperature_K, solar_irradiance_W_m2, cfg)
    dT = t_sa - np.asarray(sea_temperature_K, float)
    eps = restraint_factor * float(STEEL_ALPHA().value) * dT
    return derived(
        eps,
        "-",
        "differential thermal strain",
        [STEEL_ALPHA()],
        (
            f"R={restraint_factor:g} x alpha x (sol-air minus sea temperature); "
            f"a_s={cfg.solar_absorptivity:g}, h_o={cfg.h_outside:g} W/m^2K"
        ),
        note=(
            f"dT range {np.nanmin(dT):.2f} to {np.nanmax(dT):.2f} K; "
            "solar term is linear in a_s/h_o, both of which are assumptions"
        ),
    )
