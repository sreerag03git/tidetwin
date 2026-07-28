"""C5 - the thermal channel during neaps, and why S2 cannot carry it.

The claim splits into two halves with very different testability:

*Amplitude* (5 to 15 microstrain) needs ERA5 ``2t``, ``sst`` and ``ssrd`` at the
platform. Without CDS credentials it is ``UNTESTABLE - DATA MISSING``.

*Separability* is pure arithmetic and decides the claim on its own. Solar
heating's semidiurnal harmonic sits at exactly 12.000 h. So does S2. Their
frequency difference is zero, so no record length separates them, and any
S2-based amplitude is an unresolvable sum of tide and sunshine. M2 at 12.4206 h
is 14.77 days from S2 in Rayleigh terms and carries no solar harmonic, which is
why it has to be the carrier.

Worth stating plainly: the thermal channel is a *liability* for this method
rather than a spare carrier. The FBG cross-sensitivity in
:mod:`tidetwin.sensing.fbg` puts roughly 8 to 10 microstrain of apparent strain
on a 1 K change, against a tidal signal that C1 computes in tenths of a
microstrain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...loads.thermal import ThermalConfig, differential_thermal_strain, ssrd_to_irradiance
from ...provenance import DataUnavailable
from ...sensing.fbg import FBGSpec
from ...signal.aliasing import AliasingResult, aliasing_table

__all__ = ["ThermalResult", "thermal_channel", "CLAIMED_RANGE_USTRAIN"]

CLAIMED_RANGE_USTRAIN = (5.0, 15.0)


@dataclass
class ThermalResult:
    aliasing: AliasingResult
    amplitude_ustrain: float | None
    unavailable_reason: str
    cross_sensitivity_ustrain_per_K: float

    @property
    def inside_claimed_range(self) -> bool | None:
        if self.amplitude_ustrain is None:
            return None
        lo, hi = CLAIMED_RANGE_USTRAIN
        return bool(lo <= self.amplitude_ustrain <= hi)

    def as_artifact(self) -> dict:
        return {"aliasing": self.aliasing, "amplitude_ustrain": self.amplitude_ustrain}


def thermal_channel(
    record_days: float,
    air_temperature_K: np.ndarray | None = None,
    sea_temperature_K: np.ndarray | None = None,
    ssrd_J_m2: np.ndarray | None = None,
    restraint_factor: float = 0.3,
    cfg: ThermalConfig = ThermalConfig(),
    fbg: FBGSpec = FBGSpec(),
) -> ThermalResult:
    """Evaluate both halves of C5.

    The aliasing half always evaluates. The amplitude half returns ``None`` with
    a reason when the ERA5 inputs are absent, rather than substituting a
    plausible temperature series.
    """
    alias = aliasing_table(record_days)
    amplitude: float | None = None
    reason = ""
    try:
        irradiance = None if ssrd_J_m2 is None else ssrd_to_irradiance(ssrd_J_m2)
        q = differential_thermal_strain(
            air_temperature_K,
            sea_temperature_K,
            irradiance,
            restraint_factor=restraint_factor,
            cfg=cfg,
        )
        series = np.asarray(q.value, float)
        # Amplitude as half the peak-to-peak of the diurnal swing.
        amplitude = float(0.5 * (np.nanmax(series) - np.nanmin(series)) * 1e6)
    except DataUnavailable as exc:
        reason = f"{exc} Remedy: {exc.remedy}"

    return ThermalResult(
        aliasing=alias,
        amplitude_ustrain=amplitude,
        unavailable_reason=reason,
        cross_sensitivity_ustrain_per_K=fbg.temperature_sensitivity_ustrain_per_K,
    )
