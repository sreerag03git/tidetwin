"""Acoustic emission hit-rate model.

AE is the obvious complement to a strain-ratio method: it responds to crack
*extension* rather than to compliance, so its failure modes are uncorrelated
with the tidal channel's. The hit rate is conventionally taken proportional to a
power of the stress intensity range,

.. math:: \\dot{N} \\propto (\\Delta K)^{p}

which is the same driving force the Paris law uses, so AE activity and crack
growth rate track one another (Berkovits & Fang, "Study of fatigue crack
characteristics by acoustic emission", Engineering Fracture Mechanics
51(3):401-416, 1995).

The exponent and the proportionality constant are strongly dependent on
sensor coupling, threshold setting and ambient noise, none of which is knowable
without the specific installation. They are therefore ASSUMED, exposed, and the
resulting hit rate is reported as an order of magnitude rather than a count.
On a producing platform the ambient acoustic background - flow noise, machinery,
riser and conductor movement - is the limiting factor, and the model reports the
threshold that background implies rather than assuming it away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..provenance import Quantity, assumed, derived

__all__ = ["AESpec", "hit_rate", "detectable_growth_rate"]


@dataclass(frozen=True)
class AESpec:
    """Acoustic emission channel settings. All ASSUMED."""

    coefficient: float = 1.0e-3  # hits per cycle at unit delta_K
    exponent: float = 4.0  # power on delta_K
    background_hits_per_hour: float = 50.0
    threshold_multiple: float = 3.0  # detection needs this multiple of background

    def as_quantities(self) -> list[Quantity]:
        return [
            assumed(self.coefficient, "1/cycle", "AE hit-rate coefficient"),
            assumed(self.exponent, "-", "AE hit-rate exponent on delta K"),
            assumed(self.background_hits_per_hour, "1/h", "ambient AE background"),
            assumed(self.threshold_multiple, "-", "detection threshold over background"),
        ]


def hit_rate(
    delta_K_MPa_sqrt_m: np.ndarray | float,
    cycles_per_hour: float,
    spec: AESpec = AESpec(),
) -> Quantity:
    """Expected AE hits per hour for a given stress intensity range."""
    dk = np.maximum(np.asarray(delta_K_MPa_sqrt_m, float), 0.0)
    rate = spec.coefficient * dk**spec.exponent * cycles_per_hour
    return derived(
        rate,
        "1/h",
        "acoustic emission hit rate",
        spec.as_quantities(),
        f"coefficient x deltaK^{spec.exponent:g} x {cycles_per_hour:g} cycles/h",
        note=(
            f"ambient background is {spec.background_hits_per_hour:g} hits/h; anything below "
            f"{spec.threshold_multiple:g}x that is not distinguishable from platform noise"
        ),
    )


def detectable_growth_rate(spec: AESpec = AESpec(), cycles_per_hour: float = 300.0) -> float:
    """Smallest ``delta_K`` whose hit rate clears the ambient background.

    Inverting the hit-rate law at the detection threshold. Reported so the AE
    channel is compared on the same footing as the strain channel: both have a
    noise floor set by the environment, not by the instrument.
    """
    target = spec.threshold_multiple * spec.background_hits_per_hour
    denom = spec.coefficient * cycles_per_hour
    if denom <= 0:
        return float("inf")
    return float((target / denom) ** (1.0 / spec.exponent))
