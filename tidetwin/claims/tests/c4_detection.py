"""C4 - time to detection at SNR 3, spring against neap.

The claim is 4 to 9 days. The analysis that decides it is the split of the C3
nuisance budget into a part that averages away and a part that does not: the
``sqrt(N)`` coherent gain applies only to the first, and the second sets a floor
the record length never crosses.

Spring and neap are reported separately because the current amplitude differs
between them by the spring/neap ratio, and drag scales with the square of
velocity - so the signal is several times weaker at neaps while the systematic
floor is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...nuisance import NuisanceResult
from ...signal.detect import DetectionModel, coherent_gain_curve, detection_time_cdf

__all__ = ["DetectionCase", "detection_cases", "SNR_TARGET"]

#: The signal-to-noise ratio the claim is stated at.
SNR_TARGET = 3.0


@dataclass
class DetectionCase:
    label: str
    times_days: np.ndarray
    cdf: np.ndarray
    percentiles: dict[str, float]
    model: DetectionModel
    signature: float

    @property
    def reaches_claimed_window(self) -> bool:
        p50 = self.percentiles.get("p50", np.inf)
        return bool(np.isfinite(p50) and 4.0 <= p50 <= 9.0)


def detection_cases(
    nuisance: NuisanceResult,
    signature_fraction: float,
    record_days: float = 14.0,
    spring_neap_ratio: float = 2.33,
    seed: int = 20260728,
    n_trials: int = 2000,
) -> dict[str, DetectionCase]:
    """Detection-time distributions for spring and neap conditions.

    Drag scales as the square of current speed, so a neap current weaker by
    ``spring_neap_ratio`` weakens the strain signature by its square while the
    systematic nuisance floor stays where it is.
    """
    rnd, sysm = nuisance.split_random_systematic()
    base = abs(nuisance.baseline_ratio)
    out: dict[str, DetectionCase] = {}
    for label, scale in (
        ("spring", 1.0),
        ("neap", 1.0 / max(spring_neap_ratio, 1e-6) ** 2),
    ):
        model = DetectionModel(
            sigma_random=rnd,
            sigma_systematic=sysm,
            baseline_ratio=nuisance.baseline_ratio,
            false_alarm_rate=0.01,
        )
        signature = signature_fraction * base * scale
        xs, cdf, pct = detection_time_cdf(
            model,
            signature=signature,
            record_hours=24.0 * record_days,
            n_trials=n_trials,
            seed=seed,
        )
        out[label] = DetectionCase(label, xs, cdf, pct, model, signature)
    return out


def gain_table(model: DetectionModel, n_max: int = 100) -> dict[str, np.ndarray]:
    """Theoretical against achieved coherent averaging gain."""
    n = np.arange(1, n_max + 1)
    theoretical, achieved = coherent_gain_curve(model, n)
    return {"n_records": n, "theoretical": theoretical, "achieved": achieved}
