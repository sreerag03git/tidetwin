"""The life-extension value case.

Operators justify structural monitoring on deferred decommissioning far more
often than on avoided inspection campaigns, so it is worth separating: the base
case in :mod:`tidetwin.economics.npv` credits the system with avoiding
campaigns, while this one credits it with keeping the asset producing past its
design life.

Both cases are built entirely from ASSUMED inputs and both are downstream of C3.
A monitoring system that cannot reliably distinguish a crack from a spring tide
does not support a life-extension argument to a regulator, whatever the
arithmetic says - so the value computed here is conditional on a claim the
ledger currently records as failing, and the function returns that conditionality
alongside the number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..provenance import Quantity, assumed, derived
from .npv import EconomicInputs, NPVResult, monte_carlo_npv

__all__ = ["LifeExtensionInputs", "LifeExtensionResult", "life_extension_case"]


@dataclass(frozen=True)
class LifeExtensionInputs:
    """Assumptions specific to the life-extension case. All ASSUMED."""

    extension_years: float = 5.0
    annual_production_margin: float = 8_000_000.0
    annual_production_margin_cv: float = 0.30
    deferred_decommissioning_cost: float = 40_000_000.0
    deferred_decommissioning_cv: float = 0.25
    probability_regulator_accepts: float = 0.5

    def as_quantities(self) -> list[Quantity]:
        return [
            assumed(self.extension_years, "yr", "life extension sought"),
            assumed(self.annual_production_margin, "USD/yr", "production margin retained"),
            assumed(self.deferred_decommissioning_cost, "USD", "decommissioning deferred"),
            assumed(
                self.probability_regulator_accepts,
                "-",
                "probability the regulator accepts monitoring as evidence",
            ),
        ]


@dataclass
class LifeExtensionResult:
    base: NPVResult
    extended: NPVResult
    increment_samples: np.ndarray
    inputs: LifeExtensionInputs
    conditional_on: str

    @property
    def mean_increment(self) -> float:
        return float(np.mean(self.increment_samples))

    @property
    def probability_positive(self) -> float:
        return float(np.mean(self.increment_samples > 0.0))

    def as_quantity(self) -> Quantity:
        return derived(
            self.mean_increment,
            "USD",
            "incremental NPV from life extension",
            self.inputs.as_quantities(),
            "Monte Carlo difference between the extended and base cases",
            uncertainty=float(np.std(self.increment_samples, ddof=1)),
            note="assumption-contaminated; " + self.conditional_on,
        )


def life_extension_case(
    economics: EconomicInputs = EconomicInputs(),
    inputs: LifeExtensionInputs = LifeExtensionInputs(),
    n_samples: int = 20_000,
    seed: int = 20260728,
    detection_is_reliable: bool = False,
) -> LifeExtensionResult:
    """NPV with and without the life-extension argument.

    ``detection_is_reliable`` should carry the C3 verdict. When it is ``False``
    the regulator-acceptance probability is not silently applied at its assumed
    value: the result records that the whole case rests on a claim that has not
    been supported.
    """
    rng = np.random.default_rng(seed)
    base = monte_carlo_npv(economics, n_samples=n_samples, seed=seed)

    sigma = np.sqrt(np.log1p(inputs.annual_production_margin_cv**2))
    mu = np.log(max(inputs.annual_production_margin, 1e-9)) - 0.5 * sigma**2
    margin = rng.lognormal(mu, sigma, size=n_samples)

    accepted = rng.random(n_samples) < np.clip(inputs.probability_regulator_accepts, 0.0, 1.0)
    rate = np.clip(rng.normal(economics.discount_rate, economics.discount_rate_sd, n_samples), 0.0, 0.5)

    horizon = int(economics.horizon_years)
    increment = np.zeros(n_samples)
    for year in range(horizon + 1, horizon + int(inputs.extension_years) + 1):
        increment += margin / (1.0 + rate) ** year
    increment += inputs.deferred_decommissioning_cost * (
        1.0 / (1.0 + rate) ** horizon - 1.0 / (1.0 + rate) ** (horizon + inputs.extension_years)
    )
    increment = np.where(accepted, increment, 0.0)

    extended = NPVResult(
        samples=base.samples + increment,
        inputs=economics,
        seed=seed,
        life_extension=True,
    )
    conditional = (
        "conditional on C3, which this build records as FAILING: a method that cannot "
        "separate a crack from environmental variation does not support a life-extension "
        "submission, so this increment should be read as an upper bound on a case that "
        "has not yet been made"
        if not detection_is_reliable
        else "conditional on the C3 verdict holding"
    )
    return LifeExtensionResult(
        base=base,
        extended=extended,
        increment_samples=increment,
        inputs=inputs,
        conditional_on=conditional,
    )
