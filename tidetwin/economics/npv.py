"""Monte Carlo net present value of the monitoring system, with tornado sensitivity.

**Every input on this page is ASSUMED.** None of it derives from a solver, a
standard or a dataset; it is a set of commercial figures that vary by operator,
region, contract and year. They render red, they are all user-editable, and the
NPV they produce inherits an assumption-contaminated flag. An economic case built
on assumed inputs is a scenario, not a result, and the app says so.

The tornado plot is the useful output rather than the NPV itself: it shows which
assumption the answer is hostage to, which is a question the model can answer
honestly even when the level of the NPV cannot be trusted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from ..provenance import Quantity, assumed, derived

__all__ = ["EconomicInputs", "NPVResult", "monte_carlo_npv", "tornado"]


@dataclass(frozen=True)
class EconomicInputs:
    """Commercial assumptions. All ASSUMED, all editable.

    Distributions are lognormal on positive cost quantities (a cost cannot go
    negative and its uncertainty is multiplicative) and normal on rates.
    ``*_cv`` are coefficients of variation.
    """

    sensor_capex: float = 450_000.0
    sensor_capex_cv: float = 0.35
    rov_spread_day_rate: float = 85_000.0
    rov_spread_day_rate_cv: float = 0.30
    install_days: float = 6.0
    install_days_cv: float = 0.40
    interrogator_opex_per_year: float = 60_000.0
    interrogator_opex_cv: float = 0.30
    sensor_failure_rate_per_year: float = 0.08
    sensor_failure_rate_cv: float = 0.50
    replacement_cost: float = 120_000.0
    replacement_cost_cv: float = 0.40
    false_positive_inspection_cost: float = 500_000.0
    false_positive_cost_cv: float = 0.35
    false_positives_per_year: float = 0.3
    avoided_campaigns_per_year: float = 0.25
    avoided_campaign_cost: float = 2_000_000.0
    avoided_campaign_cost_cv: float = 0.35
    discount_rate: float = 0.09
    discount_rate_sd: float = 0.02
    horizon_years: int = 20

    def as_quantities(self) -> list[Quantity]:
        out: list[Quantity] = []
        for f in fields(self):
            if f.name.endswith(("_cv", "_sd")):
                continue
            units = (
                "USD"
                if "cost" in f.name or "capex" in f.name
                else "USD/d"
                if "day_rate" in f.name
                else "USD/yr"
                if "opex" in f.name
                else "d"
                if f.name == "install_days"
                else "yr"
                if f.name == "horizon_years"
                else "1/yr"
                if "per_year" in f.name
                else "-"
            )
            out.append(assumed(getattr(self, f.name), units, f.name.replace("_", " ")))
        return out


@dataclass
class NPVResult:
    samples: np.ndarray
    inputs: EconomicInputs
    seed: int
    life_extension: bool = False

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def median(self) -> float:
        return float(np.median(self.samples))

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.samples, q))

    @property
    def probability_positive(self) -> float:
        """The number that actually matters: odds the investment pays back at all."""
        return float(np.mean(self.samples > 0.0))

    def as_quantity(self) -> Quantity:
        return derived(
            self.mean,
            "USD",
            "expected NPV",
            self.inputs.as_quantities(),
            f"Monte Carlo over {self.samples.size} draws, seed {self.seed}",
            uncertainty=float(np.std(self.samples, ddof=1)),
            note="assumption-contaminated: every economic input is ASSUMED",
        )


def _lognormal(rng, mean: float, cv: float, size: int) -> np.ndarray:
    if mean <= 0 or cv <= 0:
        return np.full(size, max(mean, 0.0))
    sigma = np.sqrt(np.log1p(cv**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return rng.lognormal(mu, sigma, size=size)


def monte_carlo_npv(
    inputs: EconomicInputs = EconomicInputs(),
    n_samples: int = 20_000,
    seed: int = 20260728,
    life_extension_years: float = 0.0,
    life_extension_value_per_year: float = 0.0,
) -> NPVResult:
    """NPV of deploying the monitoring system over the horizon.

    ``life_extension_*`` add the value of deferring decommissioning, which is the
    case operators most often use to justify monitoring. It is kept separate from
    the base case so the two are not conflated.
    """
    rng = np.random.default_rng(seed)
    n = n_samples
    i = inputs

    capex = _lognormal(rng, i.sensor_capex, i.sensor_capex_cv, n)
    day_rate = _lognormal(rng, i.rov_spread_day_rate, i.rov_spread_day_rate_cv, n)
    days = _lognormal(rng, i.install_days, i.install_days_cv, n)
    install = capex + day_rate * days

    opex = _lognormal(rng, i.interrogator_opex_per_year, i.interrogator_opex_cv, n)
    fail_rate = np.clip(_lognormal(rng, i.sensor_failure_rate_per_year, i.sensor_failure_rate_cv, n), 0, 1)
    repl = _lognormal(rng, i.replacement_cost, i.replacement_cost_cv, n)
    fp_cost = _lognormal(rng, i.false_positive_inspection_cost, i.false_positive_cost_cv, n)
    avoided = _lognormal(rng, i.avoided_campaign_cost, i.avoided_campaign_cost_cv, n)
    rate = np.clip(rng.normal(i.discount_rate, i.discount_rate_sd, n), 0.0, 0.5)

    annual = (
        i.avoided_campaigns_per_year * avoided
        - opex
        - fail_rate * repl
        - i.false_positives_per_year * fp_cost
    )
    if life_extension_years > 0:
        annual = annual + life_extension_value_per_year

    npv = -install
    for year in range(1, int(i.horizon_years) + 1):
        npv = npv + annual / (1.0 + rate) ** year
    if life_extension_years > 0:
        for year in range(int(i.horizon_years) + 1, int(i.horizon_years + life_extension_years) + 1):
            npv = npv + life_extension_value_per_year / (1.0 + rate) ** year

    return NPVResult(
        samples=npv, inputs=i, seed=seed, life_extension=life_extension_years > 0
    )


def tornado(
    inputs: EconomicInputs = EconomicInputs(),
    n_samples: int = 4000,
    seed: int = 20260728,
    swing: float = 0.3,
) -> list[tuple[str, float, float]]:
    """One-at-a-time sensitivity: each input moved by +/- ``swing``.

    Returns ``(name, npv_low, npv_high)`` sorted by the width of the swing, which
    is the ordering a tornado plot needs.
    """
    base = asdict(inputs)
    out: list[tuple[str, float, float]] = []
    for name, value in base.items():
        if name.endswith(("_cv", "_sd")) or name == "horizon_years":
            continue
        lo_kw = dict(base)
        hi_kw = dict(base)
        lo_kw[name] = value * (1.0 - swing)
        hi_kw[name] = value * (1.0 + swing)
        lo = monte_carlo_npv(EconomicInputs(**lo_kw), n_samples, seed).mean
        hi = monte_carlo_npv(EconomicInputs(**hi_kw), n_samples, seed).mean
        out.append((name.replace("_", " "), lo, hi))
    return sorted(out, key=lambda r: -abs(r[2] - r[1]))
