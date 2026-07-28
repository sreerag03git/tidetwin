"""C8 - the economic case.

Cannot pass, and says so. Every input is an unsourced commercial assumption, so
the output is a scenario rather than a result. The tornado plot is the usable
product: which assumption the answer is hostage to is a question the model can
answer honestly even when the level of the NPV cannot be trusted.

The case is also downstream of C3. The NPV credits the system with avoiding
inspection campaigns; a method that cannot separate a crack from a spring tide
does not avoid them. Both cases are therefore reported conditionally.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...economics.life_extension import (
    LifeExtensionInputs,
    LifeExtensionResult,
    life_extension_case,
)
from ...economics.npv import EconomicInputs, NPVResult, monte_carlo_npv, tornado

__all__ = ["EconomicsResult", "run_economics", "CLAIMED_NPV_USD"]

#: The figure the abstract asserts, used only as a comparison marker on the plot.
CLAIMED_NPV_USD = 19.9e6


@dataclass
class EconomicsResult:
    base: NPVResult
    life_extension: LifeExtensionResult
    tornado_rows: list[tuple[str, float, float]]
    conditional_on_c3: str

    @property
    def headline(self) -> str:
        b = self.base
        return (
            f"Base case expected NPV {b.mean / 1e6:.2f} MUSD "
            f"(p05 {b.percentile(5) / 1e6:.2f}, p95 {b.percentile(95) / 1e6:.2f}), "
            f"{b.probability_positive * 100:.0f} percent chance of being positive; "
            f"life extension adds {self.life_extension.mean_increment / 1e6:.2f} MUSD. "
            f"Claimed: {CLAIMED_NPV_USD / 1e6:.1f} MUSD."
        )

    @property
    def dominant_assumption(self) -> str:
        return self.tornado_rows[0][0] if self.tornado_rows else "none computed"


def run_economics(
    economics: EconomicInputs = EconomicInputs(),
    life: LifeExtensionInputs = LifeExtensionInputs(),
    n_samples: int = 20_000,
    seed: int = 20260728,
    detection_is_reliable: bool = False,
) -> EconomicsResult:
    """Base case, life-extension case and tornado sensitivity."""
    base = monte_carlo_npv(economics, n_samples=n_samples, seed=seed)
    ext = life_extension_case(
        economics, life, n_samples=n_samples, seed=seed,
        detection_is_reliable=detection_is_reliable,
    )
    rows = tornado(economics, n_samples=max(2000, n_samples // 10), seed=seed)
    return EconomicsResult(
        base=base,
        life_extension=ext,
        tornado_rows=rows,
        conditional_on_c3=(
            "Every figure here is conditional on C3. The avoided-campaign benefit assumes "
            "detection works; the ledger currently records C3 as failing, so these numbers "
            "are an upper bound on a case that has not been made."
            if not detection_is_reliable
            else "Conditional on the C3 verdict holding."
        ),
    )
