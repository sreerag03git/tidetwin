"""S2-solar against S2-tidal: a separability analysis.

The solar semidiurnal tide S2 has a period of exactly 12.000 h. So does the
second harmonic of solar heating. They are the same frequency. No record length
separates them, because separability depends on frequency *difference* and the
difference here is exactly zero.

What a longer record can do is separate S2 from its neighbours - K2 at 11.9672 h
and, more usefully, M2 at 12.4206 h. The Rayleigh criterion says two
constituents are resolvable when

.. math:: |f_1 - f_2| \\ge 1/T

so the record length needed to split a pair is the reciprocal of their frequency
difference. This module reports those lengths and the resulting conclusion: any
method that leans on S2 is reading a sum of tide and sunshine that cannot be
decomposed, while M2 is 14.77 days from S2 and clean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..loads.tides import CONSTITUENT_SPEEDS_DEG_PER_HOUR, constituent_frequency

__all__ = ["SEPARATION_PAIRS", "separation_days", "aliasing_table", "AliasingResult"]

#: Pairs whose separability decides whether an M2-carrier method is sound.
SEPARATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("M2", "S2"),
    ("M2", "N2"),
    ("S2", "K2"),
    ("K1", "O1"),
    ("K1", "P1"),
)


def separation_days(a: str, b: str) -> float:
    """Record length in days needed to resolve two constituents.

    Returns ``inf`` for an exactly coincident pair.
    """
    fa = float(constituent_frequency(a).value) / (2.0 * np.pi)
    fb = float(constituent_frequency(b).value) / (2.0 * np.pi)
    df = abs(fa - fb)
    return float("inf") if df == 0.0 else float(1.0 / df / 86400.0)


@dataclass(frozen=True)
class AliasingResult:
    """Separability of each pair at a given record length."""

    record_days: float
    rows: tuple[tuple[str, str, float, bool], ...]  # (a, b, days_needed, resolved)
    solar_s2_days: float
    conclusion: str

    def resolved(self, a: str, b: str) -> bool:
        for x, y, _d, ok in self.rows:
            if {x, y} == {a, b}:
                return ok
        raise KeyError(f"pair {a}/{b} not in this analysis")


def aliasing_table(record_days: float) -> AliasingResult:
    """Which constituent pairs a record of this length can separate."""
    rows = []
    for a, b in SEPARATION_PAIRS:
        need = separation_days(a, b)
        rows.append((a, b, need, bool(record_days >= need)))

    # Solar heating's semidiurnal harmonic sits at exactly 12.000 h, which is S2.
    solar_period_h = 12.0
    s2_period_h = 360.0 / CONSTITUENT_SPEEDS_DEG_PER_HOUR["S2"]
    coincident = abs(solar_period_h - s2_period_h) < 1e-12
    solar_days = float("inf") if coincident else 1.0 / abs(
        1.0 / (solar_period_h * 3600) - 1.0 / (s2_period_h * 3600)
    ) / 86400.0

    m2s2 = separation_days("M2", "S2")
    conclusion = (
        "The solar semidiurnal heating harmonic and the S2 tidal constituent share a "
        f"period of {s2_period_h:.4f} h exactly. Their frequency difference is zero, so no "
        "record length separates them and any S2-based strain amplitude is an "
        "unresolvable sum of tidal loading and thermal response. M2 sits "
        f"{m2s2:.2f} days from S2 in Rayleigh terms and carries no solar harmonic, which "
        "is why it must be the primary carrier. "
        + (
            f"A {record_days:.0f} day record does resolve M2 from S2."
            if record_days >= m2s2
            else f"A {record_days:.0f} day record does NOT resolve M2 from S2; at least "
            f"{m2s2:.2f} days are needed, and below that the carrier is itself contaminated."
        )
    )
    return AliasingResult(
        record_days=float(record_days),
        rows=tuple(rows),
        solar_s2_days=solar_days,
        conclusion=conclusion,
    )
