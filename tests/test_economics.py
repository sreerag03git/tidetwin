"""Economics (C8): the paper's current claim is a break-even threshold, not a saving.

The earlier manuscript claimed a fleet NPV; the current paper withdraws that and
states plainly that, on avoided inspection alone, condition-based monitoring is a
net cost, breaking even only if it cuts the annual unplanned-intervention
probability by about 0.5 percentage points - a reduction it has not demonstrated.
C8 therefore tests the HONESTY of that framing, which an independent cost model
whose NPV sign is hostage to its assumptions confirms rather than refutes.
"""

from __future__ import annotations

from tidetwin.abstract import PAPER
from tidetwin.claims.registry import Artifacts, Status, _c8
from tidetwin.economics.npv import EconomicInputs, monte_carlo_npv


def test_c8_passes_on_an_honest_break_even_framing():
    """A claim that declines to claim a saving, and says so, is not refutable."""
    r = monte_carlo_npv(EconomicInputs(), n_samples=400, seed=7)
    assert r.inputs.n_jackets == PAPER.n_jackets == 30

    res = _c8(Artifacts(c8=r))
    assert res.status is Status.PASS
    assert "break-even threshold" in res.computed_text
    assert "no net saving" in res.computed_text


def test_c8_states_the_sign_is_hostage_to_assumptions():
    """The point of a threshold, not a figure, is that the inputs are all assumed."""
    r = monte_carlo_npv(EconomicInputs(), n_samples=200, seed=7)
    detail = _c8(Artifacts(c8=r)).detail
    assert "hostage to the assumptions" in detail
    # The earlier $19.9M headline is named only to say it is gone.
    assert "no longer claims a net saving" in detail
