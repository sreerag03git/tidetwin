"""Economics: the fleet-versus-per-jacket basis, which the C8 comparison rests on.

The abstract's headline is a fleet figure and everything modelled here is per
jacket. Getting that wrong does not produce an error, only a verdict that is
wrong by the fleet size, so it is pinned.
"""

from __future__ import annotations

import pytest

from tidetwin.abstract import PAPER
from tidetwin.claims.registry import Artifacts, _c8
from tidetwin.economics.npv import EconomicInputs, monte_carlo_npv


def test_the_fleet_claim_is_not_compared_against_a_per_jacket_number():
    """The abstract's 19.9 MUSD is for 30 jackets; the model is per jacket.

    Comparing them directly overstated the claim by the fleet size, and in the
    direction that made the paper look far more optimistic than it is.
    """
    r = monte_carlo_npv(EconomicInputs(), n_samples=400, seed=7)
    assert r.inputs.n_jackets == PAPER.n_jackets == 30
    assert r.fleet_mean == pytest.approx(r.mean * 30)

    res = _c8(Artifacts(c8=r))
    assert "30 jackets" in res.computed_text
    # Both bases must appear, so neither can be quoted alone.
    assert "Per jacket" in res.detail
    assert f"{PAPER.npv_usd / 1e6:.1f}" in res.detail


def test_fleet_scaling_is_stated_as_conservative():
    """Shared interrogators and spread mobilisation would raise it, not lower it."""
    r = monte_carlo_npv(EconomicInputs(), n_samples=200, seed=7)
    assert "linearly" in _c8(Artifacts(c8=r)).detail
    assert "shared" in _c8(Artifacts(c8=r)).detail
