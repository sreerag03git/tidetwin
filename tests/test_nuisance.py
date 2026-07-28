"""C3, the deciding test: convergence, interaction, correlation and break-even.

C3 decides whether the other eight claims mean anything, so the machinery that
produces it gets tested rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from tidetwin.nuisance import (
    CHANNELS,
    RANDOM_CHANNELS,
    SYSTEMATIC_CHANNELS,
    BreakEven,
    NuisanceRanges,
    NuisanceResult,
    _correlated_storm,
    convergence_trace,
    variance_decomposition,
    verdict,
    verdict_against_claimed_signature,
)


def _result(joint_sd: float, baseline: float = 2.0, per_channel=None) -> NuisanceResult:
    per_channel = per_channel or {c: joint_sd / np.sqrt(len(CHANNELS)) for c in CHANNELS}
    return NuisanceResult(
        baseline_ratio=baseline,
        per_channel_sd=per_channel,
        per_channel_samples={c: np.zeros(4) for c in CHANNELS},
        joint_sd=joint_sd,
        joint_samples=np.zeros(4),
        n_samples=200,
        record_days=14.0,
        seed=1,
        ranges=NuisanceRanges(),
    )


# ------------------------------------------------------------------ ranges


def test_scaling_narrows_every_channel_together():
    r = NuisanceRanges()
    h = r.scaled(0.5)
    assert h.wind_current_sd_ms == pytest.approx(r.wind_current_sd_ms * 0.5)
    assert h.water_level_sd_m == pytest.approx(r.water_level_sd_m * 0.5)
    assert h.fbg_drift_sd_ustrain == pytest.approx(r.fbg_drift_sd_ustrain * 0.5)
    assert h.marine_growth_mm[1] == pytest.approx(r.marine_growth_mm[1] * 0.5)
    # Scour is a retained-stiffness fraction, so halving the range moves the
    # lower bound towards 1.0 (less scour), not towards 0.
    assert h.scour_factor_range[0] > r.scour_factor_range[0]
    assert h.scour_factor_range[1] == 1.0


def test_scaling_to_zero_removes_all_variation():
    z = NuisanceRanges().scaled(0.0)
    assert z.wind_current_sd_ms == 0.0
    assert z.marine_growth_mm[1] == 0.0
    assert z.scour_factor_range == (1.0, 1.0)


def test_channel_classification_is_complete_and_disjoint():
    assert RANDOM_CHANNELS | SYSTEMATIC_CHANNELS == set(CHANNELS)
    assert not (RANDOM_CHANNELS & SYSTEMATIC_CHANNELS)


# ------------------------------------------------------------- correlation


def test_correlated_storm_draws_have_unit_variance_and_the_requested_correlation():
    rng = np.random.default_rng(0)
    for rho in (0.0, 0.3, 0.6, 1.0):
        z = np.array([_correlated_storm(rng, rho) for _ in range(40_000)])
        assert z.std(axis=0) == pytest.approx(np.ones(3), abs=0.02)
        c = np.corrcoef(z.T)
        off = c[np.triu_indices(3, k=1)]
        assert off == pytest.approx(np.full(3, rho), abs=0.03)


def test_correlation_is_clipped_to_a_valid_range():
    rng = np.random.default_rng(1)
    z = np.array([_correlated_storm(rng, 5.0) for _ in range(2000)])
    assert np.all(np.isfinite(z))
    assert z.std(axis=0) == pytest.approx(np.ones(3), abs=0.1)


# ------------------------------------------------------------- convergence


def test_convergence_trace_detects_a_settled_estimate():
    rng = np.random.default_rng(2)
    tr = convergence_trace(rng.normal(0.0, 1.0, 4000))
    assert tr.converged
    assert tr.relative_drift < 0.05
    assert tr.sigma[-1] == pytest.approx(1.0, abs=0.05)
    assert "Converged" in tr.verdict


def test_convergence_trace_flags_a_drifting_estimate():
    # Variance grows steadily through the run: never settles.
    x = np.concatenate([np.random.default_rng(3).normal(0, s, 400) for s in (0.1, 1.0, 8.0)])
    tr = convergence_trace(x)
    assert not tr.converged
    assert "NOT converged" in tr.verdict


def test_convergence_trace_handles_a_tiny_sample():
    tr = convergence_trace(np.array([1.0, 2.0]))
    assert not tr.converged
    assert not np.isfinite(tr.relative_drift)


def test_unconverged_run_withholds_the_verdict():
    """A Monte Carlo that has not converged has not decided anything."""
    from tidetwin.claims.registry import Artifacts, Status, evaluate_all

    res = _result(0.30)
    res.convergence = convergence_trace(
        np.concatenate([np.random.default_rng(4).normal(0, s, 400) for s in (0.1, 1.0, 8.0)])
    )
    art = Artifacts(c3=res)
    c3 = next(r for r in evaluate_all(art) if r.claim_id == "C3")
    assert c3.status is Status.UNTESTABLE_DATA
    assert "withheld" in c3.detail


# ---------------------------------------------------------- decomposition


def test_variance_decomposition_is_zero_for_perfectly_additive_channels():
    per = {c: 0.1 for c in CHANNELS}
    joint = np.sqrt(sum(v**2 for v in per.values()))
    d = variance_decomposition(per, joint)
    assert d.interaction == pytest.approx(0.0, abs=1e-12)
    assert abs(d.interaction_fraction) < 1e-9
    assert "close to independently" in d.interpretation


def test_variance_decomposition_names_cancellation():
    per = {c: 0.1 for c in CHANNELS}
    additive = np.sqrt(sum(v**2 for v in per.values()))
    d = variance_decomposition(per, additive * 0.6)
    assert d.interaction < 0
    assert "partially cancel" in d.interpretation


def test_variance_decomposition_names_reinforcement():
    per = {c: 0.1 for c in CHANNELS}
    additive = np.sqrt(sum(v**2 for v in per.values()))
    d = variance_decomposition(per, additive * 1.5)
    assert d.interaction > 0
    assert "reinforce" in d.interpretation


# ------------------------------------------------------------- break-even


def test_break_even_reports_an_unreachable_target():
    be = BreakEven(
        scales=np.array([0.1, 0.25, 0.5, 1.0]),
        sigmas=np.array([0.09, 0.10, 0.12, 0.14]),
        threshold=0.037,
        factor=float("nan"),
        signature_fraction=0.111,
    )
    assert not be.achievable
    assert "does not pass at any scaling" in be.statement


def test_break_even_reports_the_required_shrinkage():
    be = BreakEven(
        scales=np.array([0.1, 0.25, 0.5, 1.0]),
        sigmas=np.array([0.02, 0.05, 0.09, 0.14]),
        threshold=0.037,
        factor=0.18,
        signature_fraction=0.111,
    )
    assert not be.achievable
    assert "0.18x" in be.statement
    assert "simultaneously" in be.statement


def test_break_even_at_or_above_one_means_it_already_passes():
    be = BreakEven(np.array([1.0]), np.array([0.01]), 0.037, 1.0, 0.111)
    assert be.achievable
    assert "already passes" in be.statement


# ---------------------------------------------------------------- verdicts


def test_verdict_scales_with_the_damage_signature():
    assert verdict(0.01, 0.10)[0] == "PASS"
    assert verdict(0.02, 0.10)[0] == "MARGINAL"
    assert verdict(0.09, 0.10)[0] == "FAIL"
    assert "does not achieve reliable detection" in verdict(0.09, 0.10)[1]


def test_verdict_without_a_signature_is_untestable_not_a_pass():
    status, msg = verdict(0.01, float("nan"))
    assert status == "UNTESTABLE - DATA MISSING"
    assert "not passed" in msg
    assert verdict(0.01, 0.0)[0] == "UNTESTABLE - DATA MISSING"


def test_claimed_signature_verdict_is_independent_of_our_crack_model():
    res = _result(joint_sd=0.30, baseline=2.0)  # cv = 15 %
    status, msg = verdict_against_claimed_signature(res, 0.111)
    assert status == "FAIL"
    assert "does not depend on how the crack-to-compliance step was modelled" in msg

    quiet = _result(joint_sd=0.01, baseline=2.0)  # cv = 0.5 %
    assert verdict_against_claimed_signature(quiet, 0.111)[0] == "PASS"


def test_standard_error_shrinks_with_sample_count():
    r = _result(0.25)
    r.n_samples = 200
    se_small = r.joint_sd_standard_error
    r.n_samples = 2000
    assert r.joint_sd_standard_error < se_small
    assert r.joint_sd_standard_error == pytest.approx(se_small / np.sqrt(10), rel=0.02)


def test_random_systematic_split_reproduces_the_measured_joint_sigma():
    r = _result(0.25)
    rnd, sysm = r.split_random_systematic()
    assert float(np.hypot(rnd, sysm)) == pytest.approx(r.joint_sd, rel=1e-9)
    assert rnd > 0 and sysm > 0
