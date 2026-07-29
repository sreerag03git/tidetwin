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
    robust_scale,
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


def test_convergence_trace_flags_a_still_settling_estimate():
    """A small well-behaved sample: noisy, but nothing pathological.

    Must be reported as 'needs more samples', not as heavy tailed - the two call
    for opposite responses.
    """
    # A genuine regime shift: the scale trebles half way through, so sigma is
    # still climbing at the end. Large n, so the sampling-noise threshold is
    # small and the drift is unambiguous.
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0, 1.0, 2000), rng.normal(0, 3.0, 2000)])
    tr = convergence_trace(x)
    assert not tr.converged
    assert not tr.heavy_tailed, "finite variance: more samples would settle it"
    assert "NOT converged" in tr.verdict
    assert "Increase the sample count" in tr.verdict

    # A small clean sample IS converged, to the precision its size allows.
    # Demanding better would be demanding precision below the noise floor.
    small = convergence_trace(rng.normal(size=60))
    assert small.converged
    assert small.threshold > 0.2, "threshold must scale with the sampling error"


def test_convergence_trace_handles_a_tiny_sample():
    tr = convergence_trace(np.array([1.0, 2.0]))
    assert not tr.converged
    assert not np.isfinite(tr.relative_drift)


# ------------------------------------------------------------- heavy tails


def test_heavy_tails_are_distinguished_from_still_settling():
    """A Cauchy ratio has no variance; sigma grows with n instead of settling.

    This is the Woods Hole case: a near-rectilinear tidal current drives the
    strain ratio's denominator through zero, and the ratio becomes Cauchy-like.
    """
    rng = np.random.default_rng(21)
    tr = convergence_trace(rng.standard_cauchy(4000))
    assert not tr.converged
    assert tr.heavy_tailed
    assert "no finite variance" in tr.verdict
    assert "More samples will not help" in tr.verdict
    # The defining signature: the robust scale settles, sigma does not.
    assert tr.robust_drift <= 0.05 < tr.relative_drift

    # Detection must not hinge on one lucky draw, so check several seeds.
    flagged = sum(
        convergence_trace(np.random.default_rng(s).standard_cauchy(4000)).heavy_tailed
        for s in range(8)
    )
    assert flagged >= 7, f"only {flagged}/8 Cauchy samples detected"

    # A well-behaved sample must not be misdiagnosed, at any size or seed.
    false_positives = sum(
        convergence_trace(np.random.default_rng(s).normal(size=n)).heavy_tailed
        for s in range(8) for n in (60, 500, 4000)
    )
    assert false_positives == 0, f"{false_positives} normal samples wrongly flagged"

    # A finite-variance scale mixture is heavy relative to a normal but still
    # converges, so it must not be labelled as having no variance.
    mixture = np.concatenate([rng.normal(0, s, 8000) for s in (0.5, 1.0, 2.0)])
    assert not convergence_trace(mixture).heavy_tailed


def test_robust_scale_matches_sigma_for_a_normal_sample():
    """0.7413 * IQR is the normal-consistent estimator, so it must agree."""
    rng = np.random.default_rng(22)
    x = rng.normal(0.0, 3.0, 200_000)
    assert robust_scale(x) == pytest.approx(3.0, rel=0.02)
    assert robust_scale(x) == pytest.approx(np.std(x, ddof=1), rel=0.02)


def test_robust_scale_stays_finite_where_sigma_explodes():
    rng = np.random.default_rng(23)
    cauchy = rng.standard_cauchy(20_000)
    r = robust_scale(cauchy)
    assert np.isfinite(r)
    # The Cauchy(0,1) IQR is 2, so the normal-consistent scale is about 1.48.
    assert r == pytest.approx(1.483, rel=0.1)
    # The sample standard deviation is meaningless here and far larger.
    assert np.std(cauchy, ddof=1) > 5 * r


def test_robust_scale_needs_a_usable_sample():
    assert not np.isfinite(robust_scale(np.array([1.0, 2.0])))


def test_verdict_uses_the_robust_scale_when_the_variance_diverges():
    rng = np.random.default_rng(24)
    res = _result(joint_sd=99.0, baseline=2.0)
    res.joint_samples = 2.0 + 0.02 * rng.standard_cauchy(4000)
    res.joint_sd = float(np.std(res.joint_samples, ddof=1))
    res.convergence = convergence_trace(res.joint_samples)
    assert res.heavy_tailed
    # The robust scale is far smaller and far more stable than sigma.
    assert res.effective_sd == res.joint_robust_sd
    assert res.effective_sd < res.joint_sd
    assert "robust scale" in res.dispersion_kind

    status, msg = verdict_against_claimed_signature(res, 0.111)
    assert status in ("PASS", "MARGINAL", "FAIL")
    assert "variance itself does not converge" in msg
    assert "poor basis for a detection threshold" in msg


def test_heavy_tailed_run_is_decided_not_withheld():
    """No amount of sampling fixes an undefined variance, so do not stall forever."""
    from tidetwin.claims.registry import Artifacts, Status, evaluate_all

    rng = np.random.default_rng(25)
    res = _result(joint_sd=1.0, baseline=2.0)
    res.joint_samples = 2.0 + 0.5 * rng.standard_cauchy(4000)
    res.joint_sd = float(np.std(res.joint_samples, ddof=1))
    res.convergence = convergence_trace(res.joint_samples)
    assert res.heavy_tailed

    c3 = next(r for r in evaluate_all(Artifacts(c3=res)) if r.claim_id == "C3")
    assert c3.status is Status.FAIL, "a heavy-tailed run must still be decided"
    assert "withheld" not in c3.detail
    assert "variance of the method's own detection statistic does not exist" in c3.detail
    assert "variance undefined" in c3.computed_text


def test_the_two_failure_modes_get_opposite_treatment():
    """Still settling -> withhold and ask for more samples.

    Heavy tailed -> decide on the robust scale, because more samples cannot help.
    Conflating the two would either stall forever or quote a diverging number.
    """
    from tidetwin.claims.registry import Artifacts, Status, evaluate_all

    settling = _result(joint_sd=0.3)
    settling.joint_samples = np.concatenate([np.random.default_rng(4).normal(0, 1.0, 2000),
                                       np.random.default_rng(5).normal(0, 3.0, 2000)])
    settling.convergence = convergence_trace(settling.joint_samples)
    # Preconditions: this fixture must genuinely be the "needs more samples" case.
    assert not settling.convergence.converged
    assert not settling.convergence.heavy_tailed
    withheld = next(r for r in evaluate_all(Artifacts(c3=settling)) if r.claim_id == "C3")

    heavy = _result(joint_sd=0.3)
    heavy.joint_samples = 2.0 + 0.4 * np.random.default_rng(26).standard_cauchy(4000)
    heavy.joint_sd = float(np.std(heavy.joint_samples, ddof=1))
    heavy.convergence = convergence_trace(heavy.joint_samples)
    assert heavy.convergence.heavy_tailed
    decided = next(r for r in evaluate_all(Artifacts(c3=heavy)) if r.claim_id == "C3")

    assert withheld.status is Status.UNTESTABLE_DATA
    assert decided.status is not Status.UNTESTABLE_DATA
    assert "Increase the sample count" in withheld.detail
    assert "More samples will not help" in decided.detail


def test_unconverged_run_withholds_the_verdict():
    """A Monte Carlo that has not converged has not decided anything."""
    from tidetwin.claims.registry import Artifacts, Status, evaluate_all

    res = _result(0.30)
    res.joint_samples = np.concatenate([np.random.default_rng(4).normal(0, 1.0, 2000),
                                       np.random.default_rng(5).normal(0, 3.0, 2000)])
    res.convergence = convergence_trace(res.joint_samples)
    assert not res.convergence.converged and not res.convergence.heavy_tailed
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


# ---------------------------------------------------------------------------
# The false-alarm fraction: the C3 failure stated without statistics.


def _res(samples, baseline=2.0):
    return NuisanceResult(
        baseline_ratio=baseline,
        per_channel_sd={},
        per_channel_samples={},
        joint_sd=float(np.std(samples, ddof=1)),
        joint_samples=np.asarray(samples, float),
        n_samples=len(samples),
        record_days=14.0,
        seed=1,
        ranges=NuisanceRanges(),
    )


def test_no_false_alarms_when_the_sea_never_moves_the_ratio():
    assert _res(np.full(200, 2.0)).false_alarm_fraction(0.111) == 0.0


def test_every_draw_is_a_false_alarm_when_the_noise_dwarfs_the_signal():
    # All draws sit 50 percent away from the baseline, far past an 11.1 percent step.
    assert _res(np.full(200, 3.0)).false_alarm_fraction(0.111) == 1.0


def test_the_count_is_two_sided():
    """A detector watching for an 11.1 percent shift cannot know the sign.

    C2 finds the sign of a real crack's effect depends on which joint spring
    softens, so counting only upward excursions would understate the false-alarm
    rate by about half.
    """
    down = _res(np.full(100, 2.0 * (1 - 0.2)))
    assert down.false_alarm_fraction(0.111) == 1.0


def test_the_threshold_is_relative_to_the_baseline_not_absolute():
    """A 0.2 absolute excursion is a false alarm on a ratio of 2 but not on one of 20.

    Getting this wrong would make the verdict depend on which joint was chosen,
    since the intact ratio varies by orders of magnitude across the structure.
    """
    assert _res(np.full(50, 2.0 + 0.3), baseline=2.0).false_alarm_fraction(0.111) == 1.0
    assert _res(np.full(50, 20.0 + 0.3), baseline=20.0).false_alarm_fraction(0.111) == 0.0


def test_only_draws_past_the_signature_count():
    half = np.concatenate([np.full(50, 2.0), np.full(50, 2.0 * 1.5)])
    assert _res(half).false_alarm_fraction(0.111) == pytest.approx(0.5)


def test_a_degenerate_baseline_reports_nan_rather_than_dividing_by_zero():
    assert np.isnan(_res(np.ones(10), baseline=0.0).false_alarm_fraction())
