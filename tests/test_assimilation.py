"""Filters and calibration diagnostics.

The load-bearing test here is that the deterministic EnKF reduces *exactly* to
the Kalman filter in the linear-Gaussian limit. If that fails, nothing the
assimilation tab reports means anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from tidetwin.abstract import PAPER
from tidetwin.assimilation.baseline import prior_only, sn_miner_baseline
from tidetwin.claims.tests.c6_filter import run_comparison
from tidetwin.assimilation.calibration import (
    assess,
    coverage,
    crps_ensemble,
    pit_values,
    rank_histogram,
)
from tidetwin.assimilation.log_enkf import EnKFConfig, LogEnKF, enkf_update, inflate
from tidetwin.assimilation.particle import (
    ParticleFilter,
    effective_sample_size,
    systematic_resample,
)
from tidetwin.provenance import DataUnavailable


def _ensemble_with_exact_moments(mean: np.ndarray, cov: np.ndarray, n: int, seed: int = 0):
    """Build an ensemble whose sample mean and covariance are exactly (mean, cov)."""
    rng = np.random.default_rng(seed)
    d = mean.size
    X = rng.normal(size=(d, n))
    X -= X.mean(axis=1, keepdims=True)
    S = X @ X.T / (n - 1)
    # Whiten then colour, so the sample covariance is exactly cov.
    L_s = np.linalg.cholesky(S)
    L_c = np.linalg.cholesky(cov)
    X = L_c @ np.linalg.solve(L_s, X)
    return X + mean[:, None]


def test_deterministic_enkf_equals_the_kalman_filter():
    """Exact equivalence in the linear-Gaussian limit, to machine precision."""
    mean = np.array([1.5, -0.4])
    P = np.array([[0.9, 0.3], [0.3, 0.5]])
    H = np.array([[1.0, 2.0]])
    R = np.array([[0.25]])
    y = np.array([2.1])

    X = _ensemble_with_exact_moments(mean, P, n=200, seed=3)
    assert X.mean(axis=1) == pytest.approx(mean, abs=1e-12)
    assert np.cov(X, ddof=1) == pytest.approx(P, abs=1e-12)

    Xa = enkf_update(X, y, H, R, deterministic=True)

    # Textbook Kalman filter.
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_kf = mean + K @ (y - H @ mean)
    P_kf = (np.eye(2) - K @ H) @ P

    assert Xa.mean(axis=1) == pytest.approx(x_kf, abs=1e-10)
    assert np.cov(Xa, ddof=1) == pytest.approx(P_kf, abs=1e-10)


def test_stochastic_enkf_approaches_the_kalman_filter_for_large_ensembles():
    mean = np.array([0.0])
    P = np.array([[1.0]])
    H = np.array([[1.0]])
    R = np.array([[0.5]])
    y = np.array([1.0])
    X = _ensemble_with_exact_moments(mean, P, n=40_000, seed=5)
    Xa = enkf_update(X, y, H, R, deterministic=False, rng=np.random.default_rng(7))
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_kf = (mean + K @ (y - H @ mean)).item()
    P_kf = ((np.eye(1) - K @ H) @ P).item()
    assert float(Xa.mean()) == pytest.approx(x_kf, abs=0.02)
    assert float(np.var(Xa, ddof=1)) == pytest.approx(P_kf, rel=0.05)


def test_enkf_update_rejects_a_degenerate_ensemble():
    with pytest.raises(ValueError, match="at least two members"):
        enkf_update(np.zeros((1, 1)), np.array([0.0]), np.array([[1.0]]), np.array([[1.0]]))


def test_inflation_preserves_the_mean_and_scales_the_spread():
    X = np.array([[1.0, 2.0, 3.0, 4.0]])
    Y = inflate(X, 2.0)
    assert Y.mean() == pytest.approx(X.mean())
    assert Y.std(ddof=1) == pytest.approx(2.0 * X.std(ddof=1))
    assert inflate(X, 1.0) is X


def test_log_enkf_keeps_every_member_positive():
    """A crack depth cannot go negative; the log state guarantees it without clipping."""
    f = LogEnKF(EnKFConfig(n_members=64, seed=1))
    f.initialise(a_mean=2.0e-3, a_cv=0.6)
    assert np.all(f.depths() > 0)
    for _ in range(30):
        f.forecast(lambda a, dt: a + 1e-5 * np.sqrt(a) * dt, 0.5)
        # A strongly negative observation would drive a linear filter below zero.
        f.assimilate(np.log(1.0e-4), 0.2, 1.0)
        assert np.all(f.depths() > 0)


def test_particle_filter_resamples_and_tracks():
    pf = ParticleFilter(n_particles=400, seed=2)
    pf.initialise(2.0e-3, 0.7)
    truth = 2.0e-3
    for _ in range(20):
        truth = truth + 2e-5 * np.sqrt(truth) * 0.5
        pf.forecast(lambda a, dt: a + 2e-5 * np.sqrt(a) * dt, 0.5)
        pf.assimilate(truth, 0.1 * truth, lambda p: p)
    est = np.median(pf.resampled_ensemble())
    assert est == pytest.approx(truth, rel=0.3)
    assert np.all(pf.particles > 0)


def test_systematic_resample_and_ess():
    w = np.array([0.0, 0.0, 1.0, 0.0])
    idx = systematic_resample(w, np.random.default_rng(0))
    assert np.all(idx == 2)
    assert effective_sample_size(w) == pytest.approx(1.0)
    uniform = np.full(10, 0.1)
    assert effective_sample_size(uniform) == pytest.approx(10.0)


# --------------------------------------------------------------- calibration


def test_rank_histogram_is_flat_for_a_calibrated_ensemble():
    rng = np.random.default_rng(9)
    n_times, n_members = 4000, 20
    truth = rng.normal(size=n_times)
    ens = rng.normal(size=(n_times, n_members))
    ranks = rank_histogram(ens, truth)
    assert ranks.size == n_members + 1
    assert ranks.sum() == n_times
    expected = n_times / (n_members + 1)
    assert np.all(np.abs(ranks - expected) < 5 * np.sqrt(expected))


def test_rank_histogram_is_u_shaped_when_underdispersed():
    rng = np.random.default_rng(10)
    n_times, n_members = 3000, 20
    truth = rng.normal(size=n_times)
    ens = rng.normal(scale=0.2, size=(n_times, n_members))  # far too narrow
    ranks = rank_histogram(ens, truth)
    ends = ranks[0] + ranks[-1]
    middle = ranks[1:-1].sum()
    assert ends > middle, "an over-confident ensemble must pile up in the end bins"


def test_pit_is_uniform_when_calibrated():
    rng = np.random.default_rng(11)
    truth = rng.normal(size=3000)
    ens = rng.normal(size=(3000, 50))
    p = pit_values(ens, truth)
    assert p.mean() == pytest.approx(0.5, abs=0.03)
    assert p.std() == pytest.approx(1 / np.sqrt(12), abs=0.03)


def test_crps_reduces_to_absolute_error_for_a_point_forecast():
    truth = np.array([1.0, 2.0, 3.0])
    ens = np.tile(np.array([[0.0], [0.0], [0.0]]), (1, 30))
    assert crps_ensemble(ens, truth) == pytest.approx(np.mean(np.abs(truth)), rel=1e-12)


def test_crps_rewards_a_sharper_correct_forecast():
    rng = np.random.default_rng(12)
    truth = np.zeros(500)
    tight = rng.normal(0, 0.2, size=(500, 60))
    loose = rng.normal(0, 2.0, size=(500, 60))
    assert crps_ensemble(tight, truth) < crps_ensemble(loose, truth)


def test_coverage_matches_the_nominal_level_when_calibrated():
    rng = np.random.default_rng(13)
    truth = rng.normal(size=4000)
    ens = rng.normal(size=(4000, 200))
    assert coverage(ens, truth, 0.90) == pytest.approx(0.90, abs=0.03)


def test_assess_names_overconfidence_plainly():
    rng = np.random.default_rng(14)
    truth = rng.normal(size=1000)
    narrow = rng.normal(scale=0.05, size=(1000, 40))
    r = assess("narrow", narrow, truth)
    assert r.verdict == "MISCALIBRATED - OVERCONFIDENT"
    assert "worse than the status quo" in r.comment

    wide = rng.normal(scale=20.0, size=(1000, 40))
    assert assess("wide", wide, truth).verdict == "MISCALIBRATED - UNDERCONFIDENT"

    good = rng.normal(size=(1000, 40))
    assert assess("good", good, truth).verdict == "CALIBRATED"


def test_no_update_baseline_never_narrows():
    """With no observations the spread can only grow."""
    traj = prior_only(2e-3, 0.5, lambda a, dt: a + 1e-5 * np.sqrt(a) * dt, 0.5, 20, 400, seed=1)
    spread = traj.std(axis=1)
    assert spread[-1] >= spread[0] * 0.999


def test_sn_baseline_is_honestly_unavailable():
    with pytest.raises(DataUnavailable) as exc:
        sn_miner_baseline()
    assert "DNV-RP-C203" in str(exc.value)
    assert exc.value.remedy


# ---------------------------------------------------------------------------
# The abstract's own convergence criterion: within 8 percent of true damage by
# month 18. Testable here, unlike its remaining-life interval, so it is tested.


def _fc(**kw):
    return run_comparison(n_steps=20, n_members=32, n_particles=64, **kw)


def test_convergence_requires_staying_inside_the_band_not_just_touching_it():
    """An estimate that crosses the band on its way elsewhere has not converged.

    Under structural model error a filter can drift through the tolerance and
    out again, and reporting that first crossing as the convergence time would
    flatter it.
    """
    fc = _fc()
    touch = fc.first_within("log-EnKF", 0.08, hold=False)
    stay = fc.first_within("log-EnKF", 0.08, hold=True)
    assert np.isnan(stay) or stay >= touch


def test_a_criterion_the_do_nothing_baseline_also_passes_is_flagged():
    """The 8-percent-by-18-months test must not be quoted as evidence if the
    no-update baseline clears it too."""
    fc = _fc()
    ac = fc.abstract_convergence()
    if ac["met"] and ac["baseline_also_meets"]:
        assert not ac["discriminates"]
    assert set(ac) >= {"met", "baseline_also_meets", "discriminates",
                       "error_at_deadline", "baseline_error_at_deadline"}


def test_the_convergence_numbers_come_from_the_abstract():
    """Defaults must be the paper's own figures, not ones chosen here."""
    fc = _fc()
    ac = fc.abstract_convergence()
    assert ac["tolerance"] == pytest.approx(PAPER.enkf_convergence_error)
    assert ac["by_years"] * 12.0 == pytest.approx(PAPER.enkf_convergence_months)


def test_relative_error_is_zero_when_the_ensemble_sits_on_the_truth():
    fc = _fc()
    fc.ensembles["exact"] = np.repeat(fc.truth[:, None], 4, axis=1)
    assert np.allclose(fc.relative_error("exact"), 0.0)
    assert fc.first_within("exact", 1e-9) == pytest.approx(fc.times_years[0])
