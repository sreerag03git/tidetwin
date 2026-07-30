"""Harmonic regression, aliasing and detection statistics."""

from __future__ import annotations

import numpy as np
import pytest

from tidetwin.loads.tides import constituent_frequency
from tidetwin.signal.aliasing import aliasing_table, separation_days
from tidetwin.signal.detect import (
    DetectionModel,
    coherent_gain_curve,
    cusum,
    detection_time_cdf,
    glrt_statistic,
    pod_curve,
    roc_curve,
)
from tidetwin.signal.harmonic import (
    effective_sample_size,
    fit_harmonics,
    harmonic_amplitude_phase,
    rayleigh_check,
)
from tidetwin.signal.quadrature import decompose

NAMES = ("M2", "S2", "N2", "K1", "O1")
OM = np.array([float(constituent_frequency(n).value) for n in NAMES])


def _synthetic(t, amps, phases, mean=0.0, trend=0.0):
    y = np.full_like(t, mean) + trend * (t - t.mean())
    for A, p, w in zip(amps, phases, OM):
        y = y + A * np.cos(w * t - p)
    return y


def test_harmonic_regression_recovers_known_constituents():
    """With noiseless synthetic data the fit must be exact to near machine precision."""
    t = np.arange(0.0, 60.0 * 86400.0, 600.0)
    amps = np.array([1.0, 0.4, 0.2, 0.5, 0.3])
    phases = np.array([0.3, 1.2, 2.5, 4.0, 5.5])
    y = _synthetic(t, amps, phases, mean=7.0, trend=1e-8)

    f = fit_harmonics(t, y, NAMES, OM)
    assert f.amplitude == pytest.approx(amps, rel=1e-9, abs=1e-12)
    # Phase is recovered modulo 2 pi.
    dphi = np.angle(np.exp(1j * (f.phase - phases)))
    assert dphi == pytest.approx(np.zeros_like(dphi), abs=1e-9)
    assert f.mean == pytest.approx(7.0, abs=1e-9)
    assert f.trend == pytest.approx(1e-8, rel=1e-6)
    # Residual is at the conditioning limit of an 8640 x 12 least-squares solve
    # on a unit-amplitude signal, not a modelling error.
    assert f.residual_std < 1e-8


def test_harmonic_regression_uncertainty_shrinks_with_record_length():
    """Amplitude standard error must fall roughly as 1/sqrt(N)."""
    rng = np.random.default_rng(4)
    ses = []
    for days in (30.0, 120.0):
        t = np.arange(0.0, days * 86400.0, 600.0)
        y = _synthetic(t, [1.0, 0.4, 0.2, 0.5, 0.3], [0.3, 1.2, 2.5, 4.0, 5.5])
        y = y + rng.normal(0.0, 0.1, size=t.size)
        f = fit_harmonics(t, y, NAMES, OM)
        ses.append(f.amplitude_se[f.index("M2")])
    assert ses[1] < ses[0]
    assert ses[0] / ses[1] == pytest.approx(2.0, rel=0.35)


def test_rayleigh_criterion_flags_unresolvable_pairs():
    """M2 and S2 need about 14.77 days; below that they cannot be separated."""
    need = separation_days("M2", "S2")
    assert need == pytest.approx(14.765, rel=1e-3)
    short = rayleigh_check(NAMES, OM, 7.0 * 86400.0)
    assert ("M2", "S2") in short
    long = rayleigh_check(("M2", "S2"), OM[:2], 30.0 * 86400.0)
    assert long == ()


def test_short_record_fit_reports_the_conflict_or_refuses():
    t = np.arange(0.0, 5.0 * 86400.0, 600.0)
    y = _synthetic(t, [1.0, 0.4, 0.2, 0.5, 0.3], [0.0] * 5)
    f = fit_harmonics(t, y, NAMES, OM)
    assert f.unresolved, "a 5 day record cannot resolve these constituents"
    with pytest.raises(ValueError, match="cannot resolve"):
        fit_harmonics(t, y, NAMES, OM, strict_rayleigh=True)


def test_too_few_samples_is_an_error_not_a_fit():
    t = np.linspace(0.0, 3600.0, 5)
    with pytest.raises(ValueError, match="cannot determine"):
        fit_harmonics(t, np.zeros_like(t), NAMES, OM)


def test_effective_sample_size_penalises_autocorrelation():
    rng = np.random.default_rng(2)
    white = rng.normal(size=5000)
    assert effective_sample_size(white) == pytest.approx(5000, rel=0.1)
    red = np.cumsum(rng.normal(size=5000))  # strongly autocorrelated
    assert effective_sample_size(red) < 500


def test_solar_s2_and_tidal_s2_are_exactly_coincident():
    """The whole reason M2 must be the carrier."""
    a = aliasing_table(365.0)
    assert not np.isfinite(a.solar_s2_days)
    assert "no record length separates them" in a.conclusion
    assert a.resolved("M2", "S2")


def test_quadrature_split_separates_in_phase_from_quadrature():
    t = np.arange(0.0, 40.0 * 86400.0, 600.0)
    w = OM[0]
    elevation = np.cos(w * t)
    # A pure quadrature (drag-like) strain signal.
    strain = 3.0 * np.cos(w * t - np.pi / 2)
    s = decompose(t, strain, elevation, "M2")
    assert s.total == pytest.approx(3.0, rel=1e-6)
    assert abs(s.in_phase) < 1e-6
    assert abs(s.quadrature) == pytest.approx(3.0, rel=1e-6)
    assert s.drag_fraction == pytest.approx(1.0, rel=1e-6)
    # A pure in-phase (buoyancy-like) signal.
    s2 = decompose(t, 2.0 * np.cos(w * t), elevation, "M2")
    assert abs(s2.in_phase) == pytest.approx(2.0, rel=1e-6)
    assert abs(s2.quadrature) < 1e-6


# ------------------------------------------------------------------ detection


def test_systematic_noise_sets_a_floor_that_averaging_cannot_cross():
    """The central adversarial point of C4."""
    m = DetectionModel(sigma_random=1.0, sigma_systematic=0.2, baseline_ratio=2.0)
    assert m.sigma_after(1) == pytest.approx(np.hypot(1.0, 0.2))
    assert m.sigma_after(10_000) == pytest.approx(0.2, rel=2e-3)
    # Never below the systematic floor, however long the average.
    assert m.sigma_after(1e9) >= 0.2

    th, ach = coherent_gain_curve(m, np.array([1, 10, 100, 1000]))
    assert th[-1] < ach[-1], "achieved gain must fall short of the ideal once the floor bites"
    assert ach[-1] > 0.15


def test_no_systematic_noise_recovers_the_sqrt_n_law():
    m = DetectionModel(sigma_random=1.0, sigma_systematic=0.0, baseline_ratio=1.0)
    th, ach = coherent_gain_curve(m, np.array([1, 4, 16, 64]))
    assert ach == pytest.approx(th, rel=1e-12)
    assert ach == pytest.approx(1.0 / np.sqrt([1, 4, 16, 64]), rel=1e-12)


def test_cusum_responds_to_a_step_and_ignores_noise():
    rng = np.random.default_rng(5)
    flat = rng.normal(0.0, 1.0, 400)
    stepped = flat.copy()
    stepped[200:] += 4.0
    sp_flat, _ = cusum(flat, 0.0, 0.5)
    sp_step, _ = cusum(stepped, 0.0, 0.5)
    assert sp_step.max() > 5 * max(sp_flat.max(), 1e-9)


def test_glrt_peaks_near_the_true_change_point():
    rng = np.random.default_rng(6)
    x = rng.normal(0.0, 0.3, 300)
    x[180:] += 2.0
    g = glrt_statistic(x, 0.3)
    assert 160 <= int(np.argmax(g)) <= 200


def test_roc_auc_is_half_for_identical_populations_and_one_when_separated():
    rng = np.random.default_rng(8)
    a = rng.normal(0, 1, 3000)
    b = rng.normal(0, 1, 3000)
    _, _, auc_same = roc_curve(a, b)
    assert auc_same == pytest.approx(0.5, abs=0.05)
    _, _, auc_sep = roc_curve(a, rng.normal(10, 1, 3000))
    assert auc_sep > 0.99


def test_pod_curve_is_monotonic_and_reports_infinity_when_unreachable():
    m = DetectionModel(sigma_random=0.01, sigma_systematic=0.0, baseline_ratio=2.0)
    a = np.linspace(1e-3, 20e-3, 40)
    sig = a / 0.05  # signature grows with crack depth
    pod, a90, a90_95 = pod_curve(a, sig, m)
    assert np.all(np.diff(pod) >= -1e-12)
    assert np.isfinite(a90) and np.isfinite(a90_95)
    assert a90_95 >= a90, "the confidence-bounded size cannot be smaller"

    # A method whose signal never clears the noise must say so, not extrapolate.
    weak = DetectionModel(sigma_random=1e3, sigma_systematic=1e3, baseline_ratio=2.0)
    _pod, a90_weak, _ = pod_curve(a, sig, weak)
    assert not np.isfinite(a90_weak)


def test_detection_time_reports_trials_that_never_detect():
    m = DetectionModel(sigma_random=0.5, sigma_systematic=0.5, baseline_ratio=2.0)
    _xs, _cdf, pct = detection_time_cdf(m, signature=1e-6, max_records=30, n_trials=300, seed=1)
    assert pct["never_detected_fraction"] > 0.5
    assert not np.isfinite(pct["p50"])

    _xs, _cdf, pct2 = detection_time_cdf(
        m, signature=5.0, max_records=30, n_trials=300, seed=1
    )
    assert pct2["never_detected_fraction"] < 0.2
    assert np.isfinite(pct2["p50"])
    assert pct2["p05"] <= pct2["p50"] <= pct2["p95"]


def test_amplitude_phase_fast_path_is_byte_identical_to_the_full_fit():
    """The Monte Carlo fast path must return exactly what fit_harmonics does.

    It exists only to skip the standard-error machinery the nuisance budget and
    the rosette never read; the amplitude and phase it computes come from the
    identical cached design matrix and the identical lstsq call, so they must
    match to the last bit. Anything less would mean the ledger changed.
    """
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 14 * 86400.0, 1800.0)
    om = np.array([float(constituent_frequency("M2").value)])
    for _ in range(200):
        y = rng.normal(size=t.size) * 1e-6 + 2e-6 * np.cos(om[0] * t - 0.4)
        f = fit_harmonics(t, y, ("M2",), om)
        amp, pha = harmonic_amplitude_phase(t, y, om)
        assert amp[0] == f.amplitude[0]
        assert pha[0] == f.phase[0]


def test_cached_design_matrix_does_not_confuse_different_grids():
    """The cache is keyed on the exact grid bytes, so two grids never collide."""
    a = np.arange(0.0, 14 * 86400.0, 1800.0)
    b = np.arange(0.0, 7 * 86400.0, 1800.0)  # different length and span
    om = np.array([float(constituent_frequency("M2").value)])
    ya = np.cos(om[0] * a)
    yb = np.cos(om[0] * b)
    assert harmonic_amplitude_phase(a, ya, om)[0][0] == pytest.approx(
        fit_harmonics(a, ya, ("M2",), om).amplitude[0])
    assert harmonic_amplitude_phase(b, yb, om)[0][0] == pytest.approx(
        fit_harmonics(b, yb, ("M2",), om).amplitude[0])
