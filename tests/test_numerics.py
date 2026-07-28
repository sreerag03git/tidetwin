"""The hand-rolled numerics stand in for scipy subpackages, so they get tested."""

from __future__ import annotations

import numpy as np
import pytest

from tidetwin.numerics import (
    BilinearGrid,
    bisect_root,
    chi2_cdf,
    ecdf,
    interp1d_linear,
    ks_statistic,
    norm_cdf,
    norm_ppf,
    norm_sf,
    weighted_percentile,
)


def test_bisect_root_finds_known_roots():
    assert bisect_root(lambda x: x**2 - 2.0, 0.0, 2.0) == pytest.approx(np.sqrt(2), abs=1e-13)
    # Clamped-free beam eigenvalue equation.
    r = bisect_root(lambda x: np.cos(x) * np.cosh(x) + 1.0, 1.8, 1.95)
    assert r == pytest.approx(1.8751040687, abs=1e-9)


def test_bisect_root_requires_a_bracket():
    with pytest.raises(ValueError):
        bisect_root(lambda x: x**2 + 1.0, -1.0, 1.0)


def test_normal_distribution_round_trip():
    p = np.array([1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-6])
    assert norm_cdf(norm_ppf(p)) == pytest.approx(p, rel=1e-9)
    assert norm_cdf(0.0) == pytest.approx(0.5)
    # Known quantiles.
    assert norm_ppf(0.975) == pytest.approx(1.959963985, abs=1e-8)
    assert norm_ppf(0.95) == pytest.approx(1.644853627, abs=1e-8)
    # Survival function stays accurate where 1 - cdf would cancel to zero.
    assert norm_sf(8.0) == pytest.approx(6.22096057e-16, rel=1e-6)


def test_chi2_cdf_against_closed_forms():
    # k = 2 is exponential: F(x) = 1 - exp(-x/2).
    x = np.array([0.5, 1.0, 3.0, 7.0])
    assert chi2_cdf(x, 2) == pytest.approx(1.0 - np.exp(-x / 2.0), rel=1e-12)
    # k = 1 is the folded normal: F(x) = 2*Phi(sqrt(x)) - 1.
    assert chi2_cdf(x, 1) == pytest.approx(2.0 * norm_cdf(np.sqrt(x)) - 1.0, rel=1e-10)


def test_weighted_percentile_matches_numpy_when_uniform():
    rng = np.random.default_rng(3)
    v = rng.normal(size=501)
    w = np.ones_like(v)
    q = np.array([5.0, 50.0, 95.0])
    assert weighted_percentile(v, w, q) == pytest.approx(np.percentile(v, q), rel=2e-2)


def test_weighted_percentile_respects_weights():
    v = np.array([0.0, 10.0])
    assert weighted_percentile(v, np.array([1.0, 99.0]), 50.0) == pytest.approx(10.0, abs=0.2)
    assert weighted_percentile(v, np.array([99.0, 1.0]), 50.0) == pytest.approx(0.0, abs=0.2)


def test_ecdf_and_ks_statistic():
    rng = np.random.default_rng(11)
    x = rng.normal(size=4000)
    xs, p = ecdf(x)
    assert np.all(np.diff(xs) >= 0)
    assert p[-1] == pytest.approx(1.0)
    d = ks_statistic(x, norm_cdf)
    # Asymptotic 99% critical value is 1.63/sqrt(n).
    assert d < 1.63 / np.sqrt(x.size)
    # A shifted sample must be rejected.
    assert ks_statistic(x + 1.0, norm_cdf) > 0.2


def test_bilinear_grid_reproduces_a_bilinear_function():
    x = np.array([0.0, 1.0, 2.5, 4.0])
    y = np.array([0.0, 0.5, 2.0])
    f = lambda a, b: 3.0 + 2.0 * a - 1.5 * b + 0.75 * a * b
    z = f(x[:, None], y[None, :])
    g = BilinearGrid(x, y, z)
    xi = np.array([0.3, 1.7, 3.9])
    yi = np.array([0.1, 1.0, 1.9])
    assert g(xi, yi) == pytest.approx(f(xi, yi), rel=1e-12)
    # Nodes reproduce exactly.
    assert g(x[2], y[1]) == pytest.approx(z[2, 1], rel=1e-12)


def test_bilinear_grid_clamps_and_flags_out_of_range():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    z = np.array([[0.0, 1.0], [1.0, 2.0]])
    g = BilinearGrid(x, y, z)
    assert g(5.0, 0.0) == pytest.approx(1.0)  # clamped to x = 1, not extrapolated
    assert bool(g.out_of_range(5.0, 0.0))
    assert not bool(g.out_of_range(0.5, 0.5))
    assert g.bounds == ((0.0, 1.0), (0.0, 1.0))


def test_bilinear_grid_rejects_bad_input():
    with pytest.raises(ValueError):
        BilinearGrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        BilinearGrid(np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.zeros((2, 2)))


def test_interp1d_clamps_by_default_and_extrapolates_on_request():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 2.0, 4.0])
    assert interp1d_linear(x, y, np.array([-1.0, 3.0])) == pytest.approx([0.0, 4.0])
    assert interp1d_linear(x, y, np.array([-1.0, 3.0]), extrapolate=True) == pytest.approx(
        [-2.0, 6.0]
    )
