"""Small numerical utilities implemented on numpy and ``scipy.special`` only.

The project deliberately avoids ``scipy.optimize``, ``scipy.interpolate``,
``scipy.stats``, ``scipy.signal`` and ``scipy.integrate``. Two reasons:

1. The reference development machine has an OS Application Control policy that
   blocks one of the shared libraries those subpackages load, so code depending
   on them cannot be run or verified there.
2. Streamlit Community Cloud gives 1 GB; a narrower import surface starts faster
   and has fewer ways to fail on a platform we do not control.

Everything here is elementary and covered by ``tests/test_numerics.py``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
# scipy.special is imported inside the four functions below, not here, so that
# importing this module - which happens on the deployed app's cold start,
# transitively, long before any statistics are needed - does not load scipy.
# Re-importing inside a function is a dict lookup after the first call.

__all__ = [
    "bisect_root",
    "norm_cdf",
    "norm_ppf",
    "norm_sf",
    "chi2_cdf",
    "interp1d_linear",
    "BilinearGrid",
    "weighted_percentile",
    "ecdf",
    "ks_statistic",
    "trapezoid",
]


# ------------------------------------------------------------- root finding


def bisect_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    xtol: float = 1e-14,
    max_iter: int = 200,
) -> float:
    """Bisection root of ``f`` on the bracket ``[a, b]``.

    Bisection rather than Brent: the brackets used in this project are narrow and
    the functions are smooth and monotone across them, so the extra machinery
    buys nothing, while the convergence guarantee here is unconditional.
    """
    fa, fb = float(f(a)), float(f(b))
    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)
    if fa * fb > 0.0:
        raise ValueError(f"root not bracketed on [{a}, {b}]: f(a)={fa:g}, f(b)={fb:g}")
    lo, hi = float(a), float(b)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = float(f(mid))
        if fm == 0.0 or 0.5 * (hi - lo) < xtol:
            return mid
        if fa * fm < 0.0:
            hi = mid
        else:
            lo, fa = mid, fm
    return 0.5 * (lo + hi)


# ----------------------------------------------------------- distributions


def norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF."""
    from scipy.special import ndtr
    return ndtr(x)


def norm_sf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal survival function, ``1 - Phi(x)``, accurate in the tail."""
    from scipy.special import ndtr
    return ndtr(-np.asarray(x))


def norm_ppf(p: np.ndarray | float) -> np.ndarray | float:
    """Standard normal quantile function."""
    from scipy.special import ndtri
    return ndtri(p)


def chi2_cdf(x: np.ndarray | float, df: int) -> np.ndarray | float:
    """Chi-squared CDF via the regularised lower incomplete gamma function.

    ``F(x; k) = P(k/2, x/2)`` (Abramowitz & Stegun 26.4.19).
    """
    from scipy.special import gammainc
    return gammainc(0.5 * df, 0.5 * np.asarray(x, dtype=float))


# ---------------------------------------------------------------- sampling


def weighted_percentile(
    values: np.ndarray, weights: np.ndarray, q: np.ndarray | float
) -> np.ndarray:
    """Weighted percentile(s), ``q`` in [0, 100].

    Uses the cumulative-weight midpoint convention, which reduces to numpy's
    linear interpolation when all weights are equal.
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if v.size != w.size:
        raise ValueError("values and weights must be the same length")
    order = np.argsort(v)
    v, w = v[order], w[order]
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    cw = (np.cumsum(w) - 0.5 * w) / total
    return np.interp(np.asarray(q, dtype=float) / 100.0, cw, v)


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF: sorted values and their cumulative probabilities."""
    xs = np.sort(np.asarray(x, dtype=float).ravel())
    return xs, np.arange(1, xs.size + 1) / xs.size


def ks_statistic(x: np.ndarray, cdf: Callable[[np.ndarray], np.ndarray]) -> float:
    """One-sample Kolmogorov-Smirnov statistic against a reference CDF."""
    xs, _ = ecdf(x)
    n = xs.size
    f = np.asarray(cdf(xs), dtype=float)
    d_plus = np.max(np.arange(1, n + 1) / n - f)
    d_minus = np.max(f - np.arange(0, n) / n)
    return float(max(d_plus, d_minus))


# ----------------------------------------------------------- interpolation


def interp1d_linear(
    x: np.ndarray, y: np.ndarray, xi: np.ndarray, extrapolate: bool = False
) -> np.ndarray:
    """Linear interpolation with explicit control over out-of-range behaviour.

    ``extrapolate=False`` clamps to the end values, which is the safe default for
    a physical lookup table: an interpolator must not invent behaviour outside
    the range its source data covers. Callers that care must check the range
    themselves and report it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xi = np.asarray(xi, dtype=float)
    if not extrapolate:
        return np.interp(xi, x, y)
    out = np.interp(xi, x, y)
    lo = xi < x[0]
    hi = xi > x[-1]
    if np.any(lo):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        out = np.where(lo, y[0] + slope * (xi - x[0]), out)
    if np.any(hi):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        out = np.where(hi, y[-1] + slope * (xi - x[-1]), out)
    return out


class BilinearGrid:
    """Bilinear interpolation over a regular (possibly non-uniform) 2D grid.

    Replaces ``scipy.interpolate.RegularGridInterpolator`` for the crack-to-LJF
    surface lookup. Queries outside the grid are clamped to the boundary and
    flagged, never extrapolated: the surface is only valid over the crack
    geometries it was computed for.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        if self.z.shape != (self.x.size, self.y.size):
            raise ValueError(
                f"z has shape {self.z.shape}, expected {(self.x.size, self.y.size)}"
            )
        if np.any(np.diff(self.x) <= 0) or np.any(np.diff(self.y) <= 0):
            raise ValueError("grid axes must be strictly increasing")

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (float(self.x[0]), float(self.x[-1])), (float(self.y[0]), float(self.y[-1]))

    def out_of_range(self, xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float)
        yi = np.asarray(yi, dtype=float)
        return (
            (xi < self.x[0]) | (xi > self.x[-1]) | (yi < self.y[0]) | (yi > self.y[-1])
        )

    def __call__(self, xi: np.ndarray | float, yi: np.ndarray | float) -> np.ndarray:
        xq = np.clip(np.asarray(xi, dtype=float), self.x[0], self.x[-1])
        yq = np.clip(np.asarray(yi, dtype=float), self.y[0], self.y[-1])
        i = np.clip(np.searchsorted(self.x, xq, side="right") - 1, 0, self.x.size - 2)
        j = np.clip(np.searchsorted(self.y, yq, side="right") - 1, 0, self.y.size - 2)
        x0, x1 = self.x[i], self.x[i + 1]
        y0, y1 = self.y[j], self.y[j + 1]
        tx = (xq - x0) / (x1 - x0)
        ty = (yq - y0) / (y1 - y0)
        z00 = self.z[i, j]
        z10 = self.z[i + 1, j]
        z01 = self.z[i, j + 1]
        z11 = self.z[i + 1, j + 1]
        return (
            z00 * (1 - tx) * (1 - ty)
            + z10 * tx * (1 - ty)
            + z01 * (1 - tx) * ty
            + z11 * tx * ty
        )


def trapezoid(y: np.ndarray, x: np.ndarray | None = None, dx: float = 1.0) -> float:
    """Trapezoidal integral (numpy's, wrapped so the import site is one place)."""
    return float(np.trapezoid(np.asarray(y, dtype=float), x=x, dx=dx))
