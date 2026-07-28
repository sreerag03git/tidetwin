"""Least-squares tidal harmonic regression with confidence intervals.

Fits

.. math::
    y(t) = c_0 + c_1 t + \\sum_k \\left[ a_k \\cos(\\omega_k t)
                                       + b_k \\sin(\\omega_k t) \\right]

by ordinary least squares, then converts each ``(a, b)`` pair to amplitude and
phase. This is the classical harmonic analysis of Godin (*The Analysis of
Tides*, University of Toronto Press, 1972) and Foreman (IOS Report 78-6, 1978),
without nodal corrections: over the record lengths this application considers
(weeks to a few months) the 18.6-year nodal modulation is a slowly varying
scale factor common to both sensors of a bracketing pair, so it cancels from
their ratio.

Amplitude and phase confidence intervals come from the delta method applied to
the OLS covariance of ``(a, b)``. Two things about that are worth being explicit
about, because they set how far C4's detection times can be trusted:

* OLS assumes independent residuals. Real strain residuals are strongly
  autocorrelated (wind, waves, swell), so the naive interval is optimistic. An
  effective-sample-size correction based on the residual autocorrelation is
  applied and reported.
* Two constituents separated by less than the Rayleigh limit ``1/T`` cannot be
  resolved, however good the signal-to-noise ratio. :func:`rayleigh_check`
  reports which pairs in a requested set are unresolvable at a given record
  length, and the fit refuses to pretend otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..numerics import norm_ppf

__all__ = ["HarmonicFit", "fit_harmonics", "rayleigh_check", "effective_sample_size"]


@dataclass(frozen=True)
class HarmonicFit:
    """Result of a harmonic regression."""

    names: tuple[str, ...]
    omega: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray  # radians, y = A cos(omega t - phase)
    amplitude_se: np.ndarray
    phase_se: np.ndarray
    mean: float
    trend: float
    residual_std: float
    n_samples: int
    n_effective: float
    record_length_s: float
    unresolved: tuple[tuple[str, str], ...] = ()

    def index(self, name: str) -> int:
        return self.names.index(name.upper())

    def amplitude_of(self, name: str) -> float:
        return float(self.amplitude[self.index(name)])

    def ci(self, name: str, level: float = 0.95) -> tuple[float, float]:
        """Two-sided confidence interval on the amplitude."""
        i = self.index(name)
        z = float(norm_ppf(0.5 + 0.5 * level))
        a, se = float(self.amplitude[i]), float(self.amplitude_se[i])
        return a - z * se, a + z * se

    def snr(self, name: str) -> float:
        """Amplitude divided by its standard error."""
        i = self.index(name)
        se = float(self.amplitude_se[i])
        return float(self.amplitude[i] / se) if se > 0 else np.inf


def effective_sample_size(residuals: np.ndarray) -> float:
    """Sample size adjusted for lag-1 residual autocorrelation.

    ``n_eff = n (1 - r) / (1 + r)`` for an AR(1) residual with lag-1
    correlation ``r`` (Bartlett 1935; the standard correction used in
    geophysical trend analysis). Negative ``r`` is clipped to zero so the
    correction can only ever widen intervals, never narrow them.
    """
    r = np.asarray(residuals, float)
    n = r.size
    if n < 3:
        return float(n)
    r = r - r.mean()
    denom = float(np.dot(r, r))
    if denom <= 0:
        return float(n)
    rho = float(np.dot(r[:-1], r[1:]) / denom)
    rho = float(np.clip(rho, 0.0, 0.99))
    return float(max(2.0, n * (1.0 - rho) / (1.0 + rho)))


def rayleigh_check(
    names: tuple[str, ...], omega: np.ndarray, record_length_s: float
) -> tuple[tuple[str, str], ...]:
    """Constituent pairs closer than the Rayleigh resolution limit.

    Two frequencies are separable only if ``|f1 - f2| >= 1/T``. Below that they
    are aliases of one another over the record and no amount of signal-to-noise
    recovers them separately.
    """
    if record_length_s <= 0:
        return tuple((names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names)))
    limit = 1.0 / record_length_s
    f = np.asarray(omega, float) / (2.0 * np.pi)
    out: list[tuple[str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(f[i] - f[j]) < limit:
                out.append((names[i], names[j]))
    return tuple(out)


def fit_harmonics(
    t: np.ndarray,
    y: np.ndarray,
    names: tuple[str, ...],
    omega: np.ndarray,
    include_trend: bool = True,
    strict_rayleigh: bool = False,
) -> HarmonicFit:
    """Fit constituents at known frequencies to a time series.

    Parameters
    ----------
    t
        Times in seconds, need not be uniformly spaced.
    y
        Observations (strain, elevation, ...).
    names, omega
        Constituent labels and angular frequencies in rad/s.
    include_trend
        Fit a linear drift alongside the mean. Leave on when the series may
        contain sensor drift, which for an FBG pair it always may.
    strict_rayleigh
        Raise if any requested pair is unresolvable at this record length,
        instead of fitting them and reporting the conflict.
    """
    t = np.asarray(t, float).ravel()
    y = np.asarray(y, float).ravel()
    if t.size != y.size:
        raise ValueError(f"t has {t.size} samples, y has {y.size}")
    omega = np.asarray(omega, float).ravel()
    if omega.size != len(names):
        raise ValueError("names and omega must be the same length")
    n = t.size
    T = float(t.max() - t.min()) if n > 1 else 0.0

    unresolved = rayleigh_check(tuple(names), omega, T)
    if unresolved and strict_rayleigh:
        pairs = ", ".join(f"{a}/{b}" for a, b in unresolved)
        raise ValueError(
            f"record of {T / 86400:.2f} d cannot resolve {pairs}; "
            f"needs at least {_days_needed(omega, names, unresolved):.1f} d"
        )

    cols = [np.ones(n)]
    labels = ["mean"]
    if include_trend:
        cols.append(t - t.mean())
        labels.append("trend")
    for w in omega:
        cols.append(np.cos(w * t))
        cols.append(np.sin(w * t))
    X = np.column_stack(cols)
    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"{n} samples cannot determine {X.shape[1]} parameters; lengthen the record"
        )

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(n - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    n_eff = effective_sample_size(resid)
    inflate = float(n / max(n_eff, 1.0))
    XtX_inv = np.linalg.pinv(X.T @ X)
    cov = s2 * inflate * XtX_inv

    k0 = 2 if include_trend else 1
    a = coef[k0::2][: len(names)]
    b = coef[k0 + 1 :: 2][: len(names)]
    amp = np.hypot(a, b)
    pha = np.arctan2(b, a)

    # Delta method: A = sqrt(a^2+b^2), phi = atan2(b, a).
    amp_se = np.zeros_like(amp)
    pha_se = np.zeros_like(pha)
    for k in range(len(names)):
        ia, ib = k0 + 2 * k, k0 + 2 * k + 1
        caa, cbb, cab = cov[ia, ia], cov[ib, ib], cov[ia, ib]
        A = max(amp[k], 1e-300)
        dA = np.array([a[k] / A, b[k] / A])
        dphi = np.array([-b[k] / A**2, a[k] / A**2])
        C = np.array([[caa, cab], [cab, cbb]])
        amp_se[k] = np.sqrt(max(dA @ C @ dA, 0.0))
        pha_se[k] = np.sqrt(max(dphi @ C @ dphi, 0.0))

    return HarmonicFit(
        names=tuple(n_.upper() for n_ in names),
        omega=omega,
        amplitude=amp,
        phase=pha,
        amplitude_se=amp_se,
        phase_se=pha_se,
        mean=float(coef[0]),
        trend=float(coef[1]) if include_trend else 0.0,
        residual_std=float(np.sqrt(s2)),
        n_samples=n,
        n_effective=n_eff,
        record_length_s=T,
        unresolved=unresolved,
    )


def _days_needed(omega: np.ndarray, names: tuple[str, ...], pairs) -> float:
    f = np.asarray(omega, float) / (2.0 * np.pi)
    idx = {n: i for i, n in enumerate(names)}
    worst = 0.0
    for a, b in pairs:
        df = abs(f[idx[a]] - f[idx[b]])
        if df > 0:
            worst = max(worst, 1.0 / df)
    return worst / 86400.0
