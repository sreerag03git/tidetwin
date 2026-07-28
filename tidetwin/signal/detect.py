"""Detection statistics and time-to-detection.

The quantity monitored is the M2 strain ratio between the bracketing gauges. A
crack shifts it; the question is how long a record is needed before that shift
is distinguishable from the nuisance floor C3 measured.

The central point this module makes, and the reason it does not simply divide
signal by noise, is that **coherent averaging only helps against the part of the
nuisance that is random between records.** The C3 budget contains both kinds:

* *Averageable*: FBG readout noise, wave-induced offsets, short-period wind
  drift. These are approximately independent between successive records and fall
  as ``1/sqrt(N)``.
* *Non-averageable*: marine growth accretion, scour, sensor drift, seasonal
  stratification. These are slowly varying biases. Averaging longer does not
  reduce them; it only makes them look like a trend, and a trend is exactly what
  a growing crack also looks like.

So the detection floor tends to the non-averageable component, not to zero. The
``sqrt(N)`` gain curve is reported against the theoretical one so the divergence
is visible rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..numerics import norm_cdf, norm_ppf, weighted_percentile

__all__ = [
    "DetectionModel",
    "cusum",
    "glrt_statistic",
    "roc_curve",
    "pod_curve",
    "detection_time_cdf",
    "coherent_gain_curve",
]


@dataclass(frozen=True)
class DetectionModel:
    """Noise decomposition for the ratio statistic.

    ``sigma_random`` and ``sigma_systematic`` are standard deviations of the
    strain ratio, in the same units as the ratio itself.
    """

    sigma_random: float
    sigma_systematic: float
    baseline_ratio: float
    false_alarm_rate: float = 0.01

    @property
    def sigma_total(self) -> float:
        return float(np.hypot(self.sigma_random, self.sigma_systematic))

    def sigma_after(self, n_records: float) -> float:
        """Effective noise after averaging ``n_records`` independent records."""
        n = max(float(n_records), 1.0)
        return float(np.hypot(self.sigma_random / np.sqrt(n), self.sigma_systematic))

    def detectable_shift(self, n_records: float, power: float = 0.5) -> float:
        """Smallest ratio shift detectable at the configured false-alarm rate."""
        z_fa = float(norm_ppf(1.0 - self.false_alarm_rate))
        z_pd = float(norm_ppf(power))
        return (z_fa + z_pd) * self.sigma_after(n_records)


def cusum(x: np.ndarray, target: float, slack: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided CUSUM control statistic.

    ``S+_i = max(0, S+_{i-1} + (x_i - target) - k)`` and its mirror, with ``k``
    the slack (Page, "Continuous inspection schemes", Biometrika 41:100-115,
    1954). Slack is conventionally half the shift to be detected.
    """
    x = np.asarray(x, float).ravel()
    sp = np.zeros(x.size)
    sm = np.zeros(x.size)
    for i in range(1, x.size):
        sp[i] = max(0.0, sp[i - 1] + (x[i] - target) - slack)
        sm[i] = min(0.0, sm[i - 1] + (x[i] - target) + slack)
    return sp, sm


def glrt_statistic(x: np.ndarray, sigma: float) -> np.ndarray:
    """Generalised likelihood ratio for an unknown change point and shift.

    At each candidate change point ``k`` the statistic compares the fit of a
    constant mean against a step, maximised over the pre- and post-means. Peaks
    where a step is most plausible.
    """
    x = np.asarray(x, float).ravel()
    n = x.size
    if n < 4 or sigma <= 0:
        return np.zeros(n)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    total = csum[-1]
    out = np.zeros(n)
    for k in range(1, n):
        m1 = csum[k] / k
        m2 = (total - csum[k]) / (n - k)
        out[k] = (k * (n - k) / n) * (m2 - m1) ** 2 / (2.0 * sigma**2)
    return out


def roc_curve(
    healthy: np.ndarray, damaged: np.ndarray, n_points: int = 200
) -> tuple[np.ndarray, np.ndarray, float]:
    """Empirical ROC and area under it, from two sample populations."""
    h = np.asarray(healthy, float).ravel()
    d = np.asarray(damaged, float).ravel()
    lo = min(h.min(), d.min())
    hi = max(h.max(), d.max())
    thr = np.linspace(lo, hi, n_points)
    sign = 1.0 if d.mean() >= h.mean() else -1.0
    pfa = np.array([np.mean(sign * h >= sign * t) for t in thr])
    pd = np.array([np.mean(sign * d >= sign * t) for t in thr])
    order = np.argsort(pfa)
    auc = float(np.trapezoid(pd[order], pfa[order]))
    return pfa[order], pd[order], abs(auc)


def pod_curve(
    crack_sizes: np.ndarray,
    signature_fraction: np.ndarray,
    model: DetectionModel,
    n_records: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Probability of detection against crack size, and ``a90`` / ``a90/95``.

    The signal for a crack of size ``a`` is ``signature_fraction(a) * baseline``,
    tested against the effective noise after averaging. POD follows from the
    normal tail:

    .. math:: POD(a) = \\Phi\\!\\left(\\frac{\\Delta(a)}{\\sigma_{eff}} - z_{fa}\\right)

    ``a90`` is the size detected with 90 percent probability. ``a90/95`` adds a
    95 percent lower confidence bound, approximated here by evaluating POD with
    the noise inflated by its own sampling uncertainty; where the curve never
    reaches 0.9 both are reported as ``inf``, which is the honest answer for a
    method that cannot reach the threshold at any crack size on the grid.
    """
    a = np.asarray(crack_sizes, float).ravel()
    sig = np.abs(np.asarray(signature_fraction, float).ravel()) * abs(model.baseline_ratio)
    sigma = model.sigma_after(n_records)
    z_fa = float(norm_ppf(1.0 - model.false_alarm_rate))
    pod = np.asarray(norm_cdf(sig / max(sigma, 1e-300) - z_fa), float)

    def first_crossing(curve: np.ndarray) -> float:
        idx = np.flatnonzero(curve >= 0.9)
        if idx.size == 0:
            return float("inf")
        i = int(idx[0])
        if i == 0:
            return float(a[0])
        x0, x1 = a[i - 1], a[i]
        y0, y1 = curve[i - 1], curve[i]
        return float(x0 + (0.9 - y0) * (x1 - x0) / max(y1 - y0, 1e-12))

    a90 = first_crossing(pod)
    pod_lower = np.asarray(norm_cdf(sig / max(sigma * 1.2, 1e-300) - z_fa), float)
    a90_95 = first_crossing(pod_lower)
    return pod, a90, a90_95


def detection_time_cdf(
    model: DetectionModel,
    signature: float,
    record_hours: float = 24.0 * 14.0,
    max_records: int = 200,
    n_trials: int = 2000,
    seed: int = 20260728,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Monte Carlo distribution of time to detect a step change of ``signature``.

    Each trial accumulates records, averages them, and declares detection when
    the running mean departs from the baseline by more than the false-alarm
    threshold. Systematic noise is drawn once per trial and never averages away,
    which is what makes the upper tail heavy.

    Returns ``(times_days, cdf, percentiles)``. A trial that never detects within
    ``max_records`` contributes ``inf``, and the reported percentiles say so
    rather than silently truncating.
    """
    rng = np.random.default_rng(seed)
    z_fa = float(norm_ppf(1.0 - model.false_alarm_rate))
    times = np.full(n_trials, np.inf)
    for i in range(n_trials):
        bias = rng.normal(0.0, model.sigma_systematic)
        running = 0.0
        for n in range(1, max_records + 1):
            running += rng.normal(signature, model.sigma_random)
            mean = running / n + bias
            if abs(mean) > z_fa * model.sigma_after(n):
                times[i] = n * record_hours / 24.0
                break
    finite = times[np.isfinite(times)]
    xs = np.sort(finite)
    cdf = np.arange(1, xs.size + 1) / n_trials
    pct = {
        "p05": float(weighted_percentile(times, np.ones_like(times), 5.0)),
        "p50": float(weighted_percentile(times, np.ones_like(times), 50.0)),
        "p95": float(weighted_percentile(times, np.ones_like(times), 95.0)),
        "never_detected_fraction": float(np.mean(~np.isfinite(times))),
    }
    return xs, cdf, pct


def coherent_gain_curve(
    model: DetectionModel, n_records: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Achieved noise reduction against the ideal ``1/sqrt(N)`` curve.

    Returns ``(theoretical, achieved)`` as noise relative to a single record. The
    two coincide only while the random component dominates; once the systematic
    floor is reached, the achieved curve flattens and no further averaging helps.
    """
    n = np.asarray(n_records, float)
    sigma0 = model.sigma_after(1.0)
    theoretical = model.sigma_random / np.sqrt(n) / sigma0
    achieved = np.array([model.sigma_after(x) for x in n]) / sigma0
    return theoretical, achieved
