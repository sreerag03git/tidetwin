"""Calibration diagnostics for probabilistic forecasts.

A filter that converges quickly to a confident wrong answer is worse than no
filter, because it converts an honest "we do not know" into a number an
inspection plan will be built on. Convergence alone therefore proves nothing;
these diagnostics are the ones that decide whether C6 passes.

* **Rank histogram** (Talagrand diagram): where the truth falls among the sorted
  ensemble. Flat means calibrated; U-shaped means under-dispersed (over-confident);
  dome-shaped means over-dispersed.
* **PIT** (probability integral transform): the continuous analogue. Under a
  well-calibrated forecast, PIT values are uniform on [0, 1].
* **CRPS** (continuous ranked probability score): a proper scoring rule, so it
  cannot be improved by misrepresenting uncertainty. Lower is better.
* **Empirical coverage**: the fraction of times the nominal 90 percent interval
  actually contains the truth. It should be 0.90.

References: Hamill, "Interpretation of rank histograms for verifying ensemble
forecasts", Monthly Weather Review 129:550-560, 2001; Gneiting & Raftery,
"Strictly proper scoring rules, prediction, and estimation", JASA 102:359-378,
2007.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..numerics import ks_statistic

__all__ = ["CalibrationReport", "rank_histogram", "pit_values", "crps_ensemble", "coverage", "assess"]


def rank_histogram(ensembles: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Counts of the truth's rank within each ensemble.

    ``ensembles`` is ``(n_times, n_members)``. Returns ``n_members + 1`` bins.
    Ties are broken randomly-free by counting strict inequalities, which is the
    standard treatment for continuous ensembles.
    """
    E = np.atleast_2d(np.asarray(ensembles, float))
    y = np.asarray(truth, float).ravel()
    if E.shape[0] != y.size:
        raise ValueError(f"{E.shape[0]} ensembles but {y.size} truth values")
    ranks = (E < y[:, None]).sum(axis=1)
    return np.bincount(ranks, minlength=E.shape[1] + 1)


def pit_values(ensembles: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Probability integral transform of the truth under each ensemble CDF."""
    E = np.atleast_2d(np.asarray(ensembles, float))
    y = np.asarray(truth, float).ravel()
    return (E < y[:, None]).mean(axis=1)


def crps_ensemble(ensembles: np.ndarray, truth: np.ndarray) -> float:
    """Mean CRPS, using the exact finite-ensemble estimator.

    .. math::
        CRPS = \\frac{1}{m}\\sum_i |x_i - y|
             - \\frac{1}{2m^2}\\sum_i\\sum_j |x_i - x_j|

    (Gneiting & Raftery 2007, Eq. 21). In the same units as the forecast
    variable, and it reduces to the absolute error for a deterministic forecast.
    """
    E = np.atleast_2d(np.asarray(ensembles, float))
    y = np.asarray(truth, float).ravel()
    m = E.shape[1]
    term1 = np.abs(E - y[:, None]).mean(axis=1)
    xs = np.sort(E, axis=1)
    # sum_i sum_j |x_i - x_j| computed in O(m log m) from the sorted values.
    idx = np.arange(m)
    weights = 2 * idx - m + 1
    term2 = (xs * weights).sum(axis=1) / (m * m)
    return float(np.mean(term1 - term2))


def coverage(ensembles: np.ndarray, truth: np.ndarray, level: float = 0.90) -> float:
    """Fraction of truths inside the central interval at ``level``."""
    E = np.atleast_2d(np.asarray(ensembles, float))
    y = np.asarray(truth, float).ravel()
    lo = np.percentile(E, 50 * (1 - level), axis=1)
    hi = np.percentile(E, 50 * (1 + level), axis=1)
    return float(np.mean((y >= lo) & (y <= hi)))


@dataclass(frozen=True)
class CalibrationReport:
    """Everything needed to decide whether an interval can be trusted."""

    name: str
    rmse: float
    crps: float
    coverage_90: float
    mean_interval_width_90: float
    pit: np.ndarray
    ranks: np.ndarray
    pit_ks: float
    verdict: str
    comment: str


def assess(name: str, ensembles: np.ndarray, truth: np.ndarray) -> CalibrationReport:
    """Score a filter run and state plainly whether its intervals are honest."""
    E = np.atleast_2d(np.asarray(ensembles, float))
    y = np.asarray(truth, float).ravel()
    mean = E.mean(axis=1)
    rmse = float(np.sqrt(np.mean((mean - y) ** 2)))
    cov = coverage(E, y, 0.90)
    lo = np.percentile(E, 5, axis=1)
    hi = np.percentile(E, 95, axis=1)
    width = float(np.mean(hi - lo))
    pit = pit_values(E, y)
    ks = ks_statistic(pit, lambda x: np.clip(x, 0.0, 1.0))

    if cov < 0.7:
        verdict = "MISCALIBRATED - OVERCONFIDENT"
        comment = (
            f"The nominal 90 percent interval contains the truth only {cov * 100:.0f} percent of "
            f"the time, with a mean width of {width:.3g}. A narrow interval that is wrong "
            "this often is worse than the status quo: it will be used to defer inspections "
            "that should have gone ahead."
        )
    elif cov > 0.98:
        verdict = "MISCALIBRATED - UNDERCONFIDENT"
        comment = (
            f"Coverage is {cov * 100:.0f} percent against a nominal 90. The interval is honest "
            "but so wide that it carries little decision value over the prior."
        )
    else:
        verdict = "CALIBRATED"
        comment = (
            f"Coverage {cov * 100:.0f} percent against a nominal 90, mean width {width:.3g}, "
            f"CRPS {crps_ensemble(E, y):.4g}."
        )
    return CalibrationReport(
        name=name,
        rmse=rmse,
        crps=crps_ensemble(E, y),
        coverage_90=cov,
        mean_interval_width_90=width,
        pit=pit,
        ranks=rank_histogram(E, y),
        pit_ks=float(ks),
        verdict=verdict,
        comment=comment,
    )
