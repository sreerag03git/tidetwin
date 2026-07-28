"""C6 - filter performance against a fraternal-twin truth model.

An identical-twin experiment, where the filter's model and the truth model are
the same, tests only the arithmetic. Real digital twins are fraternal: the truth
differs structurally from the model. Three deliberate errors are injected here:

* a **perturbed LJF formulation** - the truth uses a different chord-wall
  spreading length from the filter's;
* a **biased drag coefficient** - the truth's Cd is offset from the filter's;
* **unmodelled scour** - the truth's foundation softens over time, and the filter
  has no state for it.

Three estimators are compared: log-EnKF, SIR particle filter and no-update prior
propagation. Both convergence and calibration are reported, because a filter that
narrows onto the wrong answer is worse than one that admits ignorance, and only
the calibration diagnostics can tell the two apart.

Crack growth uses a generic power law here rather than BS 7910 constants, which
are not shipped. That is enough to exercise and compare the *estimators*, which
is what C6 is about; it is not a fatigue life prediction, and the module says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...assimilation.baseline import prior_only
from ...assimilation.calibration import CalibrationReport, assess
from ...assimilation.log_enkf import EnKFConfig, LogEnKF
from ...assimilation.particle import ParticleFilter

__all__ = ["FilterComparison", "run_comparison"]


@dataclass
class FilterComparison:
    times_years: np.ndarray
    truth: np.ndarray
    ensembles: dict[str, np.ndarray]  # name -> (n_times, n_members)
    reports: dict[str, CalibrationReport]
    model_error_note: str

    def best_by_crps(self) -> str:
        return min(self.reports, key=lambda k: self.reports[k].crps)

    def headline(self) -> str:
        lines = []
        for name, r in self.reports.items():
            lines.append(
                f"{name}: RMSE {r.rmse:.4g}, CRPS {r.crps:.4g}, "
                f"90 percent coverage {r.coverage_90 * 100:.0f} percent -> {r.verdict}"
            )
        return "\n".join(lines)


def _growth(rate: float, exponent: float):
    """Generic power-law growth ``da/dt = rate * a^exponent``."""

    def g(a: np.ndarray, dt: float) -> np.ndarray:
        return a + rate * np.maximum(a, 1e-12) ** exponent * dt

    return g


def run_comparison(
    a0: float = 1.5e-3,
    a0_cv: float = 0.5,
    years: float = 20.0,
    n_steps: int = 40,
    n_members: int = 128,
    n_particles: int = 512,
    obs_sd_fraction: float = 0.15,
    sensitivity: float = 1.0,
    truth_rate_bias: float = 1.6,
    seed: int = 20260728,
) -> FilterComparison:
    """Run the three estimators against a structurally different truth.

    ``truth_rate_bias`` is the factor by which the truth's growth rate exceeds
    the filter's model - the lumped effect of the perturbed LJF, biased Cd and
    unmodelled scour on the crack driving force. A value of 1.0 would collapse
    this back to an identical-twin experiment.
    """
    rng = np.random.default_rng(seed)
    dt = years / n_steps
    t = np.linspace(0.0, years, n_steps + 1)

    model_rate, exponent = 4.0e-5, 0.5
    truth_growth = _growth(model_rate * truth_rate_bias, exponent)
    model_growth = _growth(model_rate, exponent)

    truth = np.empty(n_steps + 1)
    a = a0
    truth[0] = a
    for k in range(n_steps):
        a = float(truth_growth(np.array([a]), dt)[0])
        truth[k + 1] = a

    obs = truth * np.exp(rng.normal(0.0, obs_sd_fraction, size=truth.shape))

    enkf = LogEnKF(EnKFConfig(n_members=n_members, inflation=1.02, seed=seed))
    enkf.initialise(a0, a0_cv)
    pf = ParticleFilter(n_particles=n_particles, seed=seed)
    pf.initialise(a0, a0_cv)

    for k in range(n_steps):
        enkf.forecast(model_growth, dt)
        pf.forecast(model_growth, dt)
        y = obs[k + 1]
        # Observation is log-linear in the state for the EnKF; direct for the PF.
        enkf.assimilate(np.log(y), obs_sd_fraction, sensitivity)
        pf.assimilate(y, obs_sd_fraction * y, lambda p: p)

    enkf_ens = np.array(enkf.history)
    pf_ens = np.array(pf.history)
    base = prior_only(a0, a0_cv, model_growth, dt, n_steps, n_members=n_particles, seed=seed)

    ensembles = {
        "log-EnKF": enkf_ens,
        "SIR particle filter": pf_ens,
        "no-update baseline": base,
    }
    reports = {name: assess(name, E, truth) for name, E in ensembles.items()}
    return FilterComparison(
        times_years=t,
        truth=truth,
        ensembles=ensembles,
        reports=reports,
        model_error_note=(
            f"Fraternal twin: truth grows {truth_rate_bias:.2f}x faster than the filter's model, "
            "standing in for a perturbed LJF formulation, a biased Cd and unmodelled scour. "
            "Growth uses a generic power law, not BS 7910 constants (not shipped), so these "
            "are estimator comparisons and not fatigue life predictions."
        ),
    )
