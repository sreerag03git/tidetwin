"""Sequential importance resampling particle filter, as a benchmark for the EnKF.

Included because the EnKF's Gaussian analysis is an approximation that a
particle filter does not make. Where the two agree, the Gaussian assumption is
harmless; where they disagree, the EnKF's intervals are suspect and C6 should
say so.

Systematic (stratified) resampling is used - lower variance than multinomial for
the same particle count (Kitagawa, "Monte Carlo filter and smoother for
non-Gaussian nonlinear state space models", J. Comput. Graph. Stat. 5:1-25,
1996).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

__all__ = ["ParticleFilter", "systematic_resample", "effective_sample_size"]


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices drawn by systematic resampling."""
    w = np.asarray(weights, float)
    n = w.size
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(w), positions).clip(0, n - 1)


def effective_sample_size(weights: np.ndarray) -> float:
    """``1 / sum(w^2)``: how many particles are actually contributing."""
    w = np.asarray(weights, float)
    s = float((w**2).sum())
    return float(1.0 / s) if s > 0 else 0.0


@dataclass
class ParticleFilter:
    """SIR filter over a positive scalar state (crack depth)."""

    n_particles: int = 512
    seed: int = 20260728
    resample_threshold: float = 0.5
    particles: np.ndarray | None = None
    weights: np.ndarray | None = None
    history: list[np.ndarray] = field(default_factory=list)
    ess_history: list[float] = field(default_factory=list)

    def initialise(self, a_mean: float, a_cv: float) -> None:
        rng = np.random.default_rng(self.seed)
        sigma = np.sqrt(np.log1p(a_cv**2))
        mu = np.log(a_mean) - 0.5 * sigma**2
        self.particles = rng.lognormal(mu, sigma, size=self.n_particles)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
        self.history = [self.particles.copy()]
        self.ess_history = [float(self.n_particles)]

    def forecast(self, growth: Callable[[np.ndarray, float], np.ndarray], dt: float) -> None:
        if self.particles is None:
            raise RuntimeError("call initialise() first")
        self.particles = np.maximum(growth(self.particles, dt), 1e-12)

    def assimilate(
        self, observation: float, obs_sd: float, forward: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        """Reweight on an observation with Gaussian likelihood."""
        if self.particles is None or self.weights is None:
            raise RuntimeError("call initialise() first")
        predicted = forward(self.particles)
        loglik = -0.5 * ((observation - predicted) / obs_sd) ** 2
        loglik -= loglik.max()
        w = self.weights * np.exp(loglik)
        total = w.sum()
        if total <= 0 or not np.isfinite(total):
            # Complete filter divergence: every particle is impossible under the
            # observation. Reporting this is more useful than silently resetting.
            w = np.full(self.n_particles, 1.0 / self.n_particles)
        else:
            w = w / total
        self.weights = w
        ess = effective_sample_size(w)
        self.ess_history.append(ess)
        if ess < self.resample_threshold * self.n_particles:
            rng = np.random.default_rng(self.seed + len(self.history))
            idx = systematic_resample(w, rng)
            self.particles = self.particles[idx]
            self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
        self.history.append(self.resampled_ensemble())

    def resampled_ensemble(self) -> np.ndarray:
        """Equally weighted sample, so calibration diagnostics can be applied."""
        if self.particles is None or self.weights is None:
            raise RuntimeError("call initialise() first")
        rng = np.random.default_rng(self.seed + 977 + len(self.history))
        idx = systematic_resample(self.weights, rng)
        return self.particles[idx]
