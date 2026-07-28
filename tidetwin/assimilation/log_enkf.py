"""Ensemble Kalman filter, with a log-transformed state for crack depth.

Two update forms are provided:

``deterministic`` (ensemble transform)
    The analysis mean uses the exact Kalman gain built from the ensemble
    covariance, and the anomalies are transformed by
    :math:`T = (I + S^\\top R^{-1} S)^{-1/2}` so that the analysis covariance is
    exactly :math:`(I - KH)P^f`. In the linear-Gaussian case, an ensemble whose
    sample statistics match the Kalman filter prior therefore reproduces the
    Kalman filter posterior to machine precision - which is what
    ``tests/test_assimilation.py`` checks.

``stochastic`` (perturbed observations)
    The classical Burgers/van Leeuwen/Evensen form. Correct in the limit of large
    ensembles, noisier for small ones.

Crack depth is strictly positive and its growth is multiplicative, so the state
is carried as ``log a``. That keeps every ensemble member positive without
clipping - clipping would quietly bias the ensemble and destroy exactly the
calibration this application is trying to measure.

References: Evensen, G., "The Ensemble Kalman Filter: theoretical formulation and
practical implementation", Ocean Dynamics 53:343-367, 2003; Bishop, Etherton &
Majumdar, "Adaptive sampling with the ensemble transform Kalman filter",
Monthly Weather Review 129:420-436, 2001; Anderson & Anderson (1999) for
multiplicative inflation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg as sla

__all__ = ["EnKFConfig", "enkf_update", "inflate", "LogEnKF"]


@dataclass(frozen=True)
class EnKFConfig:
    n_members: int = 64
    inflation: float = 1.02
    deterministic: bool = True
    seed: int = 20260728


def inflate(ensemble: np.ndarray, factor: float) -> np.ndarray:
    """Multiplicative inflation about the ensemble mean.

    Counteracts the covariance collapse that finite ensembles suffer under
    repeated updates. A factor of 1.0 disables it.
    """
    if factor == 1.0:
        return ensemble
    mean = ensemble.mean(axis=1, keepdims=True)
    return mean + factor * (ensemble - mean)


def enkf_update(
    ensemble: np.ndarray,
    observation: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    deterministic: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One analysis step.

    Parameters
    ----------
    ensemble
        ``(n_state, n_members)`` prior ensemble.
    observation
        ``(n_obs,)`` observed values.
    H
        ``(n_obs, n_state)`` linear observation operator.
    R
        ``(n_obs, n_obs)`` observation error covariance.
    """
    X = np.atleast_2d(np.asarray(ensemble, float))
    n_state, N = X.shape
    y = np.asarray(observation, float).ravel()
    H = np.atleast_2d(np.asarray(H, float))
    R = np.atleast_2d(np.asarray(R, float))
    if N < 2:
        raise ValueError("ensemble needs at least two members")

    xbar = X.mean(axis=1, keepdims=True)
    A = X - xbar
    Pf = A @ A.T / (N - 1)
    HPHt = H @ Pf @ H.T
    K = Pf @ H.T @ np.linalg.inv(HPHt + R)
    xa = xbar + K @ (y[:, None] - H @ xbar)

    if not deterministic:
        rng = rng or np.random.default_rng(0)
        noise = rng.multivariate_normal(np.zeros(len(y)), R, size=N).T
        return X + K @ (y[:, None] + noise - H @ X)

    # Ensemble transform: A_a = A T with T = (I + S^T R^-1 S)^(-1/2).
    S = (H @ A) / np.sqrt(N - 1)
    M = np.eye(N) + S.T @ np.linalg.inv(R) @ S
    w, V = sla.eigh(M)
    w = np.maximum(w, 1e-300)
    T = V @ np.diag(w**-0.5) @ V.T
    return xa + A @ T


@dataclass
class LogEnKF:
    """Log-state EnKF for crack depth with a user-supplied growth model.

    ``forecast`` maps ``(a, dt, member_index)`` to the propagated depth. Working
    in ``log a`` means the filter can never produce a negative or zero crack.
    """

    config: EnKFConfig = field(default_factory=EnKFConfig)
    ensemble_log_a: np.ndarray | None = None
    history: list[np.ndarray] = field(default_factory=list)

    def initialise(self, a_mean: float, a_cv: float) -> None:
        """Seed a lognormal prior with the given mean and coefficient of variation."""
        rng = np.random.default_rng(self.config.seed)
        sigma = np.sqrt(np.log1p(a_cv**2))
        mu = np.log(a_mean) - 0.5 * sigma**2
        self.ensemble_log_a = rng.normal(mu, sigma, size=(1, self.config.n_members))
        self.history = [self.depths().copy()]

    def depths(self) -> np.ndarray:
        if self.ensemble_log_a is None:
            raise RuntimeError("call initialise() first")
        return np.exp(self.ensemble_log_a[0])

    def forecast(self, growth: Callable[[np.ndarray, float], np.ndarray], dt: float) -> None:
        a = self.depths()
        a_new = np.maximum(growth(a, dt), 1e-12)
        self.ensemble_log_a = np.log(a_new)[None, :]

    def assimilate(self, observation: float, obs_sd: float, sensitivity: float) -> None:
        """Update on a scalar observation linear in ``log a``.

        ``sensitivity`` is ``d(observed quantity)/d(log a)``, so the observation
        operator stays linear in the transformed state - which is what makes the
        log transform useful rather than merely convenient.
        """
        if self.ensemble_log_a is None:
            raise RuntimeError("call initialise() first")
        X = inflate(self.ensemble_log_a, self.config.inflation)
        rng = np.random.default_rng(self.config.seed + len(self.history))
        self.ensemble_log_a = enkf_update(
            X,
            np.array([observation]),
            np.array([[sensitivity]]),
            np.array([[obs_sd**2]]),
            deterministic=self.config.deterministic,
            rng=rng,
        )
        self.history.append(self.depths().copy())
