"""Eigenvalue extraction for the frame.

Solves the undamped generalised eigenproblem :math:`K\\phi = \\omega^2 M \\phi`
on the free DOF. Used for claim C7 (modal insensitivity to local joint damage),
where the quantity of interest is the *shift* in natural frequency between the
intact and damaged frames, so consistency of the discretisation between the two
solves matters more than absolute accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["ModalResult", "eigenmodes", "frequency_shift"]


@dataclass
class ModalResult:
    """Natural frequencies and mass-normalised mode shapes."""

    frequencies_hz: np.ndarray
    omega: np.ndarray
    modes: np.ndarray  # (n_dof, n_modes), zero at restrained DOF
    free_dof: np.ndarray

    def participation(self, direction: int, M: sp.csr_matrix) -> np.ndarray:
        """Modal participation factors for a rigid-body translation direction.

        ``direction`` is 0, 1 or 2 for global X, Y, Z.
        """
        n_dof = self.modes.shape[0]
        r = np.zeros(n_dof)
        r[direction::6] = 1.0
        Mr = M @ r
        return np.asarray(self.modes.T @ Mr)

    def effective_mass_fraction(self, direction: int, M: sp.csr_matrix) -> np.ndarray:
        """Fraction of total translational mass captured by each mode."""
        p = self.participation(direction, M)
        n_dof = self.modes.shape[0]
        r = np.zeros(n_dof)
        r[direction::6] = 1.0
        total = float(r @ (M @ r))
        return (p**2) / total if total > 0 else np.zeros_like(p)


def eigenmodes(
    K: sp.csr_matrix,
    M: sp.csr_matrix,
    free_dof: np.ndarray,
    n_modes: int = 12,
    dense_threshold: int = 900,
) -> ModalResult:
    """Lowest ``n_modes`` natural frequencies and mode shapes.

    Uses a dense symmetric solver for small systems (exact, no convergence
    tolerance to tune) and shift-invert Lanczos above ``dense_threshold`` free
    DOF. The shift is placed slightly below zero so the factorisation targets the
    lowest modes without hitting the rigid-body singularity.
    """
    Kff = K[free_dof][:, free_dof]
    Mff = M[free_dof][:, free_dof]
    n_free = Kff.shape[0]
    n_modes = int(min(n_modes, max(1, n_free - 2)))

    if n_free <= dense_threshold:
        Kd = np.asarray(Kff.todense())
        Md = np.asarray(Mff.todense())
        Kd = 0.5 * (Kd + Kd.T)
        Md = 0.5 * (Md + Md.T)
        w, v = sla.eigh(Kd, Md)
    else:  # pragma: no cover - exercised only on large models
        sigma = -1.0e-3 * float(abs(Kff.diagonal()).mean())
        w, v = spla.eigsh(Kff.tocsc(), k=n_modes, M=Mff.tocsc(), sigma=sigma, which="LM")
        order = np.argsort(w)
        w, v = w[order], v[:, order]

    w = np.maximum(w[:n_modes], 0.0)
    v = v[:, :n_modes]
    omega = np.sqrt(w)

    n_dof = K.shape[0]
    modes = np.zeros((n_dof, v.shape[1]))
    modes[free_dof, :] = v
    return ModalResult(
        frequencies_hz=omega / (2.0 * np.pi),
        omega=omega,
        modes=modes,
        free_dof=free_dof,
    )


def frequency_shift(intact: ModalResult, damaged: ModalResult, n: int = 6) -> np.ndarray:
    """Relative frequency change ``(f_damaged - f_intact) / f_intact`` per mode.

    Returned as a fraction (multiply by 100 for percent). Negative values mean
    the damaged structure is softer, which is the physically expected direction
    for a stiffness-reducing crack.
    """
    k = int(min(n, len(intact.frequencies_hz), len(damaged.frequencies_hz)))
    f0 = intact.frequencies_hz[:k]
    f1 = damaged.frequencies_hz[:k]
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(f0 > 0, (f1 - f0) / f0, np.nan)
    return shift
