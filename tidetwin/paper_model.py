"""A faithful reproduction of the paper's own analytical reference model.

The adversarial FE frame in :mod:`tidetwin.analysis` is an *independent* model
built to attack the paper's claims. It deliberately does not reuse the paper's
equations, which is why its absolute numbers differ from the paper's. That is the
right design for falsification, but it also means it cannot, on its own, answer a
different and equally important question: *are the paper's own stated results
internally consistent and correctly computed from the model it defines?*

This module answers that question. It implements the paper's analytical chain -
Eqs 1-11 of the manuscript - from the inputs in its Tables 2 and 3, and nothing
else. If it reproduces the paper's headline numbers (panel shear, overturning
moment, member forces, local deformation, the intact excess ratio, and the
damage sensitivity), the paper's arithmetic is sound; if it did not, that would
be a finding in its own right. The self-test at the bottom asserts every number
the paper prints, so a regression in either this code or the recorded claim is
caught immediately.

Definitions follow the manuscript exactly:

* ``R`` is the raw ratio of the bracketing gauge to the remote reference gauge,
  Eq 7, ``R = 1 + delta/(L_g * eps_nom)``.
* ``Phi = R - 1`` is the **excess** strain ratio, Eq 8 - the observable the paper
  actually adopts, because the common unit term carries no joint information.
* ``d`` (written ``d_K`` in the manuscript, Eq 9) is the local stiffness
  reduction. The central analytical result, Eq 11, is that the excess ratio's
  fractional change is ``d/(1-d)`` - independent of geometry, gauge length and
  load amplitude - while the raw ratio moves only ``eta`` times as much, with the
  local isolation factor ``eta = 1 - 1/R_0``.

What this module is NOT: it is not evidence that a real crack produces a given
``d`` (Eq 9 is the paper's stated, unvalidated assumption and its own central
open question), nor that a real frame isolates the local deformation as cleanly
as Eq 6 assumes. Those are what the independent FE model and the nuisance budget
probe. This module only certifies that, *given the paper's model*, its numbers
follow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PaperInputs", "PaperReference", "reference_solution", "excess_ratio_change"]


@dataclass(frozen=True)
class PaperInputs:
    """Every input the paper lists in Tables 2 and 3, verbatim."""

    # Geometry (Table 2).
    chord_D: float = 0.762           # chord outside diameter, m
    chord_T: float = 0.025           # chord wall thickness, m
    brace_d: float = 0.508           # brace outside diameter, m
    brace_t: float = 0.016           # brace wall thickness, m
    brace_angle_deg: float = 45.0
    chord_area: float = 0.0579       # A_c, m^2
    brace_area: float = 0.0247       # A_b, m^2
    E: float = 210.0e9               # Young's modulus, Pa
    f_LJF: float = 4.0               # local joint flexibility factor (assumed)

    # Environment and load case (Table 3).
    water_depth: float = 30.0        # h, m
    joint_elevation: float = 15.0    # z_J, m above mudline
    leg_spacing: float = 24.0        # B_J, m
    projected_area: float = 4.06     # A_p, m^2/m
    drag_coefficient: float = 1.05   # C_D
    seawater_density: float = 1025.0 # rho, kg/m^3
    spring_current: float = 0.80     # u_s, m/s
    neap_current: float = 0.40       # m/s
    spring_tidal_range: float = 1.6  # dz_t, m
    neap_tidal_range: float = 0.6    # m
    g: float = 9.81                  # gravity, m/s^2

    # Sensing (Table 1).
    gauge_length: float = 0.55       # L_g, m
    braced_faces: int = 2            # n_f, faces resisting the shear

    @property
    def brace_angle(self) -> float:
        return np.radians(self.brace_angle_deg)

    @property
    def R0_isolation(self) -> float:
        """The local isolation factor eta = 1 - 1/R_0 is set by R_0; see solution."""
        return 1.0 - 1.0 / reference_solution(self).R0


@dataclass(frozen=True)
class PaperReference:
    """The reference intact solution, every quantity the paper's Section 4.2 prints."""

    panel_shear: float            # V, N
    overturning_moment: float     # M, N*m
    brace_axial: float            # P_b, N
    leg_axial: float              # P_c, N
    tidal_buoyancy: float         # P_t, N
    local_deformation: float      # delta, m
    nominal_strain: float         # eps_nom, strain
    bracketing_strain: float      # eps_A, strain
    reference_strain: float       # eps_B, strain
    R0: float                     # raw ratio
    Phi0: float                   # excess ratio = R0 - 1
    isolation_factor: float       # eta = 1 - 1/R0


def _panel_shear(p: PaperInputs, current: float) -> float:
    """V = integral of the Morison drag term (Eq 1) from the joint to the surface.

    f_D(z) = 1/2 rho C_D A_p u_s^2 (z/h)^(2/7); the panel shear at the joint is the
    integral of that distributed load above the joint elevation.
    """
    f0 = 0.5 * p.seawater_density * p.drag_coefficient * p.projected_area * current**2
    h, zj = p.water_depth, p.joint_elevation
    # integral_{zj}^{h} (z/h)^(2/7) dz = h * (7/9) * (1 - (zj/h)^(9/7))
    return f0 * h * (7.0 / 9.0) * (1.0 - (zj / h) ** (9.0 / 7.0))


def _overturning_moment(p: PaperInputs, current: float) -> float:
    """M = first moment of the drag load about the mudline (Eq 1 integrated with z)."""
    f0 = 0.5 * p.seawater_density * p.drag_coefficient * p.projected_area * current**2
    h = p.water_depth
    # integral_{0}^{h} (z/h)^(2/7) z dz = h^2 * 7/16
    return f0 * h**2 * (7.0 / 16.0)


def reference_solution(p: PaperInputs = PaperInputs(), spring: bool = True) -> PaperReference:
    """Reproduce the paper's intact reference solution (Section 4.2) from first principles."""
    current = p.spring_current if spring else p.neap_current
    tidal_range = p.spring_tidal_range if spring else p.neap_tidal_range

    V = _panel_shear(p, current)
    M = _overturning_moment(p, current)

    # Eq 2: brace axial force from truss action across the braced faces.
    P_b = V / (2.0 * p.braced_faces * np.cos(p.brace_angle))
    # Eq 3: leg axial force from the global overturning couple.
    P_c = M / (2.0 * p.leg_spacing)
    # Eq 4: extra leg axial force from the tidal change in submerged volume.
    P_t = p.seawater_density * p.g * (np.pi / 4.0) * p.chord_D**2 * tidal_range

    # Eq 5: local joint deformation across the intersection.
    delta = p.f_LJF * P_b / (p.E * p.chord_D)

    # Nominal chord strain the reference gauge reads: the leg axial from overturning
    # plus the tidal buoyancy axial, over the chord section.
    eps_nom = (P_c + P_t) / (p.chord_area * p.E)

    # Eq 6: the bracketing gauge reads nominal plus the local deformation spread
    # over the gauge length; the reference gauge reads nominal alone.
    eps_A = eps_nom + delta / p.gauge_length
    eps_B = eps_nom

    R0 = eps_A / eps_B               # Eq 7
    Phi0 = R0 - 1.0                  # Eq 8
    eta = 1.0 - 1.0 / R0             # local isolation factor (Eq 11)

    return PaperReference(
        panel_shear=V, overturning_moment=M, brace_axial=P_b, leg_axial=P_c,
        tidal_buoyancy=P_t, local_deformation=delta, nominal_strain=eps_nom,
        bracketing_strain=eps_A, reference_strain=eps_B, R0=R0, Phi0=Phi0,
        isolation_factor=eta,
    )


def excess_ratio_change(d: float, ref: PaperReference | None = None) -> dict[str, float]:
    """The paper's central analytical result, Eq 11, for a local stiffness reduction ``d``.

    Returns the fractional change in the excess ratio ``Phi`` and in the raw ratio
    ``R``. The excess-ratio sensitivity ``d/(1-d)`` is exact and independent of the
    reference solution; the raw-ratio sensitivity is smaller by the isolation
    factor ``eta = 1 - 1/R_0`` and so needs the reference.
    """
    if not 0.0 <= d < 1.0:
        raise ValueError("stiffness reduction d must be in [0, 1)")
    dPhi_over_Phi = d / (1.0 - d)                       # Eq 11, excess ratio
    out = {"d": d, "dPhi_over_Phi": dPhi_over_Phi}
    if ref is not None:
        out["dR_over_R"] = ref.isolation_factor * dPhi_over_Phi  # Eq 11, raw ratio
        out["eta"] = ref.isolation_factor
    return out


def _self_test() -> None:
    """Assert that this reproduction matches every number the paper prints.

    Run with ``python -m tidetwin.paper_model``. Every tolerance is tight enough
    that a real discrepancy fails, loose enough to absorb the paper's own rounding.
    """
    ref = reference_solution()
    checks = [
        ("panel shear V, kN", ref.panel_shear / 1e3, 19.3, 0.1),
        ("overturning moment M, MN.m", ref.overturning_moment / 1e6, 0.551, 0.005),
        ("brace axial P_b, kN", ref.brace_axial / 1e3, 6.81, 0.05),
        ("leg axial P_c, kN", ref.leg_axial / 1e3, 11.48, 0.05),
        ("tidal buoyancy P_t, kN", ref.tidal_buoyancy / 1e3, 7.34, 0.05),
        ("local deformation delta, um", ref.local_deformation * 1e6, 0.170, 0.003),
        ("nominal strain eps_nom, ustrain", ref.nominal_strain * 1e6, 1.548, 0.01),
        ("bracketing strain eps_A, ustrain", ref.bracketing_strain * 1e6, 1.858, 0.01),
        ("reference strain eps_B, ustrain", ref.reference_strain * 1e6, 1.548, 0.01),
        ("raw ratio R0", ref.R0, 1.200, 0.003),
        ("excess ratio Phi0", ref.Phi0, 0.200, 0.003),
        ("isolation factor eta", ref.isolation_factor, 0.167, 0.003),
    ]
    width = max(len(name) for name, *_ in checks)
    ok = True
    for name, got, want, tol in checks:
        hit = abs(got - want) <= tol
        ok = ok and hit
        print(f"{'OK ' if hit else 'XX '}{name:<{width}}  got {got:10.4f}  want {want:10.4f}")

    # Eq 11 damage sensitivity: d=0.10 -> +11.1%, d=0.20 -> +25.0% in the excess ratio.
    for d, want_pct in ((0.10, 11.11), (0.20, 25.0)):
        chg = excess_ratio_change(d, ref)
        got_pct = chg["dPhi_over_Phi"] * 100.0
        hit = abs(got_pct - want_pct) <= 0.2
        ok = ok and hit
        print(f"{'OK ' if hit else 'XX '}excess ratio change at d={d:.2f}        "
              f"got {got_pct:9.3f}%  want {want_pct:8.3f}%")
        # the raw ratio recovers only about one sixth (eta) of that sensitivity
        print(f"    (raw ratio change {chg['dR_over_R'] * 100:7.3f}%, "
              f"eta = {chg['eta']:.3f})")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    _self_test()
