"""Verification of the 3D beam element against closed-form solutions.

These are solver verification tests, not UI tests. Every reference value here is
either an analytical result quoted with its source or derived in the test
docstring so a reviewer can check the algebra.
"""

from __future__ import annotations

import numpy as np
import pytest

from tidetwin.fe.assemble import Member, Model
from tidetwin.numerics import bisect_root
from tidetwin.fe.beam3d import (
    Section,
    local_stiffness,
    tubular_section,
    transformation,
)
from tidetwin.fe.modal import eigenmodes

E = 210.0e9
NU = 0.3
G = E / (2.0 * (1.0 + NU))
RHO = 7850.0


def _rect_section(b: float, h: float, kappa: float = 0.0) -> Section:
    """Solid rectangular section, local y along ``b`` and z along ``h``."""
    A = b * h
    Iz = b**3 * h / 12.0  # bending in the local x-y plane
    Iy = b * h**3 / 12.0  # bending in the local x-z plane
    J = Iy + Iz
    return Section(A=A, Iy=Iy, Iz=Iz, J=J, E=E, G=G, rho=RHO, kappa=kappa)


def _modes_dominated_by(res, component: int, floor: float = 0.3) -> np.ndarray:
    """Frequencies of modes whose largest translation is along ``component``.

    Modes are normalised by their largest DOF amplitude before comparison. The
    ``floor`` rejects modes with no meaningful translation (pure torsion), where
    an argmax over near-zero translations would otherwise pick a direction at
    random. Rotational DOF are excluded from the comparison because in high
    bending modes the end rotation exceeds the end translation numerically
    without changing what kind of mode it is.
    """
    disp = res.modes.reshape(-1, 6, res.modes.shape[1])
    scale = np.abs(disp).max(axis=(0, 1))
    scale[scale == 0] = 1.0
    trans = np.abs(disp[:, :3, :]).max(axis=0) / scale
    dominant = np.argmax(trans, axis=0)
    keep = (dominant == component) & (trans.max(axis=0) > floor)
    return res.frequencies_hz[keep]


def _cantilever(n_el: int, sec: Section, L: float, axis: int = 0) -> Model:
    """Cantilever of length ``L`` along a global axis, fixed at the first node."""
    nodes = np.zeros((n_el + 1, 3))
    nodes[:, axis] = np.linspace(0.0, L, n_el + 1)
    m = Model(nodes=nodes)
    for i in range(n_el):
        m.members.append(Member(i, i + 1, sec, name=f"e{i}"))
    m.fixed[0] = (True, True, True, True, True, True)
    return m


# --------------------------------------------------------------- static tests


def test_euler_bernoulli_tip_deflection():
    """Tip load on a cantilever: delta = P L^3 / (3 E I).

    Timoshenko's *Strength of Materials* Part I, Table, cantilever with end load.
    The cubic Hermite element is exact for this case, so a single element must
    reproduce it to machine precision.
    """
    L, P = 3.0, 5000.0
    sec = _rect_section(0.1, 0.2, kappa=0.0)
    m = _cantilever(1, sec, L)
    f = np.zeros(m.n_dof)
    f[m.dof(1, "uy")] = P
    res = m.solve_static(f)
    expected = P * L**3 / (3.0 * E * sec.Iz)
    assert res.displacements[1, 1] == pytest.approx(expected, rel=1e-12)


def test_timoshenko_shear_contribution():
    """With shear: delta = P L^3/(3EI) + P L/(kappa G A).

    Cowper (1966); the two-node Timoshenko element is exact for constant shear,
    so one element must again match to machine precision. A stubby beam is used
    so the shear term is a large fraction of the total.
    """
    L, P = 0.5, 5000.0
    sec = _rect_section(0.2, 0.4, kappa=5.0 / 6.0)
    m = _cantilever(1, sec, L)
    f = np.zeros(m.n_dof)
    f[m.dof(1, "uy")] = P
    res = m.solve_static(f)
    bending = P * L**3 / (3.0 * E * sec.Iz)
    shear = P * L / (sec.kappa * G * sec.A)
    assert res.displacements[1, 1] == pytest.approx(bending + shear, rel=1e-12)
    # The shear term must be a meaningful share of the answer for this to bite.
    assert shear / (bending + shear) > 0.10


def test_axial_and_torsional_flexibility():
    """delta = N L / (E A) and phi = T L / (G J)."""
    L, N, T = 4.0, 1.0e6, 2.0e4
    sec = _rect_section(0.15, 0.15)
    m = _cantilever(1, sec, L)
    f = np.zeros(m.n_dof)
    f[m.dof(1, "ux")] = N
    f[m.dof(1, "rx")] = T
    res = m.solve_static(f)
    assert res.displacements[1, 0] == pytest.approx(N * L / (E * sec.A), rel=1e-12)
    assert res.displacements[1, 3] == pytest.approx(T * L / (G * sec.J), rel=1e-12)


def test_reactions_balance_applied_load():
    """Sum of reactions must equal minus the applied load (global equilibrium)."""
    sec = tubular_section(1.2, 0.035, E, NU, RHO)
    m = _cantilever(6, sec, 12.0)
    rng = np.random.default_rng(7)
    f = np.zeros(m.n_dof)
    for i in range(1, m.n_nodes):
        f[m.dof(i, "uy")] = rng.normal(0.0, 1.0e4)
        f[m.dof(i, "uz")] = rng.normal(0.0, 1.0e4)
    res = m.solve_static(f)
    assert res.reactions.sum(axis=0)[1] == pytest.approx(-f.reshape(-1, 6)[:, 1].sum(), rel=1e-9)
    assert res.reactions.sum(axis=0)[2] == pytest.approx(-f.reshape(-1, 6)[:, 2].sum(), rel=1e-9)


def test_orientation_invariance():
    """A cantilever rotated into an arbitrary direction gives the same tip magnitude.

    Exercises the direction-cosine transformation including the near-vertical
    degenerate branch.
    """
    L, P = 5.0, 1.0e4
    sec = tubular_section(0.8, 0.02, E, NU, RHO)
    ref = None
    for axis in (0, 1, 2):
        m = _cantilever(4, sec, L, axis=axis)
        # Load transverse to the member, in a direction orthogonal to its axis.
        load_dir = (axis + 1) % 3
        f = np.zeros(m.n_dof)
        f[m.dof(m.n_nodes - 1, load_dir)] = P
        res = m.solve_static(f)
        tip = np.linalg.norm(res.displacements[m.n_nodes - 1, :3])
        if ref is None:
            ref = tip
        else:
            assert tip == pytest.approx(ref, rel=1e-10)
    assert ref is not None and ref > 0


def test_transformation_is_orthonormal():
    for xi, xj in [
        ((0, 0, 0), (1, 2, 3)),
        ((0, 0, 0), (0, 0, 5)),  # vertical: degenerate reference branch
        ((1, 1, 1), (1, 1, -4)),
    ]:
        T, L = transformation(np.array(xi, float), np.array(xj, float))
        lam = T[:3, :3]
        assert np.allclose(lam @ lam.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(lam) == pytest.approx(1.0, abs=1e-12)
        assert L == pytest.approx(np.linalg.norm(np.array(xj, float) - np.array(xi, float)))


def test_stiffness_symmetry_and_rigid_body_nullspace():
    """K must be symmetric and annihilate rigid-body motion."""
    sec = tubular_section(1.0, 0.03, E, NU, RHO)
    K = local_stiffness(sec, 2.5)
    assert np.allclose(K, K.T, atol=1e-6)
    # Rigid translation along each local axis.
    for c in range(3):
        u = np.zeros(12)
        u[c] = 1.0
        u[6 + c] = 1.0
        assert np.allclose(K @ u, 0.0, atol=1e-3)
    # Rigid rotation about local z: uy = x * theta, rz = theta.
    L = 2.5
    u = np.zeros(12)
    u[1], u[5] = 0.0, 1.0
    u[7], u[11] = L, 1.0
    assert np.allclose(K @ u, 0.0, atol=1e-3)


def test_portal_frame_sidesway():
    """Fixed-base portal frame under sidesway load, slope-deflection solution.

    With column height ``h``, beam span ``L``, stiffness ratio
    ``n = (EI_b/L)/(EI_c/h)`` and axial deformation neglected, slope-deflection
    gives

        Delta = H h^3 (2 + 3n) / (12 EI_c (1 + 6n)).

    Two independent checks of that expression: as ``n -> inf`` (rigid beam) it
    tends to ``H h^3 / (24 EI_c)``, the two fixed-fixed columns in parallel; as
    ``n -> 0`` it tends to ``H h^3 / (6 EI_c)``, two fixed-free cantilevers in
    parallel. The FE model uses a large area so that the neglected axial
    flexibility stays below the tolerance.
    """
    h, span, H = 4.0, 6.0, 1.0e4
    I = 1.0e-4
    sec = Section(A=0.5, Iy=I, Iz=I, J=2 * I, E=E, G=G, rho=RHO, kappa=0.0)

    # Frame lies in the global X-Z plane; columns vertical, beam horizontal.
    nodes = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, h], [span, 0.0, h], [span, 0.0, 0.0]])
    m = Model(nodes=nodes)
    m.members += [
        Member(0, 1, sec, "col-left"),
        Member(1, 2, sec, "beam"),
        Member(3, 2, sec, "col-right"),
    ]
    m.fixed[0] = (True,) * 6
    m.fixed[3] = (True,) * 6
    # Restrain out-of-plane DOF at the free nodes so the 2D solution applies.
    for nd in (1, 2):
        m.fixed[nd] = (False, True, False, True, False, True)

    f = np.zeros(m.n_dof)
    f[m.dof(1, "ux")] = H
    res = m.solve_static(f)

    n_ratio = (E * I / span) / (E * I / h)
    expected = H * h**3 * (2.0 + 3.0 * n_ratio) / (12.0 * E * I * (1.0 + 6.0 * n_ratio))
    assert res.displacements[1, 0] == pytest.approx(expected, rel=2.0e-3)
    # Both joints sway together when the beam is axially stiff.
    assert res.displacements[2, 0] == pytest.approx(res.displacements[1, 0], rel=1e-3)


# ---------------------------------------------------------------- modal tests


def test_cantilever_natural_frequencies():
    """Bending modes of a cantilever against the Euler-Bernoulli eigenvalues.

    f_n = (beta_n L)^2 / (2 pi L^2) * sqrt(EI / (rho A)), where beta_n L are the
    roots of ``cos(x) cosh(x) + 1 = 0`` (Blevins, *Formulas for Natural Frequency
    and Mode Shape*, 1979, Table 8-1, clamped-free beam: 1.87510, 4.69409,
    7.85476). The roots are found here to machine precision rather than quoted,
    because the element is accurate enough on mode 1 that a six-figure reference
    would be the limiting error rather than the solver.
    """
    L = 6.0
    b, hgt = 0.10, 0.30
    sec = _rect_section(b, hgt, kappa=0.0)  # Iz < Iy, so x-y bending is softest
    m = _cantilever(24, sec, L)
    K, M = m.assemble()
    res = eigenmodes(K, M, m.free_dof(), n_modes=20)
    bending_y = _modes_dominated_by(res, 1)

    betaL = np.array(
        [
            bisect_root(lambda x: np.cos(x) * np.cosh(x) + 1.0, a, b, xtol=1e-14)
            for a, b in [(1.8, 1.95), (4.6, 4.8), (7.7, 7.95)]
        ]
    )
    assert betaL == pytest.approx([1.87510, 4.69409, 7.85476], abs=1e-5)
    analytic = betaL**2 / (2.0 * np.pi * L**2) * np.sqrt(E * sec.Iz / (RHO * sec.A))

    assert len(bending_y) >= 3
    assert bending_y[:3] == pytest.approx(analytic, rel=2.0e-3)
    # Consistent mass is an upper bound on the true frequency (Rayleigh quotient
    # over a restricted trial space), so the FE values may not fall below.
    assert np.all(bending_y[:3] >= analytic * (1.0 - 1e-12))


def test_axial_and_torsional_modes():
    """Rod modes: f_n = (2n-1)/(4L) * sqrt(E/rho) axially, sqrt(G/rho) in torsion
    for a section whose polar radius of gyration equals J/A appropriately.

    Only the fundamental axial mode is checked, which is the standard
    clamped-free rod result (Blevins 1979, Table 9-1).
    """
    L = 6.0
    sec = _rect_section(0.2, 0.2, kappa=0.0)
    m = _cantilever(30, sec, L)
    K, M = m.assemble()
    res = eigenmodes(K, M, m.free_dof(), n_modes=40)
    axial = _modes_dominated_by(res, 0)
    f1 = np.sqrt(E / RHO) / (4.0 * L)
    assert len(axial) >= 1
    assert axial[0] == pytest.approx(f1, rel=2e-3)


def test_modal_frequencies_drop_when_stiffness_drops():
    """Softening any member must lower every natural frequency (Rayleigh)."""
    sec = tubular_section(1.0, 0.03, E, NU, RHO)
    soft = Section(**{**sec.__dict__, "E": sec.E * 0.5, "G": sec.G * 0.5})
    m0 = _cantilever(8, sec, 10.0)
    m1 = _cantilever(8, sec, 10.0)
    m1.members[3] = Member(3, 4, soft)
    f0 = eigenmodes(*m0.assemble(), m0.free_dof(), n_modes=6).frequencies_hz
    f1 = eigenmodes(*m1.assemble(), m1.free_dof(), n_modes=6).frequencies_hz
    assert np.all(f1 <= f0 + 1e-9)
    assert np.any(f1 < f0 * 0.999)


# ------------------------------------------------------------ section checks


def test_tubular_section_properties():
    """Thin-wall limits about the *mean* radius: A -> 2 pi rm t, I -> pi rm^3 t.

    Using the outer diameter instead of the mean radius carries an O(t/D) error,
    which at t/D = 1/60 is already 5 percent in I -- large enough to matter for a
    stiffness ratio, hence the mean-radius form here.
    """
    D, t = 1.2, 0.02
    rm = 0.5 * (D - t)
    sec = tubular_section(D, t, E, NU, RHO)
    assert sec.A == pytest.approx(2 * np.pi * rm * t, rel=1e-3)
    assert sec.Iy == pytest.approx(np.pi * rm**3 * t, rel=1e-3)
    assert sec.J == pytest.approx(2.0 * sec.Iy, rel=1e-12)
    # Cowper's thin-walled round-tube limit: kappa = 2(1+nu)/(4+3nu).
    assert sec.kappa == pytest.approx(2 * (1 + NU) / (4 + 3 * NU), rel=2e-2)
    with pytest.raises(ValueError):
        tubular_section(0.1, 0.2, E, NU, RHO)
