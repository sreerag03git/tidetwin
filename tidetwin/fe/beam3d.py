"""3D two-node Timoshenko beam element, 12 degrees of freedom.

Element formulation follows Przemieniecki, *Theory of Matrix Structural
Analysis*, McGraw-Hill 1968, Section 6.4 (stiffness with transverse shear) and
Section 11.3 (consistent mass). Shear coefficients for hollow circular sections
follow Cowper, "The shear coefficient in Timoshenko's beam theory",
J. Appl. Mech. 33(2):335-340, 1966, Eq. 13.

Degree-of-freedom ordering, per node: ``[ux, uy, uz, rx, ry, rz]``; element
vector is node i followed by node j, giving 12 DOF. Local x runs from node i to
node j.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Section", "tubular_section", "local_stiffness", "local_mass", "transformation", "element_matrices"]


@dataclass(frozen=True)
class Section:
    """Cross-section and material properties for a prismatic member.

    Attributes
    ----------
    A : float
        Cross-sectional area, m^2.
    Iy, Iz : float
        Second moments of area about the local y and z axes, m^4.
    J : float
        St Venant torsion constant, m^4. For a thin-walled circular tube
        ``J = Iy + Iz``.
    E : float
        Young's modulus, Pa.
    G : float
        Shear modulus, Pa.
    rho : float
        Structural mass density, kg/m^3.
    kappa : float
        Timoshenko shear coefficient, dimensionless. ``0`` disables shear
        deformation and recovers the Euler-Bernoulli element.
    added_mass_per_length : float
        Hydrodynamic added mass plus entrained/marine-growth mass, kg/m. Applied
        to translational inertia only. Zero for dry members.
    """

    A: float
    Iy: float
    Iz: float
    J: float
    E: float
    G: float
    rho: float
    kappa: float = 0.0
    added_mass_per_length: float = 0.0

    @property
    def mass_per_length(self) -> float:
        return self.rho * self.A + self.added_mass_per_length


def tubular_section(
    outer_diameter: float,
    wall_thickness: float,
    E: float,
    nu: float,
    rho: float,
    added_mass_per_length: float = 0.0,
    include_shear: bool = True,
) -> Section:
    """Section properties of a circular hollow section.

    Shear coefficient from Cowper (1966) Eq. 13 for a hollow circular section
    with inner/outer radius ratio ``m``:

    .. math::
        \\kappa = \\frac{6(1+\\nu)(1+m^2)^2}
                       {(7+6\\nu)(1+m^2)^2 + (20+12\\nu)m^2}
    """
    ro = 0.5 * outer_diameter
    ri = ro - wall_thickness
    if ri <= 0:
        raise ValueError("wall thickness exceeds tube radius")
    A = np.pi * (ro**2 - ri**2)
    I = 0.25 * np.pi * (ro**4 - ri**4)
    J = 2.0 * I
    G = E / (2.0 * (1.0 + nu))
    if include_shear:
        m = ri / ro
        kappa = (6.0 * (1.0 + nu) * (1.0 + m**2) ** 2) / (
            (7.0 + 6.0 * nu) * (1.0 + m**2) ** 2 + (20.0 + 12.0 * nu) * m**2
        )
    else:
        kappa = 0.0
    return Section(
        A=float(A),
        Iy=float(I),
        Iz=float(I),
        J=float(J),
        E=float(E),
        G=float(G),
        rho=float(rho),
        kappa=float(kappa),
        added_mass_per_length=float(added_mass_per_length),
    )


def local_stiffness(sec: Section, L: float) -> np.ndarray:
    """12x12 element stiffness in local coordinates.

    Przemieniecki (1968) Eq. 6.32. Shear flexibility enters through

    .. math::
        \\Phi_y = \\frac{12 E I_z}{\\kappa G A L^2}, \\qquad
        \\Phi_z = \\frac{12 E I_y}{\\kappa G A L^2}

    with :math:`\\Phi = 0` when ``sec.kappa == 0`` (Euler-Bernoulli limit).
    """
    if L <= 0:
        raise ValueError("element length must be positive")
    E, G, A, Iy, Iz, J, k = sec.E, sec.G, sec.A, sec.Iy, sec.Iz, sec.J, sec.kappa
    if k > 0:
        phi_y = 12.0 * E * Iz / (k * G * A * L * L)
        phi_z = 12.0 * E * Iy / (k * G * A * L * L)
    else:
        phi_y = phi_z = 0.0

    K = np.zeros((12, 12))
    ax = E * A / L
    tor = G * J / L
    K[0, 0] = K[6, 6] = ax
    K[0, 6] = -ax
    K[3, 3] = K[9, 9] = tor
    K[3, 9] = -tor

    # Bending in the local x-y plane (uy, rz) -> uses Iz.
    c = E * Iz / (L**3 * (1.0 + phi_y))
    b1 = 12.0 * c
    b2 = 6.0 * L * c
    b3 = (4.0 + phi_y) * L * L * c
    b4 = (2.0 - phi_y) * L * L * c
    K[1, 1] = K[7, 7] = b1
    K[1, 7] = -b1
    K[1, 5] = K[1, 11] = b2
    K[5, 7] = K[7, 11] = -b2
    K[5, 5] = K[11, 11] = b3
    K[5, 11] = b4

    # Bending in the local x-z plane (uz, ry) -> uses Iy; coupling signs flip.
    c = E * Iy / (L**3 * (1.0 + phi_z))
    d1 = 12.0 * c
    d2 = 6.0 * L * c
    d3 = (4.0 + phi_z) * L * L * c
    d4 = (2.0 - phi_z) * L * L * c
    K[2, 2] = K[8, 8] = d1
    K[2, 8] = -d1
    K[2, 4] = K[2, 10] = -d2
    K[4, 8] = K[8, 10] = d2
    K[4, 4] = K[10, 10] = d3
    K[4, 10] = d4

    return K + np.triu(K, 1).T - np.diag(np.diag(np.triu(K, 1).T))


def local_mass(sec: Section, L: float, lumped: bool = False) -> np.ndarray:
    """12x12 consistent (or lumped) element mass in local coordinates.

    Consistent form from Przemieniecki (1968) Eq. 11.28, Euler-Bernoulli shape
    functions. Rotary inertia of the cross-section is retained only in torsion.
    Hydrodynamic added mass acts on translational terms via
    ``sec.mass_per_length``.
    """
    mpl = sec.mass_per_length
    M = np.zeros((12, 12))
    if lumped:
        half = 0.5 * mpl * L
        rot = 0.5 * sec.rho * sec.J * L
        for i in (0, 1, 2, 6, 7, 8):
            M[i, i] = half
        M[3, 3] = M[9, 9] = rot
        return M

    m = mpl * L / 420.0
    # Axial
    a = mpl * L / 6.0
    M[0, 0] = M[6, 6] = 2.0 * a
    M[0, 6] = a
    # Torsion (uses structural material only; entrained water adds no polar term)
    t = sec.rho * sec.J * L / 6.0
    M[3, 3] = M[9, 9] = 2.0 * t
    M[3, 9] = t
    # Bending x-y (uy, rz)
    M[1, 1] = M[7, 7] = 156.0 * m
    M[1, 7] = 54.0 * m
    M[1, 5] = 22.0 * L * m
    M[1, 11] = -13.0 * L * m
    M[5, 7] = 13.0 * L * m
    M[7, 11] = -22.0 * L * m
    M[5, 5] = M[11, 11] = 4.0 * L * L * m
    M[5, 11] = -3.0 * L * L * m
    # Bending x-z (uz, ry)
    M[2, 2] = M[8, 8] = 156.0 * m
    M[2, 8] = 54.0 * m
    M[2, 4] = -22.0 * L * m
    M[2, 10] = 13.0 * L * m
    M[4, 8] = -13.0 * L * m
    M[8, 10] = 22.0 * L * m
    M[4, 4] = M[10, 10] = 4.0 * L * L * m
    M[4, 10] = -3.0 * L * L * m

    return M + np.triu(M, 1).T - np.diag(np.diag(np.triu(M, 1).T))


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors, by the textbook formula.

    ``np.cross`` is general over broadcasting and axes, and pays for it: on the
    profile of a full nuisance budget it drove 300k calls into ``moveaxis`` and
    ``normalize_axis_tuple`` inside a hot FE loop. For fixed length-3 vectors the
    explicit form is the same three products per component and carries none of
    that dispatch. The result is bit-for-bit what np.cross returns here.
    """
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def transformation(xi: np.ndarray, xj: np.ndarray, roll: float = 0.0) -> tuple[np.ndarray, float]:
    """Direction-cosine matrix (12x12) and element length.

    Local x is the i->j axis. The local z axis is chosen to lie in the vertical
    plane containing the member; for members within 1e-6 of vertical the global
    X axis is used as the reference instead, which is the standard degenerate
    case handling (Przemieniecki 1968, Section 6.2). ``roll`` rotates the
    section about the local x axis, radians.
    """
    d = np.asarray(xj, dtype=float) - np.asarray(xi, dtype=float)
    L = float(np.linalg.norm(d))
    if L <= 0:
        raise ValueError("coincident element nodes")
    ex = d / L
    ref = np.array([1.0, 0.0, 0.0]) if abs(ex[2]) > 1.0 - 1e-6 else np.array([0.0, 0.0, 1.0])
    ey = _cross3(ref, ex)
    ny = np.linalg.norm(ey)
    if ny < 1e-12:  # pragma: no cover - guarded by the reference switch above
        ref = np.array([0.0, 1.0, 0.0])
        ey = _cross3(ref, ex)
        ny = np.linalg.norm(ey)
    ey /= ny
    ez = _cross3(ex, ey)
    if roll:
        c, s = np.cos(roll), np.sin(roll)
        ey, ez = c * ey + s * ez, -s * ey + c * ez
    lam = np.vstack([ex, ey, ez])
    T = np.zeros((12, 12))
    for b in range(4):
        T[3 * b : 3 * b + 3, 3 * b : 3 * b + 3] = lam
    return T, L


def element_matrices(
    sec: Section,
    xi: np.ndarray,
    xj: np.ndarray,
    roll: float = 0.0,
    lumped_mass: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Global-frame element stiffness and mass, plus the transformation and length.

    Returns ``(Kg, Mg, T, L)`` where ``Kg = T^T K_local T``.
    """
    T, L = transformation(xi, xj, roll)
    kl = local_stiffness(sec, L)
    ml = local_mass(sec, L, lumped=lumped_mass)
    return T.T @ kl @ T, T.T @ ml @ T, T, L
