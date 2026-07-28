"""Morison hydrodynamic loading on the jacket frame.

Force per unit length on a slender cylinder, normal to its axis
(Morison, O'Brien, Johnson & Schaaf, "The force exerted by surface waves on
piles", Petroleum Transactions AIME 189:149-154, 1950):

.. math::
    f_n = \\tfrac{1}{2}\\rho C_d D |u_n| u_n + \\rho C_m \\frac{\\pi D^2}{4} \\dot{u}_n

Only the velocity component normal to the member axis contributes; the tangential
component is dropped, which is the standard treatment for tubulars
(API RP 2A-WSD Section 2.3.1b, ISO 19902:2020 Section 9.5).

Tidal forcing is quasi-static: at an M2 period of 12.42 h the fluid acceleration
term is smaller than the drag term by roughly ``omega D / u ~ 1e-3``, and the
structure's fundamental period (order 0.4 s) is five orders of magnitude shorter
than the forcing, so no dynamic amplification arises. The inertia term is
retained anyway so the same code serves the wave-offset nuisance case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fe.assemble import Model
from ..fe.beam3d import transformation
from ..provenance import Citation

__all__ = [
    "API_RP2A",
    "DNV_RP_C205",
    "drag_inertia_coefficients",
    "current_profile_factor",
    "marine_growth_thickness",
    "member_distributed_force",
    "assemble_hydrodynamic_load",
    "HydroConfig",
]

API_RP2A = Citation(
    document="API RP 2A-WSD, Recommended Practice for Planning, Designing and Constructing "
    "Fixed Offshore Platforms - Working Stress Design",
    locator="Section 2.3.1b.4 (drag and inertia coefficients: smooth Cd 0.65 / Cm 1.6, "
    "rough Cd 1.05 / Cm 1.2)",
    year=2014,
)
DNV_RP_C205 = Citation(
    document="DNV-RP-C205, Environmental Conditions and Environmental Loads",
    locator="Section 6.7 (roughness dependence of drag), Section 4.1.2 (current profile), "
    "Section 6.7.3 (marine growth)",
    year=2021,
)

# Relative roughness bounds over which API's smooth and rough values apply.
_SMOOTH_KD = 1.0e-4
_ROUGH_KD = 1.0e-2


@dataclass(frozen=True)
class HydroConfig:
    """Fluid and coefficient settings for a hydrodynamic load case."""

    water_density: float = 1025.0
    roughness_m: float = 0.05
    current_exponent: float = 1.0 / 7.0
    water_depth: float = 50.001
    surface_elevation: float = 0.0
    marine_growth_mm: float = 0.0


def drag_inertia_coefficients(
    diameter: float, roughness_m: float
) -> tuple[float, float, str]:
    """Steady-flow ``(Cd, Cm)`` for a tubular at post-critical Reynolds number.

    API RP 2A-WSD Section 2.3.1b.4 gives ``Cd = 0.65, Cm = 1.6`` for smooth
    members and ``Cd = 1.05, Cm = 1.2`` for rough ones. Between the two the
    coefficients are interpolated linearly in ``log10(k/D)`` across the decades
    from ``k/D = 1e-4`` to ``1e-2``, which is the transition band DNV-RP-C205
    Section 6.7 describes. Values are clamped outside that band rather than
    extrapolated.
    """
    if diameter <= 0:
        raise ValueError("diameter must be positive")
    kd = max(roughness_m, 0.0) / diameter
    if kd <= _SMOOTH_KD:
        return 0.65, 1.6, "smooth (API RP 2A-WSD)"
    if kd >= _ROUGH_KD:
        return 1.05, 1.2, "fully rough (API RP 2A-WSD)"
    w = (np.log10(kd) - np.log10(_SMOOTH_KD)) / (np.log10(_ROUGH_KD) - np.log10(_SMOOTH_KD))
    return (
        float(0.65 + w * (1.05 - 0.65)),
        float(1.6 + w * (1.2 - 1.6)),
        f"transitional, k/D = {kd:.2e} (interpolated per DNV-RP-C205 Sec. 6.7)",
    )


def current_profile_factor(
    z: np.ndarray | float,
    water_depth: float,
    surface_elevation: float = 0.0,
    exponent: float = 1.0 / 7.0,
) -> np.ndarray:
    """Tidal current shape function, normalised to 1 at the free surface.

    Power-law profile ``(h/d)^(1/7)`` measured from the seabed, per
    DNV-RP-C205 Section 4.1.2 and API RP 2A-WSD Section 2.3.1c. Returns zero
    above the instantaneous free surface, so members emerging on a falling tide
    stop being loaded.
    """
    z = np.asarray(z, dtype=float)
    seabed = surface_elevation - water_depth
    height = z - seabed
    total = max(water_depth, 1e-6)
    f = np.clip(height / total, 0.0, None) ** exponent
    return np.where(z <= surface_elevation, f, 0.0)


def marine_growth_thickness(
    z: np.ndarray | float,
    profile: tuple[tuple[float, float, float], ...] | None = None,
) -> np.ndarray:
    """Depth-varying marine growth thickness in metres.

    ``profile`` is a tuple of ``(z_top, z_bottom, thickness_m)`` bands. The
    default is the DNV-RP-C205 Section 6.7.3 North Sea profile for 56-59 deg N:
    100 mm from +2 m to -40 m, 50 mm below -40 m.

    **Regional caveat.** That profile is North Sea data. It is the default only
    because it is the one published in a standard this project already cites; it
    is *not* representative of Arabian Gulf fouling, where warmer water and
    different species give different rates. For a Gulf site, supply a local
    profile - and note that doing so makes the value ASSUMED unless it comes from
    a citable survey.
    """
    z = np.asarray(z, dtype=float)
    bands = profile or ((2.0, -40.0, 0.100), (-40.0, -200.0, 0.050))
    out = np.zeros_like(z, dtype=float)
    for z_top, z_bot, t in bands:
        out = np.where((z <= z_top) & (z > z_bot), t, out)
    return out


def member_distributed_force(
    p_i: np.ndarray,
    p_j: np.ndarray,
    diameter: float,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    cfg: HydroConfig,
) -> np.ndarray:
    """Morison force per unit length on one member, global axes, N/m.

    ``velocity`` and ``acceleration`` are the fluid kinematics at the member
    mid-point in global axes. Only their components normal to the member axis
    are used.
    """
    axis = np.asarray(p_j, float) - np.asarray(p_i, float)
    L = np.linalg.norm(axis)
    if L <= 0:
        raise ValueError("zero-length member")
    e = axis / L
    u = np.asarray(velocity, float)
    a = np.asarray(acceleration, float)
    un = u - np.dot(u, e) * e
    an = a - np.dot(a, e) * e

    d_hydro = diameter + 2.0 * cfg.marine_growth_mm * 1e-3
    cd, cm, _ = drag_inertia_coefficients(d_hydro, cfg.roughness_m)
    drag = 0.5 * cfg.water_density * cd * d_hydro * np.linalg.norm(un) * un
    inertia = cfg.water_density * cm * np.pi * d_hydro**2 / 4.0 * an
    return drag + inertia


def _consistent_nodal_loads(
    model: Model, node_i: int, node_j: int, roll: float, w_global: np.ndarray
) -> np.ndarray:
    """Consistent nodal load vector (12,) for a uniform load ``w`` (N/m), global.

    For a uniform transverse load ``w`` on a beam of length ``L`` the consistent
    end actions are ``wL/2`` and ``+/- wL^2/12`` (Przemieniecki 1968, Table 5.2).
    Axial load is split evenly. Working in local axes and transforming back keeps
    the moment terms attached to the correct rotational DOF.
    """
    T, L = transformation(model.nodes[node_i], model.nodes[node_j], roll)
    lam = T[:3, :3]
    w_local = lam @ np.asarray(w_global, float)
    wx, wy, wz = w_local
    f = np.zeros(12)
    f[0] = f[6] = wx * L / 2.0
    f[1] = f[7] = wy * L / 2.0
    f[2] = f[8] = wz * L / 2.0
    f[5] = wy * L**2 / 12.0
    f[11] = -wy * L**2 / 12.0
    f[4] = -wz * L**2 / 12.0
    f[10] = wz * L**2 / 12.0
    return T.T @ f


def assemble_hydrodynamic_load(
    build,
    velocity_at: callable,
    cfg: HydroConfig,
    acceleration_at: callable | None = None,
) -> np.ndarray:
    """Global load vector from Morison forces on every submerged member.

    ``velocity_at(z, midpoint)`` returns the fluid velocity vector (global, m/s)
    at elevation ``z``; ``acceleration_at`` likewise, defaulting to zero (the
    quasi-static tidal case).

    Members straddling the free surface are integrated only over their submerged
    length, by scaling the distributed load by the wetted fraction. That matters:
    the splash zone is where the tidal water-level change does most of its work
    on the load, and treating a member as fully wet or fully dry would put a
    discontinuity into the very signal C1 and C3 are measuring.
    """
    model = build.model
    f = np.zeros(model.n_dof)
    eta = cfg.surface_elevation
    for mid in build.submerged_members:
        mem = build.member_of[mid]
        pi = model.nodes[mem.node_i]
        pj = model.nodes[mem.node_j]
        z_lo, z_hi = min(pi[2], pj[2]), max(pi[2], pj[2])
        if z_lo >= eta:
            continue
        wet = 1.0 if z_hi <= eta else (eta - z_lo) / max(z_hi - z_lo, 1e-9)
        z_mid_wet = 0.5 * (z_lo + min(z_hi, eta))
        mid_point = 0.5 * (pi + pj)
        D = float(build.tables.sections.loc[mem.group, "outer_diameter_m"])
        u = np.asarray(velocity_at(z_mid_wet, mid_point), float)
        a = (
            np.asarray(acceleration_at(z_mid_wet, mid_point), float)
            if acceleration_at is not None
            else np.zeros(3)
        )
        w = member_distributed_force(pi, pj, D, u, a, cfg) * wet
        fe = _consistent_nodal_loads(model, mem.node_i, mem.node_j, mem.roll, w)
        idx = np.concatenate(
            [
                np.arange(mem.node_i * 6, mem.node_i * 6 + 6),
                np.arange(mem.node_j * 6, mem.node_j * 6 + 6),
            ]
        )
        f[idx] += fe
    return f
