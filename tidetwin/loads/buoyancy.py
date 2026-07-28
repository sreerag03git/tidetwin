"""Tidal water level to buoyancy and hydrostatic loading.

As the tide rises and falls the wetted length of every member near the splash
zone changes, and with it the net buoyant uplift the frame carries. On the OC4
jacket the legs run continuously through the still water level and the third
X-brace level crosses at z = -1.958 m, so a metre or two of tide directly
modulates the wetted length of members close to the surface.

This channel matters for the claims because it is *in phase with elevation*,
whereas the tidal current is in quadrature with it. A strain ratio taken from a
single harmonic without separating the two will mix them. The quadrature
decomposition in :mod:`tidetwin.signal.quadrature` exists for that reason.
"""

from __future__ import annotations

import numpy as np

from ..provenance import Citation

__all__ = ["G", "ARCHIMEDES", "wetted_fraction", "assemble_buoyancy_load"]

G = 9.80665  # m/s^2, standard gravity (BIPM SI Brochure, 9th ed., 2019)

ARCHIMEDES = Citation(
    document="ISO 19902:2020 Petroleum and natural gas industries - Fixed steel offshore structures",
    locator="Section 9.3 (hydrostatic pressure and buoyancy on submerged members)",
    year=2020,
)


def wetted_fraction(z_i: float, z_j: float, surface_elevation: float) -> float:
    """Fraction of a member's length below the free surface, 0 to 1."""
    z_lo, z_hi = (z_i, z_j) if z_i <= z_j else (z_j, z_i)
    if z_hi <= surface_elevation:
        return 1.0
    if z_lo >= surface_elevation:
        return 0.0
    return float((surface_elevation - z_lo) / (z_hi - z_lo))


def assemble_buoyancy_load(
    build,
    surface_elevation: float,
    water_density: float = 1025.0,
    flooded: bool = False,
) -> np.ndarray:
    """Global load vector from buoyant uplift at a given water level.

    Buoyancy is applied as a distributed upward force over the wetted length,

        w = rho_w * g * A_displaced   [N/m],

    with ``A_displaced`` the full outside area ``pi D^2/4`` for a sealed member
    and only the steel annulus for a flooded one. Jacket legs are normally
    flooded and braces sealed; ``flooded`` sets the assumption for all members at
    once, and which way it is set is recorded on dependent claims because it
    changes the *magnitude* of the tidal buoyancy signal by roughly the ratio of
    annulus area to full area (about 7 percent for a 1.2 m x 35 mm leg).

    Consistent nodal loads are used, so the moment transfer at member ends is
    handled the same way as for the Morison loads.
    """
    from .morison import _consistent_nodal_loads

    model = build.model
    f = np.zeros(model.n_dof)
    for mid, mem in build.member_of.items():
        pi = model.nodes[mem.node_i]
        pj = model.nodes[mem.node_j]
        frac = wetted_fraction(float(pi[2]), float(pj[2]), surface_elevation)
        if frac <= 0.0:
            continue
        row = build.tables.sections.loc[mem.group]
        D = float(row.outer_diameter_m)
        area = mem.section.A if flooded else np.pi * D**2 / 4.0
        w = np.array([0.0, 0.0, water_density * G * area]) * frac
        fe = _consistent_nodal_loads(model, mem.node_i, mem.node_j, mem.roll, w)
        idx = np.concatenate(
            [
                np.arange(mem.node_i * 6, mem.node_i * 6 + 6),
                np.arange(mem.node_j * 6, mem.node_j * 6 + 6),
            ]
        )
        f[idx] += fe
    return f
