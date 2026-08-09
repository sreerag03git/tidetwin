"""Animated simulation of the jacket responding through a tidal cycle.

Not a cartoon. Each frame is a real static solve of the 3D frame under the
Morison load from the tidal current at that instant, using the same assembly,
the same coefficients and the same sparse factorisation as every claim in the
ledger. The only cosmetic step is a displacement exaggeration factor, which is
stated on the figure, because the true deflections are millimetres on a 70 m
structure and would otherwise be invisible.

What it shows, and why it is worth watching rather than just reading:

* the jacket leans with the current and the lean *rotates* through the cycle,
  because the tidal current traces an ellipse rather than reversing along a line;
* the two bracketing gauges swing in step, which is the common-mode motion the
  strain ratio is supposed to divide out;
* at a reversing-current site the deflection passes through zero twice a cycle,
  which is exactly where the strain ratio's denominator vanishes and its variance
  ceases to exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry.oc4 import JacketBuild, SensorPair
from .loads.buoyancy import assemble_buoyancy_load
from .loads.morison import HydroConfig, assemble_hydrodynamic_load, current_profile_factor

__all__ = ["TidalCycle", "simulate_cycle"]


@dataclass
class TidalCycle:
    """Displaced shapes and gauge strains over one tidal cycle."""

    hours: np.ndarray
    nodes_undeformed: np.ndarray  # (n_nodes, 3)
    displaced: np.ndarray  # (n_frames, n_nodes, 3), already exaggerated
    tip_deflection_mm: np.ndarray  # (n_frames,) true, not exaggerated
    current_u: np.ndarray
    current_v: np.ndarray
    elevation: np.ndarray
    strain_upper: np.ndarray  # microstrain
    strain_lower: np.ndarray
    exaggeration: float
    n_solves: int

    @property
    def max_true_deflection_mm(self) -> float:
        return float(np.max(np.abs(self.tip_deflection_mm)))


def simulate_cycle(
    build: JacketBuild,
    pair: SensorPair,
    constituents,
    cfg: HydroConfig,
    n_frames: int = 36,
    constituent: str = "M2",
    exaggeration: float | None = None,
    target_visible_m: float = 2.5,
) -> TidalCycle:
    """Solve the frame at ``n_frames`` phases of one tidal cycle.

    ``exaggeration`` defaults to whatever makes the largest deflection about
    ``target_visible_m`` on a 70 m structure, so the motion reads at a glance.
    The factor is returned and must be displayed alongside the animation - an
    unlabelled exaggerated deflection is a misleading picture.
    """
    from .loads.tides import constituent_period_hours

    period_h = constituent_period_hours(constituent)
    hours = np.linspace(0.0, period_h, n_frames, endpoint=False)
    t_s = hours * 3600.0

    uv = constituents.depth_averaged_current(t_s)
    eta = constituents.elevation(t_s)

    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    K, _ = build.model.assemble()
    free = build.model.free_dof()
    lu = spla.splu(sp.csc_matrix(K[free][:, free]))

    n_nodes = build.model.n_nodes
    undeformed = build.model.nodes.copy()
    raw = np.zeros((n_frames, n_nodes, 3))
    eu = np.zeros(n_frames)
    el = np.zeros(n_frames)
    solves = 0

    for k in range(n_frames):
        speed = float(np.hypot(uv[k, 0], uv[k, 1]))
        direction = np.array([uv[k, 0], uv[k, 1], 0.0])
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 0 else np.zeros(3)

        c = HydroConfig(
            water_density=cfg.water_density,
            roughness_m=cfg.roughness_m,
            current_exponent=cfg.current_exponent,
            water_depth=cfg.water_depth,
            surface_elevation=float(eta[k]),
            marine_growth_mm=cfg.marine_growth_mm,
        )

        def vel(z, _p, d=direction, s=speed, cc=c):
            return d * s * float(
                current_profile_factor(z, cc.water_depth, cc.surface_elevation,
                                       cc.current_exponent)
            )

        f = assemble_hydrodynamic_load(build, vel, c)
        f = f + assemble_buoyancy_load(build, float(eta[k]), c.water_density, flooded=True)
        u = np.zeros(build.model.n_dof)
        u[free] = lu.solve(f[free])
        solves += 1

        d = u.reshape(-1, 6)[:, :3]
        raw[k] = d
        a, b = build.pair_strains(u, pair)
        eu[k], el[k] = a * 1e6, b * 1e6

    # Deflection of the highest node, in true millimetres.
    top = int(np.argmax(undeformed[:, 2]))
    tip_mm = np.linalg.norm(raw[:, top, :], axis=1) * 1e3

    peak = float(np.max(np.linalg.norm(raw.reshape(-1, 3), axis=1)))
    if exaggeration is None:
        exaggeration = float(target_visible_m / peak) if peak > 0 else 1.0

    return TidalCycle(
        hours=hours,
        nodes_undeformed=undeformed,
        displaced=undeformed[None, :, :] + raw * exaggeration,
        tip_deflection_mm=tip_mm,
        current_u=uv[:, 0],
        current_v=uv[:, 1],
        elevation=eta,
        strain_upper=eu,
        strain_lower=el,
        exaggeration=exaggeration,
        n_solves=solves,
    )
