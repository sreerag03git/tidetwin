"""Quasi-static tidal strain response of the jacket, as a precomputed surface.

A full frame solve per time step per Monte Carlo sample would be far too slow
for the C3 nuisance budget or for a 15-second cold start. Two properties of the
problem make that unnecessary:

* The frame is linear, so response scales with load.
* Morison drag under a steady current is ``0.5 rho Cd D |u| u``: for a current of
  speed ``U`` in direction ``theta`` the load pattern depends only on ``theta``
  (and on the water level, through wetted length and the current profile), and
  its magnitude scales as ``U^2``. Reversing the current exactly negates the
  load, so ``eps(theta + pi) = -eps(theta)``.

So the response is precomputed on a ``(theta, eta)`` grid once per structural
configuration and interpolated afterwards. Every Monte Carlo sample then costs
a table lookup instead of a sparse solve. The grid is checked for interpolation
error in ``tests/test_response.py`` against direct solves at off-grid points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry.oc4 import JacketBuild, SensorPair
from .loads.buoyancy import assemble_buoyancy_load
from .loads.morison import HydroConfig, assemble_hydrodynamic_load, current_profile_factor

__all__ = [
    "ResponseSurface",
    "build_response_surface",
    "build_response_surfaces",
    "strain_series",
]


@dataclass(frozen=True)
class ResponseSurface:
    """Strain at a sensor pair as a function of current direction and water level.

    ``drag_*`` are strains per unit of ``U^2`` (i.e. for a 1 m/s surface current),
    ``buoy_*`` are strains from buoyancy at that water level, and ``still_*`` is
    the strain at mean water level with no current, which is subtracted so the
    tidal signal is what remains.
    """

    theta: np.ndarray  # rad, current direction (anticlockwise from +x)
    eta: np.ndarray  # m, water level
    drag_upper: np.ndarray  # (n_theta, n_eta)
    drag_lower: np.ndarray
    buoy_upper: np.ndarray  # (n_eta,)
    buoy_lower: np.ndarray
    still_upper: float
    still_lower: float
    pair: SensorPair
    n_solves: int

    def evaluate(
        self, speed: np.ndarray, direction: np.ndarray, water_level: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Strain at the two gauges. Arrays broadcast together."""
        U = np.asarray(speed, float)
        th = np.mod(np.asarray(direction, float), 2.0 * np.pi)
        et = np.clip(np.asarray(water_level, float), self.eta[0], self.eta[-1])
        u2 = U * np.abs(U)  # signed square: preserves the |u|u nonlinearity
        du = _bilinear_periodic(self.theta, self.eta, self.drag_upper, th, et)
        dl = _bilinear_periodic(self.theta, self.eta, self.drag_lower, th, et)
        bu = np.interp(et, self.eta, self.buoy_upper)
        bl = np.interp(et, self.eta, self.buoy_lower)
        return (
            du * u2 + bu - self.still_upper,
            dl * u2 + bl - self.still_lower,
        )


def _bilinear_periodic(
    theta: np.ndarray, eta: np.ndarray, z: np.ndarray, th: np.ndarray, et: np.ndarray
) -> np.ndarray:
    """Bilinear interpolation, periodic in theta and clamped in eta."""
    n_t = theta.size
    dth = theta[1] - theta[0]
    ft = np.mod(th - theta[0], 2.0 * np.pi) / dth
    i0 = np.floor(ft).astype(int) % n_t
    i1 = (i0 + 1) % n_t
    wt = ft - np.floor(ft)

    j0 = np.clip(np.searchsorted(eta, et, side="right") - 1, 0, eta.size - 2)
    j1 = j0 + 1
    we = (et - eta[j0]) / (eta[j1] - eta[j0])

    return (
        z[i0, j0] * (1 - wt) * (1 - we)
        + z[i1, j0] * wt * (1 - we)
        + z[i0, j1] * (1 - wt) * we
        + z[i1, j1] * wt * we
    )


def build_response_surface(
    build: JacketBuild,
    pair: SensorPair,
    cfg: HydroConfig,
    n_theta: int = 36,
    eta_levels: np.ndarray | None = None,
    flooded_legs: bool = True,
) -> ResponseSurface:
    """Precompute the strain response grid for one sensor pair.

    Thin wrapper over :func:`build_response_surfaces`; see there for the method.
    """
    return build_response_surfaces(
        build, [pair], cfg, n_theta=n_theta, eta_levels=eta_levels,
        flooded_legs=flooded_legs,
    )[0]


def build_response_surfaces(
    build: JacketBuild,
    pairs: list[SensorPair],
    cfg: HydroConfig,
    n_theta: int = 36,
    eta_levels: np.ndarray | None = None,
    flooded_legs: bool = True,
) -> list[ResponseSurface]:
    """Precompute the strain response grid for several sensor pairs at once.

    Uses one sparse LU factorisation of the stiffness matrix and reuses it for
    every right-hand side, which is what keeps the cost at seconds rather than
    minutes. Exploits ``eps(theta + pi) = -eps(theta)`` to halve the solve count.

    The key optimisation for a gauge rosette: the frame solve depends only on the
    loading, not on where the strain is read, so a single displacement solution
    feeds every pair. Reading strain at a gauge is a couple of array lookups;
    solving the frame is a sparse back-substitution. Building four pairs
    together therefore costs one solve set plus cheap strain recovery, not four
    solve sets - the ``n_solves`` reported is the same as for a single pair.
    """
    if not pairs:
        raise ValueError("build_response_surfaces needs at least one sensor pair")

    K, _ = build.model.assemble()
    free = build.model.free_dof()
    lu = spla.splu(sp.csc_matrix(K[free][:, free]))

    def solve(f: np.ndarray) -> np.ndarray:
        u = np.zeros(build.model.n_dof)
        u[free] = lu.solve(f[free])
        return u

    eta = (
        np.asarray(eta_levels, float)
        if eta_levels is not None
        else np.linspace(-3.0, 3.0, 7)
    )
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    half = n_theta // 2
    n_p = len(pairs)

    drag_u = np.zeros((n_p, n_theta, eta.size))
    drag_l = np.zeros((n_p, n_theta, eta.size))
    buoy_u = np.zeros((n_p, eta.size))
    buoy_l = np.zeros((n_p, eta.size))
    n_solves = 0

    for j, e in enumerate(eta):
        cfg_e = HydroConfig(
            water_density=cfg.water_density,
            roughness_m=cfg.roughness_m,
            current_exponent=cfg.current_exponent,
            water_depth=cfg.water_depth,
            surface_elevation=float(e),
            marine_growth_mm=cfg.marine_growth_mm,
        )
        fb = assemble_buoyancy_load(build, float(e), cfg.water_density, flooded=flooded_legs)
        ub = solve(fb)
        n_solves += 1
        for p, pair in enumerate(pairs):
            buoy_u[p, j], buoy_l[p, j] = build.pair_strains(ub, pair)

        for i in range(half):
            th = float(theta[i])
            direction = np.array([np.cos(th), np.sin(th), 0.0])

            def vel(z, _p, d=direction, c=cfg_e):
                return d * float(
                    current_profile_factor(z, c.water_depth, c.surface_elevation, c.current_exponent)
                )

            f = assemble_hydrodynamic_load(build, vel, cfg_e)
            ud = solve(f)
            n_solves += 1
            for p, pair in enumerate(pairs):
                du, dl = build.pair_strains(ud, pair)
                drag_u[p, i, j], drag_l[p, i, j] = du, dl
                # Reversing the current negates the drag load exactly.
                drag_u[p, i + half, j], drag_l[p, i + half, j] = -du, -dl

    # Reference state: mean water level, no current.
    j_ref = int(np.argmin(np.abs(eta)))
    return [
        ResponseSurface(
            theta=theta,
            eta=eta,
            drag_upper=drag_u[p],
            drag_lower=drag_l[p],
            buoy_upper=buoy_u[p],
            buoy_lower=buoy_l[p],
            still_upper=float(buoy_u[p, j_ref]),
            still_lower=float(buoy_l[p, j_ref]),
            pair=pair,
            n_solves=n_solves,
        )
        for p, pair in enumerate(pairs)
    ]


def strain_series(
    surface: ResponseSurface,
    times_s: np.ndarray,
    constituents,
    current_scale: float = 1.0,
    extra_current: np.ndarray | None = None,
    extra_water_level: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Strain time series at both gauges under tidal (and optional extra) forcing.

    ``extra_current`` is an ``(n_t, 2)`` eastward/northward residual added to the
    tidal current before the drag is evaluated - the nonlinearity means a wind
    drift cannot simply be superposed on the answer, it has to enter here.
    """
    t = np.asarray(times_s, float)
    uv = constituents.depth_averaged_current(t) * current_scale
    if extra_current is not None:
        uv = uv + np.asarray(extra_current, float).reshape(uv.shape)
    eta = constituents.elevation(t)
    if extra_water_level is not None:
        eta = eta + np.asarray(extra_water_level, float).ravel()

    speed = np.hypot(uv[:, 0], uv[:, 1])
    direction = np.arctan2(uv[:, 1], uv[:, 0])
    return surface.evaluate(speed, direction, eta)
