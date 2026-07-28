"""C2 - damage sensitivity of the intact strain ratio.

The abstract claims an 11.1 percent change. This module computes the change over
a 2D grid of ``(a/T, 2c)`` rather than reporting a single number, because a
single number hides the fact that a deep narrow flaw and a shallow long one with
the same through-wall fraction do very different things to the joint compliance.

The crack is placed on the chord side of one brace footprint at the target
joint, converted to an added local compliance by
:func:`tidetwin.damage.crack_ljf.crack_compliance`, and put in series with that
brace's LJF spring. The frame is then re-solved and the M2 strain ratio
recomputed exactly as C1 computes it, so the two are directly comparable.

Local joint flexibility must be modelled for this to mean anything: with rigid
joints there is no spring for the crack to soften, and the computed sensitivity
would be identically zero. The functions here refuse to run in that case rather
than returning a misleading zero.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...damage.crack_ljf import CrackGeometry, crack_compliance, shell_fe_status
from ...fe.ljf import LJFModel
from ...geometry.oc4 import SensorPair, brace_chord_joints, build_jacket, load_tables
from ...loads.morison import HydroConfig
from ...loads.tides import TidalConstituents
from ...nuisance import ratio_from_series
from ...response import build_response_surface, strain_series

__all__ = ["DamageGrid", "damage_sensitivity_grid", "damage_signature"]


@dataclass
class DamageGrid:
    """Fractional change in the M2 strain ratio over a crack-geometry grid."""

    a_over_T: np.ndarray
    surface_length_m: np.ndarray
    ratio: np.ndarray  # (n_aT, n_2c) absolute ratio
    delta_fraction: np.ndarray  # (n_aT, n_2c) (ratio - intact)/intact
    intact_ratio: float
    joint_id: int
    brace_member: int
    route: str
    caveats: tuple[str, ...]
    n_masked: int = 0

    def signature_at(self, a_over_T: float, surface_length_m: float) -> float:
        """Magnitude of the fractional ratio change at one crack geometry.

        Returns ``nan`` if that geometry falls in the masked ``a/c > 1`` region.
        """
        i = int(np.argmin(np.abs(self.a_over_T - a_over_T)))
        j = int(np.argmin(np.abs(self.surface_length_m - surface_length_m)))
        return float(abs(self.delta_fraction[i, j]))

    @property
    def max_signature(self) -> float:
        if np.all(np.isnan(self.delta_fraction)):
            return float("nan")
        return float(np.nanmax(np.abs(self.delta_fraction)))


def damage_sensitivity_grid(
    pair: SensorPair,
    constituents: TidalConstituents,
    cfg: HydroConfig,
    joint_id: int,
    brace_member: int | None = None,
    a_over_T: np.ndarray | None = None,
    surface_length_m: np.ndarray | None = None,
    ljf_model: LJFModel = LJFModel.SHELL,
    record_days: float = 14.0,
    sample_interval_s: float = 1800.0,
    n_theta: int = 12,
    progress=None,
) -> DamageGrid:
    """Strain-ratio change across a grid of crack geometries.

    Raises
    ------
    ValueError
        If ``ljf_model`` is RIGID, where the answer would be a meaningless zero.
    """
    if ljf_model is LJFModel.RIGID:
        raise ValueError(
            "C2 requires a local joint flexibility model: with rigid joints there is no "
            "joint compliance for a crack to change, and the computed sensitivity would "
            "be identically zero for every crack size. Select SHELL or TABULATED."
        )

    tables = load_tables()
    kj = brace_chord_joints(tables)
    if joint_id not in kj:
        raise ValueError(f"joint {joint_id} is not a braced leg joint; choose from {sorted(kj)}")
    brace_member = brace_member if brace_member is not None else kj[joint_id][0]

    a_over_T = np.asarray(a_over_T if a_over_T is not None else np.linspace(0.1, 0.8, 6), float)
    surface_length_m = np.asarray(
        surface_length_m if surface_length_m is not None else np.linspace(0.02, 0.20, 6), float
    )

    chord_T = float(
        tables.sections.loc[
            int(tables.members.loc[_leg_member_at(tables, joint_id), "prop_set"]),
            "wall_thickness_m",
        ]
    )
    brace_d = float(
        tables.sections.loc[int(tables.members.loc[brace_member, "prop_set"]), "outer_diameter_m"]
    )

    t = np.arange(0.0, record_days * 86400.0, sample_interval_s)

    def ratio_for(compliance: dict | None) -> float:
        b = build_jacket(ljf_model=ljf_model, crack_compliance=compliance, tables=tables)
        s = build_response_surface(b, pair, cfg, n_theta=n_theta, eta_levels=np.linspace(-2, 2, 3))
        eu, el = strain_series(s, t, constituents)
        return ratio_from_series(t, eu, el)

    intact = ratio_for(None)

    ratios = np.zeros((a_over_T.size, surface_length_m.size))
    total = a_over_T.size * surface_length_m.size
    k = 0
    n_masked = 0
    for i, aT in enumerate(a_over_T):
        for j, L2c in enumerate(surface_length_m):
            crack = CrackGeometry(a=float(aT) * chord_T, c=0.5 * float(L2c), T=chord_T)
            try:
                dC, _q = crack_compliance(crack, load_width=brace_d)
            except ValueError:
                # a/c > 1: outside the Newman-Raju validity envelope. Masked rather
                # than extrapolated, because the fitted M3 term diverges there.
                ratios[i, j] = np.nan
                n_masked += 1
            else:
                ratios[i, j] = ratio_for({brace_member: (dC, 0.0, 0.0)})
            k += 1
            if progress:
                progress(k / total, f"crack a/T={aT:.2f}, 2c={L2c * 1e3:.0f} mm")

    shell_ok, shell_why = shell_fe_status()
    caveats = [
        "line-spring compliance route (no shell-FE surface shipped): expected to "
        "under-predict the LJF change, making this signature a lower bound",
        "experimental overlays from Soh (2000) and Rhee (2005) are not shipped, so the "
        "comparison against measured LJF degradation is UNTESTABLE - DATA MISSING",
    ]
    if shell_ok:
        caveats = ["shell-FE surface present; see its provenance card"]
    if n_masked:
        caveats.append(
            f"{n_masked} of {total} grid cells have a/c > 1 and are masked: the "
            "Newman-Raju boundary-correction factor is fitted only for a/c <= 1 and its "
            "M3 term diverges beyond it. Those cells are not computed rather than "
            "extrapolated."
        )

    return DamageGrid(
        a_over_T=a_over_T,
        surface_length_m=surface_length_m,
        ratio=ratios,
        delta_fraction=(ratios - intact) / intact,
        intact_ratio=float(intact),
        joint_id=joint_id,
        brace_member=int(brace_member),
        route="shell-FE surface" if shell_ok else "line-spring compliance",
        caveats=tuple(caveats),
        n_masked=n_masked,
    )


def damage_signature(
    grid: DamageGrid, a_over_T: float = 0.5, surface_length_m: float = 0.10
) -> float:
    """The reference damage signature C3's verdict is measured against.

    Taken at a mid-range crack - half through-wall, 100 mm long - rather than at
    the largest crack on the grid. Using the largest would flatter the method by
    comparing the nuisance floor against a flaw that would already be found by a
    routine inspection.
    """
    return grid.signature_at(a_over_T, surface_length_m)


def _leg_member_at(tables, joint_id: int) -> int:
    for mid, m in tables.members.iterrows():
        if int(m.prop_set) in (2, 3, 4) and joint_id in (int(m.joint_i), int(m.joint_j)):
            return int(mid)
    raise ValueError(f"no leg member at joint {joint_id}")
