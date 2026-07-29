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
from ...fe.ljf import LJFModel, shell_ljf
from ...geometry.oc4 import SensorPair, brace_chord_joints, build_jacket, load_tables
from ...loads.morison import HydroConfig
from ...loads.tides import TidalConstituents
from ...nuisance import ratio_from_series
from ...response import build_response_surface, strain_series

__all__ = ["DamageGrid", "damage_sensitivity_grid", "damage_signature",
           "StiffnessReductionResult", "stiffness_reduction_test"]


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


@dataclass
class StiffnessReductionResult:
    """The abstract's own intermediate step, tested directly.

    The paper states a chain: a 20 percent through-wall crack produces a 10
    percent joint stiffness reduction, which takes the strain ratio from 1.800 to
    2.000. The second link is pure structural mechanics. Imposing exactly the
    stated stiffness reduction and re-solving tests it with no crack model, no
    shell-FE surface and no fracture mechanics in the way - so the result depends
    on nothing this application chose.
    """

    reductions: np.ndarray
    ratios: np.ndarray
    intact_ratio: float
    claimed_intact: float
    claimed_damaged: float
    claimed_signature: float
    joint_id: int
    brace_member: int

    @property
    def at_claimed_reduction(self) -> float:
        """Fractional ratio change at the stiffness reduction the paper states."""
        i = int(np.argmin(np.abs(self.reductions - 0.10)))
        return float((self.ratios[i] - self.intact_ratio) / self.intact_ratio)

    @property
    def required_reduction_for_claim(self) -> float:
        """Stiffness reduction that would actually produce the claimed 11.1 percent.

        ``nan`` if no reduction on the swept range gets there - which is itself
        the finding, since the range runs to complete loss of the joint spring.
        """
        change = np.abs((self.ratios - self.intact_ratio) / self.intact_ratio)
        if not np.any(change >= self.claimed_signature):
            return float("nan")
        return float(np.interp(self.claimed_signature, change, self.reductions))

    @property
    def verdict(self) -> str:
        got = self.at_claimed_reduction
        need = self.required_reduction_for_claim
        head = (
            f"A {self.claimed_signature * 100:.1f} percent ratio change is claimed for a 10 "
            f"percent joint stiffness reduction. Imposing exactly that reduction gives "
            f"{got * 100:.3f} percent."
        )
        if not np.isfinite(need):
            return head + (
                " Removing the joint spring entirely does not reach the claimed change, so "
                "no stiffness reduction of any size produces it. The claimed sensitivity "
                "does not follow from the structural mechanics, independently of how the "
                "crack itself is modelled."
            )
        return head + (
            f" Reaching {self.claimed_signature * 100:.1f} percent would need a "
            f"{need * 100:.0f} percent stiffness reduction, not 10 percent."
        )


def stiffness_reduction_test(
    pair: SensorPair,
    constituents,
    cfg: HydroConfig,
    joint_id: int,
    brace_member: int | None = None,
    reductions: np.ndarray | None = None,
    ljf_model: LJFModel = LJFModel.SHELL,
    record_days: float = 14.0,
    n_theta: int = 12,
    claimed_intact: float = 1.800,
    claimed_damaged: float = 2.000,
) -> StiffnessReductionResult:
    """Sweep the joint stiffness reduction and record the strain ratio.

    ``reductions`` are fractions of the intact joint stiffness removed, 0 to 1.
    A value of 1 removes the joint spring completely, which bounds what any crack
    could possibly do.
    """
    if ljf_model is LJFModel.RIGID:
        raise ValueError(
            "A stiffness reduction test needs a joint stiffness to reduce. Rigid joints "
            "have none, so the answer would be identically zero for every reduction."
        )
    tables = load_tables()
    kj = brace_chord_joints(tables)
    if joint_id not in kj:
        raise ValueError(f"joint {joint_id} is not a braced leg joint")
    brace_member = brace_member if brace_member is not None else kj[joint_id][0]
    reductions = np.asarray(
        reductions if reductions is not None
        else np.array([0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 0.90, 0.99]),
        float,
    )
    t = np.arange(0.0, record_days * 86400.0, 1800.0)

    def ratio_for(reduction: float) -> float:
        # Compliance in series: removing a fraction f of the stiffness means the
        # remaining stiffness is (1-f)k, i.e. an added compliance of f/((1-f)k).
        if reduction <= 0.0:
            extra = None
        else:
            geom = _joint_geometry(tables, brace_member, joint_id)
            k = shell_ljf(geom).k_axial
            f = min(reduction, 0.999)
            extra = {brace_member: (f / ((1.0 - f) * k), 0.0, 0.0)}
        b = build_jacket(ljf_model=ljf_model, crack_compliance=extra, tables=tables)
        s = build_response_surface(b, pair, cfg, n_theta=n_theta,
                                   eta_levels=np.linspace(-2, 2, 3))
        eu, el = strain_series(s, t, constituents)
        return ratio_from_series(t, eu, el)

    ratios = np.array([ratio_for(float(r)) for r in reductions])
    return StiffnessReductionResult(
        reductions=reductions,
        ratios=ratios,
        intact_ratio=float(ratios[0]),
        claimed_intact=claimed_intact,
        claimed_damaged=claimed_damaged,
        claimed_signature=abs(claimed_damaged - claimed_intact) / claimed_intact,
        joint_id=joint_id,
        brace_member=int(brace_member),
    )


def _joint_geometry(tables, brace_member: int, joint_id: int):
    from ...geometry.oc4 import _joint_frame

    _axes, geom = _joint_frame(
        tables, brace_member, joint_id, int(tables.members.loc[brace_member, "prop_set"])
    )
    return geom


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
