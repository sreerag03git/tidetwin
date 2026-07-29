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
from ...fe.ljf import LJFModel, LJFStiffness, shell_ljf
from ...geometry.oc4 import SensorPair, brace_chord_joints, build_jacket, load_tables
from ...loads.morison import HydroConfig
from ...loads.tides import TidalConstituents
from ...nuisance import ratio_from_series
from ...response import build_response_surface, strain_series

__all__ = ["DamageGrid", "damage_sensitivity_grid", "damage_signature",
           "StiffnessReductionResult", "stiffness_reduction_test", "ModeSet", "MODE_SETS"]


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


@dataclass(frozen=True)
class ModeSet:
    """One reading of the phrase "10 percent joint stiffness reduction"."""

    key: str
    label: str
    #: Which of the three LJF springs this reading softens.
    axial: bool
    ipb: bool
    opb: bool


#: The abstract states a stiffness reduction without saying which stiffness. A
#: cracked chord wall softens all three local springs, but not equally, and they
#: do not push the strain ratio the same way. Every reading is swept and the one
#: most favourable to the claim is the one judged.
MODE_SETS: dict[str, ModeSet] = {
    "axial": ModeSet("axial", "the axial spring alone", True, False, False),
    "ipb": ModeSet("ipb", "in-plane bending alone", False, True, False),
    "opb": ModeSet("opb", "out-of-plane bending alone", False, False, True),
    "all": ModeSet("all", "all three springs together", True, True, True),
}


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
    #: Ratio against reduction, one curve per mode set. Keys are the members of
    #: :data:`MODE_SETS`.
    ratios_by_mode: dict[str, np.ndarray]
    intact_ratio: float
    claimed_intact: float
    claimed_damaged: float
    claimed_signature: float
    joint_id: int
    brace_member: int
    springs: dict[str, float]
    #: Which joint flexibility the reductions are taken against.
    baseline_label: str = "as modelled"

    def change(self, mode: str) -> np.ndarray:
        """Signed fractional ratio change against reduction, for one mode set."""
        return (self.ratios_by_mode[mode] - self.intact_ratio) / self.intact_ratio

    @property
    def change_by_mode(self) -> dict[str, float]:
        """Signed ratio change at the paper's stated 10 percent, per mode set."""
        i = int(np.argmin(np.abs(self.reductions - 0.10)))
        return {m: float(self.change(m)[i]) for m in self.ratios_by_mode}

    @property
    def claimed_direction(self) -> float:
        """+1 if the claim says the ratio rises with damage, -1 if it falls.

        The claim is directional: 1.800 to 2.000 is an *increase*. A reduction
        that moves the ratio down by 11 percent has not reproduced it.
        """
        return 1.0 if self.claimed_damaged >= self.claimed_intact else -1.0

    @property
    def best_mode(self) -> str:
        """The mode set most favourable to the claim.

        The abstract says "10 percent stiffness reduction" without naming a mode,
        so the fair test is the reading that helps the claim most, not the one
        that happens to be easiest to compute. That turns out to matter: the
        axial spring alone moves the ratio the *wrong way*, while out-of-plane
        bending moves it the right way and about twelve times as far. Judging on
        the axial spring would have overstated the case against the paper.

        "Most favourable" means the largest movement *towards* the claimed value.
        If no reading moves the right way at all, the largest movement of any
        kind is reported instead - and that fact is itself in the verdict.
        """
        by = self.change_by_mode
        d = self.claimed_direction
        toward = {m: v for m, v in by.items() if v * d > 0}
        pool = toward or by
        return max(pool, key=lambda m: pool[m] * d if toward else abs(pool[m]))

    @property
    def any_mode_moves_the_claimed_way(self) -> bool:
        d = self.claimed_direction
        return any(v * d > 0 for v in self.change_by_mode.values())

    @property
    def ratios(self) -> np.ndarray:
        """The most claim-favourable curve."""
        return self.ratios_by_mode[self.best_mode]

    @property
    def at_claimed_reduction(self) -> float:
        """Ratio change at the paper's stated 10 percent, on its best mode."""
        return self.change_by_mode[self.best_mode]

    def _required(self, mode: str) -> float:
        """Reduction that first reaches the claimed change, for one mode set.

        The curves are not guaranteed monotonic - the axial and out-of-plane
        contributions oppose each other, so a combined sweep can turn over - and
        ``np.interp`` silently returns nonsense on non-monotonic input. This walks
        the curve and interpolates only across the bracketing interval.
        """
        # Signed, not absolute: a reduction that drives the ratio 11 percent the
        # *wrong* way has not reproduced a claim that it rises from 1.800 to 2.000.
        change = self.change(mode) * self.claimed_direction
        target = self.claimed_signature
        hits = np.nonzero(change >= target)[0]
        if hits.size == 0:
            return float("nan")
        i = int(hits[0])
        if i == 0:
            return float(self.reductions[0])
        c0, c1 = float(change[i - 1]), float(change[i])
        r0, r1 = float(self.reductions[i - 1]), float(self.reductions[i])
        if c1 == c0:
            return r1
        return r0 + (target - c0) * (r1 - r0) / (c1 - c0)

    @property
    def required_reduction_for_claim(self) -> float:
        """Smallest reduction, over every mode set, that reaches the claim.

        ``nan`` if no reduction of any mode on the swept range gets there - and
        the range runs to 99 percent, the practically complete loss of the joint.
        """
        vals = [self._required(m) for m in self.ratios_by_mode]
        finite = [v for v in vals if np.isfinite(v)]
        return min(finite) if finite else float("nan")

    @property
    def required_mode(self) -> str | None:
        """Which mode set reaches the claim first, if any does."""
        best, mode = float("inf"), None
        for m in self.ratios_by_mode:
            v = self._required(m)
            if np.isfinite(v) and v < best:
                best, mode = v, m
        return mode

    @property
    def verdict(self) -> str:
        got = self.at_claimed_reduction
        need = self.required_reduction_for_claim
        by = self.change_by_mode
        head = (
            f"A {self.claimed_signature * 100:.1f} percent ratio change is claimed for a 10 "
            f"percent joint stiffness reduction. The abstract does not say which stiffness, "
            f"so all four readings were swept and the one most favourable to the claim is "
            f"reported: {MODE_SETS[self.best_mode].label} gives {got * 100:+.3f} percent, a "
            f"factor of {abs(self.claimed_signature / got):.0f} short."
        )
        if not self.any_mode_moves_the_claimed_way:
            head += (
                " No reading moves the ratio in the claimed direction at all, so the figure "
                "above is the largest movement of any kind, not a movement towards 2.000."
            )
        elif min(by.values()) < 0 < max(by.values()):
            head += (
                " The modes work against each other - reducing the axial spring moves the "
                "ratio down while out-of-plane bending moves it up - so no combined "
                "reduction does better than the best single one."
            )
        if not np.isfinite(need):
            return head + (
                " Removing the joint spring entirely, in any mode, does not reach the "
                "claimed change, so no stiffness reduction of any size produces it. The "
                "claimed sensitivity does not follow from the structural mechanics, "
                "independently of how the crack itself is modelled."
            )
        m = self.required_mode
        return head + (
            f" The claimed change is reachable, but only at a {need * 100:.0f} percent "
            f"reduction in {MODE_SETS[m].label} - the practically complete loss of that "
            "spring, not the 10 percent the paper attributes to a 20 percent through-wall "
            "crack."
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
    baseline_springs: LJFStiffness | None = None,
    baseline_label: str = "as modelled",
) -> StiffnessReductionResult:
    """Sweep the joint stiffness reduction and record the strain ratio.

    ``reductions`` are fractions of the intact joint stiffness removed, 0 to 1.
    A value of 1 removes the joint spring completely, which bounds what any crack
    could possibly do.

    ``baseline_springs`` answers the obvious objection to this whole test - that
    it was run on the wrong joint. The abstract's K-joint (762 mm chord, 25 mm
    wall, 45 degree brace) is *softer* than any joint on the OC4 frame, by four
    to eight times in out-of-plane bending, and a softer joint carries more of
    the load path locally. Passing the paper's own springs here pre-softens the
    instrumented joint to them and takes the reductions relative to that, so the
    claim can be tested at the flexibility the paper itself specifies.

    Only softening is possible: added compliance in series cannot stiffen a
    spring. A target stiffer than the modelled joint raises ``ValueError`` rather
    than silently doing nothing.
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

    modelled = shell_ljf(_joint_geometry(tables, brace_member, joint_id))
    k = baseline_springs if baseline_springs is not None else modelled

    # Compliance needed to bring the modelled joint down to the target baseline.
    # Zero when no baseline is given, so the ordinary path is unaffected.
    _MODE_ATTRS = (("axial", "k_axial"), ("ipb", "k_ipb"), ("opb", "k_opb"))
    offset = {}
    for name, attr in _MODE_ATTRS:
        k_model, k_target = getattr(modelled, attr), getattr(k, attr)
        if k_target > k_model * (1.0 + 1e-9):
            raise ValueError(
                f"cannot raise the {name} joint stiffness from {k_model:.4g} to "
                f"{k_target:.4g}: a compliance added in series can only soften a spring. "
                "Pass a baseline no stiffer than the modelled joint."
            )
        offset[name] = max(1.0 / k_target - 1.0 / k_model, 0.0)

    def ratio_for(reduction: float, ms: ModeSet) -> float:
        # Compliance in series: removing a fraction f of the stiffness means the
        # remaining stiffness is (1-f)k, i.e. an added compliance of f/((1-f)k).
        # The reduction is taken against the baseline k, on top of any offset
        # that moved the joint to that baseline in the first place.
        f = min(reduction, 0.999) if reduction > 0.0 else 0.0
        soften = lambda kk: f / ((1.0 - f) * kk) if f else 0.0  # noqa: E731
        parts = (
            offset["axial"] + (soften(k.k_axial) if ms.axial else 0.0),
            offset["ipb"] + (soften(k.k_ipb) if ms.ipb else 0.0),
            offset["opb"] + (soften(k.k_opb) if ms.opb else 0.0),
        )
        extra = {brace_member: parts} if any(p > 0.0 for p in parts) else None
        b = build_jacket(ljf_model=ljf_model, crack_compliance=extra, tables=tables)
        s = build_response_surface(b, pair, cfg, n_theta=n_theta,
                                   eta_levels=np.linspace(-2, 2, 3))
        eu, el = strain_series(s, t, constituents)
        return ratio_from_series(t, eu, el)

    # The intact frame is the same for every mode set; solve it once.
    intact = ratio_for(0.0, MODE_SETS["all"])
    ratios_by_mode = {
        key: np.array([intact if r <= 0.0 else ratio_for(float(r), ms)
                       for r in reductions])
        for key, ms in MODE_SETS.items()
    }
    return StiffnessReductionResult(
        reductions=reductions,
        ratios_by_mode=ratios_by_mode,
        intact_ratio=float(intact),
        claimed_intact=claimed_intact,
        claimed_damaged=claimed_damaged,
        claimed_signature=abs(claimed_damaged - claimed_intact) / claimed_intact,
        joint_id=joint_id,
        brace_member=int(brace_member),
        springs={"axial": float(k.k_axial), "ipb": float(k.k_ipb), "opb": float(k.k_opb)},
        baseline_label=baseline_label,
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
