"""C7 - modal insensitivity to local joint damage.

Eigen-solves the intact and cracked frames and reports the natural frequency
shift. The claim is that the shift is below 0.5 percent, i.e. that modal methods
cannot see the damage the strain method is supposed to see.

The comparison is only fair if the modal side is given its best shot, so this
module also reports the shift a dense, long-record modal array could actually
resolve. Operational modal analysis on offshore structures typically achieves
frequency estimates to around 0.1 to 0.5 percent depending on record length and
excitation; quoting the claim against a threshold of zero would be rhetoric
rather than analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...damage.crack_ljf import CrackGeometry, crack_compliance
from ...fe.ljf import LJFModel
from ...fe.modal import eigenmodes, frequency_shift
from ...geometry.oc4 import build_jacket, load_tables

__all__ = ["ModalResult", "modal_insensitivity"]


@dataclass
class ModalResult:
    frequencies_intact: np.ndarray
    frequencies_damaged: np.ndarray
    shift_fraction: np.ndarray
    max_abs_shift: float
    resolvable_threshold: float
    crack: CrackGeometry
    n_modes: int

    @property
    def detectable_by_modal(self) -> bool:
        return bool(self.max_abs_shift > self.resolvable_threshold)


def modal_insensitivity(
    joint_id: int,
    brace_member: int,
    a_over_T: float = 0.5,
    surface_length_m: float = 0.10,
    n_modes: int = 6,
    ljf_model: LJFModel = LJFModel.SHELL,
    resolvable_threshold: float = 0.002,
    include_added_mass: bool = True,
) -> ModalResult:
    """Frequency shift between intact and cracked frames.

    ``resolvable_threshold`` defaults to 0.2 percent, a realistic figure for a
    well-instrumented operational modal analysis with a long record - deliberately
    optimistic for the competing method, so that a finding of modal insensitivity
    is not an artefact of a pessimistic assumption.
    """
    tables = load_tables()
    leg_ps = [
        int(m.prop_set)
        for _mid, m in tables.members.iterrows()
        if int(m.prop_set) in (2, 3, 4) and joint_id in (int(m.joint_i), int(m.joint_j))
    ]
    chord_T = float(tables.sections.loc[leg_ps[0], "wall_thickness_m"])
    brace_d = float(
        tables.sections.loc[int(tables.members.loc[brace_member, "prop_set"]), "outer_diameter_m"]
    )
    crack = CrackGeometry(a=a_over_T * chord_T, c=0.5 * surface_length_m, T=chord_T)
    dC, _q = crack_compliance(crack, load_width=brace_d)

    intact = build_jacket(ljf_model=ljf_model, include_added_mass=include_added_mass, tables=tables)
    damaged = build_jacket(
        ljf_model=ljf_model,
        include_added_mass=include_added_mass,
        crack_compliance={brace_member: (dC, 0.0, 0.0)},
        tables=tables,
    )
    m0 = eigenmodes(*intact.model.assemble(), intact.model.free_dof(), n_modes=n_modes)
    m1 = eigenmodes(*damaged.model.assemble(), damaged.model.free_dof(), n_modes=n_modes)
    shift = frequency_shift(m0, m1, n=n_modes)
    return ModalResult(
        frequencies_intact=m0.frequencies_hz[:n_modes],
        frequencies_damaged=m1.frequencies_hz[:n_modes],
        shift_fraction=shift,
        max_abs_shift=float(np.nanmax(np.abs(shift))),
        resolvable_threshold=float(resolvable_threshold),
        crack=crack,
        n_modes=n_modes,
    )
