"""How far do the claims move when the modelling choices move?

Two things in this application are choices rather than measurements, and both
change the answers:

* which **local joint flexibility** formulation is in force - rigid frame
  idealisation, the first-principles shell model, or a published regression;
* the shell model's **load-spreading length**, the one free parameter in its
  derivation, which cannot be derived and has to be chosen.

``fe/ljf.py`` has always said the app "exposes the choice and reports the
spread". It exposed the choice and never reported the spread. This module is
that spread, computed rather than asserted.

The point is not to find the right value. It is to establish whether the
headline finding survives every value a reasonable person might pick. A result
that holds across the whole plausible range of an unknowable parameter is a much
stronger result than one computed at its midpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .claims.tests.c1_ratio import intact_ratio
from .fe.ljf import LJFModel
from .geometry.oc4 import build_jacket, load_tables, sensor_pair
from .loads.morison import HydroConfig
from .response import build_response_surface

__all__ = ["SensitivityRow", "ljf_sensitivity", "joint_sensitivity"]


#: Strain resolution of a mid-range commercial FBG interrogator, microstrain.
#: A signal below this cannot be read at all, whatever is done downstream.
FBG_RESOLUTION_USTRAIN = 1.0


@dataclass(frozen=True)
class SensitivityRow:
    """One configuration, the intact ratio it produces, and whether it is usable."""

    label: str
    detail: str
    ratio: float
    m2_amplitude_ustrain: float

    @property
    def finite(self) -> bool:
        return bool(np.isfinite(self.ratio))

    @property
    def measurable(self) -> bool:
        """Is the underlying strain large enough for a real sensor to see?"""
        return bool(self.m2_amplitude_ustrain >= FBG_RESOLUTION_USTRAIN)

    @property
    def well_conditioned(self) -> bool:
        """A ratio is only meaningful if both gauges read something.

        Where the upper gauge sits above water or in a near-null of the tidal
        response, the denominator approaches zero and the ratio runs to hundreds
        or millions - not a large signal, an undefined one.
        """
        return bool(self.finite and self.measurable and abs(self.ratio) < 20.0)


def _ratio_for(
    tables, joint_id, offset_m, theta, cfg, constituents, ljf_model,
    spread_factor, n_theta, record_days,
) -> tuple[float, float]:
    pair = sensor_pair(tables, joint_id, offset_m, theta)
    build = build_jacket(
        ljf_model=ljf_model,
        marine_growth_mm=cfg.marine_growth_mm,
        tables=tables,
        ljf_spread_factor=spread_factor,
    )
    surf = build_response_surface(
        build, pair, cfg, n_theta=n_theta, eta_levels=np.linspace(-2.0, 2.0, 3)
    )
    r = intact_ratio(surf, constituents, record_days=record_days, sample_interval_s=1800.0)
    return r.ratio, r.amplitude_lower * 1e6


def ljf_sensitivity(
    constituents,
    cfg: HydroConfig,
    joint_id: int = 5,
    offset_m: float = 1.5,
    theta: float = 0.0,
    spread_factors: tuple[float, ...] = (0.5, 1.0, 2.0),
    n_theta: int = 12,
    record_days: float = 14.0,
) -> list[SensitivityRow]:
    """Intact strain ratio across LJF formulations and spreading lengths.

    Rigid joints are included as the bounding case: they are what a conventional
    frame analysis would do, and the gap between rigid and shell is the whole
    contribution of local joint flexibility to the answer.
    """
    tables = load_tables()
    rows: list[SensitivityRow] = []

    ratio, amp = _ratio_for(
        tables, joint_id, offset_m, theta, cfg, constituents,
        LJFModel.RIGID, 1.0, n_theta, record_days,
    )
    rows.append(SensitivityRow(
        "RIGID", "ISO 19902 frame idealisation, no joint flexibility", ratio, amp))

    for sf in spread_factors:
        ratio, amp = _ratio_for(
            tables, joint_id, offset_m, theta, cfg, constituents,
            LJFModel.SHELL, sf, n_theta, record_days,
        )
        note = {0.5: "half", 1.0: "nominal", 2.0: "double"}.get(sf, f"{sf:g}x")
        rows.append(SensitivityRow(
            f"SHELL, spread x{sf:g}", f"shell BOEF, {note} load-spreading length",
            ratio, amp))
    return rows


def joint_sensitivity(
    constituents,
    cfg: HydroConfig,
    joint_ids: tuple[int, ...],
    offset_m: float = 1.2,
    theta: float = 0.0,
    ljf_model: LJFModel = LJFModel.SHELL,
    n_theta: int = 12,
    record_days: float = 14.0,
) -> list[SensitivityRow]:
    """Intact strain ratio at every braced leg joint.

    The headline claims were computed at one joint. If the ratio and its
    behaviour vary wildly between joints then the choice of joint is itself a
    hidden assumption, and that needs saying.
    """
    tables = load_tables()
    rows: list[SensitivityRow] = []
    for j in joint_ids:
        try:
            ratio, amp = _ratio_for(
                tables, j, offset_m, theta, cfg, constituents,
                ljf_model, 1.0, n_theta, record_days,
            )
            z = float(tables.joints.loc[j, "z_m"])
            rows.append(SensitivityRow(
                f"J{j}", f"z = {z:+.2f} m", ratio, amp))
        except Exception as exc:  # noqa: BLE001 - a joint that cannot be gauged is a result
            rows.append(SensitivityRow(f"J{j}", f"not evaluable: {exc}", float("nan"), 0.0))
    return rows


def spread_summary(rows: list[SensitivityRow]) -> str:
    """How much the configuration choice matters, over the usable configurations.

    Quoting a min-to-max range over all rows would be meaningless here: the
    ill-conditioned entries produce ratios in the millions, and reporting a
    "spread of a billion percent" says nothing except that some denominators
    approach zero. The spread is therefore reported over the well-conditioned
    rows, with the excluded ones counted rather than hidden.
    """
    usable = [r for r in rows if r.well_conditioned]
    unusable = [r for r in rows if not r.well_conditioned]
    if len(usable) < 2:
        return (
            f"Only {len(usable)} of {len(rows)} configurations are usable at all, so there "
            "is no meaningful spread to report. That is itself the finding."
        )
    vals = np.array([r.ratio for r in usable])
    lo, hi, mid = float(vals.min()), float(vals.max()), float(np.median(vals))
    rel = (hi - lo) / abs(mid) if mid else float("inf")
    out = (
        f"Across the {len(usable)} usable configurations the intact strain ratio runs from "
        f"{lo:.4f} to {hi:.4f}, a spread of {rel * 100:.0f} percent about the median "
        f"{mid:.4f}. Any value quoted to better than that is quoting the configuration, "
        "not the structure."
    )
    if unusable:
        weak = sum(1 for r in unusable if not r.measurable)
        out += (
            f" {len(unusable)} of {len(rows)} are excluded as unusable"
            + (f", {weak} because the M2 strain is below the {FBG_RESOLUTION_USTRAIN:.0f} "
               "microstrain a real interrogator can resolve" if weak else "")
            + "."
        )
    return out


def usability_summary(rows: list[SensitivityRow]) -> str:
    """Where on the structure the method could work at all."""
    n = len(rows)
    meas = [r for r in rows if r.measurable]
    cond = [r for r in rows if r.well_conditioned]
    if not n:
        return ""
    return (
        f"{len(meas)} of {n} braced joints carry an M2 strain above the "
        f"{FBG_RESOLUTION_USTRAIN:.0f} microstrain an interrogator can resolve, and "
        f"{len(cond)} of {n} also give a well-conditioned ratio. The method has a narrow "
        "window on this structure: the signal is strongest at the deep joints and falls "
        "away to nothing near and above the waterline, where the upper gauge reads almost "
        "zero and the ratio stops being defined."
    )
