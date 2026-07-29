"""Is the strain ratio simply the wrong observable for damage?

C3's lesson: the physics was fine, the estimator was wrong. Ask the same of C2.

A single-pair strain ratio is dominated by AXIAL force in the leg - measured at
J5, 0.15 to 0.32 microstrain axial against 0.02 bending. Axial force in a braced
frame is a GLOBAL equilibrium quantity: it is set by the overall load path, and
one joint going soft barely redistributes it. That is a complete explanation of
why the paper's own 10 percent joint stiffness step moves the ratio 0.37 percent.

But local joint flexibility is a ROTATIONAL spring. Softening it changes how
moment is carried across the joint, and moment shows up in BENDING strain, not
axial. The rosette already separates the two. So the question is not whether the
structure responds to the crack - it is whether we have been reading the channel
that carries the response.

Sweep the joint stiffness and watch all three observables.
"""
import numpy as np
from tidetwin.abstract import PAPER
from tidetwin.analysis import AnalysisConfig
from tidetwin.claims.tests.c2_damage import _joint_geometry
from tidetwin.fe.ljf import LJFModel, shell_ljf
from tidetwin.geometry.oc4 import brace_chord_joints, build_jacket, load_tables, sensor_pair
from tidetwin.nuisance import ratio_from_series
from tidetwin.response import build_response_surface, strain_series
from tidetwin.rosette import ROSETTE_ANGLES_DEG, decompose

cfg = AnalysisConfig()
tables = load_tables()
con = cfg.constituents()
hydro = cfg.hydro()
joint = cfg.joint_id
brace = brace_chord_joints(tables)[joint][0]
k = shell_ljf(_joint_geometry(tables, brace, joint))
t = np.arange(0.0, 14 * 86400.0, 1800.0)

PAIRS = [sensor_pair(tables, joint, cfg.sensor_offset_m, np.radians(a))
         for a in ROSETTE_ANGLES_DEG]
REDUCTIONS = np.array([0.0, 0.05, 0.10, 0.20, 0.50])
CLAIM = PAPER.damage_signature


def observables(reduction, modes="o"):
    """All three ratios at one joint-stiffness reduction."""
    if reduction <= 0:
        extra = None
    else:
        f = min(reduction, 0.999)
        c = lambda kk: f / ((1.0 - f) * kk)  # noqa: E731
        extra = {brace: (c(k.k_axial) if "a" in modes else 0.0,
                         c(k.k_ipb) if "i" in modes else 0.0,
                         c(k.k_opb) if "o" in modes else 0.0)}
    b = build_jacket(ljf_model=LJFModel.SHELL, crack_compliance=extra, tables=tables)
    U, L = [], []
    for p in PAIRS:
        s = build_response_surface(b, p, hydro, n_theta=12,
                                   eta_levels=np.linspace(-2, 2, 3))
        eu, el = strain_series(s, t, con)
        U.append(eu)
        L.append(el)
    du, dl = decompose(t, U), decompose(t, L)
    return {
        "single": ratio_from_series(t, U[0], L[0]),
        "axial": dl.axial / du.axial if du.axial > 0 else np.nan,
        "bending": dl.bending / du.bending if du.bending > 0 else np.nan,
        "bend_amp_u": du.bending,
        "bend_amp_l": dl.bending,
    }


for modes, name in (("o", "out-of-plane bending spring"), ("aio", "all three springs")):
    print(f"\n=== reducing the {name} ===")
    base = observables(0.0)
    print(f"{'reduction':>10}{'single pair':>15}{'axial rosette':>16}{'BENDING rosette':>18}")
    for r in REDUCTIONS:
        o = observables(float(r), modes) if r > 0 else base
        row = "".join(
            f"{(o[q] - base[q]) / base[q] * 100:+15.3f}%" if np.isfinite(o[q]) else f"{'n/a':>16}"
            for q in ("single", "axial", "bending")
        )
        print(f"{r * 100:9.0f}%{row}")
    print(f"           claimed at 10 %: {CLAIM * 100:+.1f}%")

b0 = observables(0.0)
print(f"\nbending M2 amplitude   upper {b0['bend_amp_u'] * 1e6:.4f} ustrain   "
      f"lower {b0['bend_amp_l'] * 1e6:.4f} ustrain")
print(f"FBG resolution {PAPER.fbg_resolution_ustrain} ustrain, "
      f"noise {PAPER.fbg_drift_ustrain_per_year} ustrain")
