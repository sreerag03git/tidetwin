"""Is 0.021 ustrain actually unreadable, or does harmonic fitting beat it down?

"Below the 0.1 microstrain resolution" is a per-sample statement. A 14-day record
at half-hourly sampling is 672 points and about 27 M2 cycles, and least squares
on a known frequency drives the amplitude standard error far below the
per-sample noise. The honest test is the fitted amplitude against its own
standard error, not against the resolution of one reading.
"""
import numpy as np
from tidetwin.abstract import PAPER
from tidetwin.analysis import AnalysisConfig
from tidetwin.claims.tests.c2_damage import _joint_geometry
from tidetwin.fe.ljf import LJFModel, shell_ljf
from tidetwin.geometry.oc4 import brace_chord_joints, build_jacket, load_tables, sensor_pair
from tidetwin.response import build_response_surface, strain_series
from tidetwin.rosette import ROSETTE_ANGLES_DEG, decompose

cfg = AnalysisConfig(); tables = load_tables()
con = cfg.constituents(); hydro = cfg.hydro()
joint = cfg.joint_id; brace = brace_chord_joints(tables)[joint][0]
k = shell_ljf(_joint_geometry(tables, brace, joint))
t = np.arange(0.0, 14 * 86400.0, 1800.0)
NOISE = PAPER.fbg_drift_ustrain_per_year * 1e-6   # 0.05 ustrain rms
RED = PAPER.stiffness_reduction

def build_for(r):
    if r <= 0:
        return build_jacket(ljf_model=LJFModel.SHELL, tables=tables)
    f = min(r, 0.999)
    return build_jacket(ljf_model=LJFModel.SHELL, tables=tables,
                        crack_compliance={brace: (0.0, 0.0, f / ((1 - f) * k.k_opb))})

def clean(b):
    U, L = [], []
    for a in ROSETTE_ANGLES_DEG:
        p = sensor_pair(tables, joint, cfg.sensor_offset_m, np.radians(a))
        s = build_response_surface(b, p, hydro, n_theta=12, eta_levels=np.linspace(-2, 2, 3))
        eu, el = strain_series(s, t, con)
        U.append(eu); L.append(el)
    return np.array(U), np.array(L)

U0, L0 = clean(build_for(0.0))
U1, L1 = clean(build_for(RED))
print(f"record {len(t)} samples over 14 days; FBG noise {NOISE*1e6:.2f} ustrain rms\n")

def spread(U, L, trials=120):
    rg = np.random.default_rng(7)
    out = []
    for _ in range(trials):
        Un = U + rg.normal(0, NOISE, U.shape)
        Ln = L + rg.normal(0, NOISE, L.shape)
        du, dl = decompose(t, Un), decompose(t, Ln)
        out.append((dl.bending / du.bending, du.bending, dl.bending))
    a = np.array(out)
    return a[:, 0], a[:, 1].mean(), a[:, 2].mean()

r0, bu, bl = spread(U0, L0)
r1, _, _ = spread(U1, L1)
sd0 = r0.std(ddof=1)
print(f"bending amplitude, upper {bu*1e6:.4f} ustrain  ({bu*1e6/PAPER.fbg_resolution_ustrain:.2f}x "
      f"the per-sample resolution)")
print(f"bending RATIO with noise:  mean {r0.mean():.4f}   sd {sd0:.4f}  "
      f"({sd0/r0.mean()*100:.2f} % of the ratio)")
shift = (r1.mean() - r0.mean()) / r0.mean()
print(f"\nshift from the paper's {RED*100:.0f} % stiffness step: {shift*100:+.3f} %")
print(f"measurement noise on that shift:                {sd0/r0.mean()*100:.3f} %")
print(f"-> signal-to-noise on the damage step: {abs(shift)/(sd0/r0.mean()):.2f}")
print(f"   claimed step {PAPER.damage_signature*100:+.1f} % would give "
      f"{PAPER.damage_signature/(sd0/r0.mean()):.1f}")
