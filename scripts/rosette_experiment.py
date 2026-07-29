"""Does a direction-invariant gauge rosette fix C3?

Single pair: one gauge above and one below the joint at ONE circumferential
angle. The axial surface strain there depends on which way the current pushes,
so a rotary tide moves the ratio with the structure unchanged. That is the
dominant C3 channel.

Rosette: four angles per section (0/90/180/270). Diametrically opposed gauges
separate bending from axial exactly:

    bx = (eps(0) - eps(180)) / 2      by = (eps(90) - eps(270)) / 2
    axial = mean of all four
    bending magnitude = hypot(bx, by)     <-- invariant to load DIRECTION

The ratio of bending magnitudes between the two sections should therefore not
care where the current points, only how hard it pushes.

Paired comparison: identical tide, identical draws, identical everything. Only
the estimator differs.
"""
import numpy as np
from tidetwin.analysis import AnalysisConfig
from tidetwin.geometry.oc4 import build_jacket, load_tables, sensor_pair
from tidetwin.nuisance import (
    CHANNELS,
    NuisanceRanges,
    _correlated_storm,
    ratio_from_series,
    robust_scale,
)
from tidetwin.response import build_response_surface
from tidetwin.loads.tides import TidalConstituents
from tidetwin.signal.harmonic import fit_harmonics
from tidetwin.loads.tides import constituent_frequency

ANGLES = np.array([0.0, 90.0, 180.0, 270.0])
N = 400
SIG = 0.111

cfg = AnalysisConfig()
tables = load_tables()
con0 = cfg.constituents()
hydro = cfg.hydro()
t = np.arange(0.0, 14 * 86400.0, 1800.0)

print("building four response surfaces (one per gauge angle)...")
build = build_jacket(ljf_model=cfg.ljf_model, tables=tables)
surfaces = []
for a in ANGLES:
    p = sensor_pair(tables, cfg.joint_id, cfg.sensor_offset_m, np.radians(a))
    surfaces.append(build_response_surface(build, p, hydro, n_theta=16,
                                           eta_levels=np.linspace(-2, 2, 3)))
print(f"  done, {sum(s.n_solves for s in surfaces)} solves\n")


def series_all(d, rng):
    """Strain at every angle, upper and lower, under one nuisance draw."""
    con = con0
    if d["ellipse_ratio"] or d["direction_bias"]:
        con = TidalConstituents(
            names=con0.names, omega=con0.omega, elev_amp=con0.elev_amp,
            elev_phase=con0.elev_phase, semi_major=con0.semi_major,
            semi_minor=con0.semi_minor * (1.0 + d["ellipse_ratio"]),
            inclination=con0.inclination + d["direction_bias"],
            current_phase=con0.current_phase, provenance=con0.provenance,
            citation=con0.citation, latitude=con0.latitude, longitude=con0.longitude,
            source_note=con0.source_note)
    uv = con.depth_averaged_current(t) * d["current_scale"]
    uv = uv + np.array([d["wind_u"] + d["wave_u"], d["wind_v"] + d["wave_v"]])[None, :]
    eta = con.elevation(t) + d["water_level"]
    speed = np.hypot(uv[:, 0], uv[:, 1])
    direction = np.arctan2(uv[:, 1], uv[:, 0])

    U, L = [], []
    for s in surfaces:
        eu, el = s.evaluate(speed, direction, eta)
        if d["drift"]:
            eu = eu + d["drift"] * (t - t[0]) / max(t[-1] - t[0], 1.0)
        if d["noise"]:
            eu = eu + rng.normal(0.0, d["noise"], size=eu.shape)
            el = el + rng.normal(0.0, d["noise"], size=el.shape)
        U.append(eu)
        L.append(el)
    return np.array(U), np.array(L), eta


OM = np.array([float(constituent_frequency("M2").value)])


def m2_phasor(y):
    """Complex M2 coefficient of one gauge series."""
    f = fit_harmonics(t, y, ("M2",), OM)
    return f.amplitude_of("M2") * np.exp(1j * float(f.phase[f.index("M2")]))


def rosette_m2(e):
    """Direction-invariant M2 bending magnitude from four opposed gauges.

    Combining in the HARMONIC domain, not the time domain. hypot() on the raw
    series rectifies it - the bending components swing through zero twice a
    cycle, so |.| doubles the frequency and destroys the M2 line the ratio is
    fitted on. Combining the complex M2 coefficients keeps the constituent
    intact and is still invariant to the direction of loading.
    """
    A = [m2_phasor(x) for x in e]
    Bx = 0.5 * (A[0] - A[2])
    By = 0.5 * (A[1] - A[3])
    return float(np.hypot(abs(Bx), abs(By)))


def rosette_ratio(U, L):
    up = rosette_m2(U)
    return float(rosette_m2(L) / up) if up > 0 else float("nan")


def axial_m2(e):
    """Direction-invariant AXIAL M2 amplitude: the mean over the circumference.

    Averaging opposed gauges cancels the bending contribution exactly, leaving
    the axial (frame-action) component - which at a braced jacket leg is where
    essentially all the tidal strain is. Also averages four gauges, so the
    sensor noise falls by two.
    """
    A = [m2_phasor(x) for x in e]
    return float(abs(sum(A) / 4.0))


def axial_ratio(U, L):
    up = axial_m2(U)
    return float(axial_m2(L) / up) if up > 0 else float("nan")




# ---------------------------------------------------------------- quadrature
# Drag is in quadrature with elevation; buoyancy is in phase with it. A common
# scaling of the current scales every drag strain by the same factor, so a
# DRAG-ONLY ratio should be invariant to spring/neap range. The buoyancy part is
# what breaks that invariance, so removing it should kill the channel.
# Elevation comes from a tide gauge, which is a real measurement, so using it as
# a phase reference is not smuggling in knowledge nobody would have.

def quad_parts(y, eta_series):
    """In-phase (buoyancy-like) and quadrature (drag-like) M2 parts of a series."""
    A = m2_phasor(y)
    E = m2_phasor(eta_series)
    if abs(E) == 0:
        return float("nan"), float("nan")
    rel = A / (E / abs(E))           # rotate so elevation sits at zero phase
    return abs(rel.real), abs(rel.imag)


def axial_series(e):
    """The direction-invariant axial combination, as a time series."""
    return (e[0] + e[1] + e[2] + e[3]) / 4.0


def axial_quad_ratio(U, L, eta):
    """Axial rosette, drag component only. Invariant to direction AND to amplitude."""
    _, qu = quad_parts(axial_series(U), eta)
    _, ql = quad_parts(axial_series(L), eta)
    return float(ql / qu) if qu and np.isfinite(qu) and qu > 0 else float("nan")


def single_quad_ratio(U, L, eta):
    """Single pair, drag component only - isolates what conditioning alone buys."""
    _, qu = quad_parts(U[0], eta)
    _, ql = quad_parts(L[0], eta)
    return float(ql / qu) if qu and np.isfinite(qu) and qu > 0 else float("nan")
def draw_one(rg, rn, active):
    d = dict.fromkeys(
        ["direction_bias", "ellipse_ratio", "current_scale", "wind_u", "wind_v",
         "water_level", "growth", "wave_u", "wave_v", "scour", "drift", "noise"], 0.0)
    d["current_scale"] = 1.0
    d["scour"] = 1.0
    z = _correlated_storm(rg, rn.storm_correlation)
    if "current_direction" in active:
        d["direction_bias"] = np.radians(rg.normal(0.0, rn.direction_bias_sd_deg))
        d["ellipse_ratio"] = rg.normal(0.0, rn.ellipse_ratio_sd)
    if "spring_neap" in active:
        d["current_scale"] = max(0.05, 1.0 + rg.normal(0.0, rn.spring_neap_sd))
    if "wind_current" in active:
        d["wind_u"] = z[0] * rn.wind_current_sd_ms
        d["wind_v"] = rg.normal(0.0, rn.wind_current_sd_ms)
    if "water_level" in active:
        d["water_level"] = z[1] * rn.water_level_sd_m
    if "wave_offset" in active:
        d["wave_u"] = z[2] * rn.wave_offset_sd_ms
        d["wave_v"] = rg.normal(0.0, rn.wave_offset_sd_ms)
    if "fbg_drift" in active:
        d["drift"] = rg.normal(0.0, rn.fbg_drift_sd_ustrain) * 1e-6
    d["noise"] = rn.fbg_noise_ustrain * 1e-6
    return d



rn = NuisanceRanges()
ACTIVE = [c for c in CHANNELS if c not in ("marine_growth", "scour")]

zero = draw_one(np.random.default_rng(0), rn, set())
U0, L0, E0 = series_all(zero, np.random.default_rng(0))
b_single = ratio_from_series(t, U0[0], L0[0])
b_ax     = axial_ratio(U0, L0)
b_axq    = axial_quad_ratio(U0, L0, E0)
b_sq     = single_quad_ratio(U0, L0, E0)
print(f"baselines  single {b_single:.4f}   axial {b_ax:.4f}   "
      f"single+quad {b_sq:.4f}   axial+quad {b_axq:.4f}")

EST = [("single", b_single), ("axial", b_ax), ("sing+quad", b_sq), ("AXIAL+quad", b_axq)]


def run(active_set, label):
    rg = np.random.default_rng(4242)
    V = np.empty((4, N))
    for i in range(N):
        d = draw_one(rg, rn, active_set)
        U, L, E = series_all(d, rg)
        V[0, i] = ratio_from_series(t, U[0], L[0])
        V[1, i] = axial_ratio(U, L)
        V[2, i] = single_quad_ratio(U, L, E)
        V[3, i] = axial_quad_ratio(U, L, E)
    cells = []
    for j, (_nm, b) in enumerate(EST):
        cells.append(f"{float(np.nanstd(V[j], ddof=1)) / abs(b) * 100:11.2f}%")
    print(f"{label:<22}" + "".join(cells))
    return V


print("nuisance dispersion, as % of the intact ratio")
print(f"{'channel':<22}" + "".join(f"{nm:>12}" for nm, _ in EST))
print("-" * 72)
for ch in ACTIVE:
    run({ch}, f"  {ch}")
print("-" * 72)
V = run(set(ACTIVE), "JOINT (all)")

print()
for j, (nm, b) in enumerate(EST):
    v = V[j]
    sd = float(np.nanstd(v, ddof=1)) / abs(b)
    rb = robust_scale(v) / abs(b)
    fa = np.mean(np.abs(v - b) / abs(b) >= SIG) * 100
    verdict = "PASS" if sd / SIG <= 1/3 else ("PASS(robust)" if rb / SIG <= 1/3 else "FAIL")
    print(f"{nm:<12} sigma/signal {sd/SIG:5.2f}x   robust/signal {rb/SIG:5.2f}x   "
          f"false alarms {fa:5.1f}%   -> {verdict}")
