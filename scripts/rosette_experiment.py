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
    return np.array(U), np.array(L)


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
# growth and scour need structural rebuilds; excluded from BOTH estimators so
# the comparison stays paired. Direction is the dominant channel either way.
ACTIVE = [c for c in CHANNELS if c not in ("marine_growth", "scour")]

zero = draw_one(np.random.default_rng(0), rn, set())
U0, L0 = series_all(zero, np.random.default_rng(0))
base_single = ratio_from_series(t, U0[0], L0[0])
base_ros = rosette_ratio(U0, L0)
base_ax = axial_ratio(U0, L0)
print(f"baseline ratio   single pair {base_single:.4f}   rosette {base_ros:.4f}\n")


def run(active_set, label):
    rg = np.random.default_rng(4242)
    s_vals, r_vals, a_vals = np.empty(N), np.empty(N), np.empty(N)
    for i in range(N):
        d = draw_one(rg, rn, active_set)
        U, L = series_all(d, rg)
        s_vals[i] = ratio_from_series(t, U[0], L[0])
        r_vals[i] = rosette_ratio(U, L)
        a_vals[i] = axial_ratio(U, L)
    def cv(v, b):
        sd = float(np.nanstd(v, ddof=1))
        rb = robust_scale(v)
        return sd / abs(b) * 100, rb / abs(b) * 100
    ss, _ = cv(s_vals, base_single)
    rs, _ = cv(r_vals, base_ros)
    as_, ar_ = cv(a_vals, base_ax)
    print(f"{label:<24}{ss:9.2f}%{rs:12.2f}%{as_:12.2f}%  (robust {ar_:5.2f}%)")
    return s_vals, r_vals, a_vals


print("nuisance dispersion, as % of the intact ratio")
print(f"{'channel':<24}{'single':>10}{'bend-rose':>13}{'AXIAL-rose':>13}")
print("-" * 76)
for ch in ACTIVE:
    run({ch}, f"  {ch}")
print("-" * 76)
sv, rv, av = run(set(ACTIVE), "JOINT (all channels)")

for label, v, b in (("single pair    ", sv, base_single),
                    ("bending rosette", rv, base_ros),
                    ("AXIAL rosette  ", av, base_ax)):
    rel = np.abs(v - b) / abs(b)
    sd = float(np.nanstd(v, ddof=1)) / abs(b)
    print(f"\n{label}:  sigma/signal = {sd / SIG:.2f}x (limit 0.33)   "
          f"false alarms = {np.mean(rel >= SIG) * 100:.1f}%   "
          f"-> {'PASS' if sd / SIG <= 1 / 3 else 'FAIL'}")
