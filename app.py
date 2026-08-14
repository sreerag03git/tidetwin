"""TideTwin - adversarial test bench for the tidal-calibration fatigue twin claims.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make `import tidetwin` work without PYTHONPATH being set. Streamlit Cloud runs
# the entrypoint with the repository root as the working directory, which is not
# necessarily this file's directory, so neither `python -m` nor an editable
# install can be assumed.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from tidetwin.analysis import AnalysisConfig, normalise, run_full, run_quick
from tidetwin.diagram import method_chain_svg
from tidetwin.robustness import (
    FBG_RESOLUTION_USTRAIN,
    joint_sensitivity,
    ljf_sensitivity,
    spread_summary,
    usability_summary,
)
from tidetwin.claims.ledger import build_stamp, ledger_frame, to_csv, to_latex
from tidetwin.claims.registry import CLAIMS, Artifacts, Status, evaluate_all
from tidetwin.abstract import LOWER_ZAKUM_JOINT, PAPER
from tidetwin.claims.tests.c2_damage import MODE_SETS
from tidetwin.economics.npv import EconomicInputs, tornado
from tidetwin.fe.ljf import JointGeometry, LJFModel, ljf_quantity, shell_ljf
from tidetwin.geometry.oc4 import (
    OC4_CITATION,
    WATER_DEPTH,
    brace_chord_joints,
    load_tables,
)
from tidetwin.loads.era5 import credentials_status
from tidetwin.loads.morison import API_RP2A, drag_inertia_coefficients
from tidetwin.loads.noaa import available_cached
from tidetwin.loads.tides import PLACEHOLDER_CONSTITUENTS, tide_model_status
from tidetwin.nuisance import CHANNEL_LABELS, NuisanceRanges
from tidetwin.provenance import assumed, derived, measured, published
from tidetwin.report import ReportInputs, to_html, to_markdown, to_text
from tidetwin.unlock import gate_status
from tidetwin.ui import (
    cover,
    running_reporter,
    svg_figure,
    dataframe,
    figure_block,
    claims_strip,
    inject_theme,
    panel,
    loading_screen,
    masthead,
    provenance_legend,
    quantity,
    section,
    status_chip,
    unavailable_panel,
)

st.set_page_config(
    layout="wide", page_title="TideTwin", initial_sidebar_state="expanded", page_icon=None
)
inject_theme()

TABLES = load_tables()
K_JOINTS = brace_chord_joints(TABLES)


# --------------------------------------------------------------------- sidebar


TIDE_COLUMNS = [
    "elev_amp",
    "elev_phase_deg",
    "semi_major",
    "semi_minor",
    "inclination_deg",
    "current_phase_deg",
]


def tide_source_input(s) -> str | None:
    """Choose the tidal forcing: a real station, or a hand-entered table.

    Only the NOAA route yields MEASURED provenance. Anything typed into the
    sidebar is ASSUMED, however plausible it looks.
    """
    cached = available_cached()
    labels = {sp.current_id: sp.label for sp in cached}
    options: list[str | None] = [sp.current_id for sp in cached] + [None]
    if not cached:
        s.warning("No cached tidal constants. Run `python scripts/fetch_tides.py`.")
        return None
    choice = s.selectbox(
        "Tidal forcing",
        options,
        index=0,
        format_func=lambda k: (labels[k] + "  (MEASURED)") if k else "Placeholder / custom (ASSUMED)",
        help="NOAA CO-OPS published harmonic constants. These are real stations, but they "
        "are not the Arabian Gulf platform site: for that a TPXO or FES extraction is "
        "needed. C3's verdict is the same at every station tested.",
    )
    if choice:
        s.caption(f"MEASURED — {labels[choice]}")
    return choice


def tide_inputs(s) -> tuple[dict | None, str]:
    """Let the user supply harmonic constants for any site, or upload them.

    Whatever is entered is ASSUMED: this widget cannot make a number measured.
    Supplying a real TPXO/FES extraction is what changes that, and the
    Provenance tab says so.
    """
    with s.expander("Custom harmonic constants (ASSUMED)"):
        st.caption(
            "Edit for your site, or upload a CSV with columns: constituent, "
            + ", ".join(TIDE_COLUMNS)
            + ". Rows may be added or removed; any IHO constituent name is accepted."
        )
        up = st.file_uploader("Upload constituents CSV", type=["csv"], key="tide_csv")
        base = pd.DataFrame(PLACEHOLDER_CONSTITUENTS).T.reset_index(names="constituent")
        source = "labelled placeholder set (not a real tide)"
        if up is not None:
            try:
                base = pd.read_csv(up)
                missing = [c for c in ["constituent", *TIDE_COLUMNS] if c not in base.columns]
                if missing:
                    st.error(f"CSV is missing column(s): {', '.join(missing)}. Using defaults.")
                    base = pd.DataFrame(PLACEHOLDER_CONSTITUENTS).T.reset_index(names="constituent")
                else:
                    source = f"uploaded CSV '{up.name}'"
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read that CSV ({type(exc).__name__}: {exc}). Using defaults.")
        edited = st.data_editor(
            base, num_rows="dynamic", width="stretch", key="tide_editor"
        )
        try:
            table = {
                str(row["constituent"]).strip().upper(): {
                    c: float(row.get(c, 0.0) or 0.0) for c in TIDE_COLUMNS
                }
                for _i, row in edited.iterrows()
                if str(row.get("constituent", "")).strip()
            }
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not parse the table ({exc}); using defaults.")
            return None, source
        if not table:
            return None, source
        if source.startswith("labelled") and not edited.equals(base):
            source = "user-edited harmonic constants"
        return table, source


def run_control(container, key, *, prominent: bool, overlay=None) -> None:
    """The Run button, and an honest statement of whether it is needed.

    Moving a slider deliberately does not recompute - a Monte Carlo over nine
    claims takes about a minute, and firing it on every drag would make the app
    unusable. But that decision only works if the user can see it: without a
    visible "your inputs are ahead of these figures" signal and a button right
    next to it, changing an input looks exactly like a control that does
    nothing. This renders both, at the top of the sidebar where the inputs are,
    as well as beside the headline numbers.
    """
    ran = st.session_state.get("full") is not None
    stale = ran and st.session_state.get("full_key") != key
    if not ran:
        label, kind = "Run the analysis", "primary"
    elif stale:
        label, kind = "Re-run with the new inputs", "primary"
        container.warning(
            "**Inputs changed.** The figures on the page are still from the previous "
            "run. Press below to recompute them.",
            icon=":material/sync_problem:",
        )
    else:
        label, kind = "Re-run analysis", "secondary"
        if prominent:
            container.success("Figures are up to date with these inputs.",
                              icon=":material/check_circle:")
    if container.button(label, type=kind, width="stretch",
                        key=f"run_{'side' if prominent else 'top'}"):
        # Render the live pipeline in the main-area overlay when we have one, so
        # pressing Run shows the machine working across the top of the page rather
        # than a thin grey bar tucked in the sidebar. Fall back to a plain bar if
        # no overlay was passed.
        report = running_reporter(overlay) if overlay is not None else None
        bar = None if overlay is not None else container.progress(0.0, "starting")

        def _progress(f: float, m: str) -> None:
            if report is not None:
                report(f, m)
            else:
                bar.progress(min(f, 1.0), m)

        try:
            _progress(0.0, "assembling the frame")
            _cfg = _cfg_from_key(key)
            st.session_state.full = run_full(_cfg, _progress)
            st.session_state.full_key = key
            st.session_state.full_from_bundle = False
            # Kept so the stamp and captions can describe the run being shown
            # rather than whatever the sidebar says later.
            st.session_state.full_cfg = _cfg
        finally:
            if overlay is not None:
                overlay.empty()
            else:
                bar.empty()
        st.rerun()
    if prominent:
        container.caption(
            "Takes about a minute: nine claims, a Monte Carlo over eight nuisance "
            "channels, and a few hundred frame solves. Nothing recomputes while you "
            "are still adjusting inputs."
        )


def sidebar() -> AnalysisConfig:
    s = st.sidebar
    s.markdown("### TideTwin")
    s.caption("Every red chip is an assumption. Results inherit them.")
    #: Filled in after the config is built, since the staleness check needs its key.
    st.session_state["_run_slot"] = s.container()
    s.divider()

    s.markdown("**Platform**")
    lat = s.number_input("Latitude, deg N", -90.0, 90.0, 24.9, 0.1, format="%.4f")
    lon = s.number_input("Longitude, deg E", -180.0, 180.0, 53.2, 0.1, format="%.4f")

    tide_station = tide_source_input(s)
    tide_table, tide_source = tide_inputs(s)

    s.markdown("**Joint and sensors**")
    joint_options = sorted(K_JOINTS)
    joint = s.selectbox(
        "Target K-joint",
        joint_options,
        index=joint_options.index(5) if 5 in joint_options else 0,
        format_func=lambda j: f"J{j}  (z = {TABLES.joints.loc[j, 'z_m']:+.2f} m, "
        f"{len(K_JOINTS[j])} braces)",
    )
    offset = s.slider("Gauge offset from joint, m", 0.5, 4.0, 1.5, 0.1)
    theta = s.slider("Gauge circumferential angle, deg", 0, 350, 0, 10)

    s.markdown("**Structural model**")
    ljf = s.selectbox(
        "Local joint flexibility",
        [LJFModel.SHELL, LJFModel.RIGID, LJFModel.TABULATED],
        format_func=lambda m: m.value,
        help="RIGID is the ISO 19902 frame idealisation and makes C2 and C7 vacuous. "
        "TABULATED needs published coefficients that are not shipped.",
    )
    roughness = s.slider("Member roughness k, mm", 0.0, 100.0, 50.0, 5.0) * 1e-3
    growth = s.slider("Marine growth, mm", 0.0, 150.0, 0.0, 5.0)

    _mode_label = {
        "single": "Single gauge pair (as the paper specifies)",
        "rosette": "Four-gauge rosette (proposed fix)",
    }
    measurement_mode = s.selectbox(
        "Sensor layout for C3",
        ["single", "rosette"],
        format_func=lambda m: _mode_label[m],
        help="The deciding test C3 depends on the sensor layout. The single pair is what "
        "the paper specifies and what the headline verdict is computed on. The four-gauge "
        "rosette is this app's proposed fix: a direction-and-amplitude-invariant estimator "
        "that takes the nuisance dispersion from ~11 % of the ratio to ~1 %, enough to pass. "
        "It costs about four times the run time because it solves four gauge positions.",
    )
    if measurement_mode == "rosette":
        s.caption(
            ":material/science: Proposed instrumentation, not the paper's. A C3 pass here "
            "is what the improved layout achieves, and is labelled as such throughout."
        )

    s.markdown("**Crack geometry**")
    s.caption("a/T and 2c are independent. There is no single percent-through-wall.")
    a_over_T = s.slider("a / T  (depth over wall thickness)", 0.05, 0.90, 0.50, 0.05)
    two_c = s.slider("2c  surface length, mm", 20.0, 400.0, 100.0, 10.0) * 1e-3

    s.markdown("**Record and Monte Carlo**")
    days = s.slider("Record length, days", 7.0, 90.0, 30.0, 1.0)
    n_mc = s.select_slider("MC samples per channel", [50, 100, 200, 400, 800], value=100)
    s.caption(
        "100 keeps the app responsive and the C3 verdict is robust well below it; the "
        "convergence trace flags honestly if a run has not settled. The exported ledger "
        "uses 900 for publication precision. Raise this for a tighter interactive estimate."
    )
    n_theta = s.select_slider("Response surface directions", [12, 24, 36, 72], value=24)
    seed = s.number_input("Random seed", 0, 2**31 - 1, 20260728, 1)
    s.caption(f"Seed {seed} is exported with every figure and table.")

    with s.expander("Nuisance ranges (all ASSUMED)"):
        ranges = NuisanceRanges(
            direction_bias_sd_deg=st.slider("Current direction bias, deg 1sd", 0.0, 45.0, 15.0),
            ellipse_ratio_sd=st.slider("Ellipse shape 1sd", 0.0, 0.5, 0.15),
            spring_neap_sd=st.slider("Spring/neap current scale 1sd", 0.0, 0.6, 0.25),
            wind_current_sd_ms=st.slider("Wind-driven current, m/s 1sd", 0.0, 0.3, 0.08),
            water_level_sd_m=st.slider("Residual water level, m 1sd", 0.0, 1.0, 0.20),
            marine_growth_mm=(0.0, st.slider("Marine growth after 20 y, mm", 0.0, 150.0, 100.0)),
            wave_offset_sd_ms=st.slider("Wave velocity offset, m/s 1sd", 0.0, 0.5, 0.10),
            scour_factor_range=(st.slider("Foundation stiffness retained", 0.01, 1.0, 0.15), 1.0),
            fbg_drift_sd_ustrain=st.slider("Differential FBG drift, ustrain 1sd", 0.0, 5.0, 0.05),
            fbg_noise_ustrain=st.slider("FBG noise, ustrain rms", 0.0, 2.0, 0.05),
        )

    with s.expander("Economic inputs (all ASSUMED)"):
        econ = EconomicInputs(
            sensor_capex=st.number_input("Sensor capex, USD", 0.0, 5e6, 450_000.0, 10_000.0),
            rov_spread_day_rate=st.number_input("ROV spread, USD/day", 0.0, 5e5, 85_000.0, 5_000.0),
            install_days=st.number_input("Install duration, days", 0.0, 60.0, 6.0, 0.5),
            interrogator_opex_per_year=st.number_input(
                "Interrogator + telemetry opex, USD/yr", 0.0, 1e6, 60_000.0, 5_000.0
            ),
            sensor_failure_rate_per_year=st.number_input(
                "Sensor failure rate, 1/yr", 0.0, 1.0, 0.08, 0.01
            ),
            replacement_cost=st.number_input("Replacement cost, USD", 0.0, 2e6, 120_000.0, 10_000.0),
            false_positive_inspection_cost=st.number_input(
                "False-positive inspection, USD", 0.0, 5e6, 500_000.0, 50_000.0
            ),
            false_positives_per_year=st.number_input("False positives, 1/yr", 0.0, 5.0, 0.3, 0.1),
            avoided_campaigns_per_year=st.number_input(
                "Avoided campaigns, 1/yr", 0.0, 5.0, 0.25, 0.05
            ),
            avoided_campaign_cost=st.number_input(
                "Avoided campaign cost, USD", 0.0, 2e7, 2_000_000.0, 100_000.0
            ),
            discount_rate=st.number_input("Discount rate", 0.0, 0.4, 0.09, 0.01),
            horizon_years=int(st.number_input("Horizon, years", 1, 50, 20, 1)),
        )

    return AnalysisConfig(
        latitude=lat,
        longitude=lon,
        joint_id=int(joint),
        sensor_offset_m=offset,
        sensor_theta_deg=float(theta),
        ljf_model=ljf,
        measurement_mode=measurement_mode,
        roughness_m=roughness,
        marine_growth_mm=growth,
        record_days=days,
        crack_a_over_T=a_over_T,
        crack_2c_m=two_c,
        n_mc_samples=int(n_mc),
        n_theta=int(n_theta),
        seed=int(seed),
        ranges=ranges,
        economics=econ,
        # A selected NOAA station takes precedence over any typed table in
        # constituents(), so the table is only part of the configuration when no
        # station is chosen. Carrying the ignored placeholder table otherwise put
        # a phantom difference into the cache key and stopped the precomputed
        # default from matching a fresh visitor's sidebar.
        tide_table=(tide_table if tide_station is None else None),
        tide_source=tide_source,
        tide_station=tide_station,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _quick(key: tuple) -> Artifacts:
    return run_quick(_cfg_from_key(key))


@st.cache_resource(show_spinner=False)
def _bundle():
    """The committed precomputed default result, or None if absent/stale.

    Loaded once per process. This is what lets a fresh visitor see the fully
    populated default view without the container computing anything - the reason
    the deployed app is no longer pinned on the CPU throttle. Any problem loading
    it returns None and the app computes on demand instead, so it can only ever
    make things faster, never wrong.
    """
    from tidetwin.precompute import load_bundle

    return load_bundle()


def _bundle_matches(key: tuple) -> bool:
    """Does the current config match the one the bundle was built for?

    Compared on everything but the trailing code-fingerprint element, which
    differs between the machine that built the bundle and this one but says
    nothing about the configuration.
    """
    b = _bundle()
    return b is not None and key[:-1] == _cfg_key(b["cfg"])[:-1]


@st.cache_data(show_spinner=False, max_entries=4)
def _full(key: tuple) -> Artifacts:
    """The full claims analysis, cached on the inputs so it runs once."""
    return run_full(_cfg_from_key(key))


@st.cache_data(show_spinner=False, max_entries=2)
def _sensitivity(key: tuple):
    """LJF and joint sensitivity sweeps. Returns ([], []) rather than failing.

    The 24-joint sweep is the single most expensive thing on the Structure tab,
    so for the default configuration it is served from the precomputed bundle
    rather than recomputed on load.
    """
    if _bundle_matches(key):
        return _bundle()["sensitivity"]
    cfg = _cfg_from_key(key)
    try:
        cfg, _notes = normalise(cfg)
        con = cfg.constituents()
        joints = tuple(sorted(brace_chord_joints(load_tables())))
        return (
            ljf_sensitivity(con, cfg.hydro(), joint_id=cfg.joint_id,
                            offset_m=cfg.sensor_offset_m, n_theta=12),
            joint_sensitivity(con, cfg.hydro(), joints, offset_m=0.4, n_theta=12),
        )
    except Exception:  # noqa: BLE001 - a sweep is a nicety, not a dependency
        return [], []


@st.cache_data(show_spinner=False, max_entries=4)
def _cycle(key: tuple):
    """One tidal cycle of real frame solves, for the animation."""
    if _bundle_matches(key):
        return _bundle()["cycle"]
    cfg = _cfg_from_key(key)
    try:
        from tidetwin.geometry.oc4 import build_jacket, sensor_pair
        from tidetwin.simulate import simulate_cycle

        cfg, _notes = normalise(cfg)
        tables = load_tables()
        pair = sensor_pair(tables, cfg.joint_id, cfg.sensor_offset_m,
                           np.radians(cfg.sensor_theta_deg))
        build = build_jacket(ljf_model=cfg.ljf_model,
                             marine_growth_mm=cfg.marine_growth_mm, tables=tables)
        return simulate_cycle(build, pair, cfg.constituents(), cfg.hydro(), n_frames=36)
    except Exception:  # noqa: BLE001 - the tab must still render without it
        return None


_CFG_CACHE: dict[tuple, AnalysisConfig] = {}


def _code_fingerprint() -> str:
    """A digest of the solver source, mixed into every cache key.

    Deliberately not cached. Caching the fingerprint would freeze it across the
    very hot reload it exists to notice, which is the whole point of it. It runs
    once per rerun and stats about forty files, which is far below the cost of
    anything else on the page.

    Streamlit hot-reloads the script on edit but keeps ``st.cache_data`` entries
    alive in the same process. Results computed before a change therefore
    survive it, and a cached object built by the old code can reach new code
    that expects a field it does not carry - which showed up as three tabs
    reporting ``AttributeError`` on a property that does exist.

    Keying on the source means editing a solver invalidates exactly the results
    that solver produced. Costs one directory walk per session and nothing after
    that. Missing files are ignored: a fingerprint that cannot be computed should
    degrade to a stale cache, never to a crash on startup.
    """
    import hashlib
    from pathlib import Path

    h = hashlib.sha256()
    root = Path(__file__).parent / "tidetwin"
    try:
        for p in sorted(root.rglob("*.py")):
            h.update(p.name.encode())
            h.update(str(p.stat().st_mtime_ns).encode())
    except OSError:
        return "unavailable"
    return h.hexdigest()[:16]


@st.cache_data(show_spinner=False, max_entries=1)
def _gauge_robustness_sweep() -> dict:
    """The rosette fix's strain-ratio dispersion against gauge tolerance.

    Independent of the sidebar configuration - it is a property of the four-gauge
    layout, not of the site - so it is computed once and cached. Cheap (a few
    thousand phasor evaluations), so it does not reopen the CPU-throttle question.
    """
    from tidetwin.rosette import gauge_robustness

    angles = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
    gains = angles * 0.004  # 0 to 2.8 % gain, paired with 0 to 7 deg placement
    disp = np.array([
        gauge_robustness(sigma_angle_deg=float(a), sigma_gain=float(g), n=4000).ratio_dispersion
        for a, g in zip(angles, gains)
    ])
    typ = gauge_robustness(sigma_angle_deg=3.0, sigma_gain=0.01, n=6000)
    placement_only = gauge_robustness(sigma_angle_deg=3.0, sigma_gain=0.0, n=6000)
    gain_only = gauge_robustness(sigma_angle_deg=0.0, sigma_gain=0.01, n=6000)
    return {
        "angles": angles, "gains": gains, "disp": disp, "typical": typ,
        "placement_only": placement_only.ratio_dispersion,
        "gain_only": gain_only.ratio_dispersion,
    }


@st.cache_data(show_spinner=False, max_entries=8)
def jacket_3d(joint_id: int, title: str = "OC4 jacket, target joint marked") -> go.Figure:
    """The 3D jacket blueprint, braces highlighted and the target joint marked.

    Geometry only - no frame solve - so it is cheap and cached on the joint id.
    Used on both the Overview and Structure tabs, so a reader sees the structure
    under test immediately rather than only after scrolling to Structure.
    """
    j = TABLES.joints
    fig = go.Figure()
    for _mid, m in TABLES.members.iterrows():
        a, b = j.loc[int(m.joint_i)], j.loc[int(m.joint_j)]
        ps = int(m.prop_set)
        braced = ps in (2, 3, 4)
        fig.add_trace(
            go.Scatter3d(
                x=[a.x_m, b.x_m], y=[a.y_m, b.y_m], z=[a.z_m, b.z_m], mode="lines",
                line=dict(color="#35b6c4" if braced else "#8a929b", width=4 if braced else 2),
                showlegend=False, hoverinfo="skip",
            )
        )
    sel = j.loc[joint_id]
    fig.add_trace(
        go.Scatter3d(
            x=[sel.x_m], y=[sel.y_m], z=[sel.z_m], mode="markers",
            marker=dict(size=7, color="#d1495b"), name=f"J{joint_id}",
        )
    )
    fig.update_layout(
        scene=dict(xaxis_title="x, m", yaxis_title="y, m", zaxis_title="z, m (SWL = 0)",
                   aspectmode="data"),
        title=title,
    )
    return fig


def _cfg_key(cfg: AnalysisConfig) -> tuple:
    tide = (
        tuple(sorted((k, tuple(sorted(v.items()))) for k, v in cfg.tide_table.items()))
        if cfg.tide_table
        else None
    )
    k = (
        cfg.latitude, cfg.longitude, cfg.joint_id, cfg.sensor_offset_m, cfg.sensor_theta_deg,
        cfg.ljf_model.value, cfg.roughness_m, cfg.marine_growth_mm, cfg.record_days,
        cfg.crack_a_over_T, cfg.crack_2c_m, cfg.n_mc_samples, cfg.n_theta, cfg.seed, tide,
        cfg.tide_station, cfg.measurement_mode, _code_fingerprint(),
    )
    _CFG_CACHE[k] = cfg
    return k


def _cfg_from_key(key: tuple) -> AnalysisConfig:
    return _CFG_CACHE[key]


cfg = sidebar()
key = _cfg_key(cfg)

# First load assembles the OC4 frame, factorises it and builds the response
# surface: a few seconds of real work. Show what is happening instead of a blank
# page, and clear it the moment the result is in.
_boot = st.empty()
# A main-area slot the Run button draws its live pipeline into, so a run shows
# across the top of the page rather than as a thin bar in the sidebar.
_run_overlay = st.empty()
_first = "booted" not in st.session_state
_splash_start = None
if _first:
    _boot.markdown(loading_screen(), unsafe_allow_html=True)
    _splash_start = time.perf_counter()
st.session_state.booted = True

if "full" not in st.session_state:
    st.session_state.full = None
    st.session_state.full_key = None
    st.session_state.full_cfg = None
    st.session_state.full_from_bundle = False

# On first load, show the precomputed default result instantly - do NOT compute.
# Auto-running the analysis on every cold load is what pinned the deployed app on
# the CPU throttle: a minute of solver work per visitor, per rerun, forever. The
# committed bundle carries that result for the default configuration, so a fresh
# visitor sees the fully populated app with the container doing no solving at all.
# If the bundle is missing or stale the app simply opens on its "run the analysis"
# landing; it never silently burns CPU on load again.
if _first and st.session_state.full is None:
    b = _bundle()
    if b is not None and _bundle_matches(key):
        st.session_state.full = b["full"]
        st.session_state.full_cfg = b["cfg"]
        st.session_state.full_key = _cfg_key(b["cfg"])
        # The bundle also carries the joint-sensitivity sweep and the tidal-cycle
        # simulation, so the Structure tab shows them without solving. This flag
        # lets that tab reach for them instead of blanking to "awaiting a run".
        st.session_state.full_from_bundle = True
# The precomputed result loads in a blink, so hold the branded splash on screen
# long enough to be seen and for its animation to play - a deliberate ~1.7 s
# intro on the first visit only, not a recomputed wait. It is a rendered page
# with a CSS animation, so it costs nothing on the server.
if _splash_start is not None:
    _held = time.perf_counter() - _splash_start
    if _held < 1.7:
        time.sleep(1.7 - _held)
_boot.empty()

# A code reload or a redeploy re-defines the dataclasses, which leaves anything
# held in session state an instance of the *old* class. Reading a field the new
# code expects then raises AttributeError and takes the whole page down with it.
# Discard a stale artifact instead of half-trusting it: recomputing is cheap,
# and a crash on a tab the user did not ask about is not.
if st.session_state.full is not None and not isinstance(st.session_state.full, Artifacts):
    st.session_state.full = None
    st.session_state.full_key = None
    st.session_state.full_cfg = None
    st.session_state.full_from_bundle = False
    st.info("The analysis code changed since the last run. Press Run full analysis again.")

# Which run the page is showing. Changing an input used to drop straight back to
# the quick analysis, so every Monte Carlo figure - Detection, Assimilation,
# Economics - vanished and was replaced by "Not computed" until the user found
# the Run button. Moving one slider emptied most of the app.
#
# The last full run is kept on screen instead. It is real, it is reproducible,
# and it is better than a blank page; what it is not is a result for the inputs
# now in the sidebar, so everything that describes it - the banner, the tab
# captions and the reproducibility stamp - describes the run that produced it,
# not the sidebar.
_have_full = st.session_state.full is not None
_stale = _have_full and st.session_state.full_key != key
# The quick fallback is only needed when nothing full is on screen. Computing it
# eagerly on every load cost a second or two of solver work even when the bundle
# already had the full result to show - wasted, and on a throttled container not
# free. So it is computed lazily, here, only if it is actually about to be shown.
art: Artifacts = st.session_state.full if _have_full else _quick(key)
#: The configuration the displayed figures were actually computed from.
shown_cfg: AnalysisConfig = st.session_state.get("full_cfg") if _have_full else cfg
if shown_cfg is None:
    shown_cfg = cfg
results = evaluate_all(art)
by_id = {r.claim_id: r for r in results}

if _stale:
    st.warning(
        "**These figures are from the previous run.** They are real results, but for the "
        "inputs used when the analysis was last run, not the ones now in the sidebar. "
        "Press **Re-run with the new inputs** to update them. The reproducibility stamp "
        "on the Ledger tab describes the run shown here, not the sidebar.",
        icon=":material/history:",
    )

STAMP = build_stamp(
    seed=shown_cfg.seed,
    geometry_digest=TABLES.digest,
    geometry_retrieved=str(OC4_CITATION.retrieved),
    ljf_model=shown_cfg.ljf_model.value,
    shell_fe_digest="not shipped" if not art.shell_fe_available else "shipped",
    tide_source=f"{art.tide_provenance} placeholder"
    if art.tide_provenance != "MEASURED"
    else "TPXO/FES extraction",
)

# ------------------------------------------------------------- global verdict

masthead(
    "Does the tidal-calibration fatigue method actually work? This app tries to prove it "
    "does not."
)

c3 = by_id["C3"]
CLAIMED_SIGNATURE = 11.1

# The C3 verdict used to occupy a full-width banner here, above everything, on
# every tab. It is still the load-bearing result and it is still stated plainly -
# in the scoreboard below, first in the claims strip, and in full on Detection -
# but it no longer crowds out the rest of the app before a reader has seen any of
# it. That placement was the author's call and this is the author's revision.
top = st.columns([1, 1, 1, 2])
top[0].metric("Claims supported", sum(r.status is Status.PASS for r in results))
top[1].metric("Claims refuted", sum(r.status is Status.FAIL for r in results))
top[2].metric(
    "Cannot be settled",
    sum(r.status in (Status.UNTESTABLE_DATA, Status.UNTESTABLE_PHYSICAL) for r in results),
    help="Not a failure, and not the same as 'not run'. The data or the physical test "
    "needed to settle these does not exist here; the Provenance tab lists exactly what "
    "would unlock each one.",
)
_pending = sum(r.status is Status.NOT_RUN for r in results)
if _pending:
    top[2].caption(f"{_pending} more awaiting analysis")
with top[3]:
    run_control(st.container(), key, prominent=False, overlay=_run_overlay)

# The same control at the top of the sidebar, next to the inputs it applies to.
# The slot was reserved before the widgets were drawn, so it appears above them
# rather than below sixty of them where nobody would find it.
run_control(st.session_state["_run_slot"], key, prominent=True, overlay=_run_overlay)

if art.c3 is None:
    st.info(
        "The Monte Carlo claims have not been computed for these inputs yet. Press "
        "**Re-run with current inputs** above. Structure, Environment and Sensing are "
        "available regardless."
    )

TABS = st.tabs(
    ["Overview", "Structure", "Environment", "Sensing", "Detection",
     "Assimilation", "Economics", "Ledger", "Provenance"]
)


# ----------------------------------------------------------------- Overview
with TABS[0]:
    with panel("Overview tab"):
        if art.input_notes:
            with st.expander(f"Input adjustments ({len(art.input_notes)}) — nothing corrected silently",
                             expanded=True):
                for n in art.input_notes:
                    st.markdown(f"- {n}")
        if art.errors:
            with st.expander(f"Computations that did not complete ({len(art.errors)})", expanded=True):
                for cid, why in sorted(art.errors.items()):
                    st.markdown(f"- **{cid}** — {why}")

        # What this is, what it stands on, and how it gets from a tide to a
        # verdict - before any number appears. The credentials row lists sources
        # that are actually in the code, so it is checkable on the Provenance
        # tab rather than being decoration.
        cover(
            tagline="Adversarial claims test bench · offshore jacket fatigue",
            lead=(
                "Nine claims from a paper on tidal-calibration fatigue monitoring, each "
                "recomputed here from first principles and then <b>attacked</b>. Every "
                "figure on every tab is solved at runtime and carries its provenance - "
                "measured, published, derived or assumed. Nothing is quoted from the "
                "paper except the claims themselves."
            ),
            credentials=[
                "OC4 reference jacket",
                "NOAA CO-OPS harmonics",
                "3D Timoshenko frame FE",
                "API RP 2A-WSD Morison",
                "Doodson constituents",
                "Newman-Raju SIF",
                "shell BOEF joint flexibility",
                "log-EnKF · SIR particle filter",
                f"Monte Carlo {cfg.n_mc_samples}x",
            ],
            chain=[
                ("Measured tide",
                 "Harmonic constants from a real NOAA current and elevation station pair."),
                ("Frame solve",
                 "Morison drag and buoyancy on a 3D beam model with flexible joints."),
                ("Strain ratio",
                 "M2 amplitudes at two gauges bracketing the joint; their ratio is the signal."),
                ("Nuisance budget",
                 "Eight environmental channels propagated through with no crack present."),
                ("Claims ledger",
                 "Nine verdicts: pass, marginal, fail, or honestly untestable."),
            ],
            note=(
                f"Reference structure and public tidal data. Verdicts below are for the "
                f"inputs in the sidebar, seed {shown_cfg.seed}."
            ),
        )

        section("Nine claims at a glance", "colour is the verdict; the figure is what we computed")
        claims_strip([
            (c.id, by_id[c.id].status.value.split(" - ")[0],
             by_id[c.id].status.colour, by_id[c.id].computed_text)
            for c in CLAIMS
        ])
        provenance_legend()

        # This page used to be nothing but text: a claims strip, a legend and a
        # stack of collapsed expanders. Every one of the twenty-six figures was
        # on another tab, so the first thing anyone saw was a wall of prose with
        # gaps in it, and the reasonable conclusion was that the app had no
        # graphs. The two results that carry the report are now drawn here, on
        # arrival, with the method diagram between them.
        section("How the method is supposed to work",
                "tide to strain to ratio, and where a crack would enter")
        with panel("Method diagram"):
            svg_figure(
                method_chain_svg(
                    nuisance_pct=(art.c3.effective_cv * 100) if art.c3 is not None else None,
                    signature_pct=CLAIMED_SIGNATURE,
                    verdict=c3.status.value if art.c3 is not None else "",
                ),
                height=300,
            )

        section("The structure under test", "OC4 reference jacket — drag to rotate")
        figure_block(
            jacket_3d(cfg.joint_id),
            "overview_jacket",
            f"Braces in accent, legs in grey, the target joint J{cfg.joint_id} marked. "
            "The full geometry, an animated tidal-cycle simulation and the modal shapes are "
            "on the Structure tab.",
            460,
        )

        if art.c3 is not None and art.c2_stiffness is not None:
            section("The two results the report rests on",
                    "both computed here; both shown in full on the Detection tab")
            gcol = st.columns(2)
            with gcol[0]:
                _n, _sig = art.c3, 0.111
                _b = _n.baseline_ratio
                fig = go.Figure()
                fig.add_histogram(x=_n.joint_samples, nbinsx=60, marker_color="#35b6c4")
                fig.add_vline(x=_b, line_color="#111418", line_width=2)
                for _s in (+1, -1):
                    fig.add_vline(x=_b * (1 + _s * _sig), line_color="#b3261e",
                                  line_dash="dash", line_width=2)
                fig.update_layout(
                    title="C3 · with no crack, where the sea alone puts the ratio",
                    xaxis_title="intact strain ratio (dashed = the claimed crack step)",
                    yaxis_title="draws", showlegend=False, margin=dict(t=54, b=40),
                )
                figure_block(fig, "overview_c3", height=330)
                st.caption(
                    f"**{_n.false_alarm_fraction(_sig) * 100:.0f} % of draws with no damage "
                    "at all** already move the ratio by as much as a crack is claimed to. "
                    "The dashed lines and the histogram overlap; that overlap is the finding."
                )
                if getattr(_n, "measurement_mode", "single") != "rosette":
                    st.caption(
                        ":material/lightbulb: This is the paper's single-pair layout, so C3 "
                        "fails as designed. Switch **Sensor layout for C3** in the sidebar to "
                        "the four-gauge rosette to see the proposed fix bring this to a pass."
                    )
            with gcol[1]:
                _sr = art.c2_stiffness
                fig = go.Figure()
                for _k, _arr in _sr.ratios_by_mode.items():
                    fig.add_scatter(x=_sr.reductions * 100, y=_sr.change(_k) * 100,
                                    name=MODE_SETS[_k].label, mode="lines",
                                    line=dict(width=3 if _k == _sr.best_mode else 1.5,
                                              dash=None if _k == _sr.best_mode else "dot"))
                fig.add_hline(y=_sr.claimed_signature * 100, line_color="#b3261e",
                              line_dash="dash")
                fig.add_vline(x=10.0, line_color="#b3261e", line_dash="dot")
                fig.update_layout(
                    title="C2 · what the paper's own 10 % stiffness step actually does",
                    xaxis_title="joint stiffness removed, %",
                    yaxis_title="change in the strain ratio, %",
                    legend=dict(orientation="h", y=-0.3), margin=dict(t=54, b=40),
                )
                figure_block(fig, "overview_c2", height=330)
                st.caption(
                    f"The paper's own intermediate step gives "
                    f"**{_sr.at_claimed_reduction * 100:+.3f} %** against a claimed "
                    f"**+{_sr.claimed_signature * 100:.1f} %** - short by a factor of "
                    f"{abs(_sr.claimed_signature / _sr.at_claimed_reduction):.0f}, with no "
                    "crack model involved."
                )
            st.caption(
                "Every other figure lives on the tab it belongs to: the jacket and the "
                "animated tidal cycle on **Structure**, the tides on **Environment**, the "
                "gauge signals on **Sensing**, the full C2 and C3 workings on **Detection**, "
                "the filters on **Assimilation**, and the cost model on **Economics**."
            )

        section("Claims ledger", "the full statement and verdict for each")
        rows = []
        for c in CLAIMS:
            r = by_id[c.id]
            rows.append(
                {
                    "Claim": c.id,
                    "Asserted": c.claimed_value,
                    "Computed": r.computed_text,
                    "Status": r.status.value,
                    "Statement": c.statement,
                }
            )
        dataframe(pd.DataFrame(rows))

        for c in CLAIMS:
            r = by_id[c.id]
            with st.expander(f"{c.id} — {c.statement[:80]}...", expanded=(c.id == "C3")):
                st.markdown(status_chip(r.status), unsafe_allow_html=True)
                st.markdown(f"**Asserted** {c.claimed_value} &nbsp;&nbsp; **Computed** {r.computed_text}")
                st.markdown(f"**Pass criterion** &mdash; {c.pass_criterion}")
                st.markdown(r.detail)
                if r.blocking_assumptions:
                    st.caption("Blocking assumptions: " + " ".join(r.blocking_assumptions))


# ---------------------------------------------------------------- Structure
with TABS[1]:
    with panel("Structure tab"):
        section("OC4 reference jacket", "published geometry, hash-verified on load")
        st.caption(str(OC4_CITATION))
        c = st.columns(4)
        with c[0]:
            quantity(published(len(TABLES.joints), "-", "joints", OC4_CITATION))
        with c[1]:
            quantity(published(len(TABLES.members), "-", "members", OC4_CITATION))
        with c[2]:
            quantity(published(WATER_DEPTH, "m", "water depth", OC4_CITATION))
        with c[3]:
            quantity(published(len(K_JOINTS), "-", "braced leg joints", OC4_CITATION))

        fig = jacket_3d(cfg.joint_id)
        figure_block(fig, "oc4_geometry", "Legs in accent, braces in grey. Mudline at z = -50.001 m.", 520)

        section("Tidal cycle simulation",
                "the frame solved at every phase of one M2 cycle - press play")
        # The simulation that goes with the result on screen. When that result is
        # the precomputed default, the bundle already carries it, so it shows with
        # no solving. When the user has run their own inputs, it is solved for
        # those. Only a changed-but-not-yet-run configuration has nothing to show,
        # and that must not silently trigger 36 frame solves on a throttled
        # container - so it, and only it, asks for a run.
        if st.session_state.get("full_from_bundle") and _bundle() is not None:
            cyc = _bundle()["cycle"]
        elif st.session_state.get("full_key") == key:
            with st.spinner("Solving the frame through a tidal cycle..."):
                cyc = _cycle(key)
        else:
            unavailable_panel("Awaiting a run",
                              "Press Re-run with the new inputs to solve the tidal cycle "
                              "for the current configuration.")
            cyc = None
        if cyc is None:
            pass
        else:
            j = TABLES.joints
            edges = [(int(m.joint_i), int(m.joint_j), int(m.prop_set))
                     for _mid, m in TABLES.members.iterrows()]
            idx = {jid: n for n, jid in enumerate(TABLES.joints.index)}

            def frame_traces(shape):
                xs, ys, zs = [], [], []
                for a, b, _ps in edges:
                    pa, pb = shape[idx[a]], shape[idx[b]]
                    xs += [pa[0], pb[0], None]
                    ys += [pa[1], pb[1], None]
                    zs += [pa[2], pb[2], None]
                return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                    line=dict(color="#1a5fb4", width=3),
                                    hoverinfo="skip", showlegend=False)

            base = []
            for a, b, _ps in edges:
                pa, pb = cyc.nodes_undeformed[idx[a]], cyc.nodes_undeformed[idx[b]]
                base.append((pa, pb))
            gx, gy, gz = [], [], []
            for pa, pb in base:
                gx += [pa[0], pb[0], None]; gy += [pa[1], pb[1], None]; gz += [pa[2], pb[2], None]

            fig = go.Figure(
                data=[
                    go.Scatter3d(x=gx, y=gy, z=gz, mode="lines",
                                 line=dict(color="#c9ced4", width=1),
                                 name="undeformed", hoverinfo="skip"),
                    frame_traces(cyc.displaced[0]),
                ],
                frames=[
                    go.Frame(data=[frame_traces(cyc.displaced[k])], traces=[1],
                             name=f"{cyc.hours[k]:.2f}")
                    for k in range(len(cyc.hours))
                ],
            )
            fig.update_layout(
                title=f"Jacket response through one M2 cycle "
                      f"(deflection exaggerated {cyc.exaggeration:,.0f}x)",
                scene=dict(xaxis_title="x, m", yaxis_title="y, m",
                           zaxis_title="z, m (SWL = 0)", aspectmode="data"),
                updatemenus=[dict(
                    type="buttons", showactive=False, x=0.02, y=0.06, xanchor="left",
                    buttons=[
                        dict(label="Play", method="animate",
                             args=[None, dict(frame=dict(duration=90, redraw=True),
                                              fromcurrent=True, mode="immediate")]),
                        dict(label="Pause", method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                mode="immediate")]),
                    ])],
                sliders=[dict(
                    active=0, x=0.12, len=0.84, y=0.02,
                    currentvalue=dict(prefix="hours into the cycle: "),
                    steps=[dict(method="animate", label=f"{h:.1f}",
                                args=[[f"{h:.2f}"], dict(mode="immediate",
                                      frame=dict(duration=0, redraw=True))])
                           for h in cyc.hours],
                )],
            )
            figure_block(
                fig, "tidal_cycle_simulation",
                f"{cyc.n_solves} full frame solves, one per phase. True peak deflection is "
                f"{cyc.max_true_deflection_mm:.2f} mm on a 70 m structure, so it is drawn "
                f"{cyc.exaggeration:,.0f}x larger to be visible. The lean rotates rather than "
                "reversing, because the tidal current traces an ellipse.", 560,
            )

            fig = go.Figure()
            fig.add_scatter(x=cyc.hours, y=cyc.strain_upper, name="upper gauge")
            fig.add_scatter(x=cyc.hours, y=cyc.strain_lower, name="lower gauge")
            fig.add_scatter(x=cyc.hours, y=np.hypot(cyc.current_u, cyc.current_v),
                            name="current speed, m/s", yaxis="y2",
                            line=dict(dash="dot", color="#5b646d"))
            fig.update_layout(
                title="Gauge strains through the same cycle",
                xaxis_title="hours into the M2 cycle",
                yaxis_title="axial surface strain, microstrain",
                yaxis2=dict(title="current speed, m/s", overlaying="y", side="right",
                            showgrid=False),
            )
            figure_block(
                fig, "tidal_cycle_strains",
                "Both gauges swing together — that common-mode motion is what dividing one "
                "by the other is meant to reject. Drag goes as speed squared, so the strain "
                "peaks twice per cycle while the current reverses once.", 360,
            )

        section("Local joint flexibility", "how much the chord wall gives where a brace lands")
        geom = JointGeometry(1.2, 0.035, 0.8, 0.02, np.radians(29.4))
        quantity(ljf_quantity(geom, cfg.ljf_model))
        warn = geom.validity()
        if warn:
            st.caption("Outside the usual parametric envelope: " + "; ".join(warn))
        if cfg.ljf_model is LJFModel.RIGID:
            unavailable_panel(
                "RIGID joints selected",
                "C2 and C7 are vacuous with rigid joints: there is no joint compliance for a "
                "crack to change. Select SHELL to make them meaningful.",
            )

        # The physical reason C2 fails lives here, so it belongs on the page as a
        # picture and not only as a sentence in the verdict.
        if art.c2_stiffness is not None:
            section("This joint against the abstract's own joint",
                    "the springs a 10 percent reduction is taken from")
            zak = shell_ljf(LOWER_ZAKUM_JOINT)
            here = art.c2_stiffness.springs
            modes = ["axial", "ipb", "opb"]
            labels = ["axial, N/m", "in-plane bending, N.m/rad", "out-of-plane bending, N.m/rad"]
            fig = go.Figure()
            fig.add_bar(name=f"J{cfg.joint_id}, as analysed", x=labels,
                        y=[here[m] for m in modes], marker_color="#35b6c4")
            fig.add_bar(name="the abstract's Lower Zakum K-joint", x=labels,
                        y=[zak.k_axial, zak.k_ipb, zak.k_opb], marker_color="#c8871b")
            fig.update_layout(
                title="Local joint stiffness, shell beam-on-elastic-foundation",
                yaxis_title="stiffness (log scale)", yaxis_type="log", barmode="group",
                legend=dict(orientation="h", y=-0.25),
            )
            figure_block(fig, "joint_stiffness_comparison", height=400)
            st.caption(
                f"The abstract's K-joint (762 mm chord, 25 mm wall, 45 degree brace, "
                f"gamma = {PAPER.gamma:.1f}) is **softer** than this frame's joint in every "
                "mode, most of all in out-of-plane bending. A softer joint carries more of "
                "the load path locally, so it is the case where a stiffness reduction should "
                "matter most - which is exactly why C2 is also evaluated there. It comes out "
                "slightly less sensitive, not more. Note the log scale: out-of-plane bending "
                "is two orders of magnitude softer than axial, and it is the mode that "
                "carries what little damage sensitivity the strain ratio has."
            )

        section("How much do the modelling choices matter?",
                "measured, not asserted - the spread is the honest precision")
        # The 24-joint sweep that goes with the displayed result: served from the
        # bundle for the precomputed default, computed for a user's own run, and
        # only asked-for when the configuration has changed but not yet been run -
        # so a slider move never triggers 24 sweeps on a throttled container.
        if st.session_state.get("full_from_bundle") and _bundle() is not None:
            ljf_rows, joint_rows = _bundle()["sensitivity"]
        elif st.session_state.get("full_key") == key:
            with st.spinner("Sweeping joints and joint-flexibility models..."):
                ljf_rows, joint_rows = _sensitivity(key)
        else:
            unavailable_panel("Awaiting a run",
                              "Press Re-run with the new inputs to sweep the joints and "
                              "joint-flexibility models for the current configuration.")
            ljf_rows, joint_rows = [], []

        if ljf_rows:
            cols = st.columns([3, 2])
            with cols[0]:
                dataframe(pd.DataFrame([
                    {"Configuration": r.label, "Strain ratio": round(r.ratio, 4),
                     "M2 strain, ustrain": round(r.m2_amplitude_ustrain, 3),
                     "Detail": r.detail}
                    for r in ljf_rows
                ]))
            with cols[1]:
                st.caption(spread_summary(ljf_rows))
            st.caption(
                "C1 turns out to be nearly insensitive to local joint flexibility, which "
                "makes it a more robust number than expected. C2 and C7 are the opposite: "
                "both are identically zero with rigid joints, because there is then no "
                "joint compliance for a crack to change."
            )

        if joint_rows:
            usable = [r for r in joint_rows if r.well_conditioned]
            fig = go.Figure()
            fig.add_scatter(
                x=[r.m2_amplitude_ustrain for r in joint_rows],
                y=[float(TABLES.joints.loc[int(r.label[1:]), "z_m"]) for r in joint_rows],
                mode="markers+text", text=[r.label for r in joint_rows],
                textposition="middle right", textfont=dict(size=9),
                marker=dict(size=11,
                            color=["#1a7f43" if r.well_conditioned else "#b3261e"
                                   for r in joint_rows]),
                name="braced joints",
            )
            fig.add_vline(x=FBG_RESOLUTION_USTRAIN, line_dash="dash", line_color="#b3261e",
                          annotation_text="FBG resolution")
            fig.add_hline(y=0.0, line_dash="dot", line_color="#5b646d",
                          annotation_text="still water level")
            fig.update_layout(
                title="Where on the jacket is there a signal to measure at all?",
                xaxis_title="M2 strain amplitude at the lower gauge, microstrain",
                yaxis_title="joint elevation, m (SWL = 0)", xaxis_type="log",
                hovermode="closest", showlegend=False,
            )
            figure_block(
                fig, "joint_usability",
                f"Green joints give a usable, well-conditioned ratio; red do not. "
                f"{len(usable)} of {len(joint_rows)} qualify. " + usability_summary(joint_rows),
                430,
            )

        if art.c7 is not None:
            section("C7 - modal shift", "can vibration monitoring see the same damage?")
            m = art.c7
            quantity(
                derived(m.max_abs_shift, "-", "max frequency shift", [], "intact vs cracked eigen-solve")
            )
            fig = go.Figure()
            fig.add_bar(x=[f"mode {i+1}" for i in range(m.n_modes)], y=m.shift_fraction * 100,
                        name="computed shift", marker_color="#35b6c4")
            fig.add_hline(y=-0.5, line_dash="dash", line_color="#d1495b",
                          annotation_text="claimed 0.5 % threshold")
            fig.add_hline(y=-m.resolvable_threshold * 100, line_dash="dot", line_color="#c8871b",
                          annotation_text="dense modal array resolution")
            fig.update_layout(title="Natural frequency shift from the modelled crack",
                              xaxis_title="mode", yaxis_title="frequency shift, %")
            figure_block(fig, "c7_modal_shift")
            st.markdown(by_id["C7"].detail)


# -------------------------------------------------------------- Environment
with TABS[2]:
    with panel("Environment tab"):
        section("Tidal forcing", "the constituents that drive the current and the water level")
        if art.tide_provenance == "MEASURED":
            st.success(f"MEASURED tidal forcing — {art.tide_source_note}")
            con = cfg.constituents()
            rows = []
            for i, nm in enumerate(con.names):
                rows.append({
                    "constituent": nm,
                    "elevation amp, m": round(float(con.elev_amp[i]), 4),
                    "current semi-major, m/s": round(float(con.semi_major[i]), 4),
                    "current semi-minor, m/s": round(float(con.semi_minor[i]), 4),
                    "major-axis inclination, deg": round(float(np.degrees(con.inclination[i])), 1),
                })
            dataframe(pd.DataFrame(rows))
            i = con.index("M2")
            cols = st.columns(3)
            with cols[0]:
                quantity(measured(float(con.elev_amp[i]), "m", "M2 elevation amplitude",
                                  con.citation))
            with cols[1]:
                quantity(measured(float(con.semi_major[i]), "m/s", "M2 current semi-major axis",
                                  con.citation))
            with cols[2]:
                ecc = abs(float(con.semi_minor[i]) / float(con.semi_major[i]))
                quantity(derived(ecc, "-", "M2 ellipse eccentricity", [],
                                 "|semi-minor| / semi-major",
                                 note="0 is a reversing channel current, 1 is circular. This is "
                                      "the single strongest predictor of the C3 nuisance floor "
                                      "(r = -0.95 across the stations tested)."))
            st.caption(
                "These are real published constants, but not the Arabian Gulf platform site. "
                "For that, a TPXO or FES extraction is required — see the Provenance tab. "
                "C3 returns the same verdict at every station tested."
            )
        else:
            tide_ok, tide_why = tide_model_status(cfg.tide_model_dir)
            if not tide_ok:
                unavailable_panel(
                    "Tidal constants are ASSUMED", tide_why,
                    "Select a NOAA station in the sidebar for MEASURED forcing, or configure "
                    "a TPXO/FES model for the platform site.")
            df = pd.DataFrame(PLACEHOLDER_CONSTITUENTS).T.reset_index(names="constituent")
            dataframe(df)
            for name in ("M2", "S2"):
                quantity(assumed(PLACEHOLDER_CONSTITUENTS[name]["elev_amp"], "m",
                                 f"{name} elevation amplitude"))

        try:
            _con = cfg.constituents()
        except Exception:  # noqa: BLE001 - plots are a nicety, never a failure mode
            _con = None

        if _con is not None:
            section("Tidal current ellipses",
                    "the shape that decides everything - see the C3 mechanism")
            fig = go.Figure()
            t_cyc = np.linspace(0.0, 2 * np.pi, 240)
            for nm in _con.names:
                i = _con.index(nm)
                a, b = float(_con.semi_major[i]), float(_con.semi_minor[i])
                inc = float(_con.inclination[i])
                u = a * np.cos(inc) * np.cos(t_cyc) - b * np.sin(inc) * np.sin(t_cyc)
                v = a * np.sin(inc) * np.cos(t_cyc) + b * np.cos(inc) * np.sin(t_cyc)
                fig.add_scatter(x=u, y=v, name=nm, mode="lines",
                                line=dict(width=3 if nm == "M2" else 1.5))
            fig.update_layout(
                title="Path traced by the tidal current over one cycle",
                xaxis_title="eastward current, m/s", yaxis_title="northward current, m/s",
                hovermode="closest", yaxis=dict(scaleanchor="x", scaleratio=1),
            )
            i = _con.index("M2")
            ecc = abs(float(_con.semi_minor[i]) / float(_con.semi_major[i]))
            figure_block(
                fig, "tidal_ellipses",
                f"M2 eccentricity {ecc:.3f}. A fat ellipse never goes slack and keeps the "
                "strain ratio well conditioned; a flat one passes through zero twice a cycle "
                "and the ratio degrades. This is the strongest predictor of the C3 nuisance "
                "floor.", 420,
            )

            section("Tidal forcing over time", "what actually drives the structure")
            t_s = np.arange(0.0, 14.0 * 86400.0, 600.0)
            eta = _con.elevation(t_s)
            uv = _con.depth_averaged_current(t_s)
            speed = np.hypot(uv[:, 0], uv[:, 1])
            fig = go.Figure()
            fig.add_scatter(x=t_s / 86400.0, y=eta, name="water level, m")
            fig.add_scatter(x=t_s / 86400.0, y=speed, name="current speed, m/s", yaxis="y2")
            fig.update_layout(
                title="Fourteen days of tidal elevation and current speed",
                xaxis_title="time, days", yaxis_title="water level, m",
                yaxis2=dict(title="current speed, m/s", overlaying="y", side="right",
                            showgrid=False),
            )
            figure_block(
                fig, "tidal_forcing",
                f"Spring/neap ratio {_con.spring_neap_ratio():.2f}, form factor "
                f"{_con.form_factor():.3f}. Drag scales with the square of speed, so the "
                "current trace - not the water level - carries most of the structural signal.",
                380,
            )

        section("ERA5 metocean", "wind, waves and temperature from the Copernicus reanalysis")
        era5_ok, era5_why = credentials_status()
        if not era5_ok:
            unavailable_panel("DATA UNAVAILABLE - CDS credentials not configured", era5_why)
        else:
            st.success(era5_why)

        section("Hydrodynamic coefficients", "drag and inertia, roughness-dependent")
        cd, cm, note = drag_inertia_coefficients(1.2, cfg.roughness_m)
        quantity(published(cd, "-", "drag coefficient Cd", API_RP2A, note=note))
        quantity(published(cm, "-", "inertia coefficient Cm", API_RP2A, note=note))

        section("C5 - S2 aliasing", "why M2 has to be the carrier, not S2")
        alias = art.c5["aliasing"]
        dataframe(pd.DataFrame(
            [{"pair": f"{a}/{b}", "days needed": f"{d:.2f}", "resolved": "yes" if ok else "no"}
             for a, b, d, ok in alias.rows]
        ))
        st.markdown(by_id["C5"].detail)


# ------------------------------------------------------------------ Sensing
with TABS[3]:
    with panel("Sensing tab"):
        section("C1 - intact tidal strain ratio", "what the two gauges read with no crack present")
        r = art.c1
        quantity(derived(r.ratio, "-", "M2 strain ratio (lower/upper)", [],
                         "harmonic fit of the frame solve", uncertainty=r.ratio_se))
        quantity(derived(r.amplitude_upper * 1e6, "ustrain", "M2 amplitude, upper gauge", [], "harmonic fit"))
        quantity(derived(r.amplitude_lower * 1e6, "ustrain", "M2 amplitude, lower gauge", [], "harmonic fit"))
        quantity(derived(
            r.resolution_margin, "x",
            "headroom over the sensor floor", [],
            f"weaker gauge's M2 amplitude divided by the {r.resolution_ustrain:g} microstrain "
            "resolution the abstract specifies",
            note="below 1 the signal cannot be read at all",
        ))
        if r.below_fbg_resolution:
            unavailable_panel(
                "Signal below sensor resolution",
                f"The weaker gauge reads {min(r.amplitude_upper, r.amplitude_lower) * 1e6:.3f} "
                f"microstrain against the {r.resolution_ustrain:g} microstrain resolution the "
                "abstract specifies for its FBG interrogator. The ratio between two quantities "
                "that cannot be measured is not measurable, whatever value the solver returns "
                "for it. This is the paper's own sensor specification, not a pessimistic one "
                "chosen here.",
            )
        st.markdown(by_id["C1"].detail)

        n = min(len(r.times_s), 4000)
        fig = go.Figure()
        fig.add_scatter(x=r.times_s[:n] / 86400, y=r.eps_upper[:n] * 1e6, name="upper gauge")
        fig.add_scatter(x=r.times_s[:n] / 86400, y=r.eps_lower[:n] * 1e6, name="lower gauge")
        fig.update_layout(title="Tidal strain at the bracketing pair",
                          xaxis_title="time, days", yaxis_title="axial surface strain, microstrain")
        figure_block(fig, "c1_strain_series")

        fig = go.Figure()
        names = list(r.fit_upper.names)
        fig.add_bar(x=names, y=r.fit_upper.amplitude * 1e6, name="upper",
                    error_y=dict(array=r.fit_upper.amplitude_se * 1e6))
        fig.add_bar(x=names, y=r.fit_lower.amplitude * 1e6, name="lower",
                    error_y=dict(array=r.fit_lower.amplitude_se * 1e6))
        fig.update_layout(title="Harmonic amplitudes with standard errors",
                          xaxis_title="constituent", yaxis_title="strain amplitude, microstrain",
                          barmode="group")
        figure_block(fig, "c1_harmonics")

        section("Quadrature decomposition", "separating the current-driven signal from the tide-level one")
        st.caption(
            "Drag is in quadrature with elevation; buoyancy is in phase with it. The ratio mixes "
            "both, which is why changing the current without changing the tide range moves it."
        )
        dataframe(pd.DataFrame([
            {"gauge": "upper", "in-phase (buoyancy), ustrain": r.split_upper.in_phase * 1e6,
             "quadrature (drag), ustrain": r.split_upper.quadrature * 1e6,
             "drag share of variance": f"{r.split_upper.drag_fraction * 100:.1f} %"},
            {"gauge": "lower", "in-phase (buoyancy), ustrain": r.split_lower.in_phase * 1e6,
             "quadrature (drag), ustrain": r.split_lower.quadrature * 1e6,
             "drag share of variance": f"{r.split_lower.drag_fraction * 100:.1f} %"},
        ]))


# ---------------------------------------------------------------- Detection
with TABS[4]:
    with panel("Detection tab"):
        if art.c2 is None or art.c3 is None:
            unavailable_panel("Not computed", "Press Run full analysis to compute C2, C3 and C4.")
        else:
            section("C2 - damage sensitivity", "how much a crack moves the strain ratio")

            # The paper's own intermediate step, tested directly. This is the
            # primary evidence and is drawn first: it needs no crack model, so
            # it cannot be answered by disputing how the crack was modelled.
            sr = art.c2_stiffness
            if sr is not None:
                st.markdown(
                    "**The paper's own intermediate step.** The abstract states that a 20 "
                    "percent through-wall crack produces a **10 percent joint stiffness "
                    "reduction**, and that this is what takes the ratio from 1.800 to 2.000. "
                    "The second link is pure structural mechanics, so it can be tested "
                    "exactly: impose the stated reduction, re-solve, read the ratio. No "
                    "crack model and no shell-FE surface enter."
                )
                fig = go.Figure()
                colours = {"axial": "#4a545e", "ipb": "#8a5300",
                           "opb": "#1a7f43", "all": "#35b6c4"}
                for key, arr in sr.ratios_by_mode.items():
                    fig.add_trace(go.Scatter(
                        x=sr.reductions * 100, y=sr.change(key) * 100,
                        name=MODE_SETS[key].label, mode="lines+markers",
                        line=dict(color=colours.get(key), width=3 if key == sr.best_mode else 2,
                                  dash=None if key == sr.best_mode else "dot"),
                    ))
                fig.add_hline(
                    y=sr.claimed_signature * 100, line_color="#b3261e", line_width=2,
                    annotation_text=f"claimed {sr.claimed_signature * 100:.1f} %",
                    annotation_position="top left",
                )
                fig.add_vline(
                    x=10.0, line_color="#b3261e", line_dash="dash",
                    annotation_text="the paper's 10 % reduction", annotation_position="top",
                )
                fig.update_layout(
                    title=(f"Strain-ratio change against imposed joint stiffness reduction, "
                           f"joint J{sr.joint_id}"),
                    xaxis_title="joint stiffness removed, %",
                    yaxis_title="change in the strain ratio, %",
                    hovermode="x unified", legend=dict(orientation="h", y=-0.22),
                )
                figure_block(fig, "c2_stiffness_sweep", height=440)
                st.caption(
                    "The abstract does not say *which* stiffness, so all four readings are "
                    "swept and the claim is judged on whichever helps it most (solid line). "
                    "That matters: the axial spring moves the ratio the wrong way, while "
                    "out-of-plane bending moves it the right way and roughly twelve times "
                    "as far."
                )
                mc = st.columns(len(sr.change_by_mode))
                for col, (key, val) in zip(mc, sr.change_by_mode.items()):
                    with col:
                        quantity(derived(
                            val * 100.0, "%",
                            f"reduce {MODE_SETS[key].label}", [],
                            "impose a 10 % reduction on that LJF spring and re-solve "
                            "the frame; change in the M2 strain ratio",
                            note=("most favourable to the claim" if key == sr.best_mode
                                  else ""),
                        ))
                st.markdown(f"**{sr.verdict}**")

                # Answer the "wrong joint" objection with a number, not an argument.
                pj = art.c2_stiffness_paper
                if pj is not None:
                    st.markdown(
                        "**Tested at the abstract's own joint.** The K-joint the paper "
                        "specifies - 762 mm chord, 25 mm wall, 45 degree brace - is softer "
                        "than every joint on this frame, and a softer joint takes more of "
                        "the local load path. So the sweep is repeated with the instrumented "
                        "joint softened to exactly that flexibility."
                    )
                    jc = st.columns(2)
                    with jc[0]:
                        quantity(derived(
                            sr.at_claimed_reduction * 100.0, "%",
                            "at this frame's own joint", [],
                            f"10 % reduction, {MODE_SETS[sr.best_mode].label}",
                        ))
                    with jc[1]:
                        quantity(derived(
                            pj.at_claimed_reduction * 100.0, "%",
                            "softened to the paper's K-joint", [],
                            f"10 % reduction, {MODE_SETS[pj.best_mode].label}",
                            note="the paper's own stated joint geometry",
                        ))
                    st.caption(
                        "The two agree to well within the factor of thirty that separates "
                        "either from the claim, so the verdict does not rest on the choice "
                        "of joint."
                    )
                st.divider()
                st.caption(
                    "Below: the independent crack-model route, retained as corroboration. "
                    "It reaches the same conclusion from a completely different direction."
                )

            g = art.c2
            fig = go.Figure(go.Contour(
                x=g.surface_length_m * 1e3, y=g.a_over_T, z=np.abs(g.delta_fraction) * 100,
                colorbar=dict(title="|change|, %"), colorscale="Teal",
            ))
            fig.update_layout(title=f"Strain-ratio change at joint J{g.joint_id} ({g.route})",
                              xaxis_title="surface length 2c, mm", yaxis_title="a / T")
            figure_block(fig, "c2_damage_contour")
            for cav in g.caveats:
                st.caption(cav)
            st.markdown(by_id["C2"].detail)

            section("C3 - nuisance variance budget", "how much the sea moves it with no crack at all")
            n = art.c3

            # The whole finding in one picture: the spread of the ratio with no
            # crack anywhere, against the step a crack is claimed to produce.
            sig = 0.111
            base = n.baseline_ratio
            fa = n.false_alarm_fraction(sig)
            fig = go.Figure()
            fig.add_histogram(x=n.joint_samples, nbinsx=70, marker_color="#35b6c4",
                              name="intact ratio under environmental variation")
            fig.add_vline(x=base, line_color="#111418", line_width=2,
                          annotation_text="intact", annotation_position="top left")
            for sgn, pos in ((+1, "top right"), (-1, "top left")):
                fig.add_vline(
                    x=base * (1 + sgn * sig), line_color="#b3261e", line_dash="dash",
                    line_width=2,
                    annotation_text=f"claimed damage step ({sgn * sig * 100:+.1f} %)",
                    annotation_position=pos,
                )
            fig.update_layout(
                title="With no crack at all: where the sea alone puts the strain ratio",
                xaxis_title="intact M2 strain ratio", yaxis_title="Monte Carlo draws",
                showlegend=False, hovermode="x",
            )
            figure_block(fig, "c3_noise_vs_signal", height=430)
            st.markdown(
                f"**{fa * 100:.1f} percent of draws with no damage present already move the "
                f"ratio by at least the {sig * 100:.1f} percent a crack is claimed to move "
                "it.** Every one of those is a false alarm from a detector set at the "
                "claimed signature. The dashed lines are where a crack is supposed to put "
                "the ratio; the histogram is where the sea puts it on its own. They overlap, "
                "and that overlap is the failure - no threshold placed on this axis separates "
                "a cracked structure from an intact one on a rough fortnight."
            )
            st.caption(
                "Counted two-sided on the magnitude of the change: a detector watching for an "
                "11.1 percent shift cannot know which way a real crack would push this ratio. "
                "C2 finds the sign depends on which joint spring softens - axial drives it "
                "down, out-of-plane bending drives it up."
            )

            rnd, sysm = n.split_random_systematic()
            chans = sorted(n.per_channel_sd.items(), key=lambda kv: kv[1])
            fig = go.Figure(go.Bar(
                x=[v / abs(n.baseline_ratio) * 100 for _k, v in chans],
                y=[CHANNEL_LABELS[k] for k, _v in chans], orientation="h",
                marker_color="#35b6c4",
            ))
            fig.add_vline(x=n.joint_cv * 100, line_color="#d1495b",
                          annotation_text=f"joint sigma {n.joint_cv*100:.2f} %")
            fig.update_layout(title="Nuisance contributions to the intact strain ratio",
                              xaxis_title="standard deviation, % of the intact ratio",
                              yaxis_title="", hovermode="y unified")
            figure_block(fig, "c3_nuisance_budget", height=440)

            cc = st.columns(4 if getattr(n, "heavy_tailed", False) else 3)
            if getattr(n, "heavy_tailed", False):
                with cc[3]:
                    quantity(derived(
                        n.joint_robust_sd / abs(n.baseline_ratio), "-",
                        "robust scale (used for the verdict)", [],
                        "0.7413 x IQR, the normal-consistent estimator",
                        note="Used because the variance does not exist at this site."))
            with cc[0]:
                quantity(n.as_quantity())
            with cc[1]:
                quantity(derived(rnd / abs(n.baseline_ratio), "-", "averageable (random) part", [],
                                 "quadrature sum of per-record channels"))
            with cc[2]:
                quantity(derived(sysm / abs(n.baseline_ratio), "-", "detection floor (systematic)", [],
                                 "quadrature sum of slowly varying channels"))
            for ch, why in n.gated_channels.items():
                st.caption(f"{CHANNEL_LABELS[ch]}: {why}")

            if getattr(n, "heavy_tailed", False):
                unavailable_panel(
                    "The variance of the strain ratio does not exist at this site",
                    "The ratio's denominator is the upper gauge's M2 amplitude, which approaches "
                    "zero when a reversing tidal current slackens. A ratio with a near-zero "
                    f"denominator is Cauchy-like: the standard deviation here is "
                    f"{n.joint_sd / max(n.joint_robust_sd, 1e-12):.1f} times a robust scale of the "
                    "same sample, and it grows with sample count instead of converging. More "
                    "samples cannot fix that. The verdict below uses the robust scale "
                    "(0.7413 x IQR), which is well defined.",
                    "A detection threshold cannot be set from a quantity with no second moment. "
                    "This is a finding against the method, not a limitation of this analysis.",
                )

            if getattr(n, "convergence", None) is not None:
                section("Has the deciding test converged?",
                        "an unconverged Monte Carlo has not decided anything")
                fig = go.Figure()
                fig.add_scatter(x=n.convergence.n_samples,
                                y=n.convergence.sigma / abs(n.baseline_ratio) * 100,
                                name="running sigma", mode="lines+markers")
                fig.add_hline(y=n.joint_cv * 100, line_dash="dot",
                              annotation_text="final estimate")
                fig.update_layout(title="Monte Carlo convergence of the joint sigma",
                                  xaxis_title="samples drawn",
                                  yaxis_title="sigma, % of the intact ratio")
                figure_block(fig, "c3_convergence", n.convergence.verdict, height=320)

            if getattr(n, "break_even", None) is not None:
                be = n.break_even
                section("Break-even", "how much quieter would the sea have to be for C3 to pass?")
                fig = go.Figure()
                fig.add_scatter(x=be.scales, y=be.sigmas * 100, name="joint sigma",
                                mode="lines+markers")
                fig.add_hline(y=be.threshold * 100, line_dash="dash", line_color="#b3261e",
                              annotation_text="pass threshold")
                if np.isfinite(be.factor) and be.factor < 1.0:
                    fig.add_vline(x=be.factor, line_dash="dot",
                                  annotation_text=f"break-even {be.factor:.2f}x")
                fig.update_layout(
                    title="Joint nuisance sigma against a common scaling of every assumed range",
                    xaxis_title="scale applied to every nuisance range, x",
                    yaxis_title="sigma, % of the intact ratio")
                figure_block(fig, "c3_break_even", be.statement, height=340)

            # The proposed fix, stress-tested against real gauges. This is about
            # the four-gauge rosette regardless of which layout is selected, so it
            # is shown in both modes: it is the honest question about whether the
            # fix survives an actual deployment, not just a perfect one.
            section("Does the rosette fix survive real gauges?",
                    "placement and gain tolerance a deployment can actually deliver")
            gr = _gauge_robustness_sweep()
            typ = gr["typical"]
            gcols = st.columns(3)
            with gcols[0]:
                quantity(derived(gr["placement_only"] * 100, "%",
                                 "from 3 deg placement error", [],
                                 "gauge misplacement leaks bending into the axial estimate"))
            with gcols[1]:
                quantity(derived(gr["gain_only"] * 100, "%",
                                 "from 1 % gain mismatch", [],
                                 "unmatched gains bias the four-gauge average"))
            with gcols[2]:
                quantity(derived(typ.combined_over_signature, "x",
                                 "combined vs signature (limit 0.33)", [],
                                 "gauge and environmental dispersion in quadrature, over "
                                 "the claimed damage signature",
                                 note="PASS" if typ.passes else "FAIL"))
            fig = go.Figure()
            fig.add_scatter(x=gr["angles"], y=gr["disp"] * 100, mode="lines+markers",
                            name="gauge-induced ratio dispersion")
            fig.add_hline(y=typ.environmental_cv * 100, line_dash="dot", line_color="#5b646d",
                          annotation_text="environmental floor")
            fig.add_hline(y=typ.damage_signature * typ.threshold_fraction * 100,
                          line_dash="dash", line_color="#b3261e",
                          annotation_text="C3 pass threshold (1/3 signature)")
            fig.update_layout(
                title="Strain-ratio dispersion from gauge imperfection",
                xaxis_title="gauge placement 1sd, degrees (gain scaled 0-2.8 %)",
                yaxis_title="ratio dispersion, % of the ratio")
            figure_block(fig, "c3_gauge_robustness", height=360)
            st.caption(
                "The fix holds. Even at loose tolerances the gauge-induced dispersion stays "
                "below the environmental floor it is added to, and the combined total is "
                f"{typ.combined_over_signature:.2f}x the damage signature against a 0.33 limit "
                "- a comfortable pass. And the instrumentation lesson is clear: **matching "
                "the gauge gains matters several times more than placing them precisely**, "
                "because the axial average is well conditioned and the bending a misplacement "
                "leaks is only a tenth of the signal. Specify matched interrogator channels "
                "before tight survey tolerances."
            )

            if getattr(n, "decomposition", None) is not None:
                d = n.decomposition
                section("Why the joint sigma is not the sum of the parts", "interaction between channels")
                cols = st.columns(3)
                with cols[0]:
                    quantity(derived(np.sqrt(d.sum_individual_variance) / abs(n.baseline_ratio), "-",
                                     "sum of channels, in quadrature", [],
                                     "sqrt of the summed per-channel variances"))
                with cols[1]:
                    quantity(derived(np.sqrt(max(d.joint_variance, 0.0)) / abs(n.baseline_ratio), "-",
                                     "measured jointly", [], "joint Monte Carlo"))
                with cols[2]:
                    quantity(derived(d.interaction_fraction, "-", "interaction term", [],
                                     "(joint - sum) / sum of variances"))
                st.caption(d.interpretation)

            if art.c4 is not None:
                section("C4 - time to detection", "how long before a crack is distinguishable")
                d = art.c4
                p = d["percentiles"]
                fig = go.Figure()
                if d["times_days"].size:
                    fig.add_scatter(x=d["times_days"], y=d["cdf"], name="detection time CDF")
                for q, lab in (("p05", "5th"), ("p50", "50th"), ("p95", "95th")):
                    if np.isfinite(p[q]):
                        fig.add_vline(x=p[q], line_dash="dot", annotation_text=f"{lab} {p[q]:.1f} d")
                fig.add_vrect(x0=4, x1=9, fillcolor="#d1495b", opacity=0.12, line_width=0,
                              annotation_text="claimed 4-9 d")
                fig.update_layout(title="Time to detect the modelled crack",
                                  xaxis_title="time, days", yaxis_title="cumulative probability")
                figure_block(fig, "c4_detection_cdf")
                st.caption(
                    f"{p['never_detected_fraction']*100:.0f} % of trials never detect within the "
                    "simulated horizon."
                )

                from tidetwin.signal.detect import coherent_gain_curve

                nrec = np.arange(1, 101)
                th, ach = coherent_gain_curve(d["model"], nrec)
                fig = go.Figure()
                fig.add_scatter(x=nrec, y=th, name="theoretical 1/sqrt(N)")
                fig.add_scatter(x=nrec, y=ach, name="achieved")
                fig.update_layout(title="Coherent averaging gain against the theoretical curve",
                                  xaxis_title="records averaged, N",
                                  yaxis_title="noise relative to a single record")
                figure_block(fig, "c4_coherent_gain")
                st.caption(
                    "The achieved curve flattens onto the systematic floor. Averaging longer stops "
                    "helping there, which is why the claimed detection window is not reached."
                )
                st.markdown(by_id["C4"].detail)

            if art.c9 is not None:
                section("C9 - probability of detection", "against the inspection methods it would replace")
                p9 = art.c9
                fig = go.Figure()
                fig.add_scatter(x=p9.crack_depth_m * 1e3, y=p9.pod, name="tidal strain (part-through)")
                for nm, dfc in p9.competitor_curves.items():
                    fig.add_scatter(x=dfc["crack_depth_mm"], y=dfc["pod"], name=nm)
                fig.add_hline(y=0.9, line_dash="dash", annotation_text="POD 0.9")
                fig.update_layout(title="POD(a)", xaxis_title="crack depth a, mm",
                                  yaxis_title="probability of detection")
                figure_block(fig, "c9_pod")
                if not p9.competitor_curves:
                    unavailable_panel("DATA UNAVAILABLE - competitor POD curves", p9.competitor_note)
                st.markdown(by_id["C9"].detail)


# ------------------------------------------------------------- Assimilation
with TABS[5]:
    with panel("Assimilation tab"):
        if art.c6 is None:
            unavailable_panel("Not computed", "Press Run full analysis to run the filters.")
        else:
            c6 = art.c6
            section("C6 - filters against a fraternal twin",
                    "the truth model differs from the filter's, as it would in reality")
            st.caption(c6.model_error_note)
            fig = go.Figure()
            fig.add_scatter(x=c6.times_years, y=c6.truth * 1e3, name="truth",
                            line=dict(color="#d7dde5", width=3))
            for nm, E in c6.ensembles.items():
                med = np.median(E, axis=1) * 1e3
                lo = np.percentile(E, 5, axis=1) * 1e3
                hi = np.percentile(E, 95, axis=1) * 1e3
                fig.add_scatter(x=c6.times_years, y=med, name=nm)
                fig.add_scatter(x=np.concatenate([c6.times_years, c6.times_years[::-1]]),
                                y=np.concatenate([hi, lo[::-1]]), fill="toself", opacity=0.15,
                                line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.update_layout(title="Crack depth: truth, medians and 90 % intervals",
                              xaxis_title="time, years", yaxis_title="crack depth, mm")
            figure_block(fig, "c6_filters")

            dataframe(pd.DataFrame([
                {"filter": r.name, "RMSE, mm": r.rmse * 1e3, "CRPS": r.crps,
                 "90 % coverage": f"{r.coverage_90*100:.0f} %",
                 "mean 90 % width, mm": r.mean_interval_width_90 * 1e3,
                 "verdict": r.verdict}
                for r in c6.reports.values()
            ]))
            for r in c6.reports.values():
                st.caption(f"**{r.name}** &mdash; {r.comment}")

            # The abstract's own convergence criterion, drawn so the reader can
            # see the thing that matters: the do-nothing baseline sits inside the
            # same target box, which is what makes the criterion uninformative.
            ac = c6.abstract_convergence()
            section("The abstract's own convergence criterion",
                    "within 8 percent of true damage by month 18 - and what that is worth")
            fig = go.Figure()
            fig.add_shape(
                type="rect", x0=0, x1=ac["by_years"], y0=0, y1=ac["tolerance"] * 100,
                fillcolor="#1a7f43", opacity=0.10, line=dict(width=0), layer="below",
            )
            fig.add_annotation(
                x=ac["by_years"] / 2, y=ac["tolerance"] * 100, yshift=12, showarrow=False,
                text=f"the claim: within {ac['tolerance'] * 100:.0f} % by month "
                     f"{ac['by_years'] * 12:.0f}",
                font=dict(color="#1a7f43", size=12),
            )
            for nm in c6.ensembles:
                dash = "dash" if nm == "no-update baseline" else None
                fig.add_scatter(
                    x=c6.times_years, y=c6.relative_error(nm) * 100, name=nm,
                    line=dict(dash=dash, width=3 if nm == "log-EnKF" else 2),
                )
            fig.add_hline(y=ac["tolerance"] * 100, line_dash="dot", line_color="#1a7f43")
            fig.add_vline(x=ac["by_years"], line_dash="dot", line_color="#1a7f43")
            fig.update_layout(
                title="Distance from the truth over time, against the abstract's target",
                xaxis_title="time, years", yaxis_title="|estimate - truth| / truth, %",
                hovermode="x unified", legend=dict(orientation="h", y=-0.22),
            )
            figure_block(fig, "c6_abstract_convergence", height=420)
            cc6 = st.columns(3)
            with cc6[0]:
                quantity(derived(
                    ac["error_at_deadline"] * 100.0, "%", "log-EnKF at month 18", [],
                    "|ensemble mean - truth| / truth at the abstract's deadline",
                ))
            with cc6[1]:
                quantity(derived(
                    ac["baseline_error_at_deadline"] * 100.0, "%",
                    "no-update baseline at month 18", [],
                    "the same distance for a filter that assimilates nothing",
                    note="assimilates nothing and still passes",
                ))
            with cc6[2]:
                _b = ac["baseline_error_at_deadline"]
                quantity(derived(
                    ac["error_at_deadline"] / _b if _b else float("nan"), "x",
                    "filter error / baseline error", [],
                    "ratio of the two distances above, at the abstract's month-18 checkpoint",
                    note=("above 1: further from the truth than assimilating nothing"
                          if _b and ac["error_at_deadline"] > _b else ""),
                ))
            if ac["met"] and ac["baseline_also_meets"]:
                st.warning(
                    "**The criterion is met, and it is worthless.** Both curves start inside "
                    "the target box and stay there. Assimilating nothing at all clears the "
                    "abstract's 8-percent-by-month-18 test just as comfortably as the "
                    "log-EnKF does, so passing it is not evidence that the filter works. "
                    "What separates the estimators is the calibration comparison below - "
                    "CRPS, coverage and the rank histogram - and that is what C6's status "
                    "rests on.",
                    icon=":material/warning:",
                )

            cols = st.columns(2)
            with cols[0]:
                fig = go.Figure()
                for nm, r in c6.reports.items():
                    fig.add_bar(x=np.arange(len(r.ranks)), y=r.ranks, name=nm, opacity=0.7)
                fig.update_layout(title="Rank histogram (flat = calibrated)",
                                  xaxis_title="rank of truth within the ensemble",
                                  yaxis_title="count", barmode="overlay")
                figure_block(fig, "c6_rank_histogram", height=340)
            with cols[1]:
                fig = go.Figure()
                for nm, r in c6.reports.items():
                    xs = np.sort(r.pit)
                    fig.add_scatter(x=xs, y=np.arange(1, xs.size + 1) / xs.size, name=nm)
                fig.add_scatter(x=[0, 1], y=[0, 1], name="uniform", line=dict(dash="dash"))
                fig.update_layout(title="PIT diagram", xaxis_title="PIT value",
                                  yaxis_title="empirical CDF")
                figure_block(fig, "c6_pit", height=340)
            st.markdown(by_id["C6"].detail)


# ---------------------------------------------------------------- Economics
with TABS[6]:
    with panel("Economics tab"):
        section("C8 - net present value", "a scenario, not a result")
        unavailable_panel(
            "Every input on this page is ASSUMED",
            "None of these figures derives from a solver, a standard or a dataset. The NPV is a "
            "scenario, not a result. It is also downstream of C3: a method that does not detect "
            "reliably cannot avoid the campaigns this model credits it with avoiding.",
        )
        if art.c8 is None:
            unavailable_panel("Not computed", "Press Run full analysis.")
        else:
            r8 = art.c8
            cols = st.columns(4)
            with cols[0]:
                quantity(r8.as_quantity())
            with cols[1]:
                quantity(assumed(r8.median / 1e6, "MUSD", "median NPV"))
            with cols[2]:
                quantity(assumed(r8.percentile(5) / 1e6, "MUSD", "5th percentile NPV"))
            with cols[3]:
                quantity(assumed(r8.probability_positive, "-", "probability NPV > 0"))

            # The distribution is per jacket; the abstract's 19.9 MUSD is for a
            # fleet of 30. Drawing the fleet figure on a per-jacket axis made the
            # claim look about thirty times more ambitious than it is.
            n_j = r8.inputs.n_jackets
            fig = go.Figure(go.Histogram(x=r8.samples / 1e6, nbinsx=80, marker_color="#35b6c4"))
            fig.add_vline(x=0, line_color="#d1495b")
            fig.add_vline(x=19.9 / n_j, line_dash="dash", line_color="#c8871b",
                          annotation_text=f"claimed 19.9 MUSD / {n_j} jackets "
                                          f"= {19.9 / n_j:.2f} each")
            fig.update_layout(title="NPV distribution, per jacket",
                              xaxis_title="NPV, million USD",
                              yaxis_title="count", hovermode="x")
            figure_block(fig, "c8_npv")
            st.caption(
                f"Per jacket. The abstract's headline is a fleet figure - 19.9 MUSD across "
                f"{n_j} ADNOC jackets - so it is divided by {n_j} to sit on this axis. "
                f"Across the fleet this model gives {r8.fleet_mean / 1e6:.1f} MUSD. The "
                "commercial case is not where the paper is aggressive."
            )

            # Comparing a fleet claim against a per-jacket model was a real error
            # here, and one that ran against the paper. Both bases are drawn so
            # the comparison cannot be misread again.
            fig = go.Figure()
            fig.add_bar(name="this model", x=["per jacket", f"fleet of {n_j}"],
                        y=[r8.mean / 1e6, r8.fleet_mean / 1e6], marker_color="#35b6c4",
                        text=[f"{r8.mean / 1e6:.2f}", f"{r8.fleet_mean / 1e6:.1f}"],
                        textposition="outside")
            fig.add_bar(name="the abstract", x=["per jacket", f"fleet of {n_j}"],
                        y=[19.9 / n_j, 19.9], marker_color="#c8871b",
                        text=[f"{19.9 / n_j:.2f}", "19.9"], textposition="outside")
            fig.update_layout(
                title="Like for like: the claim and this model on both bases",
                yaxis_title="net present value, million USD", barmode="group",
                legend=dict(orientation="h", y=-0.18),
            )
            figure_block(fig, "c8_basis", height=380)
            st.caption(
                "On either basis this model is the **more** optimistic of the two. C8 used to "
                f"compare {r8.mean / 1e6:.2f} MUSD per jacket against the abstract's 19.9 "
                f"MUSD fleet figure and read that as the paper being an order of magnitude "
                "out; the arithmetic was the error, not the paper. Fleet scales linearly "
                "here - no shared interrogator, no shared spares, no mobilisation spread "
                "across platforms - all of which would raise it further, so this understates "
                "the fleet case rather than flattering it."
            )

            with st.spinner("tornado sensitivity..."):
                t8 = tornado(cfg.economics, n_samples=2000, seed=cfg.seed)[:10]
            fig = go.Figure()
            labels = [r[0] for r in t8][::-1]
            lows = np.array([r[1] for r in t8][::-1]) / 1e6
            highs = np.array([r[2] for r in t8][::-1]) / 1e6
            base = r8.mean / 1e6
            fig.add_bar(y=labels, x=lows - base, base=base, orientation="h", name="-30 %",
                        marker_color="#d1495b")
            fig.add_bar(y=labels, x=highs - base, base=base, orientation="h", name="+30 %",
                        marker_color="#35b6c4")
            fig.update_layout(title="Tornado: which assumption the answer is hostage to",
                              xaxis_title="expected NPV, million USD", yaxis_title="",
                              barmode="overlay", hovermode="y unified")
            figure_block(fig, "c8_tornado", height=460)
            st.markdown(by_id["C8"].detail)


# ------------------------------------------------------------------- Ledger
with TABS[7]:
    with panel("Ledger tab"):
        section("Claims ledger", "the table a paper would cite, with the stamp that makes it checkable")
        st.caption(
            "Each claim is shown as the abstract words it, not as a paraphrase written "
            "here. A paraphrase is exactly where a claim can be quietly softened before "
            "it is judged, so the paper's own text is what appears."
        )
        for c in CLAIMS:
            r = by_id[c.id]
            cols = st.columns([1, 6, 3])
            cols[0].markdown(f"**{c.id}**")
            with cols[1]:
                if c.quote:
                    st.markdown(f"> *{c.quote}*")
                    with st.expander("as tested here"):
                        st.markdown(c.statement)
                        st.caption(f"Pass criterion: {c.pass_criterion}")
                else:
                    st.markdown(c.statement)
            cols[2].markdown(status_chip(r.status), unsafe_allow_html=True)
        st.divider()
        dataframe(ledger_frame(results))

        section("Reproducibility stamp", "what produced these verdicts, exactly")
        dataframe(STAMP.as_frame())

        cols = st.columns(2)
        with cols[0]:
            st.download_button("Export CSV", to_csv(results, STAMP).encode(),
                               file_name="tidetwin_claims_ledger.csv", mime="text/csv",
                               width="stretch")
        with cols[1]:
            st.download_button("Export LaTeX", to_latex(results, STAMP).encode(),
                               file_name="tidetwin_claims_ledger.tex", mime="text/x-tex",
                               width="stretch")

        section("Full report", "always produced, whatever the inputs and whatever is missing")
        st.caption(
            "Produced for whatever inputs and data sources are in force, including partial runs. "
            "Claims that could not be evaluated appear in the report saying so."
        )
        rep_inputs = ReportInputs.from_config(cfg)
        cols = st.columns(3)
        with cols[0]:
            st.download_button("Report (HTML)", to_html(results, art, STAMP, rep_inputs).encode(),
                               file_name="tidetwin_report.html", mime="text/html",
                               width="stretch")
        with cols[1]:
            st.download_button("Report (Markdown)",
                               to_markdown(results, art, STAMP, rep_inputs).encode(),
                               file_name="tidetwin_report.md", mime="text/markdown",
                               width="stretch")
        with cols[2]:
            st.download_button("Report (text)", to_text(results, art, STAMP).encode(),
                               file_name="tidetwin_report.txt", mime="text/plain",
                               width="stretch")
        with st.expander("Preview the report"):
            st.markdown(to_markdown(results, art, STAMP, rep_inputs))


# --------------------------------------------------------------- Provenance
with TABS[8]:
    with panel("Provenance tab"):
        section("Data source status", "what is real here, and what is honestly missing")
        st.markdown(
            '<span class="tt-chip" style="color:#1a7f43">AVAILABLE</span> &nbsp; '
            "**OC4 jacket geometry**", unsafe_allow_html=True,
        )
        st.caption(str(OC4_CITATION))
        st.markdown(
            '<span class="tt-chip" style="color:#1a7f43">AVAILABLE</span> &nbsp; '
            "**NOAA tidal harmonic constants**", unsafe_allow_html=True,
        )
        st.caption(f"{len(available_cached())} station pairs cached, water level and "
                   "tidal current ellipses, no account required.")

        section("Unlock the rest",
                "what each missing piece would buy, what it costs, where to put it")
        gates = gate_status()
        n_open = sum(1 for _g, ok, _w in gates if ok)
        st.caption(
            f"{n_open} of {len(gates)} optional sources are present. Four of the nine "
            "claims are currently UNTESTABLE for want of the items below. Each is a real "
            "document or dataset with a specific home in this repository — none is a "
            "placeholder waiting for a guess."
        )
        dataframe(pd.DataFrame([
            {
                "Source": g.name,
                "Status": "present" if ok else "missing",
                "Unlocks": ", ".join(g.claims) or "groundwork",
                "Cost": g.cost,
                "Effort": g.effort,
            }
            for g, ok, _why in gates
        ]))
        for g, ok, why in gates:
            colour = "#1a7f43" if ok else "#8a5300"
            with st.expander(
                f"{'✓' if ok else '○'}  {g.name}"
                + (f"  —  unlocks {', '.join(g.claims)}" if g.claims else "")
            ):
                st.markdown(
                    f'<span class="tt-chip" style="color:{colour}">'
                    f'{"PRESENT" if ok else "MISSING"}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**What it unlocks** — {g.unlocks}")
                st.markdown(f"**Cost** — {g.cost} &nbsp;&nbsp; **Effort** — {g.effort}")
                st.markdown(f"**How to get it** — {g.how}")
                st.markdown(f"**Where it goes** — `{g.where}`")
                if not ok:
                    st.caption(why)

        section("Modelling assumptions in force", "choices that change the answer")
        if art.modelling_assumptions:
            for a in art.modelling_assumptions:
                st.markdown(f"- {a}")
        else:
            st.caption("None recorded for this configuration.")

        section("Provenance classes", "every number on every tab carries one of these")
        provenance_legend()
        st.markdown(
            "- **MEASURED** &mdash; from a real external dataset, cited with retrieval date.\n"
            "- **PUBLISHED** &mdash; a constant from a standard or paper, cited with locator.\n"
            "- **DERIVED** &mdash; computed by our solvers from MEASURED and PUBLISHED inputs.\n"
            "- **ASSUMED** &mdash; set by you. Renders red, and everything computed from it "
            "inherits the flag and lists what it rests on."
        )
        section("Geometry manifest", "the hash that proves the shipped tables were not edited")
        st.json(TABLES.manifest)
