"""TideTwin - adversarial test bench for the tidal-calibration fatigue twin claims.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import sys
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

from tidetwin.analysis import AnalysisConfig, run_full, run_quick
from tidetwin.claims.ledger import build_stamp, ledger_frame, to_csv, to_latex
from tidetwin.claims.registry import CLAIMS, Artifacts, Status, evaluate_all
from tidetwin.damage.crack_ljf import shell_fe_status
from tidetwin.damage.paris import paris_status
from tidetwin.damage.sn import sn_status
from tidetwin.economics.npv import EconomicInputs, tornado
from tidetwin.fe.ljf import JointGeometry, LJFModel, ljf_quantity
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
from tidetwin.ui import (
    dataframe,
    figure_block,
    inject_theme,
    loading_screen,
    masthead,
    provenance_legend,
    quantity,
    section,
    status_chip,
    unavailable_panel,
    verdict_block,
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


def sidebar() -> AnalysisConfig:
    s = st.sidebar
    s.markdown("### TideTwin")
    s.caption("Every red chip is an assumption. Results inherit them.")

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

    s.markdown("**Crack geometry**")
    s.caption("a/T and 2c are independent. There is no single percent-through-wall.")
    a_over_T = s.slider("a / T  (depth over wall thickness)", 0.05, 0.90, 0.50, 0.05)
    two_c = s.slider("2c  surface length, mm", 20.0, 400.0, 100.0, 10.0) * 1e-3

    s.markdown("**Record and Monte Carlo**")
    days = s.slider("Record length, days", 7.0, 90.0, 30.0, 1.0)
    n_mc = s.select_slider("MC samples per channel", [50, 100, 200, 400, 800], value=200)
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
            fbg_drift_sd_ustrain=st.slider("Differential FBG drift, ustrain 1sd", 0.0, 5.0, 0.5),
            fbg_noise_ustrain=st.slider("FBG noise, ustrain rms", 0.0, 2.0, 0.2),
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
        tide_table=tide_table,
        tide_source=tide_source,
        tide_station=tide_station,
    )


@st.cache_data(show_spinner="Solving the frame...", max_entries=8)
def _quick(key: tuple) -> Artifacts:
    return run_quick(_cfg_from_key(key))


_CFG_CACHE: dict[tuple, AnalysisConfig] = {}


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
        cfg.tide_station,
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
if "booted" not in st.session_state:
    _boot.markdown(loading_screen(), unsafe_allow_html=True)
art_quick = _quick(key)
st.session_state.booted = True
_boot.empty()

if "full" not in st.session_state:
    st.session_state.full = None
    st.session_state.full_key = None

# A code reload or a redeploy re-defines the dataclasses, which leaves anything
# held in session state an instance of the *old* class. Reading a field the new
# code expects then raises AttributeError and takes the whole page down with it.
# Discard a stale artifact instead of half-trusting it: recomputing is cheap,
# and a crash on a tab the user did not ask about is not.
if st.session_state.full is not None and not isinstance(st.session_state.full, Artifacts):
    st.session_state.full = None
    st.session_state.full_key = None
    st.info("The analysis code changed since the last run. Press Run full analysis again.")

art: Artifacts = (
    st.session_state.full
    if st.session_state.full is not None and st.session_state.full_key == key
    else art_quick
)
results = evaluate_all(art)
by_id = {r.claim_id: r for r in results}

STAMP = build_stamp(
    seed=cfg.seed,
    geometry_digest=TABLES.digest,
    geometry_retrieved=str(OC4_CITATION.retrieved),
    ljf_model=cfg.ljf_model.value,
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
verdict_block(
    "The deciding test &mdash; C3, nuisance variance budget",
    c3.detail
    if art.c3 is not None
    else "Not computed yet. Press <b>Run full analysis</b> below. C3 asks whether "
    "environmental variation alone moves the strain ratio by as much as a crack would. "
    "If it does, the other eight claims do not matter.",
    c3.status.colour,
    status=c3.status.value,
)

top = st.columns([1, 1, 1, 2])
top[0].metric("Claims supported", sum(r.status is Status.PASS for r in results))
top[1].metric("Claims refuted", sum(r.status is Status.FAIL for r in results))
top[2].metric(
    "Cannot be settled",
    sum(r.status in (Status.UNTESTABLE_DATA, Status.UNTESTABLE_PHYSICAL) for r in results),
    help="Not a failure. The data or the physical test needed to settle these does not "
    "exist here, and saying so is more useful than guessing.",
)
with top[3]:
    if st.button("Run full analysis", type="primary", width="stretch"):
        bar = st.progress(0.0, "starting")
        try:
            st.session_state.full = run_full(cfg, lambda f, m: bar.progress(min(f, 1.0), m))
            st.session_state.full_key = key
        finally:
            bar.empty()
        st.rerun()
    if st.session_state.full is not None and st.session_state.full_key != key:
        st.caption("Inputs changed since the last full run. Showing the quick pass.")

if art.c3 is None:
    st.info(
        "**Available now:** Structure (3D jacket, modal shift), Environment (tidal "
        "ellipses and forcing), Sensing (strain time series, harmonic amplitudes). "
        "**Press Run full analysis** for the Monte Carlo work: the C2 damage contour, "
        "the C3 nuisance budget and its convergence and break-even, C4 detection times, "
        "C6 filter calibration, C8 economics and C9 probability of detection.",
        icon=None,
    )

TABS = st.tabs(
    ["Overview", "Structure", "Environment", "Sensing", "Detection",
     "Assimilation", "Economics", "Ledger", "Provenance"]
)


# ----------------------------------------------------------------- Overview
with TABS[0]:
    if art.input_notes:
        with st.expander(f"Input adjustments ({len(art.input_notes)}) — nothing corrected silently",
                         expanded=True):
            for n in art.input_notes:
                st.markdown(f"- {n}")
    if art.errors:
        with st.expander(f"Computations that did not complete ({len(art.errors)})", expanded=True):
            for cid, why in sorted(art.errors.items()):
                st.markdown(f"- **{cid}** — {why}")

    section("Claims ledger", "nine claims from the abstract, each with a verdict")
    provenance_legend()
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

    j = TABLES.joints
    fig = go.Figure()
    for _mid, m in TABLES.members.iterrows():
        a, b = j.loc[int(m.joint_i)], j.loc[int(m.joint_j)]
        ps = int(m.prop_set)
        fig.add_trace(
            go.Scatter3d(
                x=[a.x_m, b.x_m], y=[a.y_m, b.y_m], z=[a.z_m, b.z_m],
                mode="lines",
                line=dict(color="#35b6c4" if ps in (2, 3, 4) else "#8a929b", width=4 if ps in (2, 3, 4) else 2),
                showlegend=False, hoverinfo="skip",
            )
        )
    sel = j.loc[cfg.joint_id]
    fig.add_trace(
        go.Scatter3d(
            x=[sel.x_m], y=[sel.y_m], z=[sel.z_m], mode="markers",
            marker=dict(size=7, color="#d1495b"), name=f"J{cfg.joint_id}",
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="x, m", yaxis_title="y, m", zaxis_title="z, m (SWL = 0)",
            aspectmode="data",
        ),
        title="OC4 jacket, target joint marked",
    )
    figure_block(fig, "oc4_geometry", "Legs in accent, braces in grey. Mudline at z = -50.001 m.", 520)

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
    section("C1 - intact tidal strain ratio", "what the two gauges read with no crack present")
    r = art.c1
    quantity(derived(r.ratio, "-", "M2 strain ratio (lower/upper)", [],
                     "harmonic fit of the frame solve", uncertainty=r.ratio_se))
    quantity(derived(r.amplitude_upper * 1e6, "ustrain", "M2 amplitude, upper gauge", [], "harmonic fit"))
    quantity(derived(r.amplitude_lower * 1e6, "ustrain", "M2 amplitude, lower gauge", [], "harmonic fit"))
    if r.below_fbg_resolution:
        unavailable_panel(
            "Signal below sensor resolution",
            "Both M2 strain amplitudes are under 1 microstrain, the resolution of a typical "
            "commercial FBG interrogator. The ratio between two unmeasurable quantities is "
            "not measurable, whatever value the solver returns for it.",
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
    if art.c2 is None or art.c3 is None:
        unavailable_panel("Not computed", "Press Run full analysis to compute C2, C3 and C4.")
    else:
        section("C2 - damage sensitivity", "how much a crack moves the strain ratio")
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

        fig = go.Figure(go.Histogram(x=r8.samples / 1e6, nbinsx=80, marker_color="#35b6c4"))
        fig.add_vline(x=0, line_color="#d1495b")
        fig.add_vline(x=19.9, line_dash="dash", line_color="#c8871b",
                      annotation_text="claimed 19.9 MUSD")
        fig.update_layout(title="NPV distribution", xaxis_title="NPV, million USD",
                          yaxis_title="count", hovermode="x")
        figure_block(fig, "c8_npv")

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
    section("Claims ledger", "the table a paper would cite, with the stamp that makes it checkable")
    for c in CLAIMS:
        r = by_id[c.id]
        cols = st.columns([1, 6, 3])
        cols[0].markdown(f"**{c.id}**")
        cols[1].markdown(c.statement)
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
    section("Data source status", "what is real here, and what is honestly missing")
    checks = [
        ("OC4 jacket geometry", True, str(OC4_CITATION)),
        ("Tide model (TPXO/FES)", *tide_model_status(cfg.tide_model_dir)),
        ("ERA5 reanalysis", *credentials_status()),
        ("Shell-FE crack-to-LJF surface", *shell_fe_status()),
        ("DNV-RP-C203 S-N curve T", *sn_status()),
        ("BS 7910 Paris constants", *paris_status()),
    ]
    for name, ok, why in checks:
        st.markdown(
            f'<span class="tt-chip" style="color:{"#2f9e5f" if ok else "#d1495b"}">'
            f'{"AVAILABLE" if ok else "UNAVAILABLE"}</span> &nbsp; **{name}**',
            unsafe_allow_html=True,
        )
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
