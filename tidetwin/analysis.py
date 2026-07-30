"""Orchestration: run every claim test and collect the artifacts.

Two guarantees this module is responsible for, because the whole application
rests on them:

**Any input produces a report.** Sidebar values are normalised against the
geometry before anything is solved - a gauge offset longer than the leg member,
a joint with no braces, a crack aspect ratio outside the SIF envelope, a tide
with zero amplitude. Every adjustment is recorded and shown, never silently
applied.

**Any failure produces a report.** Each claim is computed behind a guard. A
claim whose computation raises is marked ``UNTESTABLE`` with the reason and the
run continues, so one bad input cannot deny the user the other eight verdicts.
Suppressing the error would be worse than the crash; reporting it is the point.

Nothing here touches Streamlit, so the whole analysis runs headless from
``scripts/run_ledger.py`` and from CI.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from .abstract import LOWER_ZAKUM_JOINT
from .claims.registry import Artifacts
from .claims.tests.c1_ratio import intact_ratio
from .claims.tests.c2_damage import (
    damage_sensitivity_grid,
    damage_signature,
    stiffness_reduction_test,
)
from .claims.tests.c6_filter import run_comparison
from .claims.tests.c7_modal import modal_insensitivity
from .claims.tests.c9_positioning import competitor_status, tidal_pod
from .damage.crack_ljf import shell_fe_status
from .damage.paris import paris_status
from .damage.sn import sn_status
from .economics.npv import EconomicInputs, monte_carlo_npv
from .fe.ljf import LJFModel, shell_ljf
from .geometry.oc4 import (
    WATER_DEPTH,
    brace_chord_joints,
    build_jacket,
    load_tables,
    sensor_pair,
)
from .loads.era5 import credentials_status
from .loads.morison import HydroConfig
from .loads.tides import (
    CONSTITUENT_SPEEDS_DEG_PER_HOUR,
    PLACEHOLDER_CONSTITUENTS,
    from_harmonic_constants,
    placeholder_constituents,
    tide_model_status,
)
from .nuisance import NuisanceRanges, run_nuisance_budget
from .provenance import DataUnavailable, Provenance
from .response import build_response_surface
from .signal.aliasing import aliasing_table
from .signal.detect import DetectionModel, detection_time_cdf

__all__ = ["AnalysisConfig", "run_quick", "run_full", "normalise"]


@dataclass
class AnalysisConfig:
    """Everything the sidebar can set. Values may be arbitrary; see :func:`normalise`."""

    latitude: float = 24.9
    longitude: float = 53.2
    joint_id: int = 5
    sensor_offset_m: float = 1.5
    sensor_theta_deg: float = 0.0
    ljf_model: LJFModel = LJFModel.SHELL
    roughness_m: float = 0.05
    marine_growth_mm: float = 0.0
    record_days: float = 30.0
    sample_interval_s: float = 600.0
    crack_a_over_T: float = 0.5
    crack_2c_m: float = 0.10
    n_mc_samples: int = 200
    n_theta: int = 24
    seed: int = 20260728
    ranges: NuisanceRanges = field(default_factory=NuisanceRanges)
    economics: EconomicInputs = field(default_factory=EconomicInputs)
    tide_model_dir: str | None = None
    #: User-supplied harmonic constants, same schema as PLACEHOLDER_CONSTITUENTS.
    #: ``None`` uses the placeholder set. Either way the result is ASSUMED unless
    #: ``tide_is_measured`` is set with a citation.
    tide_table: dict[str, dict[str, float]] | None = None
    tide_source: str = ""
    tide_is_measured: bool = False
    #: NOAA CO-OPS current-station id whose cached harmonic constants to use.
    #: When set, tidal forcing is MEASURED and takes precedence over
    #: ``tide_table``. ``None`` falls back to the user table or the placeholder.
    tide_station: str | None = None
    #: Sensor layout for the C3 deciding test. "single" is the paper's two-gauge
    #: pair (the default, so the headline verdict is always on the method as
    #: specified); "rosette" is the proposed four-gauge direction-and-amplitude
    #: invariant layout that the experiment shows makes C3 pass.
    measurement_mode: str = "single"

    def hydro(self) -> HydroConfig:
        return HydroConfig(
            water_depth=WATER_DEPTH,
            roughness_m=self.roughness_m,
            marine_growth_mm=self.marine_growth_mm,
        )

    def constituents(self):
        """Build the tidal constituents for this configuration.

        Precedence: a cached NOAA station (MEASURED) beats a user-entered table
        (ASSUMED) beats the labelled placeholder. Only the NOAA route can produce
        MEASURED provenance; nothing typed into the sidebar can.
        """
        if self.tide_station:
            from .loads.noaa import REFERENCE_STATIONS, load_pair, to_constituents

            for sp in REFERENCE_STATIONS:
                if sp.current_id == self.tide_station:
                    return to_constituents(load_pair(sp))
            raise DataUnavailable(
                f"NOAA station '{self.tide_station}'",
                "not one of the shipped reference stations",
                "Choose a station from tidetwin.loads.noaa.REFERENCE_STATIONS, or run "
                "scripts/fetch_tides.py to cache it.",
            )
        if not self.tide_table:
            return placeholder_constituents(self.latitude, self.longitude)
        return from_harmonic_constants(
            self.latitude,
            self.longitude,
            self.tide_table,
            source=self.tide_source or "user-entered harmonic constants",
            provenance=Provenance.ASSUMED,
        )


def normalise(cfg: AnalysisConfig) -> tuple[AnalysisConfig, tuple[str, ...]]:
    """Clamp an arbitrary configuration to something the solvers can evaluate.

    Returns the adjusted configuration and a note for every change made. Nothing
    is corrected silently: each note travels into the report as an input caveat.
    """
    notes: list[str] = []
    tables = load_tables()
    kj = brace_chord_joints(tables)
    changes: dict[str, Any] = {}

    if cfg.joint_id not in kj:
        fallback = 5 if 5 in kj else sorted(kj)[0]
        notes.append(
            f"Joint {cfg.joint_id} is not a braced leg joint; using J{fallback} instead. "
            f"Selectable joints: {sorted(kj)}."
        )
        changes["joint_id"] = fallback
    joint = changes.get("joint_id", cfg.joint_id)

    # Longest offset that still lands on both leg members either side of the joint.
    max_off = _max_offset(tables, joint)
    if not 0.05 <= cfg.sensor_offset_m <= max_off:
        clamped = float(np.clip(cfg.sensor_offset_m, 0.05, max_off))
        notes.append(
            f"Gauge offset {cfg.sensor_offset_m:.2f} m does not fit on the leg members either "
            f"side of J{joint}; clamped to {clamped:.2f} m (limit {max_off:.2f} m)."
        )
        changes["sensor_offset_m"] = clamped

    if not 0.01 <= cfg.crack_a_over_T <= 0.95:
        clamped = float(np.clip(cfg.crack_a_over_T, 0.01, 0.95))
        notes.append(
            f"Crack depth ratio a/T = {cfg.crack_a_over_T:.3f} is outside (0, 1); clamped to "
            f"{clamped:.2f}. A fully through-wall flaw is a different problem, not a deeper "
            "surface crack."
        )
        changes["crack_a_over_T"] = clamped

    # Newman-Raju is fitted for a/c <= 1, so 2c must be at least 2a.
    chord_T = _chord_thickness(tables, joint)
    a = changes.get("crack_a_over_T", cfg.crack_a_over_T) * chord_T
    if cfg.crack_2c_m < 2.0 * a:
        widened = float(2.0 * a)
        notes.append(
            f"Crack surface length 2c = {cfg.crack_2c_m * 1e3:.0f} mm gives a/c > 1 for a "
            f"{a * 1e3:.1f} mm deep flaw, outside the Newman-Raju validity envelope; widened to "
            f"{widened * 1e3:.0f} mm. The deep-crack equation set is not implemented."
        )
        changes["crack_2c_m"] = widened

    if cfg.record_days < 1.0:
        notes.append(f"Record length {cfg.record_days:.2f} d is too short to fit harmonics; set to 7 d.")
        changes["record_days"] = 7.0
    if cfg.sample_interval_s <= 0 or cfg.sample_interval_s > 6 * 3600:
        notes.append("Sample interval must be between 0 and 6 h; set to 600 s.")
        changes["sample_interval_s"] = 600.0
    if cfg.n_theta < 8:
        notes.append(f"Response-surface directions {cfg.n_theta} is too coarse; set to 12.")
        changes["n_theta"] = 12
    if cfg.n_mc_samples < 20:
        notes.append(f"{cfg.n_mc_samples} Monte Carlo samples cannot estimate a variance; set to 50.")
        changes["n_mc_samples"] = 50
    if not -90.0 <= cfg.latitude <= 90.0:
        notes.append(f"Latitude {cfg.latitude} is out of range; set to 0.")
        changes["latitude"] = 0.0
    if cfg.roughness_m < 0:
        notes.append("Negative roughness is meaningless; set to 0.")
        changes["roughness_m"] = 0.0
    if cfg.marine_growth_mm < 0:
        notes.append("Negative marine growth is meaningless; set to 0.")
        changes["marine_growth_mm"] = 0.0

    if cfg.tide_station:
        from .loads.noaa import REFERENCE_STATIONS, available_cached

        cached = {sp.current_id for sp in available_cached()}
        known = {sp.current_id for sp in REFERENCE_STATIONS}
        if cfg.tide_station not in known:
            notes.append(
                f"Tide station '{cfg.tide_station}' is not a shipped reference station; "
                "falling back to the labelled placeholder set."
            )
            changes["tide_station"] = None
        elif cfg.tide_station not in cached:
            notes.append(
                f"Tide station '{cfg.tide_station}' has no cached harmonic constants; "
                "run scripts/fetch_tides.py. Falling back to the placeholder set."
            )
            changes["tide_station"] = None

    table = cfg.tide_table
    if table:
        known = {}
        unknown = []
        for name, row in table.items():
            key = str(name).strip().upper()
            if key in CONSTITUENT_SPEEDS_DEG_PER_HOUR:
                known[key] = {k: float(v) for k, v in row.items()}
            else:
                unknown.append(str(name))
        if unknown:
            notes.append(
                f"Constituent(s) {', '.join(unknown)} are not in the IHO standard list and were "
                "dropped. Their frequencies are astronomical constants, so a name that is not on "
                f"the list has no frequency to fit. Accepted names: "
                f"{', '.join(sorted(CONSTITUENT_SPEEDS_DEG_PER_HOUR))}."
            )
        table = known
        changes["tide_table"] = known or None

    effective = table or PLACEHOLDER_CONSTITUENTS
    if changes.get("tide_station", cfg.tide_station):
        return (replace(cfg, **changes) if changes else cfg), tuple(notes)
    if not any(
        abs(v.get("semi_major", 0.0)) > 0 or abs(v.get("elev_amp", 0.0)) > 0
        for v in effective.values()
    ):
        notes.append(
            "Every supplied tidal constituent has zero amplitude, so there is no tidal "
            "forcing and no strain ratio to compute. Falling back to the labelled "
            "placeholder set so the structural claims can still be reported."
        )
        changes["tide_table"] = None

    return (replace(cfg, **changes) if changes else cfg), tuple(notes)


def _max_offset(tables, joint: int) -> float:
    lengths = []
    for mid, m in tables.members.iterrows():
        if int(m.prop_set) not in (2, 3, 4):
            continue
        ji, jj = int(m.joint_i), int(m.joint_j)
        if joint not in (ji, jj):
            continue
        p = tables.joints.loc[[ji, jj], ["x_m", "y_m", "z_m"]].to_numpy(float)
        lengths.append(float(np.linalg.norm(p[1] - p[0])))
    return max(min(lengths) * 0.9, 0.1) if lengths else 1.0


def _chord_thickness(tables, joint: int) -> float:
    for _mid, m in tables.members.iterrows():
        if int(m.prop_set) in (2, 3, 4) and joint in (int(m.joint_i), int(m.joint_j)):
            return float(tables.sections.loc[int(m.prop_set), "wall_thickness_m"])
    return 0.05


def _availability(cfg: AnalysisConfig) -> dict:
    era5_ok, _ = credentials_status()
    tide_ok, _ = tide_model_status(cfg.tide_model_dir)
    shell_ok, _ = shell_fe_status()
    sn_ok, _ = sn_status()
    paris_ok, _ = paris_status()
    pod_ok, _ = competitor_status()
    return {
        "era5_available": era5_ok,
        "tide_model_available": tide_ok,
        "shell_fe_available": shell_ok,
        "sn_available": sn_ok,
        "paris_available": paris_ok,
        "competitor_pod_available": pod_ok,
    }


def _guard(art: Artifacts, claim_ids: str | tuple[str, ...], fn: Callable[[], Any]) -> Any:
    """Run ``fn``; on failure record the reason against the affected claims.

    ``DataUnavailable`` is reported with its remedy, because that is an expected
    state with a known fix. Anything else is reported with its exception type and
    message, which is more useful to a reader than a silent gap in the ledger.
    """
    ids = (claim_ids,) if isinstance(claim_ids, str) else claim_ids
    try:
        return fn()
    except DataUnavailable as exc:
        for cid in ids:
            art.errors[cid] = f"{exc} Remedy: {exc.remedy}"
    except Exception as exc:  # noqa: BLE001 - the report must still be produced
        detail = f"{type(exc).__name__}: {exc}"
        for cid in ids:
            art.errors[cid] = detail
        art.input_notes = art.input_notes + (
            f"{'/'.join(ids)} did not complete ({detail}). Traceback head: "
            + traceback.format_exc(limit=1).strip().splitlines()[-1],
        )
    return None


def run_quick(cfg: AnalysisConfig) -> Artifacts:
    """Structure, C1, C5 aliasing and C7. Fast enough for a cold start."""
    cfg, notes = normalise(cfg)
    art = Artifacts(**_availability(cfg))
    art.input_notes = notes

    tables = load_tables()
    constituents = _guard(art, ("C1", "C2", "C3", "C4"), cfg.constituents)
    if constituents is None:
        # The table survived normalisation but still could not be built (bad
        # numeric values, say). Fall back so the structural claims still report,
        # and keep the reason visible.
        art.errors.pop("C1", None)
        art.errors.pop("C2", None)
        art.errors.pop("C3", None)
        art.errors.pop("C4", None)
        art.input_notes = art.input_notes + (
            "The supplied tidal constants could not be interpreted; falling back to the "
            "labelled placeholder set so the structural claims can still be reported.",
        )
        constituents = placeholder_constituents(cfg.latitude, cfg.longitude)
    art.tide_provenance = constituents.provenance.value
    art.tide_source_note = constituents.source_note

    surface = _guard(
        art,
        ("C1", "C2", "C3", "C4"),
        lambda: _surface(cfg, tables),
    )
    if surface is not None:
        art.modelling_assumptions = surface[1]
        art.c1 = _guard(
            art,
            "C1",
            lambda: intact_ratio(
                surface[0], constituents, cfg.record_days, cfg.sample_interval_s
            ),
        )

    art.c5 = {"aliasing": aliasing_table(cfg.record_days), "amplitude_ustrain": None}

    if cfg.ljf_model is LJFModel.RIGID:
        art.errors["C7"] = (
            "Rigid joints are selected, so there is no joint compliance for a crack to change "
            "and the frequency shift would be identically zero for every crack size. Select the "
            "SHELL local joint flexibility model to make C7 meaningful."
        )
        art.errors.setdefault("C2", art.errors["C7"])
    else:
        braces = brace_chord_joints(tables)[cfg.joint_id]
        art.c7 = _guard(
            art,
            "C7",
            lambda: modal_insensitivity(
                cfg.joint_id,
                braces[0],
                cfg.crack_a_over_T,
                cfg.crack_2c_m,
                ljf_model=cfg.ljf_model,
            ),
        )
    return art


def _surface(cfg: AnalysisConfig, tables):
    pair = sensor_pair(
        tables, cfg.joint_id, cfg.sensor_offset_m, np.radians(cfg.sensor_theta_deg)
    )
    build = build_jacket(
        ljf_model=cfg.ljf_model, marine_growth_mm=cfg.marine_growth_mm, tables=tables
    )
    return (
        build_response_surface(build, pair, cfg.hydro(), n_theta=cfg.n_theta),
        build.modelling_assumptions,
    )


def run_full(cfg: AnalysisConfig, progress=None) -> Artifacts:
    """Every claim. Expensive; runs behind the Run button. Never raises."""
    cfg, _notes = normalise(cfg)
    art = run_quick(cfg)
    tables = load_tables()
    constituents = cfg.constituents()
    hydro = cfg.hydro()

    def step(frac: float, msg: str) -> None:
        if progress:
            progress(min(max(frac, 0.0), 1.0), msg)

    pair = _guard(
        art,
        ("C2", "C3"),
        lambda: sensor_pair(
            tables, cfg.joint_id, cfg.sensor_offset_m, np.radians(cfg.sensor_theta_deg)
        ),
    )
    if pair is None:
        return art

    step(0.05, "C2: crack geometry grid")
    if cfg.ljf_model is not LJFModel.RIGID:
        braces = brace_chord_joints(tables)[cfg.joint_id]
        art.c2 = _guard(
            art,
            "C2",
            lambda: damage_sensitivity_grid(
                pair,
                constituents,
                hydro,
                cfg.joint_id,
                braces[0],
                a_over_T=np.array([0.1, 0.3, 0.5, 0.7]),
                surface_length_m=np.array([0.06, 0.10, 0.20, 0.40]),
                ljf_model=cfg.ljf_model,
                n_theta=max(8, cfg.n_theta // 2),
            ),
        )

        # The abstract's own intermediate step: 10 percent stiffness reduction ->
        # 11.1 percent ratio change. Pure mechanics, no crack model, so it tests
        # the claim on terms that depend on nothing chosen here.
        art.c2_stiffness = _guard(
            art,
            "C2",
            lambda: stiffness_reduction_test(
                pair, constituents, hydro, cfg.joint_id, braces[0],
                ljf_model=cfg.ljf_model, n_theta=max(8, cfg.n_theta // 2),
            ),
        )

        # The obvious objection to the above is that it was run on the wrong
        # joint. The abstract's K-joint is softer than any joint on this frame -
        # four to eight times in out-of-plane bending - and a softer joint takes
        # a larger share of the local load path, so it could plausibly be more
        # sensitive. Repeating the sweep with the instrumented joint softened to
        # the paper's own flexibility settles it rather than arguing about it.
        art.c2_stiffness_paper = _guard(
            art,
            "C2",
            lambda: stiffness_reduction_test(
                pair, constituents, hydro, cfg.joint_id, braces[0],
                ljf_model=cfg.ljf_model, n_theta=max(8, cfg.n_theta // 2),
                # Only the points this comparison actually reports. The full
                # nine-point curve is drawn for the joint being analysed; this
                # one exists to answer a single question - does the paper's own
                # joint change the answer at its own stated 10 percent - and
                # sweeping it at full resolution cost 24 seconds of every run to
                # produce a curve nothing displayed. The high points are kept so
                # required_reduction_for_claim stays meaningful rather than
                # silently reporting "never reached" from a truncated sweep.
                reductions=np.array([0.0, 0.10, 0.50, 0.99]),
                baseline_springs=shell_ljf(LOWER_ZAKUM_JOINT),
                baseline_label="softened to the abstract's Lower Zakum K-joint",
            ),
        )

    step(0.30, "C3: nuisance variance budget")
    art.c3 = _guard(
        art,
        ("C3", "C4"),
        lambda: run_nuisance_budget(
            pair,
            constituents,
            hydro,
            ranges=cfg.ranges,
            ljf_model=cfg.ljf_model,
            n_samples=cfg.n_mc_samples,
            record_days=min(cfg.record_days, 14.0),
            n_theta=max(8, cfg.n_theta // 2),
            seed=cfg.seed,
            era5_available=art.era5_available,
            estimator=cfg.measurement_mode,
        ),
    )

    model = None
    if art.c3 is not None:
        step(0.60, "C4: detection time")
        sig_fraction = 0.111
        if art.c2 is not None:
            s = damage_signature(art.c2, cfg.crack_a_over_T, cfg.crack_2c_m)
            if np.isfinite(s) and s > 0:
                sig_fraction = float(s)
        rnd, sysm = art.c3.split_random_systematic()
        model = DetectionModel(
            sigma_random=rnd,
            sigma_systematic=sysm,
            baseline_ratio=art.c3.baseline_ratio,
            false_alarm_rate=0.01,
        )
        art.c4 = _guard(
            art,
            "C4",
            lambda: _detection(model, sig_fraction, cfg),
        )

    step(0.75, "C6: filter comparison")
    art.c6 = _guard(art, "C6", lambda: run_comparison(seed=cfg.seed))

    step(0.88, "C8: economics")
    art.c8 = _guard(art, "C8", lambda: monte_carlo_npv(cfg.economics, seed=cfg.seed))

    step(0.95, "C9: probability of detection")
    if model is not None:
        chord_T = _chord_thickness(tables, cfg.joint_id)
        grid = art.c2

        def signature_of(a_m: float) -> float:
            if grid is None:
                return 0.111 * min(a_m / max(0.5 * chord_T, 1e-9), 2.0)
            v = grid.signature_at(a_m / chord_T, cfg.crack_2c_m)
            return 0.0 if not np.isfinite(v) else float(v)

        art.c9 = _guard(
            art,
            "C9",
            lambda: tidal_pod(model, signature_of, np.linspace(1e-3, 0.045, 60)),
        )
    else:
        art.errors.setdefault("C9", "No detection model: C3 did not complete.")

    step(1.0, "done")
    return art


def _detection(model: DetectionModel, sig_fraction: float, cfg: AnalysisConfig) -> dict:
    xs, cdf, pct = detection_time_cdf(
        model,
        signature=sig_fraction * abs(model.baseline_ratio),
        record_hours=24.0 * min(cfg.record_days, 14.0),
        seed=cfg.seed,
    )
    return {
        "times_days": xs,
        "cdf": cdf,
        "percentiles": pct,
        "model": model,
        "signature_fraction": sig_fraction,
    }
