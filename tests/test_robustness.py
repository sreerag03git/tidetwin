"""Any input must produce a complete report.

An adversarial test bench that falls over when handed unusual inputs cannot be
trusted to report honestly on ordinary ones. These tests feed the pipeline
nonsense on purpose and require that it still returns all nine claims, with the
bad inputs reported rather than silently corrected.
"""

from __future__ import annotations

import numpy as np
import pytest

from tidetwin.analysis import AnalysisConfig, normalise, run_full, run_quick
from tidetwin.claims.ledger import build_stamp, markdown_summary, to_csv, to_latex
from tidetwin.claims.registry import CLAIMS, Artifacts, Status, evaluate_all
from tidetwin.fe.ljf import LJFModel
from tidetwin.geometry.oc4 import OC4_CITATION, load_tables
from tidetwin.report import ReportInputs, to_html, to_markdown, to_text

FAST = dict(n_mc_samples=20, n_theta=8, record_days=7.0, sample_interval_s=3600.0)


def _stamp():
    t = load_tables()
    return build_stamp(
        seed=1,
        geometry_digest=t.digest,
        geometry_retrieved=str(OC4_CITATION.retrieved),
        ljf_model="SHELL",
    )


def _report_is_complete(results, art) -> None:
    """Every claim appears, in every format, with a status."""
    assert len(results) == len(CLAIMS) == 9
    ids = [r.claim_id for r in results]
    assert ids == [c.id for c in CLAIMS]
    for r in results:
        assert isinstance(r.status, Status)
        assert r.detail.strip(), f"{r.claim_id} has an empty detail"
        assert r.computed_text.strip()

    stamp = _stamp()
    md = to_markdown(results, art, stamp, ReportInputs.from_config(AnalysisConfig()))
    html = to_html(results, art, stamp)
    txt = to_text(results, art, stamp)
    for doc in (md, html, txt):
        assert doc.strip()
        for c in CLAIMS:
            assert c.id in doc, f"{c.id} missing from a report format"
    assert to_csv(results, stamp).count("\n") > 9
    assert r"\begin{tabular}" in to_latex(results, stamp)


# ------------------------------------------------------------- normalisation


@pytest.mark.parametrize(
    "field,value,expect_note",
    [
        ("joint_id", 99999, "not a braced leg joint"),
        ("joint_id", -1, "not a braced leg joint"),
        ("sensor_offset_m", 1e6, "does not fit"),
        ("sensor_offset_m", -5.0, "does not fit"),
        ("crack_a_over_T", 5.0, "outside"),
        ("crack_a_over_T", -0.2, "outside"),
        ("crack_2c_m", 1e-6, "a/c > 1"),
        ("record_days", 0.0, "too short"),
        ("sample_interval_s", -10.0, "Sample interval"),
        ("n_theta", 1, "too coarse"),
        ("n_mc_samples", 2, "cannot estimate a variance"),
        ("latitude", 500.0, "out of range"),
        ("roughness_m", -1.0, "Negative roughness"),
        ("marine_growth_mm", -50.0, "Negative marine growth"),
    ],
)
def test_hostile_scalar_inputs_are_clamped_and_reported(field, value, expect_note):
    cfg = AnalysisConfig(**{**FAST, field: value})
    fixed, notes = normalise(cfg)
    assert notes, f"{field}={value} should have produced a note"
    assert any(expect_note in n for n in notes), f"expected '{expect_note}' in {notes}"
    # The adjusted configuration must itself be clean.
    _again, notes2 = normalise(fixed)
    assert not any(expect_note in n for n in notes2), "normalise is not idempotent"


def test_normalise_leaves_good_inputs_alone():
    cfg = AnalysisConfig(**FAST)
    fixed, notes = normalise(cfg)
    assert notes == ()
    assert fixed.joint_id == cfg.joint_id
    assert fixed.crack_2c_m == cfg.crack_2c_m


def test_crack_geometry_is_widened_rather_than_extrapolated():
    """a/c > 1 must be fixed by widening 2c, never by evaluating outside the SIF envelope."""
    cfg = AnalysisConfig(crack_a_over_T=0.9, crack_2c_m=0.005, **FAST)
    fixed, notes = normalise(cfg)
    T = 0.05  # joint 5 lower leg
    assert fixed.crack_2c_m >= 2.0 * fixed.crack_a_over_T * T - 1e-12
    assert any("Newman-Raju" in n for n in notes)


def test_zero_amplitude_tide_falls_back_and_says_so():
    dead = {"M2": {k: 0.0 for k in
                   ("elev_amp", "elev_phase_deg", "semi_major", "semi_minor",
                    "inclination_deg", "current_phase_deg")}}
    cfg = AnalysisConfig(tide_table=dead, **FAST)
    fixed, notes = normalise(cfg)
    assert fixed.tide_table is None
    assert any("zero amplitude" in n for n in notes)


# ------------------------------------------------------------- full pipeline


def test_quick_run_with_hostile_inputs_still_reports_every_claim():
    cfg = AnalysisConfig(
        joint_id=12345, sensor_offset_m=999.0, crack_a_over_T=3.0, crack_2c_m=1e-9,
        latitude=1e4, roughness_m=-2.0, record_days=0.1, **{k: v for k, v in FAST.items()
                                                            if k not in ("record_days",)},
    )
    art = run_quick(cfg)
    results = evaluate_all(art)
    _report_is_complete(results, art)
    assert art.input_notes, "hostile inputs must be reported"


def test_rigid_joints_make_c2_and_c7_untestable_not_zero():
    """With rigid joints there is no compliance for a crack to change."""
    cfg = AnalysisConfig(ljf_model=LJFModel.RIGID, **FAST)
    art = run_quick(cfg)
    results = {r.claim_id: r for r in evaluate_all(art)}
    for cid in ("C2", "C7"):
        assert results[cid].status is Status.UNTESTABLE_DATA
        assert "rigid" in results[cid].detail.lower()
    _report_is_complete(list(results.values()), art)


def test_a_failing_claim_does_not_deny_the_others():
    """One broken computation must not take down the whole ledger."""
    art = run_quick(AnalysisConfig(**FAST))
    art.errors["C1"] = "deliberate failure injected by the test"
    results = evaluate_all(art)
    by_id = {r.claim_id: r for r in results}
    assert by_id["C1"].status is Status.UNTESTABLE_DATA
    assert "deliberate failure" in by_id["C1"].detail
    # Every other claim still got a verdict.
    assert all(r.status is not None for r in results)
    _report_is_complete(results, art)


def test_a_raising_test_function_is_reported_not_propagated():
    art = Artifacts()
    art.c3 = object()  # will make the C3 test raise on attribute access
    results = evaluate_all(art)
    c3 = next(r for r in results if r.claim_id == "C3")
    assert c3.status is Status.UNTESTABLE_DATA
    assert "raised" in c3.detail
    _report_is_complete(results, art)


def test_empty_artifacts_still_produce_a_full_report():
    """Nothing computed at all is still a valid, complete report."""
    art = Artifacts()
    results = evaluate_all(art)
    _report_is_complete(results, art)
    # "Not run" is not a verdict. Reporting these as UNTESTABLE would claim the
    # data to settle them is missing, which is a different and much stronger
    # statement than "the analysis has not been run".
    assert all(r.status is Status.NOT_RUN for r in results)
    assert not any(r.status.is_verdict for r in results)


@pytest.mark.parametrize("joint", [3, 5, 21, 23])
def test_every_selectable_joint_runs(joint):
    art = run_quick(AnalysisConfig(joint_id=joint, sensor_offset_m=0.8, **FAST))
    assert "C1" not in art.errors, art.errors.get("C1")
    assert art.c1 is not None
    assert np.isfinite(art.c1.ratio)


def test_user_supplied_tide_table_is_used_and_stays_assumed():
    table = {
        "M2": {"elev_amp": 1.2, "elev_phase_deg": 10.0, "semi_major": 0.4,
               "semi_minor": 0.1, "inclination_deg": 70.0, "current_phase_deg": 20.0},
        "K1": {"elev_amp": 0.3, "elev_phase_deg": 200.0, "semi_major": 0.1,
               "semi_minor": -0.03, "inclination_deg": 50.0, "current_phase_deg": 15.0},
    }
    cfg = AnalysisConfig(tide_table=table, tide_source="unit test", **FAST)
    art = run_quick(cfg)
    assert art.tide_provenance == "ASSUMED"
    assert art.c1 is not None and np.isfinite(art.c1.ratio)
    results = evaluate_all(art)
    c1 = next(r for r in results if r.claim_id == "C1")
    assert any("ASSUMED" in b for b in c1.blocking_assumptions)


def test_unknown_constituent_name_is_dropped_and_reported():
    """A name with no astronomical frequency has nothing to fit, so it is dropped."""
    cfg = AnalysisConfig(
        tide_table={
            "NOTATIDE": {"elev_amp": 1.0, "semi_major": 0.2},
            "M2": {"elev_amp": 0.6, "semi_major": 0.3, "elev_phase_deg": 0.0,
                   "semi_minor": 0.05, "inclination_deg": 30.0, "current_phase_deg": 60.0},
        },
        **FAST,
    )
    fixed, notes = normalise(cfg)
    assert set(fixed.tide_table) == {"M2"}
    assert any("NOTATIDE" in n for n in notes)
    art = run_quick(cfg)
    results = evaluate_all(art)
    _report_is_complete(results, art)
    assert art.c1 is not None, "the remaining valid constituent must still be used"


def test_all_unknown_constituents_falls_back_to_placeholder():
    cfg = AnalysisConfig(tide_table={"NOTATIDE": {"elev_amp": 1.0}}, **FAST)
    art = run_quick(cfg)
    results = evaluate_all(art)
    _report_is_complete(results, art)
    assert any("NOTATIDE" in n for n in art.input_notes)
    assert art.c1 is not None


@pytest.mark.slow
def test_full_run_completes_and_reports_everything():
    art = run_full(AnalysisConfig(**FAST))
    results = evaluate_all(art)
    _report_is_complete(results, art)
    # C3 is the deciding test and must always carry a verdict.
    c3 = next(r for r in results if r.claim_id == "C3")
    assert c3.status.is_verdict, "a full run must reach an actual verdict on C3"
    assert c3.status in (Status.PASS, Status.MARGINAL, Status.FAIL, Status.UNTESTABLE_DATA)


def test_not_run_is_never_presented_as_a_verdict():
    """The distinction that matters most to a reader of the ledger.

    "We have not run this yet" and "the data needed to settle this does not
    exist" are completely different statements. Conflating them made every claim
    read as UNTESTABLE on arrival, which misrepresents the ledger.
    """
    art = Artifacts()
    for r in evaluate_all(art):
        assert r.status is Status.NOT_RUN
        assert not r.status.is_verdict
        assert "UNTESTABLE" not in r.status.value
        assert "not been run" in r.detail

    stamp = _stamp()
    for doc in (to_markdown(results := evaluate_all(art), art, stamp),
                to_html(results, art, stamp),
                to_text(results, art, stamp)):
        assert "NOT RUN YET" in doc
        # A report of an unfinished analysis must say so up front.
        assert "had not been computed" in doc


def test_a_completed_claim_is_never_marked_not_run():
    art = run_quick(AnalysisConfig(**FAST))
    by_id = {r.claim_id: r for r in evaluate_all(art)}
    # C1 and C7 are computed by the quick pass, so they must carry real verdicts.
    assert by_id["C1"].status.is_verdict
    assert by_id["C7"].status.is_verdict


def test_the_readme_table_does_not_change_when_only_the_clock_does():
    """The committed claims table must depend on results, nothing else.

    It used to carry the commit hash and the generation time. Both are wrong
    there: the block lives inside the commit it names, so the hash can only ever
    be the previous one, and both fields move on every run even when no result
    does. CI regenerated and committed the table on every push as a result, and
    each of those commits collided with the next local regeneration.
    """
    import time

    art = Artifacts()
    results = evaluate_all(art)
    first = markdown_summary(results, _stamp())
    time.sleep(1.1)
    second = markdown_summary(results, _stamp())
    assert first == second

    assert "generated" not in first
    assert "commit" not in first
    # The determinants of the numbers must still be recorded.
    for needle in ("seed", "LJF", "geometry"):
        assert needle in first, f"the stamp must still record the {needle}"
