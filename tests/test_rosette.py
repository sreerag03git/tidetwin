"""The direction-invariant rosette: does the algebra do what it claims?

These are analytical checks on synthetic strain fields, not on the jacket. If
the decomposition is not exact on a field we constructed ourselves, nothing it
says about the structure is worth reading.
"""

from __future__ import annotations

import numpy as np
import pytest

from tidetwin.loads.tides import constituent_frequency
from tidetwin.rosette import (
    ROSETTE_ANGLES_DEG,
    axial_drag_ratio,
    axial_ratio,
    drag_component,
    bending_ratio,
    decompose,
    m2_phasor,
)

OM = float(constituent_frequency("M2").value)
T = np.arange(0.0, 14 * 86400.0, 1800.0)


def field(axial, bx, by, phase=0.0):
    """eps(phi) = a + bx cos(phi) + by sin(phi), all oscillating at M2."""
    carrier = np.cos(OM * T - phase)
    return [
        (axial + bx * np.cos(np.radians(p)) + by * np.sin(np.radians(p))) * carrier
        for p in ROSETTE_ANGLES_DEG
    ]


def test_the_decomposition_is_exact_on_a_field_we_built():
    d = decompose(T, field(axial=1.0e-6, bx=0.3e-6, by=-0.4e-6))
    assert d.axial == pytest.approx(1.0e-6, rel=1e-6)
    assert d.bending == pytest.approx(0.5e-6, rel=1e-6)  # hypot(0.3, 0.4)


def test_pure_axial_loading_shows_no_bending():
    d = decompose(T, field(axial=2.0e-6, bx=0.0, by=0.0))
    assert d.bending == pytest.approx(0.0, abs=1e-15)
    assert d.axial == pytest.approx(2.0e-6, rel=1e-6)


def test_pure_bending_shows_no_axial():
    d = decompose(T, field(axial=0.0, bx=1.0e-6, by=0.0))
    assert d.axial == pytest.approx(0.0, abs=1e-15)
    assert d.bending == pytest.approx(1.0e-6, rel=1e-6)


@pytest.mark.parametrize("load_deg", [0.0, 17.0, 45.0, 90.0, 133.0, 270.0, 355.0])
def test_both_invariants_are_invariant_to_load_direction(load_deg):
    """The whole point. Rotate the load; the invariants must not move.

    This is what a single gauge pair fails to do, and why the direction of the
    rotary tidal current is C3's dominant nuisance channel.
    """
    mag_a, mag_b = 1.0e-6, 0.5e-6
    r = np.radians(load_deg)
    d = decompose(T, field(axial=mag_a, bx=mag_b * np.cos(r), by=mag_b * np.sin(r)))
    assert d.axial == pytest.approx(mag_a, rel=1e-6)
    assert d.bending == pytest.approx(mag_b, rel=1e-6)


def test_a_single_gauge_is_not_invariant_to_load_direction():
    """The control. Without this the test above proves nothing about the fix."""
    mag_a, mag_b = 1.0e-6, 0.5e-6
    seen = []
    for load_deg in (0.0, 90.0, 180.0):
        r = np.radians(load_deg)
        e0 = field(axial=mag_a, bx=mag_b * np.cos(r), by=mag_b * np.sin(r))[0]
        seen.append(abs(m2_phasor(T, e0)))
    assert max(seen) - min(seen) > 0.5 * mag_b, "a single gauge should swing with direction"


def test_combining_in_the_time_domain_would_destroy_the_constituent():
    """Why the combination is done on phasors and not on the raw series.

    hypot() of the bending components rectifies them: they cross zero twice a
    cycle, so the magnitude runs at 2*M2 and the M2 line the ratio is fitted on
    largely disappears. Doing this made the rosette look four times worse than a
    single pair instead of better.
    """
    e = field(axial=0.0, bx=1.0e-6, by=0.0)
    bx = 0.5 * (e[0] - e[2])
    by = 0.5 * (e[1] - e[3])
    rectified = np.hypot(bx, by)
    assert abs(m2_phasor(T, rectified)) < 0.2 * abs(m2_phasor(T, bx)), (
        "rectifying must lose most of the M2 line - that is the trap being documented"
    )


def test_the_ratio_recovers_a_known_gradient():
    up = field(axial=1.0e-6, bx=0.2e-6, by=0.0)
    lo = field(axial=1.8e-6, bx=0.6e-6, by=0.0)
    assert axial_ratio(T, up, lo) == pytest.approx(1.8, rel=1e-5)
    assert bending_ratio(T, up, lo) == pytest.approx(3.0, rel=1e-5)


def test_a_dead_upper_section_gives_nan_not_a_division_by_zero():
    dead = [np.zeros_like(T) for _ in ROSETTE_ANGLES_DEG]
    assert np.isnan(axial_ratio(T, dead, field(1e-6, 0, 0)))


def test_the_wrong_number_of_gauges_is_refused_with_a_reason():
    with pytest.raises(ValueError, match="exactly four"):
        decompose(T, field(1e-6, 0, 0)[:3])


def test_bending_fraction_flags_a_section_with_no_usable_bending():
    """The measured situation at J5: bending is a few percent of axial."""
    d = decompose(T, field(axial=1.0e-6, bx=0.02e-6, by=0.0))
    assert d.bending_fraction < 0.1


# ---------------------------------------------------------------------------
# The second correction: projecting onto the drag (quadrature) component, which
# is what removes the current's AMPLITUDE from the ratio.


def tide(amp=1.0, phase=0.0):
    return amp * np.cos(OM * T - phase)


def test_drag_and_buoyancy_are_separated_by_their_phase():
    """Drag leads the tide by a quarter cycle; buoyancy is in step with it."""
    eta = tide()
    assert drag_component(T, tide(2.0, np.pi / 2), eta) == pytest.approx(2.0, rel=1e-6)
    assert drag_component(T, tide(2.0, 0.0), eta) == pytest.approx(0.0, abs=1e-9)


def test_a_common_current_scaling_cancels_in_a_drag_only_ratio():
    """The mechanism the fix relies on.

    Morison drag goes as U^2, so scaling the current scales every drag strain by
    one common factor, which cancels in a ratio. Spring/neap only moves the plain
    ratio because buoyancy rides along and does not scale.
    """
    eta = tide()
    for scale in (0.5, 1.0, 2.7):
        up = [tide(1.0 * scale, np.pi / 2) for _ in ROSETTE_ANGLES_DEG]
        lo = [tide(1.8 * scale, np.pi / 2) for _ in ROSETTE_ANGLES_DEG]
        assert axial_drag_ratio(T, up, lo, eta) == pytest.approx(1.8, rel=1e-6)


def test_buoyancy_contamination_is_what_breaks_that_invariance():
    """Control: with the buoyancy part left in, the ratio moves with the scaling."""
    eta = tide()
    seen = []
    for scale in (0.5, 2.0):
        up = [tide(1.0 * scale, np.pi / 2) + tide(0.5, 0.0) for _ in ROSETTE_ANGLES_DEG]
        lo = [tide(1.8 * scale, np.pi / 2) + tide(0.2, 0.0) for _ in ROSETTE_ANGLES_DEG]
        seen.append(axial_ratio(T, up, lo))
    assert abs(seen[0] - seen[1]) > 0.1, "buoyancy must make the plain ratio scale-dependent"


def test_the_combined_estimator_is_invariant_to_direction_and_amplitude_at_once():
    """Rotate the load and rescale it; the ratio must hold."""
    eta = tide()
    ref = None
    for load_deg, scale in ((0.0, 1.0), (61.0, 0.4), (200.0, 2.2), (315.0, 1.7)):
        r = np.radians(load_deg)
        def sec(a, b):
            return [(a + b * np.cos(np.radians(p) - r)) * scale * np.cos(OM * T - np.pi / 2)
                    + 0.3 * np.cos(OM * T)
                    for p in ROSETTE_ANGLES_DEG]
        got = axial_drag_ratio(T, sec(1.0, 0.4), sec(1.8, 0.9), eta)
        if ref is None:
            ref = got
        assert got == pytest.approx(ref, rel=1e-6)


def test_a_flat_tide_reference_gives_nan_rather_than_a_phase_from_nowhere():
    flat = np.zeros_like(T)
    assert np.isnan(drag_component(T, tide(1.0), flat))
    up = [tide(1.0, np.pi / 2) for _ in ROSETTE_ANGLES_DEG]
    assert np.isnan(axial_drag_ratio(T, up, up, flat))
