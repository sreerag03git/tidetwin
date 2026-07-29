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
    axial_ratio,
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
