"""Fracture mechanics: Newman-Raju verification, crack compliance, data gating.

The Newman-Raju coefficients are transcribed from a paywalled source. These
tests verify everything about them that *can* be verified independently, so the
provenance card's claim about which parts are checked is itself checked.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ellipe

from tidetwin.damage.crack_ljf import (
    CrackGeometry,
    crack_compliance,
    load_shell_fe_surface,
    shell_fe_status,
)
from tidetwin.damage.newman_raju import (
    boundary_correction_F,
    bending_multiplier_H,
    shape_factor_Q,
    sif,
)
from tidetwin.damage.paris import load_constants, paris_status
from tidetwin.damage.sn import load_curve, sn_status
from tidetwin.provenance import DataUnavailable


def test_shape_factor_matches_the_exact_elliptic_integral():
    """Q = E(k)^2 with k^2 = 1 - (a/c)^2. Pins both the 1.464 and the 1.65.

    Newman and Raju quote a maximum error of about 0.13 percent for this
    approximation; if the coefficients here were mistranscribed, this test would
    not pass at that tolerance.
    """
    r = np.linspace(0.05, 1.0, 40)
    k2 = 1.0 - r**2
    exact = ellipe(k2) ** 2
    approx = shape_factor_Q(r)
    rel = np.abs(approx - exact) / exact
    assert rel.max() < 2.0e-3, f"worst relative error {rel.max():.2e}"

    # Endpoints, checked against closed forms.
    assert shape_factor_Q(1.0) == pytest.approx((np.pi / 2) ** 2, rel=2e-3)
    assert shape_factor_Q(1e-9) == pytest.approx(1.0, abs=1e-6)


def test_shape_factor_is_symmetric_about_unity_aspect():
    """Q(a/c) = Q(c/a): the deep-crack branch mirrors the shallow one."""
    assert shape_factor_Q(0.4) == pytest.approx(shape_factor_Q(2.5), rel=1e-12)


def test_shallow_crack_limit_recovers_the_2d_edge_crack():
    """As a/c -> 0 and a/t -> 0 the solution must tend to 1.1215 sigma sqrt(pi a).

    Newman and Raju's fit gives 1.13 here, a 0.8 percent approximation to the
    exact 2D edge-crack value.
    """
    F = float(boundary_correction_F(1e-6, 0.0, np.pi / 2))
    assert F == pytest.approx(1.13, abs=1e-3)
    assert abs(F - 1.1215) / 1.1215 < 0.01


def test_sif_grows_with_depth_and_with_stress():
    a = np.linspace(1e-3, 0.02, 30)
    K = sif(100e6, 0.0, a, 0.05, 0.05)
    assert np.all(np.diff(K) > 0)
    assert sif(200e6, 0.0, 0.01, 0.05, 0.05) == pytest.approx(
        2.0 * sif(100e6, 0.0, 0.01, 0.05, 0.05), rel=1e-12
    )


def test_deepest_point_and_surface_point_differ():
    """f_phi = (a/c)^0.5 at the surface, 1 at the deepest point."""
    deep = boundary_correction_F(0.4, 0.3, np.pi / 2)
    surf = boundary_correction_F(0.4, 0.3, 0.0)
    assert not np.isclose(deep, surf)
    assert surf > 0 and deep > 0


def test_out_of_range_aspect_returns_nan_rather_than_a_divergent_value():
    """The M3 term diverges for a/c > 1; refusing is the only safe behaviour."""
    assert np.isnan(float(boundary_correction_F(1.5, 0.5, np.pi / 2)))
    assert np.isnan(float(boundary_correction_F(2.5, 0.7, np.pi / 2)))
    assert np.isfinite(float(boundary_correction_F(1.0, 0.5, np.pi / 2)))
    # Vectorised: only the offending entries are masked.
    out = boundary_correction_F(np.array([0.5, 1.2]), 0.4, np.pi / 2)
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_bending_multiplier_is_bounded_and_finite():
    for r in (0.1, 0.5, 1.0):
        for at in (0.0, 0.3, 0.8):
            H = float(bending_multiplier_H(r, at, np.pi / 2))
            assert np.isfinite(H)
            assert -2.0 < H < 2.0


# ------------------------------------------------------------- compliance


def test_crack_compliance_is_positive_and_grows_with_crack_size():
    T = 0.05
    base, _q = crack_compliance(CrackGeometry(0.4 * T, 0.05, T), load_width=0.8)
    assert base > 0
    deeper, _ = crack_compliance(CrackGeometry(0.6 * T, 0.05, T), load_width=0.8)
    longer, _ = crack_compliance(CrackGeometry(0.4 * T, 0.10, T), load_width=0.8)
    assert deeper > base
    assert longer > base


def test_crack_compliance_vanishes_as_the_crack_does():
    T = 0.05
    tiny, _ = crack_compliance(CrackGeometry(1e-6, 0.05, T), load_width=0.8)
    assert tiny >= 0
    assert tiny < 1e-16


def test_crack_compliance_refuses_out_of_envelope_aspect():
    T = 0.05
    with pytest.raises(ValueError, match="outside the range"):
        crack_compliance(CrackGeometry(a=0.02, c=0.005, T=T), load_width=0.8)


def test_crack_compliance_carries_its_provenance_and_caveat():
    T = 0.05
    _dC, q = crack_compliance(CrackGeometry(0.5 * T, 0.05, T), load_width=0.8)
    assert q.provenance.value == "DERIVED"
    assert q.citation is not None and "Tada" in q.citation.document
    assert "under-predicts" in q.note


def test_crack_geometry_keeps_depth_and_length_independent():
    c = CrackGeometry(a=0.02, c=0.05, T=0.04)
    assert c.a_over_T == pytest.approx(0.5)
    assert c.surface_length == pytest.approx(0.10)
    assert c.aspect == pytest.approx(0.4)
    assert c.validate() == []
    assert CrackGeometry(a=0.05, c=0.05, T=0.04).validate(), "a/T > 1 must be flagged"


# ------------------------------------------------------------- data gating


def test_unshipped_data_sources_report_unavailable_with_a_remedy():
    """Paywalled constants are absent by design; the app must say so precisely."""
    for status_fn, loader, needle in (
        (shell_fe_status, load_shell_fe_surface, "shell FE"),
        (sn_status, load_curve, "DNV-RP-C203"),
        (paris_status, load_constants, "BS 7910"),
    ):
        ok, why = status_fn()
        assert not ok
        assert why.strip()
        with pytest.raises(DataUnavailable) as exc:
            loader()
        assert exc.value.remedy, f"{needle} must tell the user how to supply the data"


def test_paris_gate_warns_about_units():
    """A units slip here would be wrong by orders of magnitude, silently."""
    _ok, why = paris_status()
    assert "units" in why.lower()
