"""Provenance layer: traceability is a testable property, so it is tested."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from tidetwin.provenance import (
    Citation,
    DataUnavailable,
    Provenance,
    Quantity,
    assumed,
    derived,
    measured,
    published,
)

CIT = Citation("DNV-RP-C203", "Table 2-2", 2016)


def test_measured_and_published_require_citation():
    with pytest.raises(ValueError):
        Quantity(1.0, "m", Provenance.MEASURED, "x")
    with pytest.raises(ValueError):
        Quantity(1.0, "m", Provenance.PUBLISHED, "x")
    # DERIVED and ASSUMED do not.
    assert derived(1.0, "m", "x", [], "op").provenance is Provenance.DERIVED
    assert assumed(1.0, "m", "x").provenance is Provenance.ASSUMED


def test_empty_units_rejected():
    with pytest.raises(ValueError):
        Quantity(1.0, "", Provenance.DERIVED, "x")


def test_arithmetic_builds_the_input_chain():
    a = published(3.0, "m", "a", CIT)
    b = measured(4.0, "m", "b", Citation("ERA5", variable="swh", retrieved=dt.date(2026, 1, 1)))
    c = a + b
    assert c.value == 7.0
    assert c.provenance is Provenance.DERIVED
    names = {q.name for q in c.chain()}
    assert {"a", "b"} <= names
    assert len(c.sources()) == 2


def test_assumption_contaminates_everything_downstream():
    a = published(2.0, "m", "clean", CIT)
    u = assumed(5.0, "m", "user slider")
    assert not a.contaminated
    result = (a + u) * 3.0
    assert result.contaminated
    assert result.blocking_assumptions == ["user slider"]


def test_unit_algebra():
    force = derived(10.0, "N", "F", [], "op")
    area = derived(2.0, "m^2", "A", [], "op")
    stress = force / area
    assert stress.units == "N/m^2"
    assert (force * area).units == "N.m^2"
    ratio = force / force
    assert ratio.units == "-"


def test_unit_mismatch_on_addition_is_an_error():
    a = derived(1.0, "m", "a", [], "op")
    b = derived(1.0, "s", "b", [], "op")
    with pytest.raises(ValueError):
        _ = a + b


def test_uncertainty_propagation_first_order():
    a = published(10.0, "m", "a", CIT, uncertainty=0.3)
    b = published(4.0, "m", "b", CIT, uncertainty=0.4)
    s = a + b
    assert s.uncertainty == pytest.approx(np.hypot(0.3, 0.4))
    p = a * b
    # d(ab) = b*da (+) a*db
    assert p.uncertainty == pytest.approx(np.hypot(4.0 * 0.3, 10.0 * 0.4))
    q = a / b
    assert q.uncertainty == pytest.approx(
        np.hypot(0.3 / 4.0, 10.0 * 0.4 / 16.0)
    )


def test_arrays_carry_provenance():
    arr = measured(np.arange(5.0), "m", "series", Citation("TPXO9", variable="h"))
    doubled = arr * 2.0
    assert doubled.is_array
    assert np.allclose(np.asarray(doubled), np.arange(5.0) * 2.0)
    assert "series" in {q.name for q in doubled.chain()}
    with pytest.raises(TypeError):
        float(doubled)


def test_chain_is_deduplicated_on_a_diamond():
    root = published(2.0, "m", "root", CIT)
    left = root * 3.0
    right = root * 5.0
    total = left + right
    roots = [q for q in total.chain() if q.name == "root"]
    assert len(roots) == 1


def test_format_includes_units_and_uncertainty():
    txt = published(1234.5678, "MPa", "sigma", CIT, uncertainty=12.0).format()
    assert txt.startswith("123") and txt.endswith("MPa") and "+/-" in txt
    # Dimensionless quantities render without a trailing unit token.
    assert derived(0.5, "-", "ratio", [], "op").format() == "0.5000"
    # Counts render as integers; "64.00 joints" would be false precision.
    assert derived(64.0, "-", "joints", [], "op").format() == "64"
    assert derived(112.0, "-", "members", [], "op").format() == "112"
    assert derived(-3.0, "m", "offset", [], "op").format() == "-3 m"
    # A value with a stated uncertainty keeps its significant figures.
    assert derived(64.0, "-", "n", [], "op", uncertainty=2.0).format() != "64"


def test_data_unavailable_carries_a_remedy():
    err = DataUnavailable("CDS API", "no credentials", "set CDSAPI_KEY in st.secrets")
    assert "DATA UNAVAILABLE" in str(err)
    assert err.remedy


def test_to_dict_round_trips_the_essentials():
    q = measured(
        3.5, "m", "swh", Citation("ERA5", variable="swh", retrieved=dt.date(2026, 2, 3))
    )
    d = q.to_dict(include_chain=True)
    assert d["provenance"] == "MEASURED"
    assert d["citation"]["variable"] == "swh"
    assert d["citation"]["retrieved"] == "2026-02-03"
    assert d["contaminated"] is False


def test_a_quantity_refuses_a_string_value_at_the_call_site():
    """A yes/no is not a measurement, and the error must point at the caller.

    Passing a string used to be accepted and then failed deep inside format(),
    so the traceback blamed the renderer rather than the call that made the
    mistake - which is how it reached a rendered page.
    """
    with pytest.raises(TypeError, match="numeric"):
        assumed("yes", "-", "does the criterion discriminate?")
    with pytest.raises(TypeError):
        derived("n/a", "-", "some label", [], "some operation")


def test_arrays_are_still_accepted():
    """Time series and sweeps carry provenance too; only strings are refused."""
    q = assumed(np.linspace(0.0, 1.0, 5), "m", "a swept parameter")
    assert q.value.shape == (5,)
