"""Real tidal constants from NOAA CO-OPS.

The unit and convention conversions here are ours, not NOAA's, so they are what
gets tested: cm/s to m/s, and an azimuth measured clockwise from true north into
an inclination measured anticlockwise from east. Getting that rotation wrong
would silently point every tidal current in the wrong direction, which C3 is
highly sensitive to.

These tests run against the cached extractions in ``data/constituents/`` and
make no network calls.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tidetwin.analysis import AnalysisConfig, normalise, run_quick
from tidetwin.claims.registry import evaluate_all
from tidetwin.loads.noaa import (
    REFERENCE_STATIONS,
    available_cached,
    load_pair,
    to_constituents,
)
from tidetwin.provenance import DataUnavailable, Provenance

CACHED = available_cached()
needs_cache = pytest.mark.skipif(not CACHED, reason="run scripts/fetch_tides.py first")


@needs_cache
def test_every_reference_station_is_cached():
    """The repository ships the extractions so CI needs no network."""
    assert len(CACHED) == len(REFERENCE_STATIONS), (
        f"only {len(CACHED)} of {len(REFERENCE_STATIONS)} cached; "
        "run scripts/fetch_tides.py"
    )


@needs_cache
@pytest.mark.parametrize("pair", CACHED, ids=lambda p: p.current_id)
def test_cached_payload_records_its_provenance(pair):
    p = load_pair(pair)
    assert p["retrieved"], "a retrieval date is required for MEASURED data"
    assert "NOAA" in p["source"]
    assert p["current_station"]["id"] == pair.current_id
    assert p["water_level_station"]["id"] == pair.water_level_id
    assert p["separation_km"] >= 0
    for name, c in p["constituents"].items():
        assert c["elev_amp_m"] >= 0
        assert c["major_amp_cm_s"] != 0 or name not in ("M2",)
        assert 0.0 <= c["azimuth_deg_true"] <= 360.0


@needs_cache
@pytest.mark.parametrize("pair", CACHED, ids=lambda p: p.current_id)
def test_constituents_are_measured_and_cited(pair):
    con = to_constituents(load_pair(pair))
    assert con.provenance is Provenance.MEASURED
    assert con.citation is not None
    assert con.citation.retrieved is not None, "MEASURED data needs a retrieval date"
    assert pair.current_id in con.citation.locator
    assert set(("M2", "S2", "N2", "K1", "O1")) <= set(con.names)


@needs_cache
@pytest.mark.parametrize("pair", CACHED, ids=lambda p: p.current_id)
def test_unit_conversion_from_cm_per_second(pair):
    raw = load_pair(pair)
    con = to_constituents(raw)
    for i, name in enumerate(con.names):
        assert con.semi_major[i] == pytest.approx(
            raw["constituents"][name]["major_amp_cm_s"] / 100.0, rel=1e-12
        )
        assert con.semi_minor[i] == pytest.approx(
            raw["constituents"][name]["minor_amp_cm_s"] / 100.0, rel=1e-12
        )
    # Real tidal currents are metres per second, not tens of them.
    assert 0.0 < abs(con.semi_major).max() < 5.0
    assert abs(con.semi_minor).max() <= abs(con.semi_major).max()


@needs_cache
@pytest.mark.parametrize("pair", CACHED, ids=lambda p: p.current_id)
def test_azimuth_converts_to_the_right_inclination(pair):
    """NOAA azi is clockwise from north; inclination is anticlockwise from east."""
    raw = load_pair(pair)
    con = to_constituents(raw)
    for i, name in enumerate(con.names):
        azi = raw["constituents"][name]["azimuth_deg_true"]
        expected = np.radians(90.0 - azi)
        assert con.inclination[i] == pytest.approx(expected, rel=1e-12)

    # The reconstructed current must actually point along the reported azimuth.
    i = con.index("M2")
    single = con.subset(("M2",))
    t = np.linspace(0.0, 12.42 * 3600.0, 400)
    uv = single.depth_averaged_current(t)
    k = int(np.argmax(np.hypot(uv[:, 0], uv[:, 1])))
    # Bearing clockwise from north of the strongest flow.
    bearing = np.degrees(np.arctan2(uv[k, 0], uv[k, 1])) % 180.0
    azi = raw["constituents"]["M2"]["azimuth_deg_true"] % 180.0
    assert min(abs(bearing - azi), 180 - abs(bearing - azi)) < 12.0


@needs_cache
def test_the_stations_span_genuinely_different_tidal_regimes():
    """A sweep that only covered one regime would settle nothing."""
    forms, eccs, majors = [], [], []
    for pair in CACHED:
        con = to_constituents(load_pair(pair))
        i = con.index("M2")
        forms.append(con.form_factor())
        majors.append(float(con.semi_major[i]))
        eccs.append(abs(float(con.semi_minor[i]) / float(con.semi_major[i])))
    # Semidiurnal (F < 0.25) through to mixed mainly diurnal (F > 1.5).
    assert min(forms) < 0.25 and max(forms) > 1.5
    # Near-rectilinear through to strongly rotary.
    assert min(eccs) < 0.10 and max(eccs) > 0.25
    # Nearly an order of magnitude in current amplitude.
    assert max(majors) / min(majors) > 4.0


@needs_cache
def test_selecting_a_station_gives_measured_forcing_end_to_end():
    cfg = AnalysisConfig(
        tide_station=CACHED[0].current_id,
        n_mc_samples=20, n_theta=8, record_days=7.0, sample_interval_s=3600.0,
    )
    art = run_quick(cfg)
    assert art.tide_provenance == "MEASURED"
    assert art.c1 is not None and np.isfinite(art.c1.ratio)
    # The contamination note must still flag that it is not the platform site.
    c1 = next(r for r in evaluate_all(art) if r.claim_id == "C1")
    assert any("not the platform site" in b or "published station" in b
               for b in c1.blocking_assumptions)


def test_unknown_station_falls_back_and_says_so():
    cfg = AnalysisConfig(tide_station="NOT-A-STATION", n_mc_samples=20, n_theta=8)
    fixed, notes = normalise(cfg)
    assert fixed.tide_station is None
    assert any("not a shipped reference station" in n for n in notes)


def test_uncached_station_reports_how_to_get_it():
    from tidetwin.loads.noaa import StationPair

    ghost = StationPair("nowhere", "ZZZ9999", "0000000")
    with pytest.raises(DataUnavailable) as exc:
        load_pair(ghost)
    assert "fetch_tides" in exc.value.remedy


@needs_cache
def test_cached_json_is_valid_and_committed_as_text():
    """The cache is committed, so it must be readable, diffable JSON."""
    for pair in CACHED:
        raw = pair.cache_path.read_text(encoding="utf-8")
        json.loads(raw)
        assert raw.endswith("\n")
        assert "public domain" in raw
