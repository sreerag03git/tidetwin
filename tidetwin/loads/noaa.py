"""Real tidal harmonic constants from the NOAA CO-OPS metadata API.

This is the route to **MEASURED** tidal forcing that does not require a
registered download. NOAA publishes, openly and without an API key:

* water-level harmonic constants - amplitude (m) and Greenwich phase (deg);
* tidal *current* harmonic constants - the ellipse parameters
  ``majorAmplitude`` and ``minorAmplitude`` (cm/s), the major-axis azimuth
  ``azi`` (deg true), and Greenwich phases.

The current ellipse is exactly the parameterisation
:class:`~tidetwin.loads.tides.TidalConstituents` uses, so these drop straight in
with provenance MEASURED and a retrieval date.

**What this does and does not settle.** These are US stations. They are *not*
the Arabian Gulf platform site, and this module does not pretend otherwise - the
citation names the station. What they do settle is the question that actually
matters for C3: whether the verdict is an artefact of the placeholder tide.
Running the nuisance budget across real stations spanning semidiurnal to mixed
regimes, with current amplitudes from 0.1 to 0.7 m/s and ellipse eccentricities
from rectilinear to strongly rotary, tests that directly. For the platform site
itself a TPXO or FES extraction is still needed; see
:func:`tidetwin.loads.tides.from_pytmd`.

Two approximations are worth stating, because they are ours and not NOAA's:

1. The current constants are measured in a bin at a stated depth, and are used
   here as the depth-averaged current that drives the Morison profile. For a
   near-surface bin this overstates the depth average slightly.
2. The current station and the water-level station are different locations, tens
   of kilometres apart in some pairs. Their relative phase is therefore not
   exact, which matters for the in-phase/quadrature split but not for the
   amplitudes. The separation is recorded in the cache and reported.

Data are a work of the US Government and in the public domain.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..provenance import Citation, DataUnavailable, Provenance
from .tides import CONSTITUENT_SPEEDS_DEG_PER_HOUR, TidalConstituents

__all__ = [
    "NOAA_CITATION",
    "CACHE_DIR",
    "StationPair",
    "REFERENCE_STATIONS",
    "fetch_pair",
    "load_pair",
    "to_constituents",
    "available_cached",
]

BASE = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "constituents"

NOAA_CITATION = Citation(
    document=(
        "NOAA CO-OPS published harmonic constants, Center for Operational Oceanographic "
        "Products and Services metadata API (mdapi). Water levels: harmonic constants "
        "from the accepted datum analysis. Currents: tidal current ellipse constants "
        "(major/minor amplitude, major-axis azimuth, Greenwich phase)"
    ),
    url="https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi",
)

#: Station pairs chosen to span real tidal regimes, not to flatter the method.
#: Eccentricity is |minor/major| for M2: 0 is a reversing channel current, 1 is
#: a circular rotary current.
REFERENCE_STATIONS: tuple["StationPair", ...] = ()


@dataclass(frozen=True)
class StationPair:
    """A current station and the water-level gauge used for its elevation."""

    label: str
    current_id: str
    water_level_id: str
    note: str = ""

    @property
    def cache_path(self) -> Path:
        return CACHE_DIR / f"noaa_{self.current_id}_{self.water_level_id}.json"


REFERENCE_STATIONS = (
    StationPair("Mayport, FL (St John's entrance)", "jx0101", "8720218",
                "strongly rotary current, semidiurnal Atlantic"),
    StationPair("Friday Harbor, WA (Point Colville)", "PUG1727", "9449880",
                "mixed regime, rotary, Salish Sea"),
    StationPair("Boston, MA (Boston Channel)", "BOS1113", "8443970",
                "strongly semidiurnal, weak current"),
    StationPair("Richmond, CA (San Francisco Bay)", "SFB1310", "9414863",
                "mixed mainly semidiurnal, strong current"),
    StationPair("Woods Hole, MA (Cape Cod)", "COD0910", "8447930",
                "strong reversing current, close pairing"),
    StationPair("Kings Bay, GA (Cumberland Sound)", "kb0401", "8679598",
                "strong current, 0.4 km pairing"),
)

_WANTED = ("M2", "S2", "N2", "K1", "O1")


def _get(url: str, timeout: float = 60.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DataUnavailable(
            "NOAA CO-OPS metadata API",
            f"request to {url} failed: {type(exc).__name__}: {exc}",
            "Check network access, or use a cached extraction in data/constituents/.",
        ) from exc


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_pair(pair: StationPair, cache_dir: Path | None = None) -> dict:
    """Download and cache the harmonic constants for a station pair.

    The cached payload records the retrieval date, the station coordinates and
    their separation, so the provenance card can state exactly what was used.
    """
    cache = (cache_dir or CACHE_DIR) / pair.cache_path.name
    cur_meta = _get(f"{BASE}/stations/{pair.current_id}.json?type=currentpredictions")
    cur_meta = cur_meta.get("stations", [cur_meta])[0]
    wl_meta = _get(f"{BASE}/stations/{pair.water_level_id}.json?type=waterlevels")
    wl_meta = wl_meta.get("stations", [wl_meta])[0]

    cur = _get(
        f"{BASE}/stations/{pair.current_id}/harcon.json?units=metric&type=currentpredictions"
    )
    wl = _get(f"{BASE}/stations/{pair.water_level_id}/harcon.json?units=metric")

    hc = cur.get("HarmonicConstituents", [])
    if not hc:
        raise DataUnavailable(
            f"NOAA current station {pair.current_id}",
            "the station publishes no harmonic current constants (it is probably a "
            "subordinate station, predicted by offsets from a reference station)",
            "Choose a station of type 'H' from the currentpredictions station list.",
        )
    first_bin = min(c.get("binNbr", 1) for c in hc)
    current = {
        c["constituentName"]: c for c in hc if c.get("binNbr", 1) == first_bin
    }
    elevation = {c["name"]: c for c in wl.get("HarmonicConstituents", [])}

    missing = [k for k in _WANTED if k not in current or k not in elevation]
    if missing:
        raise DataUnavailable(
            f"NOAA station pair {pair.current_id}/{pair.water_level_id}",
            f"constituent(s) {', '.join(missing)} not published for this pair",
            "Choose another pair from REFERENCE_STATIONS.",
        )

    payload = {
        "label": pair.label,
        "note": pair.note,
        "retrieved": _dt.date.today().isoformat(),
        "source": NOAA_CITATION.document,
        "url": NOAA_CITATION.url,
        "licence": "US Government work, public domain",
        "current_station": {
            "id": pair.current_id,
            "name": cur_meta.get("name"),
            "lat": cur_meta.get("lat"),
            "lng": cur_meta.get("lng"),
            "bin": first_bin,
            "bin_depth_m": current[_WANTED[0]].get("binDepth"),
            "units": cur.get("units"),
        },
        "water_level_station": {
            "id": pair.water_level_id,
            "name": wl_meta.get("name"),
            "lat": wl_meta.get("lat"),
            "lng": wl_meta.get("lng"),
            "units": wl.get("units"),
        },
        "separation_km": round(
            _haversine_km(
                float(cur_meta.get("lat", 0.0)), float(cur_meta.get("lng", 0.0)),
                float(wl_meta.get("lat", 0.0)), float(wl_meta.get("lng", 0.0)),
            ), 2,
        ),
        "constituents": {
            k: {
                "elev_amp_m": float(elevation[k]["amplitude"]),
                "elev_phase_gmt_deg": float(elevation[k]["phase_GMT"]),
                "major_amp_cm_s": float(current[k]["majorAmplitude"]),
                "minor_amp_cm_s": float(current[k]["minorAmplitude"]),
                "major_phase_gmt_deg": float(current[k]["majorPhaseGMT"]),
                "azimuth_deg_true": float(current[k]["azi"]),
            }
            for k in _WANTED
        },
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_pair(pair: StationPair, cache_dir: Path | None = None, allow_fetch: bool = False) -> dict:
    """Read a cached extraction, optionally downloading it if absent."""
    cache = (cache_dir or CACHE_DIR) / pair.cache_path.name
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    if allow_fetch:
        return fetch_pair(pair, cache_dir)
    raise DataUnavailable(
        f"NOAA harmonic constants for {pair.label}",
        f"{cache} is not present and fetching was not permitted",
        "Run scripts/fetch_tides.py to populate data/constituents/.",
    )


def available_cached(cache_dir: Path | None = None) -> list[StationPair]:
    """Station pairs whose constants are already on disk."""
    d = cache_dir or CACHE_DIR
    return [p for p in REFERENCE_STATIONS if (d / p.cache_path.name).is_file()]


def to_constituents(payload: dict) -> TidalConstituents:
    """Convert a cached NOAA payload into MEASURED tidal constituents.

    Unit and convention conversions, all of them ours to get right:

    * current amplitudes cm/s -> m/s;
    * NOAA ``azi`` is the major-axis azimuth in degrees **true** (clockwise from
      north); ``inclination`` here is measured **anticlockwise from east**, so
      ``inclination = 90 - azi``;
    * Greenwich phases are used for both elevation and current, so their
      relative phase - which sets the in-phase/quadrature split - is preserved.
    """
    c = payload["constituents"]
    names = tuple(k for k in _WANTED if k in c)
    cur = payload.get("current_station", {})
    wl = payload.get("water_level_station", {})

    return TidalConstituents(
        names=names,
        omega=np.array(
            [np.radians(CONSTITUENT_SPEEDS_DEG_PER_HOUR[n]) / 3600.0 for n in names]
        ),
        elev_amp=np.array([c[n]["elev_amp_m"] for n in names]),
        elev_phase=np.radians([c[n]["elev_phase_gmt_deg"] for n in names]),
        semi_major=np.array([c[n]["major_amp_cm_s"] / 100.0 for n in names]),
        semi_minor=np.array([c[n]["minor_amp_cm_s"] / 100.0 for n in names]),
        inclination=np.radians([90.0 - c[n]["azimuth_deg_true"] for n in names]),
        current_phase=np.radians([c[n]["major_phase_gmt_deg"] for n in names]),
        provenance=Provenance.MEASURED,
        citation=Citation(
            document=NOAA_CITATION.document,
            locator=(
                f"current station {cur.get('id')} '{cur.get('name')}' bin {cur.get('bin')} "
                f"at {cur.get('bin_depth_m')} m; water level station {wl.get('id')} "
                f"'{wl.get('name')}'; stations {payload.get('separation_km')} km apart"
            ),
            url=NOAA_CITATION.url,
            retrieved=_dt.date.fromisoformat(payload["retrieved"]),
        ),
        latitude=float(cur.get("lat", 0.0)),
        longitude=float(cur.get("lng", 0.0)),
        source_note=(
            f"NOAA {payload.get('label')} - {payload.get('note')}. Current constants are from "
            f"a bin at {cur.get('bin_depth_m')} m and are used as the depth-averaged current; "
            f"the elevation gauge is {payload.get('separation_km')} km from the current meter."
        ),
    )
