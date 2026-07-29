"""Tidal constituents and rotary current reconstruction.

Constituent *frequencies* are astronomical and PUBLISHED: they come from the
standard Doodson development of the tide-generating potential and are identical
everywhere on Earth. Constituent *amplitudes and phases* are local and MEASURED:
they must be extracted from a tide model (TPXO9-atlas or FES2014 via ``pyTMD``)
at the platform coordinates.

If no tide model is configured, this module raises
:class:`~tidetwin.provenance.DataUnavailable`. It does **not** substitute
plausible-looking amplitudes: a fabricated tidal forcing would make every
downstream claim untestable while looking as though it had been tested. The user
may instead enter harmonic constants by hand, in which case they are ASSUMED and
every dependent result is flagged assumption-contaminated.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..provenance import Citation, DataUnavailable, Provenance, Quantity, measured, published

__all__ = [
    "CONSTITUENT_SPEEDS_DEG_PER_HOUR",
    "constituent_frequency",
    "TidalConstituents",
    "from_pytmd",
    "from_harmonic_constants",
    "tide_model_status",
    "DOODSON",
]

DOODSON = Citation(
    document=(
        "Doodson, A.T., 'The harmonic development of the tide-generating potential', "
        "Proc. R. Soc. Lond. A 100:305-329; speeds as tabulated by the International "
        "Hydrographic Organization standard constituent list"
    ),
    locator="principal lunar/solar constituents",
    year=1921,
)

TPXO = Citation(
    document="TPXO9-atlas global tidal solution (Egbert & Erofeeva, OSU)",
    locator="elevation and transport constituents interpolated at the platform position",
    url="https://www.tpxo.net/global",
)

# Angular speeds in degrees per mean solar hour. These are astronomical
# constants, not fitted values; the period follows as 360/speed hours.
CONSTITUENT_SPEEDS_DEG_PER_HOUR: dict[str, float] = {
    "M2": 28.9841042,  # principal lunar semidiurnal
    "S2": 30.0000000,  # principal solar semidiurnal
    "N2": 28.4397295,  # larger lunar elliptic semidiurnal
    "K2": 30.0821373,  # lunisolar semidiurnal
    "K1": 15.0410686,  # lunisolar diurnal
    "O1": 13.9430356,  # principal lunar diurnal
    "P1": 14.9589314,  # principal solar diurnal
    "Q1": 13.3986609,  # larger lunar elliptic diurnal
    "M4": 57.9682084,  # first lunar overtide
    "MS4": 58.9841042,  # lunisolar quarter-diurnal
    "MF": 1.0980331,  # lunar fortnightly
    "MM": 0.5443747,  # lunar monthly
}


def constituent_frequency(name: str) -> Quantity:
    """Angular frequency of a tidal constituent, rad/s. PUBLISHED."""
    key = name.upper()
    if key not in CONSTITUENT_SPEEDS_DEG_PER_HOUR:
        raise KeyError(f"unknown tidal constituent '{name}'")
    speed = CONSTITUENT_SPEEDS_DEG_PER_HOUR[key]
    omega = np.radians(speed) / 3600.0
    return published(
        omega,
        "rad/s",
        f"{key} angular frequency",
        DOODSON,
        note=f"speed {speed:.7f} deg/h, period {360.0 / speed:.6f} h",
    )


def constituent_period_hours(name: str) -> float:
    return 360.0 / CONSTITUENT_SPEEDS_DEG_PER_HOUR[name.upper()]


@dataclass(frozen=True)
class TidalConstituents:
    """Harmonic constants at one location.

    Elevation is ``eta(t) = sum_i A_i cos(omega_i t - g_i)``.

    Currents are described as tidal ellipses, the standard rotary
    parameterisation (Foreman, "Manual for tidal currents analysis and
    prediction", IOS Pacific Marine Science Report 78-6, 1978):

        u(t) = a cos(inc) cos(omega t - g) - b sin(inc) sin(omega t - g)
        v(t) = a sin(inc) cos(omega t - g) + b cos(inc) sin(omega t - g)

    with ``a`` the semi-major axis, ``b`` the semi-minor axis (signed: negative
    means clockwise rotation), and ``inc`` the inclination of the major axis
    measured anticlockwise from east. A degenerate ellipse (``b = 0``) is a
    rectilinear current; a circular one (``|b| = a``) never slackens, and the
    difference between the two is exactly what the C3 nuisance budget must
    propagate.
    """

    names: tuple[str, ...]
    omega: np.ndarray  # rad/s
    elev_amp: np.ndarray  # m
    elev_phase: np.ndarray  # rad
    semi_major: np.ndarray  # m/s, depth-averaged
    semi_minor: np.ndarray  # m/s, signed
    inclination: np.ndarray  # rad, anticlockwise from east
    current_phase: np.ndarray  # rad
    provenance: Provenance
    citation: Citation | None
    latitude: float
    longitude: float
    source_note: str = ""

    def __post_init__(self) -> None:
        n = len(self.names)
        for field_name in (
            "omega",
            "elev_amp",
            "elev_phase",
            "semi_major",
            "semi_minor",
            "inclination",
            "current_phase",
        ):
            arr = getattr(self, field_name)
            if len(arr) != n:
                raise ValueError(f"{field_name} has {len(arr)} entries, expected {n}")

    def index(self, name: str) -> int:
        return self.names.index(name.upper())

    def elevation(self, t_seconds: np.ndarray) -> np.ndarray:
        """Tidal elevation relative to mean sea level, m."""
        t = np.atleast_1d(np.asarray(t_seconds, dtype=float))
        phase = self.omega[None, :] * t[:, None] - self.elev_phase[None, :]
        return (self.elev_amp[None, :] * np.cos(phase)).sum(axis=1)

    def depth_averaged_current(self, t_seconds: np.ndarray) -> np.ndarray:
        """Depth-averaged tidal current ``(n_t, 2)`` as eastward, northward m/s."""
        t = np.atleast_1d(np.asarray(t_seconds, dtype=float))
        ph = self.omega[None, :] * t[:, None] - self.current_phase[None, :]
        c, s = np.cos(ph), np.sin(ph)
        ci, si = np.cos(self.inclination)[None, :], np.sin(self.inclination)[None, :]
        a, b = self.semi_major[None, :], self.semi_minor[None, :]
        u = (a * ci * c - b * si * s).sum(axis=1)
        v = (a * si * c + b * ci * s).sum(axis=1)
        return np.column_stack([u, v])

    def subset(self, names: tuple[str, ...]) -> "TidalConstituents":
        idx = [self.index(n) for n in names]
        return TidalConstituents(
            names=tuple(self.names[i] for i in idx),
            omega=self.omega[idx],
            elev_amp=self.elev_amp[idx],
            elev_phase=self.elev_phase[idx],
            semi_major=self.semi_major[idx],
            semi_minor=self.semi_minor[idx],
            inclination=self.inclination[idx],
            current_phase=self.current_phase[idx],
            provenance=self.provenance,
            citation=self.citation,
            latitude=self.latitude,
            longitude=self.longitude,
            source_note=self.source_note,
        )

    def form_factor(self) -> float:
        """Courtier's form factor ``F = (K1 + O1) / (M2 + S2)``.

        ``F < 0.25`` semidiurnal, ``0.25-1.5`` mixed mainly semidiurnal,
        ``1.5-3.0`` mixed mainly diurnal, ``> 3.0`` diurnal. Reported because it
        governs how cleanly an M2-carrier method can work at the site.
        """
        try:
            num = self.elev_amp[self.index("K1")] + self.elev_amp[self.index("O1")]
            den = self.elev_amp[self.index("M2")] + self.elev_amp[self.index("S2")]
        except ValueError as exc:
            raise ValueError("form factor needs M2, S2, K1 and O1") from exc
        return float(num / den) if den > 0 else float("inf")

    def spring_neap_ratio(self) -> float:
        """``(M2 + S2) / (M2 - S2)``: how much the M2 carrier is modulated."""
        m2 = self.elev_amp[self.index("M2")]
        s2 = self.elev_amp[self.index("S2")]
        return float((m2 + s2) / (m2 - s2)) if m2 > s2 else float("inf")

    def as_quantities(self) -> list[Quantity]:
        out: list[Quantity] = []
        for i, n in enumerate(self.names):
            kw = dict(name=f"{n} elevation amplitude", units="m")
            if self.provenance is Provenance.MEASURED and self.citation is not None:
                out.append(measured(float(self.elev_amp[i]), citation=self.citation, **kw))
            else:
                out.append(
                    Quantity(
                        float(self.elev_amp[i]),
                        "m",
                        self.provenance,
                        f"{n} elevation amplitude",
                        self.citation,
                        note=self.source_note,
                    )
                )
        return out


def tide_model_status(model_dir: Path | str | None = None) -> tuple[bool, str]:
    """Whether a real tide model is usable here, and why not if not."""
    try:
        import pyTMD  # noqa: F401
    except ImportError:
        return False, (
            "pyTMD is not installed. Add it to requirements.txt and provide a "
            "TPXO9-atlas or FES2014 model directory."
        )
    d = Path(model_dir) if model_dir else None
    if d is None:
        return False, (
            "No tide model directory configured. Set TIDETWIN_TIDE_MODEL_DIR or pass "
            "model_dir. TPXO9-atlas requires a (free, registered) download from "
            "https://www.tpxo.net/global; FES2014 from AVISO."
        )
    if not d.is_dir():
        return False, f"Tide model directory {d} does not exist."
    return True, f"Tide model directory {d} present."


def from_pytmd(
    latitude: float,
    longitude: float,
    model_dir: Path | str,
    model_name: str = "TPXO9-atlas-v5",
    constituents: tuple[str, ...] = ("M2", "S2", "N2", "K1", "O1"),
    retrieved: _dt.date | None = None,
) -> TidalConstituents:
    """Extract harmonic constants at a point from a TPXO/FES model via ``pyTMD``.

    Raises
    ------
    DataUnavailable
        If ``pyTMD`` is not installed or the model files are absent. The tide
        model is a multi-gigabyte registered download and cannot be vendored
        into this repository.
    """
    ok, why = tide_model_status(model_dir)
    if not ok:
        raise DataUnavailable(
            "Tidal harmonic constants (TPXO/FES)",
            why,
            "Install pyTMD, download TPXO9-atlas, and set TIDETWIN_TIDE_MODEL_DIR to its path.",
        )

    raise DataUnavailable(  # pragma: no cover
        "Tidal harmonic constants (TPXO/FES)",
        (
            "pyTMD is installed and a model directory was supplied, but this build has "
            "not been exercised against a real TPXO/FES download, so the extraction path "
            "is not certified. Verify against a known station before trusting it."
        ),
        "Run scripts/extract_tides.py, check against a published station, then cache to data/constituents/.",
    )


#: Sidebar starting values so the machinery can be exercised with no tide model.
#:
#: These are NOT Arabian Gulf harmonic constants. They are round numbers chosen
#: to give a mixed, mainly semidiurnal tide with a rotary current, so that the
#: solvers, the Monte Carlo and the plots have something to chew on. Every value
#: is ASSUMED, renders red, and contaminates every result computed from it. The
#: point of the app is to replace them with a TPXO/FES extraction; until that
#: happens, C1 and C3 report numbers about a *hypothetical* tide, and the ledger
#: says so.
PLACEHOLDER_CONSTITUENTS: dict[str, dict[str, float]] = {
    "M2": {
        "elev_amp": 0.50, "elev_phase_deg": 0.0,
        "semi_major": 0.25, "semi_minor": 0.08, "inclination_deg": 40.0, "current_phase_deg": 80.0,
    },
    "S2": {
        "elev_amp": 0.20, "elev_phase_deg": 40.0,
        "semi_major": 0.10, "semi_minor": 0.03, "inclination_deg": 40.0, "current_phase_deg": 120.0,
    },
    "N2": {
        "elev_amp": 0.10, "elev_phase_deg": 340.0,
        "semi_major": 0.05, "semi_minor": 0.015, "inclination_deg": 40.0, "current_phase_deg": 60.0,
    },
    "K1": {
        "elev_amp": 0.25, "elev_phase_deg": 100.0,
        "semi_major": 0.09, "semi_minor": -0.04, "inclination_deg": 25.0, "current_phase_deg": 150.0,
    },
    "O1": {
        "elev_amp": 0.15, "elev_phase_deg": 70.0,
        "semi_major": 0.06, "semi_minor": -0.025, "inclination_deg": 25.0, "current_phase_deg": 130.0,
    },
}


def placeholder_constituents(
    latitude: float = 24.9, longitude: float = 53.2
) -> "TidalConstituents":
    """ASSUMED constituents for exercising the solvers without a tide model.

    See :data:`PLACEHOLDER_CONSTITUENTS`. Provenance is ASSUMED by construction
    and cannot be overridden here.
    """
    return from_harmonic_constants(
        latitude,
        longitude,
        PLACEHOLDER_CONSTITUENTS,
        source="PLACEHOLDER - not a real tide; replace with a TPXO/FES extraction",
        provenance=Provenance.ASSUMED,
    )


def from_harmonic_constants(
    latitude: float,
    longitude: float,
    table: dict[str, dict[str, float]],
    source: str = "user-entered",
    provenance: Provenance = Provenance.ASSUMED,
    citation: Citation | None = None,
) -> TidalConstituents:
    """Build constituents from a table of harmonic constants.

    ``table`` maps constituent name to a dict with keys ``elev_amp`` (m),
    ``elev_phase_deg``, ``semi_major`` (m/s), ``semi_minor`` (m/s, signed),
    ``inclination_deg`` and ``current_phase_deg``.

    The default provenance is ASSUMED, which is the honest classification for
    numbers typed into a sidebar. Pass ``Provenance.MEASURED`` with a citation
    only when the values genuinely come from a published tide table or a model
    extraction, and say which in ``source``.
    """
    if provenance in (Provenance.MEASURED, Provenance.PUBLISHED) and citation is None:
        raise ValueError("MEASURED or PUBLISHED harmonic constants require a citation")
    names = tuple(k.upper() for k in table)
    unknown = [n for n in names if n not in CONSTITUENT_SPEEDS_DEG_PER_HOUR]
    if unknown:
        raise KeyError(f"unknown constituent(s): {', '.join(unknown)}")

    def col(key: str, default: float = 0.0) -> np.ndarray:
        return np.array([float(table[k].get(key, default)) for k in table])

    return TidalConstituents(
        names=names,
        omega=np.array([np.radians(CONSTITUENT_SPEEDS_DEG_PER_HOUR[n]) / 3600.0 for n in names]),
        elev_amp=col("elev_amp"),
        elev_phase=np.radians(col("elev_phase_deg")),
        semi_major=col("semi_major"),
        semi_minor=col("semi_minor"),
        inclination=np.radians(col("inclination_deg")),
        current_phase=np.radians(col("current_phase_deg")),
        provenance=provenance,
        citation=citation,
        latitude=latitude,
        longitude=longitude,
        source_note=source,
    )
