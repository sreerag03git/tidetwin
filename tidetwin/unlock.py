"""What is locked, what unlocks it, and what it costs.

Four of the nine claims currently report ``UNTESTABLE - DATA MISSING``. That is
the honest state, not a shortfall: the data needed to settle them is behind
paywalls, registrations, or a finite-element run this application cannot do
inside a web container. But "honest" should not mean "opaque", so this module
enumerates every gate, what it unlocks, what it costs, and exactly where the
value goes once supplied.

Nothing here is a placeholder waiting to be filled with a guess. Each entry names
a specific document, table or dataset, and a specific file path to put it in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

__all__ = ["Gate", "GATES", "gate_status"]


@dataclass(frozen=True)
class Gate:
    """One thing you could obtain, and what it would buy."""

    key: str
    name: str
    unlocks: str
    claims: tuple[str, ...]
    cost: str
    effort: str
    how: str
    where: str
    check: Callable[[], tuple[bool, str]]


def _era5() -> tuple[bool, str]:
    from .loads.era5 import credentials_status

    return credentials_status()


def _shell_fe() -> tuple[bool, str]:
    from .damage.crack_ljf import shell_fe_status

    return shell_fe_status()


def _sn() -> tuple[bool, str]:
    from .damage.sn import sn_status

    return sn_status()


def _paris() -> tuple[bool, str]:
    from .damage.paris import paris_status

    return paris_status()


def _scf() -> tuple[bool, str]:
    from .damage.scf import scf_status

    return scf_status()


def _pod() -> tuple[bool, str]:
    from .claims.tests.c9_positioning import competitor_status

    return competitor_status()


def _tpxo() -> tuple[bool, str]:
    import os

    from .loads.tides import tide_model_status

    return tide_model_status(os.environ.get("TIDETWIN_TIDE_MODEL_DIR"))


def _ljf() -> tuple[bool, str]:
    from .provenance import DataUnavailable
    from .fe.ljf import load_tabulated

    try:
        load_tabulated("buitrago1993")
        return True, "Published LJF coefficients are available."
    except DataUnavailable as exc:
        return False, str(exc)


GATES: tuple[Gate, ...] = (
    Gate(
        key="era5",
        name="ERA5 metocean (Copernicus CDS)",
        unlocks="Real wind, wave and temperature series instead of assumed ranges. "
        "C5's thermal amplitude becomes computable, and two of C3's eight nuisance "
        "channels move from ASSUMED to MEASURED.",
        claims=("C5", "C3"),
        cost="Free",
        effort="10 minutes",
        how="Register at cds.climate.copernicus.eu, accept the ERA5 licence, copy your "
        "API key.",
        where="Set CDSAPI_KEY in .streamlit/secrets.toml, or ~/.cdsapirc locally.",
        check=_era5,
    ),
    Gate(
        key="shell_fe",
        name="Shell-FE crack-to-LJF surface",
        unlocks="C2's damage signature stops being a lower bound and becomes an answer. "
        "This is the single highest-value gate: C2 currently computes 0.004 percent "
        "against a claimed 11.1 percent, via a line-spring model documented to "
        "under-predict, so the true figure is unknown between those two.",
        claims=("C2", "C3"),
        cost="Free with CalculiX or Code_Aster; a licence if you use Abaqus or ANSYS",
        effort="Days - it needs a meshed cracked K-joint and a convergence study",
        how="Mesh the chord-brace intersection with a semi-elliptical crack, sweep "
        "(a/T, 2c), extract the joint compliance at each, and record the mesh "
        "convergence study and validation against Soh (2000) / Rhee (2005).",
        where="data/shell_fe_surface/ with manifest.json - see "
        "tidetwin.damage.crack_ljf.load_shell_fe_surface for the schema.",
        check=_shell_fe,
    ),
    Gate(
        key="tpxo",
        name="TPXO9-atlas or FES2014 tide model",
        unlocks="Site-specific tidal forcing for the actual platform, replacing the "
        "NOAA reference stations. Changes the magnitudes; the C3 verdict is the same at "
        "all six stations already tested, so this refines rather than decides.",
        claims=("C1", "C3", "C4"),
        cost="Free, registration required",
        effort="An hour, plus a multi-gigabyte download",
        how="Register at tpxo.net (or AVISO for FES2014), download the atlas, then "
        "pip install -r requirements-data.txt for pyTMD.",
        where="Point TIDETWIN_TIDE_MODEL_DIR at the unpacked model directory.",
        check=_tpxo,
    ),
    Gate(
        key="paris",
        name="BS 7910:2019 Table 8 - Paris law constants",
        unlocks="Real crack growth rates, so C6's remaining-life claim of +/-0.9 years "
        "becomes testable rather than an estimator demonstration.",
        claims=("C6",),
        cost="Paid standard",
        effort="15 minutes once you have the document",
        how="Transcribe A, m and the threshold for steels in marine environments with "
        "cathodic protection. Check the units carefully - A differs by orders of "
        "magnitude between N/mm^1.5 and MPa.m^0.5 conventions.",
        where="data/paris/bs7910_marine_cp.json - schema in tidetwin.damage.paris.",
        check=_paris,
    ),
    Gate(
        key="sn",
        name="DNV-RP-C203 Table 2-2 - S-N curve T",
        unlocks="The conventional S-N and Miner life estimate, which is the status quo "
        "C6 has to beat. Without it the comparison is against prior propagation only.",
        claims=("C6",),
        cost="Paid standard",
        effort="10 minutes",
        how="Transcribe log a1, m1, log a2, m2, the thickness exponent and t_ref for "
        "curve T in seawater with cathodic protection.",
        where="data/sn/dnv_rp_c203_T.json - schema in tidetwin.damage.sn.",
        check=_sn,
    ),
    Gate(
        key="scf",
        name="Efthymiou SCFs (DNV-RP-C203 Appendix B)",
        unlocks="Hot-spot stress at the joint, which the fatigue-life route needs. No "
        "claim currently depends on it, so this is groundwork rather than a gate.",
        claims=(),
        cost="Paid standard",
        effort="30 minutes",
        how="Transcribe the coefficients for the K-joint load cases. An SCF error "
        "propagates to the fifth power through the S-N curve, so check them twice.",
        where="data/scf/efthymiou.json - schema in tidetwin.damage.scf.",
        check=_scf,
    ),
    Gate(
        key="ljf",
        name="Published LJF regressions (Buitrago 1993 / Fessler 1986)",
        unlocks="An independent local-joint-flexibility formulation to check the "
        "first-principles shell model against. C1, C2, C3 and C7 all shift with the LJF "
        "model, so a second opinion bounds that sensitivity.",
        claims=("C1", "C2", "C7"),
        cost="Paywalled conference proceedings",
        effort="30 minutes",
        how="Transcribe the influence-factor coefficients for axial, IPB and OPB.",
        where="data/ljf/buitrago1993.json - schema in tidetwin.fe.ljf.load_tabulated.",
        check=_ljf,
    ),
    Gate(
        key="pod",
        name="Published POD curves for ROV MPI, ACFM and FMD",
        unlocks="C9's comparison against the inspection methods this one would replace. "
        "Flooded member detection is the real incumbent competitor on cost.",
        claims=("C9",),
        cost="Free if you have the source figures; digitising effort",
        effort="An hour per curve",
        how="Digitise POD against crack depth from the published qualification studies.",
        where="data/pod/<method>.csv with columns method, crack_depth_mm, pod.",
        check=_pod,
    ),
)


def gate_status() -> list[tuple[Gate, bool, str]]:
    """Every gate with its current availability. Never raises."""
    out = []
    for g in GATES:
        try:
            ok, why = g.check()
        except Exception as exc:  # noqa: BLE001
            ok, why = False, f"{type(exc).__name__}: {exc}"
        out.append((g, ok, why))
    return out
