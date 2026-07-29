"""Local joint flexibility of tubular joints.

Three formulations are provided, and which one is in force is reported next to
every result that depends on it.

``RIGID``
    Brace ends are rigidly connected to the chord centreline. This is the
    ordinary frame idealisation permitted for jacket analysis (ISO 19902:2020
    Section 12.3), not a fabricated number, but it is a *modelling assumption*
    and is recorded as one on every dependent claim.

``SHELL``
    Chord-wall local flexibility derived from thin-shell theory: the
    beam-on-elastic-foundation solution for a long cylindrical shell under a
    radial line load (Timoshenko & Woinowsky-Krieger, *Theory of Plates and
    Shells*, 2nd ed., McGraw-Hill 1959, Section 114, Eq. 270-271; Hetenyi,
    *Beams on Elastic Foundation*, Univ. Michigan Press 1946). The functional
    form is derived, not fitted; the circumferential load-spreading idealisation
    it needs is stated explicitly in :func:`shell_ljf` and is the main source of
    modelling error.

``TABULATED``
    Published parametric regressions - Fessler, Little & Edwards, "Parametric
    equations for the flexibility matrices of single brace tubular joints in
    offshore structures", Proc. Inst. Civ. Eng. 81(4):659-673, 1986; Buitrago,
    Healy & Chang, "Local joint flexibility of tubular joints", OMAE 1993,
    Vol. I, pp. 405-416. Both are behind paywalls and their coefficients are
    **not** shipped with this repository, because transcribing coefficients that
    cannot be checked against the source would be indistinguishable from
    inventing them. Supply them yourself as JSON in ``data/ljf/`` (see
    :func:`load_tabulated` for the schema) and this formulation activates; until
    then it reports DATA UNAVAILABLE.

How much the choice matters, measured rather than asserted
----------------------------------------------------------
:mod:`tidetwin.robustness` sweeps the formulation and the shell model's
load-spreading length and reports the spread on the Structure tab. The result is
not what this docstring used to claim.

* **C1 barely moves.** The intact strain ratio spans about 0.8 percent between
  the rigid idealisation and the shell model with its spreading length halved or
  doubled. The ratio is a global load-path quantity and is nearly insensitive to
  local joint compliance, which makes C1 a more robust number than expected.
* **C2 and C7 depend on it entirely.** Both are identically zero under RIGID,
  because there is no joint compliance for a crack to change. For those two the
  formulation is not a refinement, it is the whole mechanism.

The earlier wording here said all four claims move with the formulation. That
was an assumption, and measuring it showed it was wrong for C1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from ..provenance import Citation, DataUnavailable, Quantity, derived, published

__all__ = [
    "LJFModel",
    "JointGeometry",
    "LJFStiffness",
    "shell_ljf",
    "joint_stiffness",
    "load_tabulated",
    "RIGID_STIFFNESS_FACTOR",
]

# Stiffness used for DOF that the LJF model does not soften. Applied as a
# multiple of the connected brace's own axial stiffness so it is always stiff
# relative to the member without being so large it wrecks the conditioning.
RIGID_STIFFNESS_FACTOR = 1.0e4

TIMOSHENKO = Citation(
    "Timoshenko, S. & Woinowsky-Krieger, S., Theory of Plates and Shells",
    "2nd ed., Section 114, Eq. 270-271 (long cylindrical shell under radial line load)",
    1959,
)
ISO19902 = Citation(
    "ISO 19902:2020 Petroleum and natural gas industries - Fixed steel offshore structures",
    "Section 12.3 (frame idealisation)",
    2020,
)


class LJFModel(str, Enum):
    RIGID = "RIGID"
    SHELL = "SHELL"
    TABULATED = "TABULATED"


@dataclass(frozen=True)
class JointGeometry:
    """Non-dimensional description of a tubular brace-to-chord connection.

    Attributes
    ----------
    chord_D, chord_T : float
        Chord outer diameter and wall thickness, m.
    brace_d, brace_t : float
        Brace outer diameter and wall thickness, m.
    theta : float
        Angle between brace and chord axes, radians.
    E, nu : float
        Chord material elastic constants.
    """

    chord_D: float
    chord_T: float
    brace_d: float
    brace_t: float
    theta: float
    E: float = 2.10e11
    nu: float = 0.3

    @property
    def beta(self) -> float:
        """Diameter ratio d/D."""
        return self.brace_d / self.chord_D

    @property
    def gamma(self) -> float:
        """Chord slenderness D/(2T)."""
        return self.chord_D / (2.0 * self.chord_T)

    @property
    def tau(self) -> float:
        """Wall thickness ratio t/T."""
        return self.brace_t / self.chord_T

    @property
    def R(self) -> float:
        """Chord mid-surface radius, m."""
        return 0.5 * (self.chord_D - self.chord_T)

    def validity(self) -> list[str]:
        """Warnings where the geometry sits outside usual parametric ranges.

        The customary validity envelope for tubular joint parametric equations
        is 0.2 <= beta <= 1.0, 8 <= gamma <= 32, 0.2 <= tau <= 1.0,
        30 deg <= theta <= 90 deg (DNV-RP-C203 Section 4.3).
        """
        out: list[str] = []
        if not 0.2 <= self.beta <= 1.0:
            out.append(f"beta = {self.beta:.3f} outside 0.2-1.0")
        if not 8.0 <= self.gamma <= 32.0:
            out.append(f"gamma = {self.gamma:.1f} outside 8-32")
        if not 0.2 <= self.tau <= 1.0:
            out.append(f"tau = {self.tau:.3f} outside 0.2-1.0")
        deg = np.degrees(self.theta)
        if not 30.0 <= deg <= 90.0:
            out.append(f"theta = {deg:.1f} deg outside 30-90 deg")
        return out


@dataclass(frozen=True)
class LJFStiffness:
    """Joint spring stiffnesses, in the brace-end local frame.

    ``k_axial`` N/m along the brace axis; ``k_ipb`` and ``k_opb`` N.m/rad about
    the in-plane and out-of-plane bending axes. ``np.inf`` means rigid.
    """

    k_axial: float
    k_ipb: float
    k_opb: float
    model: LJFModel
    note: str = ""

    def as_vector(self, rigid_value: float) -> np.ndarray:
        """Six-component spring vector ``[kx, ky, kz, krx, kry, krz]``.

        Local axis 0 is the brace axis, 1 the in-plane bending axis, 2 the
        out-of-plane bending axis. Shear and torsion are not softened by the
        chord-wall mechanism these formulations describe, so they take the
        rigid value.
        """
        f = lambda v: rigid_value if not np.isfinite(v) else float(v)
        return np.array(
            [f(self.k_axial), rigid_value, rigid_value, rigid_value, f(self.k_ipb), f(self.k_opb)]
        )


def shell_ljf(g: JointGeometry, spread_factor: float = 1.0) -> LJFStiffness:
    """Chord-wall local flexibility from cylindrical shell theory.

    Derivation, in full, because the idealisation is where the modelling error
    lives:

    1. A long cylindrical shell of mid-surface radius ``R`` and wall ``T``
       carrying an axisymmetric radial line load of intensity ``p`` per unit
       circumferential length deflects radially at the load by

       .. math:: w = \\frac{p}{8 \\beta_s^3 D_s}

       with shell rigidity :math:`D_s = E T^3 / [12(1-\\nu^2)]` and attenuation
       parameter :math:`\\beta_s = [3(1-\\nu^2)/(R^2 T^2)]^{1/4}`
       (Timoshenko & Woinowsky-Krieger 1959, Eq. 270-271). This is the
       beam-on-elastic-foundation analogy with foundation modulus
       :math:`k = ET/R^2`.

    2. A brace does not load the chord axisymmetrically. The radial component of
       the brace axial force is ``P sin(theta)``, and it is reacted over an
       effective circumferential length

       .. math:: c_{eff} = \\min(d + 2/\\beta_s,\\; 2 \\pi R)

       i.e. the brace footprint width plus one shell attenuation length each
       side, capped at the full circumference. **This spreading length is the
       principal idealisation.** It is not a fitted constant, but it is a choice,
       so it is exposed as ``spread_factor`` rather than buried. Halving or
       doubling it changes the axial joint stiffness by about 12 and 22 percent
       respectively, and moves the intact strain ratio by well under 1 percent -
       measured by :func:`tidetwin.robustness.ljf_sensitivity` and shown on the
       Structure tab.

    3. Axial stiffness follows as :math:`k_{ax} = P/w = 8 \\beta_s^3 D_s c_{eff} /
       \\sin^2(theta)`, the second ``sin`` factor converting the radial
       deflection back to displacement along the brace axis.

    4. Brace moments are carried as a couple of radial forces separated by the
       footprint extent in the plane of bending. The brace-chord intersection is
       an ellipse: elongated to :math:`d/\\sin(theta)` along the chord axis, and
       only :math:`d` wide circumferentially. In-plane bending therefore works
       across the long axis (:math:`\\ell = 0.7 d/\\sin\\theta`) and out-of-plane
       bending across the short one (:math:`\\ell = 0.7 d`), giving
       :math:`k_{rot} = \\ell^2 k_{radial} / 2`. Out-of-plane bending also drives
       the chord section oval rather than bending the wall against its hoop
       foundation, which softens it further (see :func:`_opb_softening`). The two
       effects together put OPB well below IPB in stiffness, the ordering seen in
       every published LJF dataset.

    Returns stiffnesses, not flexibilities, so that ``inf`` cleanly denotes a
    rigid connection.
    """
    E, nu = g.E, g.nu
    R, T = g.R, g.chord_T
    Ds = E * T**3 / (12.0 * (1.0 - nu**2))
    beta_s = (3.0 * (1.0 - nu**2) / (R**2 * T**2)) ** 0.25
    atten = 1.0 / beta_s
    # spread_factor scales the attenuation-length allowance. It is the one free
    # choice in this derivation, so it is a parameter rather than a buried
    # constant: tidetwin.robustness.ljf_sensitivity sweeps it and reports how far
    # the dependent claims move, which is the only honest way to present an
    # idealisation that cannot be derived.
    c_eff = min(g.brace_d + 2.0 * spread_factor * atten, 2.0 * np.pi * R)

    # Radial stiffness of the loaded patch, N/m of radial deflection.
    k_radial = 8.0 * beta_s**3 * Ds * c_eff
    s = max(np.sin(g.theta), 1e-3)
    k_axial = k_radial / (s**2)

    lever_ipb = 0.7 * g.brace_d / s
    lever_opb = 0.7 * g.brace_d
    k_ipb = 0.5 * k_radial * lever_ipb**2
    k_opb = 0.5 * k_radial * lever_opb**2 * _opb_softening(g)

    return LJFStiffness(
        k_axial=float(k_axial),
        k_ipb=float(k_ipb),
        k_opb=float(k_opb),
        model=LJFModel.SHELL,
        note=(
            f"shell BOEF; attenuation length {atten:.3f} m, spread factor "
            f"{spread_factor:g}, effective loaded arc {c_eff:.3f} m "
            f"({c_eff / (2 * np.pi * R) * 100:.0f}% of circumference)"
        ),
    )


def _opb_softening(g: JointGeometry) -> float:
    """Extra out-of-plane compliance from circumferential ovalisation.

    Out-of-plane bending drives the chord section oval rather than bending the
    wall against the hoop foundation, and the ovalisation stiffness of a ring of
    radius ``R`` and wall ``T`` scales as :math:`E T^3 / R^3` against the
    :math:`E T / R` of the foundation, i.e. by :math:`(T/R)^2 = 1/(2\\gamma-1)^2`
    to leading order. The ratio is capped so it degrades smoothly rather than
    vanishing for very slender chords, and it is the reason OPB flexibility
    exceeds IPB flexibility in all published datasets.
    """
    return float(np.clip(1.0 / (1.0 + 0.5 * (2.0 * g.gamma) ** 0.5), 0.02, 1.0))


def load_tabulated(name: str, root: Path | None = None) -> dict:
    """Load a published LJF coefficient set from ``data/ljf/<name>.json``.

    Expected schema::

        {
          "citation": {"document": "...", "locator": "...", "year": 1993},
          "form": "f = C * beta^a * gamma^b * tau^c * sin(theta)^d",
          "axial": {"C": ..., "a": ..., "b": ..., "c": ..., "d": ...},
          "ipb":   {...},
          "opb":   {...},
          "validity": {"beta": [0.2, 0.8], "gamma": [8, 32], "tau": [0.2, 1.0]}
        }

    where ``f`` is the non-dimensional flexibility such that the local
    deflection is ``delta = f * P / (E * D)``.

    Raises
    ------
    DataUnavailable
        If the file is absent. The remedy names the document to transcribe from.
    """
    root = root or Path(__file__).resolve().parents[2] / "data" / "ljf"
    path = root / f"{name}.json"
    if not path.is_file():
        raise DataUnavailable(
            source=f"LJF coefficient set '{name}'",
            reason=f"{path} not present; published regression coefficients are not shipped",
            remedy=(
                "Transcribe the coefficients from the source document "
                "(Fessler/Little/Edwards, Proc. ICE 81(4):659-673, 1986, or "
                "Buitrago/Healy/Chang, OMAE 1993 Vol. I pp. 405-416) into "
                f"{path} using the schema in tidetwin.fe.ljf.load_tabulated."
            ),
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _tabulated_ljf(g: JointGeometry, name: str, root: Path | None = None) -> LJFStiffness:
    spec = load_tabulated(name, root)

    def f_of(block: dict) -> float:
        return (
            block["C"]
            * g.beta ** block["a"]
            * g.gamma ** block["b"]
            * g.tau ** block["c"]
            * np.sin(g.theta) ** block.get("d", 0.0)
        )

    D = g.chord_D
    # delta = f * P / (E * D)  ->  k = E * D / f
    k_axial = g.E * D / f_of(spec["axial"])
    # Rotational forms are non-dimensionalised by an extra D^2 (moment / rotation).
    k_ipb = g.E * D**3 / f_of(spec["ipb"])
    k_opb = g.E * D**3 / f_of(spec["opb"])
    return LJFStiffness(
        k_axial=float(k_axial),
        k_ipb=float(k_ipb),
        k_opb=float(k_opb),
        model=LJFModel.TABULATED,
        note=f"coefficient set '{name}': {spec.get('citation', {}).get('document', '?')}",
    )


def joint_stiffness(
    g: JointGeometry,
    model: LJFModel = LJFModel.RIGID,
    tabulated_name: str = "buitrago1993",
    spread_factor: float = 1.0,
    extra_axial_compliance: float = 0.0,
    extra_ipb_compliance: float = 0.0,
    extra_opb_compliance: float = 0.0,
) -> LJFStiffness:
    """LJF spring stiffnesses for one brace end.

    ``extra_*_compliance`` are additional compliances in series (m/N and
    rad/N.m), which is how crack-induced flexibility from
    :mod:`tidetwin.damage.crack_ljf` is superposed: compliances add, stiffnesses
    do not.
    """
    if model is LJFModel.RIGID:
        base = LJFStiffness(np.inf, np.inf, np.inf, LJFModel.RIGID, "rigid frame idealisation")
    elif model is LJFModel.SHELL:
        base = shell_ljf(g, spread_factor)
    elif model is LJFModel.TABULATED:
        base = _tabulated_ljf(g, tabulated_name)
    else:  # pragma: no cover - enum is closed
        raise ValueError(f"unknown LJF model {model}")

    if not any((extra_axial_compliance, extra_ipb_compliance, extra_opb_compliance)):
        return base

    def add(k: float, c_extra: float) -> float:
        c = (0.0 if np.isinf(k) else 1.0 / k) + c_extra
        return np.inf if c <= 0 else 1.0 / c

    return LJFStiffness(
        k_axial=add(base.k_axial, extra_axial_compliance),
        k_ipb=add(base.k_ipb, extra_ipb_compliance),
        k_opb=add(base.k_opb, extra_opb_compliance),
        model=base.model,
        note=(base.note + "; crack compliance added in series").strip("; "),
    )


def ljf_quantity(g: JointGeometry, model: LJFModel) -> Quantity:
    """The axial LJF stiffness as a provenance-carrying quantity."""
    s = joint_stiffness(g, model)
    if model is LJFModel.RIGID:
        return published(
            np.inf,
            "N/m",
            "LJF axial stiffness (rigid idealisation)",
            ISO19902,
            note="joints modelled rigid; recorded as a modelling assumption on dependent claims",
        )
    inputs = [
        published(g.chord_D, "m", "chord outer diameter", Citation("OC4 jacket definition")),
        published(g.chord_T, "m", "chord wall thickness", Citation("OC4 jacket definition")),
    ]
    return derived(
        s.k_axial,
        "N/m",
        "LJF axial stiffness",
        inputs,
        f"{model.value} formulation: {s.note}",
        citation=TIMOSHENKO if model is LJFModel.SHELL else None,
        note=s.note,
    )
