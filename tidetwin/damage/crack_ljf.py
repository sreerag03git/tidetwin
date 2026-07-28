"""Crack-induced local joint flexibility.

Two routes, and the app reports which one produced any number it shows.

**Route A - shipped shell-FE surface (preferred).** A precomputed crack-to-LJF
surface over ``(a/T, 2c)``, generated offline with 3D shell elements and shipped
as a versioned table with its mesh convergence study and validation against
Soh (2000) and Rhee (2005). Full shell FE will not run inside a Streamlit
container, so it is never regenerated in-app - only interpolated.
**No such surface is shipped with this repository.** Generating one honestly
needs a shell FE code, a mesh convergence study and digitised experimental data,
none of which can be conjured from a language model without fabricating them. The
loader looks for it, reports its provenance card if present, and says DATA
UNAVAILABLE if not.

**Route B - line-spring compliance from fracture mechanics (default).** The extra
compliance a crack adds is obtained from its own stress-intensity factor by
Irwin's relation and Castigliano's theorem. For a crack whose SIF is
proportional to the applied load, ``K = P k(A)``, the strain energy stored by
the crack is ``U_c = (P^2/E') \\int k^2 dA`` and hence

.. math::
    \\Delta C = \\frac{\\partial^2 U_c}{\\partial P^2}
              = \\frac{2}{E'} \\int_A k(A)^2 \\, dA

with ``E' = E/(1-nu^2)`` for plane strain. This is the standard compliance-from-
SIF route set out in Tada, Paris & Irwin, *The Stress Analysis of Cracks
Handbook*, and applied to cracked structural members by Dimarogonas, "Vibration
of cracked structures: a state of the art review", Engineering Fracture
Mechanics 55(5):831-857, 1996. Combined with the Newman-Raju SIF for a
semi-elliptical surface crack it gives a runtime-computable
``crack geometry -> added compliance`` map with no fitted constants.

Route B is a *line-spring* idealisation: it treats the cracked chord wall as a
locally weakened strip rather than resolving the shell. It captures the right
scaling in ``a/T`` and crack length but not the chord ovalisation coupling that
a shell model would. It should be expected to under-predict the LJF change, and
therefore to make C2's damage signature *smaller* and C3's verdict *harsher*.
That direction is the conservative one for a claim that says detection works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..numerics import BilinearGrid
from ..provenance import Citation, DataUnavailable, Quantity, derived
from .newman_raju import NEWMAN_RAJU, boundary_correction_F, shape_factor_Q

__all__ = [
    "TADA_PARIS",
    "CrackGeometry",
    "crack_compliance",
    "ShellFESurface",
    "load_shell_fe_surface",
    "shell_fe_status",
]

TADA_PARIS = Citation(
    document=(
        "Tada, H., Paris, P.C. & Irwin, G.R., The Stress Analysis of Cracks Handbook; "
        "compliance-from-SIF applied to cracked members by Dimarogonas, A.D., 'Vibration of "
        "cracked structures: a state of the art review', Engineering Fracture Mechanics "
        "55(5):831-857"
    ),
    locator="Castigliano/Irwin compliance integral, dC = (2/E') int k^2 dA",
    year=1996,
)


@dataclass(frozen=True)
class CrackGeometry:
    """A semi-elliptical surface crack in the chord wall at the weld toe.

    ``a`` and ``c`` are kept as independent quantities throughout: a deep narrow
    flaw and a shallow long one with the same "percent through-wall" behave
    entirely differently, and collapsing them to one number is exactly the
    simplification this application refuses to make.
    """

    a: float  # depth, m
    c: float  # surface half-length, m
    T: float  # wall thickness, m

    @property
    def a_over_T(self) -> float:
        return self.a / self.T

    @property
    def surface_length(self) -> float:
        """``2c``, the full surface length, m."""
        return 2.0 * self.c

    @property
    def aspect(self) -> float:
        return self.a / max(self.c, 1e-12)

    def validate(self) -> list[str]:
        out = []
        if not 0.0 < self.a_over_T < 1.0:
            out.append(f"a/T = {self.a_over_T:.3f} outside (0, 1); a through-wall crack is a different problem")
        if self.aspect > 1.0:
            out.append(f"a/c = {self.aspect:.2f} > 1; Newman-Raju F is fitted for a/c <= 1")
        return out


def crack_compliance(
    crack: CrackGeometry,
    load_width: float,
    E: float = 2.10e11,
    nu: float = 0.3,
    n_steps: int = 200,
) -> tuple[float, Quantity]:
    """Additional axial compliance from a surface crack, m/N.

    The cracked region is idealised as a strip of the chord wall of width
    ``load_width`` (taken as the brace footprint) and thickness ``T`` carrying
    the local load ``P``, so the membrane stress is ``sigma = P/(w T)`` and the
    SIF per unit load is

    .. math:: k(\\alpha) = \\frac{F(\\alpha)}{wT} \\sqrt{\\frac{\\pi\\alpha}{Q}}

    Integrating over the growth of a semi-ellipse of fixed aspect ratio, whose
    area is ``A = (pi/2) a c`` so ``dA/da = (pi/2)(c/a) \\cdot 2a = pi c``:

    .. math::
        \\Delta C = \\frac{2}{E'} \\int_0^a k(\\alpha)^2 \\, \\pi c \\frac{\\alpha}{a}
                    \\, d\\alpha

    with the ``alpha/a`` factor carrying the self-similar growth of the ellipse
    half-length alongside the depth. The integral is evaluated at the deepest
    point (``phi = pi/2``), which dominates the compliance for the aspect ratios
    fatigue cracks actually take.

    Returns the compliance and a provenance-carrying quantity describing it.
    """
    problems = crack.validate()
    if crack.aspect > 1.0:
        raise ValueError(
            f"a/c = {crack.aspect:.2f} exceeds 1, outside the range the Newman-Raju "
            "boundary-correction factor is fitted for. Its M3 term diverges beyond that "
            "point, so evaluating here would return a large but meaningless compliance. "
            "Either restrict the crack grid to 2c >= 2a, or implement the deep-crack "
            "(a/c > 1) equation set."
        )
    E_prime = E / (1.0 - nu**2)
    w = max(load_width, 1e-6)
    alpha = np.linspace(1e-6, crack.a, n_steps)
    ratio = crack.aspect
    at = np.clip(alpha / crack.T, 0.0, 0.999)
    Q = shape_factor_Q(np.full_like(alpha, ratio))
    F = boundary_correction_F(np.full_like(alpha, ratio), at, np.pi / 2)
    k = F * np.sqrt(np.pi * alpha / Q) / (w * crack.T)
    integrand = k**2 * np.pi * crack.c * (alpha / max(crack.a, 1e-12))
    dC = float(2.0 / E_prime * np.trapezoid(integrand, alpha))

    q = derived(
        dC,
        "m/N",
        "crack-induced axial compliance",
        [],
        (
            f"Castigliano/Irwin compliance integral over a semi-elliptical surface crack, "
            f"a/T={crack.a_over_T:.3f}, 2c={crack.surface_length * 1e3:.0f} mm, "
            f"load width {w * 1e3:.0f} mm; Newman-Raju SIF"
        ),
        citation=TADA_PARIS,
        note=(
            "line-spring idealisation: no chord ovalisation coupling, so this "
            "under-predicts the true LJF change"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )
    return dC, q


# ------------------------------------------------------- shell-FE surface


@dataclass(frozen=True)
class ShellFESurface:
    """A precomputed crack-to-LJF surface with its provenance manifest."""

    a_over_T: np.ndarray
    surface_length_m: np.ndarray
    compliance_ratio: np.ndarray  # (n_aT, n_2c), C_cracked / C_intact
    manifest: dict
    digest: str

    def interpolator(self) -> BilinearGrid:
        return BilinearGrid(self.a_over_T, self.surface_length_m, self.compliance_ratio)

    def provenance_card(self) -> dict:
        m = self.manifest
        return {
            "element type": m.get("element_type", "?"),
            "mesh size at crack front": m.get("mesh_size_mm", "?"),
            "mesh convergence study": m.get("convergence_study", "?"),
            "validation": m.get("validation", "?"),
            "generated": m.get("generated", "?"),
            "surface hash": self.digest,
        }


def shell_fe_status(root: Path | None = None) -> tuple[bool, str]:
    """Whether a shell-FE surface is available, and what is missing if not."""
    d = root or Path(__file__).resolve().parents[2] / "data" / "shell_fe_surface"
    manifest = d / "manifest.json"
    if not manifest.is_file():
        return False, (
            "No shell-FE crack-to-LJF surface is shipped. Generating one requires a 3D "
            "shell FE code, a mesh convergence study, and digitised experimental data "
            "from Soh (2000) and Rhee (2005) for validation. Until one is generated "
            "offline and placed in data/shell_fe_surface/, C2 uses the line-spring "
            "compliance route, which is documented to under-predict the LJF change."
        )
    return True, f"Shell-FE surface present at {d}."


def load_shell_fe_surface(root: Path | None = None) -> ShellFESurface:
    """Load the shipped surface.

    Raises
    ------
    DataUnavailable
        If no surface is shipped. The caller must fall back to
        :func:`crack_compliance` and say so, or mark the claim untestable.
    """
    import hashlib

    d = root or Path(__file__).resolve().parents[2] / "data" / "shell_fe_surface"
    ok, why = shell_fe_status(d)
    if not ok:
        raise DataUnavailable(
            "Shell-FE crack-to-LJF surface",
            why,
            "Generate offline with scripts/generate_shell_fe_surface.py against a shell FE "
            "solver, then commit the table, its convergence study and its manifest.",
        )
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    table = d / manifest["table"]
    raw = table.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    if manifest.get("sha256", "").startswith(digest[:8]) is False and "sha256" in manifest:
        raise DataUnavailable(
            "Shell-FE crack-to-LJF surface",
            f"{table.name} hash {digest} does not match manifest",
            "restore the table or regenerate the manifest",
        )
    import pandas as pd

    df = pd.read_parquet(table) if table.suffix == ".parquet" else pd.read_csv(table)
    a_over_T = np.unique(df["a_over_T"].to_numpy(float))
    lengths = np.unique(df["surface_length_m"].to_numpy(float))
    grid = (
        df.pivot(index="a_over_T", columns="surface_length_m", values="compliance_ratio")
        .to_numpy(float)
    )
    return ShellFESurface(a_over_T, lengths, grid, manifest, digest)
