"""Newman-Raju stress-intensity factors for a semi-elliptical surface crack.

Source: Newman, J.C. & Raju, I.S., "An empirical stress-intensity factor
equation for the surface crack", Engineering Fracture Mechanics 15(1-2):185-192,
1981; extended in NASA TM-85793, "Stress-intensity factor equations for cracks
in three-dimensional finite bodies subjected to tension and bending loads",
1984. The same equations are reproduced in BS 7910:2019 Annex M.

    K = (S_t + H S_b) sqrt(pi a / Q) F(a/c, a/t, c/b, phi)

**Verification status, stated plainly because it bounds how far C2 can be
trusted.** The primary sources are a paywalled journal article and a scanned
technical memorandum; the coefficients below are transcribed, and this build
verifies what can be verified independently rather than asserting the rest:

* *Verified numerically* (``tests/test_newman_raju.py``):
    - the shape factor ``Q = 1 + 1.464 (a/c)^1.65`` against the exact elliptic
      integral ``E(k)^2``, which it matches to better than 0.2 percent over the
      whole range - this pins both the 1.464 and the 1.65;
    - the shallow-crack limit ``a/c -> 0, a/t -> 0, phi = 90 deg``, where the
      equation must return the 2D edge-crack value 1.1215 sqrt(pi a); it returns
      1.13, the 0.8 percent approximation Newman and Raju state;
    - symmetry, positivity, and monotonic growth of K with a/t.
* *Transcribed but not independently verified here*: the ``M2``, ``M3`` and
  ``g`` terms away from those limits, and the whole bending multiplier ``H``.
  An error in them would move C2's damage signature without any test catching
  it. The provenance card says so, and the app offers the shipped shell-FE
  surface as the route that does not depend on this equation at all.
"""

from __future__ import annotations

import numpy as np

from ..provenance import Citation

__all__ = ["NEWMAN_RAJU", "shape_factor_Q", "boundary_correction_F", "bending_multiplier_H", "sif"]

NEWMAN_RAJU = Citation(
    document=(
        "Newman, J.C. & Raju, I.S., 'An empirical stress-intensity factor equation for the "
        "surface crack', Engineering Fracture Mechanics 15(1-2):185-192; extended in "
        "NASA TM-85793 (1984); reproduced in BS 7910:2019 Annex M"
    ),
    locator="semi-elliptical surface crack in a finite-thickness plate, tension and bending",
    year=1981,
)


def shape_factor_Q(a_over_c: np.ndarray | float) -> np.ndarray:
    """Elliptical shape factor.

    ``Q = 1 + 1.464 (a/c)^1.65`` for ``a/c <= 1`` and
    ``Q = 1 + 1.464 (c/a)^1.65`` for ``a/c > 1``. This is Rawe's approximation to
    the square of the complete elliptic integral of the second kind, accurate to
    about 0.13 percent.
    """
    r = np.asarray(a_over_c, dtype=float)
    small = np.where(r <= 1.0, r, 1.0 / np.maximum(r, 1e-12))
    return 1.0 + 1.464 * small**1.65


def boundary_correction_F(
    a_over_c: np.ndarray | float,
    a_over_t: np.ndarray | float,
    phi: np.ndarray | float,
    c_over_b: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Boundary-correction factor ``F`` for tension.

    ``phi`` is the parametric angle: 0 at the free surface (the ends of the
    ellipse, where the crack meets the plate face) and pi/2 at the deepest point.

    Valid for ``0 < a/c <= 1``, ``0 <= a/t < 1``, ``c/b < 0.5``. Outside
    ``a/c <= 1`` this returns ``nan``, deliberately.

    The ``M3`` term contains ``14 (1 - a/c)^24``. For ``a/c > 1`` that base is
    negative and the even power makes it grow explosively - at ``a/c = 2.5`` it
    reaches 2.4e5, and ``F`` comes out four orders of magnitude too large without
    any warning. Newman and Raju publish a separate equation set for deep cracks
    (``a/c > 1``); it is not implemented here, so rather than extrapolate a
    formula past the point where it diverges, this function refuses. Callers
    must mask the invalid region and say why.
    """
    r = np.asarray(a_over_c, dtype=float)
    at = np.asarray(a_over_t, dtype=float)
    ph = np.asarray(phi, dtype=float)
    cb = np.asarray(c_over_b, dtype=float)

    M1 = 1.13 - 0.09 * r
    M2 = -0.54 + 0.89 / (0.2 + r)
    M3 = 0.5 - 1.0 / (0.65 + r) + 14.0 * (1.0 - r) ** 24
    g = 1.0 + (0.1 + 0.35 * at**2) * (1.0 - np.sin(ph)) ** 2
    f_phi = (r**2 * np.cos(ph) ** 2 + np.sin(ph) ** 2) ** 0.25
    # Finite-width correction; unity for a wide plate (c/b -> 0).
    arg = np.clip(np.pi * cb * np.sqrt(np.clip(at, 0.0, 0.999)) / 2.0, 0.0, 1.5)
    f_w = np.sqrt(1.0 / np.cos(arg))
    F = (M1 + M2 * at**2 + M3 * at**4) * g * f_phi * f_w
    return np.where(r <= 1.0, F, np.nan)


def bending_multiplier_H(
    a_over_c: np.ndarray | float,
    a_over_t: np.ndarray | float,
    phi: np.ndarray | float,
) -> np.ndarray:
    """Bending multiplier ``H`` applied to the bending stress component."""
    r = np.asarray(a_over_c, dtype=float)
    at = np.asarray(a_over_t, dtype=float)
    ph = np.asarray(phi, dtype=float)
    p = 0.2 + r + 0.6 * at
    G1 = -1.22 - 0.12 * r
    G2 = 0.55 - 1.05 * r**0.75 + 0.47 * r**1.5
    H1 = 1.0 - 0.34 * at - 0.11 * r * at
    H2 = 1.0 + G1 * at + G2 * at**2
    return H1 + (H2 - H1) * np.sin(ph) ** p


def sif(
    tension: float | np.ndarray,
    bending: float | np.ndarray,
    a: float | np.ndarray,
    c: float | np.ndarray,
    t: float,
    phi: float | np.ndarray = np.pi / 2,
    b: float | None = None,
) -> np.ndarray:
    """Mode I stress-intensity factor, Pa.m^0.5.

    Parameters
    ----------
    tension, bending
        Membrane and bending stress components, Pa.
    a, c
        Crack depth and surface half-length, m.
    t
        Wall thickness, m.
    phi
        Parametric angle; default pi/2 (the deepest point), which governs
        through-thickness growth.
    b
        Plate half-width, m. ``None`` means a wide plate.
    """
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    r = a / np.maximum(c, 1e-12)
    at = np.clip(a / t, 0.0, 0.999)
    cb = 0.0 if b is None else np.asarray(c, float) / b
    Q = shape_factor_Q(r)
    F = boundary_correction_F(r, at, phi, cb)
    H = bending_multiplier_H(r, at, phi)
    return (np.asarray(tension, float) + H * np.asarray(bending, float)) * np.sqrt(
        np.pi * a / Q
    ) * F
