"""No-update baseline: prior propagation with no assimilation.

This is the status quo the digital twin has to beat - a fatigue assessment made
at design time and never revised. It is included because "the filter converged"
means nothing without knowing what not filtering would have given, and because a
miscalibrated filter can easily be *worse* than this.

Two variants:

``prior_only``
    Propagate the initial crack-size ensemble through the growth model with no
    observations. Always available.

``sn_miner``
    The conventional S-N plus Miner's rule life estimate, which needs
    DNV-RP-C203 curve parameters. Those are in a paid standard and are not
    shipped, so this variant reports DATA UNAVAILABLE until the user supplies
    them - see :mod:`tidetwin.damage.sn`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..provenance import DataUnavailable

__all__ = ["prior_only", "sn_miner_baseline"]


def prior_only(
    a_mean: float,
    a_cv: float,
    growth: Callable[[np.ndarray, float], np.ndarray],
    dt: float,
    n_steps: int,
    n_members: int = 512,
    seed: int = 20260728,
) -> np.ndarray:
    """Ensemble trajectory with no assimilation. Returns ``(n_steps+1, n_members)``."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.log1p(a_cv**2))
    mu = np.log(a_mean) - 0.5 * sigma**2
    a = rng.lognormal(mu, sigma, size=n_members)
    out = [a.copy()]
    for _ in range(n_steps):
        a = np.maximum(growth(a, dt), 1e-12)
        out.append(a.copy())
    return np.array(out)


def sn_miner_baseline(*_args, **_kw):
    """Conventional S-N/Miner life estimate. Not available without curve data."""
    raise DataUnavailable(
        "DNV-RP-C203 S-N curve T parameters",
        "curve constants are not shipped; DNV-RP-C203 is a paid standard",
        "Enter log a1, m1, log a2, m2, the thickness exponent and t_ref from "
        "DNV-RP-C203 Table 2-2 (seawater with cathodic protection) into "
        "data/sn/dnv_rp_c203_T.json - see tidetwin.damage.sn.load_curve.",
    )
