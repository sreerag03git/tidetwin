"""C3 - the deciding test.

The machinery lives in :mod:`tidetwin.nuisance`, which is imported by the
response layer as well; this module is the claim-facing entry point so that
``claims/tests/`` carries one module per claim as the brief specifies.

Two verdicts are produced and both are reported:

``verdict``
    Against the damage signature this application computes from the line-spring
    crack model. That model is documented to under-predict, so this comparison
    is the harsher of the two.

``verdict_against_claimed_signature``
    Against the 11.1 percent the abstract asserts. This is the robust one: if
    the nuisance floor exceeds a third of the signal strength the paper claims
    for itself, the conclusion does not depend on any modelling choice made
    here.
"""

from __future__ import annotations

from ...nuisance import (
    CHANNEL_LABELS,
    CHANNELS,
    RANDOM_CHANNELS,
    SYSTEMATIC_CHANNELS,
    NuisanceRanges,
    NuisanceResult,
    ratio_from_series,
    run_nuisance_budget,
    verdict,
    verdict_against_claimed_signature,
)

__all__ = [
    "CHANNELS",
    "CHANNEL_LABELS",
    "RANDOM_CHANNELS",
    "SYSTEMATIC_CHANNELS",
    "NuisanceRanges",
    "NuisanceResult",
    "ratio_from_series",
    "run_nuisance_budget",
    "verdict",
    "verdict_against_claimed_signature",
    "CLAIMED_DAMAGE_SIGNATURE",
]

#: The damage signature the abstract asserts, used only as the right-hand side
#: of a comparison. It is the hypothesis under test, never a computed result,
#: and the UI labels it as such wherever it appears.
CLAIMED_DAMAGE_SIGNATURE = 0.111
