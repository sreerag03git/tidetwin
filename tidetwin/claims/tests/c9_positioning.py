"""C9 - positioning against the incumbent inspection methods.

POD(a) for the tidal-strain method is computed from the detection model, so it is
DERIVED and moves with whatever C2 and C3 produce.

Competitor curves - ROV magnetic particle inspection, ACFM, and flooded member
detection - are **not** shipped. Published POD curves for these exist in the
offshore inspection literature and in operator qualification datasets, but
digitising them requires the source figures, and inventing plausible ``a90/95``
values to compare against would fabricate exactly the comparison the claim rests
on. Supply them as CSV in ``data/pod/`` and the overlay activates.

The part-through versus through-wall distinction is kept explicit throughout,
because it is where the comparison is usually smudged: flooded member detection
is the real incumbent competitor on cost, but it can only respond *after* a crack
has gone through-wall and the member has flooded. It and a part-through method
are not substitutes, and a POD curve that plots them on one axis without saying
so is misleading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...provenance import DataUnavailable
from ...signal.detect import DetectionModel, pod_curve

__all__ = ["PODResult", "tidal_pod", "load_competitor_pod", "competitor_status"]

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "pod"

SCHEMA = (
    "CSV with columns: method,crack_depth_mm,pod   (one row per point on the curve). "
    "Add a companion <method>.json with {'citation': {...}, 'detects': 'part-through' "
    "or 'through-wall', 'conditions': '...'}"
)


@dataclass
class PODResult:
    crack_depth_m: np.ndarray
    pod: np.ndarray
    a90: float
    a90_95: float
    detects: str
    competitor_curves: dict[str, pd.DataFrame]
    competitor_note: str

    @property
    def reaches_90_percent(self) -> bool:
        return bool(np.isfinite(self.a90))


def competitor_status(root: Path | None = None) -> tuple[bool, str]:
    d = root or DATA_DIR
    files = list(d.glob("*.csv")) if d.is_dir() else []
    if files:
        return True, f"{len(files)} competitor POD curve(s) found in {d}."
    return False, (
        "DATA UNAVAILABLE - no published POD curves shipped for ROV MPI, ACFM or flooded "
        f"member detection. Digitise them from the source figures into {d}. {SCHEMA}"
    )


def load_competitor_pod(root: Path | None = None) -> dict[str, pd.DataFrame]:
    ok, why = competitor_status(root)
    if not ok:
        raise DataUnavailable("Competitor POD curves", why, SCHEMA)
    d = root or DATA_DIR
    return {p.stem: pd.read_csv(p) for p in sorted(d.glob("*.csv"))}


def tidal_pod(
    model: DetectionModel,
    signature_fraction_of: callable,
    crack_depth_m: np.ndarray | None = None,
    n_records: float = 1.0,
) -> PODResult:
    """POD(a) for the tidal-strain method.

    ``signature_fraction_of(a)`` returns the fractional strain-ratio change for a
    crack of depth ``a``, normally interpolated from the C2 grid.
    """
    a = (
        np.asarray(crack_depth_m, float)
        if crack_depth_m is not None
        else np.linspace(1e-3, 30e-3, 60)
    )
    sig = np.asarray([signature_fraction_of(float(x)) for x in a], float)
    pod, a90, a90_95 = pod_curve(a, sig, model, n_records=n_records)
    try:
        curves = load_competitor_pod()
        note = "Competitor curves loaded from data/pod/."
    except DataUnavailable as exc:
        curves = {}
        note = f"{exc}\n\nRemedy: {exc.remedy}"
    return PODResult(
        crack_depth_m=a,
        pod=pod,
        a90=a90,
        a90_95=a90_95,
        detects="part-through",
        competitor_curves=curves,
        competitor_note=note,
    )
