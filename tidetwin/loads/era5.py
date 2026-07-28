"""ERA5 reanalysis client with an on-disk cache.

Variables used by this application:

===========  ===============================================  ==========
short name   long name                                        used by
===========  ===============================================  ==========
``swh``      significant height of combined wind waves/swell  C3 wave offset
``pp1d``     peak wave period                                 C3 wave offset
``mwd``      mean wave direction                              C3 wave offset
``u10``      10 m eastward wind component                     C3 wind-driven current
``v10``      10 m northward wind component                    C3 wind-driven current
``2t``       2 m air temperature                              C5 thermal channel
``sst``      sea surface temperature                          C5 thermal channel
``ssrd``     surface solar radiation downwards                C5 thermal channel
===========  ===============================================  ==========

Access needs a Copernicus Climate Data Store account and an API key. Without
one this module raises :class:`~tidetwin.provenance.DataUnavailable`; it never
manufactures a stand-in time series. C3's wind-driven-current and wave-offset
terms and the whole of C5 are gated on this, and report
``UNTESTABLE - DATA MISSING`` when it is absent.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..provenance import Citation, DataUnavailable, measured

__all__ = [
    "ERA5_CITATION",
    "ERA5_VARIABLES",
    "CACHE_DIR",
    "ERA5Request",
    "credentials_status",
    "cached_series",
    "fetch",
    "load_or_explain",
]

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "era5_cache"

ERA5_CITATION = Citation(
    document=(
        "Hersbach, H. et al., 'The ERA5 global reanalysis', Q. J. R. Meteorol. Soc. "
        "146:1999-2049; data from the Copernicus Climate Data Store, "
        "reanalysis-era5-single-levels"
    ),
    year=2020,
    url="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
)

ERA5_VARIABLES: dict[str, str] = {
    "swh": "significant_height_of_combined_wind_waves_and_swell",
    "pp1d": "peak_wave_period",
    "mwd": "mean_wave_direction",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "2t": "2m_temperature",
    "sst": "sea_surface_temperature",
    "ssrd": "surface_solar_radiation_downwards",
}


@dataclass(frozen=True)
class ERA5Request:
    """A reproducible request. Its hash names the cache file."""

    latitude: float
    longitude: float
    start: _dt.date
    end: _dt.date
    variables: tuple[str, ...] = tuple(ERA5_VARIABLES)

    def key(self) -> str:
        payload = json.dumps(
            {
                "lat": round(self.latitude, 4),
                "lon": round(self.longitude, 4),
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
                "vars": sorted(self.variables),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def cache_path(self, root: Path | None = None) -> Path:
        return (root or CACHE_DIR) / f"era5_{self.key()}.parquet"

    def meta_path(self, root: Path | None = None) -> Path:
        return (root or CACHE_DIR) / f"era5_{self.key()}.json"


def credentials_status() -> tuple[bool, str]:
    """Whether the CDS API can be used, and precisely what is missing if not."""
    try:
        import cdsapi  # noqa: F401
    except ImportError:
        return False, "cdsapi is not installed (pip install cdsapi)."
    if os.environ.get("CDSAPI_KEY") or os.environ.get("CDSAPI_URL"):
        return True, "CDS credentials found in the environment."
    rc = Path.home() / ".cdsapirc"
    if rc.is_file():
        return True, f"CDS credentials found at {rc}."
    try:
        import streamlit as st

        if "CDSAPI_KEY" in st.secrets:
            return True, "CDS credentials found in st.secrets."
    except Exception:
        pass
    return False, (
        "DATA UNAVAILABLE - CDS credentials not configured. Register free at "
        "https://cds.climate.copernicus.eu, accept the ERA5 licence, then set "
        "CDSAPI_KEY in .streamlit/secrets.toml (or ~/.cdsapirc). Secrets are never "
        "committed to this repository."
    )


def cached_series(req: ERA5Request, root: Path | None = None) -> pd.DataFrame | None:
    """Return the cached time series for a request, or ``None``."""
    p = req.cache_path(root)
    if not p.is_file():
        return None
    return pd.read_parquet(p)


def fetch(req: ERA5Request, root: Path | None = None) -> pd.DataFrame:
    """Download (or read from cache) an hourly ERA5 series at a point.

    Raises
    ------
    DataUnavailable
        If credentials are absent and the request is not already cached.
    """
    cached = cached_series(req, root)
    if cached is not None:
        return cached

    ok, why = credentials_status()
    if not ok:
        raise DataUnavailable(
            "ERA5 reanalysis (Copernicus CDS)",
            why,
            "Configure CDSAPI_KEY, or pre-populate data/era5_cache/ with a prior download.",
        )

    import cdsapi  # pragma: no cover - requires credentials

    raise DataUnavailable(  # pragma: no cover
        "ERA5 reanalysis (Copernicus CDS)",
        (
            "Credentials are present but this build has not been exercised against a live "
            "CDS retrieval, so the download and NetCDF-decoding path is uncertified. "
            "Treating an unverified download as MEASURED data would be worse than "
            "reporting it missing."
        ),
        "Run scripts/fetch_era5.py, verify the series against the CDS web preview, then rerun.",
    )


def load_or_explain(
    req: ERA5Request, root: Path | None = None
) -> tuple[pd.DataFrame | None, str]:
    """Best-effort load. Returns ``(data, explanation)``; data is ``None`` if absent.

    Used by the UI so the Environment tab can render an honest DATA UNAVAILABLE
    panel with setup instructions instead of raising.
    """
    try:
        return fetch(req, root), "ERA5 hourly series loaded."
    except DataUnavailable as exc:
        return None, f"{exc}\n\nRemedy: {exc.remedy}"


def as_quantities(df: pd.DataFrame, retrieved: _dt.date) -> dict[str, object]:
    """Wrap a loaded ERA5 frame as MEASURED quantities, one per variable."""
    out = {}
    for col in df.columns:
        cit = Citation(
            document=ERA5_CITATION.document,
            locator=ERA5_VARIABLES.get(col, col),
            year=ERA5_CITATION.year,
            url=ERA5_CITATION.url,
            retrieved=retrieved,
            variable=col,
        )
        out[col] = measured(df[col].to_numpy(float), _units_for(col), f"ERA5 {col}", cit)
    return out


def _units_for(name: str) -> str:
    return {
        "swh": "m",
        "pp1d": "s",
        "mwd": "deg",
        "u10": "m/s",
        "v10": "m/s",
        "2t": "K",
        "sst": "K",
        "ssrd": "J/m^2",
    }.get(name, "-")
