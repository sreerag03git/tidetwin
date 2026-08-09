"""The single configuration the app opens on.

The sidebar's default widget values and the precomputed bundle must describe the
same run. If they drift, a fresh visitor either recomputes for nothing (defeating
the precompute that keeps the deployed app off the CPU throttle) or sees results
that do not match the sidebar. Both read their defaults from here.

No Streamlit import: this must be callable from the plain precompute script and
from the tests, not only from inside a running app.
"""

from __future__ import annotations

from .analysis import AnalysisConfig
from .economics.npv import EconomicInputs
from .fe.ljf import LJFModel
from .nuisance import NuisanceRanges

__all__ = ["default_config", "default_tide_station"]


def default_tide_station() -> str | None:
    """The first cached NOAA station, which the sidebar selects by default."""
    from .loads.noaa import available_cached

    cached = available_cached()
    return cached[0].current_id if cached else None


def default_config() -> AnalysisConfig:
    """The configuration a visitor sees before touching any control.

    These values are the sidebar widget defaults. The nuisance ranges are
    ``NuisanceRanges()`` - in particular the FBG drift and noise are the paper's
    own 0.05 microstrain, not a harsher figure - so the app's default C3 matches
    the exported ledger rather than being quietly more pessimistic than it.
    """
    return AnalysisConfig(
        latitude=24.9,
        longitude=53.2,
        joint_id=5,
        sensor_offset_m=1.5,
        sensor_theta_deg=0.0,
        ljf_model=LJFModel.SHELL,
        measurement_mode="single",
        roughness_m=50.0e-3,
        marine_growth_mm=0.0,
        record_days=30.0,
        crack_a_over_T=0.50,
        crack_2c_m=100.0e-3,
        n_mc_samples=100,
        n_theta=24,
        seed=20260728,
        tide_station=default_tide_station(),
        ranges=NuisanceRanges(),
        economics=EconomicInputs(),
    )
