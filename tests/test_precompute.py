"""The precomputed bundle must stay in step with the source it was built from.

The deployed app opens on this bundle instead of computing, so if it drifts out
of date the app silently falls back to computing on load - the exact CPU-throttle
behaviour the bundle exists to prevent. These tests make that drift a red build
rather than a slow deploy: if any solver source changed without the bundle being
regenerated (python scripts/precompute_default.py), the fingerprint no longer
matches and this fails.
"""

from __future__ import annotations

import numpy as np

from tidetwin.appdefaults import default_config
from tidetwin.claims.registry import Artifacts
from tidetwin.precompute import BUNDLE_PATH, load_bundle, source_fingerprint


def test_the_committed_bundle_exists_and_is_current():
    assert BUNDLE_PATH.is_file(), (
        "no precomputed bundle committed; run: python scripts/precompute_default.py"
    )
    bundle = load_bundle()
    assert bundle is not None, (
        "the committed bundle is stale or unloadable - its fingerprint does not match the "
        "current tidetwin source. Regenerate it: python scripts/precompute_default.py"
    )


def test_the_bundle_carries_a_full_result_for_the_default_config():
    bundle = load_bundle()
    assert bundle is not None
    assert isinstance(bundle["full"], Artifacts)
    # The pieces the app reads straight out of the bundle.
    assert bundle["full"].c3 is not None, "the bundle must carry the deciding C3 result"
    ljf_rows, joint_rows = bundle["sensitivity"]
    assert len(joint_rows) > 0, "the joint-sensitivity sweep must be present"
    assert bundle["cycle"] is not None, "the tidal-cycle simulation must be present"


def test_the_bundle_config_is_the_documented_default():
    """It must match default_config, or a fresh visitor's sidebar will not match
    the bundle and the app will prompt for a run it does not need."""
    bundle = load_bundle()
    assert bundle is not None
    a, b = bundle["cfg"], default_config()
    assert a.joint_id == b.joint_id
    assert a.n_mc_samples == b.n_mc_samples
    assert a.measurement_mode == b.measurement_mode
    assert a.ranges.fbg_drift_sd_ustrain == b.ranges.fbg_drift_sd_ustrain
    assert a.tide_station == b.tide_station


def test_the_fingerprint_is_content_based_not_mtime_based():
    """It must survive a checkout: two reads of unchanged files agree, and it is
    a hex digest, not a timestamp."""
    f = source_fingerprint()
    assert f == source_fingerprint()
    assert f != "unavailable"
    int(f, 16)  # a hex digest


def test_the_default_app_config_matches_the_ledger_fbg_spec():
    """The app's default nuisance ranges must be the paper's own FBG figures, so
    the interactive C3 is not quietly harsher than the exported ledger."""
    from tidetwin.nuisance import NuisanceRanges

    r = default_config().ranges
    assert r.fbg_drift_sd_ustrain == NuisanceRanges().fbg_drift_sd_ustrain == 0.05
    assert r.fbg_noise_ustrain == NuisanceRanges().fbg_noise_ustrain == 0.05


def test_the_bundle_c3_matches_a_fresh_run_of_the_same_config():
    """The precomputed C3 dispersion must equal what run_full produces now - the
    bundle is a cache of the real computation, not a different one."""
    from tidetwin.analysis import run_full

    bundle = load_bundle()
    assert bundle is not None
    fresh = run_full(default_config())
    assert fresh.c3.joint_cv == np.float64(bundle["full"].c3.joint_cv) or (
        abs(fresh.c3.joint_cv - bundle["full"].c3.joint_cv) < 1e-12
    )
