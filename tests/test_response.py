"""Verification of the precomputed strain response surface.

The response surface is the load-bearing optimisation in this application: C3
evaluates it thousands of times instead of solving the frame, so if it does not
reproduce a direct solve then every claim downstream of it is wrong by an
unknown amount. These tests check it against direct solves at points that are
deliberately *not* on the grid, and check the two symmetries the physics
requires.
"""

from __future__ import annotations

import numpy as np
import pytest

from tidetwin.fe.ljf import LJFModel
from tidetwin.geometry.oc4 import WATER_DEPTH, build_jacket, load_tables, sensor_pair
from tidetwin.loads.buoyancy import assemble_buoyancy_load, wetted_fraction
from tidetwin.loads.morison import (
    HydroConfig,
    assemble_hydrodynamic_load,
    current_profile_factor,
    drag_inertia_coefficients,
    member_distributed_force,
)
from tidetwin.response import build_response_surface

CFG = HydroConfig(water_depth=WATER_DEPTH, roughness_m=0.05)


def _s(x) -> float:
    """Scalar from a length-1 array. numpy 2 refuses float() on non-0-d arrays."""
    return float(np.asarray(x).item())


@pytest.fixture(scope="module")
def rig():
    tables = load_tables()
    pair = sensor_pair(tables, 5, offset_m=1.5)
    build = build_jacket(ljf_model=LJFModel.RIGID, tables=tables)
    surface = build_response_surface(
        build, pair, CFG, n_theta=36, eta_levels=np.linspace(-3.0, 3.0, 7)
    )
    return build, pair, surface


def _direct_solve(build, pair, speed: float, direction: float, eta: float):
    """Strain at the gauge pair from a full frame solve, no surface involved."""
    cfg = HydroConfig(
        water_density=CFG.water_density,
        roughness_m=CFG.roughness_m,
        current_exponent=CFG.current_exponent,
        water_depth=CFG.water_depth,
        surface_elevation=eta,
        marine_growth_mm=CFG.marine_growth_mm,
    )
    d = np.array([np.cos(direction), np.sin(direction), 0.0])

    def vel(z, _p):
        f = float(
            current_profile_factor(z, cfg.water_depth, cfg.surface_elevation, cfg.current_exponent)
        )
        return d * speed * f

    f_drag = assemble_hydrodynamic_load(build, vel, cfg)
    f_buoy = assemble_buoyancy_load(build, eta, cfg.water_density, flooded=True)
    f_ref = assemble_buoyancy_load(build, 0.0, cfg.water_density, flooded=True)
    K, _ = build.model.assemble()
    u = build.model.solve_static(f_drag + f_buoy, K).displacements.reshape(-1)
    u_ref = build.model.solve_static(f_ref, K).displacements.reshape(-1)
    hi, lo = build.pair_strains(u, pair)
    hi0, lo0 = build.pair_strains(u_ref, pair)
    return hi - hi0, lo - lo0


@pytest.mark.parametrize(
    "speed,direction_deg,eta",
    [
        (0.25, 17.0, 0.0),
        (0.40, 103.0, 1.4),
        (0.15, 249.0, -1.1),
        (0.33, 331.0, 2.2),
    ],
)
def test_surface_reproduces_a_direct_solve_off_grid(rig, speed, direction_deg, eta):
    """Interpolated response must match a full frame solve at off-grid points.

    The grid has 10 degree spacing in direction and 1 m in water level; these
    query points sit between nodes on both axes, so any interpolation error
    shows up here.
    """
    build, pair, surface = rig
    direction = np.radians(direction_deg)
    got_u, got_l = surface.evaluate(
        np.array([speed]), np.array([direction]), np.array([eta])
    )
    exp_u, exp_l = _direct_solve(build, pair, speed, direction, eta)

    scale = max(abs(exp_u), abs(exp_l), 1e-12)
    assert abs(_s(got_u) - exp_u) / scale < 0.02
    assert abs(_s(got_l) - exp_l) / scale < 0.02
    # The ratio is what the claims key on, so it gets a tighter bound.
    if abs(exp_u) > 1e-15:
        assert _s(got_l) / _s(got_u) == pytest.approx(exp_l / exp_u, rel=0.02)


def test_surface_is_exact_on_its_own_grid_nodes(rig):
    """At a node the interpolation must be the stored value, not near it."""
    build, pair, surface = rig
    i, j = 7, 4
    theta = float(surface.theta[i])
    eta = float(surface.eta[j])
    got_u, _got_l = surface.evaluate(np.array([1.0]), np.array([theta]), np.array([eta]))
    expected = surface.drag_upper[i, j] + surface.buoy_upper[j] - surface.still_upper
    assert _s(got_u) == pytest.approx(expected, rel=1e-12)


def test_reversing_the_current_negates_the_drag_response(rig):
    """eps(theta + pi) = -eps(theta) exactly: |u|u is odd and the frame is linear.

    The surface exploits this to halve its solve count, so it has to hold.
    """
    build, pair, surface = rig
    for theta in (0.3, 1.9, 4.4):
        a_u, a_l = surface.evaluate(np.array([0.3]), np.array([theta]), np.array([0.0]))
        b_u, b_l = surface.evaluate(
            np.array([0.3]), np.array([theta + np.pi]), np.array([0.0])
        )
        # Buoyancy at eta = 0 is the reference state, so only drag remains.
        assert _s(a_u) == pytest.approx(-_s(b_u), rel=1e-9, abs=1e-18)
        assert _s(a_l) == pytest.approx(-_s(b_l), rel=1e-9, abs=1e-18)


def test_drag_response_scales_as_speed_squared(rig):
    """Morison drag is quadratic, which is what lets the surface store one speed."""
    _build, _pair, surface = rig
    one, _ = surface.evaluate(np.array([1.0]), np.array([0.7]), np.array([0.0]))
    two, _ = surface.evaluate(np.array([2.0]), np.array([0.7]), np.array([0.0]))
    half, _ = surface.evaluate(np.array([0.5]), np.array([0.7]), np.array([0.0]))
    assert _s(two) == pytest.approx(4.0 * _s(one), rel=1e-9)
    assert _s(half) == pytest.approx(0.25 * _s(one), rel=1e-9)


def test_zero_current_at_reference_level_gives_zero_strain(rig):
    """The reference state is subtracted, so it must return exactly zero."""
    _build, _pair, surface = rig
    u, l = surface.evaluate(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert _s(u) == pytest.approx(0.0, abs=1e-18)
    assert _s(l) == pytest.approx(0.0, abs=1e-18)


def test_water_level_outside_the_grid_is_clamped_not_extrapolated(rig):
    """A surface must never invent behaviour beyond the range it was built over."""
    _build, _pair, surface = rig
    edge, _ = surface.evaluate(
        np.array([0.3]), np.array([1.0]), np.array([surface.eta[-1]])
    )
    beyond, _ = surface.evaluate(np.array([0.3]), np.array([1.0]), np.array([99.0]))
    assert _s(beyond) == pytest.approx(_s(edge), rel=1e-12)


def test_solve_count_uses_the_half_plane_symmetry(rig):
    """n_theta/2 drag solves plus one buoyancy solve per water level."""
    _build, _pair, surface = rig
    n_theta, n_eta = surface.theta.size, surface.eta.size
    assert surface.n_solves == n_eta * (n_theta // 2 + 1)


# ---------------------------------------------------------------- load layer


def test_wetted_fraction_covers_the_splash_zone():
    assert wetted_fraction(-10.0, -5.0, 0.0) == 1.0
    assert wetted_fraction(5.0, 10.0, 0.0) == 0.0
    assert wetted_fraction(-1.0, 1.0, 0.0) == pytest.approx(0.5)
    assert wetted_fraction(-1.0, 1.0, 0.5) == pytest.approx(0.75)
    # Order of the endpoints must not matter.
    assert wetted_fraction(1.0, -1.0, 0.0) == pytest.approx(0.5)


def test_current_profile_is_zero_above_the_surface_and_one_at_it():
    d = 50.0
    assert float(current_profile_factor(0.0, d)) == pytest.approx(1.0)
    assert float(current_profile_factor(5.0, d)) == 0.0
    assert float(current_profile_factor(-d, d)) == pytest.approx(0.0)
    # Monotonic increase from seabed to surface.
    z = np.linspace(-d, 0.0, 40)
    f = current_profile_factor(z, d)
    assert np.all(np.diff(f) >= -1e-12)


def test_drag_force_is_quadratic_and_aligned_with_the_flow():
    pi_, pj = np.array([0.0, 0.0, -10.0]), np.array([0.0, 0.0, 0.0])  # vertical member
    u = np.array([1.0, 0.0, 0.0])
    f1 = member_distributed_force(pi_, pj, 1.2, u, np.zeros(3), CFG)
    f2 = member_distributed_force(pi_, pj, 1.2, 2.0 * u, np.zeros(3), CFG)
    assert np.allclose(f2, 4.0 * f1, rtol=1e-12)
    # Force is along the flow, and a vertical member feels the full horizontal flow.
    assert f1[0] > 0 and abs(f1[1]) < 1e-12 and abs(f1[2]) < 1e-12


def test_flow_along_a_member_axis_produces_no_normal_force():
    """Only the velocity component normal to the axis loads a tubular."""
    pi_, pj = np.array([0.0, 0.0, -10.0]), np.array([0.0, 0.0, 0.0])
    axial = np.array([0.0, 0.0, 1.0])
    f = member_distributed_force(pi_, pj, 1.2, axial, np.zeros(3), CFG)
    assert np.allclose(f, 0.0, atol=1e-12)


def test_roughness_moves_cd_between_the_api_smooth_and_rough_values():
    cd_s, cm_s, _ = drag_inertia_coefficients(1.2, 0.0)
    cd_r, cm_r, _ = drag_inertia_coefficients(1.2, 0.5)
    assert (cd_s, cm_s) == (0.65, 1.6)
    assert (cd_r, cm_r) == (1.05, 1.2)
    cd_m, cm_m, note = drag_inertia_coefficients(1.2, 1.2e-3)
    assert 0.65 < cd_m < 1.05 and 1.2 < cm_m < 1.6
    assert "transitional" in note
    with pytest.raises(ValueError):
        drag_inertia_coefficients(0.0, 0.05)


def test_shared_solve_matches_separate_builds_exactly():
    """The rosette optimisation: one frame solve feeds every gauge angle.

    build_response_surfaces must return, for each pair, exactly what
    build_response_surface would have returned building it alone - the frame
    solve does not depend on where strain is read, so sharing it changes
    nothing but the cost. A single digit of difference would mean the shared
    solve had corrupted one pair's recovery.
    """
    from tidetwin.rosette import ROSETTE_ANGLES_DEG
    from tidetwin.response import build_response_surfaces

    tables = load_tables()
    build = build_jacket(ljf_model=LJFModel.RIGID, tables=tables)
    eta = np.linspace(-2.0, 2.0, 5)
    pairs = [sensor_pair(tables, 5, 1.5, np.radians(a)) for a in ROSETTE_ANGLES_DEG]

    separate = [build_response_surface(build, p, CFG, n_theta=16, eta_levels=eta) for p in pairs]
    shared = build_response_surfaces(build, pairs, CFG, n_theta=16, eta_levels=eta)

    for sep, sh in zip(separate, shared):
        assert np.array_equal(sep.drag_upper, sh.drag_upper)
        assert np.array_equal(sep.drag_lower, sh.drag_lower)
        assert np.array_equal(sep.buoy_upper, sh.buoy_upper)
        assert np.array_equal(sep.buoy_lower, sh.buoy_lower)
        assert sep.still_upper == sh.still_upper

    # Four gauges, but the solve count is that of a single pair - the whole point.
    assert shared[0].n_solves == separate[0].n_solves


def test_shared_build_rejects_an_empty_pair_list():
    from tidetwin.response import build_response_surfaces

    tables = load_tables()
    build = build_jacket(ljf_model=LJFModel.RIGID, tables=tables)
    with pytest.raises(ValueError, match="at least one"):
        build_response_surfaces(build, [], CFG, n_theta=8)
