"""OC4 geometry ingestion, strain recovery and local joint flexibility."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from tidetwin.fe.assemble import Member, Model
from tidetwin.fe.beam3d import tubular_section
from tidetwin.fe.ljf import (
    JointGeometry,
    LJFModel,
    joint_stiffness,
    load_tabulated,
    shell_ljf,
)
from tidetwin.fe.modal import eigenmodes
from tidetwin.geometry.oc4 import (
    BASE_JOINTS,
    MUDLINE_Z,
    WATER_DEPTH,
    brace_chord_joints,
    build_jacket,
    load_tables,
    sensor_pair,
)
from tidetwin.provenance import DataUnavailable

E = 210.0e9
NU = 0.3


# ------------------------------------------------------------ table integrity


def test_published_tables_load_with_expected_shape():
    t = load_tables()
    assert len(t.joints) == 64
    assert len(t.members) == 112
    assert len(t.sections) == 6
    # Published OC4 dimensions: 1.2 m legs, 0.8 m braces, 12 m mudline footprint.
    assert set(t.sections.outer_diameter_m.round(3)) == {0.8, 1.2, 2.082}
    assert t.joints.z_m.min() == pytest.approx(MUDLINE_Z)
    assert WATER_DEPTH == pytest.approx(50.001)
    footprint = t.joints.loc[list(BASE_JOINTS), ["x_m", "y_m"]].abs().to_numpy()
    assert np.allclose(footprint, 6.0)
    # Narrows to 8 m x 8 m at the transition piece.
    assert np.allclose(t.joints.loc[[53, 54, 55, 56], ["x_m", "y_m"]].abs().to_numpy(), 4.0)


def test_tampered_geometry_is_rejected(tmp_path):
    """A silently edited geometry table would invalidate every structural claim."""
    from tidetwin.geometry.oc4 import DATA_DIR

    for p in DATA_DIR.glob("*"):
        shutil.copy(p, tmp_path / p.name)
    target = tmp_path / "oc4_joints.csv"
    text = target.read_text(encoding="utf-8").replace("6.00000,6.00000,-45.50000", "6.10000,6.00000,-45.50000")
    target.write_text(text, encoding="utf-8")
    with pytest.raises(DataUnavailable, match="does not match manifest"):
        load_tables(str(tmp_path))


def test_missing_geometry_reports_data_unavailable(tmp_path):
    with pytest.raises(DataUnavailable, match="missing table"):
        load_tables(str(tmp_path))


def test_k_joints_are_the_four_legs_by_six_levels():
    t = load_tables()
    kj = brace_chord_joints(t)
    # Six braced levels on each of four legs.
    assert len(kj) == 24
    zs = sorted({round(float(t.joints.loc[j, "z_m"]), 3) for j in kj})
    assert zs == [-44.001, -43.127, -24.614, -8.922, 4.378, 15.651]
    # The mid-height joints are multiplanar KK joints carrying four braces.
    assert len(kj[5]) == 4
    assert len(kj[21]) == 4


# ---------------------------------------------------------- strain recovery


def test_strain_recovery_matches_cantilever():
    """Surface strain from internal actions against ``eps = -M y /(E I)``.

    A cantilever loaded +y at the tip must put the +y fibre at the root into
    compression; getting this sign wrong would invert the whole strain-ratio
    claim, so it is pinned here.
    """
    L, P, n_el = 8.0, 2.0e4, 4
    sec = tubular_section(1.2, 0.035, E, NU, 7850.0)
    nodes = np.zeros((n_el + 1, 3))
    nodes[:, 0] = np.linspace(0.0, L, n_el + 1)
    m = Model(nodes=nodes)
    for i in range(n_el):
        m.members.append(Member(i, i + 1, sec))
    m.fixed[0] = (True,) * 6
    f = np.zeros(m.n_dof)
    f[m.dof(n_el, "uy")] = P
    u = m.solve_static(f).displacements.reshape(-1)

    r = 0.6
    for s_global in (0.3, 1.0, 1.7):
        mem = m.members[0]
        Mz_exact = P * (L - s_global)
        eps = m.member_strain(mem, u, s_global, theta=0.0, radius=r)
        assert eps == pytest.approx(-Mz_exact * r / (E * sec.Iz), rel=1e-9)
        # Opposite fibre is the mirror image.
        assert m.member_strain(mem, u, s_global, np.pi, r) == pytest.approx(-eps, rel=1e-9)
        # Neutral axis carries no bending strain.
        assert m.member_strain(mem, u, s_global, np.pi / 2, r) == pytest.approx(0.0, abs=1e-15)


def test_strain_recovery_axial_is_uniform_around_the_section():
    L, N = 5.0, 3.0e6
    sec = tubular_section(1.2, 0.05, E, NU, 7850.0)
    m = Model(nodes=np.array([[0.0, 0, 0], [L, 0, 0]]))
    m.members.append(Member(0, 1, sec))
    m.fixed[0] = (True,) * 6
    f = np.zeros(m.n_dof)
    f[m.dof(1, "ux")] = N
    u = m.solve_static(f).displacements.reshape(-1)
    for theta in (0.0, 1.0, 2.5, 4.0):
        assert m.member_strain(m.members[0], u, 2.0, theta, 0.6) == pytest.approx(
            N / (E * sec.A), rel=1e-9
        )


def test_station_outside_the_member_is_rejected():
    sec = tubular_section(1.2, 0.05, E, NU, 7850.0)
    m = Model(nodes=np.array([[0.0, 0, 0], [3.0, 0, 0]]))
    m.members.append(Member(0, 1, sec))
    m.fixed[0] = (True,) * 6
    u = np.zeros(m.n_dof)
    with pytest.raises(ValueError, match="outside member"):
        m.member_strain(m.members[0], u, 4.0, 0.0, 0.6)


# ------------------------------------------------------------- jacket build


def test_jacket_assembles_and_has_no_mechanism():
    b = build_jacket(include_added_mass=True)
    assert b.model.n_nodes == 64
    assert len(b.model.members) == 112
    K, M = b.model.assemble()
    res = eigenmodes(K, M, b.model.free_dof(), n_modes=6)
    # Clamped at the mudline: no rigid-body modes, and the fundamental of a
    # 70 m braced steel jacket with a free top belongs in the low single Hz.
    assert np.all(res.frequencies_hz > 0.5)
    assert 1.0 < res.frequencies_hz[0] < 10.0
    # Four-fold symmetry gives a degenerate first pair.
    assert res.frequencies_hz[1] == pytest.approx(res.frequencies_hz[0], rel=1e-6)


def test_local_joint_flexibility_softens_the_frame():
    """LJF may only reduce stiffness, never increase it."""
    rigid = build_jacket(ljf_model=LJFModel.RIGID, include_added_mass=True)
    shell = build_jacket(ljf_model=LJFModel.SHELL, include_added_mass=True)
    assert len(shell.model.springs) == 72
    assert shell.model.n_nodes > rigid.model.n_nodes

    def tip_deflection(b):
        K, _ = b.model.assemble()
        f = np.zeros(b.model.n_dof)
        top = list(b.tables.joints.index).index(53)
        f[b.model.dof(top, "ux")] = 1.0e6
        return b.model.solve_static(f, K).displacements[top, 0]

    assert tip_deflection(shell) > tip_deflection(rigid)
    f_rigid = eigenmodes(*rigid.model.assemble(), rigid.model.free_dof(), 4).frequencies_hz
    f_shell = eigenmodes(*shell.model.assemble(), shell.model.free_dof(), 4).frequencies_hz
    assert f_shell[0] < f_rigid[0]


def test_rigid_joint_choice_is_recorded_as_an_assumption():
    b = build_jacket(ljf_model=LJFModel.RIGID)
    assert any("rigid" in a for a in b.modelling_assumptions)
    assert not any("rigid" in a for a in build_jacket(ljf_model=LJFModel.SHELL).modelling_assumptions)


def test_sensor_pair_brackets_the_joint():
    t = load_tables()
    p = sensor_pair(t, 5, offset_m=1.5)
    assert p.joint_id == 5
    assert p.upper_member != p.lower_member
    # Joint 5 sits between leg members 4 (below) and 17 (above).
    assert {p.upper_member, p.lower_member} == {4, 17}
    with pytest.raises(ValueError, match="exceeds"):
        sensor_pair(t, 3, offset_m=50.0)


def test_marine_growth_adds_mass_and_lowers_frequencies():
    clean = build_jacket(include_added_mass=True)
    fouled = build_jacket(include_added_mass=True, marine_growth_mm=100.0)
    f0 = eigenmodes(*clean.model.assemble(), clean.model.free_dof(), 3).frequencies_hz
    f1 = eigenmodes(*fouled.model.assemble(), fouled.model.free_dof(), 3).frequencies_hz
    assert np.all(f1 < f0)


# ---------------------------------------------------------------------- LJF


def test_shell_ljf_orders_opb_softer_than_ipb():
    """Out-of-plane bending is the most flexible mode in all published data."""
    g = JointGeometry(1.2, 0.035, 0.8, 0.02, np.radians(45.0))
    s = shell_ljf(g)
    assert s.k_opb < s.k_ipb
    assert np.isfinite(s.k_axial) and s.k_axial > 0


def test_shell_ljf_stiffens_with_thicker_chord():
    """Flexibility is a chord-wall mechanism: thicker wall, stiffer joint."""
    thin = shell_ljf(JointGeometry(1.2, 0.020, 0.8, 0.02, np.radians(45.0)))
    thick = shell_ljf(JointGeometry(1.2, 0.060, 0.8, 0.02, np.radians(45.0)))
    assert thick.k_axial > thin.k_axial
    assert thick.k_ipb > thin.k_ipb


def test_rigid_model_returns_infinite_stiffness():
    g = JointGeometry(1.2, 0.035, 0.8, 0.02, np.radians(45.0))
    s = joint_stiffness(g, LJFModel.RIGID)
    assert not np.isfinite(s.k_axial)
    v = s.as_vector(rigid_value=1.0e12)
    assert np.all(np.isfinite(v)) and np.all(v == 1.0e12)


def test_crack_compliance_adds_in_series():
    g = JointGeometry(1.2, 0.035, 0.8, 0.02, np.radians(45.0))
    base = joint_stiffness(g, LJFModel.SHELL)
    extra_c = 1.0 / base.k_axial  # doubling the compliance
    cracked = joint_stiffness(g, LJFModel.SHELL, extra_axial_compliance=extra_c)
    assert cracked.k_axial == pytest.approx(0.5 * base.k_axial, rel=1e-12)
    # A crack can only soften.
    assert cracked.k_axial < base.k_axial


def test_validity_envelope_flags_out_of_range_geometry():
    ok = JointGeometry(1.2, 0.035, 0.8, 0.02, np.radians(45.0))
    assert ok.validity() == []
    # The real OC4 braces meet the legs near 29 deg, just outside the envelope.
    shallow = JointGeometry(1.2, 0.035, 0.8, 0.02, np.radians(29.4))
    assert any("theta" in w for w in shallow.validity())


def test_unshipped_published_coefficients_report_data_unavailable():
    """Paywalled regression coefficients are absent by design, not by accident."""
    with pytest.raises(DataUnavailable) as exc:
        load_tabulated("buitrago1993")
    assert "Buitrago" in exc.value.remedy
    assert exc.value.remedy  # must tell the user how to supply them
