"""OC4 reference jacket geometry.

The node, member and section tables are shipped as CSV under ``data/geometry/``
and are transcribed from NREL's FAST CertTest SubDyn input deck
``NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat``, which encodes the jacket defined
in Vorpahl, Popko & Kaufer (2013), "Description of a basic model of the 'UpWind
reference jacket' for code comparison in the OC4 project under IEA Wind Annex
30", Fraunhofer IWES. The manifest records the retrieval date and a SHA-256 of
each table so a reviewer can confirm the numbers were not edited afterwards.

Layout, as published: four battered legs on a 12 m x 12 m mudline footprint
narrowing to 8 m x 8 m at the transition piece, four levels of X-bracing, one
horizontal mud brace frame, piles clamped at z = -50.001 m, still water level at
z = 0.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..fe.assemble import Member, Model, SpringLink
from ..fe.beam3d import Section, tubular_section
from ..fe.ljf import RIGID_STIFFNESS_FACTOR, JointGeometry, LJFModel, joint_stiffness
from ..provenance import Citation, DataUnavailable

__all__ = [
    "DATA_DIR",
    "OC4_CITATION",
    "JacketTables",
    "load_tables",
    "JacketBuild",
    "build_jacket",
    "SensorPair",
    "brace_chord_joints",
    "MUDLINE_Z",
    "SWL_Z",
    "WATER_DEPTH",
]

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "geometry"

OC4_CITATION = Citation(
    document=(
        "NREL FAST CertTest deck NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat, encoding the "
        "OC4/UpWind reference jacket of Vorpahl, Popko & Kaufer, 'Description of a basic "
        "model of the UpWind reference jacket for code comparison in the OC4 project under "
        "IEA Wind Annex 30', Fraunhofer IWES"
    ),
    locator="JOINTS, MEMBERS and MEMBER X-SECTION PROPERTY tables",
    year=2013,
    url="https://github.com/old-NWTC/FAST/blob/master/CertTest/5MW_Baseline/NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat",
    retrieved=_dt.date(2026, 7, 28),
)

MUDLINE_Z = -50.001
SWL_Z = 0.0
WATER_DEPTH = SWL_Z - MUDLINE_Z

# Joints clamped at the mudline (SubDyn "base reaction joints").
BASE_JOINTS = (61, 62, 63, 64)
# Joints where the transition piece attaches (SubDyn "interface joints").
INTERFACE_JOINTS = (24, 28, 32, 36, 53, 54, 55, 56)
LEG_PROP_SETS = (2, 3, 4)
BRACE_PROP_SETS = (1,)
PILE_PROP_SETS = (5, 6)


@dataclass(frozen=True)
class JacketTables:
    """The three published tables plus their integrity manifest."""

    joints: pd.DataFrame
    members: pd.DataFrame
    sections: pd.DataFrame
    manifest: dict

    @property
    def digest(self) -> str:
        """Short hash of all three tables, stamped onto ledger exports."""
        h = hashlib.sha256()
        for name in ("joints", "members", "sections"):
            h.update(self.manifest["files"][f"oc4_{name}.csv"]["sha256"].encode())
        return h.hexdigest()[:12]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_tables(root: str | None = None) -> JacketTables:
    """Read the OC4 tables and verify them against the shipped manifest.

    Raises
    ------
    DataUnavailable
        If a table is missing, or if its SHA-256 does not match the manifest.
        A silently edited geometry table would invalidate every structural claim
        in the ledger, so a mismatch is fatal rather than a warning.
    """
    d = Path(root) if root else DATA_DIR
    files = {n: d / f"oc4_{n}.csv" for n in ("joints", "members", "sections")}
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise DataUnavailable(
            "OC4 jacket geometry",
            f"missing table(s): {', '.join(missing)}",
            "restore data/geometry/ from the repository",
        )
    manifest_path = d / "MANIFEST.json"
    if not manifest_path.is_file():
        raise DataUnavailable(
            "OC4 geometry manifest",
            f"{manifest_path} not present",
            "run scripts/write_geometry_manifest.py",
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, path in files.items():
        want = manifest["files"][path.name]["sha256"]
        got = _sha256(path)
        if want != got:
            raise DataUnavailable(
                "OC4 jacket geometry",
                f"{path.name} SHA-256 {got[:12]} does not match manifest {want[:12]}",
                "the geometry table has been modified since it was transcribed; restore it",
            )
    return JacketTables(
        joints=pd.read_csv(files["joints"]).set_index("joint_id").sort_index(),
        members=pd.read_csv(files["members"]).set_index("member_id").sort_index(),
        sections=pd.read_csv(files["sections"]).set_index("prop_set").sort_index(),
        manifest=manifest,
    )


# ------------------------------------------------------------------ sections


def _section_for(
    row: pd.Series,
    marine_growth_t: float = 0.0,
    marine_growth_rho: float = 1325.0,
    added_mass_per_length: float = 0.0,
) -> Section:
    """Build a :class:`Section` from a property-set row.

    Marine growth adds mass but no structural stiffness (DNV-RP-C205 Section
    6.7.3: growth is treated as non-structural mass and increased hydrodynamic
    diameter). Its density defaults to 1325 kg/m^3, the value DNV-RP-C205 gives
    for hard growth.
    """
    D = float(row.outer_diameter_m)
    t = float(row.wall_thickness_m)
    E = float(row.E_Pa)
    G = float(row.G_Pa)
    nu = E / (2.0 * G) - 1.0
    growth_mass = 0.0
    if marine_growth_t > 0:
        growth_mass = marine_growth_rho * np.pi * (D + marine_growth_t) * marine_growth_t
    return tubular_section(
        outer_diameter=D,
        wall_thickness=t,
        E=E,
        nu=nu,
        rho=float(row.density_kg_m3),
        added_mass_per_length=added_mass_per_length + growth_mass,
    )


# ------------------------------------------------------------------- joints


def brace_chord_joints(tables: JacketTables) -> dict[int, list[int]]:
    """Map each leg joint to the brace members that land on it.

    A leg joint carrying two or more braces is a K (uniplanar) or KK
    (multiplanar) joint; these are the fatigue-critical locations and the ones
    the sensor pairs bracket.
    """
    leg_joints: set[int] = set()
    for _mid, m in tables.members.iterrows():
        if int(m.prop_set) in LEG_PROP_SETS:
            leg_joints.update((int(m.joint_i), int(m.joint_j)))
    out: dict[int, list[int]] = {}
    for mid, m in tables.members.iterrows():
        if int(m.prop_set) not in BRACE_PROP_SETS:
            continue
        for j in (int(m.joint_i), int(m.joint_j)):
            if j in leg_joints:
                out.setdefault(j, []).append(int(mid))
    return {k: sorted(v) for k, v in sorted(out.items()) if len(v) >= 2}


def _leg_members_at(tables: JacketTables, joint: int) -> list[int]:
    return [
        int(mid)
        for mid, m in tables.members.iterrows()
        if int(m.prop_set) in LEG_PROP_SETS
        and joint in (int(m.joint_i), int(m.joint_j))
    ]


@dataclass(frozen=True)
class SensorPair:
    """Two member-aligned gauges bracketing a joint on the chord.

    ``upper``/``lower`` each name the leg member the gauge sits on, the station
    along that member measured from its node i, and the circumferential angle.
    The claim under test is the ratio of their tidal strain amplitudes.
    """

    joint_id: int
    upper_member: int
    upper_station: float
    lower_member: int
    lower_station: float
    theta: float
    offset_m: float

    def label(self) -> str:
        return f"J{self.joint_id} pair, +/-{self.offset_m:.2f} m, theta={np.degrees(self.theta):.0f} deg"


def sensor_pair(
    tables: JacketTables, joint: int, offset_m: float = 1.5, theta: float = 0.0
) -> SensorPair:
    """Place a bracketing gauge pair on the leg either side of ``joint``.

    ``offset_m`` is the axial distance from the joint centre along the leg. DNV
    and the tubular-joint literature place the extrapolation region beyond
    0.4*sqrt(r*T) from the weld toe; a default of 1.5 m sits well outside the
    notch field of a 1.2 m chord while remaining on the same member.
    """
    legs = _leg_members_at(tables, joint)
    if len(legs) < 2:
        raise ValueError(
            f"joint {joint} has {len(legs)} leg member(s); a bracketing pair needs two"
        )
    zs = tables.joints.loc[joint, "z_m"]
    above: list[tuple[int, float]] = []
    below: list[tuple[int, float]] = []
    for mid in legs:
        m = tables.members.loc[mid]
        ji, jj = int(m.joint_i), int(m.joint_j)
        other = jj if ji == joint else ji
        z_other = tables.joints.loc[other, "z_m"]
        length = float(
            np.linalg.norm(
                tables.joints.loc[jj, ["x_m", "y_m", "z_m"]].to_numpy(float)
                - tables.joints.loc[ji, ["x_m", "y_m", "z_m"]].to_numpy(float)
            )
        )
        # Station measured from node i of that member.
        station = offset_m if ji == joint else length - offset_m
        if not 0.0 <= station <= length:
            raise ValueError(
                f"offset {offset_m} m exceeds the {length:.2f} m leg member {mid} at joint {joint}"
            )
        (above if z_other > zs else below).append((mid, station))
    if not above or not below:
        raise ValueError(f"joint {joint} does not have leg members on both sides")
    return SensorPair(
        joint_id=joint,
        upper_member=above[0][0],
        upper_station=above[0][1],
        lower_member=below[0][0],
        lower_station=below[0][1],
        theta=theta,
        offset_m=offset_m,
    )


# -------------------------------------------------------------------- build


@dataclass
class JacketBuild:
    """An assembled jacket model plus the bookkeeping needed to interpret it."""

    model: Model
    tables: JacketTables
    ljf_model: LJFModel
    member_of: dict[int, Member] = field(default_factory=dict)
    submerged_members: tuple[int, ...] = ()
    spring_joint_of: dict[int, int] = field(default_factory=dict)
    modelling_assumptions: tuple[str, ...] = ()

    def member(self, member_id: int) -> Member:
        return self.member_of[member_id]

    def strain(self, u: np.ndarray, member_id: int, station: float, theta: float) -> float:
        mem = self.member_of[member_id]
        radius = 0.5 * float(self.tables.sections.loc[mem.group, "outer_diameter_m"])
        return self.model.member_strain(mem, u, station, theta, radius)

    def pair_strains(self, u: np.ndarray, pair: SensorPair) -> tuple[float, float]:
        return (
            self.strain(u, pair.upper_member, pair.upper_station, pair.theta),
            self.strain(u, pair.lower_member, pair.lower_station, pair.theta),
        )


def build_jacket(
    ljf_model: LJFModel = LJFModel.RIGID,
    marine_growth_mm: float = 0.0,
    include_added_mass: bool = False,
    water_density: float = 1025.0,
    added_mass_coefficient: float = 1.0,
    crack_compliance: dict[int, tuple[float, float, float]] | None = None,
    foundation_stiffness: float | None = None,
    tables: JacketTables | None = None,
) -> JacketBuild:
    """Assemble the OC4 jacket.

    Parameters
    ----------
    ljf_model
        Which local joint flexibility formulation to apply at brace-to-leg
        connections. See :mod:`tidetwin.fe.ljf`.
    marine_growth_mm
        Uniform marine growth thickness applied to submerged members as
        non-structural mass and hydrodynamic diameter (DNV-RP-C205 Section
        6.7.3). Depth-varying profiles are applied by the loading module; the
        structural mass effect is taken as uniform because it is a second-order
        contributor to a quasi-static strain ratio.
    include_added_mass
        Add hydrodynamic added mass ``rho_w * Ca * pi D^2/4`` to submerged
        members. Required for modal claims (C7), irrelevant to static ones.
    crack_compliance
        ``{member_id: (axial, ipb, opb)}`` extra compliances in series at that
        brace's LJF spring, from :mod:`tidetwin.damage.crack_ljf`.
    foundation_stiffness
        If given, the mudline joints are supported on six equal-diagonal springs
        of this stiffness instead of being clamped, which is how scour is
        represented in the C3 nuisance budget. ``None`` clamps them, matching
        the published SubDyn deck.
    """
    tables = tables or load_tables()
    joints = tables.joints
    ids = list(joints.index)
    index_of = {j: i for i, j in enumerate(ids)}
    coords = joints.loc[ids, ["x_m", "y_m", "z_m"]].to_numpy(float)

    nodes = [coords]
    n_next = len(ids)
    members: list[Member] = []
    springs: list[SpringLink] = []
    member_of: dict[int, Member] = {}
    spring_joint_of: dict[int, int] = {}
    submerged: list[int] = []
    assumptions: list[str] = []

    kj = brace_chord_joints(tables)
    ljf_targets = set(kj.keys())

    for mid, m in tables.members.iterrows():
        mid = int(mid)
        ps = int(m.prop_set)
        row = tables.sections.loc[ps]
        ji, jj = int(m.joint_i), int(m.joint_j)
        zmid = 0.5 * (joints.loc[ji, "z_m"] + joints.loc[jj, "z_m"])
        is_submerged = zmid < SWL_Z
        D = float(row.outer_diameter_m)
        am = 0.0
        if include_added_mass and is_submerged:
            am = water_density * added_mass_coefficient * np.pi * D**2 / 4.0
        sec = _section_for(
            row,
            marine_growth_t=(marine_growth_mm * 1e-3 if is_submerged else 0.0),
            added_mass_per_length=am,
        )
        ni, nj = index_of[ji], index_of[jj]
        member_length = float(np.linalg.norm(coords[nj] - coords[ni]))

        # Insert an LJF spring where a brace lands on a leg joint.
        if ljf_model is not LJFModel.RIGID and ps in BRACE_PROP_SETS:
            for end, (jt, _n) in enumerate(((ji, ni), (jj, nj))):
                if jt not in ljf_targets:
                    continue
                new_idx = n_next
                n_next += 1
                nodes.append(coords[index_of[jt]][None, :])
                axes, geom = _joint_frame(tables, mid, jt, ps)
                extra = (crack_compliance or {}).get(mid, (0.0, 0.0, 0.0))
                st = joint_stiffness(
                    geom,
                    ljf_model,
                    extra_axial_compliance=extra[0],
                    extra_ipb_compliance=extra[1],
                    extra_opb_compliance=extra[2],
                )
                rigid_val = RIGID_STIFFNESS_FACTOR * sec.E * sec.A / max(member_length, 1e-6)
                springs.append(
                    SpringLink(
                        node_a=new_idx,
                        node_b=index_of[jt],
                        stiffness=st.as_vector(rigid_val),
                        name=f"LJF m{mid} @ J{jt}",
                        local_axes=axes,
                    )
                )
                spring_joint_of[new_idx] = jt
                if end == 0:
                    ni = new_idx
                else:
                    nj = new_idx

        mem = Member(ni, nj, sec, name=f"M{mid}", group=ps)
        members.append(mem)
        member_of[mid] = mem
        if is_submerged:
            submerged.append(mid)

    model = Model(nodes=np.vstack(nodes), members=members, springs=springs)
    for j in BASE_JOINTS:
        if foundation_stiffness is None:
            model.fixed[index_of[j]] = (True,) * 6
        else:
            model.fixed[index_of[j]] = (False,) * 6
    if foundation_stiffness is not None:
        assumptions.append("mudline restraint replaced by finite foundation springs (scour case)")
        model.nodes = np.vstack([model.nodes, model.nodes[[index_of[j] for j in BASE_JOINTS]]])
        for k, j in enumerate(BASE_JOINTS):
            ground = len(model.nodes) - len(BASE_JOINTS) + k
            model.fixed[ground] = (True,) * 6
            model.springs.append(
                SpringLink(
                    node_a=index_of[j],
                    node_b=ground,
                    stiffness=np.full(6, float(foundation_stiffness)),
                    name=f"foundation J{j}",
                )
            )
    model.node_labels = {index_of[j]: f"J{j}" for j in ids}

    if ljf_model is LJFModel.RIGID:
        assumptions.append(
            "brace-to-leg connections modelled rigid (ISO 19902 frame idealisation); "
            "local joint flexibility not represented"
        )
    if marine_growth_mm > 0:
        assumptions.append(f"uniform {marine_growth_mm:.0f} mm marine growth mass on submerged members")

    return JacketBuild(
        model=model,
        tables=tables,
        ljf_model=ljf_model,
        member_of=member_of,
        submerged_members=tuple(submerged),
        spring_joint_of=spring_joint_of,
        modelling_assumptions=tuple(assumptions),
    )


def _joint_frame(
    tables: JacketTables, brace_member: int, joint: int, brace_ps: int
) -> tuple[np.ndarray, JointGeometry]:
    """Local axes for a brace-end LJF spring, and the joint's geometry.

    Row 0 is the brace axis (pointing away from the joint), row 1 the in-plane
    bending axis (normal to the plane containing brace and chord), row 2 the
    out-of-plane bending axis. Matches the ordering
    :meth:`tidetwin.fe.ljf.LJFStiffness.as_vector` assumes.
    """
    j = tables.joints
    m = tables.members.loc[brace_member]
    ji, jj = int(m.joint_i), int(m.joint_j)
    other = jj if ji == joint else ji
    p0 = j.loc[joint, ["x_m", "y_m", "z_m"]].to_numpy(float)
    p1 = j.loc[other, ["x_m", "y_m", "z_m"]].to_numpy(float)
    ex = p1 - p0
    ex /= np.linalg.norm(ex)

    legs = _leg_members_at(tables, joint)
    lm = tables.members.loc[legs[0]]
    la = j.loc[int(lm.joint_j), ["x_m", "y_m", "z_m"]].to_numpy(float) - j.loc[
        int(lm.joint_i), ["x_m", "y_m", "z_m"]
    ].to_numpy(float)
    la /= np.linalg.norm(la)

    n = np.cross(ex, la)
    if np.linalg.norm(n) < 1e-8:  # brace collinear with the chord
        n = np.cross(ex, np.array([0.0, 0.0, 1.0]))
    n /= np.linalg.norm(n)
    e2 = np.cross(ex, n)
    axes = np.vstack([ex, n, e2])

    chord = tables.sections.loc[int(lm.prop_set)]
    brace = tables.sections.loc[brace_ps]
    # Standard tubular-joint convention: theta is the included angle between the
    # brace and chord axes. The OC4 X-braces meet the legs near 29 deg, just
    # below the usual 30 deg validity floor, which JointGeometry.validity()
    # reports rather than silently clipping.
    theta = float(np.arccos(np.clip(abs(float(np.dot(ex, la))), -1.0, 1.0)))
    geom = JointGeometry(
        chord_D=float(chord.outer_diameter_m),
        chord_T=float(chord.wall_thickness_m),
        brace_d=float(brace.outer_diameter_m),
        brace_t=float(brace.wall_thickness_m),
        theta=theta,
        E=float(chord.E_Pa),
        nu=float(chord.E_Pa) / (2.0 * float(chord.G_Pa)) - 1.0,
    )
    return axes, geom
