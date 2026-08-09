"""Global assembly and sparse solution of the 3D frame.

Assembles the structure stiffness and mass from :mod:`tidetwin.fe.beam3d`
elements, applies boundary conditions by row/column elimination, and solves.
Local joint flexibility is introduced as node-pair spring elements (see
:mod:`tidetwin.fe.ljf`), which is the standard treatment for tubular joints in
frame analysis (ISO 19902:2020 Section 13.7, Buitrago et al. 1993).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # scipy is imported lazily inside functions at runtime;
    import scipy.sparse as sp  # this makes the sparse-matrix annotations resolvable

from dataclasses import dataclass, field

import numpy as np

from .beam3d import Section, element_matrices

__all__ = ["Model", "Member", "SpringLink", "SolveResult"]

DOF_PER_NODE = 6
DOF_NAMES = ("ux", "uy", "uz", "rx", "ry", "rz")


@dataclass
class Member:
    """A prismatic frame member between two nodes."""

    node_i: int
    node_j: int
    section: Section
    name: str = ""
    roll: float = 0.0
    group: int | str = ""  # property-set id, used to recover the section table row


@dataclass
class SpringLink:
    """A six-component elastic link between two coincident nodes.

    Used to represent local joint flexibility: the brace-end node is duplicated
    and connected to the chord node through diagonal springs whose stiffnesses
    come from the LJF parametric formulae. ``stiffness`` is ordered
    ``[kx, ky, kz, krx, kry, krz]`` in the *global* frame if ``local_axes`` is
    ``None``, otherwise in the frame whose rows are the supplied unit vectors.

    A stiffness of ``np.inf`` means the DOF is rigidly tied; it is implemented as
    a large penalty relative to the mean diagonal of the assembled stiffness.
    """

    node_a: int
    node_b: int
    stiffness: np.ndarray
    name: str = ""
    local_axes: np.ndarray | None = None


@dataclass
class SolveResult:
    """Displacements and derived member responses from a static solve."""

    displacements: np.ndarray  # (n_nodes, 6)
    reactions: np.ndarray  # (n_nodes, 6), zero at free DOF
    free_dof: np.ndarray
    condition_estimate: float | None = None

    def node(self, i: int) -> np.ndarray:
        return self.displacements[i]


@dataclass
class Model:
    """A 3D frame: nodes, members, springs, restraints.

    Nodes are an ``(n, 3)`` array of coordinates in metres, global Z up with
    Z = 0 at mean sea level unless the geometry module states otherwise.
    """

    nodes: np.ndarray
    members: list[Member] = field(default_factory=list)
    springs: list[SpringLink] = field(default_factory=list)
    fixed: dict[int, tuple[bool, bool, bool, bool, bool, bool]] = field(default_factory=dict)
    node_labels: dict[int, str] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_dof(self) -> int:
        return self.n_nodes * DOF_PER_NODE

    def dof(self, node: int, component: int | str) -> int:
        """Global DOF index for a node and component (index or name)."""
        c = DOF_NAMES.index(component) if isinstance(component, str) else int(component)
        return node * DOF_PER_NODE + c

    # ------------------------------------------------------------- assembly

    def assemble(self, lumped_mass: bool = False) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Assemble global stiffness ``K`` and mass ``M`` as sparse CSR."""
        import scipy.sparse as sp

        n = self.n_dof
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        kvals: list[np.ndarray] = []
        mvals: list[np.ndarray] = []

        for mem in self.members:
            Kg, Mg, _T, _L = element_matrices(
                mem.section,
                self.nodes[mem.node_i],
                self.nodes[mem.node_j],
                roll=mem.roll,
                lumped_mass=lumped_mass,
            )
            idx = np.concatenate(
                [
                    np.arange(mem.node_i * DOF_PER_NODE, mem.node_i * DOF_PER_NODE + 6),
                    np.arange(mem.node_j * DOF_PER_NODE, mem.node_j * DOF_PER_NODE + 6),
                ]
            )
            R, C = np.meshgrid(idx, idx, indexing="ij")
            rows.append(R.ravel())
            cols.append(C.ravel())
            kvals.append(Kg.ravel())
            mvals.append(Mg.ravel())

        K = sp.coo_matrix(
            (np.concatenate(kvals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        ).tocsr()
        M = sp.coo_matrix(
            (np.concatenate(mvals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        ).tocsr()

        if self.springs:
            K = K + self._spring_stiffness(K)
        return K.tocsr(), M.tocsr()

    def _spring_stiffness(self, K: sp.csr_matrix) -> sp.csr_matrix:
        """Assemble spring links. Infinite stiffness becomes a scaled penalty."""
        import scipy.sparse as sp

        n = self.n_dof
        diag = K.diagonal()
        scale = float(np.mean(diag[diag > 0])) if np.any(diag > 0) else 1.0
        penalty = 1.0e6 * scale
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for s in self.springs:
            k = np.asarray(s.stiffness, dtype=float).copy()
            k[~np.isfinite(k)] = penalty
            a0 = s.node_a * DOF_PER_NODE
            b0 = s.node_b * DOF_PER_NODE
            if s.local_axes is None:
                for c in range(6):
                    ia, ib = a0 + c, b0 + c
                    rows += [ia, ib, ia, ib]
                    cols += [ia, ib, ib, ia]
                    vals += [k[c], k[c], -k[c], -k[c]]
            else:
                R = np.asarray(s.local_axes, dtype=float)  # rows are local unit vectors
                for block in (0, 3):
                    kb = np.diag(k[block : block + 3])
                    kg = R.T @ kb @ R
                    for p in range(3):
                        for q in range(3):
                            ia, ib = a0 + block + p, b0 + block + q
                            ja, jb = a0 + block + q, b0 + block + p
                            rows += [ia, ib, ia, ib]
                            cols += [ja, jb, ib, ia]
                            vals += [kg[p, q], kg[p, q], -kg[p, q], -kg[p, q]]
        return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    # ------------------------------------------------------ boundary handling

    def free_dof(self) -> np.ndarray:
        """Indices of unrestrained DOF."""
        mask = np.ones(self.n_dof, dtype=bool)
        for node, flags in self.fixed.items():
            for c, f in enumerate(flags):
                if f:
                    mask[node * DOF_PER_NODE + c] = False
        return np.flatnonzero(mask)

    # ---------------------------------------------------------------- solve

    def solve_static(
        self, load: np.ndarray, K: sp.csr_matrix | None = None
    ) -> SolveResult:
        """Solve ``K u = f`` for a global load vector.

        ``load`` may be ``(n_dof,)`` or ``(n_nodes, 6)``.
        """
        import scipy.sparse.linalg as spla

        f = np.asarray(load, dtype=float).reshape(-1)
        if f.size != self.n_dof:
            raise ValueError(f"load vector has {f.size} entries, expected {self.n_dof}")
        if K is None:
            K, _ = self.assemble()
        free = self.free_dof()
        Kff = K[free][:, free].tocsc()
        u = np.zeros(self.n_dof)
        try:
            u[free] = spla.spsolve(Kff, f[free])
        except Exception as exc:  # pragma: no cover - singular systems
            raise RuntimeError(
                "frame solve failed; the model is likely a mechanism (check restraints "
                "and that every LJF spring connects two existing nodes)"
            ) from exc
        if not np.all(np.isfinite(u)):
            raise RuntimeError("frame solve produced non-finite displacements (singular stiffness)")
        r = K @ u - f
        r[free] = 0.0
        return SolveResult(
            displacements=u.reshape(-1, 6),
            reactions=r.reshape(-1, 6),
            free_dof=free,
        )

    # ----------------------------------------------------------- post-process

    def member_end_forces(self, mem: Member, u: np.ndarray) -> np.ndarray:
        """Local-frame end force vector (12,) for one member.

        Sign convention is the element's own: entries 0-5 act on node i, 6-11 on
        node j, in local axes.
        """
        from .beam3d import local_stiffness, transformation

        T, L = transformation(self.nodes[mem.node_i], self.nodes[mem.node_j], mem.roll)
        ug = np.concatenate(
            [
                u.reshape(-1, 6)[mem.node_i],
                u.reshape(-1, 6)[mem.node_j],
            ]
        )
        return local_stiffness(mem.section, L) @ (T @ ug)

    def member_internal_forces(
        self, mem: Member, u: np.ndarray, s: float
    ) -> tuple[float, float, float]:
        """Internal ``(N, My, Mz)`` at distance ``s`` from node i, local axes.

        All loading in this model is applied at nodes, so within an element the
        shear is constant and the bending moments vary linearly. The internal
        moments are therefore obtained by interpolating the end values:

            N(s)  = fe[6]                              (tension positive)
            Mz(s) = -fe[5] (1 - s/L) + fe[11] (s/L)
            My(s) = -fe[4] (1 - s/L) + fe[10] (s/L)

        where ``fe = K_local u_local`` are the forces the nodes apply to the
        element. The leading minus at node i converts the applied end action into
        the internal action on the adjacent cut.
        """
        from .beam3d import transformation

        _T, L = transformation(self.nodes[mem.node_i], self.nodes[mem.node_j], mem.roll)
        if not -1e-9 <= s <= L + 1e-9:
            raise ValueError(f"station {s} m lies outside member of length {L:.3f} m")
        fe = self.member_end_forces(mem, u)
        xi = float(np.clip(s / L, 0.0, 1.0))
        N = float(fe[6])
        Mz = float(-fe[5] * (1.0 - xi) + fe[11] * xi)
        My = float(-fe[4] * (1.0 - xi) + fe[10] * xi)
        return N, My, Mz

    def member_strain(
        self, mem: Member, u: np.ndarray, s: float, theta: float, radius: float
    ) -> float:
        """Axial surface strain a member-aligned gauge would read. Tension positive.

        For a beam along local x, the fibre at local coordinates
        ``(y, z) = (r cos(theta), r sin(theta))`` carries

        .. math::
            \\varepsilon = \\frac{N}{EA}
                         - \\frac{M_z\\, y}{E I_z}
                         + \\frac{M_y\\, z}{E I_y}

        The signs follow from :math:`\\kappa_{xy} = M_z/(EI_z)`,
        :math:`\\kappa_{xz} = -M_y/(EI_y)` and
        :math:`\\varepsilon = -y \\kappa_{xy} - z \\kappa_{xz}`, with the
        right-handed convention that a positive ``ry`` rotates +z toward +x.
        Verified in ``tests/test_oc4.py::test_strain_recovery_matches_cantilever``.

        ``theta`` is measured from the local +y axis about the member axis;
        ``radius`` is the gauge circle radius (the outer radius for an
        externally bonded FBG).
        """
        N, My, Mz = self.member_internal_forces(mem, u, s)
        sec = mem.section
        y = radius * np.cos(theta)
        z = radius * np.sin(theta)
        return float(
            N / (sec.E * sec.A) - Mz * y / (sec.E * sec.Iz) + My * z / (sec.E * sec.Iy)
        )
