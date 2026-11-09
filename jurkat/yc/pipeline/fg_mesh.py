"""FG-repeat mesh of the nuclear pores (inventory v3, item F2).

The NPC library entry (7N9F, 216,568 residues) models only the folded parts of the FG nucleoporins. Their disordered
FG-repeat domains are added as self-avoiding residue chains (one bead per residue, CA_STEP apart, low persistence)
anchored at the first modelled residue of each chain:
  NSP1   residues 1-636     (48 copies per pore in 7N9F: channel and both rings)
  NUP57  residues 1-286     (32 copies)
  NUP49  residues 1-269     (32 copies)
  NUP159 residues 388-1102  (16 copies; after its beta-propeller, before the modelled C-terminal coiled coil)
Chains start pointing at the pore axis and must stay in the pore channel, cytoplasm or nucleoplasm, within MAX_REACH
of their anchor, clear of membranes, the NPC scaffold, chromatin and each other. FG nucleoporins that 7N9F does not
contain (NUP100, NUP116, NUP145N, NUP42, NUP1, NUP60, NUP2) are not added.
"""
import os
import numpy as np
import gemmi
import pack3d as p3
import cell3d as c3
from config import LIB

FG = {"P14907": ("NSP1", None), "P48837": ("NUP57", None), "Q02199": ("NUP49", None), "P40477": ("NUP159", 388)}
CA_STEP = 0.38
BEAD_R = 0.3
PERSIST = 0.6
MAX_REACH = 60.0      # nm: an FG domain (~600 residues) stays within the pore and its immediate surroundings
ANCHOR_F = f"{LIB}/NPC_fg_anchors.npz"


def anchors():
    """per FG chain of the NPC library entry: anchor point, inward direction, residues to add (library frame)"""
    if os.path.exists(ANCHOR_F):
        z = np.load(ANCHOR_F)
        return z["pos"], z["dir"], z["n"], z["gene"]
    import extend_complexes as xc, assemblies as A, membrane_orient as mo
    st = gemmi.read_structure(xc.fetch_assembly("7N9F")); st.remove_ligands_and_waters(); st.setup_entities()
    xyz, kind, acc, pos = mo.beads_with_positions(st, align=A.entity_align("7N9F"))
    L = np.load(f"{LIB}/NPC.npz")["beads"].astype(np.float64)
    assert len(L) == len(xyz)
    pc, qc = xyz.mean(0), L.mean(0); U, S, Vt = np.linalg.svd((xyz - pc).T @ (L - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T; t = qc - R @ pc
    X = xyz @ R.T + t                                   # structure -> library frame (pore axis = z)
    chain = np.array([c.name for c in st[0] for r in c
                      if gemmi.find_tabulated_residue(r.name) and (r.find_atom("CA", "*") or r.find_atom("P", "*"))])
    P, D, N, Gn = [], [], [], []
    for a, (gene, first) in FG.items():
        for c in np.unique(chain[acc == a]):
            k = np.where((chain == c) & (acc == a))[0]
            i0 = k[np.argmin(pos[k])]; p0 = int(pos[i0])
            n = p0 - 1 if first is None else p0 - first
            if n < 20:
                continue
            v = -np.r_[X[i0, :2], 0.0]; v /= max(np.linalg.norm(v), 1e-9)      # toward the pore axis
            P.append(X[i0]); D.append(v); N.append(n); Gn.append(gene)
    np.savez(ANCHOR_F, pos=np.array(P), dir=np.array(D), n=np.array(N), gene=np.array(Gn))
    return np.array(P), np.array(D), np.array(N), np.array(Gn)


def place(cell, pre, obstacles, seed=37, log=print):
    from scipy.spatial.transform import Rotation
    import mrna
    P, D, N, Gn = anchors()
    starts, dirs, lens = [], [], []
    for p, q in zip(pre["npc_pos"], pre["npc_q"]):
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        starts.append(P @ R.T + p); dirs.append(D @ R.T); lens.append(N)
    starts = np.concatenate(starts).astype(np.float64); dirs = np.concatenate(dirs).astype(np.float64)
    lens = np.concatenate(lens).astype(np.int64)
    OC, OR = obstacles
    G = cell.G; glo = np.full(3, -G[0] - 10.0); ghi = -glo
    a = cell.args(glo, ghi, margin=50.0)
    cap_s = int(lens.sum()) + len(OR) + 1000
    mask = (1 << c3.PORE) | (1 << c3.CYTO) | (1 << c3.NUCP)
    pts, off = mrna.thread_all(seed, glo, ghi, *a, OC.astype(np.float64), OR.astype(np.float64), starts, dirs, lens,
                               G[2] - 30.0, cap_s, int(cap_s * 3 + 8 * len(OR)), CA_STEP, BEAD_R, PERSIST, mask, MAX_REACH, 60)
    got = np.diff(off)
    per = {g: (got[np.tile(Gn, len(pre['npc_pos'])) == g].sum(), lens[np.tile(Gn, len(pre['npc_pos'])) == g].sum()) for g in np.unique(Gn)}
    log(f"FG mesh: {len(lens)} FG domains on {len(pre['npc_pos'])} pores, {got.sum() / 1e6:.2f}M of {lens.sum() / 1e6:.2f}M residues "
        f"({got.sum() / lens.sum():.0%}); " + ", ".join(f"{g} {a_ / b_:.0%}" for g, (a_, b_) in per.items()))
    return dict(fg_pts=pts, fg_off=off)
