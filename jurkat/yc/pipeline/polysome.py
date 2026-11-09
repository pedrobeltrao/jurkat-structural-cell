"""Polysomes (inventory v3, item C1): cytosolic 80S ribosomes threaded on mRNAs.

Geometry: the mRNA channel of the 80S library structure (4V88, which has no mRNA) is taken from PDB 7ZW0 (yeast 80S
with 29 nt of mRNA, superposed on 4V88 by matching UniProt residues; median deviation 0.16 nm). The channel's 5' end is
the exit and its 3' end the entry. A polysome is grown as a self-avoiding chain: each next ribosome is placed so that
its mRNA entry lies a short linker (LINK_MIN..LINK_MAX nm) beyond the previous ribosome's exit, inside a cone around the
exit direction, with a random spin; ribosomes must lie in the cytosol and not clash. The mRNA polyline of each
ribosome is its channel segment plus the linker to the next ribosome (schematic beads, 0.5 nm apart, like linker DNA).
Polysome ribosomes are placed before packing (like NPCs and nucleosomes) and replace free cytosolic ribosomes.

Assumptions: POLY_FRACTION of cytosolic ribosomes are in polysomes; polysome sizes 1 + Poisson(MEAN_SIZE - 1)
(capped at MAX_SIZE); each polysome is one mRNA, so free mRNA threads are reduced by the number of polysomes.
"""
import os
import numpy as np
import numba as nb
import pack3d as p3
import cell3d as c3
import structlib as sl
from build_structs import rot_to_z
from config import LIB

RIBO = "ribosome_80S"
POLY_FRACTION = 0.7       # share of cytosolic ribosomes in polysomes (log phase, rich medium; assumption)
MEAN_SIZE = 5.0           # mean ribosomes per polysome (assumption)
MAX_SIZE = 15
LINK_MIN, LINK_MAX = 3.0, 12.0
CONE = 0.7                # rad, max deviation of the linker from the exit direction
H_WALK = 20.0             # nm, collision grid cell for the whole-cell walk
NT_STEP = 0.5             # nm between schematic mRNA beads (< 0.635 nm encoder delta limit)
CHANNEL_F = f"{LIB}/{RIBO}_mrna_channel.npy"


def channel_path():
    """mRNA channel of the 80S library structure, 5'->3', in the library frame (nm); computed once from 7ZW0."""
    if os.path.exists(CHANNEL_F):
        return np.load(CHANNEL_F)
    import gemmi, extend_complexes as xc, assemblies as A, membrane_orient as mo

    def ca(pid):
        st = gemmi.read_structure(xc.fetch_assembly(pid)); st.setup_entities()
        xyz, kind, acc, pos = mo.beads_with_positions(st, align=A.entity_align(pid))
        return st, xyz, acc, pos

    def kabsch(P, Q):
        pc, qc = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
        d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
        return R, qc - R @ pc

    xs, _, _ = sl.beads_from_structure(gemmi.read_structure(xc.fetch_assembly("4V88")))
    _, x0, a0, p0 = ca("4V88")
    key0 = {}
    for i, k in enumerate(zip(a0, p0)):
        if k[0] and k[1] > 0:
            key0.setdefault(k, i)
    st1, x1, a1, p1 = ca("7ZW0")
    seen, pairs = set(), []
    for j, k in enumerate(zip(a1, p1)):
        if k in key0 and k not in seen:
            seen.add(k); pairs.append((key0[k], j))
    I, J = np.array(pairs).T
    P, Q = x1[J], x0[I]
    R, t = kabsch(P, Q)
    for _ in range(4):        # trim flexible parts (40S head, stalks)
        res = np.linalg.norm(P @ R.T + t - Q, axis=1)
        keep = res < max(0.3, np.percentile(res, 75)); R, t = kabsch(P[keep], Q[keep])
    desc = A.entity_desc("7ZW0"); sub2ent = {s: e.name for e in st1.entities for s in e.subchains}
    nts = sorted((r.seqid.num, (r.find_atom("P", "*") or r.find_atom("C4'", "*")).pos) for ch in st1[0] for r in ch
                 if desc.get(sub2ent.get(r.subchain, ""), "").lower() == "mrna" and (r.find_atom("P", "*") or r.find_atom("C4'", "*")))
    m = np.array([[p.x, p.y, p.z] for _, p in nts]) / 10.0 @ R.T + t - xs.mean(0)
    np.save(CHANNEL_F, m.astype(np.float32))
    return m


def _mat2quat(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])          # pack3d convention (w, x, y, z)


@nb.njit(cache=True)
def _qmul(a, b, out):
    out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]


@nb.njit(cache=True)
def walk(seed, glo, ghi, G, pores, pidx, PT, PQ, PC, stc, str_, br, mask, sizes, ex, tex, en, q_en, OC, OR, cap_s, cap_e):
    """Grow one chain per entry of sizes. ex/tex: exit point and outward exit direction (library frame);
    en: entry point; q_en: quaternion taking the outward entry direction to +z. Returns pos, quat, chain index."""
    np.random.seed(seed)
    dims, head, nxt, sid, cen, rad, cnt = p3.grid_new(glo, ghi, H_WALK, cap_s, cap_e)
    if OC.shape[0]:
        p3.g_insert(glo, H_WALK, dims, head, nxt, sid, cen, rad, cnt, OC, OR)     # e.g. actin cables
    ntot = sizes.sum()
    pos = np.empty((ntot, 3), np.float32); qu = np.empty((ntot, 4), np.float32); chain = np.empty(ntot, np.int32)
    out = np.zeros(c3.NMEM); M = np.empty((3, 3)); q = np.empty(4); q2 = np.empty(4); t = np.empty(3)
    P = np.empty((stc.shape[0], 3)); D = np.empty(3); negD = np.empty(3)
    exw = np.empty(3); texw = np.empty(3)
    n = 0
    R0 = G[2] - 30.0
    for c in range(sizes.shape[0]):
        placed = 0; fails = 0
        while placed < sizes[c]:
            if placed == 0:
                while True:
                    for k in range(3):
                        t[k] = (np.random.random() * 2 - 1) * R0
                    if t[0] * t[0] + t[1] * t[1] + t[2] * t[2] < R0 * R0:
                        break
                p3.rand_quat(q)
            else:
                while True:   # direction within CONE of the previous exit direction
                    for k in range(3):
                        D[k] = np.random.normal()
                    dn = np.sqrt(D[0] ** 2 + D[1] ** 2 + D[2] ** 2)
                    s = 0.0
                    for k in range(3):
                        D[k] = D[k] / dn
                        s += D[k] * texw[k]
                    if s > np.cos(CONE):
                        break
                L = LINK_MIN + np.random.random() * (LINK_MAX - LINK_MIN)
                for k in range(3):
                    negD[k] = -D[k]
                p3.quat_z_to(negD, np.random.random() * 2 * np.pi, q2)   # outward entry direction -> -D
                _qmul(q2, q_en, q)
                p3.quat_mat(q, M)
                for k in range(3):
                    t[k] = exw[k] + L * D[k] - (M[k, 0] * en[0] + M[k, 1] * en[1] + M[k, 2] * en[2])
            p3.quat_mat(q, M); p3.transform(stc, M, t, P)
            good = not p3.g_overlap(glo, H_WALK, dims, head, nxt, sid, cen, rad, P, str_, p3.TOL)
            if good:
                good = p3.check_volume(P, str_, t[0], t[1], t[2], br, mask, G, pores, pidx, PT, PQ, PC, out)
            if not good:
                fails += 1
                if placed > 0 and fails > 60:
                    break            # polysome ends shorter
                continue
            if not p3.g_insert(glo, H_WALK, dims, head, nxt, sid, cen, rad, cnt, P, str_):
                return pos[:n], qu[:n], chain[:n]
            for k in range(3):
                exw[k] = t[k] + M[k, 0] * ex[0] + M[k, 1] * ex[1] + M[k, 2] * ex[2]
                texw[k] = M[k, 0] * tex[0] + M[k, 1] * tex[1] + M[k, 2] * tex[2]
                pos[n, k] = t[k]
            for k in range(4):
                qu[n, k] = q[k]
            chain[n] = c
            n += 1; placed += 1; fails = 0
    return pos[:n], qu[:n], chain[:n]


def _resample(P, step=NT_STEP):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    u = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.c_[[np.interp(u, s, P[:, k]) for k in range(3)]].T


def _hermite(a, ta, b, tb):
    L = np.linalg.norm(b - a)
    s = np.linspace(0, 1, max(3, int(L * 2)))[:, None]
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
    return h00 * a + h10 * ta * L + h01 * b + h11 * tb * L


def place(cell, lp, n_ribo, seed=17, obstacles=None, log=print):
    """Place polysomes holding ~n_ribo cytosolic ribosomes. Returns dict with ribosome instances, per-ribosome mRNA
    polylines (channel + linker to the next ribosome) and the linker beads as packing obstacles."""
    from scipy.spatial.transform import Rotation
    li = int(np.where(lp["ids"] == RIBO)[0][0])
    z = np.load(f"{LIB}/{RIBO}.npz")
    stc, str_ = sl.sphere_tree(z["beads"].astype(np.float64), per=550)       # ~30 coarse spheres for the walk
    ch = channel_path().astype(np.float64)
    ex, en = ch[0], ch[-1]
    tex = ch[0] - ch[3]; tex /= np.linalg.norm(tex)
    ten = ch[-1] - ch[-4]; ten /= np.linalg.norm(ten)
    q_en = _mat2quat(rot_to_z(ten))
    rng = np.random.default_rng(seed)
    sizes = []
    while sum(sizes) < n_ribo:
        sizes.append(int(min(MAX_SIZE, 1 + rng.poisson(MEAN_SIZE - 1))))
    sizes = np.array(sizes, np.int64)
    G = cell.G
    glo = np.full(3, -G[0] - 10.0); ghi = -glo
    a = cell.args(glo, ghi, margin=50.0)
    OC, OR = obstacles if obstacles is not None else (np.zeros((0, 3)), np.zeros(0))
    ns = len(str_) * len(sizes) * int(MEAN_SIZE + 2) + len(OR)
    pos, qu, chn = walk(seed, glo, ghi, *a, stc, str_, float(np.max(np.linalg.norm(stc, axis=1) + str_)), 1 << c3.CYTO,
                        sizes, ex, tex, en, q_en, OC.astype(np.float64), OR.astype(np.float64), ns + 1000, ns * 14 + 10000)
    # mRNA polylines
    Rm = Rotation.from_quat(np.c_[qu[:, 1:], qu[:, :1]].astype(np.float64)).as_matrix()
    pieces, links = [], []
    for i in range(len(pos)):
        P = ch @ Rm[i].T + pos[i]
        if i + 1 < len(pos) and chn[i + 1] == chn[i]:
            Pn = ch @ Rm[i + 1].T + pos[i + 1]
            lk = _hermite(P[0], Rm[i] @ tex, Pn[-1], -(Rm[i + 1] @ ten))
            links.append(lk[1:-1]); P = np.vstack([P[::-1], lk[1:]])
        else:
            P = P[::-1]
        pieces.append(_resample(P))
    off = np.r_[0, np.cumsum([len(p) for p in pieces])].astype(np.int64)
    n_chain = len(np.unique(chn))
    log(f"polysomes: {n_chain} chains, {len(pos)}/{sizes.sum()} ribosomes (mean {len(pos) / max(n_chain, 1):.1f} per mRNA), "
        f"{off[-1]} mRNA beads")
    return dict(poly_pos=pos, poly_q=qu, poly_chain=chn, li_ribo=li, poly_mrna=np.concatenate(pieces).astype(np.float32),
                poly_mrna_off=off, poly_link=np.concatenate(links).astype(np.float32) if links else np.zeros((0, 3), np.float32),
                poly_chains=n_chain)
