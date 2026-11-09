"""Microtubules from the spindle pole body (inventory v3, item D): unbudded G1 cell, one SPB in the nuclear envelope.

Geometry from PDB 8QV3 (native yeast gamma-TuRC capping a spindle microtubule, cryo-ET subtomogram average 8.2 A)
and 5W3F (yeast microtubule, 3.7 A; the lattice dimer):
two neighbouring protofilaments give the lateral screw (~28 deg, ~0.9 nm). The lattice is the ideal 13-protofilament
3-start B-lattice (360/13 deg and 1.5 dimers / 13 per protofilament, seam between protofilaments 12 and 0) with an
8.3 nm dimer repeat; +z points to the plus end (alpha -> beta). Library types:
  MT_tubulin   alpha/beta-tubulin (TUB1/TUB2) from 5W3F, axis-aligned frame like F_actin
  gTuRC_nuc    gamma-TuRC: 7 gamma-TuSCs (TUB4 x2, SPC97, SPC98) + SPC110 fragments, capping nuclear minus ends
  gTuRC_cyto   the same ring without SPC110 (cytoplasmic MTs are anchored by SPC72, not in 8QV3)

Layout: minus ends on the SPB plaques (inner plaque for nuclear MTs, outer for cytoplasmic), MTs straight (persistence
length ~mm). N_KMT kinetochore MTs (one per chromosome; their plus ends seed the chromosome walks, Rabl-like), a few
other nuclear MTs and N_CYTO cytoplasmic MTs. Each MT grows layer by layer until its target length or until it meets
a membrane or an obstacle. Lengths are G1 estimates (EM reconstructions of yeast SPBs, e.g. Winey et al. 1995) and the
total polymer is capped at POLYMER_CAP of the alpha/beta-tubulin pool.
"""
import os, json
import numpy as np
import pack3d as p3
import cell3d as c3
import tiles3d as t3
import build_structs as bs
from config import LIB

LIB_MT, LIB_RING_NUC, LIB_RING_CYTO = "MT_tubulin", "gTuRC_nuc", "gTuRC_cyto"
LIB_IDS = (LIB_MT, LIB_RING_NUC, LIB_RING_CYTO)
NPF = 13
DIMER = 8.3                 # nm, dimer repeat: the shortest at which 5W3F dimers stack without interpenetrating
                            # (only the alpha N-terminal Met within 0.3 nm); yeast lattices are 8.2-8.4 nm
N_KMT, KMT_LEN = 16, (250.0, 60.0)          # kinetochore MTs: mean, sd length (nm)
N_NUC, NUC_LEN = 4, (400.0, 100.0)          # other nuclear MTs
N_CYTO, CYTO_LEN = 3, (1200.0, 300.0)       # cytoplasmic MTs
MIN_LEN = 60.0
POLYMER_CAP = 0.7           # at most this share of alpha/beta-tubulin dimers in MTs
PLAQUE_R = 90.0             # radius over which minus ends are spread on an SPB plaque
PLAQUE_GAP = 25.0           # ring centre distance from the membrane midsurface
CONE, CONE_CYTO = np.radians(30.0), np.radians(70.0)   # splay of MTs at the plaque edge
BEND = 8.2 / 300.0          # max bend per dimer layer (radius of curvature >= 300 nm)
LOOK, SAFE = 40.0, 60.0     # steer away from a membrane closer than SAFE, LOOK nm ahead of the tip
H = 20.0
PARAM_F = f"{LIB}/{LIB_MT}_lattice.json"
TUBULIN_ING = "CPX_1424_Tubulin_alpha_beta_heterodimeric"
TUSC_ING = "CPX_1198_Gamma_tubulin_small_complex"
RING_GENES = {"TUB4": 13, "SPC97": 7, "SPC98": 6}       # per ring; SPC110 x14 in nuclear rings


def _ca(st, c):
    return {r.seqid.num: np.array(r.find_atom("CA", "*").pos.tolist()) / 10 for r in st[0][c] if r.find_atom("CA", "*")}


def _screw(st, a, b):
    from scipy.spatial.transform import Rotation
    A, B = {}, {}
    for x, y in zip(a, b):
        da, db = _ca(st, x), _ca(st, y)
        for i in sorted(set(da) & set(db)):
            A[(x, i)] = da[i]; B[(x, i)] = db[i]
    X = np.array(list(A.values())); Y = np.array(list(B.values()))
    pa, pb = X - X.mean(0), Y - Y.mean(0)
    U, S, Vt = np.linalg.svd(pa.T @ pb); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T; t = Y.mean(0) - R @ X.mean(0)
    rv = Rotation.from_matrix(R).as_rotvec()
    return R, t, rv / np.linalg.norm(rv), np.linalg.norm(rv)


def model():
    """library entries + lattice parameters (MT frame: axis = z through the origin, plus end +z,
    protofilament 0 dimer 0 centroid at (r0, 0, 0))"""
    if os.path.exists(PARAM_F) and all(os.path.exists(f"{LIB}/{x}.npz") for x in LIB_IDS):
        return json.load(open(PARAM_F))
    import gemmi, extend_complexes as xc, structlib as sl
    st = gemmi.read_structure(xc.fetch_assembly("8QV3"))
    # chains: gamma-tubulin c/d, SPC97 E, SPC98 F, SPC110 Sc/Sd, alpha Ac/Ad, beta Bc/Bd (protofilaments c -> d)
    R, t, u, ang = _screw(st, ["Ac", "Bc"], ["Ad", "Bd"])
    xa = np.array(list(_ca(st, "Ac").values())); xb = np.array(list(_ca(st, "Bc").values()))
    if (xb.mean(0) - xa.mean(0)) @ u < 0:      # +axis toward the plus end (alpha -> beta)
        u, ang = -u, -ang
    rise = float(t @ u)
    p0 = np.linalg.lstsq(np.eye(3) - R, t - (t @ u) * u, rcond=None)[0]   # a point on the screw axis
    # lattice subunit: the 3.7 A yeast MT dimer 5W3F superposed on 8QV3's first dimer (8QV3's 8 A model is ~4% longer
    # along the axis: alpha-beta 4.35 vs 4.18 nm, so its dimers would interpenetrate at a normal repeat)
    s5 = gemmi.read_structure(xc.fetch_assembly("5W3F"))
    P, Q = [], []
    for c5, c8 in (("A", "Ac"), ("B", "Bc")):
        d5, d8 = _ca(s5, c5), _ca(st, c8)
        for i in sorted(set(d5) & set(d8)):
            P.append(d5[i]); Q.append(d8[i])
    P, Q = np.array(P), np.array(Q)
    U, S_, Vt = np.linalg.svd((P - P.mean(0)).T @ (Q - Q.mean(0))); dd = np.sign(np.linalg.det(Vt.T @ U.T))
    R5 = Vt.T @ np.diag([1, 1, dd]) @ U.T; t5 = Q.mean(0) - R5 @ P.mean(0)
    rms5 = float(np.sqrt((((P @ R5.T + t5) - Q) ** 2).sum(1).mean()))
    xyz, kind, cid5 = sl.beads_from_structure(s5, chains=["A", "B"]); xyz = xyz @ R5.T + t5; cid5 = np.array(cid5)
    # protofilaments run parallel to the axis in a 13_3 lattice: take the axis direction from the dimer (alpha -> beta),
    # through the screw-axis point (the two-protofilament screw tilts it by ~1.7 deg)
    u_screw = u; u = xyz[cid5 == "B"].mean(0) - xyz[cid5 == "A"].mean(0); u /= np.linalg.norm(u)
    c = xyz.mean(0)
    rvec = (c - p0) - ((c - p0) @ u) * u; r0 = float(np.linalg.norm(rvec))
    ex = rvec / r0; F = np.c_[ex, np.cross(u, ex), u]
    o = p0 + ((c - p0) @ u) * u                 # MT-frame origin: on the axis, level with dimer 0
    phi = np.sign(ang) * 2 * np.pi / NPF; dz = np.sign(rise) * 1.5 * DIMER / NPF
    bs.save(LIB_MT, (xyz - c) @ F, kind, extra=dict(pdb="5W3F", src="PDB", nchain=2), per=30)
    # ring: the gamma-TuSC of 8QV3 repeated by two protofilament steps (7 copies, 14 gamma-tubulin positions)
    for lid, chains in ((LIB_RING_NUC, ["c", "d", "E", "F", "Sc", "Sd"]), (LIB_RING_CYTO, ["c", "d", "E", "F"])):
        xyz, kind, cid = sl.beads_from_structure(st, chains=chains)
        X = (xyz - o) @ F; cid = np.array(cid)
        parts, kinds = [], []
        for m in range(7):
            # the 7th gamma-TuSC's second half (gamma-tubulin 14 + SPC98) overlaps the first one in the native ring;
            # on the ideal lattice helix it would sit inside protofilament 0, so it is left out
            keep = ~np.isin(cid, ["d", "F"]) if m == 6 else np.ones(len(cid), bool)
            a = 2 * m * phi; Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
            parts.append(X[keep] @ Rz.T + [0, 0, 2 * m * dz]); kinds.append(kind[keep])
        X = np.concatenate(parts); K = np.concatenate(kinds); rc = X.mean(0)
        bs.save(lid, X - rc, K, extra=dict(pdb="8QV3", src="PDB", nchain=7 * len(chains) - 2), per=30)
        if lid == LIB_RING_NUC:
            ring_c = rc
    prm = dict(phi_deg=float(np.degrees(phi)), dz=float(dz), dimer=DIMER, r0=r0, ring_c=[float(v) for v in ring_c],
               measured=dict(phi_deg=float(np.degrees(ang)), rise=rise, rmsd_5W3F_on_8QV3=rms5,
                             axis_tilt_deg=float(np.degrees(np.arccos(min(1.0, abs(u @ u_screw))))),
                             alpha_beta_axial=float((xb.mean(0) - xa.mean(0)) @ u)))
    json.dump(prm, open(PARAM_F, "w"))
    return prm


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _quat(Rm):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(Rm).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def _frame(w, rng):
    e1 = np.cross(w, rng.normal(size=3)); e1 /= np.linalg.norm(e1)
    return np.c_[e1, np.cross(w, e1), w]


def place(cell, lp, n_dimers, obstacles=None, seed=31, log=print):
    """MTs + rings. n_dimers: alpha/beta-tubulin dimer pool. Returns instances (mt_pos/q/type/reg), kMT tips."""
    prm = model()
    li = {k: int(np.where(lp["ids"] == k)[0][0]) for k in LIB_IDS}
    phi, dz, D, r0 = np.radians(prm["phi_deg"]), prm["dz"], prm["dimer"], prm["r0"]
    ring_c = np.array(prm["ring_c"])
    rng = np.random.default_rng(seed)
    G = cell.G; cn = G[3:6]
    nrm = G[7:10] - cn; nrm /= np.linalg.norm(nrm)
    spb_dir = -nrm                                   # SPB opposite the nucleolus (as in the chromatin pre-pass)
    _, n_out = c3.nuc_surface(G, spb_dir)            # NE normal at the SPB
    plaques = {"nuc": (c3.nuc_surface(G, spb_dir, -(c3.PERI_W + PLAQUE_GAP))[0], -n_out, 1 << c3.NUCP, t3.R_NUCP, LIB_RING_NUC),
               "cyto": (c3.nuc_surface(G, spb_dir, PLAQUE_GAP)[0], n_out, 1 << c3.CYTO, t3.R_CYTO, LIB_RING_CYTO)}
    # target lengths, scaled down if the pool is short
    plan = [("nuc", "kMT", rng.normal(*KMT_LEN)) for _ in range(N_KMT)] + \
           [("nuc", "nMT", rng.normal(*NUC_LEN)) for _ in range(N_NUC)] + [("cyto", "cMT", rng.normal(*CYTO_LEN)) for _ in range(N_CYTO)]
    need = sum(max(L, MIN_LEN) / D * NPF for _, _, L in plan)
    scale = min(1.0, POLYMER_CAP * n_dimers / need)
    glo = np.full(3, -G[0] - 10.0); ghi = -glo
    a = cell.args(glo, ghi, margin=50.0)
    cap = int(need * 1.2) * 40 + 100000 + (len(obstacles[1]) if obstacles is not None else 0)
    grid = p3.grid_new(glo, ghi, H, cap, cap * 14)
    if obstacles is not None and len(obstacles[1]):
        OC, OR = obstacles
        k = np.all((OC > glo) & (OC < ghi), axis=1)
        p3.g_insert(glo, H, *grid, OC[k], OR[k])
    sub = {}
    for k, v in li.items():
        s0, s1 = lp["st_off"][v], lp["st_off"][v + 1]
        sub[k] = (lp["STC"][s0:s1], lp["STR"][s0:s1], float(lp["BR"][v]))
    out = np.zeros(c3.NMEM); M = np.empty((3, 3))

    def body(lid, centre, Rm, mask):
        stc, str_, br = sub[lid]
        P = np.empty((len(stc), 3)); p3.quat_mat(_quat(Rm), M); p3.transform(stc, M, centre, P)
        if p3.g_overlap(glo, H, *grid[:6], P, str_, p3.TOL):
            return None
        if not p3.check_volume(P, str_, centre[0], centre[1], centre[2], br, mask, *a, out):
            return None
        return P

    def clearance(p):
        c3.mem_sdfs(p[0], p[1], p[2], *a, out)
        return c3.min_mem_dist(out, -1)

    def steer(p, w):
        """bend toward more membrane clearance (at most BEND per layer) when a membrane is within SAFE ahead"""
        q = p + w * LOOK
        if clearance(q) > SAFE:
            return w
        g = np.array([clearance(q + e * 4.0) - clearance(q - e * 4.0) for e in np.eye(3)])
        g -= (g @ w) * w; n = np.linalg.norm(g)
        if n < 1e-6:
            return w
        w2 = np.cos(BEND) * w + np.sin(BEND) * g / n
        return w2 / np.linalg.norm(w2)

    def grow(O, w, ring, mask, nlay):
        W = _frame(w, rng)
        Pr = body(ring, O + W @ ring_c, W, mask)
        if Pr is None:
            return None, O, W
        got = [(ring, O + W @ ring_c, W, Pr)]; Oj = O.copy()
        for j in range(nlay):
            layer = []
            for k in range(NPF):
                Rk = W @ _rz(k * phi)
                cen = Oj + Rk @ np.array([r0, 0.0, 0.0]) + W[:, 2] * (k * dz)
                P = body(LIB_MT, cen, Rk, mask)
                if P is None:
                    return got, Oj, W
                layer.append((LIB_MT, cen, Rk, P))
            got += layer
            Oj = Oj + W[:, 2] * D
            w2 = steer(Oj, W[:, 2])                  # parallel-transport the frame to the new direction
            ax = np.cross(W[:, 2], w2); sn = np.linalg.norm(ax)
            if sn > 1e-9:
                from scipy.spatial.transform import Rotation
                W = Rotation.from_rotvec(ax / sn * np.arcsin(min(sn, 1.0))).as_matrix() @ W
        return got, Oj, W

    pos, rot, typ, reg, mt_id = [], [], [], [], []
    tips, kinds, lengths = [], [], []
    for side, kind, L in plan:
        L = max(MIN_LEN, L * scale); nlay = int(round(L / D))
        centre0, normal, mask, region, ring = plaques[side]
        Wn = _frame(normal, rng)
        best = None
        for _ in range(80):                          # random minus-end site; MTs splay out from the plaque (bouquet)
            r = PLAQUE_R * np.sqrt(rng.random()); th = rng.uniform(0, 2 * np.pi)
            radial = np.cos(th) * Wn[:, 0] + np.sin(th) * Wn[:, 1]
            O = centre0 + r * radial + normal * rng.uniform(0, 15.0)
            cone = CONE if side == "nuc" else CONE_CYTO
            w = normal + np.tan(cone) * (r / PLAQUE_R) * radial + 0.15 * rng.normal(size=3); w /= np.linalg.norm(w)
            got, Oe, We = grow(O, w, ring, mask, nlay)
            if got is None:
                continue
            if best is None or len(got) > len(best[0]):
                best = (got, Oe, We)
            if len(got) - 1 >= nlay * NPF:
                break
        if best is None or (len(best[0]) - 1) / NPF * D < MIN_LEN:
            continue
        got, Oe, We = best
        for lid, cen, Rm, P in got:
            p3.g_insert(glo, H, *grid, P, sub[lid][1])
            pos.append(cen); rot.append(_quat(Rm)); typ.append(li[lid]); reg.append(region); mt_id.append(len(lengths))
        n_l = (len(got) - 1) // NPF
        tips.append(Oe + We[:, 2] * 12.0); kinds.append(kind); lengths.append(n_l * D)
    # chromosomes without a kinetochore MT start next to the inner plaque
    while sum(k == "kMT" for k in kinds) < N_KMT:
        tips.append(plaques["nuc"][0] + plaques["nuc"][1] * 40.0); kinds.append("kMT"); lengths.append(np.nan)
    lengths = np.array(lengths); kinds = np.array(kinds)
    kmt_tips = np.array([t for t, k in zip(tips, kinds) if k == "kMT"], np.float64).reshape(-1, 3)
    keep = ~np.isnan(lengths); lengths, kinds = lengths[keep], kinds[keep]
    n_tub = int(sum(t == li[LIB_MT] for t in typ))
    log(f"microtubules: {len(lengths)} MTs ({', '.join(f'{k} {int((kinds == k).sum())}' for k in ('kMT', 'nMT', 'cMT'))}), "
        f"{n_tub} tubulin dimers ({n_tub / max(n_dimers, 1):.0%} of pool, length scale {scale:.2f}); "
        f"mean length kMT {lengths[kinds == 'kMT'].mean() if (kinds == 'kMT').any() else 0:.0f} nm, "
        f"cMT {lengths[kinds == 'cMT'].mean() if (kinds == 'cMT').any() else 0:.0f} nm; lattice {prm}")
    return dict(mt_pos=np.array(pos, np.float32), mt_q=np.array(rot, np.float32), mt_type=np.array(typ, np.int32),
                mt_reg=np.array(reg, np.int16), mt_id=np.array(mt_id, np.int32), mt_len=lengths.astype(np.float32),
                mt_kind=kinds.astype("U3"), kmt_tips=kmt_tips,
                n_rings_nuc=int(sum(t == li[LIB_RING_NUC] for t in typ)), n_rings_cyto=int(sum(t == li[LIB_RING_CYTO] for t in typ)),
                n_tub=n_tub)
