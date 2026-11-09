"""Cytoskeleton pre-pass for the Jurkat cell (cell 1): centrosome, microtubules (MTs) and cortical F-actin.

Decisions (PLANNING_NOTES.md §4c / §7a): total MT length 2.8 mm (GT-annotation density 6.8 um/um3 x cytosol), ~560 MTs
of ~5 um nucleated at the centrosome; F-actin = 50% of ACTB+ACTG1 actin, in a cortical band 8-150 nm below the PM.

Centrosome (not segmented): the most open cytosol point 300-800 nm from the outer nuclear membrane within 2 um of the
(procedural) Golgi centroid, score = min(membrane clearance excluding the NE, 600 nm) - 0.25 x distance to the Golgi
centroid. Two orthogonal centrioles (mother 450 nm, daughter 250 nm long, 9 triplets at 85 nm radius) built from the
same MT lattice (complete 13-protofilament tubes for the A/B/C microtubules: a simplification).

Libraries (new entries in lib/, human sequences on curated lattices):
  MT_TUBA1B-TUBB4B : alpha/beta-tubulin dimer, AFDB v6 TUBA1B (P68363) and TUBB4B (P68371) monomers (pLDDT >= 50)
                     superposed (sequence-aligned Calpha, Kabsch) on 5W3F chains A/B placed on the 8QV3 lattice (as in
                     the yeast microtubules.model); MT frame: x radial, z = axis toward the plus end (alpha -> beta)
  F_ACTB           : ACTB (P60709) AFDB monomer superposed on a 9GO5 (yeast ADP-F-actin, 2.5 A) subunit; filament frame
                     x radial, z = filament axis; helix parameters from 9GO5 (rise ~2.76 nm, twist ~-166.6 deg)
MT lattice: ideal 13_3 B-lattice, DIMER repeat, phi = 360/13 (sign from 8QV3), dz = 1.5 DIMER / 13.

Placement (whole cell, model frame of jgeom_cell1.npz):
  - MT paths grow from minus ends on a 450-900 nm shell around the centrosome (initial direction ~radial), in chunks of 8 dimer layers; persistence
    length LP_MT; a chunk is accepted while every axis point is cytosol with membrane clearance >= CLR_MT
    (min_m |d_m| - MEM_HALF_m) and its axis spheres (R_MT_AX) do not overlap earlier MTs/centrioles; a PM-limited
    failure ends the MT at the cortex, other failures steer toward larger clearance (<= BEND per layer; 0.4 rad
    random turn after an MT collision); after a blocked chunk, up to 8 candidate turns are tested on the next
    chunk (look-ahead) and the first free one is taken; an MT ends after MAX_FAILS failed chunks.
    MTs are added until the total length target (or the dimer pool) is reached.
  - F-actin: straight-ish filaments (persistence 10 um), 200-1000 nm, seeded in the band 40-110 nm below the PM and
    oriented tangentially; truncated where they leave the 8-150 nm band, come within 6 nm of another membrane or hit
    earlier filaments/MTs (axis spheres R_ACT_AX every 2 subunits); kept if >= 100 nm. Added until the F-actin target.
Collision representation is coarse (filament axis spheres), adequate within this module; clashes with chromatin/
polysomes/other pre-pass content are resolved at integration.
Output: data/derived/prepass/cytoskeleton_cell1.{npz,json}, data/derived/qc/cytoskeleton_cell1.png
usage: NUMBA_NUM_THREADS=8 python jcytoskeleton.py [mt_total_um=2800] [factin_frac=0.5] [seed=31]
"""
import os, sys, json, time, faulthandler
faulthandler.enable()
import numpy as np
import pandas as pd
import gemmi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3, pack3d as p3, build_structs as bs, structlib as sl, tm_graft
from config import LIB

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"
OUT = f"{D}/prepass"
LIB_MT, LIB_ACT = "MT_TUBA1B-TUBB4B", "F_ACTB"
ING_TUB, ING_ACT = "TUBA1B-TUBB4B", "ACTB"
NPF = 13
DIMER = 8.2                  # nm, mammalian GDP-lattice dimer repeat (checked for bead overlap in mt_model)
LP_MT = 20000.0              # nm, effective MT persistence length in cytoplasm (bending, not thermal)
CLR_MT = float(os.environ.get("CLR_MT", 30.0))   # nm, axis clearance to any membrane surface (MT outer radius 12.5)
R_MT_AX = 12.5
BEND = DIMER / 300.0         # max steering bend per layer (radius of curvature >= 300 nm)
CHUNK = 8
MAX_FAILS = int(os.environ.get("MAX_FAILS", 40))
SHELL = (300.0, 500.0)       # nm, pericentriolar nucleation shell (v2)
L_MAX = 20000.0              # nm, maximal MT length (v2)
CORTEX_D = 300.0             # nm below the PM: switch to cortex-following (v2)
CORTEX_BAND = (150.0, 300.0) # nm, depth held along the cortex (inside the F-actin band would be 8-150 nm)
RADIAL_BIAS = 0.004          # per-layer radial pull (v2)
N_REROUTE = 32
MAX_BACK = 4                 # progressive backtracks (1, 2, ... chunks) before an MT ends (v2)
MIN_KEEP = 1500.0            # nm: shorter MTs are discarded (directions that cannot grow are not kept)
OUTWARD = 1.0                # weight of the radial (away from centrosome) component in escape turns
MT_MEAN, MT_SD = 5000.0, 2000.0
N_MT_MAX = 3000
R_ACT_AX = 3.6
ACT_BAND = (8.0, 150.0)
ACT_SEED = (40.0, 110.0)
ACT_LEN = (200.0, 1000.0)
LP_ACT = 10000.0
VOXS2 = np.array([16.0, 16.0, 13.76])
H = 30.0                     # nm, collision grid spacing
DBG = {}


# ------------------------------------------------------------------ structures
def _ca_chain(st, cname):
    out = []
    for r in st[0][cname]:
        a = r.find_atom("CA", "*")
        if a is not None:
            out.append((r.seqid.num, gemmi.find_tabulated_residue(r.name).one_letter_code.upper(), np.array(a.pos.tolist()) / 10))
    return out


def _align_map(qseq, tseq):
    """pairs (iq, it) of aligned residue indices from a gemmi global alignment"""
    al = gemmi.align_string_sequences(list(qseq), list(tseq), [])
    iq = it = 0; pairs = []
    import re
    for n, op in re.findall(r"(\d+)([MID])", al.cigar_str()):
        n = int(n)
        if op == "M":
            pairs += [(iq + k, it + k) for k in range(n)]; iq += n; it += n
        elif op == "I":
            iq += n
        else:
            it += n
    return pairs, al.calculate_identity()


def _kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc)); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def _human_onto(acc, tmpl):
    """AFDB monomer of acc (pLDDT >= 50 kept) superposed on template residues tmpl [(num, aa, xyz)]; returns beads, rmsd, id"""
    pos, xyz, pl = tm_graft.af_model(acc)
    st_seq = None
    # AF sequence from the tar model is not returned by af_model; rebuild it from the model file via the index
    import gzip, json as _j
    idx = _j.load(open(tm_graft.IDX)); off, size = idx[acc]
    with open(tm_graft.TAR, "rb") as f:
        f.seek(off); data = gzip.decompress(f.read(size)).decode()
    st = gemmi.make_structure_from_block(gemmi.cif.read_string(data)[0])
    hum = _ca_chain(st, st[0][0].name)
    hseq = "".join(a for _, a, _ in hum); tseq = "".join(a for _, a, _ in tmpl)
    pairs, ident = _align_map(hseq, tseq)
    hp = np.array([hum[i][2] for i, j in pairs]); tp = np.array([tmpl[j][2] for i, j in pairs])
    R, t = _kabsch(hp, tp)
    for _ in range(3):                                       # trim outliers (flexible loops/tails) and refit
        d = np.linalg.norm(hp @ R.T + t - tp, axis=1); k = d < max(0.2, np.percentile(d, 85))
        R, t = _kabsch(hp[k], tp[k])
    rmsd = float(np.sqrt(((hp @ R.T + t - tp) ** 2).sum(1).mean()) * 10)
    pl_map = dict(zip(pos, pl))
    keep = np.array([pl_map.get(n, 100) >= 50 for n, _, _ in hum])
    beads = np.array([x for _, _, x in hum])[keep] @ R.T + t
    return beads, rmsd, ident


def mt_model():
    """MT_TUBA1B-TUBB4B library entry + lattice parameters (yeast microtubules.model geometry, human subunits)"""
    pf = f"{LIB}/{LIB_MT}_lattice.json"
    if os.path.exists(pf) and os.path.exists(f"{LIB}/{LIB_MT}.npz"):
        return json.load(open(pf))
    import jlib_build as jb
    from scipy.spatial.transform import Rotation
    st = gemmi.read_structure(jb.pdb_file("8qv3")); s5 = gemmi.read_structure(jb.pdb_file("5w3f"))

    def ca(s, c):
        return {r.seqid.num: np.array(r.find_atom("CA", "*").pos.tolist()) / 10 for r in s[0][c] if r.find_atom("CA", "*")}
    # screw between neighbouring protofilaments c -> d (alpha Ac/Ad, beta Bc/Bd)
    A, B = [], []
    for x, y in (("Ac", "Ad"), ("Bc", "Bd")):
        da, db = ca(st, x), ca(st, y)
        for i in sorted(set(da) & set(db)):
            A.append(da[i]); B.append(db[i])
    X, Y = np.array(A), np.array(B)
    R, t = _kabsch(X, Y)
    rv = Rotation.from_matrix(R).as_rotvec(); ang = np.linalg.norm(rv); u = rv / ang
    xa = np.array(list(ca(st, "Ac").values())); xb = np.array(list(ca(st, "Bc").values()))
    if (xb.mean(0) - xa.mean(0)) @ u < 0:
        u, ang = -u, -ang
    rise = float(t @ u)
    p0 = np.linalg.lstsq(np.eye(3) - R, t - (t @ u) * u, rcond=None)[0]
    # 5W3F dimer superposed on 8QV3's first dimer (Ac, Bc)
    P, Q = [], []
    for c5, c8 in (("A", "Ac"), ("B", "Bc")):
        d5, d8 = ca(s5, c5), ca(st, c8)
        for i in sorted(set(d5) & set(d8)):
            P.append(d5[i]); Q.append(d8[i])
    R5, t5 = _kabsch(np.array(P), np.array(Q))
    # human subunits onto the (placed) 5W3F chains
    parts, info = [], {}
    for acc, ch in (("P68363", "A"), ("P68371", "B")):
        tmpl = [(n, a, x @ R5.T + t5) for n, a, x in _ca_chain(s5, ch)]
        beads, rmsd, ident = _human_onto(acc, tmpl)
        parts.append(beads); info[acc] = dict(rmsd_A=round(rmsd, 2), identity=round(ident, 1), beads=len(beads))
    xyz = np.vstack(parts)
    u_ax = parts[1].mean(0) - parts[0].mean(0); u_ax /= np.linalg.norm(u_ax)
    c = xyz.mean(0)
    rvec = (c - p0) - ((c - p0) @ u_ax) * u_ax; r0 = float(np.linalg.norm(rvec))
    ex = rvec / r0; F = np.c_[ex, np.cross(u_ax, ex), u_ax]
    local = (xyz - c) @ F
    phi = np.sign(ang) * 2 * np.pi / NPF
    # dimer repeat fixed at the mammalian 8.2 nm; AF loops at the longitudinal interface leave a few close Calpha pairs
    from scipy.spatial import cKDTree
    rep = DIMER
    dmin = cKDTree(local).query(local + [0, 0, rep])[0]
    clash = dict(min_CA_CA_nm=round(float(dmin.min()), 2), n_lt_0p5nm=int((dmin < 0.5).sum()))
    dz = np.sign(rise) * 1.5 * rep / NPF
    bs.save(LIB_MT, local, np.zeros(len(local), np.int8),
            extra=dict(src="lattice 8QV3/5W3F; AFDB P68363+P68371 superposed", nchain=2), per=30)
    prm = dict(phi_deg=float(np.degrees(phi)), dz=float(dz), dimer=rep, r0=r0, measured_rise=rise,
               measured_phi_deg=float(np.degrees(ang)), superposition=info, longitudinal_contacts=clash)
    json.dump(prm, open(pf, "w"), indent=1)
    return prm


def actin_model():
    pf = f"{LIB}/{LIB_ACT}_helix.json"
    if os.path.exists(pf) and os.path.exists(f"{LIB}/{LIB_ACT}.npz"):
        return json.load(open(pf))
    import jlib_build as jb
    from scipy.spatial.transform import Rotation
    st = gemmi.read_structure(jb.pdb_file("9go5"))
    ch = {}
    for cc in st[0]:
        d = [(r.seqid.num, gemmi.find_tabulated_residue(r.name).one_letter_code.upper(), np.array(r.find_atom("CA", "*").pos.tolist()) / 10)
             for r in cc if r.find_atom("CA", "*") and gemmi.find_tabulated_residue(r.name) and gemmi.find_tabulated_residue(r.name).is_amino_acid()]
        if len(d) > 300:
            ch[cc.name] = d
    names = list(ch)
    common = sorted(set.intersection(*[set(n for n, _, _ in d) for d in ch.values()]))
    X = {n: np.array([x for k, _, x in ch[n] if k in set(common)]) for n in names}
    best = None
    for a in names:
        for b in names:
            if a == b:
                continue
            R, t = _kabsch(X[a], X[b])
            rv = Rotation.from_matrix(R).as_rotvec(); ang = np.linalg.norm(rv); u = rv / ang
            if abs(abs(t @ u) - 2.76) < 0.3:
                if t @ u < 0:
                    u, ang = -u, -ang
                best = (a, R, t, u, ang); break
        if best:
            break
    a, R, t, u, ang = best
    p0 = np.linalg.lstsq(np.eye(3) - R, t - (t @ u) * u, rcond=None)[0]
    beads, rmsd, ident = _human_onto("P60709", ch[a])
    c = beads.mean(0)
    r = (c - p0) - ((c - p0) @ u) * u; r0 = float(np.linalg.norm(r))
    ex = r / r0; F = np.c_[ex, np.cross(u, ex), u]
    local = (beads - c) @ F
    bs.save(LIB_ACT, local, np.zeros(len(local), np.int8), extra=dict(src="helix 9GO5; AFDB P60709 superposed"), per=30)
    prm = dict(phi_deg=float(np.degrees(ang)), rise=float(t @ u), r0=r0, superposition=dict(rmsd_A=round(rmsd, 2), identity=round(ident, 1)))
    json.dump(prm, open(pf, "w"), indent=1)
    return prm


# ------------------------------------------------------------------ geometry helpers
def quats(Rm):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(Rm).as_quat()
    q = np.c_[q[:, 3], q[:, 0], q[:, 1], q[:, 2]]
    q[q[:, 0] < 0] *= -1
    return q


def rz_stack(angles):
    c, s = np.cos(angles), np.sin(angles)
    Rz = np.zeros((len(angles), 3, 3)); Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s; Rz[:, 1, 0] = s; Rz[:, 1, 1] = c; Rz[:, 2, 2] = 1
    return Rz


def perp(w, rng):
    e = np.cross(w, rng.normal(size=3)); return e / np.linalg.norm(e)


def orthonormal(W):
    """re-orthonormalise a frame (columns x, y, z) keeping z"""
    z = W[:, 2] / np.linalg.norm(W[:, 2]); x = W[:, 0] - (W[:, 0] @ z) * z; x /= np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z]


def rot_between(a, b):
    """rotation matrix taking unit a to unit b (Rodrigues)"""
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(a @ b)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        e = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]); e /= np.linalg.norm(e)
        return 2 * np.outer(e, e) - np.eye(3)                  # proper 180 deg rotation about e
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    return np.eye(3) + s * K + (1 - c) * K @ K


class Geo:
    def __init__(self, cell):
        self.cell = cell

    def mem(self, P):
        g = self.cell.mems(np.atleast_2d(P).astype(np.float64))
        clr = np.abs(g[:, :c3.NMEM - 1]) - c3.MEM_HALF[:c3.NMEM - 1]
        return g, clr

    def lab(self, P):
        return self.cell.labels(np.atleast_2d(P).astype(np.float64))


# ------------------------------------------------------------------ centrosome
def centrosome(cell, geo, centre_nm, rng):
    """most open cytosol point 300-800 nm from the nucleus within 2 um of the Golgi centroid. Distances from Euclidean
    distance transforms of the 16 nm label grid (cropped around the Golgi; SDF fields only hold narrow bands)"""
    from scipy import ndimage as ndi
    z = np.load(f"{D}/cell1_labels_s2.npz"); L = z["labels"]; o2 = z["origin_s2_xyz"]
    gi = np.argwhere(L == 5); g0 = gi.mean(0)
    half = np.ceil(2600.0 / VOXS2).astype(int)
    lo = np.maximum(np.round(g0).astype(int) - half, 0); hi = np.minimum(np.round(g0).astype(int) + half, L.shape)
    sub = L[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    d_nuc = ndi.distance_transform_edt(sub != 2, sampling=VOXS2)
    d_open = ndi.distance_transform_edt(sub == 1, sampling=VOXS2)          # distance to the nearest non-cytosol voxel
    gc = (o2 + g0) * VOXS2 - centre_nm
    idx = np.argwhere((sub == 1) & (d_nuc >= 300) & (d_nuc <= 800))
    P = (o2 + lo + idx) * VOXS2 - centre_nm
    dg = np.linalg.norm(P - gc, axis=1); k = dg < 2000
    idx, P, dg = idx[k], P[k], dg[k]
    score = np.minimum(d_open[tuple(idx.T)], 600.0) - 0.25 * dg
    i = int(np.argmax(score)); c = P[i]
    if geo.lab(c)[0] != c3.CYTO:
        raise RuntimeError("centrosome candidate not in cytosol of the SDF geometry")
    return c, gc, dict(ne_dist_nm=float(d_nuc[tuple(idx[i])]), open_radius_nm=float(d_open[tuple(idx[i])]),
                       dist_golgi_centroid_nm=float(dg[i]), sdf_clearance_nm=float(geo.mem(c)[1].min()))


# ------------------------------------------------------------------ MT lattice
def lattice(axis_pts, frames, prm):
    """dimers of a 13-pf tube along axis points (one layer each) with frames (n,3,3: x,y,z columns)"""
    phi, dz, r0 = np.radians(prm["phi_deg"]), prm["dz"], prm["r0"]
    n = len(axis_pts); Rz = rz_stack(np.arange(NPF) * phi)
    Rk = np.einsum("nij,kjl->nkil", frames, Rz).reshape(-1, 3, 3)
    cen = (axis_pts[:, None, :] + np.einsum("nkij,j->nki", Rk.reshape(n, NPF, 3, 3), np.array([r0, 0, 0]))
           + frames[:, None, :, 2] * (np.arange(NPF) * dz)[None, :, None]).reshape(-1, 3)
    return cen, quats(Rk)


def centrioles(cen0, rng, prm):
    """mother (450 nm) and orthogonal daughter (250 nm): 9 triplets (A,B,C tubes) at radius 85 nm"""
    D_ = prm["dimer"]; out_c, out_q, ax_pts = [], [], []
    a1 = perp(rng.normal(size=3), rng); a2 = np.cross(a1, perp(a1, rng)); a2 /= np.linalg.norm(a2)
    for axis, length, off in ((a1, 450.0, np.zeros(3)), (a2, 250.0, a1 * 120.0 + perp(a1, rng) * 0)):
        e1 = perp(axis, rng); e2 = np.cross(axis, e1)
        base = cen0 + off - axis * length / 2
        nl = int(length / D_)
        for j in range(9):
            th = 2 * np.pi * j / 9
            for t, dr in enumerate((-0.0, 20.0, 40.0)):                 # A, B, C tubes stepping outward/sideways
                rad = 85.0 - 12.0 + dr * 0.6; tang = dr * 0.8
                c0 = base + (np.cos(th) * e1 + np.sin(th) * e2) * rad + (-np.sin(th) * e1 + np.cos(th) * e2) * tang
                pts = c0 + np.outer(np.arange(nl) * D_, axis)
                Fm = np.c_[np.cos(th) * e1 + np.sin(th) * e2, np.cross(axis, np.cos(th) * e1 + np.sin(th) * e2), axis]
                cc, qq = lattice(pts, np.repeat(Fm[None], nl, 0), prm)
                out_c.append(cc); out_q.append(qq); ax_pts.append(pts)
    return np.vstack(out_c), np.vstack(out_q), np.vstack(ax_pts)


# ------------------------------------------------------------------ growth
def grow_mts(geo, grid, glo, cen0, prm, target_nm, max_dimers, rng, log):
    """MT paths from the centrosome aster (see module docstring); steering never turns back toward the centrosome"""
    D_ = prm["dimer"]
    sig = np.sqrt(D_ / LP_MT)
    mts, total, n_dim, tried = [], 0.0, 0, 0
    reasons = {"cortex": 0, "steer_fail": 0, "target": 0, "nonfinite": 0, "blocked": 0}
    blocks = {}
    while total < target_nm and len(mts) < N_MT_MAX and tried < 30 * N_MT_MAX:
        tried += 1
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        p = cen0 + d * rng.uniform(450, 900)
        if geo.lab(p)[0] != c3.CYTO or geo.mem(p)[1].min() < CLR_MT:
            continue
        if p3.g_overlap(glo, H, *grid[:6], p[None].astype(np.float64), np.array([R_MT_AX]), p3.TOL):
            continue
        w = d + 0.1 * rng.normal(size=3); w /= np.linalg.norm(w)
        Wf = np.c_[perp(w, rng), np.zeros(3), w]; Wf[:, 1] = np.cross(w, Wf[:, 0])
        L_target = max(500.0, rng.normal(MT_MEAN, MT_SD))
        pts, frames = [p], [Wf.copy()]
        fails, pending, end = 0, [], None
        while len(pts) * D_ < L_target and fails < MAX_FAILS:
            prop, fr, ww, WW = [], [], w.copy(), Wf.copy()
            for _ in range(CHUNK):
                nw = ww + sig * np.cross(ww, rng.normal(size=3)); nw /= np.linalg.norm(nw)
                rr = (prop[-1] if prop else pts[-1]) - cen0; rr /= np.linalg.norm(rr)
                if nw @ rr < 0.2:                                            # keep growing away from the centrosome
                    nw = nw + 0.05 * rr; nw /= np.linalg.norm(nw)
                WW = orthonormal(rot_between(ww, nw) @ WW); ww = WW[:, 2].copy()
                prop.append((prop[-1] if prop else pts[-1]) + D_ * ww); fr.append(WW.copy())
            prop = np.array(prop)
            if not np.isfinite(prop).all():
                end = "nonfinite"; break
            g, clr = geo.mem(prop); lab = geo.lab(prop)
            bad = (clr.min(1) < CLR_MT) | (lab != c3.CYTO)
            f = int(np.argmax(bad)) if bad.any() else CHUNK
            coll = CHUNK
            for k in range(f):
                if p3.g_overlap(glo, H, *grid[:6], prop[k:k + 1], np.array([R_MT_AX]), p3.TOL):
                    coll = k; break
            n_ok = min(f, coll)
            pts += list(prop[:n_ok]); frames += fr[:n_ok]
            if n_ok:
                w, Wf = (prop[n_ok - 1] - (prop[n_ok - 2] if n_ok > 1 else pts[-n_ok - 1])), fr[n_ok - 1]
                w = Wf[:, 2].copy()
            if n_ok < CHUNK:
                if coll < f and os.environ.get("CYTO_DEBUG"):
                    q = prop[coll]; own = np.array(pts[:max(len(pts) - 4, 0)]) if len(pts) > 4 else np.zeros((0, 3))  # (self: not checked any more)
                    d_own = np.linalg.norm(own - q, axis=1).min() if len(own) else 1e9
                    d_cen = np.linalg.norm(DBG["cax"] - q, axis=1).min()
                    d_oth = min([np.linalg.norm(m_[0] - q, axis=1).min() for m_ in mts] + [1e9])
                    k_ = "self" if d_own < 25.3 else ("centriole" if d_cen < 25.3 else ("other_mt" if d_oth < 25.3 else "unknown"))
                    blocks["coll_" + k_] = blocks.get("coll_" + k_, 0) + 1
                why = "collision" if coll < f else ("label" if lab[f] != c3.CYTO else "mem:" + c3.MEM_NAMES[int(np.argmin(clr[f]))])
                blocks[why] = blocks.get(why, 0) + 1
                if f < coll and bad[f] and lab[f] == c3.CYTO and int(np.argmin(clr[f])) == c3.M_PM:
                    end = "cortex"; break                        # reached the cortex
                fails += 1
                tip = pts[-1]; found = False
                for _try in range(8):                                 # look-ahead: first candidate turn with a free path
                    turn = (0.15 + 0.08 * _try) if coll < f else BEND * CHUNK * (1 + _try)
                    rad = tip - cen0; rad /= np.linalg.norm(rad)                 # outward bias: no turning back into the aster
                    tgt = perp(w, rng) + OUTWARD * (rad - (rad @ w) * w); tgt -= (tgt @ w) * w; tgt /= np.linalg.norm(tgt)
                    nw = np.cos(turn) * w + np.sin(turn) * tgt; nw /= np.linalg.norm(nw)
                    if nw @ rad < 0.2:
                        continue
                    look = tip + np.outer(np.arange(1, CHUNK + 1) * D_, nw)
                    gl, cl = geo.mem(look); ll = geo.lab(look)
                    if (cl.min(1) >= CLR_MT).all() and (ll == c3.CYTO).all() and not any(
                            p3.g_overlap(glo, H, *grid[:6], look[k:k + 1], np.array([R_MT_AX]), p3.TOL) for k in range(0, CHUNK, 2)):
                        found = True; break
                if not np.isfinite(nw).all():
                    end = "nonfinite"; break
                Wf = orthonormal(rot_between(w, nw) @ Wf); w = Wf[:, 2].copy()
                if not found and fails >= MAX_FAILS // 4 and coll >= f:
                    end = "blocked"; break
        L = (len(pts) - 1) * D_
        if L < 300 or not np.isfinite(np.array(pts)).all():
            continue
        reasons[end or ("target" if len(pts) * D_ >= L_target else "steer_fail")] += 1
        nd = len(pts) * NPF
        if n_dim + nd > max_dimers:
            break
        P_ = np.array(pts)
        p3.g_insert(glo, H, *grid, P_, np.full(len(P_), R_MT_AX))      # only accepted MTs become obstacles
        mts.append((P_, np.array(frames))); total += L; n_dim += nd
        if len(mts) % 100 == 0:
            log(f"  MTs {len(mts)}, total {total / 1000:.0f} um, dimers {n_dim:,}, ends {reasons}")
    log("MT ends:", reasons, "| chunk blocks:", dict(sorted(blocks.items(), key=lambda kv: -kv[1])))
    return mts, total, n_dim


class Depth:
    """true depth below the plasma membrane (nm) from a Euclidean distance transform of the 16 nm cell mask (the PM SDF
    field is only stored within a narrow band and clamps at 160 nm inside the cell)"""
    def __init__(self, centre_nm):
        from scipy import ndimage as ndi
        z = np.load(f"{D}/cell1_labels_s2.npz"); L = z["labels"] > 0; self.o2 = z["origin_s2_xyz"]; self.c = centre_nm
        self.edt = ndi.distance_transform_edt(np.pad(L, 1), sampling=VOXS2)[1:-1, 1:-1, 1:-1].astype(np.float32)

    def __call__(self, P):
        from scipy.ndimage import map_coordinates
        P = np.atleast_2d(P)
        return map_coordinates(self.edt, ((P + self.c) / VOXS2 - self.o2).T, order=1, mode="nearest")

    def normal(self, p, e=24.0):
        g = np.array([self(p + e * u)[0] - self(p - e * u)[0] for u in np.eye(3)])
        return g / (np.linalg.norm(g) + 1e-12)                                  # points into the cell


def grow_mts_v2(geo, grid, glo, cen0, prm, target_nm, max_dimers, rng, log, depth=None):
    """Lymphocyte-like MT array (v2): MTs nucleate on a pericentriolar shell (SHELL nm) with radial directions and grow
    without a length target (<= L_MAX) in chunks of CHUNK layers: persistence LP_MT plus a weak radial bias; a chunk is
    accepted if every axis point is cytosol, >= CLR_MT from all membranes and outside other MTs (axis spheres of the
    real 12.5 nm radius, i.e. >= 25 nm centre spacing). Blocked chunks are re-routed by look-ahead: up to N_REROUTE
    candidate deflections (0.1-1.5 rad, random azimuth, no turning back toward the centrosome) are tested over 2 chunks
    and the first free one is taken; if none is free, up to MAX_BACK progressive backtracks (1, 2, ... chunks) are
    tried; then the MT ends. MTs shorter than MIN_KEEP are discarded. Within CORTEX_D nm of the PM (PM SDF depth) the MT switches to cortex mode: its direction is projected on
    the local cortex tangent plane, depth is held in CORTEX_BAND (just inside the F-actin band), and it continues for
    a random 0.5-3 um along the cortex. MTs are added until the total length target or the dimer pool is reached."""
    D_ = prm["dimer"]; sig = np.sqrt(D_ / LP_MT)
    mts, total, n_dim, tried = [], 0.0, 0, 0
    stats = {"cortex": 0, "blocked": 0, "max_len": 0}
    LAST = {"why": ""}; why_blocked = {}
    P = np.array([R_MT_AX])

    def depth_normal(p):
        return float(depth(p)[0]), depth.normal(p)

    def free(pts_, cortex):
        g, clr = geo.mem(pts_); lab = geo.lab(pts_)
        if cortex:
            clr = clr.copy(); dep = depth(pts_)
            okpm = (dep > CORTEX_BAND[0] - 40) & (dep < CORTEX_D + 200)
            other = np.delete(clr, c3.M_PM, axis=1).min(1) >= CLR_MT
            ok = okpm & other & (lab == c3.CYTO)
        else:
            ok = (clr.min(1) >= CLR_MT) & (lab == c3.CYTO)
        if not ok.all():
            i = int(np.argmin(ok))
            LAST["why"] = "label:" + c3.LABEL_NAMES[int(lab[i])] if lab[i] != c3.CYTO else "mem:" + c3.MEM_NAMES[int(np.argmin(clr[i]))]
            return i
        for k in range(len(pts_)):
            if p3.g_overlap(glo, H, *grid[:6], pts_[k:k + 1], P, p3.TOL):
                LAST["why"] = "collision"; return k
        return len(pts_)

    while total < target_nm and tried < 20000:
        tried += 1
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        p = cen0 + d * rng.uniform(*SHELL)
        if free(p[None], False) < 1:
            continue
        w = d + 0.1 * rng.normal(size=3); w /= np.linalg.norm(w)
        Wf = orthonormal(np.c_[perp(w, rng), np.zeros(3), w])
        pts, frames = [p], [Wf.copy()]
        cortex, cortex_left, end, backed = False, 0.0, None, 0
        while True:
            if (len(pts) - 1) * D_ > L_MAX:
                end = "max_len"; break
            dep, nrm = depth_normal(pts[-1])
            if not cortex and dep < CORTEX_D:
                cortex = True; cortex_left = rng.uniform(500, 3000)
            if cortex and cortex_left <= 0:
                end = "cortex"; break
            # proposal
            prop, fr, ww, WW = [], [], w.copy(), Wf.copy()
            for _ in range(CHUNK):
                q = prop[-1] if prop else pts[-1]
                nw = ww + sig * np.cross(ww, rng.normal(size=3))
                if cortex:
                    dq, nq = dep, nrm
                    nw = nw - (nw @ nq) * nq + 0.02 * nq * np.sign(np.mean(CORTEX_BAND) - dq)   # tangent + depth hold
                else:
                    rr = q - cen0; rr /= np.linalg.norm(rr); nw = nw + RADIAL_BIAS * rr
                nw /= np.linalg.norm(nw)
                WW = orthonormal(rot_between(ww, nw) @ WW); ww = WW[:, 2].copy()
                prop.append(q + D_ * ww); fr.append(WW.copy())
            prop = np.array(prop)
            n_ok = free(prop, cortex)
            if n_ok == CHUNK:
                pts += list(prop); frames += fr; w, Wf = ww, WW
                if cortex: cortex_left -= CHUNK * D_
                continue
            # re-route by look-ahead
            found = False
            for k in range(N_REROUTE):
                turn = 0.1 + 1.4 * k / N_REROUTE
                tgt = perp(w, rng); nw = np.cos(turn) * w + np.sin(turn) * tgt
                if cortex:
                    nw = nw - (nw @ nrm) * nrm
                else:
                    rr = pts[-1] - cen0; rr /= np.linalg.norm(rr)
                    if nw @ rr < -0.1:
                        continue
                nw /= np.linalg.norm(nw)
                look = pts[-1] + np.outer(np.arange(1, 2 * CHUNK + 1) * D_, nw)
                if free(look, cortex) == 2 * CHUNK:
                    found = True; break
            if found:
                Wf = orthonormal(rot_between(w, nw) @ Wf); w = Wf[:, 2].copy(); continue
            nb = (backed + 1) * CHUNK
            if backed < MAX_BACK and len(pts) > nb + CHUNK:
                del pts[-nb:]; del frames[-nb:]; Wf = frames[-1].copy(); w = Wf[:, 2].copy(); backed += 1
                if cortex and depth(pts[-1])[0] >= CORTEX_D:
                    cortex = False
                continue
            end = "cortex" if cortex else "blocked"
            if end == "blocked":
                why_blocked[LAST["why"]] = why_blocked.get(LAST["why"], 0) + 1
            break
        L = (len(pts) - 1) * D_
        P_ = np.array(pts)
        if L < MIN_KEEP or not np.isfinite(P_).all():
            continue
        nd = len(pts) * NPF
        if n_dim + nd > max_dimers:
            break
        p3.g_insert(glo, H, *grid, P_, np.full(len(P_), R_MT_AX))
        mts.append((P_, np.array(frames))); total += L; n_dim += nd; stats[end] += 1
        if len(mts) % 50 == 0:
            log(f"  MTs {len(mts)}, total {total / 1000:.0f} um, mean {total / len(mts):.0f} nm, ends {stats}, tries {tried}")
    log("MT ends:", stats, "tries", tried, "| blocked by:", dict(sorted(why_blocked.items(), key=lambda kv: -kv[1])))
    return mts, total, n_dim


def place_actin(geo, grid, glo, ghi, cell_bb, prm, n_target, rng, log):
    # pool of seed points in the cortical band
    seeds = []
    while sum(len(s) for s in seeds) < 400000:
        P = rng.uniform(cell_bb[0], cell_bb[1], (2_000_000, 3))
        g, clr = geo.mem(P)
        dep = g[:, c3.M_PM]
        other = np.delete(clr, c3.M_PM, axis=1).min(1)
        k = (dep > ACT_SEED[0]) & (dep < ACT_SEED[1]) & (other > 10)
        seeds.append(P[k])
    seeds = np.vstack(seeds); rng.shuffle(seeds)
    log(f"  actin seed pool {len(seeds):,}")
    rise, phi, r0 = prm["rise"], np.radians(prm["phi_deg"]), prm["r0"]
    sig = np.sqrt(rise / LP_ACT)
    cen_all, q_all, placed, nfil, si, tried = [], [], 0, 0, 0, 0
    lens = []
    while placed < n_target and tried < 20 * n_target / 200:
        tried += 1
        s = seeds[si % len(seeds)]; si += 1
        gp = s + np.vstack([np.eye(3) * 4.0, -np.eye(3) * 4.0])
        gg = geo.mem(gp)[0][:, c3.M_PM]; n = gg[:3] - gg[3:]; n /= (np.linalg.norm(n) + 1e-12)   # inward PM normal
        t = perp(n, rng)
        L = rng.uniform(*ACT_LEN); ns = int(L / rise)
        # straight-ish axis: small random bending
        dirs = np.empty((ns, 3)); w = t.copy()
        for k in range(ns):
            dirs[k] = w
            if k % 8 == 7:
                w = w + np.sqrt(8) * sig * np.cross(w, rng.normal(size=3)); w /= np.linalg.norm(w)
        ax = s + np.cumsum(dirs, 0) * rise - dirs[0] * rise
        g, clr = geo.mem(ax)
        dep = g[:, c3.M_PM]; other = np.delete(clr, c3.M_PM, axis=1).min(1)
        ok = (dep > ACT_BAND[0]) & (dep < ACT_BAND[1]) & (other > 6.0)
        f = int(np.argmin(ok)) if not ok.all() else ns
        chk = np.arange(0, f, 2)
        coll = f
        for k in chk:
            if p3.g_overlap(glo, H, *grid[:6], ax[k:k + 1], np.array([R_ACT_AX]), p3.TOL):
                coll = k; break
        m = min(f, coll)
        if m * rise < 100.0:
            continue
        ax = ax[:m]; dd = dirs[:m]
        p3.g_insert(glo, H, *grid, ax[::2].copy(), np.full(len(ax[::2]), R_ACT_AX))
        ex = np.cross(dd, n); ex /= np.linalg.norm(ex, axis=1, keepdims=True)
        Fm = np.stack([ex, np.cross(dd, ex), dd], axis=2)                    # columns x, y, z
        spin = rng.uniform(0, 2 * np.pi)
        Rk = np.einsum("nij,njk->nik", Fm, rz_stack(spin + phi * np.arange(m)))
        cen = ax + np.einsum("nij,j->ni", Rk, np.array([r0, 0, 0]))
        cen_all.append(cen.astype(np.float32)); q_all.append(quats(Rk).astype(np.float32))
        placed += m; nfil += 1; lens.append(m * rise)
        if nfil % 10000 == 0:
            log(f"  actin filaments {nfil:,}, subunits {placed:,}")
    return np.vstack(cen_all), np.vstack(q_all), nfil, np.array(lens)


def main():
    t0 = time.time()
    mt_total = float(sys.argv[1]) * 1000 if len(sys.argv) > 1 else 2.8e6
    fa_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 31
    os.makedirs(OUT, exist_ok=True); os.makedirs(f"{D}/qc", exist_ok=True)
    rng = np.random.default_rng(seed)
    log = lambda *a: print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)
    z = np.load(f"{D}/jgeom_cell1.npz"); cell = c3.Cell(z["G"], z["pores"]); centre_nm = z["centre_nm"]
    geo = Geo(cell)
    pm = mt_model(); pa = actin_model()
    log("MT lattice", {k: v for k, v in pm.items() if k != "superposition"}, pm["superposition"])
    log("actin helix", pa)
    I = pd.read_csv(f"{D}/ingredients_v0.tsv", sep="\t").set_index("ing_id")
    inv = pd.read_csv(f"{D}/inventory_v0_complexed.tsv", sep="\t", index_col="gene")
    tub_pool = int(I.copies.get(ING_TUB, 0))
    act_total = int(inv.copies[["ACTB", "ACTG1"]].sum()); act_target = int(min(fa_frac * act_total, I.copies.get(ING_ACT, 0)))
    # cell bounding box (model frame) from the label grid
    L = np.load(f"{D}/cell1_labels_s2.npz"); lab = L["labels"]; o2 = L["origin_s2_xyz"]
    idx = np.argwhere(lab.any(axis=(1, 2)))[[0, -1], 0], np.argwhere(lab.any(axis=(0, 2)))[[0, -1], 0], np.argwhere(lab.any(axis=(0, 1)))[[0, -1], 0]
    bb = np.array([[(o2[a] + idx[a][0]) * VOXS2[a], (o2[a] + idx[a][1]) * VOXS2[a]] for a in range(3)]).T - centre_nm
    glo = bb[0] - 200.0; ghi = bb[1] + 200.0
    cap_s = 9_000_000; cap_e = cap_s * 10
    grid = p3.grid_new(glo, ghi, H, cap_s, cap_e)
    log("grid", grid[0].tolist(), "bbox", bb.round(0).tolist())
    # centrosome + centrioles
    cen0, gc, cinfo = centrosome(cell, geo, centre_nm, rng)
    cc, cq, cax = centrioles(cen0, rng, pm)
    DBG["cax"] = cax
    p3.g_insert(glo, H, *grid, cax.astype(np.float64), np.full(len(cax), R_MT_AX))
    log("centrosome", cen0.round(0).tolist(), cinfo, "centriole dimers", len(cc))
    if os.environ.get("CYTO_STOP_AFTER_CENTROSOME"):
        return
    # MTs
    depth = Depth(centre_nm)
    mts, total, n_dim = grow_mts_v2(geo, grid, glo, cen0, pm, mt_total, tub_pool - len(cc), rng, log, depth=depth)
    mc, mq = [], []
    for pts, fr in mts:
        c_, q_ = lattice(pts, fr, pm); mc.append(c_.astype(np.float32)); mq.append(q_.astype(np.float32))
    mc = np.vstack(mc) if mc else np.zeros((0, 3), np.float32); mq = np.vstack(mq) if mq else np.zeros((0, 4), np.float32)
    lens = np.array([(len(p) - 1) * pm["dimer"] for p, _ in mts])
    tipd = np.array([np.linalg.norm(p[-1] - cen0) for p, _ in mts]) if mts else np.zeros(1)
    log(f"MTs: {len(mts)}, total {total / 1e6:.2f} mm, mean {lens.mean():.0f} nm, dimers {len(mc):,} (+{len(cc):,} centrioles); pool {tub_pool:,}; "
        f"plus-end distance from centrosome median {np.median(tipd):.0f} nm, p90 {np.percentile(tipd, 90):.0f} nm")
    tip_dep = depth(np.array([p[-1] for p, _ in mts])) if mts else np.zeros(1)
    frac_cortex = float(np.mean(tip_dep < 300.0))
    log(f"plus ends within 300 nm of the PM: {frac_cortex:.1%}; length quartiles {np.percentile(lens, [10, 25, 50, 75, 90]).round(0).tolist()}")
    # cortical actin
    ac, aq, nfil, flen = place_actin(geo, grid, glo, ghi, bb, pa, act_target, rng, log)
    log(f"F-actin: {nfil:,} filaments, {len(ac):,} subunits ({len(ac) * pa['rise'] / 1e6:.1f} mm), target {act_target:,}")
    lib_ids = np.array([LIB_MT, LIB_ACT])
    pos = np.vstack([cc.astype(np.float32), mc, ac]); quat = np.vstack([cq.astype(np.float32), mq, aq])
    types = np.r_[np.zeros(len(cc) + len(mc), np.int32), np.ones(len(ac), np.int32)]
    kind = np.r_[np.full(len(cc), 0, np.int8), np.full(len(mc), 1, np.int8), np.full(len(ac), 2, np.int8)]   # 0 centriole 1 MT 2 F-actin
    mt_id = np.r_[np.full(len(cc), -1, np.int32), np.repeat(np.arange(len(mts), dtype=np.int32), [len(p) * NPF for p, _ in mts]), np.full(len(ac), -1, np.int32)]
    np.savez_compressed(f"{OUT}/cytoskeleton_cell1.npz", lib_ids=lib_ids, types=types, pos=pos, quat=quat, kind=kind, mt_id=mt_id)
    info = dict(
        consumed={ING_TUB: int(len(cc) + len(mc)), ING_ACT: int(len(ac))},
        lib_to_ingredient={LIB_MT: ING_TUB, LIB_ACT: ING_ACT},
        microtubules=dict(version="v2 lymphocyte array (grow_mts_v2)", frac_plus_ends_within_300nm_of_PM=round(frac_cortex, 3),
                          length_percentiles_nm={str(q): float(v) for q, v in zip((10, 25, 50, 75, 90), np.percentile(lens, [10, 25, 50, 75, 90]).round(0))},
                          plus_end_dist_centrosome_median_nm=float(np.median(tipd)),
                          n=len(mts), total_length_mm=round(total / 1e6, 3), mean_length_nm=round(float(lens.mean()), 0) if len(lens) else 0,
                          median_length_nm=round(float(np.median(lens)), 0) if len(lens) else 0, dimers=int(len(mc)), target_mm=mt_total / 1e6,
                          tubulin_pool=tub_pool, lattice=pm),
        centrosome=dict(model_nm=cen0.round(1).tolist(), label_frame_nm=(cen0 + centre_nm).round(1).tolist(),
                        golgi_centroid_model_nm=gc.round(1).tolist(), centriole_dimers=int(len(cc)), **cinfo),
        f_actin=dict(filaments=int(nfil), subunits=int(len(ac)), length_mm=round(len(ac) * pa["rise"] / 1e6, 2),
                     mean_filament_nm=round(float(flen.mean()), 0) if len(flen) else 0, target_subunits=act_target,
                     actb_actg1_inventory=act_total, fraction_of_actin=round(len(ac) / act_total, 3), helix=pa),
        params=dict(SHELL=SHELL, L_MAX=L_MAX, CORTEX_D=CORTEX_D, CORTEX_BAND=CORTEX_BAND, RADIAL_BIAS=RADIAL_BIAS, N_REROUTE=N_REROUTE, DIMER=pm["dimer"], LP_MT=LP_MT, CLR_MT=CLR_MT, R_MT_AX=R_MT_AX, ACT_BAND=ACT_BAND, ACT_LEN=ACT_LEN,
                    R_ACT_AX=R_ACT_AX, seed=seed, grid_H=H),
        runtime_s=round(time.time() - t0, 0),
        caveats=["collision on filament-axis spheres only (no MT-actin lattice-level check)",
                 "centrioles: complete 13-pf A/B/C tubes, approximate triplet geometry",
                 "F-actin uses ACTB beads for all subunits (free ACTG1 pool ~1e4 at 100 pg)",
                 "MTs stop at the cortex (PM clearance) or after 20 failed steering attempts"])
    json.dump(info, open(f"{OUT}/cytoskeleton_cell1.json", "w"), indent=1)
    # QC figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for a_, (i, j) in zip(ax, ((0, 1), (0, 2))):
        for pts, _ in mts[::2]:
            a_.plot(pts[:, i], pts[:, j], lw=0.3, c="tab:blue", alpha=0.6)
        sm = ac[rng.choice(len(ac), min(len(ac), 60000), replace=False)] if len(ac) else ac
        a_.scatter(sm[:, i], sm[:, j], s=0.05, c="tab:red", alpha=0.3)
        a_.scatter([cen0[i]], [cen0[j]], c="k", s=40, marker="*")
        a_.set_aspect("equal"); a_.set_xlabel("xyz"[i] + " (nm)"); a_.set_ylabel("xyz"[j] + " (nm)")
    ax[0].set_title(f"MTs ({len(mts)}, every 2nd shown, blue), cortical F-actin sample (red), centrosome (*)")
    fig.tight_layout(); fig.savefig(f"{D}/qc/cytoskeleton_cell1.png", dpi=80)
    log("done", json.dumps({k: info[k] for k in ("consumed",)}))


if __name__ == "__main__":
    main()
