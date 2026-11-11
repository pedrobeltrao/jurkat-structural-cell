"""v7 cytoskeleton pre-pass (PLANNING_NOTES "v7 prep"; DECISION stage 3 -> v7: MT variant b).

Builds data/derived/prepass_v7/cytoskeleton_cell1.{npz,json} from
  - MTs: cytoskeleton stage 3 variant b (data/derived/cytoskel/stage3/mt_stage3_b_cell1.npz; EM scaffold + 40 aster MTs,
    1.456 mm, 2.33 M TUBA1B-TUBB4B dimers), replacing the old procedural kind == 1 rows;
  - centrioles (kind 0): the old pair (prepass/cytoskeleton_cell1.npz, 29,484 dimers) moved rigidly to the MTOC candidate
    of stage 3 (model (-1, 1239, 2755) nm; low confidence): candidate centres = cytosol grid points (60 nm) within 3 um with
    >= 150 nm membrane clearance, nearest first; for each, N_ROT random rotations of a 1,000-dimer subsample; accepted when
    no dimer violates the geometry (label cytosol, >= CLR_C nm from every membrane); the first with no MT clash (dimer
    centre >= 18 nm from an MT axis) wins, else the one with the fewest; MT dimers < 9 nm from a centriole dimer are dropped;
  - cortical F-actin (kind 2): the old filaments, trimmed where a subunit lies within 16.5 nm of an MT axis (12.5 MT radius
    + 3.6 actin radius + 0.4), within 8 nm of a centriole dimer, or violates the v7 geometry (label cytosol, >= 3.6 nm
    membrane clearance); filaments are re-split at removed subunits (consecutive subunits > 6 nm apart = new filament)
    and pieces < 100 nm (36 subunits) are dropped.
Clashes of the new MTs / centrioles with other pre-pass content are resolved in their favour (EM-scaffolded MTs):
  - polysome ribosomes (prepass/polysomes_cell1.npz) with centre < 26 nm from an MT axis or < 20 nm from a centriole
    dimer are dropped -> prepass_v7/polysomes_cell1.{npz,json} (consumed reduced: the copies return to the random pool);
  - mRNA beads < 14 nm from an MT axis or < 7 nm from a centriole dimer are listed in prepass_v7/mrna_drop.npy (jwc index
    skips them; the 1.3 GB bead file is not copied);
  - NPC scaffold beads (7R5K) are kept; MT dimers whose centre is closer than bead radius + 5 nm to one are dropped.
usage: NUMBA_NUM_THREADS=8 python jcytoskel_v7.py [geometry=data/derived/jgeom_cell1_v7.npz]"""
import os, sys, json, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3
from config import LIB

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"; P0 = f"{D}/prepass"; P7 = f"{D}/prepass_v7"; S3 = f"{D}/cytoskel/stage3"
R_MT = 12.5; R_ACT = 3.6
CLR_C = 10.0                 # nm, centriole dimer centre clearance to membrane surfaces
MT_C = 18.0                  # nm, centriole dimer centre to MT axis
ACT_MT = 16.5; ACT_C = 8.0; ACT_CLR = 3.6; ACT_BREAK = 6.0; ACT_MIN = 36
RIB_MT = 26.0; RIB_C = 20.0; MRNA_MT = 14.0; MRNA_C = 7.0
N_ROT = 24; SEARCH_R = 3000.0; SEARCH_H = 40.0; CEN_CLR = 150.0; N_CAND = 3000
SEED = 20261111


class Axes:
    """polyline MT axes; exact point-to-segment distances for candidates near the segment midpoints"""
    def __init__(self, pts, off):
        a, b = [], []
        for i in range(len(off) - 1):
            p = pts[off[i]:off[i + 1]].astype(np.float64)
            if len(p) > 1: a.append(p[:-1]); b.append(p[1:])
        self.A = np.concatenate(a); self.B = np.concatenate(b); self.M = 0.5 * (self.A + self.B)
        self.hl = 0.5 * np.linalg.norm(self.B - self.A, axis=1).max()
        self.tree = cKDTree(self.M)

    def dist(self, P, thr):
        """min distance to any axis, exact below thr (else >= thr, reported as thr)"""
        P = np.asarray(P, np.float64); out = np.full(len(P), thr)
        for s in range(0, len(P), 2_000_000):
            Q = P[s:s + 2_000_000]
            lst = self.tree.query_ball_point(Q, thr + self.hl)
            for i, js in enumerate(lst):
                if not js: continue
                js = np.asarray(js); a = self.A[js]; d = self.B[js] - a
                t = np.clip(np.einsum("ij,ij->i", Q[i] - a, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-9), 0, 1)
                out[s + i] = min(thr, float(np.sqrt(((a + t[:, None] * d - Q[i]) ** 2).sum(1)).min()))
        return out

    def near(self, P, thr):
        """boolean: within thr of an axis (vectorised prefilter on midpoints, exact test on the candidates)"""
        P = np.asarray(P, np.float64); d0, _ = self.tree.query(P, k=1, distance_upper_bound=thr + self.hl)
        cand = np.nonzero(np.isfinite(d0))[0]; res = np.zeros(len(P), bool)
        if len(cand): res[cand] = self.dist(P[cand], thr) < thr
        return res


def _cs(oj):
    import ast
    c = oj["centrosome"]; return ast.literal_eval(c) if isinstance(c, str) else c


def clearance(cell, P):
    S = cell.mems(P)
    return np.min(np.abs(S) - c3.MEM_HALF[None, :], axis=1), cell.labels(P)


def segments(P, brk):
    """filament pieces of consecutive subunits (start indices); a gap > brk starts a new piece"""
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.r_[0, np.nonzero(d > brk)[0] + 1, len(P)]


def main(geom=f"{D}/jgeom_cell1_v7.npz"):
    t0 = time.time(); rng = np.random.default_rng(SEED); log = {}
    os.makedirs(P7, exist_ok=True)
    z = np.load(geom); cell = c3.Cell(z["G"], z["pores"])
    old = np.load(f"{P0}/cytoskeleton_cell1.npz"); oj = json.load(open(f"{P0}/cytoskeleton_cell1.json"))
    lib_ids = list(old["lib_ids"]); iMT, iA = lib_ids.index("MT_TUBA1B-TUBB4B"), lib_ids.index("F_ACTB")
    mt = np.load(f"{S3}/mt_stage3_b_cell1.npz"); mj = json.load(open(f"{S3}/mt_stage3_b_cell1.json"))
    cl = np.load(f"{S3}/mt_stage3_b_centerlines.npz")
    ax = Axes(cl["points_model_nm"], cl["offsets"])
    mtoc = np.array([-1.0, 1239.0, 2755.0])
    mpos = mt["pos"].astype(np.float64); mquat = mt["quat"]; mid = mt["mt_id"]
    print(f"MT b: {len(mpos):,} dimers on {len(cl['offsets']) - 1} centerlines ({time.time() - t0:.0f}s)", flush=True)
    mtree = cKDTree(mpos)

    # ---------------------------------------------------------------- centrioles to the MTOC candidate
    k0 = old["kind"] == 0
    cp, cq = old["pos"][k0].astype(np.float64), old["quat"][k0].astype(np.float64)
    c0 = cp.mean(0); rel = cp - c0; rmax = np.linalg.norm(rel, axis=1).max()
    # free-space clearance: the stored SDFs are clamped at 48-64 nm, so take the Euclidean distance transform of the
    # "free" grid (cytosol and >= 30 nm from every membrane) on a SEARCH_H grid around the candidate
    from scipy.ndimage import distance_transform_edt
    g = np.arange(-SEARCH_R - 600, SEARCH_R + 601, SEARCH_H); ng = len(g)
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) + mtoc
    clr0, lab = clearance(cell, X)
    free = ((lab == c3.CYTO) & (clr0 >= 30.0)).reshape(ng, ng, ng)
    clr = distance_transform_edt(free).ravel() * SEARCH_H
    ok = (np.linalg.norm(X - mtoc, axis=1) <= SEARCH_R) & (clr >= CEN_CLR)
    log["centriole_search"] = dict(grid_pts=int(len(X)), free_frac=round(float(free.mean()), 3), max_clearance_nm=float(clr.max()),
                                   candidates=int(ok.sum()))
    print("centriole search:", log["centriole_search"], flush=True)
    X, clr = X[ok], clr[ok]
    order = np.argsort(np.linalg.norm(X - mtoc, axis=1))[:N_CAND]
    rots = Rotation.random(N_ROT, random_state=SEED)
    sub = rel[rng.choice(len(rel), 1000, replace=False)]
    best = None; tested = 0                                  # (membrane violations, MT clashes, rank) -> lexicographic min
    for rank, i in enumerate(order):
        tested += 1
        for r in range(N_ROT):
            Q = X[i] + rots[r].apply(sub)
            cq_, lq = clearance(cell, Q)
            bad = int(((lq != c3.CYTO) | (cq_ < CLR_C)).sum())
            if bad: continue
            mtc = int(ax.near(Q, MT_C).sum())
            if best is None or mtc < best[2]: best = (i, r, mtc)
            if mtc == 0: break
        if best is not None and best[2] == 0: break
    if best is None: raise SystemExit("no membrane-clear centriole placement found")
    cen = X[best[0]]; R = rots[best[1]]
    ncp = cen + R.apply(rel)
    cq_, lq = clearance(cell, ncp); full_bad = int(((lq != c3.CYTO) | (cq_ < CLR_C)).sum()); full_mt = int(ax.near(ncp, MT_C).sum())
    qold = Rotation.from_quat(np.c_[cq[:, 1:], cq[:, :1]])
    qn = (R * qold).as_quat(); ncq = np.c_[qn[:, 3:], qn[:, :3]].astype(np.float32)
    log["centrioles"] = dict(old_centre_nm=c0.round(1).tolist(), new_centre_nm=cen.round(1).tolist(), mtoc_candidate_nm=mtoc.tolist(),
                             dist_to_mtoc_nm=round(float(np.linalg.norm(cen - mtoc)), 1), centre_clearance_nm=round(float(clr[best[0]]), 1),
                             moved_nm=round(float(np.linalg.norm(cen - c0)), 1), rotation_index=int(best[1]), candidates_tested=tested,
                             dimers=int(len(ncp)), full_check_membrane_violations=full_bad, full_check_mt_clashes=full_mt,
                             extent_radius_nm=round(float(rmax), 1),
                             golgi_centroid_dist_nm=round(float(np.linalg.norm(cen - np.array(_cs(oj)["golgi_centroid_model_nm"]))), 1))
    print("centrioles:", log["centrioles"], f"({time.time() - t0:.0f}s)", flush=True)
    ctree = cKDTree(ncp)
    dmc, _ = ctree.query(mpos, k=1, distance_upper_bound=9.0); clash_c = np.isfinite(dmc)   # dimer centres < 9 nm apart
    mpos, mquat, mid = mpos[~clash_c], mquat[~clash_c], mid[~clash_c]
    log["centrioles"]["mt_dimers_dropped_at_centrioles"] = int(clash_c.sum())

    # ---------------------------------------------------------------- NPC beads vs MT dimers
    npc = np.load(f"{P0}/npc_cell1.npz"); L = np.load(f"{LIB}/NPC.npz"); stc, str_ = L["st_c"].astype(np.float64), L["st_r"].astype(np.float64)
    nq = npc["quat"].astype(np.float64); Rn = Rotation.from_quat(np.c_[nq[:, 1:], nq[:, :1]])
    NB = np.concatenate([npc["pos"][k].astype(np.float64) + Rn[k].apply(stc) for k in range(len(nq))]); NR = np.tile(str_, len(nq))
    rmaxb = float(NR.max())
    near_npc = np.zeros(len(mpos), bool)
    for j, nb_ in enumerate(cKDTree(NB).query_ball_point(mpos, rmaxb + 5.0 + 4.0)):
        if nb_:
            d = np.linalg.norm(NB[nb_] - mpos[j], axis=1)
            near_npc[j] = bool((d < NR[nb_] + 5.0).any())
    log["npc_vs_mt"] = dict(npc_beads=int(len(NB)), mt_dimers_dropped=int(near_npc.sum()))
    keep_mt = ~near_npc
    mpos, mquat, mid = mpos[keep_mt], mquat[keep_mt], mid[keep_mt]
    print("NPC vs MT:", log["npc_vs_mt"], flush=True)

    # ---------------------------------------------------------------- F-actin trim
    k2 = old["kind"] == 2
    ap, aq = old["pos"][k2].astype(np.float64), old["quat"][k2]
    rm_mt = ax.near(ap, ACT_MT)
    dc, _ = ctree.query(ap, k=1, distance_upper_bound=ACT_C); rm_c = np.isfinite(dc)
    clr_a = np.full(len(ap), 1e9); lab_a = np.full(len(ap), c3.CYTO, np.int8)
    for s in range(0, len(ap), 2_000_000):
        clr_a[s:s + 2_000_000], lab_a[s:s + 2_000_000] = clearance(cell, ap[s:s + 2_000_000])
    rm_g = (lab_a != c3.CYTO) | (clr_a < ACT_CLR)
    rm = rm_mt | rm_c | rm_g
    seg0 = segments(ap, ACT_BREAK)
    keep = ~rm
    # re-split: a removed subunit breaks its filament; drop short pieces
    pid = np.zeros(len(ap), np.int64); pid[seg0[1:-1]] = 1
    brk = np.r_[True, rm[:-1]] | (pid == 1)            # new piece starts after a removed subunit or at an old filament start
    piece = np.cumsum(brk) - 1
    piece_k = piece[keep]; cnt = np.bincount(piece_k, minlength=piece.max() + 1)
    keep2 = keep.copy(); keep2[keep] = cnt[piece_k] >= ACT_MIN
    log["f_actin"] = dict(subunits_old=int(len(ap)), filaments_old=int(len(seg0) - 1), removed_mt=int(rm_mt.sum()), removed_centriole=int(rm_c.sum()),
                          removed_geometry=int(rm_g.sum()), removed_short_pieces=int((keep & ~keep2).sum()), subunits_new=int(keep2.sum()),
                          filaments_new=int((np.bincount(piece[keep2]) >= ACT_MIN).sum()))
    ap, aq = ap[keep2], aq[keep2]
    print("F-actin:", log["f_actin"], f"({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- polysome ribosomes and mRNA beads
    pz = np.load(f"{P0}/polysomes_cell1.npz"); pj = json.load(open(f"{P0}/polysomes_cell1.json"))
    rp = pz["pos"].astype(np.float64)
    d_r = ax.near(rp, RIB_MT); dcr, _ = ctree.query(rp, k=1, distance_upper_bound=RIB_C); d_rc = np.isfinite(dcr)
    drop_r = d_r | d_rc
    out = {k: pz[k][~drop_r] for k in pz.files if k != "lib_ids"}; out["lib_ids"] = pz["lib_ids"]
    np.savez_compressed(f"{P7}/polysomes_cell1.npz", **out)
    pj2 = dict(pj); cons = pj["consumed"] if isinstance(pj["consumed"], dict) else json.loads(str(pj["consumed"]).replace("'", '"'))
    cons = {k: int(v) - (int(drop_r.sum()) if k == "RIBO80S" else 0) for k, v in cons.items()}
    pj2["consumed"] = cons; pj2["v7_dropped"] = dict(mt_axis_lt_26nm=int(d_r.sum()), centriole_lt_20nm=int(d_rc.sum()), total=int(drop_r.sum()),
                                                     source="prepass/polysomes_cell1.npz; jcytoskel_v7.py")
    json.dump(pj2, open(f"{P7}/polysomes_cell1.json", "w"), indent=1)
    log["ribosomes"] = pj2["v7_dropped"] | dict(kept=int((~drop_r).sum()))
    print("ribosomes:", log["ribosomes"], f"({time.time() - t0:.0f}s)", flush=True)
    mr = np.load(f"{P0}/mrna_cell1.npz", mmap_mode="r"); pts = mr["pts"]; inrib = mr["inrib"]
    drops = []
    lo_b, hi_b = ax.M.min(0) - 50, ax.M.max(0) + 50
    for s in range(0, len(pts), 10_000_000):
        Q = np.asarray(pts[s:s + 10_000_000], np.float64)
        sel = np.nonzero((inrib[s:s + 10_000_000] == 0) & np.all((Q > lo_b) & (Q < hi_b), axis=1))[0]
        if not len(sel): continue
        hit = ax.near(Q[sel], MRNA_MT)
        dcm, _ = ctree.query(Q[sel], k=1, distance_upper_bound=MRNA_C); hit |= np.isfinite(dcm)
        drops.append(s + sel[hit])
    drops = np.concatenate(drops) if drops else np.zeros(0, np.int64)
    np.save(f"{P7}/mrna_drop.npy", drops.astype(np.int64))
    log["mrna_beads_dropped"] = int(len(drops))
    print("mRNA beads dropped:", len(drops), f"({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- write
    T = np.r_[np.full(len(ncp), iMT), np.full(len(mpos), iMT), np.full(len(ap), iA)].astype(np.int32)
    Ppos = np.concatenate([ncp, mpos, ap]).astype(np.float32)
    Qq = np.concatenate([ncq, mquat.astype(np.float32), aq.astype(np.float32)])
    kind = np.r_[np.zeros(len(ncp)), np.ones(len(mpos)), np.full(len(ap), 2)].astype(np.int8)
    mtid = np.r_[np.full(len(ncp), -1), mid, np.full(len(ap), -1)].astype(np.int32)
    np.savez_compressed(f"{P7}/cytoskeleton_cell1.npz", lib_ids=np.array(lib_ids), types=T, pos=Ppos, quat=Qq, kind=kind, mt_id=mtid)
    consumed = {"TUBA1B-TUBB4B": int(len(ncp) + len(mpos)), "ACTB": int(len(ap))}
    J = dict(version="v7", consumed=consumed, lib_to_ingredient=oj["lib_to_ingredient"], geometry=os.path.basename(geom),
             sources=dict(mt="cytoskel/stage3/mt_stage3_b_cell1.npz (variant b)", centrioles_factin="prepass/cytoskeleton_cell1.npz (v6 pre-pass)"),
             microtubules=dict(variant="b", dimers=int(len(mpos)), centerlines=int(len(cl["offsets"]) - 1), total_um=mj.get("microtubules", {}) if not isinstance(mj.get("microtubules"), str) else mj["microtubules"][:200]),
             params=dict(CLR_C=CLR_C, MT_C=MT_C, ACT_MT=ACT_MT, ACT_C=ACT_C, ACT_CLR=ACT_CLR, ACT_BREAK=ACT_BREAK, ACT_MIN=ACT_MIN,
                         RIB_MT=RIB_MT, RIB_C=RIB_C, MRNA_MT=MRNA_MT, MRNA_C=MRNA_C, N_ROT=N_ROT, SEARCH_R=SEARCH_R, seed=SEED),
             log=log, runtime_s=round(time.time() - t0))
    json.dump(J, open(f"{P7}/cytoskeleton_cell1.json", "w"), indent=1, default=str)
    print("consumed:", consumed, f"written {P7}/cytoskeleton_cell1.npz ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
