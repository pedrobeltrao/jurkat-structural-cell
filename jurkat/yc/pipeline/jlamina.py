"""Nuclear lamina lattice pre-pass (v7; PLANNING_NOTES "v7 prep", rule L2).

v5/v6 packed the lamins (LMNB1 2.08 M, LMNB2 0.37 M, LMNA 0.08 M copies; inventory key "lamina") as random rigid bodies in
the 25 nm lamina shell under the INM: 0.68 placed (RSA jamming of 29 nm rods in a thin shell). Here they are laid out
as an ordered two-layer rod lattice, which is how lamins are organised (filaments of head-to-tail polymers, 3.5 nm
thick, in a ~14 nm meshwork under the INM; Turgay et al. 2017 Nature 543:261; filaments in different layers cross):
  - each library is used in its principal frame (AFDB monomer with low-pLDDT tails dropped: a 28.5-29.8 x ~5 x ~4 nm rod;
    smallest axis along the local INM normal);
  - the INM is divided into patches (Voronoi cells of seeds PATCH nm apart on the HMFF NE midsurface; random in-plane
    orientation per patch = lamina domains); layer 1 rods run along e1 at depth D1 below the INM midsurface, layer 2 along
    e2 = n x e1 at D2 (orthogonal crossing layers); rods head-to-tail every PERIOD nm (filaments), filaments LAT nm apart
    (rows staggered by PERIOD/2); LAT is set so that the lattice holds the lamina copies (not below LAT_MIN = 6.5 nm, the
    non-overlap spacing of the rods), surplus sites are thinned at random;
  - sites are projected onto the depth isosurface of the NE field (3 Newton steps), framed by the local normal; dropped
    when the rod centre or ends are not nucleoplasm (label NUCP; not pore, nucleolus, speckle), within PORE_X nm of a pore
    axis (NPC nuclear ring / basket), closer than the rod half-thickness to the INM, or when the rod capsule overlaps a
    rod of a neighbouring patch (bead-level test at patch borders);
  - DECISION (v7 prep): the lamina has priority over the chromatin in its ~16 nm layer: chromatin pre-pass instances whose
    bounding sphere overlaps a rod capsule are listed in chromatin_drop_lamina.npy and dropped at jwc indexing (the current
    walk chromatin fills the nucleoplasm up to the INM; the genome-mapped chromatin should leave the band free, then this
    list is ~empty). LAMINA_PRIORITY=chromatin drops the rods instead;
  - species are assigned at random in proportion to their lamina copies; unplaced copies stay in the random lamina shell.
Not modelled: the filament structure itself (tetramers, 3.5 nm coiled-coil filaments, tail domains), A/B meshwork
segregation, lamina-chromatin (LAD) contacts. The chromatin pre-pass is an input: re-run when it changes.
usage: NUMBA_NUM_THREADS=8 python jlamina.py [geometry] [chromatin npz] [ingredients tsv] [out dir]"""
import os, sys, json, time
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3
from config import LIB

V = (__import__("os").environ.get("VCELL_ROOT", ".") + ""); D = f"{V}/data/derived"
GENES = ["LMNB1", "LMNB2", "LMNA"]
PERIOD = 30.5; LAT_MIN = 6.5; PATCH = 500.0   # PERIOD / LAT_MIN / D1 / D2 are recomputed from the rod boxes in main()
D1, D2 = 6.0, 13.0              # nm below the INM midsurface (INM half thickness 2.01 + rod half-thickness + 0.5)
PORE_X = 80.0                   # nm from a pore axis
SEED = 20261112
LAT_F = float(os.environ.get("LAMINA_LAT_F", 1.0))      # lateral spacing / rod box width (bent rods nest partly)
PRIORITY = os.environ.get("LAMINA_PRIORITY", "lamina")   # lamina: rods kept, clashing chromatin dropped; chromatin: rods dropped


def frames(names):
    out = {}
    for n in names:
        z = np.load(f"{LIB}/{n}.npz"); c = z["st_c"].astype(np.float64); r = z["st_r"].astype(np.float64)
        w = r ** 3; m = (c * w[:, None]).sum(0) / w.sum(); X = c - m
        _, U = np.linalg.eigh((X * w[:, None]).T @ X); U = U[:, [2, 1, 0]]
        if np.linalg.det(U) < 0: U[:, 2] *= -1
        Y = X @ U
        bc = 0.5 * ((Y + r[:, None]).max(0) + (Y - r[:, None]).min(0))   # bounding-box centre in the principal frame
        Y = Y - bc; m = m + U @ bc                                       # rods are framed on their bounding box
        ext = np.abs(Y).max(0) + r.max()
        out[n] = dict(m=m, U=U, Y=Y, r=r, half_len=float(ext[0]), half_w=float(ext[1]), half_thick=float(ext[2]))
    return out


def project(cell, P, depth, it=3, h=0.5):
    """move points onto the NE-distance isosurface out[M_NE] = depth on the nucleoplasm side; returns points and unit
    normals pointing into the nucleus (direction of increasing distance)"""
    P = P.copy()
    for _ in range(it):
        S = cell.mems(P)[:, c3.M_NE].astype(np.float64)
        g = np.stack([(cell.mems(P + h * e)[:, c3.M_NE] - cell.mems(P - h * e)[:, c3.M_NE]) / (2 * h) for e in np.eye(3)], 1)
        gn = np.linalg.norm(g, axis=1, keepdims=True); g /= np.maximum(gn, 1e-9)
        P -= (S - depth)[:, None] * g
    S = cell.mems(P)[:, c3.M_NE].astype(np.float64)
    good = (np.abs(S - depth) < 1.0) & (gn[:, 0] > 0.5)          # converged, and inside the stored NE band
    return P, g, good


def main(geom=f"{D}/jgeom_cell1_v7.npz", chrom=f"{D}/prepass/chromatin_cell1.npz", ing=f"{D}/ingredients_v0.tsv",   # lamina targets: pre-L3 inventory (v6), not ingredients_v7
         out_dir=f"{D}/prepass_v7"):
    import trimesh
    t0 = time.time(); rng = np.random.default_rng(SEED)
    if not os.path.exists(ing): ing = f"{D}/ingredients_v0.tsv"
    z = np.load(geom); cell = c3.Cell(z["G"], z["pores"]); centre = z["centre_nm"]
    I = pd_read(ing).set_index("ing_id")
    tgt = {g: int(round(I.at[g, "copies"] * json.loads(I.at[g, "comp_weights"]).get("lamina", 0.0))) for g in GENES}
    ntot = sum(tgt.values())
    Fm = frames(GENES)
    global PERIOD, LAT_MIN, D1, D2
    hlm = max(Fm[g]["half_len"] for g in GENES); hwm = max(Fm[g]["half_w"] for g in GENES); htm = max(Fm[g]["half_thick"] for g in GENES)
    PERIOD = 2 * hlm + 0.5; LAT_MIN = LAT_F * 2 * hwm               # box-framed neighbours; overlaps resolved below
    D1 = float(c3.MEM_HALF[c3.M_NE]) + htm + 0.5; D2 = D1 + 2 * htm + 0.5
    m = trimesh.load(f"{D}/cell1_NE_hmff_midsurface.ply", process=False)
    Vt = np.asarray(m.vertices, np.float64) - centre; area = float(m.area)            # NE midsurface ~ INM area
    Vn = np.asarray(m.vertex_normals, np.float64)
    if np.mean(np.einsum("ij,ij->i", Vn, Vt - Vt.mean(0))) < 0: Vn = -Vn           # outward (to the cytoplasm)
    lat = max(LAT_MIN, 2 * area / ntot / PERIOD)                                     # two layers
    print(f"lamina copies {tgt} (total {ntot:,}); NE midsurface {area / 1e6:.1f} um2; lateral spacing {lat:.2f} nm "
          f"(needed {2 * area / ntot / PERIOD:.2f})", flush=True)
    # patch seeds: greedy subsample of mesh vertices PATCH apart
    order = rng.permutation(len(Vt)); seeds = []; snorm = []
    tr_v = None
    for i in order:
        p = Vt[i]
        if seeds and tr_v is not None and tr_v.query(p)[0] < PATCH: continue
        if seeds and np.min(np.linalg.norm(np.array(seeds[-200:]) - p, axis=1)) < PATCH: continue
        seeds.append(p); snorm.append(Vn[i])
        if len(seeds) % 200 == 0: tr_v = cKDTree(np.array(seeds))
    seeds = np.array(seeds); stree = cKDTree(seeds)
    print(f"{len(seeds)} patches ({time.time() - t0:.0f}s)", flush=True)
    sites, dirs, layer, patch = [], [], [], []
    half = 1.1 * PATCH
    for k, p in enumerate(seeds):
        n = -snorm[k]                                                              # into the nucleus
        a = rng.normal(size=3); e1 = a - (a @ n) * n; e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
        for L, (d, u) in enumerate(((D1, e1), (D2, e2))):
            v = np.cross(n, u)
            us = np.arange(-half, half, PERIOD); vs = np.arange(-half, half, lat)
            U_, V_ = np.meshgrid(us, vs, indexing="ij"); U_ = U_ + (np.arange(len(vs))[None, :] % 2) * PERIOD / 2
            Q = p + (0.5 * c3.PERI_W + d) * n + U_.ravel()[:, None] * u + V_.ravel()[:, None] * v
            sites.append(Q); dirs.append(np.repeat(u[None], len(Q), 0)); layer.append(np.full(len(Q), L)); patch.append(np.full(len(Q), k))
    S = np.concatenate(sites); Dr = np.concatenate(dirs); Ly = np.concatenate(layer); Pt = np.concatenate(patch)
    print(f"{len(S):,} raw sites ({time.time() - t0:.0f}s)", flush=True)
    # project to depth, Voronoi assignment
    P = np.empty_like(S); N = np.empty_like(S); G_ok = np.zeros(len(S), bool)
    for L, d in ((0, D1), (1, D2)):
        s = Ly == L
        for a in range(0, int(s.sum()), 2_000_000):
            idx = np.nonzero(s)[0][a:a + 2_000_000]
            P[idx], N[idx], G_ok[idx] = project(cell, S[idx], d)
    dd, kk = stree.query(P, k=2)
    keep = (kk[:, 0] == Pt) & G_ok
    P, N, Dr, Ly, Pt, dd = P[keep], N[keep], Dr[keep], Ly[keep], Pt[keep], dd[keep]
    Dr = Dr - np.einsum("ij,ij->i", Dr, N)[:, None] * N; Dr /= np.linalg.norm(Dr, axis=1, keepdims=True)
    border = (dd[:, 1] - dd[:, 0]) < 2 * 35.0
    print(f"{len(P):,} sites after Voronoi assignment ({int(border.sum()):,} at patch borders) ({time.time() - t0:.0f}s)", flush=True)
    # species (random by copies), thinning to the target
    sp = rng.choice(len(GENES), size=len(P), p=np.array([tgt[g] for g in GENES]) / ntot)
    hl = np.array([Fm[g]["half_len"] for g in GENES])[sp]; ht = np.array([Fm[g]["half_thick"] for g in GENES])[sp]
    # geometry checks: centre and both rod ends nucleoplasm, depth, pores
    ok = np.ones(len(P), bool); reason = {}
    for nm_, Q in (("centre", P), ("end+", P + (hl - 1.5)[:, None] * Dr), ("end-", P - (hl - 1.5)[:, None] * Dr)):
        lab = cell.labels(Q); Sq = cell.mems(Q)
        bad = (lab != c3.NUCP) | (Sq[:, c3.M_NE] - c3.MEM_HALF[c3.M_NE] < ht - 0.5)
        reason[nm_] = int((bad & ok).sum()); ok &= ~bad
    pr = z["pores"]; ptree = cKDTree(pr[:, :3])
    near = ptree.query_ball_point(P, PORE_X + 60.0)
    bad = np.zeros(len(P), bool)
    for i, js in enumerate(near):
        if not js or not ok[i]: continue
        v = P[i] - pr[js, :3]; zz = np.einsum("ij,ij->i", v, pr[js, 3:]); rho = np.sqrt(np.maximum((v ** 2).sum(1) - zz ** 2, 0))
        bad[i] = bool(((rho < PORE_X) & (np.abs(zz) < 150)).any())
    reason["pore"] = int(bad.sum()); ok &= ~bad
    # chromatin: instances within 40 nm of the INM; capsule (rod axis +-(hl - r), radius ht) vs bounding sphere
    ch = np.load(chrom); cid = list(ch["lib_ids"]); cbr = []
    for l in cid:
        zl = np.load(f"{LIB}/{l}.npz"); cbr.append(float((np.linalg.norm(zl["st_c"], axis=1) + zl["st_r"]).max()))
    cbr = np.array(cbr)
    cp = ch["pos"].astype(np.float64); sel = np.zeros(len(cp), bool)
    lo, hi = P.min(0) - 50, P.max(0) + 50
    inb = np.all((cp > lo) & (cp < hi), axis=1)
    idx = np.nonzero(inb)[0]
    for a in range(0, len(idx), 5_000_000):
        j = idx[a:a + 5_000_000]; sel[j] = cell.mems(cp[j])[:, c3.M_NE] < 45.0
    cP = cp[sel]; cR = cbr[ch["types"][sel]]
    print(f"chromatin instances within 45 nm of the INM: {int(sel.sum()):,} of {len(cp):,} ({time.time() - t0:.0f}s)", flush=True)
    cidx = np.nonzero(sel)[0]
    ctree = cKDTree(cP); bad = np.zeros(len(P), bool); hits = [[] for _ in range(len(P))]
    rq = float(hl.max() + cR.max() + ht.max())
    for i, js in enumerate(ctree.query_ball_point(P, rq)):
        if not js or not ok[i]: continue
        v = cP[js] - P[i]; t = np.clip(v @ Dr[i], -(hl[i] - 1.2), hl[i] - 1.2)
        d = np.linalg.norm(v - t[:, None] * Dr[i], axis=1)
        hb = d < cR[js] + ht[i] - 0.5
        if hb.any(): bad[i] = True; hits[i] = list(cidx[np.asarray(js)[hb]])
    if PRIORITY == "chromatin":
        reason["chromatin"] = int(bad.sum()); ok &= ~bad; chrom_drop = np.zeros(0, np.int64)
    else:                                   # lamina first: clashing chromatin instances are dropped at jwc indexing
        reason["chromatin_clash_sites_kept"] = int(bad.sum()); chrom_drop = None
    hits = [hits[i] for i in np.nonzero(ok)[0]]
    P, N, Dr, Ly, Pt, sp, border = P[ok], N[ok], Dr[ok], Ly[ok], Pt[ok], sp[ok], border[ok]
    # rotation: library principal frame (x = rod, z = smallest) -> world (Dr, N x Dr, N)
    W1 = Dr; W3 = N; W2 = np.cross(W3, W1)
    Rw = np.stack([W1, W2, W3], 2)                          # columns = world images of the principal axes
    # overlap resolution, bead level, all rods (curved / folded NE and patch borders): patches in order, each rod accepted
    # when it overlaps neither an accepted rod of an earlier patch nor an earlier-accepted rod of its own patch
    Yb = {s_: (Fm[g]["Y"], Fm[g]["r"]) for s_, g in enumerate(GENES)}
    rmx = max(float(Fm[g]["r"].max()) for g in GENES)
    acc = np.zeros(len(P), bool); accB = {}; order_p = np.argsort(Pt, kind="stable"); bounds = np.searchsorted(Pt[order_p], np.arange(len(seeds) + 1))
    for kp in range(len(seeds)):
        ii = order_p[bounds[kp]:bounds[kp + 1]]
        if not len(ii): continue
        rng.shuffle(ii)
        B = np.concatenate([P[i] + Yb[sp[i]][0] @ Rw[i].T for i in ii]); Rb = np.concatenate([Yb[sp[i]][1] for i in ii])
        Ob = np.concatenate([np.full(len(Yb[sp[i]][1]), j) for j, i in enumerate(ii)])
        bad = np.zeros(len(ii), bool)
        nb_ = [q for q in stree.query_ball_point(seeds[kp], 3.0 * PATCH) if q in accB]
        if nb_:
            AB = np.concatenate([accB[q][0] for q in nb_]); AR = np.concatenate([accB[q][1] for q in nb_])
            tA = cKDTree(AB); d0, _ = tA.query(B, k=1)
            for j in np.nonzero(d0 < (Rb + rmx) * 0.95)[0]:
                if bad[Ob[j]]: continue
                js = tA.query_ball_point(B[j], Rb[j] + rmx)
                if js and (np.linalg.norm(AB[js] - B[j], axis=1) < (AR[js] + Rb[j]) * 0.95).any(): bad[Ob[j]] = True
        pr = cKDTree(B).query_pairs(2 * rmx, output_type="ndarray")
        if len(pr):
            x_, y_ = pr[:, 0], pr[:, 1]
            hit = (Ob[x_] != Ob[y_]) & (np.linalg.norm(B[x_] - B[y_], axis=1) < (Rb[x_] + Rb[y_]) * 0.95)
            conf = {}
            for u_, v_ in zip(Ob[x_][hit], Ob[y_][hit]):
                conf.setdefault(max(u_, v_), set()).add(min(u_, v_))
            ok_ = ~bad
            for j in range(len(ii)):                          # greedy in the shuffled order
                if ok_[j] and any(ok_[q] for q in conf.get(j, ())): ok_[j] = False
        else:
            ok_ = ~bad
        acc[ii[ok_]] = True
        keepb = ok_[Ob]; accB[kp] = (B[keepb], Rb[keepb])
    reason["rod_overlap"] = int((~acc).sum())
    hits = [hits[i] for i in np.nonzero(acc)[0]]
    P, N, Dr, Ly, Pt, sp, Rw = P[acc], N[acc], Dr[acc], Ly[acc], Pt[acc], sp[acc], Rw[acc]
    # thin to the target per species
    keep = np.zeros(len(P), bool)
    for s_, g in enumerate(GENES):
        ii = np.nonzero(sp == s_)[0]; rng.shuffle(ii); keep[ii[:tgt[g]]] = True
    hits = [hits[i] for i in np.nonzero(keep)[0]]
    P, N, Dr, Ly, Pt, sp, Rw = P[keep], N[keep], Dr[keep], Ly[keep], Pt[keep], sp[keep], Rw[keep]
    Ulib = np.stack([Fm[g]["U"] for g in GENES])[sp]; mlib = np.stack([Fm[g]["m"] for g in GENES])[sp]
    R = Rw @ np.transpose(Ulib, (0, 2, 1))                    # world <- library
    pos = P - np.einsum("nij,nj->ni", R, mlib)
    drop = np.zeros(len(P), bool)
    k = ~drop
    if chrom_drop is None:
        chrom_drop = np.unique(np.array([x for i in np.nonzero(k)[0] for x in hits[i]], np.int64))
    pos, R, sp, Ly = pos[k], R[k], sp[k], Ly[k]
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat(); quat = np.c_[q[:, 3:], q[:, :3]].astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f"{out_dir}/lamina_cell1.npz", lib_ids=np.array(GENES), types=sp.astype(np.int32), pos=pos.astype(np.float32),
                        quat=quat, layer=Ly.astype(np.int8))
    placed = {g: int((sp == s_).sum()) for s_, g in enumerate(GENES)}
    np.save(f"{out_dir}/chromatin_drop_lamina.npy", chrom_drop)
    json.dump(dict(target=os.path.relpath(chrom, V), n=int(len(ch["types"])), drop=os.path.relpath(f"{out_dir}/chromatin_drop_lamina.npy", V),
                   why="overlap with the lamina lattice (jlamina.py, lamina priority)"), open(f"{out_dir}/chromatin_drop_lamina.json", "w"), indent=1)
    cl = list(ch["lib_ids"]); ct = ch["types"][chrom_drop] if len(chrom_drop) else np.zeros(0, int)
    reason["chromatin_instances_dropped"] = int(len(chrom_drop))
    reason["nucleosomes_dropped"] = int((ct == cl.index("NUCLEOSOME")).sum()) if "NUCLEOSOME" in cl else 0
    J = dict(version="v7", consumed=placed, consumed_key="lamina", target=tgt, lib_to_ingredient={g: g for g in GENES}, geometry=os.path.basename(geom),
             chromatin=os.path.relpath(chrom, D), chromatin_n=int(len(ch["types"])), priority=PRIORITY,
             chromatin_drop="chromatin_drop_lamina.npy (indices into the chromatin file above; jwc index drops them only when the"
                            " configured chromatin file and instance count match)", ingredients=os.path.relpath(ing, D), lateral_spacing_nm=round(lat, 2),
             params=dict(LAT_F=LAT_F, PERIOD=PERIOD, LAT_MIN=LAT_MIN, PATCH=PATCH, D1=D1, D2=D2, PORE_X=PORE_X, seed=SEED),
             dropped=reason, patches=int(len(seeds)), placed_frac=round(sum(placed.values()) / max(ntot, 1), 4),
             rods={g: dict(half_len=round(Fm[g]["half_len"], 2), half_thick=round(Fm[g]["half_thick"], 2)) for g in GENES},
             density_per_um2=round(sum(placed.values()) / (area / 1e6), 1), runtime_s=round(time.time() - t0))
    json.dump(J, open(f"{out_dir}/lamina_cell1.json", "w"), indent=1)
    print(json.dumps(J, indent=1), flush=True)


def pd_read(f):
    import pandas as pd
    return pd.read_csv(f, sep="\t")


if __name__ == "__main__":
    main(*sys.argv[1:5])
