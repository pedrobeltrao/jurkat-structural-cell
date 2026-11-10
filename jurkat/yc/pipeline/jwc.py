"""Whole-cell packing driver for the Jurkat cell (stage 3), adapted from yeast wc3d.py.

Steps (python jwc.py <step> [args]):
  tiles    tile list (TILE nm cubes intersecting the cell) + per-tile region volumes (8 nm label voxels) and membrane
           areas (8 nm marching cubes per membrane SDF)                         -> wc/tiles.parquet
  counts   expected copies per (tile, library, region): cell totals (ingredients minus pre-pass consumption) mapped to
           regions as in jpack_test (signal-peptide lumen rule, OM/IM, cristae and Golgi-stack area factors) and split
           over tiles by a multinomial proportional to the tile measure       -> wc/counts.npz, wc/libs.json
  index    pre-pass instances (data/derived/prepass/*_cell1.npz) and mRNA beads bucketed by tile for obstacle lookup
                                                                              -> wc/prepass_{types,pos,quat}.npy + offsets
  pack     checkerboard phases (2x2x2 colours); each tile is packed with pack3d.pack_tile; obstacles = pre-pass instances
           within tile + halo and the instances of already packed neighbour tiles -> wc/tiles/<i>.npz
  summary  totals per region and placement fractions                         -> wc/summary.json
Tiles of 500 nm (yeast: 200 nm) keep the per-tile collision grid < ~1 GB per worker (job memory cap 128 GB).
Model frame = cell3d frame (nm, origin at the cell centre).
v5: open face (cell3d G_FACE): membrane triangles below the face are dropped (measures and packing), so nothing is placed on
it; cell copy numbers are scaled to the modelled part of the cell (summary open_face: f_pm for the PM and PM shells, f_vol
for everything else; applied to the copies left after pre-pass consumption); cortical F-actin pre-pass filaments that run
along the face (within FACE_ACTIN nm of it and > 150 nm, the cortex band depth, below the side-wall PM;
the cell SDF is clamped at 160 nm inside) are dropped at indexing.
Shell regions (jpack_test.SHELL_REG) are split over tiles by their membrane's area.
v6: nuclear speckles (cell3d SPK, jspeckles.py) are a volume region; chromatin pre-pass instances (nucleosomes, H1) whose
centre lies inside a speckle are dropped at indexing (interchromatin space). Shells of a membrane with an area factor
(Golgi: GOLGI_STACK_F, cisternae not built) are packed at the modelled share, as its TM proteins (JWC_GOLGI_SHELL_FULL=1
restores v5). The NPC-proximal shell sh_NPC is split over tiles by NE area."""
import os, sys, json, time
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")   # fork-safe (OpenMP aborts after fork)
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3
from config import LIB
from skimage.measure import marching_cubes

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"
W = f"{D}/wc"
TILE = 500.0
VOX = 4.0; MESH_H = 8.0; MEAS_H = 8.0
HALO = 40.0
FACE_ACTIN = 160.0
NPROC = int(os.environ.get("JWC_NPROC", 24))
GOLGI_SHELL_FULL = os.environ.get("JWC_GOLGI_SHELL_FULL", "0") == "1"   # 1 = v5 behaviour (no area factor on shells)
import jpack_test as jp                                  # region definitions, mapping tables, area factors
VOL_REG, MEM_REG, RNAMES, NV = jp.VOL_REG, jp.MEM_REG, jp.RNAMES, jp.NV
NSH0 = jp.NSH0
_S = {}


def load_cell():
    z = np.load(f"{D}/jgeom_cell1.npz")
    return c3.Cell(z["G"], z["pores"]), json.loads(str(z["summary"]))


def face_keep(tri, G):
    """triangles (n, 9) whose centroid lies in the modelled domain (above the open face)"""
    if G[c3.G_FACE] <= 0 or len(tri) == 0:
        return tri
    return tri[(tri[:, 2] + tri[:, 5] + tri[:, 8]) / 3.0 >= G[c3.G_ZFACE]]


# ------------------------------------------------------------------ tiles + measures
def _measure(idx):
    cell = _S["cell"]; out = []
    for i in idx:
        lo = _S["T"][i] * TILE + _S["lo0"]
        n = int(TILE / MEAS_H); g = (np.arange(n) + 0.5) * MEAS_H
        X, Y, Z = np.meshgrid(lo[0] + g, lo[1] + g, lo[2] + g, indexing="ij")
        lab = cell.labels(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1))
        vol = [float(np.isin(lab, labs).sum() * MEAS_H ** 3) for _, labs in VOL_REG]
        m = n + 1; gm = np.arange(m) * MEAS_H
        X, Y, Z = np.meshgrid(lo[0] + gm, lo[1] + gm, lo[2] + gm, indexing="ij")
        S = cell.mems(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)).reshape(m, m, m, c3.NMEM)
        ar = []
        for _, mi in MEM_REG:
            s = S[..., mi]
            if s.min() < 0 < s.max():
                v, f, _, _ = marching_cubes(s, 0.0, spacing=(MEAS_H,) * 3)
                tt = face_keep((v + lo)[f].reshape(-1, 9), cell.G)
                ar.append(float(0.5 * np.linalg.norm(np.cross(tt[:, 3:6] - tt[:, 0:3], tt[:, 6:9] - tt[:, 0:3]), axis=1).sum()))
            else:
                ar.append(0.0)
        out.append([i] + vol + ar)
    return out


def step_tiles():
    os.environ["NUMBA_NUM_THREADS"] = "1"
    cell, summ = load_cell(); R = cell.G[0]
    lo0 = -np.full(3, np.ceil(R / TILE) * TILE); nt = int(2 * np.ceil(R / TILE))
    I, J, K = np.meshgrid(*[np.arange(nt)] * 3, indexing="ij"); T = np.stack([I.ravel(), J.ravel(), K.ravel()], 1)
    C = T * TILE + lo0 + TILE / 2
    dep = cell.mems(C.astype(np.float64))[:, c3.M_OUTER]
    keep = dep > -TILE * 0.87                               # tile centre within a half diagonal of the cell
    T = T[keep]
    print(f"candidate tiles {nt ** 3:,} -> {len(T):,} near the cell", flush=True)
    _S.update(cell=cell, T=T, lo0=lo0)
    import multiprocessing as mp
    chunks = np.array_split(np.arange(len(T)), NPROC * 8)
    t0 = time.time()
    with mp.get_context("fork").Pool(NPROC) as pool:
        res = [r for part in pool.imap_unordered(_measure, chunks) for r in part]
    cols = ["tile"] + [r[0] for r in VOL_REG] + ["mem_" + m[0] for m in MEM_REG]
    M = pd.DataFrame(res, columns=cols).sort_values("tile").reset_index(drop=True)
    M[["ti", "tj", "tk"]] = T[M.tile.values]
    M = M[(M[[r[0] for r in VOL_REG]].sum(axis=1) > 0) | (M[["mem_" + m[0] for m in MEM_REG]].sum(axis=1) > 0)].reset_index(drop=True)
    M["tile"] = np.arange(len(M))
    os.makedirs(W, exist_ok=True)
    M.to_parquet(f"{W}/tiles.parquet"); json.dump(dict(lo0=lo0.tolist(), TILE=TILE), open(f"{W}/tiles_meta.json", "w"))
    print(f"{len(M):,} tiles with cell content in {time.time() - t0:.0f}s; volumes um3:",
          (M[[r[0] for r in VOL_REG]].sum() / 1e9).round(1).to_dict(), "| areas um2:", (M[["mem_" + m[0] for m in MEM_REG]].sum() / 1e6).round(1).to_dict())


# ------------------------------------------------------------------ expected counts
def prepass_files():
    P = f"{D}/prepass"
    return sorted(f for f in os.listdir(P) if f.endswith("_cell1.npz") and not f.startswith("mrna"))


def step_counts(seed=3):
    M = pd.read_parquet(f"{W}/tiles.parquet"); cell, summ = load_cell()
    I = pd.read_csv(f"{D}/ingredients_v0.tsv", sep="\t")
    consumed = {}
    for f in prepass_files():
        js = f"{D}/prepass/{f[:-4]}.json"
        for k, v in json.load(open(js)).get("consumed", {}).items(): consumed[k] = consumed.get(k, 0) + v
    I["copies"] = (I.copies - I.ing_id.map(consumed).fillna(0)).clip(lower=0).round().astype(np.int64)
    have = {f[:-4] for f in os.listdir(LIB)}
    if os.environ.get("JWC_LIBS_FROM"):                  # keep an earlier library order (tiles store indices into libs.json)
        order = json.load(open(os.environ["JWC_LIBS_FROM"]))["libs"]
        I = I.set_index("ing_id").loc[order].reset_index()
    else:
        I = I[I.ing_id.isin(have) & (I.copies > 0)].reset_index(drop=True)
    memb = np.array([("membrane" in np.load(f"{LIB}/{i}.npz").files) for i in I.ing_id])
    # cell measures from the tiles (consistent with the per-tile distribution); membrane totals include area factors
    cvol = {r: float(M[r].sum()) for r, _ in VOL_REG}
    carea = {m: float(M["mem_" + m].sum()) for m, _ in MEM_REG}
    # cristae / Golgi stacks: the inventory covers the full membrane (IM = CRISTAE_F x OM area, Golgi = GOLGI_STACK_F x blob);
    # only the share present in the geometry is packed. With procedural cristae (jcristae.py) the measured IM area replaces
    # the factor (share = measured IM / (CRISTAE_F x OM), capped at 1).
    fac = {"MITO_IM": jp.CRISTAE_F, "GOLGI": jp.GOLGI_STACK_F}
    if c3.n_fields(cell.G) and cell.G[c3.G_CRISTAE] > 0:
        fac["MITO_IM"] = max(1.0, jp.CRISTAE_F * carea["MITO_OM"] / max(carea["MITO_IM"], 1.0))
    print("membrane area factors (inventory / modelled):", {k: round(v, 2) for k, v in fac.items()}, flush=True)
    of = summ.get("open_face") if cell.G[c3.G_FACE] > 0 else None
    f_vol, f_pm = (of["f_vol"], of["f_pm"]) if of else (1.0, 1.0)
    pm_regs = {"PM"} | {s_[0] for s_ in jp.SHELL_REG if s_[1] == c3.M_PM}
    print(f"open-face scaling of cell copy numbers: f_vol {f_vol}, f_pm {f_pm} (PM regions {sorted(pm_regs)})", flush=True)
    rng = np.random.default_rng(seed); rows = []
    tot = {}
    for li, r in I.iterrows():
        for comp, wt in json.loads(r.comp_weights).items():
            if memb[li] and r.transmembrane:
                reg = jp.TMR.get(comp, "ER")
                if reg == "MITO_IM" and r.mito_om_frac >= 0.5: reg = "MITO_OM"
                share = 1.0 / fac.get(reg, 1.0)              # cristae / Golgi stacks not built: only the modelled share
            else:
                reg = jp.SOL.get(comp, "cytosol")
                if reg in ("ER_lumen", "golgi_lumen", "endo_lumen") and r.signal_frac < 0.5: reg = "cytosol"
                share = 1.0 / fac.get(jp.SHELL_MEMNAME.get(reg), 1.0) if not GOLGI_SHELL_FULL else 1.0
                # v6: shells of a membrane with an area factor (Golgi cisternae not built) are packed at the modelled share,
                # as the TM proteins of that membrane (v5 packed them at full copies on the 1/4.5 blob: 35% placed)
            share *= f_pm if reg in pm_regs else f_vol
            tot[(li, RNAMES.index(reg))] = tot.get((li, RNAMES.index(reg)), 0.0) + r.copies * wt * share
    col = {ri: (RNAMES[ri] if ri < NV else "mem_" + RNAMES[ri] if ri < NSH0 else "mem_" + jp.SHELL_MEMNAME[RNAMES[ri]])
           for ri in range(len(RNAMES))}
    pcache = {}
    for (li, ri), x in tot.items():
        n = int(x) + (rng.random() < x - int(x))
        if n <= 0: continue
        if ri not in pcache:
            w = M[col[ri]].values.astype(np.float64); s = w.sum()
            pcache[ri] = (np.nonzero(w)[0], w[w > 0] / s) if s > 0 else (None, None)
        nz, p = pcache[ri]
        if nz is None: continue
        k = rng.multinomial(n, p); sel = k > 0
        rows.append(np.stack([nz[sel], np.full(sel.sum(), li), np.full(sel.sum(), ri), k[sel]], 1))
    C = np.concatenate(rows).astype(np.int64)
    C = C[np.lexsort((C[:, 1], C[:, 0]))]
    np.savez_compressed(f"{W}/counts.npz", tile=C[:, 0].astype(np.int32), lib=C[:, 1].astype(np.int32), region=C[:, 2].astype(np.int16), n=C[:, 3].astype(np.int32))
    json.dump(dict(libs=list(I.ing_id), mass_da=I.mass_da.tolist(), consumed=consumed, f_vol=f_vol, f_pm=f_pm), open(f"{W}/libs.json", "w"))
    print(f"counts: {len(C):,} (tile, lib, region) entries, {C[:, 3].sum():,} instances; per region:",
          {RNAMES[ri]: int(C[C[:, 2] == ri, 3].sum()) for ri in range(len(RNAMES))})


# ------------------------------------------------------------------ pre-pass index
def step_index():
    meta = json.load(open(f"{W}/tiles_meta.json")); lo0 = np.array(meta["lo0"])
    M = pd.read_parquet(f"{W}/tiles.parquet")
    key = {tuple(t): i for i, t in enumerate(M[["ti", "tj", "tk"]].values)}
    libs, T_, P_, Q_ = [], [], [], []
    for f in prepass_files():
        z = np.load(f"{D}/prepass/{f}")
        ids = list(z["lib_ids"]); base = len(libs); libs += ids
        T_.append(z["types"].astype(np.int32) + base); P_.append(z["pos"].astype(np.float32)); Q_.append(z["quat"].astype(np.float32))
        print("  ", f, len(z["types"]), flush=True)
    T_ = np.concatenate(T_); P_ = np.concatenate(P_); Q_ = np.concatenate(Q_)
    cell, _ = load_cell()
    if cell.G[c3.G_FACE] > 0:                               # cortical F-actin laid along the (former) face membrane
        fa = np.array([l == "F_ACTB" for l in libs])[T_]
        near = fa & (P_[:, 2] < cell.G[c3.G_ZFACE] + FACE_ACTIN)
        if near.any():
            dep = cell.mems(P_[near].astype(np.float64))[:, c3.M_PM]
            drop = np.zeros(len(T_), bool); drop[np.nonzero(near)[0][dep > 150.0]] = True   # deeper than the 8-150 nm cortex band
            print(f"open face: dropped {int(drop.sum()):,} cortical F-actin subunits along the face", flush=True)
            T_, P_, Q_ = T_[~drop], P_[~drop], Q_[~drop]
    if cell.G[c3.G_SPECKLE] > 0:                            # v6: speckles are interchromatin space
        isch = np.array([l == "NUCLEOSOME" or l.startswith("H1-") for l in libs])[T_]
        o_ = int(cell.G[c3.G_SPECKLE]); n_ = int(cell.G[o_]); S_ = cell.G[o_ + 1:o_ + 1 + 4 * n_].reshape(n_, 4)
        from scipy.spatial import cKDTree
        d_, k_ = cKDTree(S_[:, :3]).query(P_[isch].astype(np.float64), k=1)
        drop = np.zeros(len(T_), bool); drop[np.nonzero(isch)[0][d_ < S_[k_, 3]]] = True
        print(f"speckles: dropped {int(drop.sum()):,} chromatin pre-pass instances (nucleosomes, H1) inside {n_} speckles", flush=True)
        T_, P_, Q_ = T_[~drop], P_[~drop], Q_[~drop]
    tij = np.floor((P_ - lo0) / TILE).astype(np.int64)
    tid = np.array([key.get(tuple(t), -1) for t in tij]) if len(tij) < 5_000_000 else _tile_ids(tij, M)
    keep = tid >= 0
    o = np.argsort(tid[keep], kind="stable")
    for name, arr in (("types", T_[keep][o]), ("pos", P_[keep][o]), ("quat", Q_[keep][o])):
        np.save(f"{W}/prepass_{name}.npy", arr)
    off = np.searchsorted(tid[keep][o], np.arange(len(M) + 1))
    np.save(f"{W}/prepass_off.npy", off); json.dump(libs, open(f"{W}/prepass_libs.json", "w"))
    print(f"pre-pass instances {len(T_):,}, indexed {int(keep.sum()):,} into {len(M):,} tiles", flush=True)
    mf = f"{D}/prepass/mrna_cell1.npz"                    # mRNA beads (obstacles; beads inside ribosome channels skipped)
    if os.path.exists(mf):
        z = np.load(mf); pts = z["pts"]; keepb = z["inrib"] == 0 if "inrib" in z.files else np.ones(len(pts), bool)
        pts = pts[keepb]; tid = _tile_ids(np.floor((pts - lo0) / TILE).astype(np.int64), M)
        k = tid >= 0; o = np.argsort(tid[k], kind="stable")
        np.save(f"{W}/mrna_pts.npy", pts[k][o].astype(np.float32))
        np.save(f"{W}/mrna_off.npy", np.searchsorted(tid[k][o], np.arange(len(M) + 1)))
        json.dump(dict(bead_radius=float(z["bead_radius"])), open(f"{W}/mrna_meta.json", "w"))
        print(f"mRNA beads indexed: {int(k.sum()):,}")


def _tile_ids(tij, M):
    dims = M[["ti", "tj", "tk"]].values.max(0) + 2
    lut = -np.ones(dims, np.int64); lut[M.ti, M.tj, M.tk] = np.arange(len(M))
    ok = np.all((tij >= 0) & (tij < dims), axis=1)
    out = -np.ones(len(tij), np.int64); out[ok] = lut[tij[ok, 0], tij[ok, 1], tij[ok, 2]]
    return out


# ------------------------------------------------------------------ packing
def _libpack(ids):
    so = [0]; C, R, B = [], [], []
    for i in ids:
        z = np.load(f"{LIB}/{i}.npz"); c = z["st_c"].astype(np.float64); r = z["st_r"].astype(np.float64)
        C.append(c); R.append(r); so.append(so[-1] + len(r)); B.append(float((np.linalg.norm(c, axis=1) + r).max()))
    return np.array(so, np.int64), np.concatenate(C), np.concatenate(R), np.array(B)


def _neigh(t):
    out = []
    for d in np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 3):
        if d.any():
            j = _S["key"].get(tuple(t + d))
            if j is not None: out.append(j)
    return out


def _pack_tile(ti):
    import pack3d as p3
    cell = _S["cell"]; t = _S["Tij"][ti]; lo = t * TILE + _S["lo0"]
    glo, ghi = lo - HALO, lo + TILE + HALO
    out_f = f"{W}/tiles/{ti}.npz"
    if os.path.exists(out_f): return ti, -1, -1, False, 0.0
    t0 = time.time()
    # seeds per volume region (4 nm voxels)
    n = int(TILE / VOX); g = (np.arange(n) + 0.5) * VOX
    X, Y, Z = np.meshgrid(lo[0] + g, lo[1] + g, lo[2] + g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1); lab = cell.labels(P)
    vox_off, VP = [0], []
    for name, labs in VOL_REG:
        m = np.isin(lab, labs); VP.append(P[m]); vox_off.append(vox_off[-1] + int(m.sum()))
    vox_off += [vox_off[-1]] * len(MEM_REG); VOXP = np.concatenate(VP); del P, X, Y, Z, lab
    # membrane triangles
    m_ = int(TILE / MESH_H) + 1; gm = np.arange(m_) * MESH_H
    X, Y, Z = np.meshgrid(lo[0] + gm, lo[1] + gm, lo[2] + gm, indexing="ij")
    S = cell.mems(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)).reshape(m_, m_, m_, c3.NMEM)
    TRI, tri_off = [], [0]
    for name, mi in MEM_REG:
        s = S[..., mi]
        if s.min() < 0 < s.max():
            v, fc, _, _ = marching_cubes(s, 0.0, spacing=(MESH_H,) * 3); tt = face_keep((v + lo)[fc].reshape(-1, 9), cell.G)
        else:
            tt = np.zeros((0, 9))
        TRI.append(tt); tri_off.append(tri_off[-1] + len(tt))
    TRI = np.concatenate(TRI); del S
    ar = 0.5 * np.linalg.norm(np.cross(TRI[:, 3:6] - TRI[:, 0:3], TRI[:, 6:9] - TRI[:, 0:3]), axis=1)
    CUMA = np.cumsum(ar) if len(ar) else np.zeros(1)
    # jobs: membranes first, then volume; largest bounding radius first
    a, b = _S["coff"][ti], _S["coff"][ti + 1]
    jl, jr, jn = _S["clib"][a:b].astype(np.int64), _S["creg"][a:b].astype(np.int64), _S["cn"][a:b].astype(np.int64)
    if len(jl):
        cls = np.where(jr >= NSH0, 1, np.where(jr >= NV, 0, 2))           # membranes, then shells, then volumes
        o = np.lexsort((-_S["BR"][jl], cls)); jl, jr, jn = jl[o], jr[o], jn[o]
    # obstacles: pre-pass instances in this tile + neighbours, and packed neighbour tiles
    OCs, ORs = [], []
    for j in [ti] + _neigh(t):
        p0, p1 = _S["poff"][j], _S["poff"][j + 1]
        if p1 > p0:
            C0, R0 = p3.emit_spheres(_S["ptypes"][p0:p1].astype(np.int64), _S["ppos"][p0:p1].astype(np.float64), _S["pquat"][p0:p1].astype(np.float64),
                                     _S["pso"], _S["pSTC"], _S["pSTR"], _S["pBR"], glo, ghi)
            OCs.append(C0); ORs.append(R0)
        if "moff" in _S:
            m0, m1 = _S["moff"][j], _S["moff"][j + 1]
            if m1 > m0:
                mp_ = np.asarray(_S["mpts"][m0:m1], np.float64)
                kk = np.all((mp_ >= glo - 1) & (mp_ <= ghi + 1), axis=1)
                OCs.append(mp_[kk]); ORs.append(np.full(int(kk.sum()), _S["mrad"]))
        f = f"{W}/tiles/{j}.npz"
        if j != ti and os.path.exists(f):
            z = np.load(f)
            C1, R1 = p3.emit_spheres(z["types"].astype(np.int64), z["pos"].astype(np.float64), z["quat"].astype(np.float64),
                                     _S["so"], _S["STC"], _S["STR"], _S["BR"], glo, ghi)
            OCs.append(C1); ORs.append(R1)
    OC = np.concatenate(OCs) if OCs else np.zeros((0, 3)); OR = np.concatenate(ORs) if ORs else np.zeros(0)
    nsph = int((np.diff(_S["so"])[jl] * jn).sum()) if len(jl) else 0
    cap_s = nsph + len(OR) + 1000; cap_e = cap_s * 14
    e = np.zeros(0, np.int64)
    res = p3.pack_tile(int(ti) * 7919 + 11, lo, TILE, HALO, cell.G, cell.pores, cell.pidx, cell.PT, cell.PQ, cell.PC,
                       _S["so"], _S["STC"], _S["STR"], _S["BR"], jp.regmask(), jl, jr, jn, NV, np.array(vox_off, np.int64), VOXP, VOX,
                       np.array(tri_off, np.int64), TRI, CUMA, OC, OR, 0, 1500.0, 0, 200, 25, cap_s, cap_e,
                       e, np.zeros((0, 3)), e, e, np.zeros(0), e, np.zeros(7), _S["QF"], jp.regmem(), *jp.shell_arrays())
    types, pos, quat, placed, _, _, overflow, ns, ne, _ = res
    tmp_f = f"{W}/tiles/.{ti}.tmp.npz"
    np.savez(tmp_f, types=types.astype(np.int32), pos=pos, quat=quat, job_lib=jl.astype(np.int32), job_region=jr.astype(np.int16),
             job_n=jn.astype(np.int32), job_placed=placed.astype(np.int32), overflow=overflow)
    os.replace(tmp_f, out_f)                                        # atomic: a killed run never leaves a partial tile
    return ti, int(jn.sum()), int(placed.sum()), bool(overflow), time.time() - t0


def step_pack(limit=None):
    os.environ["NUMBA_NUM_THREADS"] = "1"
    M = pd.read_parquet(f"{W}/tiles.parquet"); meta = json.load(open(f"{W}/tiles_meta.json"))
    cell, _ = load_cell()
    L = json.load(open(f"{W}/libs.json"))["libs"]; so, STC, STR, BR = _libpack(L)
    pl = json.load(open(f"{W}/prepass_libs.json")); pso, pSTC, pSTR, pBR = _libpack(pl)
    cz = np.load(f"{W}/counts.npz")
    coff = np.searchsorted(cz["tile"], np.arange(len(M) + 1))
    Tij = M[["ti", "tj", "tk"]].values
    _S.update(cell=cell, Tij=Tij, lo0=np.array(meta["lo0"]), key={tuple(t): i for i, t in enumerate(Tij)},
              so=so, STC=STC, STR=STR, BR=BR, pso=pso, pSTC=pSTC, pSTR=pSTR, pBR=pBR, QF=(np.zeros((0, 4)) if os.environ.get("JWC_NO_LUMEN_ALIGN") else jp.qflat(so, STC, STR)),
              clib=cz["lib"], creg=cz["region"], cn=cz["n"], coff=coff,
              # loaded fully into RAM before the fork (shared copy-on-write): random mmap reads on Lustre left workers I/O-bound
              ptypes=np.load(f"{W}/prepass_types.npy"), ppos=np.load(f"{W}/prepass_pos.npy"),
              pquat=np.load(f"{W}/prepass_quat.npy"), poff=np.load(f"{W}/prepass_off.npy"))
    if os.path.exists(f"{W}/mrna_off.npy"):
        _S.update(mpts=np.load(f"{W}/mrna_pts.npy"), moff=np.load(f"{W}/mrna_off.npy"),
                  mrad=json.load(open(f"{W}/mrna_meta.json"))["bead_radius"])
    os.makedirs(f"{W}/tiles", exist_ok=True)
    sel = np.arange(len(M)) if limit is None else limit
    colour = (Tij[sel, 0] % 2) * 4 + (Tij[sel, 1] % 2) * 2 + (Tij[sel, 2] % 2)
    import multiprocessing as mp
    log = open(f"{W}/pack_log.tsv", "a")
    for ph in range(8):
        tiles = sel[colour == ph]
        t0 = time.time(); done = 0; inst = plc = 0; ovf = 0
        with mp.get_context("fork").Pool(NPROC) as pool:
            for ti, ni, np_, ov, dt in pool.imap_unordered(_pack_tile, tiles, chunksize=1):
                done += 1
                if ni >= 0:
                    inst += ni; plc += np_; ovf += ov; log.write(f"{ph}\t{ti}\t{ni}\t{np_}\t{int(ov)}\t{dt:.1f}\n")
                if done % 200 == 0:
                    log.flush(); print(f"  phase {ph}: {done}/{len(tiles)} tiles, placed {plc:,}/{inst:,}, {time.time() - t0:.0f}s", flush=True)
        log.flush()
        print(f"phase {ph} done: {len(tiles)} tiles, placed {plc:,}/{inst:,} ({plc / max(inst, 1):.1%}), overflow {ovf}, {time.time() - t0:.0f}s", flush=True)


def step_summary():
    M = pd.read_parquet(f"{W}/tiles.parquet"); L = json.load(open(f"{W}/libs.json")); mass = np.array(L["mass_da"])
    exp = {}; plc = {}; ninst = 0; nt = 0
    for ti in range(len(M)):
        f = f"{W}/tiles/{ti}.npz"
        if not os.path.exists(f): continue
        z = np.load(f); nt += 1; ninst += len(z["types"])
        for l, r, n, p in zip(z["job_lib"], z["job_region"], z["job_n"], z["job_placed"]):
            exp[r] = exp.get(r, 0.0) + n * mass[l]; plc[r] = plc.get(r, 0.0) + p * mass[l]
    out = {RNAMES[r]: dict(expected_pg=round(exp[r] / 6.022e23 * 1e12, 2), placed_pg=round(plc[r] / 6.022e23 * 1e12, 2),
                           frac=round(plc[r] / max(exp[r], 1), 3)) for r in sorted(exp)}
    tot_e, tot_p = sum(exp.values()), sum(plc.values())
    S = dict(tiles_packed=nt, instances=ninst, placed_pg=round(tot_p / 6.022e23 * 1e12, 2), expected_pg=round(tot_e / 6.022e23 * 1e12, 2),
             mass_frac=round(tot_p / max(tot_e, 1), 3), regions=out)
    json.dump(S, open(f"{W}/summary.json", "w"), indent=1); print(json.dumps(S, indent=1))


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "tiles": step_tiles()
    elif step == "counts": step_counts()
    elif step == "index": step_index()
    elif step == "pack":
        lim = None
        if len(sys.argv) > 2:                      # pack a sub-box: x0 y0 z0 x1 y1 z1 (model nm)
            M = pd.read_parquet(f"{W}/tiles.parquet"); meta = json.load(open(f"{W}/tiles_meta.json"))
            b = np.array([float(x) for x in sys.argv[2:8]]); c = M[["ti", "tj", "tk"]].values * TILE + np.array(meta["lo0"]) + TILE / 2
            lim = np.where(np.all((c >= b[:3]) & (c <= b[3:]), axis=1))[0]; print("sub-box tiles:", len(lim))
        step_pack(lim)
    elif step == "summary": step_summary()
