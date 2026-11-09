"""Driver for the 3D whole-cell packing: library pack, expected counts per tile, NPC + chromatin pre-pass,
checkerboard-phased parallel tile packing."""
import os, json, glob, time
import numpy as np, pandas as pd
import numba as nb
import cell3d as c3, tiles3d as t3, pack3d as p3

from config import W3D as W, LIB, RAW, NPROC
CHROM_KB = [230, 813, 316, 1532, 577, 270, 1091, 563, 440, 746, 667, 1078, 924, 784, 1091, 948]  # chr I-XVI
N_TRNA = 3.3e6
N_MRNA = 3.0e4
MRNA_MEAN_NT = 1500


# ------------------------------------------------------------------ library
def build_libpack(lib_ids):
    st_off = [0]; C, R, BR, MEM, NB = [], [], [], [], []
    for i in lib_ids:
        z = np.load(f"{LIB}/{i}.npz")
        c = z["st_c"].astype(np.float64); r = z["st_r"].astype(np.float64)
        C.append(c); R.append(r); st_off.append(st_off[-1] + len(r))
        BR.append(float((np.linalg.norm(c, axis=1) + r).max()))
        MEM.append("membrane" in z.files); NB.append(len(z["beads"]))
    return dict(ids=np.array(lib_ids), st_off=np.array(st_off, np.int64), STC=np.concatenate(C), STR=np.concatenate(R),
                BR=np.array(BR), memb=np.array(MEM), nbeads=np.array(NB))


# ------------------------------------------------------------------ expected totals per (lib_id, region)
R = t3


SOLUBLE_ONLY = ("cytosol", "nucleoplasm", "nucleolus")
# outer nuclear membrane (half of the 18.3 um2 NE) / (cortical ER 51.8 um2 + ONM 9.1 um2), from tile_measure areas
ER_ON_NE = 0.15


def expected_totals(ing2, oligo_map):
    rows = []
    for r in ing2.itertuples():
        if r.ing_id in ("NPC", "nucleosome"):
            continue
        lib_id, n = r.ing_id, r.copies_per_cell
        if lib_id in oligo_map:
            lib_id, k = oligo_map[lib_id]; n = n / k
        if not os.path.exists(f"{LIB}/{lib_id}.npz"):
            rows.append((lib_id, r.ing_id, "no_structure", n)); continue
        ntm = float(r.n_tm) if r.n_tm == r.n_tm else 0.0
        tm = ntm > 0 or (isinstance(r.membrane, str) and r.membrane in ("PM", "NE", "VAC"))
        cw = json.loads(r.comp_weights)
        if tm:   # an integral membrane protein listed in soluble and membrane compartments lives in the membranes
            mem = {c: f for c, f in cw.items() if c not in SOLUBLE_ONLY}
            if mem:
                z = sum(mem.values()); cw = {c: f / z for c, f in mem.items()}
        for c, f in cw.items():
            cp = n * f
            if cp <= 0:
                continue
            def add(reg, x=cp):
                rows.append((lib_id, r.ing_id, reg, x))
            if c == "cytosol":
                add("cytosol" if not tm else "dropped_TM_in_soluble_compartment")
            elif c == "nucleoplasm":
                add("nucleoplasm" if not tm else "dropped_TM_in_soluble_compartment")
            elif c == "nucleolus":
                add("nucleolus" if not tm else "dropped_TM_in_soluble_compartment")
            elif c == "mito_IM":
                add("mem_MITO_IM" if tm else "mito_matrix")     # peripheral IM proteins: mostly matrix-side (F1, bc1 core)
            elif c == "mito_OM":
                add("mem_MITO_OM" if tm else "cytosol")         # peripheral OM proteins face the cytosol
            elif c == "mito_IMS":
                add("mem_MITO_IM" if tm else "mito_IMS")
            elif c == "mito_matrix":
                add("mem_MITO_IM" if tm else "mito_matrix")
            elif c == "mitochondrion":
                if tm:
                    add("mem_MITO_IM", 0.75 * cp); add("mem_MITO_OM", 0.25 * cp)
                else:
                    add("mito_matrix", 0.9 * cp); add("mito_IMS", 0.1 * cp)
            elif c == "ER":
                if tm:   # the outer nuclear membrane is ER: share ER membrane proteins by membrane area
                    add("mem_ER", (1 - ER_ON_NE) * cp); add("mem_NE", ER_ON_NE * cp)
                elif r.ing_id == "ribosome_80S":
                    add("ER_peripheral")
                else:
                    add("ER_lumen")
            elif c == "plasma_membrane":
                add("mem_PM" if tm else "PM_peripheral")
            elif c == "nuclear_envelope":
                add("mem_NE" if tm else "NE_peripheral")
            elif c == "vacuole":
                add("mem_VAC" if tm else "vacuole")
            elif c == "golgi_endosome":
                add("mem_GOLGI" if tm else "golgi_endosome")
            elif c == "peroxisome":
                add("mem_PEROX" if tm else "peroxisome")
            elif c == "lipid_droplet":
                add("mem_ER" if tm else "LD_peripheral")
            elif c == "cell_wall":
                add("cell_wall")
    rows.append(("tRNA", "tRNA", "cytosol", N_TRNA))
    e = pd.DataFrame(rows, columns=["lib_id", "ing_id", "region", "expected"])
    return e.groupby(["lib_id", "ing_id", "region"], as_index=False).expected.sum()


def crowding_factors(e, M, reg_idx, logmod, vol, iters=30):
    """Per region, tile factors c_t for the modulated types such that the expected macromolecular volume per tile
    stays proportional to the region's measure in the tile (composition varies between tiles, crowding does not)"""
    out = {}
    for reg, g in e[e.lib_id.isin(logmod)].groupby("region"):
        ri = reg_idx[reg]; W = M[:, ri]; nz = np.nonzero(W)[0]
        if len(nz) == 0:
            continue
        A = np.stack([np.exp(logmod[l][nz] - logmod[l][nz].max()) for l in g.lib_id])         # rows x tiles
        V = (g.expected.values * np.array([vol[l] for l in g.lib_id]))[:, None]
        c = np.ones(len(nz))
        for _ in range(iters):
            Wc = A * (W[nz] * c)[None]
            occ = (V * Wc / Wc.sum(1, keepdims=True)).sum(0)
            ratio = (W[nz] / W[nz].sum()) / (occ / occ.sum())
            c *= ratio
        cf = np.zeros(len(W)); cf[nz] = c / c.mean(); out[ri] = cf
    return out


def distribute(exp, measure, lib_index, seed=3, logmod=None, vol=None):
    """Multinomial split of each (lib, region) total over tiles proportional to the tile's region measure, times
    exp(logmod[lib]) per tile if given (organisation.tile_logmod: large-scale co-enrichment of STRING modules) and
    the crowding correction for those types (crowding_factors, needs vol: lib id -> volume).
    Returns long table tile_i, lib_i, region_i, n (cell totals conserved after stochastic rounding)."""
    rng = np.random.default_rng(seed)
    M = measure.values
    reg_idx = {n: i for i, n in enumerate(t3.REGION_NAMES)}
    e = exp[exp.region.isin(reg_idx)].groupby(["lib_id", "region"], as_index=False).expected.sum()
    cf = crowding_factors(e, M, reg_idx, logmod, vol) if logmod is not None else {}
    T, Li, Ri, Nn = [], [], [], []
    for r in e.itertuples():
        ri = reg_idx[r.region]
        w = M[:, ri]
        if w.sum() <= 0:
            continue
        N = int(np.floor(r.expected) + (rng.random() < r.expected - np.floor(r.expected)))
        if N == 0:
            continue
        nz = np.nonzero(w)[0]
        if logmod is not None and r.lib_id in logmod:
            lw = logmod[r.lib_id][nz]
            w = np.zeros_like(w); w[nz] = M[nz, ri] * np.exp(lw - lw.max()) * (cf[ri][nz] if ri in cf else 1.0)
        cnt = rng.multinomial(N, w[nz] / w[nz].sum())
        k = cnt > 0
        T.append(nz[k]); Li.append(np.full(k.sum(), lib_index[r.lib_id])); Ri.append(np.full(k.sum(), ri)); Nn.append(cnt[k])
    d = pd.DataFrame(dict(tile=np.concatenate(T).astype(np.int32), lib=np.concatenate(Li).astype(np.int32),
                          region=np.concatenate(Ri).astype(np.int16), n=np.concatenate(Nn).astype(np.int32)))
    return d.sort_values(["tile"]).reset_index(drop=True)


# ------------------------------------------------------------------ pre-pass: NPCs + chromatin
def prepass(cell, lp, n_nuc, seed=5, log=print, before_chromatin=None):
    """NPCs, then chromatin. before_chromatin(npc_spheres) may place more objects first and return a dict with
    'obstacles' (centres, radii) for the walks and optional 'starts' (one start point per chromosome)."""
    li_npc = int(np.where(lp["ids"] == "NPC")[0][0]); li_nuc = int(np.where(lp["ids"] == "nucleosome")[0][0])
    # NPC instances: library +z (cytoplasmic) along the pore normal
    npc_pos = cell.npc_pos
    q = np.empty(4); npc_q = []
    rng = np.random.default_rng(seed)
    for u in cell.pores[:, 3:6]:          # NE normals
        p3.quat_z_to(u.astype(np.float64), rng.uniform(0, 2 * np.pi), q); npc_q.append(q.copy())
    npc_q = np.array(npc_q)
    types = np.full(len(npc_pos), li_npc, np.int32)
    cn = cell.G[3:6]; rn = float(cell.G[11:14].max())
    glo = cn - rn - 80; ghi = cn + rn + 80
    OC, OR = p3.emit_spheres(types, npc_pos.astype(np.float32), npc_q.astype(np.float32), lp["st_off"], lp["STC"], lp["STR"], lp["BR"], glo, ghi)
    lens = np.array(CHROM_KB, float); lens = np.round(lens / lens.sum() * n_nuc).astype(np.int64)
    dirn = cell.G[7:10] - cn; dirn /= np.linalg.norm(dirn)
    spb = c3.nuc_surface(cell.G, -dirn, -(c3.PERI_W + 40))[0]
    starts, r_start = np.tile(spb, (len(lens), 1)), 150.0
    extra = {}
    if before_chromatin is not None:
        extra = before_chromatin(p3.emit_spheres(types, npc_pos.astype(np.float32), npc_q.astype(np.float32), lp["st_off"],
                                                 lp["STC"], lp["STR"], lp["BR"], np.full(3, -1e4), np.full(3, 1e4)))
        if "obstacles" in extra:
            C, R = extra["obstacles"]; k = np.all((C > glo) & (C < ghi), axis=1)
            OC = np.concatenate([OC, C[k]]); OR = np.concatenate([OR, R[k]])
        if len(extra.get("starts", ())) == len(lens):     # chromosome walks start at the kinetochore-MT plus ends
            starts, r_start = np.asarray(extra["starts"], np.float64), 20.0
    a = cell.args(glo, ghi, margin=200.0)
    s0, s1 = lp["st_off"][li_nuc], lp["st_off"][li_nuc + 1]
    mask = 1 << c3.NUCP
    nsph = (s1 - s0) * lens.sum() + len(OR)
    t0 = time.time()
    pos, qu, chain, brk = p3.chromatin_walk(seed, glo, ghi, *a, lp["STC"][s0:s1], lp["STR"][s0:s1], lp["BR"][li_nuc], mask,
                                            OC, OR, lens, starts, r_start, 12.5, int(nsph * 1.1) + 1000, int(nsph * 10) + 10000)
    log(f"chromatin: {len(pos)}/{lens.sum()} nucleosomes, {int(brk.sum())} fragments, {time.time()-t0:.0f}s")
    out = dict(npc_pos=npc_pos.astype(np.float32), npc_q=npc_q.astype(np.float32), li_npc=li_npc, li_nuc=li_nuc,
               nuc_pos=pos, nuc_q=qu, nuc_chain=chain, nuc_brk=brk, chain_target=lens)
    out.update({k: v for k, v in extra.items() if k not in ("obstacles", "starts")})
    return out


# ------------------------------------------------------------------ tile worker
_S = {}   # shared state (set before fork)


PRE_GROUPS = [("npc", "li_npc"), ("nuc", "li_nuc"), ("poly", "li_ribo"), ("act", "li_actin"), ("tpm", "li_tpm"), ("mt", None), ("pat", None), ("atp", "li_atp")]


def pre_instances(pre):
    """types, positions, quaternions of all pre-placed instances: NPCs, nucleosomes, polysome ribosomes, actin,
    microtubules (+ gamma-TuRC caps), actin patches (actin, Arp2/3, capping protein); per-instance types in *_type"""
    parts = [(k, li) for k, li in PRE_GROUPS if f"{k}_pos" in pre]
    gt = np.concatenate([pre[f"{k}_type"].astype(np.int32) if li is None else np.full(len(pre[f"{k}_pos"]), int(pre[li]), np.int32)
                         for k, li in parts])
    gp = np.concatenate([pre[f"{k}_pos"] for k, _ in parts]); gq = np.concatenate([pre[f"{k}_q"] for k, _ in parts])
    return gt, gp, gq


def setup_shared(cell, lp, counts, tiles, pre, halo, partners=None, prm=None):
    """partners, prm: organisation.partners() CSR arrays and organisation.params() (None: uniform random placement)"""
    _S.update(cell=cell, lp=lp, tiles=tiles, halo=halo, tindex={t: i for i, t in enumerate(tiles)})
    _S["partners"] = partners; _S["prm"] = prm
    _S["counts"] = {k: g for k, g in counts.groupby("tile")}
    gt, gp, gq = pre_instances(pre)
    _S["glob"] = (gt, gp, gq)
    # pre-placed RNA beads (polysome linkers, full-length mRNA threads) as packing obstacles, bucketed by tile
    pb = [pre[k].astype(np.float32) for k in ("poly_link", "mrna_pts", "fg_pts") if k in pre and len(pre[k])]
    pb = np.concatenate(pb) if pb else np.zeros((0, 3), np.float32)
    key = np.floor(pb / t3.TILE).astype(np.int64)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0])); pb = pb[order]; key = key[order]
    brk = np.r_[0, np.where(np.any(np.diff(key, axis=0) != 0, axis=1))[0] + 1, len(key)]
    _S["rna_beads"] = pb
    _S["rna_bucket"] = {tuple(key[a]): (a, b) for a, b in zip(brk[:-1], brk[1:]) if b > a}
    _S["mrna"] = None


def neighbours(t):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx or dy or dz:
                    yield (t[0] + dx, t[1] + dy, t[2] + dz)


def pack_one(ti, n_mrna, seed, max_tries=200, give_up=25):
    cell, lp, halo = _S["cell"], _S["lp"], _S["halo"]
    t = _S["tiles"][ti]
    lo = np.array(t, float) * t3.TILE; hi = lo + t3.TILE
    glo = lo - halo; ghi = hi + halo
    a = cell.args(lo, hi, margin=80.0)
    g = np.load(f"{t3.GDIR}/{t3.tile_name(t)}.npz")
    # membrane triangles grouped by membrane id 0..7
    V, F, FM = g["verts"].astype(np.float64), g["faces"], g["face_mem"]
    order = np.argsort(FM, kind="stable")
    tri = V[F[order]].reshape(-1, 9) if len(F) else np.zeros((0, 9))
    fm = FM[order]
    tri_off = np.searchsorted(fm, np.arange(9)).astype(np.int64)
    ar = 0.5 * np.linalg.norm(np.cross(tri[:, 3:6] - tri[:, 0:3], tri[:, 6:9] - tri[:, 0:3]), axis=1) if len(tri) else np.zeros(0)
    cuma = np.cumsum(ar) if len(ar) else np.zeros(1)
    # voxel seeds per volume region
    n = int(t3.TILE / t3.VOX)
    band_m = np.array([t3.BANDS[r][0] for r in t3.BAND_ORDER], np.int64)
    band_w = np.array([t3.BANDS[r][1] for r in t3.BAND_ORDER], np.float64)
    lab, bits = t3._voxel_regions(lo, n, t3.VOX, *a, band_m, band_w, t3.WALL_DEPTH)
    masks = t3.region_masks(lab, bits)
    vox_off = [0]; VP = []
    for r in range(t3.NREG):
        if r < t3.MEM_REGION0:
            ijk = np.argwhere(masks[r]); VP.append(lo + (ijk + 0.5) * t3.VOX)
            vox_off.append(vox_off[-1] + len(ijk))
        else:
            vox_off.append(vox_off[-1])
    VOXP = np.concatenate(VP) if VP else np.zeros((0, 3))
    # jobs: membranes first, then volume; each block largest bounding radius first
    c = _S["counts"].get(ti)
    if c is None:
        jl = np.zeros(0, np.int64); jr = np.zeros(0, np.int64); jn = np.zeros(0, np.int64)
    else:
        c = c.assign(br=lp["BR"][c.lib.values], mem=(c.region.values >= t3.MEM_REGION0))
        c = c.sort_values(["mem", "br"], ascending=[False, False])
        jl = c.lib.values.astype(np.int64); jr = c.region.values.astype(np.int64); jn = c.n.values.astype(np.int64)
    # obstacles: global (NPC, chromatin) + finished neighbour tiles
    gt, gp, gq = _S["glob"]
    OCs, ORs = [], []
    ATs, APs = [], []         # anchors for partner attraction: placed instances around the tile
    k = np.all((gp >= glo) & (gp <= ghi), axis=1); ATs.append(gt[k]); APs.append(gp[k])
    C0, R0 = p3.emit_spheres(gt, gp, gq, lp["st_off"], lp["STC"], lp["STR"], lp["BR"], glo, ghi)
    OCs.append(C0); ORs.append(R0)
    sl_ = [(_S["rna_bucket"][nt]) for nt in [t] + list(neighbours(t)) if nt in _S["rna_bucket"]]   # pre-placed RNA beads
    if sl_:
        pl = np.concatenate([_S["rna_beads"][a:b] for a, b in sl_]).astype(np.float64)
        k = np.all((pl >= glo - 1) & (pl <= ghi + 1), axis=1)
        OCs.append(pl[k]); ORs.append(np.full(k.sum(), 0.5))
    for nt in neighbours(t):
        f = f"{RAW}/{t3.tile_name(nt)}.npz"
        if nt in _S["tindex"] and os.path.exists(f):
            z = np.load(f)
            C1, R1 = p3.emit_spheres(z["types"], z["pos"], z["quat"], lp["st_off"], lp["STC"], lp["STR"], lp["BR"], glo, ghi)
            OCs.append(C1); ORs.append(R1)
            k = np.all((z["pos"] >= glo) & (z["pos"] <= ghi), axis=1); ATs.append(z["types"][k]); APs.append(z["pos"][k])
            mp_ = z["mrna_pts"]
            if len(mp_):
                k = np.all((mp_ >= glo - 1) & (mp_ <= ghi + 1), axis=1)
                OCs.append(mp_[k].astype(np.float64)); ORs.append(np.full(k.sum(), 0.5))
    OC = np.concatenate(OCs); OR = np.concatenate(ORs)
    nsph = int((np.diff(lp["st_off"])[jl] * jn).sum()) if len(jl) else 0
    cap_s = len(OR) + nsph + n_mrna * 8000 + 1000
    cap_e = cap_s * 14
    t0 = time.time()
    pr = _S.get("partners")
    if pr is None:
        pr = dict(off=np.zeros(0, np.int64), idx=np.zeros(0, np.int64), strength=np.zeros(0), kind=np.zeros(0, np.int64))
    prm = _S.get("prm") if _S.get("prm") is not None else np.zeros(7)
    AT = np.concatenate(ATs).astype(np.int64); AP = np.concatenate(APs).astype(np.float64)
    res = p3.pack_tile(seed, lo, t3.TILE, halo, *a, lp["st_off"], lp["STC"], lp["STR"], lp["BR"], t3.REG_MASK,
                       jl, jr, jn, t3.MEM_REGION0, np.array(vox_off, np.int64), VOXP, t3.VOX,
                       tri_off, tri, cuma, OC, OR, n_mrna, MRNA_MEAN_NT, t3.R_CYTO, max_tries, give_up, cap_s, cap_e,
                       AT, AP, pr["off"], pr["idx"], pr["strength"], pr["kind"], prm)
    types, pos, quat, placed, mpts, moff, overflow, ns, ne, att = res
    np.savez(f"{RAW}/{t3.tile_name(t)}.npz", types=types, pos=pos, quat=quat, job_lib=jl.astype(np.int32),
             job_region=jr.astype(np.int16), job_n=jn.astype(np.int32), job_placed=placed.astype(np.int32),
             mrna_pts=mpts, mrna_off=moff, overflow=overflow)
    return ti, int(jn.sum()), int(placed.sum()), bool(overflow), time.time() - t0, len(moff) - 1, int(att[0]), int(att[1])


def _pw(args):
    nb.set_num_threads(1)
    return pack_one(*args)


def run_phase(tile_ids, mrna_counts, nproc=NPROC, seed0=1000, log=print):
    import multiprocessing as mp
    os.makedirs(RAW, exist_ok=True)
    jobs = [(ti, int(mrna_counts[ti]), seed0 + ti) for ti in tile_ids]
    # heavier tiles first for load balance
    jobs.sort(key=lambda j: -int(_S["counts"][j[0]].n.sum()) if j[0] in _S["counts"] else 0)
    out = []
    with mp.get_context("fork").Pool(nproc) as pool:
        for r in pool.imap_unordered(_pw, jobs, chunksize=1):
            out.append(r)
    return out


def phase_of(t):
    return (t[0] & 1) + 2 * (t[1] & 1) + 4 * (t[2] & 1)
