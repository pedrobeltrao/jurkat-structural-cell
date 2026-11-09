"""Chromatin pre-pass for the Jurkat cell (fork B, 2026-11-08).

Nucleosome chains: one persistent self-avoiding walk per chromosome (46: 22 autosome pairs + X + Y, GRCh38 lengths,
near-diploid male Jurkat), 3.1e7 nucleosomes in total (DNA-based, PLANNING_NOTES §4c), count per chromosome
proportional to length. Adapted from yeast pack3d.chromatin_walk (step 12.5 nm, persistence 0.5, fragment restart
after 40 consecutive failures), with Jurkat changes:
  - chromosome territories: nucleoplasm voxels (32 nm grid; label NUCP from cell3d, minus nucleoli from EM
    nucleolus_seg) are split into 46 power-diagram regions with volume proportional to chromosome length (k-means
    centroids, then additive weights balanced iteratively). Radial order: centroids sorted interior -> periphery
    (depth below the NE midsurface) are assigned to chromosomes by decreasing gene density (gene-dense 19, 17, 22
    interior; gene-poor 18, 13, 4, X, Y peripheral; Croft 1999 / Boyle 2001 trend). Each walk is confined to its
    territory; a one-voxel (32 nm) band along territory borders is forbidden, so territories can be walked
    independently (in parallel) without cross-territory overlap checks
  - restarts (fragment starts) are drawn inside the chromosome's own territory (yeast restarted near G[3:6])
  - collision model of the walk: coarse nucleosome sphere tree (structlib.sphere_tree, per=300 -> 5 spheres); the
    output instances use the full NUCLEOSOME library entry (1KX5 beads, 43 spheres) downstream
  - exclusion from nucleoli (they are mostly rRNA/protein), the NE/perinuclear space, pores: check_volume with
    label mask NUCP + territory grid
Linker histone H1: after each chromosome's walk, H1 copies (H1-0, -1, -2, -3, -4, -5, -10 in proportion to their
free copies, budget = total H1 copies) are placed with their library centroid on the nucleosome dyad axis
(direction from the nucleosome centre to the mid-point of the DNA ends, in the disk plane) at 6-9 nm from the
nucleosome centre, random orientation, overlap check that ignores the parent nucleosome.
Not modelled: linker DNA beads (procedural in the viewer, as yeast), heterochromatin compaction at the lamina,
loop/TAD structure, sister chromatids (G1 assumed).

usage:
  jchromatin.py setup                 -> data/derived/prepass/chromatin_territories.npz (+ QC)
  jchromatin.py block                 -> test on the nucleus part of the 1 um packing-test block
  jchromatin.py chrom <i> [<j> ...]   -> walk chromosomes i.. (0-45) -> data/derived/prepass/chromatin_parts/chr_<i>.npz
  jchromatin.py merge                 -> data/derived/prepass/chromatin_cell1.npz + json + QC png
"""
import os, sys, json, time
import numpy as np, numba as nb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3, pack3d as p3, structlib as sl
from config import LIB

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"; OUT = f"{D}/prepass"; PARTS = f"{OUT}/chromatin_parts"
os.makedirs(PARTS, exist_ok=True); os.makedirs(f"{D}/qc", exist_ok=True)
N_NUC = 31_000_000
STEP = 12.5
TG_H = 32.0
# GRCh38 primary assembly chromosome lengths (bp)
CHR_LEN = {"1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555, "5": 181538259, "6": 170805979, "7": 159345973,
           "8": 145138636, "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309, "13": 114364328, "14": 107043718,
           "15": 101991189, "16": 90338345, "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167, "21": 46709983,
           "22": 50818468, "X": 156040895, "Y": 57227415}
# gene density rank (most -> least gene-dense), approximate genes/Mb ordering
GENE_RANK = ["19", "17", "22", "16", "20", "11", "1", "15", "12", "14", "9", "10", "7", "3", "6", "2", "8", "21", "5", "X", "4", "13", "18", "Y"]
H1_GENES = ["H1-0", "H1-1", "H1-2", "H1-3", "H1-4", "H1-5", "H1-10"]


def chromosomes():
    names = []
    for c in [str(i) for i in range(1, 23)]:
        names += [c + "a", c + "b"]
    names += ["Xa", "Ya"]
    L = np.array([CHR_LEN[n[:-1]] for n in names], float)
    n = np.floor(N_NUC * L / L.sum()).astype(np.int64)
    n[np.argmax(L)] += N_NUC - n.sum()
    return names, L, n


def load_cell():
    g = np.load(f"{D}/jgeom_cell1.npz")
    return c3.Cell(g["G"], g["pores"]), g["centre_nm"]


# ------------------------------------------------------------------ territories
def setup():
    sys.path.insert(0, V); from jurkat.em.io import read as em_read
    t0 = time.time()
    cell, centre = load_cell()
    names, L, nn = chromosomes()
    # nucleus bbox from a coarse label scan
    R = cell.G[0]
    g = np.arange(-R, R, 128.0)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij"); P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    lab = cell.labels(P); Pn = P[lab == c3.NUCP]
    lo = Pn.min(0) - 256; hi = Pn.max(0) + 256
    dims = np.ceil((hi - lo) / TG_H).astype(int)
    gx = [lo[a] + (np.arange(dims[a]) + 0.5) * TG_H for a in range(3)]
    X, Y, Z = np.meshgrid(*gx, indexing="ij"); P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    lab = cell.labels(P).reshape(dims)
    nuc = lab == c3.NUCP
    del X, Y, Z
    # nucleoli from EM (label frame s2 grid 16 x 16 x 13.76; model = label nm - centre)
    vox = np.array([16.0, 16.0, 13.76])
    llo = lo + centre; lhi = hi + centre
    sl_ = tuple(slice(max(int(np.floor(a / v)), 0), int(np.ceil(b / v))) for a, b, v in zip(llo, lhi, vox))
    nl = em_read("nucleolus_seg", "s2", sl_) > 0
    o = np.array([s.start for s in sl_]) * vox - centre
    idx = np.clip(((P - o) / vox).astype(int), 0, np.array(nl.shape) - 1)
    nucl = nl[idx[:, 0], idx[:, 1], idx[:, 2]].reshape(dims) & nuc
    allowed = nuc & ~nucl
    print(f"territory grid {dims.tolist()} @ {TG_H} nm; nucleoplasm {nuc.sum() * TG_H**3 / 1e9:.1f} um3, nucleoli {nucl.sum() * TG_H**3 / 1e9:.1f} um3, "
          f"allowed {allowed.sum() * TG_H**3 / 1e9:.1f} um3 ({time.time() - t0:.0f}s)", flush=True)
    pts = P[allowed.ravel()]
    # depth below the NE midsurface (positive inside)
    depth = -cell.mems(pts)[:, c3.M_NE] - 0.5 * c3.PERI_W
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(0)
    sub = pts[rng.choice(len(pts), min(400_000, len(pts)), replace=False)]
    km = MiniBatchKMeans(n_clusters=len(names), random_state=0, n_init=3, batch_size=20000).fit(sub)
    C = km.cluster_centers_
    # radial assignment: interior centroids <- gene-dense chromosomes
    cdep = -cell.mems(C)[:, c3.M_NE]
    rank = {c: i for i, c in enumerate(GENE_RANK)}
    chr_order = sorted(range(len(names)), key=lambda i: (rank[names[i][:-1]], names[i]))
    cen_order = np.argsort(-cdep)
    C = C[cen_order][np.argsort(chr_order)]          # C[i] = centroid of chromosome i
    # power-diagram balancing to volumes proportional to length
    target = L / L.sum()
    w = np.zeros(len(names))
    def assign(Q, w):
        out = np.empty(len(Q), np.int64)
        for a in range(0, len(Q), 500_000):
            d = ((Q[a:a + 500_000, None, :] - C[None]) ** 2).sum(-1) - w[None]
            out[a:a + 500_000] = d.argmin(1)
        return out
    subb = pts[rng.choice(len(pts), min(300_000, len(pts)), replace=False)]
    scale = np.mean(((subb[:2000, None] - C[None]) ** 2).sum(-1))
    step = 0.02 * scale; best = (1e9, w.copy())
    for it in range(400):
        a = assign(subb, w); f = np.bincount(a, minlength=len(names)) / len(a)
        err = np.abs(f / target - 1).max()
        if err < best[0]: best = (err, w.copy())
        if err < 0.05: break
        w += step * np.clip((target - f) / np.maximum(target, 1e-9), -1, 1)
        if it % 50 == 49: step *= 0.7
    w = best[1]; print(f"power-diagram balancing: max |vol/target - 1| = {best[0]:.3f} after {it + 1} iterations", flush=True)
    a = assign(pts, w)
    f = np.bincount(a, minlength=len(names)) / len(a)
    print("territory volume / target: min %.2f max %.2f" % ((f / target).min(), (f / target).max()), flush=True)
    TG = -np.ones(dims, np.int16)
    TG.reshape(-1)[np.flatnonzero(allowed.ravel())] = a
    # forbid a one-voxel band along territory borders (and keep only interior of each territory)
    band = np.zeros(dims, bool)
    for ax in range(3):
        for sh in (1, -1):
            r = np.roll(TG, sh, axis=ax)
            band |= (TG >= 0) & (r >= 0) & (r != TG)
    TGw = TG.copy(); TGw[band] = -1
    np.savez_compressed(f"{OUT}/chromatin_territories.npz", TG=TGw, TG_raw=TG, lo=lo, h=TG_H, names=np.array(names), n_nuc=nn,
                        frac=f, target=target, centroids=C, nucleolus=nucl)
    print(f"setup done {time.time() - t0:.0f}s; band voxels {band.sum()} ({band.sum() / allowed.sum():.1%})", flush=True)


# ------------------------------------------------------------------ numba core
@nb.njit(cache=True)
def _tg_ok(x, y, z, TG, tlo, th, cid):
    i = int((x - tlo[0]) / th); j = int((y - tlo[1]) / th); k = int((z - tlo[2]) / th)
    if i < 0 or j < 0 or k < 0 or i >= TG.shape[0] or j >= TG.shape[1] or k >= TG.shape[2]:
        return False
    return TG[i, j, k] == cid


@nb.njit(cache=True)
def walk(seed, glo, ghi, G, pores, pidx, PT, PQ, PC, stc, str_, br, mask, TG, tlo, th, cid, starts, n_target, step, cap_s, cap_e):
    np.random.seed(seed)
    dims, head, nxt, sid, cen, rad, cnt = p3.grid_new(glo, ghi, p3.H_GRID, cap_s, cap_e)
    pos = np.empty((n_target, 3), np.float32); qu = np.empty((n_target, 4), np.float32); brk = np.zeros(n_target, np.int8)
    out = np.zeros(c3.NMEM); M = np.empty((3, 3)); q = np.empty(4); t = np.empty(3)
    P = np.empty((stc.shape[0], 3))
    n = 0; have_prev = False
    dx, dy, dz = 0.0, 0.0, 0.0; px, py, pz = 0.0, 0.0, 0.0
    fails = 0; restarts = 0; overflow = False
    while n < n_target and restarts < 200000:
        if not have_prev:
            s = int(np.random.random() * starts.shape[0])
            for k in range(3):
                t[k] = starts[s, k] + (np.random.random() - 0.5) * th
        else:
            ex = np.random.normal(); ey = np.random.normal(); ez = np.random.normal()
            en = np.sqrt(ex * ex + ey * ey + ez * ez)
            ex = 0.5 * dx + ex / en; ey = 0.5 * dy + ey / en; ez = 0.5 * dz + ez / en
            en = np.sqrt(ex * ex + ey * ey + ez * ez)
            ex /= en; ey /= en; ez /= en
            t[0] = px + step * ex; t[1] = py + step * ey; t[2] = pz + step * ez
        good = _tg_ok(t[0], t[1], t[2], TG, tlo, th, cid)
        if good:
            p3.rand_quat(q); p3.quat_mat(q, M); p3.transform(stc, M, t, P)
            good = not p3.g_overlap(glo, p3.H_GRID, dims, head, nxt, sid, cen, rad, P, str_, p3.TOL)
            if good:
                good = p3.check_volume(P, str_, t[0], t[1], t[2], br, mask, G, pores, pidx, PT, PQ, PC, out)
        if not good:
            fails += 1
            if fails % 40 == 0 or not have_prev:
                have_prev = False; restarts += 1
            continue
        if not p3.g_insert(glo, p3.H_GRID, dims, head, nxt, sid, cen, rad, cnt, P, str_):
            overflow = True
            break
        if have_prev:
            dx = (t[0] - px) / step; dy = (t[1] - py) / step; dz = (t[2] - pz) / step
        else:
            brk[n] = 1
        px, py, pz = t[0], t[1], t[2]; have_prev = True
        for k in range(3):
            pos[n, k] = t[k]
        for k in range(4):
            qu[n, k] = q[k]
        n += 1; fails = 0
    return pos[:n], qu[:n], brk[:n], dims, head, nxt, sid, cen, rad, cnt, overflow, restarts


@nb.njit(cache=True)
def _overlap_skip(lo, h, dims, head, nxt, sid, cen, rad, P, R, tol, s0, s1):
    for s in range(P.shape[0]):
        x, y, z, r = P[s, 0], P[s, 1], P[s, 2], R[s]
        i0, i1 = p3._rng3(x, r, lo[0], h, dims[0]); j0, j1 = p3._rng3(y, r, lo[1], h, dims[1]); k0, k1 = p3._rng3(z, r, lo[2], h, dims[2])
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    e = head[(i * dims[1] + j) * dims[2] + k]
                    while e >= 0:
                        o = sid[e]
                        if o < s0 or o >= s1:
                            dxx = cen[o, 0] - x; dyy = cen[o, 1] - y; dzz = cen[o, 2] - z
                            rr = rad[o] + r - tol
                            if dxx * dxx + dyy * dyy + dzz * dzz < rr * rr:
                                return True
                        e = nxt[e]
    return False


@nb.njit(cache=True)
def place_h1(seed, glo, dims, head, nxt, sid, cen, rad, cnt, npos, nqu, nsp, dyad, h_off, h_stc, h_str, h_br, budget, tries,
             G, pores, pidx, PT, PQ, PC, mask):
    """H1 instance types (index into h_off groups), positions, quats; budget[k] copies per H1 type"""
    np.random.seed(seed)
    nt = budget.shape[0]
    rem = budget.copy()
    tot = 0
    for k in range(nt):
        tot += rem[k]
    nmax = min(tot, npos.shape[0])
    ht = np.empty(nmax, np.int32); hp = np.empty((nmax, 3), np.float32); hq = np.empty((nmax, 4), np.float32); hn = np.empty(nmax, np.int64)
    order = np.random.permutation(npos.shape[0])
    out = np.zeros(c3.NMEM); M = np.empty((3, 3)); Mn = np.empty((3, 3)); q = np.empty(4); qn = np.empty(4); t = np.empty(3)
    m = 0
    for oi in range(order.shape[0]):
        if m >= nmax:
            break
        i = order[oi]
        # choose H1 type proportional to remaining copies
        s = 0
        for k in range(nt):
            s += rem[k]
        if s <= 0:
            break
        x = np.random.random() * s; ty = nt - 1
        for k in range(nt):
            x -= rem[k]
            if x <= 0:
                ty = k; break
        a0 = h_off[ty]; a1 = h_off[ty + 1]
        stc = h_stc[a0:a1]; str_ = h_str[a0:a1]
        P = np.empty((a1 - a0, 3))
        for k in range(4):
            qn[k] = nqu[i, k]
        p3.quat_mat(qn, Mn)
        dw = np.empty(3)
        for k in range(3):
            dw[k] = Mn[k, 0] * dyad[0] + Mn[k, 1] * dyad[1] + Mn[k, 2] * dyad[2]
        ok = False
        for tr in range(tries):
            d = 6.0 + 3.0 * np.random.random()
            for k in range(3):
                t[k] = npos[i, k] + d * dw[k]
            p3.rand_quat(q); p3.quat_mat(q, M); p3.transform(stc, M, t, P)
            if _overlap_skip(glo, p3.H_GRID, dims, head, nxt, sid, cen, rad, P, str_, p3.TOL, i * nsp, (i + 1) * nsp):
                continue
            if not p3.check_volume(P, str_, t[0], t[1], t[2], h_br[ty], mask, G, pores, pidx, PT, PQ, PC, out):
                continue
            if not p3.g_insert(glo, p3.H_GRID, dims, head, nxt, sid, cen, rad, cnt, P, str_):
                break
            ok = True
            break
        if ok:
            ht[m] = ty; hn[m] = i
            for k in range(3):
                hp[m, k] = t[k]
            for k in range(4):
                hq[m, k] = q[k]
            rem[ty] -= 1; m += 1
    return ht[:m], hp[:m], hq[:m], hn[:m]


# ------------------------------------------------------------------ drivers
def nucleosome_models():
    z = np.load(f"{LIB}/NUCLEOSOME.npz"); b = z["beads"].astype(np.float64); k = z["kind"]
    c, r = sl.sphere_tree(b, per=300, pad=sl.BEAD_R)                       # coarse collision tree for the walk
    br = float((np.linalg.norm(c, axis=1) + r).max())
    d = b[k == 1]
    w, Vv = np.linalg.eigh(np.cov(d.T)); nrm = Vv[:, 0]
    jumps = np.linalg.norm(np.diff(d, axis=0), axis=1); cut = int(np.argmax(jumps))
    ends = np.array([d[0], d[cut], d[cut + 1], d[-1]]).mean(0)
    dy = ends - b.mean(0); dy -= nrm * dy.dot(nrm); dy /= np.linalg.norm(dy)
    return c, r, br, dy


def h1_models(I):
    import pandas as pd
    sub = I[I.ing_id.isin(H1_GENES)]
    ids, budget, C, R, BR, off = [], [], [], [], [], [0]
    for rr in sub.itertuples():
        f = f"{LIB}/{rr.ing_id}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f); c = z["st_c"].astype(np.float64); r = z["st_r"].astype(np.float64)
        ids.append(rr.ing_id); budget.append(int(rr.copies)); C.append(c); R.append(r); off.append(off[-1] + len(r))
        BR.append(float((np.linalg.norm(c, axis=1) + r).max()))
    return ids, np.array(budget, np.int64), np.array(off, np.int64), np.concatenate(C), np.concatenate(R), np.array(BR)


def run_region(cid, n_target, h1_budget, T, cell, models, seed, box=None, tries=30):
    """walk chromosome cid (territory id) for n_target nucleosomes, then H1; box = (lo, hi) clip (test block)"""
    TG = T["TG"]
    if box is not None:                                               # clip the territory grid to the test box
        lo_t, h = T["lo"], float(T["h"])
        gi = [lo_t[a] + (np.arange(TG.shape[a]) + 0.5) * h for a in range(3)]
        inb = ((gi[0][:, None, None] >= box[0][0]) & (gi[0][:, None, None] < box[1][0]) & (gi[1][None, :, None] >= box[0][1]) &
               (gi[1][None, :, None] < box[1][1]) & (gi[2][None, None, :] >= box[0][2]) & (gi[2][None, None, :] < box[1][2]))
        TG = np.where(inb, TG, -1).astype(np.int16)
    ijk = np.argwhere(TG == cid)
    if len(ijk) == 0 or n_target <= 0:
        return None
    tlo = np.asarray(T["lo"], np.float64); th = float(T["h"])
    starts = tlo + (ijk + 0.5) * th
    glo = starts.min(0) - 2 * th; ghi = starts.max(0) + 2 * th
    nc, nr, nbr, dyad = models["nuc"]
    nsp = len(nr)
    hids, hb, hoff, hC, hR, hBR = models["h1"]
    nh = int(min(h1_budget.sum(), n_target))
    cap_s = int((n_target * nsp + nh * 6) * 1.05) + 1000; cap_e = cap_s * 12
    mask = np.int64(1 << c3.NUCP)
    t0 = time.time()
    pos, qu, brk, dims, head, nxt, sid, cen, rad, cnt, overflow, restarts = walk(
        seed, glo, ghi, cell.G, cell.pores, cell.pidx, cell.PT, cell.PQ, cell.PC, nc, nr, nbr, mask,
        TG, tlo, th, np.int16(cid), starts, int(n_target), STEP, cap_s, cap_e)
    t1 = time.time()
    ht, hp, hq, hn = place_h1(seed + 7, glo, dims, head, nxt, sid, cen, rad, cnt, pos.astype(np.float64), qu.astype(np.float64), nsp,
                              dyad, hoff, hC, hR, hBR, h1_budget.astype(np.int64), tries,
                              cell.G, cell.pores, cell.pidx, cell.PT, cell.PQ, cell.PC, mask)
    return dict(pos=pos, quat=qu, brk=brk, h1_type=ht, h1_pos=hp, h1_quat=hq, h1_nuc=hn, overflow=overflow, restarts=restarts,
                t_walk=t1 - t0, t_h1=time.time() - t1, n_target=n_target, territory_voxels=len(ijk))


def common():
    import pandas as pd
    cell, centre = load_cell()
    T = dict(np.load(f"{OUT}/chromatin_territories.npz", allow_pickle=True))
    I = pd.read_csv(f"{D}/ingredients_v0.tsv", sep="\t")
    models = dict(nuc=nucleosome_models(), h1=h1_models(I))
    return cell, centre, T, models


def h1_share(models, frac):
    hb = models["h1"][1]
    return np.floor(hb * frac).astype(np.int64)


def block():
    cell, centre, T, models = common()
    z = np.load(f"{D}/packtest/block1000_seed1.npz"); lo = z["lo"]; size = float(z["size"]); box = (lo, lo + size)
    names, L, nn = [str(x) for x in T["names"]], None, T["n_nuc"]
    tot_vox = (T["TG_raw"] >= 0).sum()
    res = {}; t0 = time.time()
    for cid in range(len(names)):
        frac_box_terr = None
        # nucleosome target in the box: chromosome's density x its territory volume inside the box
        lo_t, h = T["lo"], float(T["h"])
        TG = T["TG_raw"]
        gi = [lo_t[a] + (np.arange(TG.shape[a]) + 0.5) * h for a in range(3)]
        ix = [(g >= box[0][a]) & (g < box[1][a]) for a, g in enumerate(gi)]
        sub = TG[np.ix_(*ix)]
        nv_box = int((sub == cid).sum()); nv = int((TG == cid).sum())
        if nv_box == 0:
            continue
        n_t = int(round(nn[cid] * nv_box / nv))
        hb = h1_share(models, n_t / N_NUC)
        r = run_region(cid, n_t, hb, T, cell, models, seed=100 + cid, box=box)
        if r is not None:
            res[cid] = r
            print(f"chr {names[cid]}: target {n_t}, placed {len(r['pos'])}, H1 {len(r['h1_pos'])}, fragments {int(r['brk'].sum())}, "
                  f"walk {r['t_walk']:.1f}s H1 {r['t_h1']:.1f}s", flush=True)
    save(res, models, names, f"{OUT}/chromatin_block", cell, box=box, t_total=time.time() - t0)


def chrom(cids):
    cell, centre, T, models = common()
    names = [str(x) for x in T["names"]]; nn = T["n_nuc"]
    for cid in cids:
        t0 = time.time()
        hb = h1_share(models, nn[cid] / N_NUC)
        r = run_region(cid, int(nn[cid]), hb, T, cell, models, seed=1000 + cid)
        np.savez(f"{PARTS}/chr_{cid:02d}.npz", **{k: v for k, v in r.items()})
        print(f"chr {names[cid]}: target {nn[cid]}, placed {len(r['pos'])} ({len(r['pos'])/nn[cid]:.1%}), H1 {len(r['h1_pos'])}, "
              f"fragments {int(r['brk'].sum())}, overflow {r['overflow']}, walk {r['t_walk']:.0f}s, H1 {r['t_h1']:.0f}s, total {time.time()-t0:.0f}s", flush=True)


def save(res, models, names, stem, cell, box=None, t_total=0.0):
    hids = models["h1"][0]
    lib_ids = ["NUCLEOSOME"] + hids
    pos, qu, ty, chain, brk = [], [], [], [], []
    for cid, r in sorted(res.items()):
        pos.append(r["pos"]); qu.append(r["quat"]); ty.append(np.zeros(len(r["pos"]), np.int32))
        chain.append(np.full(len(r["pos"]), cid, np.int32)); brk.append(r["brk"])
        pos.append(r["h1_pos"]); qu.append(r["h1_quat"]); ty.append(r["h1_type"].astype(np.int32) + 1)
        chain.append(np.full(len(r["h1_pos"]), cid, np.int32)); brk.append(np.zeros(len(r["h1_pos"]), np.int8))
    pos = np.concatenate(pos); qu = np.concatenate(qu); ty = np.concatenate(ty); chain = np.concatenate(chain); brk = np.concatenate(brk)
    np.savez_compressed(f"{stem}.npz", lib_ids=np.array(lib_ids), types=ty, pos=pos, quat=qu, chain=chain, brk=brk)
    n_nuc = int((ty == 0).sum()); n_h1 = int((ty > 0).sum())
    # QC: labels of the placed centres and density
    lab = cell.labels(pos[ty == 0].astype(np.float64)) if n_nuc else np.zeros(0)
    inside = float(np.mean(lab == c3.NUCP)) if n_nuc else float("nan")
    consumed = {"NUCLEOSOME": n_nuc}
    for k, h in enumerate(hids):
        consumed[h] = int((ty == k + 1).sum())
    meta = dict(n_nucleosomes=n_nuc, n_h1=n_h1, h1_per_nucleosome=round(n_h1 / max(n_nuc, 1), 3), centres_in_nucleoplasm=round(inside, 5),
                fragments=int(brk[ty == 0].sum()), consumed=consumed,
                per_chromosome={names[c]: dict(target=int(r["n_target"]), placed=int(len(r["pos"])), h1=int(len(r["h1_pos"])),
                                                fragments=int(r["brk"].sum()), walk_s=round(r["t_walk"], 1), h1_s=round(r["t_h1"], 1),
                                                overflow=bool(r["overflow"])) for c, r in res.items()},
                params=dict(step_nm=STEP, persistence=0.5, territory_grid_nm=TG_H, coarse_nucleosome_spheres=len(models["nuc"][1]),
                            h1_dyad_offset_nm=[6, 9], h1_tries=30), runtime_s=round(t_total, 1))
    if box is not None:
        bl, bh = box
        P = bl + np.random.default_rng(0).uniform(0, 1, (200000, 3)) * (bh - bl)
        vn = np.mean(cell.labels(P) == c3.NUCP) * np.prod(bh - bl) / 1e9
        m_pg = (n_nuc * (108.5e3 + 200 * 650) + n_h1 * 21.5e3) / 6.022e23 * 1e12
        meta.update(box_lo=bl.tolist(), box_hi=bh.tolist(), nucleoplasm_um3=round(vn, 4), chromatin_mg_per_ml=round(m_pg / vn * 1e3, 1),
                    nucleosomes_per_um3=round(n_nuc / vn), expected_nucleosomes_per_um3=round(N_NUC / 266.0))
    json.dump(meta, open(f"{stem}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "per_chromosome"}, indent=1))


def merge():
    cell, centre, T, models = common()
    names = [str(x) for x in T["names"]]
    res = {}
    for cid in range(len(names)):
        f = f"{PARTS}/chr_{cid:02d}.npz"
        if os.path.exists(f):
            res[cid] = {k: (v.item() if v.shape == () else v) for k, v in np.load(f).items()}
    save(res, models, names, f"{OUT}/chromatin_cell1", cell, t_total=sum(r["t_walk"] + r["t_h1"] for r in res.values()))
    qc_png(cell, T, f"{OUT}/chromatin_cell1.npz")


def qc_png(cell, T, fn, slab=60.0):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    z = np.load(fn); p = z["pos"]; ty = z["types"]; ch = z["chain"]
    m = ty == 0; p = p[m]; ch = ch[m]
    zc = np.median(p[:, 2]); s = np.abs(p[:, 2] - zc) < slab
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    ax[0].scatter(p[s, 0], p[s, 1], c=ch[s] % 20, cmap="tab20", s=0.3, lw=0)
    ax[0].set_title(f"nucleosome centres |z - {zc:.0f}| < {slab:.0f} nm, coloured by chromosome"); ax[0].set_aspect("equal")
    TG = T["TG_raw"]; k = int((zc - T["lo"][2]) / T["h"])
    ax[1].imshow(np.where(TG[:, :, k] >= 0, TG[:, :, k] % 20, np.nan).T, cmap="tab20", origin="lower",
                 extent=[T["lo"][0], T["lo"][0] + TG.shape[0] * T["h"], T["lo"][1], T["lo"][1] + TG.shape[1] * T["h"]])
    nl = T["nucleolus"][:, :, k]
    ax[1].contour(np.linspace(T["lo"][0], T["lo"][0] + TG.shape[0] * T["h"], TG.shape[0]), np.linspace(T["lo"][1], T["lo"][1] + TG.shape[1] * T["h"], TG.shape[1]),
                  nl.T.astype(float), [0.5], colors="k", linewidths=0.8)
    ax[1].set_title("territories (black: nucleoli)"); ax[1].set_aspect("equal")
    fig.tight_layout(); fig.savefig(f"{D}/qc/chromatin_cell1.png", dpi=70)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "setup":
        setup()
    elif cmd == "block":
        block()
    elif cmd == "chrom":
        chrom([int(x) for x in sys.argv[2:]])
    elif cmd == "merge":
        merge()
