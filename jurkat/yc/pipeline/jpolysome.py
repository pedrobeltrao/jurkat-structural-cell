"""Polysome pre-pass for the Jurkat cell (PLANNING_NOTES §4-results DECISION): place the EM-segmented ribosomes of
cell 1 at their measured positions, then add ribosomes as polysome neighbours of EM seeds until the inventory count of
80S ribosomes (ingredients RIBO80S) is reached.

  EM ribosomes: data/derived/ribo_seg_instances.parquet (centroids in s0 voxels of the LABEL frame, class 1 cytosolic,
    2 nuclear, 3 ER sheet, 4 ER tube; cell2 = cell id). Model frame nm = (x*4, y*4, z*3.44) - centre_nm (jgeom_cell1.npz).
    Cytosolic / ER classes are kept if the centre lies in cytosol >= R_EXCL from every membrane (else nudged up to
    NUDGE nm along the membrane-normal direction); nuclear-class ribosomes (likely pre-ribosomes / misclassified) are
    not placed as 80S (reported).
  Polysome nuclei: EM ribosomes linked within LINK nm (single linkage; fragments of one polysome at EM recall ~0.4).
  Added ribosomes: grown from members of nuclei of >= 2 ribosomes (singletons: LAM_SINGLE, mostly monosomes) so that the
    total reaches the target; the next member sits D_MIN..D_MAX nm from the previous one inside a cone (CONE rad) around the
    chain direction; cytosolic chains stay in cytosol >= R_EXCL from membranes; ER-seeded chains stay ER_BAND nm from
    the ER membrane (membrane-bound polysomes). Ribosome-ribosome exclusion by centre distance >= D_EXCL (coarse;
    exact sphere-tree collisions are resolved when tiles are packed, with these ribosomes as fixed obstacles).
  Orientations: random (v0; the mRNA channel / exit-tunnel orientation toward the ER translocon is not modelled yet).
Output: data/derived/prepass/polysomes_cell1.npz (lib_ids, types, pos, quat, origin 0=EM 1=added, er 0/1, chain id)
and polysomes_cell1.json."""
import os, sys, json, time, numpy as np, pandas as pd, numba as nb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"
R_EXCL = 14.0                 # nm, ribosome centre clearance to membranes (80S radius ~13-14 nm)
D_EXCL = 22.0                 # nm, min centre distance between ribosomes (coarse; EM centroids of touching ribosomes ~22-28 nm)
D_MIN, D_MAX = 23.0, 29.0     # nm, centre spacing along a polysome (v3: tighter, from the EM NN comparison, validate/ribosome_nn.py)
CONE = 1.0                    # v3: more curvature -> compact spiral-like chains
ER_BAND = (12.0, 28.0)        # nm, centre distance to the ER membrane for ER-bound ribosomes
NUDGE = 10.0
LINK = 34.0                   # nm, single-linkage distance grouping EM ribosomes into polysome nuclei
LAM_SINGLE = 0.3              # mean extensions for EM singletons (most stay monosomes)
H = 24.0                      # grid cell (>= D_EXCL)


@nb.njit(cache=True)
def _cell_of(p, lo, dims):
    i = int((p[0] - lo[0]) / H); j = int((p[1] - lo[1]) / H); k = int((p[2] - lo[2]) / H)
    if i < 0 or j < 0 or k < 0 or i >= dims[0] or j >= dims[1] or k >= dims[2]:
        return -1
    return (i * dims[1] + j) * dims[2] + k


@nb.njit(cache=True)
def _free(p, lo, dims, head, nxt, P):
    i0 = int((p[0] - lo[0]) / H); j0 = int((p[1] - lo[1]) / H); k0 = int((p[2] - lo[2]) / H)
    for i in range(max(i0 - 1, 0), min(i0 + 2, dims[0])):
        for j in range(max(j0 - 1, 0), min(j0 + 2, dims[1])):
            for k in range(max(k0 - 1, 0), min(k0 + 2, dims[2])):
                e = head[(i * dims[1] + j) * dims[2] + k]
                while e >= 0:
                    dx = P[e, 0] - p[0]; dy = P[e, 1] - p[1]; dz = P[e, 2] - p[2]
                    if dx * dx + dy * dy + dz * dz < D_EXCL * D_EXCL:
                        return False
                    e = nxt[e]
    return True


@nb.njit(cache=True)
def _ok_site(p, er, G, pores, pidx, PT, PQ, PC, out):
    m = c3.mem_sdfs(p[0], p[1], p[2], G, pores, pidx, PT, PQ, PC, out)
    if c3.label_from(p[0], p[1], p[2], G, pores, out, m) != c3.CYTO:
        return False
    if er:
        d = out[c3.M_ER]
        if d < ER_BAND[0] or d > ER_BAND[1]:
            return False
        for mm in range(c3.NMEM):
            if mm != c3.M_ER and mm != c3.M_OUTER and abs(out[mm]) < R_EXCL:
                return False
        return True
    return c3.min_mem_dist(out, -1) - c3.HALF >= R_EXCL


@nb.njit(cache=True)
def grow(seed, P0, er0, extra, clus, lo, dims, G, pores, pidx, PT, PQ, PC, cap):
    """insert EM ribosomes P0 (already validated), then grow extra[i] members from each seed i.
    Returns positions, origin, er flag, chain id, number placed."""
    np.random.seed(seed)
    P = np.empty((cap, 3)); orig = np.zeros(cap, np.int8); erf = np.zeros(cap, np.int8); chain = np.empty(cap, np.int32)
    emi = np.empty(cap, np.int64)
    head = -np.ones(dims[0] * dims[1] * dims[2], np.int64); nxt = -np.ones(cap, np.int64)
    out = np.zeros(c3.NMEM); n = 0
    for i in range(P0.shape[0]):
        if not _free(P0[i], lo, dims, head, nxt, P):
            continue
        c = _cell_of(P0[i], lo, dims)
        if c < 0:
            continue
        P[n] = P0[i]; orig[n] = 0; erf[n] = er0[i]; chain[n] = clus[i]; emi[n] = i
        nxt[n] = head[c]; head[c] = n; n += 1
    nseed = n
    t = np.empty(3); u = np.empty(3)
    for s in range(nseed):
        k = extra[emi[s]]
        cur = P[s].copy()
        u[0] = np.random.normal(); u[1] = np.random.normal(); u[2] = np.random.normal()
        un = np.sqrt((u * u).sum()); u /= un
        for m in range(k):
            placed = False
            for tr in range(60):
                v = np.empty(3)
                v[0] = np.random.normal(); v[1] = np.random.normal(); v[2] = np.random.normal()
                v /= np.sqrt((v * v).sum())
                c_ = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
                if c_ < np.cos(CONE):                       # rejection-sample a direction inside the cone
                    if tr < 50:
                        continue
                d = D_MIN + (D_MAX - D_MIN) * np.random.random()
                for a in range(3):
                    t[a] = cur[a] + d * v[a]
                if not _ok_site(t, erf[s], G, pores, pidx, PT, PQ, PC, out):
                    continue
                if not _free(t, lo, dims, head, nxt, P):
                    continue
                c = _cell_of(t, lo, dims)
                if c < 0 or n >= cap:
                    continue
                P[n] = t; orig[n] = 1; erf[n] = erf[s]; chain[n] = chain[s]; emi[n] = emi[s]
                nxt[n] = head[c]; head[c] = n; n += 1
                for a in range(3):
                    u[a] = v[a]; cur[a] = t[a]
                placed = True
                break
            if not placed:
                break
    return P[:n], orig[:n], erf[:n], chain[:n], nseed


def main(seed=17):
    t0 = time.time()
    z = np.load(f"{D}/jgeom_cell1.npz"); cell = c3.Cell(z["G"], z["pores"]); centre = z["centre_nm"]
    I = pd.read_csv(f"{D}/ingredients_v0.tsv", sep="\t").set_index("ing_id")
    target = int(I.copies["RIBO80S"])
    r = pd.read_parquet(f"{D}/ribo_seg_instances.parquet"); r = r[r.cell2 == 1]
    n_nuc = int((r.cls == 2).sum()); r = r[r.cls != 2]
    P0 = r[["x", "y", "z"]].values * np.array([4.0, 4.0, 3.44]) - centre
    er0 = r.cls.isin([3, 4]).values.astype(np.int8)
    # validate / nudge EM sites
    M = cell.mems(P0); L = cell.labels(P0)
    def site_ok(M, L, er):
        """cytosolic: >= R_EXCL from every membrane; ER-bound: ER membrane within ER_BAND, others >= R_EXCL"""
        others = np.abs(M[:, :c3.NMEM - 1]) - c3.MEM_HALF[:c3.NMEM - 1]
        o_noer = np.delete(others, c3.M_ER, axis=1).min(1)
        ok_c = (L == c3.CYTO) & (others.min(1) >= R_EXCL)
        ok_e = (L == c3.CYTO) & (M[:, c3.M_ER] >= ER_BAND[0]) & (M[:, c3.M_ER] <= ER_BAND[1]) & (o_noer >= R_EXCL)
        return np.where(er == 1, ok_e, ok_c)
    ok = site_ok(M, L, er0)
    bad = np.where(~ok)[0]
    rng = np.random.default_rng(seed); fixed = 0
    for i in bad:                                           # nudge: random small displacements, keep the first valid
        for _ in range(12):
            q = P0[i] + rng.normal(0, NUDGE / 2, 3)
            if np.linalg.norm(q - P0[i]) > NUDGE: continue
            if site_ok(cell.mems(q[None]), cell.labels(q[None]), er0[i:i + 1])[0]:
                P0[i] = q; ok[i] = True; fixed += 1; break
    P0v, er0v = P0[ok], er0[ok]
    print(f"EM ribosomes cell 1: {len(r) + n_nuc:,} (nuclear-class skipped {n_nuc:,}); valid as-is {int(ok.sum()) - fixed:,}, "
          f"nudged {fixed:,}, dropped {int((~ok).sum()):,}; target RIBO80S {target:,}", flush=True)
    n_add = max(target - len(P0v), 0)
    # polysome nuclei: EM ribosomes linked within LINK nm (single linkage) = fragments of one polysome (EM recall ~35-50%)
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pr = cKDTree(P0v).query_pairs(LINK, output_type="ndarray")
    ncl, clus = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(P0v),) * 2), directed=False)
    csz = np.bincount(clus)[clus]
    # additions: singletons mostly stay monosomes (LAM_SINGLE), members of groups >= 2 share the rest
    n_single = int((csz == 1).sum()); lam_c = max(n_add - LAM_SINGLE * n_single, 0) / max(len(P0v) - n_single, 1)
    lam_i = np.where(csz == 1, LAM_SINGLE, lam_c)
    extra = rng.poisson(lam_i * 1.45).astype(np.int64)              # 45% overshoot: extensions fail in crowded cytosol
    lam = n_add / max(len(P0v), 1)
    print(f"EM polysome nuclei (link {LINK} nm): {ncl:,} groups, singletons {n_single:,}; extension rate per member {lam_c:.2f}", flush=True)
    lo = np.array([-z["R"]] * 3, float) if "R" in z.files else -np.full(3, cell.G[0])
    R = float(cell.G[0]); lo = -np.full(3, R); dims = np.array([int(2 * R / H) + 1] * 3, np.int64)
    print(f"grid {dims.tolist()} ({np.prod(dims) * 8 / 1e9:.1f} GB head); growing ~{n_add:,} added ribosomes (mean {lam:.2f} per seed)", flush=True)
    P, orig, erf, chain, nseed = grow(seed, P0v, er0v, extra, clus.astype(np.int32), lo, dims, cell.G, cell.pores, cell.pidx, cell.PT, cell.PQ, cell.PC,
                                      int(len(P0v) + extra.sum() + 10))
    # trim overshoot to the target (drop the last-added members first)
    if len(P) > target:
        keep = np.ones(len(P), bool); add_idx = np.where(orig == 1)[0]
        keep[add_idx[-(len(P) - target):]] = False
        P, orig, erf, chain = P[keep], orig[keep], erf[keep], chain[keep]
    q = rng.normal(size=(len(P), 4)); q /= np.linalg.norm(q, axis=1, keepdims=True); q[q[:, 0] < 0] *= -1
    os.makedirs(f"{D}/prepass", exist_ok=True)
    np.savez_compressed(f"{D}/prepass/polysomes_cell1.npz", lib_ids=np.array(["RIBO80S"]), types=np.zeros(len(P), np.int32),
                        pos=P.astype(np.float32), quat=q.astype(np.float32), origin=orig, er=erf, chain=chain)
    sizes = pd.Series(chain).value_counts()
    info = dict(target=target, placed=int(len(P)), em_placed=int((orig == 0).sum()), added=int((orig == 1).sum()),
                er_bound=int(erf.sum()), er_bound_frac=round(float(erf.mean()), 3), nuclear_class_skipped=n_nuc,
                em_dropped=int((~ok).sum()), mean_polysome_size=round(float(sizes.mean()), 2), max_polysome_size=int(sizes.max()),
                consumed={"RIBO80S": int(len(P))}, runtime_s=round(time.time() - t0),
                params=dict(R_EXCL=R_EXCL, D_EXCL=D_EXCL, D_MIN=D_MIN, D_MAX=D_MAX, CONE=CONE, ER_BAND=ER_BAND, NUDGE=NUDGE),
                n_polysomes=int((pd.Series(chain).value_counts() >= 2).sum()), n_monosomes=int((pd.Series(chain).value_counts() == 1).sum()),
                caveats="random orientations; coarse centre-distance exclusion; chain = EM linkage group + extensions")
    json.dump(info, open(f"{D}/prepass/polysomes_cell1.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
