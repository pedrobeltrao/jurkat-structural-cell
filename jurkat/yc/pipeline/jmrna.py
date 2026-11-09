"""mRNA pre-pass for the Jurkat cell (cell 1): every mRNA as a coarse-grained thread placed before tile packing.

Total mRNA per cell (assumption, documented in the json): N_MRNA = 250,000.
  - Jurkat 10x 3' scRNA-seq: median 10-15k UMI/cell (GSE263194 v3.1: 10.0k / 11.0k; 10x Jurkat v1: 14.6k); typical 10x
    3' molecule capture efficiency ~5-10% (no Jurkat smFISH counts in hand) -> ~1.5-3e5 mRNAs.
  - lower bound from the polysome pre-pass: 170k polysomes (one mRNA each).
  - literature range for mammalian cells of ~700-1000 um3: 1-3e5.
Per-gene copies: E6-1 consensus TPM (rna_e61.tsv, protein-coding, renormalised) x N_MRNA, integer by multinomial.
Lengths: RSEM effective length (DepMap 24Q4 Jurkat, expression-weighted over isoforms) + 350 nt (fragment-length
  correction ~250 nt + poly(A) ~100 nt); genes without a value: lognormal (median 2,500 nt, sigma 0.5).
Pools: nuclear NUC_FRAC = 14% (random genes, multinomial by abundance; nascent/nuclear-retained transcripts);
  polysomal = one mRNA per polysome of >= 2 ribosomes (polysomes_cell1.npz chains); longer mRNAs go to larger polysomes
  (rank matching on log-length with N(0, 0.35) jitter); rest = free cytosolic mRNPs. Monosomes (chain size 1) are
  treated as vacant 80S without mRNA (in mammalian cells ~10-20% of 80S are non-translating).
Representation: one bead per 5 nt, STEP = 3 nm apart (0.6 nm/nt contour), BEAD_R = 1.0 nm (ssRNA + bases).
  Polysomal threads: 5' tail -> through ribosome 1 (entry->exit straight segment, beads flagged inrib=1) -> linker ->
  ribosome 2 -> ... -> 3' tail. Ribosome order within a chain: greedy nearest-neighbour path from the chain end farthest
  from the chain centroid. Entry/exit = centre -/+ R_RIBO along the local chain direction (schematic: the 6QZP mRNA
  channel is not used; ribosome orientations in the polysome pre-pass are random). ORF footprint 30 nt per ribosome;
  remaining nt split 5' fraction U(0.1, 0.4) / 3'.
  Tails and free threads: persistent random walk (PERSIST) in the allowed compartment (cytosol; nucleoplasm for the
  nuclear pool), >= MEM_CLR nm from membranes, >= RIB_CLR nm from every ribosome centre; a thread stuck for MAX_STUCK
  tries ends early (truncation reported). Threads do not avoid each other, nucleosomes or the cytoskeleton (coarse;
  clashes are left to integration / the packer treats beads as obstacles).
usage: jmrna.py ; writes data/derived/prepass/mrna_cell1.npz/.json and data/derived/qc/mrna_cell1.png"""
import os, sys, json, time, numpy as np, pandas as pd, numba as nb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"
N_MRNA = 250_000
NUC_FRAC = 0.14
NT_PER_BEAD = 5
STEP = 3.0
BEAD_R = 1.0
R_RIBO = 13.0
RIB_CLR = 14.0
MEM_CLR = 1.5
PERSIST = 0.6
MAX_STUCK = 60
FOOT_NT = 30
H = 24.0


# ------------------------------------------------------------------ ribosome centre grid
@nb.njit(cache=True)
def build_grid(P, lo, dims):
    head = -np.ones(dims[0] * dims[1] * dims[2], np.int32); nxt = -np.ones(P.shape[0], np.int32)
    for i in range(P.shape[0]):
        a = int((P[i, 0] - lo[0]) / H); b = int((P[i, 1] - lo[1]) / H); c = int((P[i, 2] - lo[2]) / H)
        if a < 0 or b < 0 or c < 0 or a >= dims[0] or b >= dims[1] or c >= dims[2]:
            continue
        k = (a * dims[1] + b) * dims[2] + c
        nxt[i] = head[k]; head[k] = i
    return head, nxt


@nb.njit(cache=True)
def near_ribo(p, P, lo, dims, head, nxt, r):
    a0 = int((p[0] - lo[0]) / H); b0 = int((p[1] - lo[1]) / H); c0 = int((p[2] - lo[2]) / H)
    for a in range(max(a0 - 1, 0), min(a0 + 2, dims[0])):
        for b in range(max(b0 - 1, 0), min(b0 + 2, dims[1])):
            for c in range(max(c0 - 1, 0), min(c0 + 2, dims[2])):
                e = head[(a * dims[1] + b) * dims[2] + c]
                while e >= 0:
                    dx = P[e, 0] - p[0]; dy = P[e, 1] - p[1]; dz = P[e, 2] - p[2]
                    if dx * dx + dy * dy + dz * dz < r * r:
                        return True
                    e = nxt[e]
    return False


@nb.njit(cache=True)
def ok_bead(p, lab, G, pores, pidx, PT, PQ, PC, out, P, lo, dims, head, nxt):
    m = c3.mem_sdfs(p[0], p[1], p[2], G, pores, pidx, PT, PQ, PC, out)
    l = c3.label_from(p[0], p[1], p[2], G, pores, out, m)
    if lab == 0:
        if l != c3.CYTO:
            return False
    else:
        if l != c3.NUCP:
            return False
    if c3.min_mem_dist(out, -1) - c3.HALF < MEM_CLR + BEAD_R:
        return False
    return not near_ribo(p, P, lo, dims, head, nxt, RIB_CLR)


@nb.njit(cache=True)
def walk(pts, k0, n, start, d0, lab, G, pores, pidx, PT, PQ, PC, out, P, lo, dims, head, nxt):
    """persistent random walk of up to n beads starting one step from start along d0 (|d0| = 0: random direction);
    writes into pts[k0:]; returns number of beads written"""
    d = d0.copy()
    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] == 0.0:
        for a in range(3):
            d[a] = np.random.normal()
    dn = np.sqrt((d * d).sum()); d /= dn
    cur = start.copy(); t = np.empty(3); k = 0; stuck = 0
    while k < n and stuck < MAX_STUCK:
        for a in range(3):
            t[a] = d[a] * PERSIST + np.random.normal() * (1.0 - PERSIST) * 1.6
        tn = np.sqrt((t * t).sum())
        for a in range(3):
            t[a] = cur[a] + STEP * t[a] / tn
        if not ok_bead(t, lab, G, pores, pidx, PT, PQ, PC, out, P, lo, dims, head, nxt):
            stuck += 1
            continue
        for a in range(3):
            d[a] = (t[a] - cur[a]) / STEP; cur[a] = t[a]; pts[k0 + k, a] = t[a]
        k += 1; stuck = 0
    return k


@nb.njit(cache=True)
def random_start(lab, R, G, pores, pidx, PT, PQ, PC, out, P, lo, dims, head, nxt, s):
    for _ in range(20000):
        for a in range(3):
            s[a] = (np.random.random() * 2 - 1) * R
        if ok_bead(s, lab, G, pores, pidx, PT, PQ, PC, out, P, lo, dims, head, nxt):
            return True
    return False


@nb.njit(cache=True, parallel=True)
def thread_all(seed, kind, n5, n3, nlen, rstart, rcount, order, P, cap_off, G, pores, pidx, PT, PQ, PC, lo, dims, head, nxt, R):
    """kind 0: polysomal (ribosomes order[rstart:rstart+rcount], tails n5/n3 beads); kind 1/2: free cytosolic / nuclear
    thread of nlen beads. Writes into pts[cap_off[i]:cap_off[i+1]]; returns pts, inrib flags, beads used per thread."""
    m = kind.shape[0]
    pts = np.zeros((cap_off[m], 3), np.float32); inrib = np.zeros(cap_off[m], np.int8); used = np.zeros(m, np.int64)
    for i in nb.prange(m):
        np.random.seed(seed + i)
        out = np.zeros(c3.NMEM); k0 = cap_off[i]; k = 0
        tmp = np.empty((cap_off[i + 1] - cap_off[i] + 8, 3), np.float32)
        if kind[i] == 0:
            nr = rcount[i]
            ent = np.empty((nr, 3)); ext = np.empty((nr, 3))
            for j in range(nr):
                c = order[rstart[i] + j]
                if nr == 1:
                    dvec = np.array([1.0, 0.0, 0.0])
                else:
                    ja = order[rstart[i] + max(j - 1, 0)]; jb = order[rstart[i] + min(j + 1, nr - 1)]
                    dvec = P[jb] - P[ja]
                dn = np.sqrt((dvec * dvec).sum()) + 1e-9
                for a in range(3):
                    ent[j, a] = P[c, a] - R_RIBO * dvec[a] / dn; ext[j, a] = P[c, a] + R_RIBO * dvec[a] / dn
            # 5' tail grown outward from the first entry, then reversed so the thread runs 5'->3'
            d0 = ent[0] - ext[0]
            n5w = walk(tmp, 0, n5[i], ent[0], d0, 0, G, pores, pidx, PT, PQ, PC, out, P, lo, dims, head, nxt)
            for b in range(n5w):
                for a in range(3):
                    pts[k0 + k, a] = tmp[n5w - 1 - b, a]
                k += 1
            for j in range(nr):
                # through the ribosome (schematic straight channel)
                L = 2 * R_RIBO; nb_ = max(int(L / STEP), 1)
                for b in range(nb_ + 1):
                    f = b / nb_
                    for a in range(3):
                        pts[k0 + k, a] = ent[j, a] + f * (ext[j, a] - ent[j, a])
                    inrib[k0 + k] = 1; k += 1
                if j < nr - 1:
                    gv = ent[j + 1] - ext[j]; gl = np.sqrt((gv * gv).sum()); nl = int(gl / STEP)
                    for b in range(1, nl):
                        f = b / nl
                        for a in range(3):
                            pts[k0 + k, a] = ext[j, a] + f * gv[a]
                        k += 1
            d0 = ext[nr - 1] - ent[nr - 1]
            n3w = walk(pts, k0 + k, n3[i], ext[nr - 1], d0, 0, G, pores, pidx, PT, PQ, PC, out, P, lo, dims, head, nxt)
            k += n3w
        else:
            lab = 0 if kind[i] == 1 else 1
            s = np.empty(3)
            if random_start(lab, R, G, pores, pidx, PT, PQ, PC, out, P, lo, dims, head, nxt, s):
                for a in range(3):
                    pts[k0, a] = s[a]
                k = 1 + walk(pts, k0 + 1, nlen[i] - 1, s, np.zeros(3), lab, G, pores, pidx, PT, PQ, PC, out, P, lo, dims, head, nxt)
        used[i] = k
    return pts, inrib, used


@nb.njit(cache=True)
def chain_orders(P, chain_sorted_idx, starts, counts):
    """greedy nearest-neighbour path per chain, from the member farthest from the chain centroid"""
    order = np.empty(chain_sorted_idx.shape[0], np.int64)
    for c in range(starts.shape[0]):
        s, n = starts[c], counts[c]
        idx = chain_sorted_idx[s:s + n]
        if n == 1:
            order[s] = idx[0]; continue
        cen = np.zeros(3)
        for j in range(n):
            cen += P[idx[j]]
        cen /= n
        used = np.zeros(n, np.bool_); best = 0; bd = -1.0
        for j in range(n):
            d = ((P[idx[j]] - cen) ** 2).sum()
            if d > bd:
                bd = d; best = j
        cur = best
        for t in range(n):
            order[s + t] = idx[cur]; used[cur] = True
            nb_ = -1; nd = 1e30
            for j in range(n):
                if not used[j]:
                    d = ((P[idx[j]] - P[idx[cur]]) ** 2).sum()
                    if d < nd:
                        nd = d; nb_ = j
            if nb_ < 0:
                break
            cur = nb_
    return order


def main(seed=29):
    t0 = time.time(); rng = np.random.default_rng(seed)
    z = np.load(f"{D}/jgeom_cell1.npz"); cell = c3.Cell(z["G"], z["pores"]); R = float(cell.G[0])
    pz = np.load(f"{D}/prepass/polysomes_cell1.npz"); P = pz["pos"].astype(np.float64); chain = pz["chain"].astype(np.int64)
    # chains
    srt = np.argsort(chain, kind="stable"); ch_s = chain[srt]
    uniq, starts, counts = np.unique(ch_s, return_index=True, return_counts=True)
    order = chain_orders(P, srt.astype(np.int64), starts.astype(np.int64), counts.astype(np.int64))
    poly = counts >= 2
    p_start, p_count = starts[poly], counts[poly]
    n_poly = int(poly.sum()); n_mono = int((~poly).sum())
    print(f"polysomes {n_poly:,} (ribosomes {p_count.sum():,}), monosomes {n_mono:,}; chain order {time.time() - t0:.0f}s", flush=True)
    # genes, copies, lengths
    rna = pd.read_csv(f"{D}/rna_e61.tsv", sep="\t", index_col="gene")
    tpm = rna.tpm_median.clip(lower=0); tpm = tpm[tpm > 0]
    el = pd.read_csv(f"{V}/data/transcriptomics/depmap_24Q4/jurkat/Jurkat_AllGenes_effective_length.tsv", sep="\t")
    el["sym"] = el.gene.str.split(" ").str[0]; el = el.groupby("sym").effective_length.max()
    genes = tpm.index.values
    L = el.reindex(genes).values + 350.0
    miss = ~np.isfinite(L) | (L < 400)
    L[miss] = np.exp(rng.normal(np.log(2500), 0.5, miss.sum()))
    w = tpm.values / tpm.values.sum()
    copies = rng.multinomial(N_MRNA, w)
    gidx = np.repeat(np.arange(len(genes)), copies); rng.shuffle(gidx)
    nt = np.round(L[gidx]).astype(np.int64)
    n_nuc = int(round(NUC_FRAC * N_MRNA)); n_rest = N_MRNA - n_nuc
    n_polym = min(n_poly, n_rest)
    nuc_sel = np.arange(n_nuc); rest = np.arange(n_nuc, N_MRNA)
    # longer mRNAs -> larger polysomes (rank matching with jitter)
    key = np.log(nt[rest]) + rng.normal(0, 0.35, len(rest))
    rest_sorted = rest[np.argsort(-key)]
    poly_mrna = rest_sorted[:n_polym]; free_mrna = rest_sorted[n_polym:]
    pkey = np.log(p_count) + rng.normal(0, 0.35, n_poly)
    poly_rank = np.argsort(-pkey)[:n_polym]
    # build thread table
    kind = np.r_[np.zeros(n_polym, np.int64), np.ones(len(free_mrna), np.int64), np.full(n_nuc, 2, np.int64)]
    mids = np.r_[poly_mrna, free_mrna, nuc_sel]
    ntv = nt[mids]
    rstart = np.zeros(len(kind), np.int64); rcount = np.zeros(len(kind), np.int64); poly_id = -np.ones(len(kind), np.int64)
    rstart[:n_polym] = p_start[poly_rank]; rcount[:n_polym] = p_count[poly_rank]; poly_id[:n_polym] = uniq[poly][poly_rank]
    tot_beads = np.maximum(ntv // NT_PER_BEAD, 2)
    rem = np.maximum(ntv - FOOT_NT * rcount, 0) // NT_PER_BEAD
    f5 = rng.uniform(0.1, 0.4, len(kind))
    n5 = np.where(kind == 0, (rem * f5).astype(np.int64), 0); n3 = np.where(kind == 0, rem - n5, 0)
    nlen = np.where(kind == 0, 0, tot_beads)
    # capacity per thread: tails + ribosome channels (9 beads) + linkers (<= ~22 beads)
    cap = np.where(kind == 0, n5 + n3 + rcount * 40 + 8, nlen + 2)
    cap_off = np.r_[0, np.cumsum(cap)].astype(np.int64)
    est_gb = cap_off[-1] * 13 / 1e9
    print(f"threads {len(kind):,} (polysomal {n_polym:,}, free {len(free_mrna):,}, nuclear {n_nuc:,}); mean length {ntv.mean():.0f} nt; "
          f"bead capacity {cap_off[-1]:,} (~{est_gb:.1f} GB)", flush=True)
    lo = -np.full(3, R); dims = np.array([int(2 * R / H) + 1] * 3, np.int64)
    head, nxt = build_grid(P, lo, dims)
    t1 = time.time()
    pts, inrib, used = thread_all(seed, kind, n5, n3, nlen, rstart, rcount, order, P, cap_off, cell.G, cell.pores, cell.pidx,
                                  cell.PT, cell.PQ, cell.PC, lo, dims, head, nxt, R)
    print(f"threaded in {time.time() - t1:.0f}s", flush=True)
    # compact
    keep = np.zeros(len(pts), bool)
    for i in range(len(kind)):
        keep[cap_off[i]:cap_off[i] + used[i]] = True
    pts = pts[keep]; inrib = inrib[keep]; off = np.r_[0, np.cumsum(used)].astype(np.int64)
    want = np.where(kind == 0, n5 + n3, nlen)
    got = np.where(kind == 0, used - (rcount * (int(2 * R_RIBO / STEP) + 1)), used)
    trunc = float(np.clip(1 - got.sum() / max(want.sum(), 1), 0, 1))
    empty = int((used == 0).sum())
    os.makedirs(f"{D}/prepass", exist_ok=True)
    np.savez_compressed(f"{D}/prepass/mrna_cell1.npz", pts=pts, off=off, inrib=inrib, gene=genes[gidx[mids]].astype("U16"),
                        kind=kind.astype(np.int8), poly_id=poly_id, nt=ntv, bead_radius=np.float32(BEAD_R))
    info = dict(total_mrna=N_MRNA, polysomal=int(n_polym), free_cytosolic=int(len(free_mrna)), nuclear=n_nuc,
                monosomes_without_mrna=n_mono, threads_empty=empty, beads=int(len(pts)), beads_in_ribosomes=int(inrib.sum()),
                bead_radius_nm=BEAD_R, step_nm=STEP, nt_per_bead=NT_PER_BEAD, mean_length_nt=round(float(ntv.mean())),
                total_nt=int(ntv.sum()), tail_truncation_frac=round(trunc, 4), memory_pts_gb=round(pts.nbytes / 1e9, 2),
                genes_expressed=int((copies > 0).sum()),
                top_genes=pd.Series(copies, index=genes).sort_values(ascending=False).head(10).to_dict(),
                runtime_s=round(time.time() - t0),
                assumptions=dict(N_MRNA="250k: 10x median 10-15k UMI / ~5-10% capture; >= 170k polysomes; literature 1-3e5",
                                 lengths="DepMap RSEM effective length + 350 nt; lognormal median 2.5 kb if missing",
                                 nuclear_frac=NUC_FRAC, monosomes="vacant 80S (no mRNA)",
                                 polysome_assignment="rank match log length vs log polysome size, N(0,0.35) jitter",
                                 path="schematic: straight channel entry->exit along chain direction (6QZP mRNA channel not used)",
                                 collisions="threads avoid membranes and ribosome centres only (not each other, nucleosomes, cytoskeleton)"))
    json.dump(info, open(f"{D}/prepass/mrna_cell1.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k not in ("assumptions", "top_genes")}, indent=1), flush=True)
    # QC figure: 300 nm slab through the cell centre
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sl = np.abs(pts[:, 2]) < 150
    kb = np.repeat(kind, used)
    for a_, (nm, ks) in zip(ax, (("polysomal (blue) / free (orange) / nuclear (green)", (0, 1, 2)), ("zoom 2 um", (0, 1, 2)))):
        for kk, col in zip(ks, ("tab:blue", "tab:orange", "tab:green")):
            m = sl & (kb == kk)
            a_.scatter(pts[m, 0], pts[m, 1], s=0.02, c=col, alpha=0.4, rasterized=True)
        rs = np.abs(P[:, 2]) < 150
        a_.scatter(P[rs, 0], P[rs, 1], s=0.05, c="k", alpha=0.3, rasterized=True)
        a_.set_aspect("equal"); a_.set_title(nm)
    ax[1].set_xlim(-1000, 1000); ax[1].set_ylim(-1000, 1000)
    os.makedirs(f"{D}/qc", exist_ok=True); fig.tight_layout(); fig.savefig(f"{D}/qc/mrna_cell1.png", dpi=90)


if __name__ == "__main__":
    main()
