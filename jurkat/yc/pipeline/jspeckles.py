"""Nuclear speckles for the Jurkat cell (pack v6, PLANNING_NOTES "v6 implementation", rule R6b).

There is no speckle label in the EM data, so speckles are procedural: N_SPK spheres of radius R_SPK placed in the
interchromatin space of the nucleoplasm (label NUCP), at the most chromatin-poor sites (nucleosome pre-pass density
smoothed over the sphere volume), >= D_EDGE from anything that is not nucleoplasm (NE, pores, nucleolus),
>= D_SEP apart (centre to centre), every sphere fully inside NUCP (100 surface probes).
The section n, n x (cx, cy, cz, r) is appended to G and G[cell3d.G_SPECKLE] points at it; cell3d labels SPK inside.
Literature: 20-50 speckles per mammalian nucleus, 0.5-2 um, in interchromatin regions (Spector & Lamond 2011, CSH Perspect
Biol 3:a000646). We use the small end (0.5 um) as the volume the packer can fill at a moderate density.

usage: NUMBA_NUM_THREADS=8 python jspeckles.py [n=30] [r_nm=250] [seed=5]
v7: JGEOM / JCHROM env select the geometry file (rewritten in place) and the chromatin pre-pass; a Golgi cisternae
section (jgolgi.py) after the speckle section is preserved. Backs up nothing itself (run after copying jgeom_cell1.npz to jgeom_cell1_v5.npz); idempotent: an existing speckle
section is replaced (G is rebuilt from the part before it)."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3

V = (__import__("os").environ.get("VCELL_ROOT", ".") + ""); D = f"{V}/data/derived"
N_SPK = int(sys.argv[1]) if len(sys.argv) > 1 else 30
R_SPK = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 5
D_EDGE, D_SEP, H = 350.0, 1000.0, 100.0
GEOM = os.environ.get("JGEOM", f"{D}/jgeom_cell1.npz")          # v7: JGEOM=data/derived/jgeom_cell1_v7.npz
CHROM = os.environ.get("JCHROM", f"{D}/prepass/chromatin_cell1.npz")   # v7: the genome-mapped chromatin


def main():
    t0 = time.time()
    z = dict(np.load(GEOM)); G = z["G"]; summ = json.loads(str(z["summary"]))
    gsec = None                                                # v7: a Golgi cisternae section appended after the speckles
    if G[c3.G_GOLGI_CIS] > 0 and G[c3.G_GOLGI_CIS] > G[c3.G_SPECKLE] > 0:
        go = int(G[c3.G_GOLGI_CIS]); gsec = G[go:].copy(); G = G[:go].copy(); G[c3.G_GOLGI_CIS] = 0.0
    if G[c3.G_SPECKLE] > 0:                                    # replace an earlier section (the last one appended)
        G = G[: int(G[c3.G_SPECKLE])].copy(); G[c3.G_SPECKLE] = 0.0
    cell = c3.Cell(G, z["pores"])
    ch = np.load(CHROM); ids = list(ch["lib_ids"])
    nuc = ch["pos"][ch["types"] == ids.index("NUCLEOSOME")].astype(np.float64)
    lo = nuc.min(0) - 500; hi = nuc.max(0) + 500; n = np.ceil((hi - lo) / H).astype(int)
    print(f"{len(nuc):,} nucleosomes; grid {n.tolist()} at {H:.0f} nm", flush=True)
    hist = np.histogramdd(nuc, bins=n, range=list(zip(lo, lo + n * H)))[0].astype(np.float32)
    from scipy.ndimage import uniform_filter
    k = int(round(2 * R_SPK / H)) | 1
    dens = uniform_filter(hist, size=k, mode="constant")       # nucleosomes per H^3 averaged over the sphere's cube
    g = [lo[i] + (np.arange(n[i]) + 0.5) * H for i in range(3)]
    X, Y, Zg = np.meshgrid(*g, indexing="ij"); P = np.stack([X.ravel(), Y.ravel(), Zg.ravel()], 1)
    lab = cell.labels(P)
    cand = np.nonzero(lab == c3.NUCP)[0]
    # clearance from everything that is not nucleoplasm (NE / perinuclear / pores, nucleolus, cytoplasm): exact EDT on the
    # 100 nm label grid (the stored SDFs are clamped at 64-160 nm, too short for this test)
    from scipy.ndimage import distance_transform_edt
    clr = distance_transform_edt((lab == c3.NUCP).reshape(tuple(n))).ravel() * H
    cand = cand[clr[cand] >= D_EDGE + R_SPK]
    d = dens.ravel()[cand]
    rng = np.random.default_rng(SEED)
    order = cand[np.lexsort((rng.random(len(cand)), d))]       # most chromatin-poor first, random tie-break
    print(f"{len(cand):,} candidate centres; nucleosome density (per {H:.0f} nm voxel, sphere-averaged) "
          f"min {d.min():.2f}, median {np.median(d):.2f}, all-nucleus median {np.median(dens.ravel()[lab == c3.NUCP]):.2f}", flush=True)
    u = rng.normal(size=(100, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    sel = []
    for i in order:
        c = P[i]
        if any(np.linalg.norm(c - P[j]) < D_SEP for j in sel): continue
        if not np.all(cell.labels(c + 0.98 * R_SPK * u) == c3.NUCP): continue
        sel.append(i)
        if len(sel) == N_SPK: break
    C = P[sel]
    sec = np.r_[float(len(sel)), np.c_[C, np.full(len(sel), R_SPK)].ravel()]
    G2 = np.r_[G, sec]; G2[c3.G_SPECKLE] = len(G)
    if gsec is not None:                                       # re-append the Golgi section with its offsets shifted
        import jgolgi
        o = len(G2); W, S = gsec[0], gsec[1]; head = gsec[2:12]; ni = int(gsec[13] - gsec[12])
        G2 = jgolgi.append_section(G2, head, gsec[14:14 + ni], gsec[14 + ni:], W, S)
    cell2 = c3.Cell(G2, z["pores"])
    # check: label volume by Monte Carlo inside the spheres and chromatin load
    u3 = rng.normal(size=(len(C), 400, 3)); q = (C[:, None, :] + R_SPK * np.cbrt(rng.random((len(C), 400, 1))) * u3 /
         np.linalg.norm(u3, axis=2, keepdims=True)).reshape(-1, 3)
    frac_spk = float((cell2.labels(q) == c3.SPK).mean())
    vol = len(sel) * 4 / 3 * np.pi * (R_SPK / 1000) ** 3
    inside = np.zeros(len(nuc), bool)
    from scipy.spatial import cKDTree
    tr = cKDTree(C); dd, _ = tr.query(nuc, k=1); inside = dd < R_SPK
    nuc_dens_spk = inside.sum() / vol
    nuc_dens_all = len(nuc) / (summ.get("nucleus_um3") or 254.5)
    S = dict(n=len(sel), r_nm=R_SPK, volume_um3=round(vol, 3), label_check_frac=round(frac_spk, 4), seed=SEED,
             d_edge_nm=D_EDGE, sep_nm=D_SEP, centres_nm=np.round(C, 1).tolist(),
             nucleosomes_inside=int(inside.sum()), nucleosome_density_inside_per_um3=round(float(nuc_dens_spk), 1),
             nucleosome_density_nucleus_per_um3=round(float(nuc_dens_all), 1),
             source="procedural (no EM label): most chromatin-poor NUCP sites; jspeckles.py")
    summ["speckles"] = S
    z["G"] = G2; z["summary"] = np.array(json.dumps(summ))
    tmp = GEOM + ".tmp.npz"; np.savez_compressed(tmp, **z); os.replace(tmp, GEOM)
    print(json.dumps({k: v for k, v in S.items() if k != "centres_nm"}, indent=1), f"\n{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
