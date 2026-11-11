"""Juxta-centrosomal procedural Golgi (v7; DECISION Pedro 2026-11-10: centrioles stay at the EM-derived MTOC candidate and
the Golgi is moved to surround the centrosome; this also removes the v6 Golgi/mito/ER overlap of the smoothed EM blob).

The EM Golgi blob (label 5, 4.2 um3) sat 6.4 um from the MTOC candidate and overlapped mitochondria and ER after
smoothing. Here the Golgi occupancy is rebuilt as a shell around the centriole pair: the free cytoplasm (cytosol label of
the geometry without Golgi, >= CLR nm from every other membrane, >= PM_CLR below the PM (cortex), >= MT_CLR from an MT
axis, >= C_CLR from a centriole dimer, outside the PCM sphere R_IN) is filled nearest-first from the centrosome until the
EM blob volume is reached. The occupancy is smoothed and turned into an SDF exactly as jgeom_build does (mask_sdf, 16 nm
grid, sigma 16 nm); its deep SDF feeds the cisternae section (jgolgi.py layout, cell3d G_GOLGI_CIS). The inventory area
definition is unchanged (GOLGI_STACK_F x EM blob area; jwc uses inventory / measured).
Layout of G after this script: fields + cristae, then the new Golgi blob bricks (header of field F_GOLGI re-pointed; the
old EM bricks stay as dead data), then the speckle section, then the cisternae section (so jspeckles keeps both).
Pre-pass content inside the new Golgi (prepass_v7) is removed: MT dimers whose centre is < MT_DROP from the Golgi blob
surface or inside it, F-actin subunits < 4 nm, ribosomes < 15 nm, mRNA beads < 3 nm (appended to mrna_drop.npy).
usage: NUMBA_NUM_THREADS=8 python jgolgi_ribbon.py   (input jgeom_cell1_v7.npz, backup jgeom_cell1_v7_emgolgi.npz)"""
import os, sys, json, time, shutil
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3
import jgeom_build as jb
import jgolgi

V = (__import__("os").environ.get("VCELL_ROOT", ".") + ""); D = f"{V}/data/derived"; P7 = f"{D}/prepass_v7"
GEO = f"{D}/jgeom_cell1_v7.npz"; BAK = f"{D}/jgeom_cell1_v7_emgolgi.npz"
H = jb.H
BOX = 3600.0           # nm half-size of the search box around the centrosome
CLR = 40.0             # nm from every other membrane surface
PM_CLR = 150.0         # nm below the PM (cortical F-actin band 8-150 nm; the cell field is clamped at 160 inside)
MT_CLR = 48.0          # nm from an MT axis (stage-3 clearance 20 nm + smoothing + 16 nm voxels)
C_CLR = 60.0           # nm from a centriole dimer
R_IN = 450.0           # nm: PCM sphere (jvirtual.R_PCM 400) + margin
MT_DROP = 8.0          # nm: MT dimer centre (on the wall, ~12 nm off axis) closer than this to the Golgi -> axis < ~20 nm


def no_golgi(G):
    G = G.copy(); o = c3.G_N + 1 + 12 * c3.F_GOLGI
    G[o + 4:o + 7] = 0.0; G[c3.G_GOLGI_CIS] = 0.0           # field returns its outside clamp everywhere
    return G


def main():
    t0 = time.time()
    if not os.path.exists(BAK): shutil.copy(GEO, BAK)
    z = dict(np.load(BAK)); G = z["G"].copy(); summ = json.loads(str(z["summary"]))
    J = json.load(open(f"{P7}/cytoskeleton_cell1.json")); cen = np.array(J["log"]["centrioles"]["new_centre_nm"])
    cell0 = c3.Cell(no_golgi(G), z["pores"])
    # grid aligned with the jgeom_build Golgi grid
    oh = c3.G_N + 1 + 12 * c3.F_GOLGI; org = G[oh:oh + 3]
    i0 = np.floor((cen - BOX - org) / H).astype(int); n = int(np.ceil(2 * BOX / H)) + 1
    lo = org + i0 * H
    ax = [lo[i] + np.arange(n) * H for i in range(3)]
    X = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    print(f"grid {n}^3 = {len(X):,} voxels at {H} nm around {cen.round(0).tolist()}", flush=True)
    free = np.zeros(len(X), bool)
    for s in range(0, len(X), 5_000_000):
        P = X[s:s + 5_000_000]; lab = cell0.labels(P); S = cell0.mems(P)
        oth = np.min(np.abs(S[:, [c3.M_NE, c3.M_ER, c3.M_OM, c3.M_IM, c3.M_ENDO]]) - c3.MEM_HALF[[c3.M_NE, c3.M_ER, c3.M_OM, c3.M_IM, c3.M_ENDO]], axis=1)
        free[s:s + len(P)] = (lab == c3.CYTO) & (oth >= CLR) & (S[:, c3.M_PM] >= PM_CLR) & (S[:, c3.M_OUTER] >= PM_CLR)
    free = free.reshape(n, n, n)
    print(f"free cytoplasm {free.sum() * H ** 3 / 1e9:.2f} um3 ({time.time() - t0:.0f}s)", flush=True)
    # MT axes and centriole dimers rasterised; Euclidean distance on the grid
    cl = np.load(f"{D}/cytoskel/stage3/mt_stage3_b_centerlines.npz"); mp = cl["points_model_nm"].astype(np.float64)
    cz = np.load(f"{P7}/cytoskeleton_cell1.npz"); cp = cz["pos"][cz["kind"] == 0].astype(np.float64)
    for pts, clr_, nm in ((mp, MT_CLR, "MT"), (cp, C_CLR, "centriole")):
        ij = np.round((pts - lo) / H).astype(int); ok = np.all((ij >= 0) & (ij < n), axis=1)
        occ = np.zeros((n, n, n), bool); occ[tuple(ij[ok].T)] = True
        d = ndi.distance_transform_edt(~occ) * H
        free &= d >= clr_ + 0.5 * H * np.sqrt(3)             # voxel quantisation margin
        print(f"  after {nm} clearance: {free.sum() * H ** 3 / 1e9:.2f} um3", flush=True)
    dc = np.linalg.norm(X - cen, axis=1).reshape(n, n, n)
    free &= dc >= R_IN
    vt = float(summ["golgi"].get("vol_sdf_um3") or summ["golgi"]["vol_mask_um3"])
    idx = np.nonzero(free.ravel())[0]; o = idx[np.argsort(dc.ravel()[idx], kind="stable")]
    k = int(round(vt * 1e9 / H ** 3 * 1.04))              # +4%: smoothing erodes the surface slightly
    sel = o[:k]; occ = np.zeros(n ** 3, bool); occ[sel] = True; occ = occ.reshape(n, n, n)
    r_max = float(dc.ravel()[sel].max())
    print(f"target {vt:.2f} um3: {k:,} voxels, radius {R_IN:.0f}-{r_max:.0f} nm from the centrosome ({time.time() - t0:.0f}s)", flush=True)
    # SDF as jgeom_build (exact near the surface), blob bricks and deep bricks
    sdf, area = jb.mask_sdf(occ, lo, jb.SPEC["golgi"][1])
    # mask_sdf pads by one voxel and returns the grid of occ: same origin lo
    bh, bi, bd = jb.bricks(sdf, lo, jb.SPEC["golgi"][2])
    dh, di, dd = jb.bricks(sdf, lo, (jgolgi.DEEP_IN, 64.0))
    # rebuild G: [fields + cristae] + new Golgi bricks + speckles + cisternae
    G0 = jgolgi.strip(G)                                  # without the old cisternae section
    so = int(G0[c3.G_SPECKLE]); ns = int(G0[so]); spk = G0[so:so + 1 + 4 * ns].copy()
    Gb = G0[:so].copy()
    io = len(Gb); Gb = np.r_[Gb, bi]; do = len(Gb); Gb = np.r_[Gb, bd]
    Gb[oh:oh + 10] = bh; Gb[oh + 10] = io; Gb[oh + 11] = do
    Gb[c3.G_SPECKLE] = len(Gb); Gb = np.r_[Gb, spk]
    G2 = jgolgi.append_section(Gb, dh, di, dd)
    cell = c3.Cell(G2, z["pores"])
    # ---------------------------------------------------------------- checks
    rng = np.random.default_rng(4)
    Pb = X[sel] + rng.uniform(-H / 2, H / 2, (len(sel), 3)); Pb = Pb[rng.choice(len(Pb), min(400000, len(Pb)), replace=False)]
    S = cell.mems(Pb); blob = np.array([c3.field_sdf(p[0], p[1], p[2], G2, c3.F_GOLGI) for p in Pb[:200000]])
    inb = blob < 0; Sb = S[:200000][inb]
    chk = dict(sample_inside_blob=int(inb.sum()), inside_mito=int((Sb[:, c3.M_OM] < 0).sum()), inside_ER=int((Sb[:, c3.M_ER] < 0).sum()),
               inside_endo=int((Sb[:, c3.M_ENDO] < 0).sum()), within_10nm_other_membrane=int((np.min(np.abs(Sb[:, [c3.M_NE, c3.M_ER, c3.M_OM, c3.M_ENDO]]) - 2.0, axis=1) < 10).sum()),
               nucleus_side=int((cell0.labels(Pb[:200000][inb]) != c3.CYTO).sum()))
    # MT axis points: Golgi SDF clearance (>= 20 nm required: 12.5 + 7.5)
    Sm = cell.mems(mp); gm = Sm[:, c3.M_GOLGI]; lm = cell.labels(mp)
    chk["mt_axis_points_golgi_clearance_lt20nm"] = int((np.abs(gm) - c3.MEM_HALF[c3.M_GOLGI] < 20).sum())
    chk["mt_axis_points_not_cytosol"] = int((lm != c3.CYTO).sum())
    lab_all = cell.labels(Pb); chk["blob_label_counts"] = {c3.LABEL_NAMES[i]: int(c) for i, c in enumerate(np.bincount(lab_all, minlength=14)) if c}
    print("checks:", chk, flush=True)
    # ---------------------------------------------------------------- pre-pass trimming
    def gdist(P):
        return np.array(cell.mems(P)[:, c3.M_GOLGI], np.float64)      # GOLGI SDF (cisternal; < 0 lumen)
    def blobd(P):
        Gx = G2.copy(); Gx[c3.G_GOLGI_CIS] = 0.0                       # blob only: lumen + intercisternal = inside
        return np.array(c3.Cell(Gx, z["pores"]).mems(P)[:, c3.M_GOLGI], np.float64)
    near = lambda P, r: np.all(np.abs(P - cen) < BOX + 200, axis=1)
    trim = {}
    kind = cz["kind"]; pos = cz["pos"].astype(np.float64)
    nb_ = near(pos, 0); bad = np.zeros(len(pos), bool)
    ii = np.nonzero(nb_)[0]; bd_ = blobd(pos[ii])
    bad[ii[(kind[ii] == 1) & (bd_ < MT_DROP)]] = True; bad[ii[(kind[ii] == 2) & (bd_ < 4.0)]] = True
    bad[ii[(kind[ii] == 0) & (bd_ < 4.0)]] = True
    trim["mt_dimers"] = int((bad & (kind == 1)).sum()); trim["factin"] = int((bad & (kind == 2)).sum()); trim["centriole"] = int((bad & (kind == 0)).sum())
    if bad.any():
        cyt = {k_: cz[k_][~bad] for k_ in cz.files if k_ != "lib_ids"}; cyt["lib_ids"] = cz["lib_ids"]
        shutil.copy(f"{P7}/cytoskeleton_cell1.npz", f"{P7}/pre_ribbon_cytoskeleton_cell1.npz")
        shutil.copy(f"{P7}/cytoskeleton_cell1.json", f"{P7}/pre_ribbon_cytoskeleton_cell1.json")
        np.savez_compressed(f"{P7}/cytoskeleton_cell1.npz", **cyt)
        ids = list(cz["lib_ids"]); T_ = cyt["types"]
        J["consumed"] = {"TUBA1B-TUBB4B": int((T_ == ids.index("MT_TUBA1B-TUBB4B")).sum()), "ACTB": int((T_ == ids.index("F_ACTB")).sum())}
    J["log"]["golgi_ribbon_trim"] = trim; J["geometry"] = os.path.basename(GEO)
    json.dump(J, open(f"{P7}/cytoskeleton_cell1.json", "w"), indent=1, default=str)
    pz = np.load(f"{P7}/polysomes_cell1.npz"); pj = json.load(open(f"{P7}/polysomes_cell1.json")); rp = pz["pos"].astype(np.float64)
    ii = np.nonzero(near(rp, 0))[0]; badr = np.zeros(len(rp), bool); badr[ii[blobd(rp[ii]) < 15.0]] = True
    if badr.any():
        shutil.copy(f"{P7}/polysomes_cell1.npz", f"{P7}/pre_ribbon_polysomes_cell1.npz")
        np.savez_compressed(f"{P7}/polysomes_cell1.npz", **({k_: pz[k_][~badr] for k_ in pz.files if k_ != "lib_ids"} | {"lib_ids": pz["lib_ids"]}))
        pj["consumed"] = {k_: (int(v) - int(badr.sum()) if k_ == "RIBO80S" else v) for k_, v in pj["consumed"].items()}
    pj["v7_golgi_ribbon_dropped"] = int(badr.sum()); json.dump(pj, open(f"{P7}/polysomes_cell1.json", "w"), indent=1)
    trim["ribosomes"] = int(badr.sum())
    mr = np.load(f"{D}/prepass/mrna_cell1.npz", mmap_mode="r"); pts = mr["pts"]; inrib = mr["inrib"]
    lob, hib = cen - BOX - 200, cen + BOX + 200; extra = []
    for s in range(0, len(pts), 10_000_000):
        Q = np.asarray(pts[s:s + 10_000_000], np.float64); m_ = np.nonzero((inrib[s:s + 10_000_000] == 0) & np.all((Q > lob) & (Q < hib), axis=1))[0]
        if len(m_): extra.append(s + m_[blobd(Q[m_]) < 3.0])
    extra = np.concatenate(extra) if extra else np.zeros(0, np.int64)
    old = np.load(f"{P7}/mrna_drop.npy")
    if not os.path.exists(f"{P7}/pre_ribbon_mrna_drop.npy"): shutil.copy(f"{P7}/mrna_drop.npy", f"{P7}/pre_ribbon_mrna_drop.npy")
    np.save(f"{P7}/mrna_drop.npy", np.union1d(old, extra).astype(np.int64)); trim["mrna_beads"] = int(len(extra))
    print("pre-pass trimmed:", trim, flush=True)
    # ---------------------------------------------------------------- write
    summ["golgi_ribbon"] = dict(method="free cytoplasm filled nearest-first around the centrioles (jgolgi_ribbon.py)", centre_nm=cen.round(1).tolist(),
                                target_um3=vt, voxels=k, r_in_nm=R_IN, r_max_nm=round(r_max, 1), area_um2_blob=round(area, 1),
                                clr_nm=dict(membranes=CLR, pm=PM_CLR, mt_axis=MT_CLR, centriole=C_CLR), checks=chk, prepass_trim=trim,
                                em_blob_kept_as="jgeom_cell1_v7_emgolgi.npz")
    summ.pop("golgi_cisternae", None)
    z["G"] = G2; z["summary"] = np.array(json.dumps(summ))
    tmp = GEO + ".tmp.npz"; np.savez_compressed(tmp, **z); os.replace(tmp, GEO)
    print(f"written {GEO} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
