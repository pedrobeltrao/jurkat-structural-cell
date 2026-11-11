"""Procedural cisternal Golgi for the Jurkat cell geometry (v7; PLANNING_NOTES "v7 prep", rule R5b).

The EM Golgi segmentation (label 5) is one smoothed blob (4.15 um3, 63 um2, median half-thickness ~40 nm: a ribbon of
~80 nm sheets/tubules), so packs v1-v6 accounted for the cisternal membrane with GOLGI_STACK_F = 4.5 (only 1/4.5 of the
Golgi TM inventory and, from v6, of the Golgi shells was packed). Here the blob interior is divided into cisternae that run
parallel to the blob surface (cell3d.golgi_sdf): lumen layers of midsurface separation 2W = 20 nm alternating with
intercisternal cytosol of 20 nm (period S = 40 nm); the blob surface is the outer membrane of the outermost cisterna.
Values: cisternal lumen ~10-20 nm, intercisternal spacing ~10-20 nm (Ladinsky et al. 1999 JCB 144:1135, NRK Golgi
tomography); the 4 nm bilayers leave ~16 nm free on each side. Thin pieces (half-thickness < 2W) stay a single lumen
(vesicles / tubules). Simplifications: no fenestrae, rims, cis-trans polarity or ribbon gaps; layers follow the blob
surface everywhere (onion-like in tubular parts).
A deep Golgi SDF (same mask, smoothing and exact-distance method as jgeom_build, inside clamp DEEP_IN nm) is stored in the
section so the depth is known beyond the +-64 nm band of the blob field. MTs and other pre-pass content built on the v6
geometry stay valid: outside the blob the GOLGI SDF is unchanged, inside it only rises (max).
usage: jgolgi.py [in=data/derived/jgeom_cell1.npz] [out=data/derived/jgeom_cell1_v7.npz]
The input is never modified; an existing cisternae section in the input is replaced."""
import os, sys, json, time, numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3
import jgeom_build as jb

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"
W_CIS = 10.0          # nm, half the membrane-midsurface separation across a cisternal lumen
S_CIS = 40.0          # nm, layer period (lumen 2W + intercisternal 2W)
DEEP_IN = 320.0       # nm, inside clamp of the deep field (max half-thickness of the blob 254 nm)


def golgi_deep_field(cell=1):
    """exact deep SDF of the Golgi mask on the jgeom_build grid (same occupancy, extrusion and smoothing)"""
    z = np.load(f"{D}/cell{cell}_labels_s2.npz"); L = z["labels"]; o2 = z["origin_s2_xyz"]
    Li = ndi.zoom(L, (1, 1, jb.VOXS2[2] / jb.H), order=0)
    lo_frame = o2 * jb.VOXS2
    idx = np.argwhere(Li > 0); bb_lo = lo_frame + idx.min(0) * jb.H; bb_hi = lo_frame + idx.max(0) * jb.H
    centre = (bb_lo + bb_hi) / 2
    lo = lo_frame - centre - 2 * jb.H
    open_face = o2[2] == 0 and bool((Li[:, :, 0] > 0).any())
    Li = np.pad(Li, 2); lo_e = lo.copy()
    occ = Li == 5
    if open_face:
        lo_e = lo - np.array([0.0, 0.0, jb.EXTRUDE * jb.H])
        f = occ[:, :, 2:3]
        occ = np.concatenate([np.zeros_like(occ[:, :, :2]), np.repeat(f, jb.EXTRUDE, axis=2), occ[:, :, 2:]], axis=2)
    sdf, area = jb.mask_sdf(occ, lo_e, jb.SPEC["golgi"][1])
    # deeper than EXACT_BAND the mask_sdf interior is the voxel EDT: fine for layer depth at 16 nm sampling
    return jb.bricks(sdf, lo_e, (DEEP_IN, 64.0)), area, centre


def strip(G):
    """G without an earlier cisternae section (it may sit before or after the speckle section)"""
    G = G.copy()
    o = int(G[c3.G_GOLGI_CIS])
    if o <= 0: return G
    G[c3.G_GOLGI_CIS] = 0.0
    if int(G[c3.G_SPECKLE]) > o:           # speckles appended after the Golgi section: move them down
        so = int(G[c3.G_SPECKLE]); n = int(G[so]); spk = G[so:so + 1 + 4 * n].copy()
        G = G[:o]; G[c3.G_SPECKLE] = len(G); return np.r_[G, spk]
    return G[:o]


def append_section(G, head, idx, data, W=W_CIS, S=S_CIS):
    o = len(G)
    ho = o + 2 + 12                          # start of the index block
    hdr = list(head) + [ho, ho + len(idx)]
    sec = np.r_[W, S, np.array(hdr, np.float64), idx, data]
    G2 = np.r_[G, sec]; G2[c3.G_GOLGI_CIS] = o
    return G2


def main(fin=f"{D}/jgeom_cell1.npz", fout=f"{D}/jgeom_cell1_v7.npz"):
    t0 = time.time()
    z = dict(np.load(fin)); G0 = strip(z["G"]); summ = json.loads(str(z["summary"]))
    (head, idx, data), area, centre = golgi_deep_field()
    assert np.allclose(centre, z["centre_nm"]), "grid centre differs from the geometry"
    G = append_section(G0, head, idx, data)
    cell0, cell = c3.Cell(G0, z["pores"]), c3.Cell(G, z["pores"])
    # checks on a random sample inside the blob bounding region + marching-cubes area in boxes
    rng = np.random.default_rng(11)
    lo_, hi_ = np.array(head[:3]), np.array(head[:3]) + np.array(head[4:7]) * head[7] * head[3]
    P = rng.uniform(lo_, hi_, (3_000_000, 3))
    l0, l1 = cell0.labels(P), cell.labels(P)
    vbox = float(np.prod(hi_ - lo_)) / 1e9
    v_blob = (l0 == c3.GOLGI).mean() * vbox; v_lum = (l1 == c3.GOLGI).mean() * vbox
    to_cyto = ((l0 == c3.GOLGI) & (l1 == c3.CYTO)).mean() * vbox
    changed_other = int(((l0 != l1) & ~((l0 == c3.GOLGI) & (l1 == c3.CYTO))).sum())
    out_chg = 0
    m0, m1 = cell0.mems(P[l0 != c3.GOLGI][:500000]), cell.mems(P[l0 != c3.GOLGI][:500000])
    out_chg = int((np.abs(m0[:, c3.M_GOLGI] - m1[:, c3.M_GOLGI]) > 1e-3).sum())   # outside the blob: must be 0
    # area: marching cubes of the GOLGI SDF at 4 nm over the whole blob bounding box in 1 um sub-boxes (sampled)
    from skimage.measure import marching_cubes
    a_new = a_old = vol_s = 0.0; h = 4.0; nb = 0
    gp = P[l0 == c3.GOLGI]                                    # 24 boxes of 0.5 um centred on random blob points
    boxes = [tuple(p - 250.0) for p in gp[rng.choice(len(gp), 24, replace=False)]]
    sel = range(len(boxes))
    for b in sel:
        b0 = np.array(boxes[b]); ax = [b0[i] + np.arange(0, 500.0 + h, h) for i in range(3)]
        X = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3); sh = tuple(len(a) for a in ax)
        for nm, cc in (("old", cell0), ("new", cell)):
            s = cc.mems(X)[:, c3.M_GOLGI].reshape(sh)
            if s.min() < 0 < s.max():
                v, fc, _, _ = marching_cubes(s, 0.0, spacing=(h,) * 3)
                a = float(np.linalg.norm(np.cross(v[fc[:, 1]] - v[fc[:, 0]], v[fc[:, 2]] - v[fc[:, 0]]), axis=1).sum() / 2) / 1e6
                if nm == "old": a_old += a
                else: a_new += a
        nb += 1
    ratio = a_new / max(a_old, 1e-9)
    S = dict(W_nm=W_CIS, S_nm=S_CIS, deep_inside_clamp_nm=DEEP_IN, blob_area_um2_rebuilt=round(area, 1),
             blob_area_um2=summ["golgi"]["area_um2"], blob_lumen_um3_mc=round(v_blob, 3), cisternal_lumen_um3_mc=round(v_lum, 3),
             blob_to_cytosol_um3=round(to_cyto, 3), other_label_changes_in_sample=changed_other,
             golgi_sdf_changed_outside_blob=out_chg, area_ratio_new_over_blob_sampled_boxes=round(ratio, 2),
             area_new_um2_est=round(ratio * summ["golgi"]["area_um2"], 1),
             note="area estimate = sampled-box marching-cubes ratio x blob area; jwc.step_counts uses the measured tile areas")
    summ["golgi_cisternae"] = S
    z["G"] = G; z["summary"] = np.array(json.dumps(summ))
    tmp = fout + ".tmp.npz"; np.savez_compressed(tmp, **z); os.replace(tmp, fout)
    print(json.dumps(S, indent=1), f"\nwritten {fout} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(a[:2]))
