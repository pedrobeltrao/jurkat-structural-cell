# Compartment label grid for a target cell at 16 nm (s2 grid, label frame) -> input for geometry/packing.
# Labels: 0 outside, 1 cytosol, 2 nucleus, 3 mitochondrion, 4 ER, 5 golgi, 6 endolysosomal.
#  - nucleus, mito, endo/lyso/vesicle: _seg layers (GT Dice: nucleus 0.997, mito 0.96; endolysosomal merged, under-called)
#  - ER: un-flipped er_pred thresholded so that ER volume = 0.75 x er_seg volume in the cell (GT: er_seg over-calls 1.33x)
#  - golgi: _seg if >= 1 um3 in the cell; otherwise procedural (cell 1, PLANNING_NOTES §1): template = largest golgi_seg
#    component of cell 10, eroded/dilated to TARGET_GOLGI um3, centred on the centroid of the residual golgi_pred > 60 signal,
#    clipped to cytosol
#  - ld_seg is ignored (false positive, raw-EM check) -> cytosol
# usage: build_labels.py CELL ; writes data/derived/cell{CELL}_labels_s2.npz
import sys, numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, (__import__("os").environ.get("VCELL_ROOT", ".") + ""))
from jurkat.em.io import read
CELL = int(sys.argv[1]) if len(sys.argv) > 1 else 1
V3 = 16 * 16 * 13.76e-9
z = np.load(f"data/derived/cell{CELL}_s2.npz"); cell = z["cell"]; o = z["origin_s2_xyz"]
sl = tuple(slice(int(a), int(a) + n) for a, n in zip(o, cell.shape))
L = np.zeros(cell.shape, np.uint8); L[cell] = 1
nuc = (read("nucleus_seg", "s2", sl) == CELL) & cell              # nucleus instance id == cell id (seeded watershed)
mito = (read("mito_seg", "s2", sl) > 0) & cell & ~nuc
endo = ((read("endo_seg", "s2", sl) > 0) | (read("lyso_seg", "s2", sl) > 0) | (read("vesicle_seg", "s2", sl) > 0)) & cell & ~nuc & ~mito
er_seg_v = ((read("er_seg", "s2", sl) > 0) & cell & ~nuc & ~mito).sum()
erp = read("er_pred", "s2", sl); erp[~cell | nuc | mito | endo] = 0
target = 0.75 * er_seg_v
vals = np.sort(erp[erp > 0].ravel())[::-1]
thr = vals[int(min(target, len(vals) - 1))] if len(vals) else 255
er = erp > thr
print(f"ER: er_seg {er_seg_v*V3:.1f} um3 -> er_pred threshold {thr} gives {er.sum()*V3:.1f} um3 (target {target*V3:.1f})")
gol = (read("golgi_seg", "s2", sl) > 0) & cell & ~nuc & ~mito
TARGET_GOLGI = 0.0054 * cell.sum() * V3                            # median golgi volume fraction of cells with a segmented Golgi
if gol.sum() * V3 < 1.0:
    gp = read("golgi_pred", "s2", sl); sig = (gp > 60) & cell & ~nuc
    c0 = np.array(ndi.center_of_mass(sig)) if sig.any() else np.array(ndi.center_of_mass(cell & ~nuc))
    z10 = np.load("data/derived/cell10_s2.npz"); o10 = z10["origin_s2_xyz"]
    sl10 = tuple(slice(int(a), int(a) + n) for a, n in zip(o10, z10["cell"].shape))
    g10 = (read("golgi_seg", "s2", sl10) > 0) & z10["cell"]
    lab, n = ndi.label(g10); tpl = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    idx = np.argwhere(tpl); tpl = tpl[tuple(slice(a, b + 1) for a, b in zip(idx.min(0), idx.max(0)))]
    tpl = np.pad(tpl, 6)
    while tpl.sum() * V3 > TARGET_GOLGI * 1.05: tpl = ndi.binary_erosion(tpl)
    while tpl.sum() * V3 < TARGET_GOLGI * 0.95: tpl = ndi.binary_dilation(tpl)
    # placement: the most open perinuclear cytosol spot (0.1-1 um from the nucleus) within 2.5 um of the signal centroid
    free = cell & ~nuc & ~mito & ~endo
    dt_free = ndi.distance_transform_edt(free, sampling=(16, 16, 13.76))
    dn = ndi.distance_transform_edt(~nuc, sampling=(16, 16, 13.76))
    cand = free & (dn > 100) & (dn < 1000)
    g = np.argwhere(cand); dc = np.linalg.norm((g - c0) * [16, 16, 13.76], axis=1)
    g, dc = g[dc < 2500], dc[dc < 2500]
    if len(g): c0 = g[np.argmax(dt_free[tuple(g.T)])].astype(float)
    gol = np.zeros(cell.shape, bool)
    st = np.round(c0 - np.array(tpl.shape) / 2).astype(int)
    st = np.clip(st, 0, np.array(cell.shape) - np.array(tpl.shape))
    gol[tuple(slice(a, a + b) for a, b in zip(st, tpl.shape))] = tpl
    gol &= cell & ~nuc & ~mito             # golgi overrides ER: er_pred may contain misclassified Golgi membranes here
    for _ in range(20):                    # geodesic growth into free cytoplasm until the target volume (template clipped by nucleus/mito)
        if gol.sum() * V3 >= 0.95 * TARGET_GOLGI: break
        gol = ndi.binary_dilation(gol) & cell & ~nuc & ~mito & ~endo
    print(f"golgi: procedural, template from cell 10, centred at s2 crop {np.round(c0).astype(int).tolist()} -> {gol.sum()*V3:.2f} um3 "
          f"(target {TARGET_GOLGI:.2f})")
else:
    print(f"golgi: segmented {gol.sum()*V3:.2f} um3")
L[er] = 4; L[endo] = 6; L[gol] = 5; L[mito] = 3; L[nuc] = 2
names = ["outside", "cytosol", "nucleus", "mitochondrion", "er", "golgi", "endolysosomal"]
print("volumes um3:", {names[k]: round(float((L == k).sum() * V3), 1) for k in range(1, 7)})
np.savez_compressed(f"data/derived/cell{CELL}_labels_s2.npz", labels=L, names=np.array(names), origin_s2_xyz=o, voxel_nm=z["voxel_nm"])
