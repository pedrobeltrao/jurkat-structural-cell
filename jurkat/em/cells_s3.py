# Phase 1a (coarse, 32 nm): cell instances = nucleus-seeded watershed on PM evidence within foreground & not-ECS.
# v2 (2026-11-08): *_pred layers are read un-flipped (jurkat.em.io); v1 (scratch_checks/cells_s3_watershed.py) used
# y-flipped pm_pred/ecs_pred against nucleus_seg seeds. Writes data/derived/cells_s3.npz (+ per-cell table).
import sys, numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
sys.path.insert(0, (__import__("os").environ.get("VCELL_ROOT", ".") + ""))
from jurkat.em.io import read
fg = read("masks/foreground", "s2") > 0
ecsp = read("ecs_pred", "s3"); pmp = read("pm_pred", "s3"); nuc = read("nucleus_seg", "s3")
sh = tuple(min(a, b) for a, b in zip(fg.shape, nuc.shape)); fg, ecsp, pmp, nuc = [a[:sh[0], :sh[1], :sh[2]] for a in (fg, ecsp, pmp, nuc)]
vox = 32 * 32 * 27.52e-9
print("shape", sh, "foreground um3", round(fg.sum() * vox), "| ecs_pred>127 inside fg um3", round(((ecsp > 127) & fg).sum() * vox),
      "| ecs_pred>127 overlapping nuclei um3", round(((ecsp > 127) & (nuc > 0)).sum() * vox, 1))
interior = fg & (ecsp <= 96) & ~(pmp > 200)
interior = ndi.binary_opening(interior, iterations=1) | (nuc > 0)
elev = pmp.astype(np.float32) + ndi.gaussian_filter(ecsp.astype(np.float32), 1)
lab = watershed(elev, markers=nuc.astype(np.int32), mask=interior)
faces = np.zeros(sh, bool); faces[[0, -1], :, :] = faces[:, [0, -1], :] = faces[:, :, [0, -1]] = True
rows = []
for i in np.unique(nuc)[1:]:
    c = lab == i; v = c.sum() * vox; nv = (nuc == i).sum() * vox
    surf = c & ~ndi.binary_erosion(c); onface = (surf & faces).sum() / max(surf.sum(), 1)
    pm_on = (surf & ndi.binary_dilation(pmp > 127, iterations=1)).sum() / max(surf.sum(), 1)
    idx = np.argwhere(c); lo, hi = idx.min(0) * [.032, .032, .02752], idx.max(0) * [.032, .032, .02752]
    rows.append(dict(cell=int(i), vol_um3=round(v, 1), nucleus_um3=round(nv, 1), nuc_frac=round(nv / v, 3), surf_on_faces=round(onface, 3),
                     surf_on_pm=round(pm_on, 3), bbox_lo_um=lo.round(1).tolist(), bbox_hi_um=hi.round(1).tolist()))
T = pd.DataFrame(rows); print(T.to_string(index=False))
print("unassigned interior (no nucleus reached) um3", round((interior & (lab == 0)).sum() * vox, 1))
np.savez_compressed("data/derived/cells_s3.npz", labels=lab.astype(np.uint8), voxel_nm=np.array([32, 32, 27.52]), axes="xyz")
T.to_csv("data/derived/cells_s3_table.tsv", sep="\t", index=False)
