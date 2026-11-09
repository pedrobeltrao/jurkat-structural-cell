# Phase 1b: refine cell CELL (argv[1], default 1) at 16 nm (s2 grid, 16x16x13.76 nm, xyz). Seeded 3-label watershed:
#   1 = cell 1 (eroded 32 nm label + its nucleus), 2 = other cells (eroded), 3 = outside (confident ECS or background).
# Elevation = pm_pred + smoothed ecs_pred, so boundaries settle on the PM ridge / ECS edge.
# Output: data/derived/cell{CELL}_s2.npz (uint8 labels in the crop + origin), summary printed.
import numpy as np, tensorstore as ts, time
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage import measure
ROOT = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
import sys as _s; CELL = int(_s.argv[1]) if len(_s.argv) > 1 else 1
B = f"{ROOT}/data/em/jrc_jurkat-1/jrc_jurkat-1.n5/labels/"
import sys; sys.path.insert(0, ROOT)
from jurkat.em.io import read as em_read          # un-flips *_pred layers
rd = lambda p, sl: em_read(p.rsplit("/", 1)[0], p.rsplit("/", 1)[1], sl)
VOX = np.array([16, 16, 13.76]); V3 = VOX.prod() * 1e-9
t0 = time.time()
lab32 = np.load(f"{ROOT}/data/derived/cells_s3.npz")["labels"]
i = np.argwhere(lab32 == CELL); m = 16
lo = np.maximum((i.min(0) - m) * 2, 0); hi = np.minimum((i.max(0) + 1 + m) * 2, [2500, 750, 2140])
sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
pm, ecs = rd("pm_pred/s2", sl).astype(np.float32), rd("ecs_pred/s2", sl).astype(np.float32)
nuc, fg = rd("nucleus_seg/s2", sl), rd("masks/foreground/s1", sl) > 0
sl32 = tuple(slice(int(a) // 2, (int(b) + 1) // 2) for a, b in zip(lo, hi))
prior = np.repeat(np.repeat(np.repeat(lab32[sl32], 2, 0), 2, 1), 2, 2)[:pm.shape[0], :pm.shape[1], :pm.shape[2]]
print("crop", pm.shape, "lo", lo, "read s", round(time.time() - t0, 1), flush=True)

# only a band of +-BAND (16 nm voxels) around the 32 nm boundary of cell 1 is left unlabelled; deeper voxels are seeds.
# Memory-lean (job limit 128 GB): distance transforms on the 32 nm grid, then 2x upsampled.
BAND32 = 16
p32 = lab32[sl32]
up2 = lambda a: np.repeat(np.repeat(np.repeat(a, 2, 0), 2, 1), 2, 2)[:pm.shape[0], :pm.shape[1], :pm.shape[2]]
deep_in = up2(ndi.distance_transform_edt(p32 == CELL) > BAND32)
far_out = up2(ndi.distance_transform_edt(p32 != CELL) > BAND32)
mk = np.zeros(pm.shape, np.uint8)
mk[far_out & ((prior > 0) & (prior != CELL))] = 2
mk[far_out & (prior == 0)] = 3
mk[(ndi.gaussian_filter(ecs, 1) > 180) & (prior != CELL) | ~fg] = 3
mk[(nuc > 0) & (nuc != CELL)] = 2
mk[deep_in | (nuc == CELL)] = 1
del deep_in, far_out
print("unlabelled band voxels", int((mk == 0).sum()), flush=True)
elev = pm + ndi.gaussian_filter(ecs, 1.5); del ecs
ws = watershed(elev, markers=mk); del elev
c = ws == 1
c = ndi.binary_fill_holes(ndi.binary_opening(c, iterations=2))
lb, n = ndi.label(c); c = lb == (np.argmax(np.bincount(lb.ravel())[1:]) + 1)
vol = c.sum() * V3; nv = (nuc == CELL).sum() * V3
v, f, _, _ = measure.marching_cubes(ndi.gaussian_filter(np.pad(c, 2).astype(np.float32), 1.0), 0.5, spacing=VOX)
area = measure.mesh_surface_area(v, f) / 1e6
prior_vol = (prior == CELL).sum() * V3
agree = 2 * (c & (prior == CELL)).sum() / (c.sum() + (prior == CELL).sum())
# fraction of cell-1 surface voxels sitting on a PM ridge (pm_pred>127 within 2 voxels)
surf = c & ~ndi.binary_erosion(c); near_pm = ndi.binary_dilation(pm > 127, iterations=2)
faces = np.zeros(c.shape, bool); g = [lo[k] == 0 or hi[k] >= [2500, 750, 2140][k] for k in range(3)]
for k in range(3):
    if g[k]:
        s = [slice(None)] * 3; s[k] = 0 if lo[k] == 0 else -1; faces[tuple(s)] = True
print(f"cell{CELL} @16nm: vol {vol:.1f} um3 (32nm prior {prior_vol:.1f}, Dice vs prior {agree:.3f}) | nucleus {nv:.1f} ({nv/vol:.0%}) | "
      f"surface area {area:.0f} um2 | surface on PM ridge {(surf & near_pm).sum()/surf.sum():.1%} | on volume faces {(surf & faces).sum()/surf.sum():.1%}", flush=True)
cyto = c & (nuc != CELL)
print(f"cytoplasm (cell minus nucleus) {cyto.sum()*V3:.1f} um3; equivalent-sphere diameter {(6*vol/np.pi)**(1/3):.2f} um", flush=True)
np.savez_compressed(f"{ROOT}/data/derived/cell{CELL}_s2.npz", cell=c, origin_s2_xyz=lo, voxel_nm=VOX)
print("done s", round(time.time() - t0, 1))
