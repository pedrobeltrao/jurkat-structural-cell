# Remove thin bridged attachments (substrate lamellae, touching fragments) from a refined cell mask (16 nm):
# opening (r voxels) -> largest component -> geodesic reconstruction inside the original mask, limited to R voxels, so
# microvilli attached to the main body come back but bodies connected only through thin bridges do not.
# usage: clean_cell.py CELL [r=3] [R=6]; writes data/derived/cell{CELL}_s2_clean.npz
import sys, numpy as np
from scipy import ndimage as ndi
from skimage import measure
CELL = int(sys.argv[1]); r = int(sys.argv[2]) if len(sys.argv) > 2 else 3; R = int(sys.argv[3]) if len(sys.argv) > 3 else 6
z = np.load(f"data/derived/cell{CELL}_s2.npz"); c = z["cell"]; V3 = 16 * 16 * 13.76e-9
ball = ndi.generate_binary_structure(3, 1)
op = ndi.binary_opening(c, structure=ball, iterations=r)
lab, n = ndi.label(op); core = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
rec = core.copy()
for _ in range(R): rec = ndi.binary_dilation(rec, structure=ball) & c
rec = ndi.binary_fill_holes(rec)
lab, n = ndi.label(rec); rec = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
def area(m):
    v, f, _, _ = measure.marching_cubes(ndi.gaussian_filter(np.pad(m, 2).astype(np.float32), 1.0), 0.5, spacing=(16, 16, 13.76))
    return measure.mesh_surface_area(v, f) / 1e6
print(f"cell {CELL}: volume {c.sum()*V3:.1f} -> {rec.sum()*V3:.1f} um3 (removed {(c & ~rec).sum()*V3:.1f}); area {area(c):.0f} -> {area(rec):.0f} um2")
np.savez_compressed(f"data/derived/cell{CELL}_s2_clean.npz", cell=rec, origin_s2_xyz=z["origin_s2_xyz"], voxel_nm=z["voxel_nm"])
