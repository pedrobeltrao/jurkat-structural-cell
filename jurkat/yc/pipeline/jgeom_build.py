"""Build the Jurkat cell geometry (cell3d.py fields) from the 16 nm compartment label grid and the HMFF NE mesh.

Inputs (data/derived): cell{N}_labels_s2.npz (uint8 labels on the s2 grid 16 x 16 x 13.76 nm, label frame),
cell{N}_NE_hmff_midsurface.ply (nm, label frame), np_seg_instances.parquet (NPC centroids, s0 voxels).
Output: data/derived/jgeom_cell{N}.npz with G (enclosing radius + sparse-brick SDF fields), pores (centre on the
perinuclear midsurface + outward unit normal), centre_nm (cell-centre in the label frame; model coordinates = label
frame nm - centre_nm), and per-field summaries.

Per mask field: resample to isotropic 16 nm (nearest), Gaussian-smooth the occupancy (sigma per field), level 0.5,
marching cubes, exact distance to the triangles within EXACT_BAND, Euclidean distance transform beyond.
NE field: exact distance to the HMFF midsurface triangles (sign from ray-parity voxelisation of the closed mesh).
Pores: np_seg centroids of the cell projected onto the NE midsurface, then random additional positions (>= MIN_SEP
apart) up to N_PORES (PLANNING_NOTES §4c: 1,000 NPCs)."""
import sys, json, numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

H = 16.0
BRICK = 8
EXACT_BAND = 96.0
N_PORES = 1000
MIN_SEP = 150.0
VOXS2 = np.array([16.0, 16.0, 13.76])
SPEC = {   # field: (label ids, smoothing sigma nm, brick band (inside, outside) nm)
    "cell": ((1, 2, 3, 4, 5, 6), 16.0, (160.0, 64.0)),
    "mito": ((3,), 12.0, (64.0, 64.0)),
    "er": ((4,), 10.0, (48.0, 48.0)),
    "golgi": ((5,), 16.0, (64.0, 64.0)),
    "endo": ((6,), 12.0, (64.0, 64.0)),
}


def _tri_min_dist(P, V, F, k=12):
    """exact unsigned distance from points P to the triangle mesh (V, F), candidates = k nearest triangle centroids"""
    import importlib, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cellshape as cs
    cen = V[F].mean(1)
    _, cand = cKDTree(cen).query(P, k=k)
    return cs._min_tri_dist(P.astype(np.float64), cand.astype(np.int64), V.astype(np.float64), F.astype(np.int64))


def mask_sdf(occ, lo, sigma):
    level = ndi.gaussian_filter(occ.astype(np.float32), sigma / H) - 0.5
    verts, faces, _, _ = marching_cubes(np.pad(level, 1, constant_values=-0.5), 0.0, spacing=(H, H, H))
    verts = verts + lo - H
    a = np.linalg.norm(np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]]), axis=1)
    faces = faces[a > 1e-6]
    inside = level > 0
    approx = np.where(inside, -ndi.distance_transform_edt(inside, sampling=H), ndi.distance_transform_edt(~inside, sampling=H)).astype(np.float32)
    sel = np.abs(approx) < EXACT_BAND + 2 * H
    idx = np.argwhere(sel); P = lo + idx * H
    d = _tri_min_dist(P, verts, faces)
    approx[sel] = np.where(inside[sel], -d, d)
    area = float(a.sum() / 2 / 1e6)
    return approx, area


def mesh_sdf(V, F, lo, shape):
    import cellshape as cs
    occ = np.zeros(shape, bool)
    # ray-parity voxelisation (cellshape.voxelize uses its module H; temporarily set it)
    old = cs.H; cs.H = H
    try:
        occ = cs.voxelize(V, F, np.asarray(lo, np.float64), shape)
    finally:
        cs.H = old
    approx = np.where(occ, -ndi.distance_transform_edt(occ, sampling=H), ndi.distance_transform_edt(~occ, sampling=H)).astype(np.float32)
    sel = np.abs(approx) < 200.0
    idx = np.argwhere(sel); P = lo + idx * H
    d = _tri_min_dist(P, V, F)
    approx[sel] = np.where(occ[sel], -d, d)
    return approx


def bricks(sdf, lo, band):
    bin_, bout = band
    n = np.array(sdf.shape); nb_ = -(-(n - 1) // BRICK); pad = nb_ * BRICK + 1 - n
    S = np.clip(np.pad(sdf, [(0, p) for p in pad], mode="edge"), -bin_, bout)
    idx = np.empty(tuple(nb_), np.int64); data = []
    for bx in range(nb_[0]):
        for by in range(nb_[1]):
            for bz in range(nb_[2]):
                blk = S[bx * BRICK:bx * BRICK + BRICK + 1, by * BRICK:by * BRICK + BRICK + 1, bz * BRICK:bz * BRICK + BRICK + 1]
                if blk.max() <= -bin_: idx[bx, by, bz] = -1
                elif blk.min() >= bout: idx[bx, by, bz] = -2
                else: idx[bx, by, bz] = len(data); data.append(blk.ravel())
    head = [lo[0], lo[1], lo[2], H, nb_[0], nb_[1], nb_[2], BRICK, bin_, bout]
    dat = np.concatenate(data) if data else np.zeros(0)
    return np.array(head, np.float64), idx.ravel().astype(np.float64), dat.astype(np.float64)


def main(cell=1, root=(__import__("os").environ.get("VCELL_ROOT", ".") + "")):
    import trimesh
    import pandas as pd
    import cell3d as c3
    D = f"{root}/data/derived"
    z = np.load(f"{D}/cell{cell}_labels_s2.npz"); L = z["labels"]; o2 = z["origin_s2_xyz"]
    Li = ndi.zoom(L, (1, 1, VOXS2[2] / H), order=0)                          # isotropic 16 nm
    lo_frame = o2 * VOXS2                                                     # label-frame nm of voxel (0,0,0)
    idx = np.argwhere(Li > 0); bb_lo = lo_frame + idx.min(0) * H; bb_hi = lo_frame + idx.max(0) * H
    centre = (bb_lo + bb_hi) / 2
    lo = lo_frame - centre - 2 * H                                            # model-frame grid origin (2-voxel pad)
    Li = np.pad(Li, 2)
    R = float(np.linalg.norm((bb_hi - bb_lo) / 2) + 100.0)
    print("grid", Li.shape, "centre (label nm)", centre.round(1).tolist(), "enclosing R", round(R), flush=True)
    fields, summ = {}, {}
    for name, (ids, sig, band) in SPEC.items():
        occ = np.isin(Li, ids)
        sdf, area = mask_sdf(occ, lo, sig)
        fields[name] = bricks(sdf, lo, band)
        summ[name] = dict(vol_mask_um3=round(float(occ.sum() * H ** 3 / 1e9), 2), vol_sdf_um3=round(float((sdf < 0).sum() * H ** 3 / 1e9), 2),
                          area_um2=round(area, 1), bricks_stored=int((fields[name][1] >= 0).sum()))
        print(name, summ[name], flush=True)
    m = trimesh.load(f"{D}/cell{cell}_NE_hmff_midsurface.ply", process=False)
    V = np.asarray(m.vertices, np.float64) - centre; F = np.asarray(m.faces, np.int64)
    ne = mesh_sdf(V, F, lo, Li.shape)
    fields["ne_mid"] = bricks(ne, lo, (120.0, 120.0))
    summ["ne_mid"] = dict(vol_inside_um3=round(float((ne < 0).sum() * H ** 3 / 1e9), 1), area_um2=round(float(m.area / 1e6), 1))
    print("ne_mid", summ["ne_mid"], flush=True)
    order = [fields[n] for n in c3.FIELD_NAMES]
    G = c3.pack_fields(R, order)
    # pores: np_seg centroids -> nearest point on the NE mesh; then random fill
    npc = pd.read_parquet(f"{D}/np_seg_instances.parquet"); npc = npc[npc.cell2 == cell]
    P = npc[["x", "y", "z"]].values * np.array([4.0, 4.0, 3.44]) - centre
    tri_c = V[F].mean(1); nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    # orient normals outward (away from the mesh centroid on average)
    if np.mean(np.einsum("ij,ij->i", nrm, tri_c - V.mean(0))) < 0: nrm = -nrm
    tree = cKDTree(tri_c)
    d, j = tree.query(P)
    keep = d < 300.0
    pores = [np.r_[tri_c[k], nrm[k]] for k in j[keep]]
    sel = []
    for p in pores:                                                           # de-duplicate detections
        if all(np.linalg.norm(p[:3] - q[:3]) >= MIN_SEP for q in sel): sel.append(p)
    n_det = len(sel)
    area_w = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1); area_w /= area_w.sum()
    rng = np.random.default_rng(7); tries = 0
    pts = np.array([s[:3] for s in sel]) if sel else np.zeros((0, 3))
    while len(sel) < N_PORES and tries < 200000:
        k = rng.choice(len(F), p=area_w); tries += 1
        c = tri_c[k]
        if len(pts) and np.min(np.linalg.norm(pts - c, axis=1)) < MIN_SEP: continue
        sel.append(np.r_[c, nrm[k]]); pts = np.vstack([pts, c])
    pores = np.array(sel)
    print(f"pores: {n_det} from np_seg (of {len(P)}, {keep.sum()} within 300 nm of the NE), {len(pores) - n_det} added -> {len(pores)}", flush=True)
    np.savez_compressed(f"{D}/jgeom_cell{cell}.npz", G=G, pores=pores, centre_nm=centre, R=R, summary=json.dumps(summ))
    print("G size", G.size, "->", f"{D}/jgeom_cell{cell}.npz")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
