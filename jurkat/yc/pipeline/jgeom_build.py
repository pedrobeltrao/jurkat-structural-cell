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
apart) up to N_PORES (PLANNING_NOTES §4c: 1,000 NPCs).

v5: (1) open boundary. Cell 1 is truncated by the EM volume's z = 0 face (raw-EM checked). Every mask field is built on a
grid extruded EXTRUDE voxels below the face (the face slice repeated), so the smoothed surfaces run straight through the
face and no membrane closes it; G[G_FACE] = 1 and G[G_ZFACE] = z of the first imaged slice (model nm), where cell3d clips
the domain (label OUTV below). The NE (HMFF mesh) does not reach the face in cell 1. The summary gets an "open_face"
section: footprint, modelled PM area/volume, and the missing cap (side-wall extrapolation of the cross-section area
A(z) = pi r(z)^2 with r^2 quadratic in z, fitted over the first FIT_UM above the face) -> copy-number scale factors.
(2) nucleolus field from EM nucleolus_seg (instance labels > 0) inside the nucleus label, smoothed like the organelles."""
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
NUCL_SIGMA = 16.0
EXTRUDE = 24          # voxels (384 nm) of extrusion below the open face
FIT_UM = 1.0


def _tri_min_dist(P, V, F, k=12):
    """exact unsigned distance from points P to the triangle mesh (V, F), candidates = k nearest triangle centroids"""
    import importlib, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cellshape as cs
    cen = V[F].mean(1)
    _, cand = cKDTree(cen).query(P, k=k)
    return cs._min_tri_dist(P.astype(np.float64), cand.astype(np.int64), V.astype(np.float64), F.astype(np.int64))


def mask_sdf(occ, lo, sigma, zclip=None):
    level = ndi.gaussian_filter(occ.astype(np.float32), sigma / H) - 0.5
    verts, faces, _, _ = marching_cubes(np.pad(level, 1, constant_values=-0.5), 0.0, spacing=(H, H, H))
    verts = verts + lo - H
    a = np.linalg.norm(np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]]), axis=1)
    faces = faces[a > 1e-6]; a = a[a > 1e-6]
    inside = level > 0
    approx = np.where(inside, -ndi.distance_transform_edt(inside, sampling=H), ndi.distance_transform_edt(~inside, sampling=H)).astype(np.float32)
    sel = np.abs(approx) < EXACT_BAND + 2 * H
    idx = np.argwhere(sel); P = lo + idx * H
    d = _tri_min_dist(P, verts, faces)
    approx[sel] = np.where(inside[sel], -d, d)
    if zclip is not None:                                  # area above the open face only
        a = a[verts[faces].mean(1)[:, 2] >= zclip]
    area = float(a.sum() / 2 / 1e6)
    return approx, area


def face_summary(inside, zz, area_um2, vol_um3, z_face):
    """missing cap below the open face from a side-wall extrapolation: r(z)^2 = A(z)/pi fitted by a quadratic in z over
    the first FIT_UM above the face, extended below it to r = 0; cap volume = integral of A, cap area = surface of the
    revolved profile. Scale factors: f_vol = modelled / (modelled + cap) volume, f_pm = modelled PM / (modelled + cap) area."""
    A = inside.sum((0, 1)) * H * H / 1e6                     # um2 per z slice
    z = (zz - z_face) / 1e3                                  # um above the face
    m = z <= FIT_UM
    c = np.polyfit(z[m], A[m] / np.pi, 2)                    # r^2(z)
    roots = np.roots(c); roots = roots[np.isreal(roots)].real; roots = roots[roots < 0]
    z0 = roots.max() if len(roots) else -1.0
    zs = np.linspace(z0, 0.0, 400); r2 = np.clip(np.polyval(c, zs), 0, None); r = np.sqrt(r2)
    cap_vol = float(np.trapezoid(np.pi * r2, zs))
    dr = np.gradient(r, zs); cap_area = float(np.trapezoid(2 * np.pi * r * np.sqrt(1 + dr ** 2), zs))
    foot = float(A[0])
    return dict(z_face_nm=round(z_face, 1), footprint_um2=round(foot, 2), cap_height_um=round(-z0, 3),
                cap_vol_um3=round(cap_vol, 2), cap_area_um2=round(cap_area, 2), cell_vol_modelled_um3=round(vol_um3, 1),
                pm_area_modelled_um2=round(area_um2, 1), f_vol=round(vol_um3 / (vol_um3 + cap_vol), 4),
                f_pm=round(area_um2 / (area_um2 + cap_area), 4), fit=dict(r2_poly=[round(x, 3) for x in c], fit_um=FIT_UM),
                A_first_um=[round(float(a), 1) for a in A[m][::4]])


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
    open_face = o2[2] == 0 and bool((Li[:, :, 0] > 0).any())
    z_face = float(lo_frame[2] - centre[2]) if open_face else None          # first imaged slice (voxel-centre convention)
    # nucleolus occupancy on the same isotropic grid (instance labels > 0, inside the nucleus label)
    sys.path.insert(0, root); from jurkat.em.io import read as em_read
    sl_ = tuple(slice(o2[i], o2[i] + L.shape[i]) for i in range(3))
    NL = ndi.zoom((em_read("nucleolus_seg", "s2", sl_) > 0).astype(np.uint8), (1, 1, VOXS2[2] / H), order=0)[:, :, :Li.shape[2]]
    NL = (NL > 0) & (Li == 2)
    Li = np.pad(Li, 2); NL = np.pad(NL, 2)
    lo_e = lo.copy()
    if open_face:                                                             # extrude every mask through the face
        lo_e = lo - np.array([0.0, 0.0, EXTRUDE * H])
    R = float(np.linalg.norm((bb_hi - bb_lo) / 2) + 100.0)
    print("grid", Li.shape, "centre (label nm)", centre.round(1).tolist(), "enclosing R", round(R), "open face z", z_face, flush=True)

    def ext(occ):
        if not open_face: return occ
        f = occ[:, :, 2:3]                                                    # face slice (after the 2-voxel pad)
        return np.concatenate([np.zeros_like(occ[:, :, :2]), np.repeat(f, EXTRUDE, axis=2), occ[:, :, 2:]], axis=2)
    fields, summ = {}, {}
    zc = z_face if open_face else None
    for name, (ids, sig, band) in SPEC.items():
        occ = ext(np.isin(Li, ids))
        sdf, area = mask_sdf(occ, lo_e, sig, zclip=zc)
        fields[name] = bricks(sdf, lo_e, band)
        zz = lo_e[2] + np.arange(sdf.shape[2]) * H
        keepz = (zz >= z_face) if open_face else np.ones(sdf.shape[2], bool)
        summ[name] = dict(vol_mask_um3=round(float(occ[:, :, keepz].sum() * H ** 3 / 1e9), 2),
                          vol_sdf_um3=round(float((sdf[:, :, keepz] < 0).sum() * H ** 3 / 1e9), 2),
                          area_um2=round(area, 1), bricks_stored=int((fields[name][1] >= 0).sum()))
        print(name, summ[name], flush=True)
        if name == "cell" and open_face:
            summ["open_face"] = face_summary((sdf < 0)[:, :, keepz], zz[keepz], area, summ[name]["vol_sdf_um3"], z_face)
            print("open_face", summ["open_face"], flush=True)
    sdf, _ = mask_sdf(NL, lo, NUCL_SIGMA)
    fields["nucleolus"] = bricks(sdf, lo, (64.0, 64.0))
    lab_nl, n_nl = ndi.label(sdf < 0)
    summ["nucleolus"] = dict(vol_mask_um3=round(float(NL.sum() * H ** 3 / 1e9), 2), vol_sdf_um3=round(float((sdf < 0).sum() * H ** 3 / 1e9), 2),
                             components=int(n_nl), bricks_stored=int((fields["nucleolus"][1] >= 0).sum()))
    print("nucleolus", summ["nucleolus"], flush=True)
    m = trimesh.load(f"{D}/cell{cell}_NE_hmff_midsurface.ply", process=False)
    V = np.asarray(m.vertices, np.float64) - centre; F = np.asarray(m.faces, np.int64)
    ne = mesh_sdf(V, F, lo, Li.shape)
    fields["ne_mid"] = bricks(ne, lo, (120.0, 120.0))
    summ["ne_mid"] = dict(vol_inside_um3=round(float((ne < 0).sum() * H ** 3 / 1e9), 1), area_um2=round(float(m.area / 1e6), 1))
    print("ne_mid", summ["ne_mid"], flush=True)
    order = [fields[n] for n in c3.FIELD_NAMES]
    G = c3.pack_fields(R, order)
    if open_face:
        G[c3.G_FACE] = 1.0; G[c3.G_ZFACE] = z_face
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
