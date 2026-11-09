"""Real shapes for the composite model cell: cell outline, nucleus and vacuoles as smoothed signed-distance fields.

Input: cell_meshes.npz (scripts/cell_meshes.py), segmented cryo-FIB/SEM meshes of EMPIAR-11416 (Dumoux et al. 2023):
the outline, vacuoles and lipid droplets of cell 08 (the only cell not cut by the edge of the volume) and the
nucleus of cell 05. The cell is scaled to the inventory cell volume; the nucleus to its volume fraction and placed
where the segmentation pipeline put cell 08's synthetic nucleus (its most open region).

Processing (nm):
  1. each part is voxelised on an H grid (ray parity along z), closed morphologically (CLOSE: fills the grooves a
     curtaining stripe of the FIB data cut into the outline and some vacuoles) and Gaussian-smoothed, which removes the
     mesh facets and the 50 nm slice steps; vacuoles are smoothed more along z, which rounds the flat caps left by the
     slice-wise segmentation. The smoothed level set is the new surface
  2. vacuoles: the segmentation loses their tops and bottoms (z extent / in-plane extent 0.55-0.61 over 107 vacuoles,
     against 1.02 for the cells; 0.2-0.4 for most of cell 08's). The in-plane outlines are kept; each lobe is stretched
     along z about its centroid until its z extent matches its smaller in-plane extent (at most Z_MAX), and its
     distance field is blended with that of its moment-matched ellipsoid (ELL_BLEND), which rounds the missing caps.
     Lobes with an in-plane aspect above MAX_ASPECT are dropped (thin streaks along a curtaining stripe). The lobes are
     scaled together to the target volume and carved apart where they come closer than VAC_GAP to each other or to the
     nucleus, or closer than VAC_CORTEX to the outer cell surface; the carved shape is smoothed again
  3. the surface is re-meshed (marching cubes) and the exact distance to it is computed at the grid nodes near it
     (closest triangle via a k-d tree on triangle centroids); further away a Euclidean distance transform is used
  4. for the packing and the viewer, the grids are stored as sparse bricks of BRICK^3 voxels within a band around
     the surface (cell3d.field_sdf); outside the band only the sign is kept
"""
import numpy as np
import numba as nb
from config import ROOT

MESH_F = f"{ROOT}/cell_meshes.npz"
H = 20.0               # nm, grid spacing
BRICK = 8              # voxels per brick edge
SMOOTH = dict(cell=50.0, nucleus=40.0, vacuole=(30.0, 30.0, 90.0), carved=20.0)    # nm, Gaussian sigma (x, y, z)
CLOSE = dict(cell=100.0, nucleus=60.0, vacuole=60.0)    # nm, radius of the morphological closing
Z_MAX = 3.0            # largest z stretch of a vacuole lobe
ELL_BLEND = 0.5        # weight of the moment-matched ellipsoid in a lobe's distance field
MAX_ASPECT = 4.0       # in-plane aspect above which a segmented "vacuole" is treated as an artefact
VAC_GAP = 30.0         # nm between vacuole lobes, and between vacuoles and the nucleus
VAC_CORTEX = 220.0     # nm: vacuoles and the nucleus stay this deep below the outer cell surface (wall + cortical ER)
NUC_CORTEX = 260.0
EXACT_BAND = 160.0     # nm: exact distances within this band; EDT beyond
BANDS = dict(cell=(260.0, 100.0), nucleus=(130.0, 130.0), vacuole=(90.0, 90.0))   # (inside, outside) kept in bricks


# ----------------------------------------------------------------- voxelisation
@nb.njit(cache=True)
def _column_hits(V, F, lo, h, nx, ny):
    """z of every triangle crossing per (x, y) node column (two passes: count, fill)"""
    cnt = np.zeros(nx * ny, np.int64)
    ex = 0.38196601125 * 1e-3 * h                       # irrational offset: no ray hits an edge exactly
    for f in range(F.shape[0]):
        a = V[F[f, 0]]; b = V[F[f, 1]]; c = V[F[f, 2]]
        i0 = max(0, int(np.ceil((min(a[0], min(b[0], c[0])) - lo[0] - ex) / h)))
        i1 = min(nx - 1, int(np.floor((max(a[0], max(b[0], c[0])) - lo[0] - ex) / h)))
        j0 = max(0, int(np.ceil((min(a[1], min(b[1], c[1])) - lo[1] - ex) / h)))
        j1 = min(ny - 1, int(np.floor((max(a[1], max(b[1], c[1])) - lo[1] - ex) / h)))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                x = lo[0] + i * h + ex; y = lo[1] + j * h + ex
                d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
                if d == 0.0:
                    continue
                l1 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / d
                l2 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / d
                if l1 >= 0 and l2 >= 0 and l1 + l2 <= 1:
                    cnt[i * ny + j] += 1
    off = np.zeros(nx * ny + 1, np.int64)
    for k in range(nx * ny):
        off[k + 1] = off[k] + cnt[k]
    zs = np.empty(off[-1]); fill = off[:-1].copy()
    for f in range(F.shape[0]):
        a = V[F[f, 0]]; b = V[F[f, 1]]; c = V[F[f, 2]]
        i0 = max(0, int(np.ceil((min(a[0], min(b[0], c[0])) - lo[0] - ex) / h)))
        i1 = min(nx - 1, int(np.floor((max(a[0], max(b[0], c[0])) - lo[0] - ex) / h)))
        j0 = max(0, int(np.ceil((min(a[1], min(b[1], c[1])) - lo[1] - ex) / h)))
        j1 = min(ny - 1, int(np.floor((max(a[1], max(b[1], c[1])) - lo[1] - ex) / h)))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                x = lo[0] + i * h + ex; y = lo[1] + j * h + ex
                d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
                if d == 0.0:
                    continue
                l1 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / d
                l2 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / d
                if l1 >= 0 and l2 >= 0 and l1 + l2 <= 1:
                    zs[fill[i * ny + j]] = l1 * a[2] + l2 * b[2] + (1 - l1 - l2) * c[2]
                    fill[i * ny + j] += 1
    return zs, off


@nb.njit(cache=True)
def _fill(zs, off, lo, h, nx, ny, nz):
    occ = np.zeros((nx, ny, nz), np.uint8)
    for i in range(nx):
        for j in range(ny):
            z = np.sort(zs[off[i * ny + j]:off[i * ny + j + 1]])
            for p in range(0, z.shape[0] - 1, 2):
                k0 = max(0, int(np.ceil((z[p] - lo[2]) / h))); k1 = min(nz - 1, int(np.floor((z[p + 1] - lo[2]) / h)))
                for k in range(k0, k1 + 1):
                    occ[i, j, k] = 1
    return occ


def voxelize(V, F, lo, shape):
    zs, off = _column_hits(V.astype(np.float64), F.astype(np.int64), lo.astype(np.float64), H, shape[0], shape[1])
    return _fill(zs, off, lo.astype(np.float64), H, *shape)


# ----------------------------------------------------------------- exact distances
@nb.njit(cache=True, fastmath=True)
def _tri_dist2(p, a, b, c):
    """squared distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5)"""
    ab = b - a; ac = c - a; ap = p - a
    d1 = ab @ ap; d2 = ac @ ap
    if d1 <= 0 and d2 <= 0:
        q = a
    else:
        bp = p - b; d3 = ab @ bp; d4 = ac @ bp
        if d3 >= 0 and d4 <= d3:
            q = b
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0 and d1 >= 0 and d3 <= 0:
                q = a + ab * (d1 / (d1 - d3))
            else:
                cp = p - c; d5 = ab @ cp; d6 = ac @ cp
                if d6 >= 0 and d5 <= d6:
                    q = c
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0 and d2 >= 0 and d6 <= 0:
                        q = a + ac * (d2 / (d2 - d6))
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
                            q = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
                        else:
                            s = va + vb + vc
                            if s <= 1e-30:                      # degenerate triangle
                                return min(ap @ ap, min(bp @ bp, cp @ cp))
                            q = a + ab * (vb / s) + ac * (vc / s)
    e = p - q
    return e @ e


@nb.njit(cache=True, parallel=True)
def _min_tri_dist(P, cand, V, F):
    out = np.empty(P.shape[0])
    for n in nb.prange(P.shape[0]):
        best = 1e30
        for k in range(cand.shape[1]):
            f = cand[n, k]
            d = _tri_dist2(P[n], V[F[f, 0]], V[F[f, 1]], V[F[f, 2]])
            if d < best:
                best = d
        out[n] = np.sqrt(best)
    return out


def exact_sdf(level, lo):
    """signed distance (negative inside) at the grid nodes to the level-0 surface of level (positive inside):
    exact within EXACT_BAND, Euclidean distance transform beyond. Returns (sdf, verts, faces)."""
    from skimage.measure import marching_cubes
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    verts, faces, _, _ = marching_cubes(level, 0.0, spacing=(H, H, H))
    verts = verts + lo
    area = np.linalg.norm(np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]]), axis=1)
    faces = faces[area > 1e-6]                           # marching cubes emits some zero-area triangles
    inside = level > 0
    approx = np.where(inside, -distance_transform_edt(inside, sampling=H), distance_transform_edt(~inside, sampling=H))
    sel = np.abs(approx) < EXACT_BAND + 2 * H
    idx = np.argwhere(sel)
    P = lo + idx * H
    cen = verts[faces].mean(1)
    _, cand = cKDTree(cen).query(P, k=8)
    d = _min_tri_dist(P, cand.astype(np.int64), verts.astype(np.float64), faces.astype(np.int64))
    sdf = approx.astype(np.float64)
    sdf[sel] = np.where(inside[sel], -d, d)
    return sdf.astype(np.float32), verts, faces


def smooth_level(occ, sigma, close=0.0):
    from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes
    if close > 0:
        r = int(round(close / H)); g = np.arange(-r, r + 1)
        ball = (g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2) <= r * r
        occ = binary_fill_holes(binary_closing(np.pad(occ.astype(bool), r + 1), ball))[r + 1:-r - 1, r + 1:-r - 1, r + 1:-r - 1]
    sig = np.atleast_1d(np.asarray(sigma, np.float64)) / H
    return gaussian_filter(occ.astype(np.float32), sig if len(sig) == 3 else float(sig[0])) - 0.5


def mesh_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


# ----------------------------------------------------------------- sampling and bricks
class Grid:
    """dense signed-distance grid with trilinear sampling (builder use)"""
    def __init__(self, sdf, lo):
        self.sdf, self.lo = sdf, np.asarray(lo, np.float64)

    def __call__(self, P):
        from scipy.ndimage import map_coordinates
        P = np.atleast_2d(np.asarray(P, np.float64))
        return map_coordinates(self.sdf, ((P - self.lo) / H).T, order=1, mode="nearest")

    def grad(self, P, e=2.0):
        P = np.atleast_2d(np.asarray(P, np.float64))
        g = np.stack([self(P + e * u) - self(P - e * u) for u in np.eye(3)], 1)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)


def bricks(sdf, lo, band):
    """sparse brick encoding: (header values, index, data). index per brick: -1 inside, -2 outside, else brick id;
    data per stored brick: (BRICK+1)^3 node values (shared faces duplicated, so sampling never leaves the brick)"""
    bin_, bout = band
    n = np.array(sdf.shape)
    nb_ = -(-(n - 1) // BRICK)
    pad = nb_ * BRICK + 1 - n
    S = np.pad(sdf, [(0, p) for p in pad], mode="edge")
    S = np.clip(S, -bin_, bout)
    idx = np.empty(tuple(nb_), np.int64); data = []
    for bx in range(nb_[0]):
        for by in range(nb_[1]):
            for bz in range(nb_[2]):
                blk = S[bx * BRICK:bx * BRICK + BRICK + 1, by * BRICK:by * BRICK + BRICK + 1, bz * BRICK:bz * BRICK + BRICK + 1]
                if blk.max() <= -bin_:
                    idx[bx, by, bz] = -1
                elif blk.min() >= bout:
                    idx[bx, by, bz] = -2
                else:
                    idx[bx, by, bz] = len(data); data.append(blk.ravel())
    head = [lo[0], lo[1], lo[2], H, nb_[0], nb_[1], nb_[2], BRICK, bin_, bout]
    return np.array(head, np.float64), idx.ravel().astype(np.float64), np.concatenate(data).astype(np.float64)


# ----------------------------------------------------------------- the composite cell
def load_parts(cell_vol_um3, nuc_vol_um3):
    """meshes in nm: cell scaled to cell_vol_um3 about its centroid (= origin), nucleus to nuc_vol_um3"""
    z = np.load(MESH_F)
    Vc, Fc = z["cell_V"].astype(np.float64), z["cell_F"]
    s = (cell_vol_um3 / mesh_volume(Vc, Fc)) ** (1 / 3)
    sc = 1000.0 * s
    parts = dict(scale=s, cell=(Vc * sc, Fc),
                 vacuoles=[(z[f"vac{i}_V"].astype(np.float64) * sc, z[f"vac{i}_F"]) for i in range(int(z["n_vac"]))],
                 droplets=[(z[f"ld{i}_V"].astype(np.float64) * sc, z[f"ld{i}_F"]) for i in range(int(z["n_ld"]))])
    Vn, Fn = z["nuc_V"].astype(np.float64), z["nuc_F"]
    parts["nucleus"] = (Vn * 1000.0 * (nuc_vol_um3 / mesh_volume(Vn, Fn)) ** (1 / 3), Fn)
    parts["nuc_hint"] = (z["nuc_hint_centre"] * sc, z["nuc_hint_axes"])
    return parts


def _principal(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    dv = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    cen = (dv[:, None] * (a + b + c) / 4.0).sum(0) / dv.sum()
    a, b, c = a - cen, b - cen, c - cen; s = a + b + c
    M = (dv[:, None, None] * (np.einsum("ni,nj->nij", a, a) + np.einsum("ni,nj->nij", b, b)
                              + np.einsum("ni,nj->nij", c, c) + np.einsum("ni,nj->nij", s, s))).sum(0)
    w, U = np.linalg.eigh(M)
    return cen, U[:, ::-1].T                     # rows: long, middle, short


def build(geo, rng, log=print):
    """the composite cell's shapes. Returns dict with dense Grids (cell, nucleus, vacuole), brick encodings,
    the nucleus centre and frame, lipid-droplet spheres (centre, radius) and summary numbers."""
    V = geo["volumes_um3"]
    nuc_vol = V["nucleoplasm"] + V["nucleolus"]
    parts = load_parts(geo["cell_volume_um3"], nuc_vol)
    Vc, Fc = parts["cell"]
    lo = np.floor(Vc.min(0) / H) * H - 8 * H
    shape = tuple((np.ceil((Vc.max(0) + 12 * H - lo) / H)).astype(int) + 1)
    # ---- cell (rescaled once so the smoothed outline has the inventory volume)
    for _ in range(2):
        lev = smooth_level(voxelize(Vc, Fc, lo, shape), SMOOTH["cell"], CLOSE["cell"])
        f = geo["cell_volume_um3"] * 1e9 / ((lev > 0).sum() * H ** 3)
        if abs(f - 1) < 0.003:
            break
        Vc = Vc * f ** (1 / 3); parts["scale"] *= f ** (1 / 3)
    cell_sdf, cv, cf = exact_sdf(lev, lo)
    cell = Grid(cell_sdf, lo)
    log(f"cell: outline of segmented cell 08 scaled x{parts['scale']:.3f} to {geo['cell_volume_um3']} um3; smoothed volume "
        f"{(cell_sdf < 0).sum() * H ** 3 / 1e9:.2f} um3, {len(cf)} triangles, extent {np.round(cv.min(0))} .. {np.round(cv.max(0))} nm")
    # ---- nucleus: cell 05's, oriented like the hint (long axis to long axis), pushed inward until NUC_CORTEX deep
    Vn, Fn = parts["nucleus"]
    _, Un = _principal(Vn, Fn)
    hc, Uh = parts["nuc_hint"]
    R = Uh.T @ Un                                  # nucleus frame -> hint frame
    if np.linalg.det(R) < 0:
        R = Uh.T @ (Un * np.array([1, 1, -1])[:, None])
    Vn = Vn @ R.T
    c = hc.copy()
    for _ in range(60):
        dep = -cell(Vn + c)
        k = int(np.argmin(dep))
        if dep[k] >= NUC_CORTEX:
            break
        c += cell.grad(Vn[k] + c)[0] * -(NUC_CORTEX - dep[k] + 10.0)
    Vn = Vn + c
    lev = smooth_level(voxelize(Vn, Fn, lo, shape), SMOOTH["nucleus"], CLOSE["nucleus"])
    nuc_sdf, nv, nf = exact_sdf(lev, lo)
    nuc = Grid(nuc_sdf, lo)
    log(f"nucleus: segmented nucleus of cell 05 scaled to {nuc_vol:.2f} um3, centre {np.round(c)} nm "
        f"(moved {np.linalg.norm(c - hc):.0f} nm from the hint), smoothed volume {(nuc_sdf < 0).sum() * H ** 3 / 1e9:.2f} um3")
    # ---- vacuoles: z-corrected, scaled to the target volume, carved apart, smoothed
    target = V["vacuole"] * 1e9
    lobes, dropped, stretch = [], 0, []
    for Vv, Fv in parts["vacuoles"]:
        e = Vv.max(0) - Vv.min(0)
        if max(e[0], e[1]) / min(e[0], e[1]) > MAX_ASPECT:
            dropped += 1
            continue
        cen, _ = _principal(Vv, Fv)
        sz = float(np.clip(min(e[0], e[1]) / e[2], 1.0, Z_MAX)); stretch.append(sz)
        lobes.append((cen, (Vv - cen) * np.array([1.0, 1.0, sz]), Fv))
    k = (target / sum(abs(mesh_volume(D, F)) for _, D, F in lobes)) ** (1 / 3)
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    for it in range(4):
        min1 = np.full(shape, 1e4, np.float32); min2 = min1.copy(); arg = np.full(shape, -1, np.int16)
        subs = []
        for i, (cen, D, Fv) in enumerate(lobes):
            Vv = cen + D * k
            a = np.maximum(np.floor((Vv.min(0) - lo) / H).astype(int) - 12, 0)
            b = np.minimum(np.ceil((Vv.max(0) - lo) / H).astype(int) + 13, np.array(shape))
            occ = voxelize(Vv, Fv, lo + a * H, tuple(b - a))
            inside = smooth_level(occ, SMOOTH["vacuole"], CLOSE["vacuole"]) > 0
            d = np.where(inside, -distance_transform_edt(inside, sampling=H), distance_transform_edt(~inside, sampling=H))
            # blend with the moment-matched ellipsoid (rounds the caps the segmentation lost)
            ec, eU = _principal(Vv, Fv); eax = _semi_axes(Vv, Fv)
            g = np.stack(np.meshgrid(*[lo[j] + H * np.arange(a[j], b[j]) for j in range(3)], indexing="ij"), -1) - ec
            L = (g @ eU.T) / eax
            k0 = np.linalg.norm(L, axis=-1); k1 = np.linalg.norm(L / eax, axis=-1)
            d = ((1 - ELL_BLEND) * d + ELL_BLEND * k0 * (k0 - 1) / np.maximum(k1, 1e-12)).astype(np.float32)
            sl = tuple(slice(a[j], b[j]) for j in range(3))
            m1, m2, ag = min1[sl], min2[sl], arg[sl]
            better = d < m1
            m2[:] = np.where(better, m1, np.minimum(m2, d)); m1[:] = np.where(better, d, m1); ag[:] = np.where(better, i, ag)
            subs.append((sl, d))
        U = np.full(shape, 1e4, np.float32)
        for i, (sl, d) in enumerate(subs):
            others = np.where(arg[sl] == i, min2[sl], min1[sl])
            U[sl] = np.minimum(U[sl], np.maximum(d, VAC_GAP - others))
        U = np.maximum(U, VAC_GAP - nuc_sdf)
        U = np.maximum(U, cell_sdf + VAC_CORTEX)
        U = gaussian_filter(np.clip(U, -2000, 2000), SMOOTH["carved"] / H)
        vol = (U < 0).sum() * H ** 3
        log(f"  vacuoles pass {it + 1}: scale {k:.3f}, volume {vol / 1e9:.3f} um3 (target {target / 1e9:.3f})")
        if abs(vol / target - 1) < 0.01:
            break
        k *= (target / vol) ** (1 / 3)
    vac_sdf, vv, vf = exact_sdf(-U, lo)
    vac = Grid(vac_sdf, lo)
    from scipy.ndimage import label
    n_comp = label(vac_sdf < 0)[1]
    log(f"vacuoles: {len(lobes)} segmented lobes of cell 08 ({dropped} streak artefact dropped), z stretch "
        f"{min(stretch):.1f}-{max(stretch):.1f} (median {np.median(stretch):.1f}), scale {k:.3f}, carved {VAC_GAP:.0f} nm apart -> "
        f"{n_comp} compartments, {(vac_sdf < 0).sum() * H ** 3 / 1e9:.2f} um3, surface {_area(vv, vf) / 1e6:.1f} um2")
    # ---- lipid droplets: spheres of the segmented volumes at their centroids (positions checked by the builder)
    droplets = []
    for Vd, Fd in parts["droplets"]:
        cen, _ = _principal(Vd, Fd)
        droplets.append((cen, (3 * abs(mesh_volume(Vd, Fd)) / (4 * np.pi)) ** (1 / 3)))
    return dict(lo=lo, shape=shape, cell=cell, nucleus=nuc, vacuole=vac, nuc_centre=c,
                nuc_extent=float(np.linalg.norm(nv - c, axis=1).max()), droplets=droplets,
                meshes=dict(cell=(cv, cf), nucleus=(nv, nf), vacuole=(vv, vf)),
                bricks=dict(cell=bricks(cell_sdf, lo, BANDS["cell"]), nucleus=bricks(nuc_sdf, lo, BANDS["nucleus"]),
                            vacuole=bricks(vac_sdf, lo, BANDS["vacuole"])))


def _semi_axes(V, F):
    """semi-axes (long, middle, short) of the solid ellipsoid with the same second moments"""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    dv = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    cen = (dv[:, None] * (a + b + c) / 4.0).sum(0) / dv.sum()
    a, b, c = a - cen, b - cen, c - cen; s = a + b + c
    M = (dv[:, None, None] * (np.einsum("ni,nj->nij", a, a) + np.einsum("ni,nj->nij", b, b)
                              + np.einsum("ni,nj->nij", c, c) + np.einsum("ni,nj->nij", s, s))).sum(0) / 20.0
    return np.sqrt(5.0 * np.clip(np.sort(np.linalg.eigvalsh(M / dv.sum()))[::-1], 1e-9, None))


def _area(V, F):
    return float(0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())
