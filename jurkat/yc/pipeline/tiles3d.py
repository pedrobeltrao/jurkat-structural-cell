"""Tile decomposition of the 3D cell (200 nm cubes) and per-tile geometry: region volumes, membrane areas,
8-nm marching-cubes membrane meshes (seamless across tiles because tiles share lattice planes)."""
import os, json
import numpy as np
import numba as nb
import cell3d as c3

TILE = 200.0
VOX = 4.0        # voxel size for region measures and placement seeds
MESH_H = 8.0     # marching-cubes lattice
from config import GDIR, NPROC

# packing regions
(R_CYTO, R_NUCP, R_NUCL, R_ERL, R_VAC, R_MATRIX, R_IMS, R_GOLGI, R_PEROX, R_WALL,
 R_PM_PERI, R_NE_PERI, R_ER_PERI, R_LD_PERI) = range(14)
VOL_REGIONS = ["cytosol", "nucleoplasm", "nucleolus", "ER_lumen", "vacuole", "mito_matrix", "mito_IMS",
               "golgi_endosome", "peroxisome", "cell_wall", "PM_peripheral", "NE_peripheral", "ER_peripheral",
               "LD_peripheral"]
MEM_REGION0 = 14     # region id = 14 + membrane id (PM, ER, NE, VAC, OM, IM, GOLGI, PEROX)
MEM_REGIONS = ["PM", "ER", "NE", "VAC", "MITO_OM", "MITO_IM", "GOLGI", "PEROX"]
REGION_NAMES = VOL_REGIONS + ["mem_" + m for m in MEM_REGIONS]
NREG = len(REGION_NAMES)
# allowed label bitmask for each volume region
L = c3
REG_MASK = np.zeros(NREG, np.int64)
for r, labs in {R_CYTO: [L.CYTO], R_NUCP: [L.NUCP], R_NUCL: [L.NUCL], R_ERL: [L.CER, L.PERI], R_VAC: [L.VAC],
                R_MATRIX: [L.MATRIX], R_IMS: [L.IMS], R_GOLGI: [L.GOLGI], R_PEROX: [L.PEROX],
                R_WALL: [L.MANNAN, L.GLUCAN], R_PM_PERI: [L.CYTO], R_NE_PERI: [L.CYTO], R_ER_PERI: [L.CYTO],
                R_LD_PERI: [L.CYTO]}.items():
    for l in labs:
        REG_MASK[r] |= 1 << l
# peripheral bands: (membrane, max |d|)
BANDS = {R_PM_PERI: (c3.M_PM, 22.0), R_NE_PERI: (c3.M_NE, 20.0), R_ER_PERI: (c3.M_ER, 25.0), R_LD_PERI: (c3.M_LD, 12.0)}
WALL_DEPTH = 60.0    # wall proteins: outer 60 nm of the wall (mannoprotein layer + outer glucan)


def tile_list(cell):
    """tiles intersecting the cell (its enclosing sphere, and with a real outline the outline itself)"""
    R = cell.G[0]
    n = int(np.ceil(R / TILE))
    out = []
    for ix in range(-n, n):
        for iy in range(-n, n):
            for iz in range(-n, n):
                lo = np.array([ix, iy, iz]) * TILE; hi = lo + TILE
                q = np.clip(0.0, lo, hi)
                if np.linalg.norm(q) < R:
                    if c3.n_fields(cell.G) > c3.F_CELL:     # 3x3x3 samples, each within 87 nm of any tile point;
                        clamp = cell.G[c3.G_N + 1 + 12 * c3.F_CELL + 9]    # the field is exact to `clamp` outside
                        g = lo + TILE * np.array([[i, j, k] for i in (0, .5, 1) for j in (0, .5, 1) for k in (0, .5, 1)])
                        if max(c3.outer_depth(p[0], p[1], p[2], cell.G) for p in g) <= -min(clamp - 1.0, 88.0):
                            continue
                    out.append((ix, iy, iz))
    return out


def tile_name(t):
    return f"t_{t[0]:+03d}_{t[1]:+03d}_{t[2]:+03d}"


@nb.njit(cache=True)
def _voxel_regions(lo, n, h, G, pores, pidx, PT, PQ, PC, band_m, band_w, wall_depth):
    """label + region membership bits for voxel centres (serial)."""
    lab = np.zeros((n, n, n), np.int8)
    bits = np.zeros((n, n, n), np.int32)
    out = np.zeros(c3.NMEM)
    for i in range(n):
        x = lo[0] + (i + 0.5) * h
        for j in range(n):
            y = lo[1] + (j + 0.5) * h
            for k in range(n):
                z = lo[2] + (k + 0.5) * h
                l = c3.label_of(x, y, z, G, pores, pidx, PT, PQ, PC, out)
                lab[i, j, k] = l
                if l == c3.CYTO:
                    c3.mem_sdfs(x, y, z, G, pores, pidx, PT, PQ, PC, out)
                    for b in range(band_m.shape[0]):
                        if abs(out[band_m[b]]) < band_w[b]:
                            bits[i, j, k] |= 1 << b
                if l == c3.MANNAN or l == c3.GLUCAN:
                    if out[c3.M_OUTER] < wall_depth:      # label_of filled out
                        bits[i, j, k] |= 1 << 8
    return lab, bits


@nb.njit(cache=True)
def _mem_lattice(lo, n, h, G, pores, pidx, PT, PQ, PC):
    g = np.zeros((c3.NMEM, n, n, n), np.float32)
    out = np.zeros(c3.NMEM)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c3.mem_sdfs(lo[0] + i * h, lo[1] + j * h, lo[2] + k * h, G, pores, pidx, PT, PQ, PC, out)
                for m in range(c3.NMEM):
                    g[m, i, j, k] = out[m]
    return g


BAND_ORDER = [R_PM_PERI, R_NE_PERI, R_ER_PERI, R_LD_PERI]


def region_masks(lab, bits):
    """dict region -> boolean voxel mask (volume regions)."""
    m = {}
    for r in range(MEM_REGION0):
        mm = np.zeros(lab.shape, bool)
        for l in range(15):
            if REG_MASK[r] >> l & 1:
                mm |= lab == l
        if r in BANDS:
            mm &= (bits >> BAND_ORDER.index(r) & 1).astype(bool)
        if r == R_WALL:
            mm &= (bits >> 8 & 1).astype(bool)
        m[r] = mm
    return m


def tile_geometry(cell, t):
    from skimage.measure import marching_cubes
    lo = np.array(t, float) * TILE
    a = cell.args(lo, lo + TILE, margin=80.0)
    n = int(TILE / VOX)
    band_m = np.array([BANDS[r][0] for r in BAND_ORDER], np.int64)
    band_w = np.array([BANDS[r][1] for r in BAND_ORDER], np.float64)
    lab, bits = _voxel_regions(lo, n, VOX, *a, band_m, band_w, WALL_DEPTH)
    masks = region_masks(lab, bits)
    vol = np.zeros(NREG)
    for r, mm in masks.items():
        vol[r] = mm.sum() * VOX ** 3
    labvol = np.bincount(lab.ravel(), minlength=15) * VOX ** 3
    nm = int(TILE / MESH_H) + 1
    g = _mem_lattice(lo, nm, MESH_H, *a)
    verts, faces, mids = [], [], []
    area = np.zeros(c3.NMEM)
    nv = 0
    for m in range(c3.NMEM - 1):        # skip CELL_OUTER
        f = g[m]
        if not (f.min() < 0 < f.max()):
            continue
        v, fc, _, _ = marching_cubes(f, level=0.0, spacing=(MESH_H,) * 3)
        if len(fc) == 0:
            continue
        v = v + lo
        tri = v[fc]
        area[m] = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        verts.append(v.astype(np.float32)); faces.append((fc + nv).astype(np.int32)); mids.append(np.full(len(fc), m, np.int8))
        nv += len(v)
    for m in range(8):
        vol[MEM_REGION0 + m] = area[m]
    V = np.concatenate(verts) if verts else np.zeros((0, 3), np.float32)
    F = np.concatenate(faces) if faces else np.zeros((0, 3), np.int32)
    M = np.concatenate(mids) if mids else np.zeros(0, np.int8)
    return dict(lo=lo, measure=vol, labvol=labvol, verts=V, faces=F, face_mem=M, area=area)


def _worker(args):
    gfile, tl = args
    nb.set_num_threads(1)
    cell = c3.Cell.load(gfile)
    res = []
    for t in tl:
        d = tile_geometry(cell, t)
        np.savez_compressed(f"{GDIR}/{tile_name(t)}.npz", **d)
        res.append((t, d["measure"], d["labvol"], len(d["faces"])))
    return res


def run_geometry(gfile, tiles, nproc=NPROC):
    import multiprocessing as mp
    os.makedirs(GDIR, exist_ok=True)
    chunks = [tiles[i::nproc * 4] for i in range(nproc * 4)]
    with mp.get_context("fork").Pool(nproc) as pool:
        out = []
        for r in pool.imap_unordered(_worker, [(gfile, c) for c in chunks]):
            out += r
    return out
