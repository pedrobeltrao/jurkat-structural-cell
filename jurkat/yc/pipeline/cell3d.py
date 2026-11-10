"""3D whole-cell geometry for the Jurkat T cell (jrc_jurkat-1, cell 1) as signed-distance fields.
Replaces the yeast analytic geometry with the same numba interface used by pack3d.py (mem_sdfs, label_from,
min_mem_dist, in_pore, outer_depth, label_of, label_points, NMEM, HALF, PORE_R, PERI_W, M_NE, M_OUTER, CYTO).
The yeast version is kept as cell3d_yeast.py.

Units nm, origin at the cell centre (bounding-box centre of the refined cell mask), axes x, y, z of the EM label frame.
G layout: [0] radius of a sphere enclosing the cell, [1:G_N] reserved; G[G_N] = number of fields, then 12 header values
per field (origin xyz, spacing h, bricks along x/y/z, voxels per brick, clamp inside, clamp outside, offset of the brick
index, offset of the brick data) and the index/data sections (jgeom_build.bricks; same encoding as yeast cellshape).

Fields (F_*): cell (plasma-membrane midsurface), NE perinuclear midsurface (HMFF fit), mito outer membrane, ER, Golgi,
endolysosomal, nucleolus (v5; EM nucleolus_seg inside the nucleus; label NUCL inside the nucleoplasm, no membrane).
v5 open boundary: cell 1 is truncated by the EM volume's z = 0 face. When G[G_FACE] > 0 the mask fields were extruded
below the face before the SDF was computed (no membrane on the face), and the model domain is clipped at z = G[G_ZFACE]:
label OUTV below it, out[M_OUTER] = min(PM depth, z - z_face) so the face is a plain hard wall (not a membrane). Derived: ONM/INM at +-PERI_W/2 around the NE midsurface; mito inner membrane IMS_W inside the outer one, plus
procedural lamellar cristae when G[G_CRISTAE] points to a cristae section (jcristae.py).

Membrane SDF sign convention (as yeast): negative on the lumenal / non-cytoplasmic side, so the gradient points to the
cytoplasmic (N-) side, = +z of membrane-protein libraries. Mito IM: matrix is the N-side, SDF_IM = -SDF_matrix."""
import numpy as np
import numba as nb

# ----------------------------------------------------------------- labels / membranes
EXT, CYTO, PERI, NUCP, NUCL, IMS, MATRIX, ERL, GOLGI, ENDO, PORE, OUTV = range(12)
LABEL_NAMES = ["exterior", "cytosol", "perinuclear", "nucleoplasm", "nucleolus", "mito_IMS", "mito_matrix",
               "ER_lumen", "golgi_lumen", "endolysosomal_lumen", "pore_channel", "outside_EM_volume"]
M_PM, M_NE, M_ER, M_OM, M_IM, M_GOLGI, M_ENDO, M_OUTER = range(8)
MEM_NAMES = ["PM", "NE", "ER", "MITO_OM", "MITO_IM", "GOLGI", "ENDO", "CELL_OUTER"]
NMEM = 8
HALF = 2.0           # reference bilayer half thickness (pack3d adds it to clearances; see min_mem_dist)
# per-membrane exclusion half thickness = D_P-P / 2 from Pogozheva et al. 2022 SI Table S20 (data/lipids_jurkat.json)
MEM_HALF = np.array([2.275, 2.01, 2.01, 1.975, 1.955, 2.095, 2.245, 2.275])   # PM, NE, ER, MITO_OM, MITO_IM, GOLGI, ENDO, CELL_OUTER
PORE_R = 45.0        # pore-membrane radius at the NE midplane
PERI_W = 34.0        # perinuclear space (ONM-INM midsurface distance)
IMS_W = 8.0          # OM-IM midsurface distance
G_N = 8
G_CRISTAE = 1        # G[1]: offset of the cristae section (jcristae.py), 0 = no cristae (IM = boundary membrane only)
G_FACE = 2           # G[2] > 0: open boundary (EM-volume truncation face) at z = G[G_ZFACE] (model nm); domain is z >= z_face
G_ZFACE = 3
CR_W = 10.0          # crista: half the IM midsurface separation across the intracristal lumen
CR_GAP = 12.0        # lamellae end this far inside the inner boundary membrane (junction ribbons bridge the gap)
CR_JW = 10.0         # junction ribbon half-width
F_CELL, F_NE, F_MITO, F_ER, F_GOLGI, F_ENDO, F_NUCLEOLUS = range(7)
FIELD_NAMES = ["cell", "ne_mid", "mito", "er", "golgi", "endo", "nucleolus"]


# ----------------------------------------------------------------- numba evaluators
@nb.njit(cache=True, fastmath=True)
def n_fields(G):
    return int(G[G_N]) if G.shape[0] > G_N else 0


@nb.njit(cache=True, fastmath=True)
def field_sdf(px, py, pz, G, k):
    """signed distance field k (negative inside): trilinear within a stored brick, +-clamp elsewhere"""
    o = G_N + 1 + 12 * k
    h = G[o + 3]; B = int(G[o + 7])
    fx = (px - G[o]) / h; fy = (py - G[o + 1]) / h; fz = (pz - G[o + 2]) / h
    nbx = int(G[o + 4]); nby = int(G[o + 5]); nbz = int(G[o + 6])
    if fx < 0.0 or fy < 0.0 or fz < 0.0 or fx >= nbx * B or fy >= nby * B or fz >= nbz * B:
        return G[o + 9]
    ix = int(fx); iy = int(fy); iz = int(fz)
    bx = ix // B; by = iy // B; bz = iz // B
    bi = int(G[int(G[o + 10]) + (bx * nby + by) * nbz + bz])
    if bi == -1:
        return -G[o + 8]
    if bi == -2:
        return G[o + 9]
    n1 = B + 1
    tx = fx - ix; ty = fy - iy; tz = fz - iz
    b = int(G[o + 11]) + bi * n1 * n1 * n1 + ((ix - bx * B) * n1 + (iy - by * B)) * n1 + (iz - bz * B)
    sx = n1 * n1; sy = n1
    c00 = G[b] + (G[b + sx] - G[b]) * tx
    c01 = G[b + 1] + (G[b + sx + 1] - G[b + 1]) * tx
    c10 = G[b + sy] + (G[b + sx + sy] - G[b + sy]) * tx
    c11 = G[b + sy + 1] + (G[b + sx + sy + 1] - G[b + sy + 1]) * tx
    c0 = c00 + (c10 - c00) * ty; c1 = c01 + (c11 - c01) * ty
    return c0 + (c1 - c0) * tz


@nb.njit(cache=True, fastmath=True)
def face_dist(pz, G):
    """height above the open (truncation) face; 1e9 when the geometry has no open face"""
    if G[G_FACE] > 0.0:
        return pz - G[G_ZFACE]
    return 1e9


@nb.njit(cache=True, fastmath=True)
def outer_depth(px, py, pz, G):
    """depth below the plasma membrane (negative outside the cell)"""
    return -field_sdf(px, py, pz, G, F_CELL)


@nb.njit(cache=True, fastmath=True)
def in_pore(px, py, pz, pores, rc):
    """index of the pore whose axis (through its centre along the NE normal) is within rc of p, or -1"""
    for k in range(pores.shape[0]):
        vx = px - pores[k, 0]; vy = py - pores[k, 1]; vz = pz - pores[k, 2]
        z = vx * pores[k, 3] + vy * pores[k, 4] + vz * pores[k, 5]
        if z < -150.0 or z > 150.0:
            continue
        rx = vx - z * pores[k, 3]; ry = vy - z * pores[k, 4]; rz = vz - z * pores[k, 5]
        if rx * rx + ry * ry + rz * rz < rc * rc:
            return k
    return -1


@nb.njit(cache=True, fastmath=True)
def crista_lumen(px, py, pz, G, b):
    """signed distance to the intracristal lumen (negative inside) of the mitochondrion at p; b = boundary-IM matrix SDF.
    Section (jcristae.py): n, id-grid origin xyz, spacing, dims xyz, n x 12 params (axis xyz, phase, spacing S, junction
    direction xyz, centroid xyz, enabled), then the dense id grid (-1 = none)."""
    o = int(G[G_CRISTAE])
    n = int(G[o]); h = G[o + 4]; nx = int(G[o + 5]); ny = int(G[o + 6]); nz = int(G[o + 7])
    ix = int(np.floor((px - G[o + 1]) / h)); iy = int(np.floor((py - G[o + 2]) / h)); iz = int(np.floor((pz - G[o + 3]) / h))
    if ix < 0 or iy < 0 or iz < 0 or ix >= nx or iy >= ny or iz >= nz:
        return 1e9
    k = int(G[o + 8 + 12 * n + (ix * ny + iy) * nz + iz])
    if k < 0:
        return 1e9
    q = o + 8 + 12 * k
    if G[q + 11] == 0.0:
        return 1e9
    S = G[q + 4]
    t = G[q] * px + G[q + 1] * py + G[q + 2] * pz - G[q + 3]
    u = t - S * np.floor(t / S + 0.5)
    slab = abs(u) - CR_W
    lam = max(slab, b + CR_GAP)
    w = abs(G[q + 5] * (px - G[q + 8]) + G[q + 6] * (py - G[q + 9]) + G[q + 7] * (pz - G[q + 10])) - CR_JW
    return min(lam, max(slab, w))


@nb.njit(cache=True, fastmath=True)
def mem_sdfs(px, py, pz, G, pores, pidx, PT, PQ, PC, out):
    """Fill out[NMEM] with signed distances to each membrane (negative = lumenal side). Returns the matrix SDF.
    pidx/PT/PQ/PC are unused (kept for the yeast call signature)."""
    dep = outer_depth(px, py, pz, G)
    out[M_PM] = dep
    out[M_OUTER] = min(dep, face_dist(pz, G))          # domain depth: PM or the open face, whichever is nearer
    hp = field_sdf(px, py, pz, G, F_NE)                 # 0 at the perinuclear midsurface, negative toward nucleoplasm
    d = abs(hp) - 0.5 * PERI_W
    rc = PORE_R + 0.5 * PERI_W
    if abs(hp) < rc + PERI_W:
        k = in_pore(px, py, pz, pores, rc)
        if k >= 0:
            vx = px - pores[k, 0]; vy = py - pores[k, 1]; vz = pz - pores[k, 2]
            z = vx * pores[k, 3] + vy * pores[k, 4] + vz * pores[k, 5]
            rho = np.sqrt(max(vx * vx + vy * vy + vz * vz - z * z, 0.0))
            d = np.sqrt((rho - rc) ** 2 + hp * hp) - 0.5 * PERI_W
    out[M_NE] = d
    om = field_sdf(px, py, pz, G, F_MITO)
    out[M_OM] = om
    matrix = om + IMS_W
    if G[G_CRISTAE] > 0.0 and matrix < CR_W:
        matrix = max(matrix, -crista_lumen(px, py, pz, G, matrix))
    out[M_IM] = -matrix
    out[M_ER] = field_sdf(px, py, pz, G, F_ER)
    out[M_GOLGI] = field_sdf(px, py, pz, G, F_GOLGI)
    out[M_ENDO] = field_sdf(px, py, pz, G, F_ENDO)
    return matrix


@nb.njit(cache=True, fastmath=True)
def label_from(px, py, pz, G, pores, out, matrix):
    """Compartment label given precomputed membrane SDFs (out) and matrix SDF."""
    if G[G_FACE] > 0.0 and pz < G[G_ZFACE]:
        return OUTV
    if out[M_OUTER] < 0:
        return EXT
    if out[M_NE] < 0:
        return PERI
    hp = field_sdf(px, py, pz, G, F_NE)
    if hp < -0.5 * PERI_W:
        if hp > -PERI_W - 8.0 and in_pore(px, py, pz, pores, PORE_R + 0.5 * PERI_W) >= 0:
            return PORE
        if n_fields(G) > F_NUCLEOLUS and field_sdf(px, py, pz, G, F_NUCLEOLUS) < 0:
            return NUCL
        return NUCP
    if hp < 0:                                           # pore channel between midplane and ONM
        return PORE
    if out[M_OM] < 0:
        return MATRIX if matrix < 0 else IMS
    if out[M_ER] < 0:
        return ERL
    if out[M_GOLGI] < 0:
        return GOLGI
    if out[M_ENDO] < 0:
        return ENDO
    return CYTO


@nb.njit(cache=True, fastmath=True)
def label_of(px, py, pz, G, pores, pidx, PT, PQ, PC, out):
    matrix = mem_sdfs(px, py, pz, G, pores, pidx, PT, PQ, PC, out)
    return label_from(px, py, pz, G, pores, out, matrix)


@nb.njit(cache=True, fastmath=True)
def min_mem_dist(out, skip):
    """smallest membrane clearance, corrected for each membrane's thickness: min_m(|d_m| - MEM_HALF[m]) + HALF,
    so pack3d's tests 'min_mem_dist(...) < r + HALF' mean 'closer than r to the membrane surface'.
    M_OUTER (= PM depth, or the open face where that is nearer) acts as a hard wall, except for PM proteins (skip = M_PM),
    whose face clearance pack3d.check_membrane tests separately."""
    d = 1e9
    for m in range(NMEM):
        if m != skip and not (m == M_OUTER and skip == M_PM):
            d = min(d, abs(out[m]) - MEM_HALF[m])
    return d + HALF


@nb.njit(cache=True, parallel=True)
def label_points(P, G, pores, pidx, PT, PQ, PC):
    lab = np.zeros(P.shape[0], np.int8)
    for i in nb.prange(P.shape[0]):
        out = np.zeros(NMEM)
        lab[i] = label_of(P[i, 0], P[i, 1], P[i, 2], G, pores, pidx, PT, PQ, PC, out)
    return lab


@nb.njit(cache=True, parallel=True)
def mem_points(P, G, pores, pidx, PT, PQ, PC):
    g = np.zeros((P.shape[0], NMEM), np.float32)
    for i in nb.prange(P.shape[0]):
        out = np.zeros(NMEM)
        mem_sdfs(P[i, 0], P[i, 1], P[i, 2], G, pores, pidx, PT, PQ, PC, out)
        for m in range(NMEM):
            g[i, m] = out[m]
    return g


# ----------------------------------------------------------------- python container
def pack_fields(R, fields):
    """G with the enclosing radius R and the distance fields appended (list of (header, index, data), order F_*)"""
    G = [float(R)] + [0.0] * (G_N - 1)
    head = len(G) + 1 + 12 * len(fields)
    secs, off, hdr = [], head, []
    for h, idx, data in fields:
        hdr += list(h) + [off, off + len(idx)]
        secs += [idx, data]; off += len(idx) + len(data)
    return np.concatenate([np.array(G + [len(fields)] + hdr, np.float64)] + secs)


class Cell:
    """geometry container: G (fields), pores (centre on the perinuclear midsurface + outward normal), empty prims"""
    def __init__(self, G, pores=None):
        self.G = np.asarray(G, np.float64)
        self.pores = np.zeros((0, 6)) if pores is None else np.asarray(pores, np.float64).reshape(-1, 6)
        self.PT = np.zeros(0, np.int64); self.PQ = np.zeros((0, 8)); self.PC = np.zeros(0, np.int64)
        self.pidx = np.zeros(0, np.int64)

    @classmethod
    def load(cls, fn):
        z = np.load(fn)
        return cls(z["G"], z["pores"])

    def labels(self, P):
        return label_points(np.asarray(P, np.float64), self.G, self.pores, self.pidx, self.PT, self.PQ, self.PC)

    def mems(self, P):
        return mem_points(np.asarray(P, np.float64), self.G, self.pores, self.pidx, self.PT, self.PQ, self.PC)
