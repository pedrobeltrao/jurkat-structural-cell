"""3D whole-cell geometry for a haploid S. cerevisiae cell as analytic signed-distance primitives.
Units nm, origin at cell centre. Organelle volumes/areas are data-driven (inventory geo dict); nucleus shape, vacuole
lobes and lipid droplets match statistics of segmented cryo-FIB/SEM cells (organelle_stats.json); the arrangement
is illustrative (no single tomogram is copied).

G (cell constants): [0] cell outer radius, [1] mannan/glucan boundary, [2] plasma membrane, [3:6] nucleus centre,
[6] nucleus equivalent-sphere radius (ONM midsurface), [7:10] nucleolus sphere centre, [10] its radius,
[11:14] nucleus semi-axes (ONM midsurface), [14:23] nucleus rotation rows (local = R (p - centre)).
Pores: per pore the centre on the perinuclear midsurface and the outward NE normal (6 values).

Real shapes (cellshape.py): the cell outline, the nucleus and the vacuoles can instead be signed-distance fields,
appended to G after G_N: G[G_N] = number of fields, then 12 header values per field (grid origin xyz, spacing, bricks
along x/y/z, voxels per brick, clamp inside, clamp outside, offset of the brick index, offset of the brick data) and
the index and data sections (field_sdf). With a cell field, G[0] is the radius of a sphere enclosing the cell and
G[0] - G[1], G[0] - G[2] are the mannan-layer and wall depths below the outer surface.

Membrane SDF sign convention: negative on the *lumenal / non-cytoplasmic* side; the gradient therefore
points to the cytoplasmic (N-) side, which is +z in the membrane-protein library frame.
For the mitochondrial inner membrane the matrix is the N-side, so SDF_IM = -SDF_matrix."""
import json
import numpy as np
import numba as nb

# ----------------------------------------------------------------- labels / membranes
EXT, MANNAN, GLUCAN, CYTO, CER, PERI, NUCP, NUCL, VAC, IMS, MATRIX, LD, GOLGI, PEROX, PORE = range(15)
LABEL_NAMES = ["exterior", "wall_mannan", "wall_glucan", "cytosol", "cER_lumen", "perinuclear", "nucleoplasm",
               "nucleolus", "vacuole", "mito_IMS", "mito_matrix", "lipid_droplet", "golgi_endosome",
               "peroxisome", "pore_channel"]
M_PM, M_ER, M_NE, M_VAC, M_OM, M_IM, M_GOLGI, M_PEROX, M_LD, M_OUTER = range(10)
MEM_NAMES = ["PM", "ER", "NE", "VAC", "MITO_OM", "MITO_IM", "GOLGI", "PEROX", "LD", "CELL_OUTER"]
NMEM = 10
# primitive types and classes
T_SPHERE, T_CAPSULE, T_PILL, T_CERPATCH = range(4)
C_VAC, C_LD, C_GOLGI, C_PEROX, C_OM, C_IM, C_CRISTA, C_CER = range(8)
HALF = 2.0           # bilayer half thickness used for exclusion (as in the column model)
PORE_R = 45.0        # pore-membrane radius at the NE midplane
PERI_W = 34.0        # perinuclear space (ONM-INM midsurface distance)
IMS_W = 8.0          # OM-IM midsurface distance
MITO_R = 150.0
CRISTA_HALF = 9.0    # crista membrane half-separation = radius of the rim tube (fits ATP synthase dimers, see atp_dimers.py)
CER_GAP = 27.0       # PM midsurface -> outer cER membrane midsurface
CER_LUMEN = 20.0     # cER membrane midsurface separation


G_N = 23                 # analytic constants in G; distance fields follow
F_CELL, F_VAC, F_NUC = 0, 1, 2


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
def outer_depth(px, py, pz, G):
    """depth below the outer cell surface (negative outside the cell)"""
    if n_fields(G) > F_CELL:
        return -field_sdf(px, py, pz, G, F_CELL)
    return G[0] - np.sqrt(px * px + py * py + pz * pz)


@nb.njit(cache=True, fastmath=True)
def nuc_sdf(px, py, pz, G):
    if n_fields(G) > F_NUC:
        return field_sdf(px, py, pz, G, F_NUC)
    return nuc_ellipsoid(px, py, pz, G)


@nb.njit(cache=True, fastmath=True)
def nuc_ellipsoid(px, py, pz, G):
    """signed distance to the ONM midsurface (ellipsoid; negative inside), first-order approximation
    k0 (k0 - 1) / k1: exact on the surface, error ~ d^2 x curvature difference nearby (< 0.5 nm across the NE)"""
    dx = px - G[3]; dy = py - G[4]; dz = pz - G[5]
    lx = (G[14] * dx + G[15] * dy + G[16] * dz) / G[11]
    ly = (G[17] * dx + G[18] * dy + G[19] * dz) / G[12]
    lz = (G[20] * dx + G[21] * dy + G[22] * dz) / G[13]
    k0 = np.sqrt(lx * lx + ly * ly + lz * lz)
    k1 = np.sqrt((lx / G[11]) ** 2 + (ly / G[12]) ** 2 + (lz / G[13]) ** 2)
    if k1 < 1e-12:
        return -min(G[11], min(G[12], G[13]))
    return k0 * (k0 - 1.0) / k1


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
def sdf_cer(px, py, pz, q, pm, Rp):
    """cortical-ER patch: a sheet at constant depth (Rp - q[5]) below the plasma membrane (pm: signed PM distance,
    positive in the cytoplasm), within the angular cap q[0:3], cos q[3], q[4]; half-thickness q[6]"""
    r = np.sqrt(px * px + py * py + pz * pz)
    c = (px * q[0] + py * q[1] + pz * q[2]) / max(r, 1e-9)
    c = min(1.0, max(-1.0, c))
    s = max(np.arccos(c) - q[4], 0.0) * r
    dr = (Rp - q[5]) - pm
    return np.sqrt(s * s + dr * dr) - q[6]


@nb.njit(cache=True, fastmath=True)
def sdf_prim(px, py, pz, t, q):
    if t == T_SPHERE:
        dx = px - q[0]; dy = py - q[1]; dz = pz - q[2]
        return np.sqrt(dx * dx + dy * dy + dz * dz) - q[3]
    if t == T_CAPSULE:
        ax, ay, az = px - q[0], py - q[1], pz - q[2]
        bx, by, bz = q[3] - q[0], q[4] - q[1], q[5] - q[2]
        h = (ax * bx + ay * by + az * bz) / (bx * bx + by * by + bz * bz)
        h = min(max(h, 0.0), 1.0)
        dx = ax - bx * h; dy = ay - by * h; dz = az - bz * h
        return np.sqrt(dx * dx + dy * dy + dz * dz) - q[6]
    if t == T_PILL:   # points within q[7] of a flat disk (centre q0-2, normal q3-5, radius q6)
        vx, vy, vz = px - q[0], py - q[1], pz - q[2]
        z = vx * q[3] + vy * q[4] + vz * q[5]
        rx, ry, rz = vx - z * q[3], vy - z * q[4], vz - z * q[5]
        rho = np.sqrt(rx * rx + ry * ry + rz * rz)
        s = max(rho - q[6], 0.0)
        return np.sqrt(s * s + z * z) - q[7]
    # T_CERPATCH: spherical shell patch (axis q0-2, cos half-angle q3, half-angle q4, mid radius q5, half thick q6)
    r = np.sqrt(px * px + py * py + pz * pz)
    c = (px * q[0] + py * q[1] + pz * q[2]) / max(r, 1e-9)
    c = min(max(c, -1.0), 1.0)
    th = np.arccos(c)
    s = max(th - q[4], 0.0) * r
    dr = r - q[5]
    return np.sqrt(s * s + dr * dr) - q[6]


@nb.njit(cache=True, fastmath=True)
def mem_sdfs(px, py, pz, G, pores, pidx, PT, PQ, PC, out):
    """Fill out[NMEM] with signed distances to each membrane (negative = lumenal side).
    Returns the mito-matrix SDF (negative inside matrix)."""
    dep = outer_depth(px, py, pz, G)
    out[M_PM] = dep - (G[0] - G[2])
    out[M_OUTER] = dep
    # nuclear envelope lumen with toroidal pores (hp: offset from the perinuclear midsurface)
    hp = nuc_sdf(px, py, pz, G) + 0.5 * PERI_W
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
    for m in (M_ER, M_VAC, M_OM, M_GOLGI, M_PEROX, M_LD):
        out[m] = 1e9
    if n_fields(G) > F_VAC:
        out[M_VAC] = field_sdf(px, py, pz, G, F_VAC)
    im = 1e9; cr = 1e9
    for ii in range(pidx.shape[0]):
        i = pidx[ii]
        if PT[i] == T_CERPATCH:
            d = sdf_cer(px, py, pz, PQ[i], out[M_PM], G[2])
        else:
            d = sdf_prim(px, py, pz, PT[i], PQ[i])
        c = PC[i]
        if c == C_CER:
            out[M_ER] = min(out[M_ER], d)
        elif c == C_VAC:
            out[M_VAC] = min(out[M_VAC], d)
        elif c == C_OM:
            out[M_OM] = min(out[M_OM], d)
        elif c == C_IM:
            im = min(im, d)
        elif c == C_CRISTA:
            cr = min(cr, d)
        elif c == C_GOLGI:
            out[M_GOLGI] = min(out[M_GOLGI], d)
        elif c == C_PEROX:
            out[M_PEROX] = min(out[M_PEROX], d)
        elif c == C_LD:
            out[M_LD] = min(out[M_LD], d)
    matrix = max(im, -cr)
    out[M_IM] = -matrix
    return matrix


@nb.njit(cache=True, fastmath=True)
def label_from(px, py, pz, G, pores, out, matrix):
    """Compartment label given precomputed membrane SDFs (out) and matrix SDF."""
    if out[M_OUTER] < 0:
        return EXT
    if out[M_OUTER] < G[0] - G[1]:
        return MANNAN
    if out[M_PM] < 0:
        return GLUCAN
    if out[M_ER] < 0:
        return CER
    if out[M_NE] < 0:
        return PERI
    ns = nuc_sdf(px, py, pz, G)
    if ns < -0.5 * PERI_W:
        if ns > -PERI_W - 8.0:
            if in_pore(px, py, pz, pores, PORE_R + 0.5 * PERI_W) >= 0:
                return PORE
        wx, wy, wz = px - G[7], py - G[8], pz - G[9]
        if ns < -PERI_W and wx * wx + wy * wy + wz * wz < G[10] * G[10]:
            return NUCL
        return NUCP
    if ns < 0:         # pore channel between midplane and ONM
        return PORE
    if out[M_OM] < 0:
        return MATRIX if matrix < 0 else IMS
    if out[M_VAC] < 0:
        return VAC
    if out[M_LD] < 0:
        return LD
    if out[M_GOLGI] < 0:
        return GOLGI
    if out[M_PEROX] < 0:
        return PEROX
    return CYTO


@nb.njit(cache=True, fastmath=True)
def label_of(px, py, pz, G, pores, pidx, PT, PQ, PC, out):
    matrix = mem_sdfs(px, py, pz, G, pores, pidx, PT, PQ, PC, out)
    return label_from(px, py, pz, G, pores, out, matrix)


@nb.njit(cache=True, fastmath=True)
def min_mem_dist(out, skip):
    d = 1e9
    for m in range(NMEM):
        if m != skip:
            d = min(d, abs(out[m]))
    return d


@nb.njit(cache=True, parallel=True)
def label_grid(xs, ys, zs, G, pores, pidx, PT, PQ, PC):
    lab = np.zeros((xs.shape[0], ys.shape[0], zs.shape[0]), np.int8)
    for i in nb.prange(xs.shape[0]):
        out = np.zeros(NMEM)
        for j in range(ys.shape[0]):
            for k in range(zs.shape[0]):
                lab[i, j, k] = label_of(xs[i], ys[j], zs[k], G, pores, pidx, PT, PQ, PC, out)
    return lab


@nb.njit(cache=True, parallel=True)
def mem_grid(xs, ys, zs, G, pores, pidx, PT, PQ, PC):
    g = np.zeros((NMEM, xs.shape[0], ys.shape[0], zs.shape[0]), np.float32)
    for i in nb.prange(xs.shape[0]):
        out = np.zeros(NMEM)
        for j in range(ys.shape[0]):
            for k in range(zs.shape[0]):
                mem_sdfs(xs[i], ys[j], zs[k], G, pores, pidx, PT, PQ, PC, out)
                for m in range(NMEM):
                    g[m, i, j, k] = out[m]
    return g


@nb.njit(cache=True, parallel=True)
def label_points(P, G, pores, pidx, PT, PQ, PC):
    lab = np.zeros(P.shape[0], np.int8)
    for i in nb.prange(P.shape[0]):
        out = np.zeros(NMEM)
        lab[i] = label_of(P[i, 0], P[i, 1], P[i, 2], G, pores, pidx, PT, PQ, PC, out)
    return lab


# ----------------------------------------------------------------- python geometry container
def pack_fields(G, fields):
    """G (analytic constants) with distance fields appended; fields: list of cellshape.bricks() outputs, in the order
    F_CELL, F_VAC, F_NUC"""
    G = list(np.asarray(G, np.float64)[:G_N])
    head = len(G) + 1 + 12 * len(fields)
    secs, off = [], head
    hdr = []
    for h, idx, data in fields:
        hdr += list(h) + [off, off + len(idx)]
        secs += [idx, data]; off += len(idx) + len(data)
    return np.concatenate([np.array(G + [len(fields)] + hdr, np.float64)] + secs)


class Cell:
    def __init__(self, d, fields=None):
        self.d = d
        self.G = np.array(d["G"], np.float64)
        self.fields = fields                      # list of (header, index, data) or None (analytic cell)
        if fields:
            self.G = pack_fields(self.G, fields)
        self.pores = np.array(d["pores"], np.float64).reshape(-1, 6)    # centre (perinuclear midsurface), normal
        self.PT = np.array(d["PT"], np.int64)
        self.PQ = np.array(d["PQ"], np.float64).reshape(-1, 8)
        self.PC = np.array(d["PC"], np.int64)
        self.bc = np.array(d["bc"], np.float64).reshape(-1, 3)   # bounding spheres of prims
        self.br = np.array(d["br"], np.float64)
        self.npc_pos = np.array(d["npc_pos"], np.float64).reshape(-1, 3)

    def to_json(self, fn):
        import os
        d = dict(self.d)
        if self.fields:
            ff = os.path.splitext(fn)[0] + "_fields.npz"
            np.savez_compressed(ff, **{f"{n}{k}": a for k, (h, i, x) in enumerate(self.fields) for n, a in (("h", h), ("i", i), ("x", x))})
            d["fields"] = os.path.basename(ff)
        json.dump(d, open(fn, "w"))

    @staticmethod
    def load(fn):
        import os
        d = json.load(open(fn))
        fields = None
        if d.get("fields"):
            z = np.load(os.path.join(os.path.dirname(fn), d["fields"]))
            fields = [(z[f"h{k}"], z[f"i{k}"], z[f"x{k}"]) for k in range(len([f for f in z.files if f[0] == "h"]))]
        return Cell(d, fields)

    def select(self, lo, hi, margin=80.0):
        """Indices of prims / pores whose bounding sphere intersects box [lo,hi] grown by margin."""
        lo = np.asarray(lo) - margin; hi = np.asarray(hi) + margin
        q = np.clip(self.bc, lo, hi)
        pidx = np.where(((q - self.bc) ** 2).sum(1) <= self.br ** 2)[0].astype(np.int64)
        # pores: pore centre at NE midsurface
        pc = self.pores[:, :3]
        q = np.clip(pc, lo, hi)
        pk = np.where(((q - pc) ** 2).sum(1) <= (PORE_R + PERI_W + 40) ** 2)[0]
        return pidx, np.ascontiguousarray(self.pores[pk])

    def args(self, lo, hi, margin=80.0):
        pidx, pores = self.select(lo, hi, margin)
        return self.G, pores, pidx, self.PT, self.PQ, self.PC

    def label(self, P, margin=80.0):
        P = np.atleast_2d(P)
        lo, hi = P.min(0), P.max(0)
        a = self.args(lo, hi, margin)
        out = np.zeros(NMEM)
        return np.array([label_of(p[0], p[1], p[2], *a, out) for p in P], np.int8)


# ----------------------------------------------------------------- builder
def _rand_unit(rng, n=None):
    v = rng.normal(size=(n or 1, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v if n else v[0]


@nb.njit(cache=True)
def _many(P, G, which):
    out = np.empty(P.shape[0])
    for n in range(P.shape[0]):
        if which == 0:
            out[n] = nuc_sdf(P[n, 0], P[n, 1], P[n, 2], G)
        else:
            out[n] = outer_depth(P[n, 0], P[n, 1], P[n, 2], G)
    return out


def nuc_sdf_np(P, G):
    """nuc_sdf for an array of points (n x 3)"""
    return _many(np.atleast_2d(np.asarray(P, np.float64)), np.asarray(G, np.float64), 0)


def ray_depth(G, u, depth, depth_fn=None):
    """radius along the unit direction u from the cell centre at which the depth below the outer cell surface is
    depth (exact for a spherical cell). The cell field in G is clamped (cellshape.BANDS); for deeper points pass
    depth_fn (e.g. the builder's dense grid)."""
    G = np.asarray(G, np.float64); u = np.asarray(u, np.float64) / np.linalg.norm(u)
    if n_fields(G) <= F_CELL:
        return G[0] - depth
    if depth_fn is None:
        assert depth < G[G_N + 1 + 12 * F_CELL + 8] - 5.0, "depth beyond the stored band: pass depth_fn"
        depth_fn = lambda p: outer_depth(p[0], p[1], p[2], G)
    lo_, hi_ = 0.0, G[0] + 50.0
    for _ in range(40):
        m = 0.5 * (lo_ + hi_)
        if float(np.ravel(depth_fn(m * u))[0]) > depth:
            lo_ = m
        else:
            hi_ = m
    return 0.5 * (lo_ + hi_)


def _grad(f, p, e=1.0):
    g = np.array([f(*(p + e * u)) - f(*(p - e * u)) for u in np.eye(3)])
    return g / max(np.linalg.norm(g), 1e-12)


def nuc_surface(G, u, offset=0.0):
    """point on the nuclear envelope along direction u from the nucleus centre, moved by offset (nm) along the
    outward normal (e.g. -PERI_W / 2: perinuclear midsurface), and that normal"""
    G = np.asarray(G, np.float64)
    if n_fields(G) > F_NUC:
        u = np.asarray(u, np.float64) / np.linalg.norm(u); c = G[3:6]
        lo_, hi_ = 0.0, 3000.0
        for _ in range(40):
            m = 0.5 * (lo_ + hi_)
            if nuc_sdf(*(c + m * u), G) < 0:
                lo_ = m
            else:
                hi_ = m
        p = c + 0.5 * (lo_ + hi_) * u
        n = _grad(lambda x, y, z: nuc_sdf(x, y, z, G), p)
        return p + offset * n, n
    R = G[14:23].reshape(3, 3); ax = G[11:14]
    ul = R @ (np.asarray(u, np.float64) / np.linalg.norm(u))
    pl = ul / np.sqrt(((ul / ax) ** 2).sum())                 # on the ellipsoid
    nl = pl / ax ** 2; nl /= np.linalg.norm(nl)
    n = R.T @ nl
    return G[3:6] + R.T @ pl + offset * n, n


def _sample_volumes(rng, pooled, target, jitter=0.25):
    """volumes drawn from the pooled real ones (with lognormal smoothing) until they add up to target; the last
    draw is rescaled so the sum is exact. Largest first."""
    out = []
    while sum(out) < target:
        out.append(float(rng.choice(pooled)) * float(np.exp(rng.normal(0.0, jitter))))
    out[-1] -= sum(out) - target
    if out[-1] < 0.25 * min(pooled):
        out.pop(); out = [v * target / sum(out) for v in out]
    return sorted(out, reverse=True)


def organelle_stats(fn=None):
    import os
    fn = fn or os.path.join(os.path.dirname(os.path.abspath(__file__)), "organelle_stats.json")
    st = json.load(open(fn))["cells"]
    frac = lambda key: np.concatenate([np.array(c[key]) / c["cell_volume_um3"] for c in st.values()])
    return dict(vacuole=frac("vacuole_volumes_um3"), lipid_droplet=frac("lipid_droplet_volumes_um3"),
                vac_rms_rel=float(np.median([c["vacuole_cluster_rms_um"] / (3 * c["cell_volume_um3"] / (4 * np.pi)) ** (1 / 3)
                                             for c in st.values()])))


NUC_AXES = (1.25, 1.10, 1.0)    # nucleus axis ratios: cell 05's nucleus (1.24 : 1.10 : 1) and the in-plane ratio of cell
                                # 04's (1.27; its short axis lies along the milling direction of a truncated cell)
NUC_OFFSET = 0.30               # nucleus centre offset / protoplast radius (two real nuclei: 0.13, 0.29)
VAC_OFFSET = 0.30               # vacuole cluster centre offset, opposite the nucleus (five cells: 0.04-0.33)
VAC_GAP = (30.0, 60.0)          # nm between touching vacuole lobes (real lobes are pressed together)


def build(geo, seed=11, log=print, shapes=None):
    """shapes: cellshape.build() output (real cell outline, nucleus and vacuoles as distance fields); None: an
    analytic cell (sphere, ellipsoidal nucleus, vacuole lobes resampled from the segmented cells)"""
    rng = np.random.default_rng(seed)
    V = geo["volumes_um3"]
    Rcell = geo["cell_radius_um"] * 1000.0
    rn = geo["nucleus_radius_um"] * 1000.0
    WALL = geo["wall_thickness_um"] * 1000.0
    stats = organelle_stats()
    PT, PQ, PC, bc, br = [], [], [], [], []

    def add(t, q, c, cen, rad):
        q = list(q) + [0.0] * (8 - len(q)); PT.append(t); PQ.append(q); PC.append(c); bc.append(list(cen)); br.append(rad)

    occupied = []
    fields = None
    if shapes is None:
        Rp = Rcell - WALL
        side = np.array([-990.0, 160.0, 90.0]); side /= np.linalg.norm(side)
        cn = side * NUC_OFFSET * Rp
        cv0 = -side * VAC_OFFSET * Rp
        # nucleus: ellipsoid of the inventory volume; long axis tangential, middle axis radial
        ax = np.array(NUC_AXES) * rn / np.prod(NUC_AXES) ** (1 / 3)
        e1 = np.cross(side, _rand_unit(rng)); e1 /= np.linalg.norm(e1); e3 = np.cross(e1, side)
        Rn = np.array([e1, side, e3])                       # rows: local x (long), y (middle, radial), z (short)
        G = [Rcell, Rcell - 35.0, Rp, *cn, rn, 0.0, 0.0, 0.0, 0.0, *ax, *Rn.ravel()]
        Gq = np.array(G)
        nuc_d = lambda P: nuc_sdf_np(P, Gq)
        depth = lambda P: Rcell - np.linalg.norm(np.atleast_2d(P), axis=1)
        nuc_ext = ax.max()
    else:
        cellg, nucg, vacg = shapes["cell"], shapes["nucleus"], shapes["vacuole"]
        Rcell = float(np.linalg.norm(shapes["meshes"]["cell"][0], axis=1).max()) + 20.0     # encloses the cell
        Rp = Rcell - WALL
        cn = np.asarray(shapes["nuc_centre"], np.float64); nuc_ext = shapes["nuc_extent"] + 40.0
        G = [Rcell, Rcell - 35.0, Rp, *cn, rn, 0.0, 0.0, 0.0, 0.0, nuc_ext, nuc_ext, nuc_ext, *np.eye(3).ravel()]
        fields = [shapes["bricks"]["cell"], shapes["bricks"]["vacuole"], shapes["bricks"]["nucleus"]]
        Gq = pack_fields(G, fields)
        nuc_d = nucg
        depth = lambda P: -cellg(P)
        vin_ = np.argwhere(vacg.sdf < 0)
        import cellshape
        cv = shapes["lo"] + vin_.mean(0) * cellshape.H              # vacuole centroid
        side = (cn - cv) / np.linalg.norm(cn - cv)
    # nucleolus crescent: sphere centred beyond the NE on the side away from the vacuole, radius solved for
    # 20% of nuclear volume (MC over the nucleoplasm)
    cno = cn + 0.8 * (nuc_surface(Gq, side)[0] - cn)
    box = rng.uniform(-1, 1, (600000, 3)) * nuc_ext + cn
    nb_ = nuc_d(box)
    inn = box[nb_ < -PERI_W]
    vin = (nb_ < -PERI_W).mean() * (2 * nuc_ext) ** 3
    target = 0.2 * 4 / 3 * np.pi * rn ** 3
    lo_, hi_ = 50.0, 2 * rn
    for _ in range(40):
        mid = 0.5 * (lo_ + hi_)
        f = (np.linalg.norm(inn - cno, axis=1) < mid).mean() * vin
        lo_, hi_ = (mid, hi_) if f < target else (lo_, mid)
    rno = 0.5 * (lo_ + hi_)
    G[7:11] = [*cno, rno]
    Gq[7:11] = G[7:11]

    def free(c, r, gap):
        if depth(c)[0] - WALL < r + gap + CER_GAP + CER_LUMEN + 20:
            return False
        if nuc_d(c)[0] < r + gap:
            return False
        if vac_d(c)[0] < r + gap:
            return False
        return all(np.linalg.norm(c - o) > r + ro + gap for o, ro in occupied)

    if shapes is None:
        vac_d = lambda P: np.full(len(np.atleast_2d(P)), 1e9)
        # ---- vacuole: lobes with the size distribution of the segmented cells, grown as touching clusters around a
        # centre opposite the nucleus (real lobes are pressed together and spread over about half the cell radius)
        vols = _sample_volumes(rng, stats["vacuole"] * geo["cell_volume_um3"] * 1e9, V["vacuole"] * 1e9)
        lobes = []
        for v in vols:
            r = (3 * v / (4 * np.pi)) ** (1 / 3)
            for att in range(40000):
                if not lobes:      # the largest lobe: as close to the cluster centre as it fits
                    c = cv0 + _rand_unit(rng) * rng.uniform(0, 0.6 * Rp) * min(1.0, att / 2000)
                elif rng.random() < 0.8:
                    cj, rj = lobes[int(rng.integers(len(lobes)))]
                    c = cj + _rand_unit(rng) * (r + rj + rng.uniform(*VAC_GAP))
                else:
                    c = cv0 + rng.normal(0, 0.35 * Rp, 3)
                if free(c, r, VAC_GAP[0]):
                    break
            else:
                raise RuntimeError("could not place a vacuole lobe")
            lobes.append((c, r)); occupied.append((c, r)); add(T_SPHERE, [*c, r], C_VAC, c, r)
        lr = np.array([r for _, r in lobes]); lv = 4 / 3 * np.pi * lr ** 3; lc = np.array([c for c, _ in lobes])
        vcen = (lc * lv[:, None]).sum(0) / lv.sum()
        rms = np.sqrt((lv * ((lc - vcen) ** 2).sum(1)).sum() / lv.sum())
        log(f"vacuole: {len(lobes)} lobes, largest r {lobes[0][1]:.0f} nm, cluster rms {rms / Rp:.2f} x protoplast radius "
            f"(segmented cells {stats['vac_rms_rel']:.2f})")
        cv = vcen
        vac_d = lambda P: (np.linalg.norm(np.atleast_2d(P)[:, None] - lc[None], axis=2) - lr[None]).min(1)
    else:
        vac_d = vacg

    # ---- NPCs on the NE, >=150 nm apart, uniform over the envelope area
    pores = []
    if shapes is None:
        amax = None
        while len(pores) < geo["n_npc"]:
            s, n = nuc_surface(G, _rand_unit(rng), -0.5 * PERI_W)
            # area element along a random direction ~ |p|^3 / (p . n) relative; accept proportionally
            p = s - cn; w = np.linalg.norm(p) ** 3 / max(np.dot(p, n), 1e-9)
            amax = amax or 1.3 * (ax.max() ** 3 / ax.min())
            if rng.random() * amax > w:
                continue
            if all(np.linalg.norm(s - q[:3]) > 150.0 for q in pores):
                pores.append(np.r_[s, n])
    else:
        nv, nf = shapes["meshes"]["nucleus"]
        tri = nv[nf]; ar = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        cum = np.cumsum(ar) / ar.sum()
        while len(pores) < geo["n_npc"]:
            t = tri[int(np.searchsorted(cum, rng.random()))]
            a_, b_ = rng.random(2)
            if a_ + b_ > 1:
                a_, b_ = 1 - a_, 1 - b_
            p = t[0] + a_ * (t[1] - t[0]) + b_ * (t[2] - t[0])
            n = _grad(lambda x, y, z: nuc_sdf(x, y, z, Gq), p)
            s = p - 0.5 * PERI_W * n
            if all(np.linalg.norm(s - q[:3]) > 150.0 for q in pores):
                pores.append(np.r_[s, n])
    pores = np.array(pores)
    npc_pos = pores[:, :3]

    # ---- mitochondrial network: cortical branched tubules, total length ~ V/(pi r^2)
    Vtarget = V["mitochondrion"] * 1e9
    sph = 4 / 3 * np.pi * MITO_R ** 3
    segs = []
    total = 0.0; vol = 0.0
    tries = 0
    while vol < Vtarget and tries < 5000:
        tries += 1
        parent = -1
        if segs and rng.random() < 0.4:   # branch from an existing segment
            parent = int(rng.integers(len(segs)))
            a, b = segs[parent]
            start = a + (b - a) * rng.uniform(0.3, 0.7)
        else:
            u_ = _rand_unit(rng)                                   # 457-597 nm below the outer surface
            start = u_ * ray_depth(Gq, u_, rng.uniform(457, 597), depth)
        d = _rand_unit(rng); d -= start * np.dot(d, start) / np.dot(start, start); d /= np.linalg.norm(d)
        p = start; nseg = 0
        for k in range(rng.integers(4, 12)):
            seglen = rng.uniform(250, 420)
            d = d + rng.normal(0, 0.3, 3); d -= p * np.dot(d, p) / np.dot(p, p); d /= np.linalg.norm(d)
            q = p + d * seglen
            uq = q / np.linalg.norm(q)                             # keep within 437-637 nm of the outer surface
            q = uq * np.clip(np.linalg.norm(q), ray_depth(Gq, uq, 637.0, depth), ray_depth(Gq, uq, 437.0, depth))
            samp = p + (q - p) * np.linspace(0.0, 1, 10)[:, None]
            okk = (nuc_d(samp) > MITO_R + 40).all() and (vac_d(samp) > MITO_R + 40).all()
            ignore = {parent} if nseg == 0 else {len(segs) - 1}
            if okk and segs:
                A = np.array([s[0] for s in segs]); B = np.array([s[1] for s in segs])
                ab = B - A
                for s in samp[3:]:
                    h = np.clip(((s - A) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
                    dd = np.linalg.norm(s - (A + ab * h[:, None]), axis=1)
                    dd[list(i for i in ignore if i >= 0)] = 1e9
                    if (dd < 2 * MITO_R + 60).any():
                        okk = False; break
            if not okk:
                break
            segs.append((p.copy(), q.copy())); total += np.linalg.norm(q - p); p = q; nseg += 1
            vol += np.pi * MITO_R ** 2 * np.linalg.norm(segs[-1][1] - segs[-1][0])
            if nseg == 1:
                vol += sph * (0.5 if parent < 0 else 0.0) + sph * 0.5   # free end(s) of the new chain/branch
            if vol >= Vtarget:
                break
    log(f"mito: {len(segs)} segments, length {total:.0f} nm, est. volume {vol/1e9:.3f} um3 (target {Vtarget/1e9:.3f})")
    n_cristae = 0
    for a, b in segs:
        cen = (a + b) / 2; rad = np.linalg.norm(b - a) / 2 + MITO_R
        add(T_CAPSULE, [*a, *b, MITO_R], C_OM, cen, rad)
        add(T_CAPSULE, [*a, *b, MITO_R - IMS_W], C_IM, cen, rad)
        ax = (b - a) / np.linalg.norm(b - a); Ls = np.linalg.norm(b - a)
        for s in np.arange(30.0, Ls - 30.0, 42.0):
            e = _rand_unit(rng); e -= ax * np.dot(e, ax); e /= np.linalg.norm(e)
            off = rng.uniform(30, 50); R = (MITO_R - IMS_W + 2.0) - off
            c = a + ax * (s + rng.uniform(-6, 6)) + e * off
            add(T_PILL, [*c, *ax, R, CRISTA_HALF], C_CRISTA, c, R + CRISTA_HALF + 1)
            n_cristae += 1
    for a, b in segs:
        for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 100))):
            occupied.append((a + (b - a) * t, MITO_R))
    log(f"cristae: {n_cristae}")

    def place_sphere(r, cls, gap=40.0, rmin=0, rmax=None):
        for _ in range(20000):
            c = _rand_unit(rng) * rng.uniform(rmin, rmax or Rcell)
            if free(c, r, gap):
                add(T_SPHERE, [*c, r], cls, c, r); occupied.append((c, r)); return c
        raise RuntimeError("could not place sphere")

    # ---- lipid droplets: sizes from the segmented cells (depositor labels). Real cell: the segmented droplets as
    # volume-equivalent spheres at their positions (nudged where the reshaped vacuoles or nucleus now are)
    if shapes is None:
        ldr = [(3 * v / (4 * np.pi)) ** (1 / 3) for v in
               _sample_volumes(rng, stats["lipid_droplet"] * geo["cell_volume_um3"] * 1e9, V["lipid_droplet"] * 1e9)]
        for r in ldr:
            place_sphere(r, C_LD)
    else:
        ldr, moved = [], 0
        for c, r in shapes["droplets"]:
            c = np.asarray(c, np.float64)
            for t in range(4000):
                cc = c if t == 0 else c + _rand_unit(rng) * rng.uniform(0, 20 + 0.3 * t)
                if free(cc, r, 40.0):
                    break
            else:
                cc = None
            if cc is None:
                place_sphere(r, C_LD)
            else:
                add(T_SPHERE, [*cc, r], C_LD, cc, r); occupied.append((cc, r))
            moved += t > 0
            ldr.append(r)
        log(f"lipid droplets: {len(ldr)} segmented droplets, {moved} moved to free space")
    rld = np.median(ldr); nld = len(ldr)
    # ---- peroxisomes
    npx = 4
    rpx = (V["peroxisome"] * 1e9 * 3 / (4 * np.pi * npx)) ** (1 / 3)
    for _ in range(npx):
        place_sphere(rpx, C_PEROX)
    # ---- Golgi cisternae (dispersed, unstacked in S. cerevisiae), endosomes, vesicles
    vg = V["golgi_endosome"] * 1e9
    n_cis, n_end, n_ves = 14, 8, 8
    vol_cis = 0.0
    for _ in range(n_cis):
        R = rng.uniform(170, 260); h = 10.0
        for _ in range(20000):
            c = _rand_unit(rng) * rng.uniform(300, Rcell)
            if free(c, R + h, 40.0):
                break
        nrm = _rand_unit(rng)
        add(T_PILL, [*c, *nrm, R, h], C_GOLGI, c, R + h); occupied.append((c, R + h))
        vol_cis += 2 * np.pi * R * R * h + np.pi ** 2 * h * h * R + 4 / 3 * np.pi * h ** 3
    rves = 35.0
    vol_ves = n_ves * 4 / 3 * np.pi * rves ** 3
    rend = ((vg - vol_cis - vol_ves) * 3 / (4 * np.pi * n_end)) ** (1 / 3)
    for _ in range(n_end):
        place_sphere(rend, C_GOLGI)
    for _ in range(n_ves):
        place_sphere(rves, C_GOLGI, gap=30.0)
    log(f"LD n={nld} median r={rld:.0f}  perox r={rpx:.0f}  endosome r={rend:.0f}  golgi cisternae {n_cis}")

    # ---- cortical ER: spherical shell patches covering geo['cer_cover'] of the PM area
    rc = Rp - CER_GAP - 0.5 * CER_LUMEN
    test = _rand_unit(rng, 200000)
    covered = np.zeros(len(test), bool)
    caps = []
    while covered.mean() < geo["cer_cover"]:
        u = _rand_unit(rng); al = rng.uniform(0.12, 0.32)
        cand = test @ u > np.cos(al)
        # keep patches mostly separate (sheets with fenestrae-like gaps)
        if (cand & covered).mean() > 0.15 * cand.mean():
            continue
        caps.append((u, al)); covered |= cand
    for u, al in caps:
        rr = ray_depth(Gq, u, WALL + CER_GAP + 0.5 * CER_LUMEN, depth)    # the sheet's radius along the patch axis
        add(T_CERPATCH, [*u, np.cos(al), al, rc, 0.5 * CER_LUMEN], C_CER, u * rr, al * rr + 20 + abs(rr - rc) + (150 if fields else 0))
    log(f"cER: {len(caps)} patches, PM coverage {covered.mean():.3f}")

    d = dict(G=G, pores=pores.ravel().tolist(), PT=PT, PQ=np.array(PQ).ravel().tolist(), PC=PC,
             bc=np.array(bc).ravel().tolist(), br=br, npc_pos=npc_pos.ravel().tolist(),
             mito_segments=[[list(a), list(b)] for a, b in segs], seed=seed,
             consts=dict(HALF=HALF, PORE_R=PORE_R, PERI_W=PERI_W, IMS_W=IMS_W, MITO_R=MITO_R,
                         CER_GAP=CER_GAP, CER_LUMEN=CER_LUMEN),
             label_names=LABEL_NAMES, mem_names=MEM_NAMES)
    return Cell(d, fields)
