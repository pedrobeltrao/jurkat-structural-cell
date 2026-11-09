"""Numba random-sequential-addition packer with sphere-tree collision detection on a uniform grid,
periodic in x/y, hard bounds in z. Regions are axis-aligned slabs/boxes with optional exclusion boxes."""
import numpy as np
import numba as nb


@nb.njit(cache=True)
def _cell(c, h, n):
    i = int(np.floor(c / h))
    return i


class Grid:
    def __init__(self, L, Z, h=6.0, cap=30_000_000):
        self.L, self.Z, self.h = L, Z, h
        self.nx = int(np.ceil(L / h)); self.nz = int(np.ceil(Z / h)) + 2
        self.head = -np.ones(self.nx * self.nx * self.nz, np.int64)
        self.nxt = -np.ones(cap, np.int64)
        self.sid = np.zeros(cap, np.int64)
        self.cen = np.zeros((cap // 4, 3), np.float64)
        self.rad = np.zeros(cap // 4, np.float64)
        self.n_ent = np.zeros(1, np.int64); self.n_sph = np.zeros(1, np.int64)


@nb.njit(cache=True)
def _overlap(head, nxt, sid, cen, rad, nx, nz, h, L, P, R, tol):
    for s in range(P.shape[0]):
        x, y, z, r = P[s, 0], P[s, 1], P[s, 2], R[s]
        i0 = int(np.floor((x - r) / h)); i1 = int(np.floor((x + r) / h))
        j0 = int(np.floor((y - r) / h)); j1 = int(np.floor((y + r) / h))
        k0 = max(int(np.floor((z - r) / h)), -1); k1 = min(int(np.floor((z + r) / h)), nz - 2)
        for i in range(i0, i1 + 1):
            ii = i % nx
            for j in range(j0, j1 + 1):
                jj = j % nx
                for k in range(k0, k1 + 1):
                    c = (ii * nx + jj) * nz + (k + 1)
                    e = head[c]
                    while e >= 0:
                        q = sid[e]
                        dx = x - cen[q, 0]; dx -= L * np.round(dx / L)
                        dy = y - cen[q, 1]; dy -= L * np.round(dy / L)
                        dz = z - cen[q, 2]
                        rr = r + rad[q] - tol
                        if dx * dx + dy * dy + dz * dz < rr * rr:
                            return True
                        e = nxt[e]
    return False


@nb.njit(cache=True)
def _insert(head, nxt, sid, cen, rad, n_ent, n_sph, nx, nz, h, P, R):
    for s in range(P.shape[0]):
        q = n_sph[0]; n_sph[0] += 1
        cen[q, 0] = P[s, 0]; cen[q, 1] = P[s, 1]; cen[q, 2] = P[s, 2]; rad[q] = R[s]
        x, y, z, r = P[s, 0], P[s, 1], P[s, 2], R[s]
        i0 = int(np.floor((x - r) / h)); i1 = int(np.floor((x + r) / h))
        j0 = int(np.floor((y - r) / h)); j1 = int(np.floor((y + r) / h))
        k0 = max(int(np.floor((z - r) / h)), -1); k1 = min(int(np.floor((z + r) / h)), nz - 2)
        for i in range(i0, i1 + 1):
            ii = i % nx
            for j in range(j0, j1 + 1):
                jj = j % nx
                for k in range(k0, k1 + 1):
                    c = (ii * nx + jj) * nz + (k + 1)
                    e = n_ent[0]; n_ent[0] += 1
                    sid[e] = q; nxt[e] = head[c]; head[c] = e


@nb.njit(cache=True)
def _in_region(P, R, zlo, zhi, boxes_in, boxes_out, L):
    """All spheres within z-slab [zlo,zhi]; if boxes_in non-empty, every sphere center must lie inside
    at least one 'in' box (x0,x1,z0,z1, with radius margin); no sphere may touch any 'out' box."""
    for s in range(P.shape[0]):
        x = P[s, 0] % L; z = P[s, 2]; r = R[s]
        if z - r < zlo or z + r > zhi:
            return False
        if boxes_in.shape[0] > 0:
            ok = False
            for b in range(boxes_in.shape[0]):
                if x - r >= boxes_in[b, 0] and x + r <= boxes_in[b, 1] and z - r >= boxes_in[b, 2] and z + r <= boxes_in[b, 3]:
                    ok = True
                    break
            if not ok:
                return False
        for b in range(boxes_out.shape[0]):
            # distance from sphere center to box in (x,z), periodic in x
            for shift in (-L, 0.0, L):
                xx = x + shift
                dx = max(boxes_out[b, 0] - xx, 0.0, xx - boxes_out[b, 1])
                dz = max(boxes_out[b, 2] - z, 0.0, z - boxes_out[b, 3])
                if dx * dx + dz * dz < r * r:
                    return False
    return True


@nb.njit(cache=True)
def _radial_ok(P, R, cx, cy, rmin, L):
    """sphere centers must be at radial distance >= rmin + r from (cx,cy) (e.g., outside an NPC hole)."""
    if rmin <= 0:
        return True
    for s in range(P.shape[0]):
        dx = P[s, 0] - cx; dx -= L * np.round(dx / L)
        dy = P[s, 1] - cy; dy -= L * np.round(dy / L)
        if dx * dx + dy * dy < (rmin + R[s]) ** 2:
            return False
    return True


def random_rotation(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def rot_z(t, flip=False):
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    if flip:
        R = R @ np.diag([1, -1, -1.0])
    return R


class Packer:
    def __init__(self, L, Z, h=6.0, seed=1, tol=0.3):
        self.g = Grid(L, Z, h)
        self.L, self.Z, self.tol = L, Z, tol
        self.rng = np.random.default_rng(seed)
        self.placed = []  # (ing_id, R(3x3), t(3))

    def overlaps(self, P, R):
        g = self.g
        return _overlap(g.head, g.nxt, g.sid, g.cen, g.rad, g.nx, g.nz, g.h, self.L, P, R, self.tol)

    def insert(self, P, R):
        g = self.g
        _insert(g.head, g.nxt, g.sid, g.cen, g.rad, g.n_ent, g.n_sph, g.nx, g.nz, g.h, P, R)

    def try_place(self, ing_id, st_c, st_r, region, n, max_tries=400):
        """region: dict(kind='volume'|'membrane', zlo, zhi, boxes_in, boxes_out, z0, flip, hole=(cx,cy,r))"""
        ok = 0
        bin_ = np.asarray(region.get("boxes_in", np.zeros((0, 4))), float).reshape(-1, 4)
        bout = np.asarray(region.get("boxes_out", np.zeros((0, 4))), float).reshape(-1, 4)
        hole = region.get("hole", (0.0, 0.0, 0.0))
        for _ in range(n):
            for t in range(max_tries):
                if region["kind"] == "membrane":
                    Rm = rot_z(self.rng.uniform(0, 2 * np.pi), region.get("flip", False))
                    pos = np.array([self.rng.uniform(0, self.L), self.rng.uniform(0, self.L), region["z0"]])
                else:
                    Rm = random_rotation(self.rng)
                    pos = np.array([self.rng.uniform(0, self.L), self.rng.uniform(0, self.L),
                                    self.rng.uniform(region["zlo"], region["zhi"])])
                P = st_c @ Rm.T + pos
                if region["kind"] == "membrane":
                    if bin_.shape[0] and not _in_region(P[:, :], np.zeros(len(P)), -1e9, 1e9, bin_[:, :], np.zeros((0, 4)), self.L):
                        continue
                    if not _radial_ok(P, st_r, hole[0], hole[1], hole[2], self.L):
                        continue
                else:
                    if not _in_region(P, st_r, region["zlo"], region["zhi"], bin_, bout, self.L):
                        continue
                    if not _radial_ok(P, st_r, hole[0], hole[1], hole[2], self.L):
                        continue
                if self.overlaps(P, st_r):
                    continue
                self.insert(P, st_r)
                self.placed.append((ing_id, Rm, pos))
                ok += 1
                break
        return ok
