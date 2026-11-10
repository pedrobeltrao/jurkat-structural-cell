"""Sanity-check a Jurkat viewer data folder (jenc.py output) with the same decoding rules as the web app.

Decodes tiles/index.json, one tile (YCT1 binary, as worker.ts decodeTile), library/types.json + beads.bin +
spheretree.bin (as data.ts loadLibrary), geometry.json + geometry_fields.bin (as fields.ts loadFields / fieldSdf) and
applies the compartment label rule of geometry.ts / shaders.ts FILL (= cell3d.mem_sdfs + label_from). Prints stats,
the field values and label at the cube centre, re-derives every tile's label_mask (5^3 samples, as jenc.py) and,
with --ref, compares labels at random points with the pipeline's own cell3d evaluation of the full-resolution fields.

Whole-cell builds (jenc_full.py; index.json 'shard'): tiles are read as byte ranges of their block shard (as worker.ts
getTileBytes), types_tiles.bin holds uint32 tile indices, one overview block is decoded (as tiles.ts Overview), and the
label_mask check runs on a random sample of tiles. DATA may be a folder or an http(s) URL (then every read goes
through HTTP, tiles with Range requests, which must answer 206).

usage: check_data.py [DATA_DIR | URL] [--tile ix iy iz] [--ref] [--n 20000] [--sample 300]
python: python (conda env, see README)"""
import os, sys, json, gzip, math, argparse, urllib.request
import numpy as np

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
REGION_NAMES = ["cytosol", "nucleoplasm", "matrix", "ER_lumen", "golgi_lumen", "endo_lumen", "perinuclear", "mem_PM", "mem_NE",
                "mem_ER", "mem_MITO_OM", "mem_MITO_IM", "mem_GOLGI", "mem_ENDO", "chromatin", "polysome_ribosome", "cytoskeleton", "NPC",
                # model v5 (data.ts REGION_NAMES 18-30)
                "nucleolus", "sh_PM_cyto", "sh_PM_exo", "sh_PM_cortex", "sh_NE_lamina", "sh_NE_nuc", "sh_ER_cyto", "sh_ER_lumen",
                "sh_GOLGI_cyto", "sh_GOLGI_lumen", "sh_ENDO_cyto", "sh_ENDO_lumen", "sh_OM_cyto",
                # model v6 (data.ts REGION_NAMES 31; pack region sh_NPC maps to NPC 17)
                "speckle"]
MEM_NAMES = ["PM", "NE", "ER", "MITO_OM", "MITO_IM", "GOLGI", "ENDO"]
LABEL_NAMES = ["exterior", "cytosol", "perinuclear", "nucleoplasm", "nucleolus", "mito_IMS", "mito_matrix", "ER_lumen",
               "golgi_lumen", "endolysosomal_lumen", "pore_channel", "outside_EM_volume", "speckle"]
OUTV, SPK = 11, 12
CR_HDR = 32                          # fields.ts: cristae header words appended to F
TILE, Q_ORG, QSTEP = 200.0, 80.0, 360.0 / 65535.0
OFF_BITS = 26                       # shaders.ts draw-list offset width


def _raw(path, rng=None):
    """bytes of a local file or URL; rng = (offset, n) reads a byte range (HTTP: Range request, must answer 206)"""
    if path.startswith(("http://", "https://")):
        req = urllib.request.Request(path, headers={"Range": f"bytes={rng[0]}-{rng[0] + rng[1] - 1}"} if rng else {})
        with urllib.request.urlopen(req) as r:
            if rng: assert r.status == 206, f"Range request answered {r.status} for {path}"
            return r.read()
    with open(path, "rb") as f:
        if rng: f.seek(rng[0]); return f.read(rng[1])
        return f.read()


SPLIT = {}                            # split.json of the data folder: {path: n_parts} (data.ts fetchBin)


def read(path, rng=None):
    b = None
    for base in SPLIT.get("_base", []):
        if path.startswith(base) and path[len(base):] in SPLIT:
            b = b"".join(_raw(f"{path}.{i:03d}") for i in range(SPLIT[path[len(base):]]))
    if b is None: b = _raw(path, rng)
    return gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b       # the app inflates gzip by its magic bytes


# ------------------------------------------------------------------ fields (fields.ts)
class FArr(np.ndarray):
    """float32 F with the cristae header and speckle section offsets (fields.ts F.crOff, F.spOff)"""
    crOff = 0
    spOff = 0


def load_fields(geo, d):
    fd = geo["fields"]; buf = read(os.path.join(d, fd["file"])); n = len(fd["head"])
    total = 1 + 12 * n + sum(fd["n_index"]) + sum(fd["n_data"])
    if fd.get("total"): total = fd["total"]
    cr = fd.get("cristae"); crOff = -(-total // 16) * 16 if cr else 0
    end = crOff + CR_HDR if cr else total
    sp = (geo.get("speckles") or {}).get("spheres") or []       # v6 (fields.ts: appended at spOff, F[spOff] = n)
    spOff = -(-end // 16) * 16 if sp else 0
    F = np.zeros(spOff + 1 + len(sp) if sp else end, np.float32).view(FArr); F[0] = n; off = 0
    for k in range(n):
        for j in (10, 11):
            assert float(np.float32(fd["head"][k][j])) == fd["head"][k][j], f"field {k} offset {fd['head'][k][j]} not exact in float32"
    for k in range(n):
        F[1 + 12 * k:13 + 12 * k] = fd["head"][k]
        ni, nd = fd["n_index"][k], fd["n_data"][k]
        idx = np.frombuffer(buf, "<i4", ni, off); off += 4 * ni
        F[int(fd["head"][k][10]):int(fd["head"][k][10]) + ni] = idx
        dat = np.frombuffer(buf, "<i2", nd, off); off += 2 * nd; off = (off + 3) // 4 * 4
        d0 = int(fd["head"][k][11]); F[d0:d0 + nd] = dat.astype(np.float32) * np.float32(fd["q"])
    if cr:                                            # fields.ts loadFields: cristae section + header at crOff
        F[cr["off_par"]:cr["off_par"] + cr["n_par"]] = np.frombuffer(buf, "<f4", cr["n_par"], off); off += 4 * cr["n_par"]
        F[cr["off_idx"]:cr["off_idx"] + cr["n_index"]] = np.frombuffer(buf, "<i4", cr["n_index"], off); off += 4 * cr["n_index"]
        F[cr["off_dat"]:cr["off_dat"] + cr["n_data"]] = np.frombuffer(buf, "<i2", cr["n_data"], off); off += 2 * cr["n_data"]; off = (off + 3) // 4 * 4
        c = cr["consts"]
        F[crOff:crOff + 18] = [cr["n"], *cr["origin"], cr["h"], *cr["dims"], cr["B"], *cr["nb"], cr["off_par"], cr["off_idx"], cr["off_dat"],
                               c["CR_W"], c["CR_GAP"], c["CR_JW"]]
    assert off == len(buf), f"geometry_fields.bin: decoded {off} of {len(buf)} bytes"
    F.crOff = crOff if cr else 0
    if sp: F[spOff] = len(sp) // 4; F[spOff + 1:spOff + 1 + len(sp)] = sp
    F.spOff = spOff
    return F



def field_sdf(F, x, y, z, k):
    o = 1 + 12 * k; h = F[o + 3]; B = int(F[o + 7])
    fx, fy, fz = (x - F[o]) / h, (y - F[o + 1]) / h, (z - F[o + 2]) / h
    nbx, nby, nbz = int(F[o + 4]), int(F[o + 5]), int(F[o + 6])
    if fx < 0 or fy < 0 or fz < 0 or fx >= nbx * B or fy >= nby * B or fz >= nbz * B:
        return float(F[o + 9])
    ix, iy, iz = math.floor(fx), math.floor(fy), math.floor(fz)
    bx, by, bz = ix // B, iy // B, iz // B
    bi = int(F[int(F[o + 10]) + (bx * nby + by) * nbz + bz])
    if bi == -1: return -float(F[o + 8])
    if bi == -2: return float(F[o + 9])
    n1 = B + 1; sx, sy = n1 * n1, n1
    tx, ty, tz = fx - ix, fy - iy, fz - iz
    b = int(F[o + 11]) + bi * n1 ** 3 + ((ix - bx * B) * n1 + (iy - by * B)) * n1 + (iz - bz * B)
    c00 = F[b] + (F[b + sx] - F[b]) * tx; c01 = F[b + 1] + (F[b + sx + 1] - F[b + 1]) * tx
    c10 = F[b + sy] + (F[b + sx + sy] - F[b + sy]) * tx; c11 = F[b + sy + 1] + (F[b + sx + sy + 1] - F[b + sy + 1]) * tx
    c0 = c00 + (c10 - c00) * ty; c1 = c01 + (c11 - c01) * ty
    return float(c0 + (c1 - c0) * tz)


# ------------------------------------------------------------------ label rule (geometry.ts / shaders.ts FILL)
class Geo:
    def __init__(self, geo, F):
        self.F = F; c = geo["consts"]; self.PERI, self.IMS, self.PORE_R = c["PERI_W"], c["IMS_W"], c["PORE_R"]
        fc = geo.get("face") or {}; self.face_on = bool(fc.get("on")); self.zf = float(fc.get("z", -1e9)) if self.face_on else -1e9
        self.crOff = getattr(F, "crOff", 0); self.spOff = getattr(F, "spOff", 0); self.nF = int(F[0])
        half = geo["cube_nm"] / 2; rc = self.PORE_R + 0.5 * self.PERI; reach = math.hypot(150, rc) + 2
        P = np.array(geo["pores"], float).reshape(-1, 6)
        self.n_pores_all = len(P)
        self.P = P[np.all(np.abs(P[:, :3]) < half + reach, axis=1)]      # as prepareGeometry

    def in_pore(self, x, y, z, rc):
        for p in self.P:
            v = np.array([x, y, z]) - p[:3]; zz = v @ p[3:]
            if abs(zz) > 150: continue
            r = v - zz * p[3:]
            if r @ r < rc * rc: return float(np.sqrt(r @ r))
        return -1.0

    def in_speckle(self, x, y, z):
        """geometry.ts inSpeckle (= cell3d.in_speckle) on the F section at spOff"""
        if not self.spOff: return -1
        F, o = self.F, self.spOff
        for k in range(int(F[o])):
            q = o + 1 + 4 * k; dx, dy, dz = x - F[q], y - F[q + 1], z - F[q + 2]
            if dx * dx + dy * dy + dz * dz < F[q + 3] * F[q + 3]: return k
        return -1

    def mem_sdfs(self, x, y, z):
        d = [0.0] * 7; rc = self.PORE_R + 0.5 * self.PERI
        d[0] = -field_sdf(self.F, x, y, z, 0)
        hp = field_sdf(self.F, x, y, z, 1); dne = abs(hp) - 0.5 * self.PERI
        if abs(hp) < rc + self.PERI:
            rho = self.in_pore(x, y, z, rc)
            if rho >= 0: dne = math.sqrt((rho - rc) ** 2 + hp * hp) - 0.5 * self.PERI
        d[1] = dne
        om = field_sdf(self.F, x, y, z, 2); d[3] = om; matrix = om + self.IMS
        if self.crOff and matrix < self.F[self.crOff + 15]: matrix = max(matrix, -self.crista_lumen(x, y, z, matrix))
        d[4] = -matrix
        d[2] = field_sdf(self.F, x, y, z, 3); d[5] = field_sdf(self.F, x, y, z, 4); d[6] = field_sdf(self.F, x, y, z, 5)
        return d, matrix, hp

    def crista_lumen(self, x, y, z, b):
        """geometry.ts cristaLumen (= cell3d.crista_lumen) on the F layout of fields.ts"""
        F, o = self.F, self.crOff; h = F[o + 4]
        ix, iy, iz = math.floor((x - F[o + 1]) / h), math.floor((y - F[o + 2]) / h), math.floor((z - F[o + 3]) / h)
        if ix < 0 or iy < 0 or iz < 0 or ix >= F[o + 5] or iy >= F[o + 6] or iz >= F[o + 7]: return 1e9
        B = int(F[o + 8]); bx, by, bz = ix // B, iy // B, iz // B
        slot = int(F[int(F[o + 13]) + (bx * int(F[o + 10]) + by) * int(F[o + 11]) + bz])
        if slot < 0: return 1e9
        k = int(F[int(F[o + 14]) + slot * B ** 3 + ((ix - bx * B) * B + (iy - by * B)) * B + (iz - bz * B)])
        if k < 0: return 1e9
        q = int(F[o + 12]) + 12 * k
        if F[q + 11] == 0: return 1e9
        S = float(F[q + 4]); CR_W, CR_GAP, CR_JW = (float(v) for v in F[o + 15:o + 18])
        t = float(F[q]) * x + float(F[q + 1]) * y + float(F[q + 2]) * z - float(F[q + 3])
        u = t - S * math.floor(t / S + 0.5); slab = abs(u) - CR_W
        lam = max(slab, b + CR_GAP)
        w = abs(float(F[q + 5]) * (x - F[q + 8]) + float(F[q + 6]) * (y - F[q + 9]) + float(F[q + 7]) * (z - F[q + 10])) - CR_JW
        return min(lam, max(slab, w))

    def label(self, x, y, z):
        d, matrix, hp = self.mem_sdfs(x, y, z); PERI = self.PERI
        if self.face_on and z < self.zf: return OUTV
        if d[0] < 0: return 0
        if d[1] < 0: return 2
        if hp < -0.5 * PERI:
            if hp > -PERI - 8 and self.in_pore(x, y, z, self.PORE_R + 0.5 * PERI) >= 0: return 10
            if self.nF > 6 and field_sdf(self.F, x, y, z, 6) < 0: return 4
            if self.in_speckle(x, y, z) >= 0: return SPK
            return 3
        if hp < 0: return 10
        if d[3] < 0: return 6 if matrix < 0 else 5
        if d[2] < 0: return 7
        if d[5] < 0: return 8
        if d[6] < 0: return 9
        return 1


# ------------------------------------------------------------------ tile (worker.ts decodeTile)
def decode_tile(buf):
    h = np.frombuffer(buf, "<u4", 16, 0)
    assert h[0] == 0x31544359, "bad tile magic"
    t = np.frombuffer(buf, "<i4", 3, 4); lo = t * TILE
    nI, nG, nM, nMB, nL, nLB, nV, nT, fl, nX, nXB = (int(x) for x in h[4:15])
    off = 64
    inst = np.frombuffer(buf, "<u4", 4 * nI, off).reshape(-1, 4); off += 16 * nI
    ghost = np.frombuffer(buf, "<u4", 4 * nG, off).reshape(-1, 4); off += 16 * nG
    mrna = []
    if nM:                                            # worker.ts decodePolylines (MR_Q = 0.005 nm)
        lens = np.frombuffer(buf, "<u4", nM, off); off += 4 * nM
        st = np.frombuffer(buf, "<i4", 3 * nM, off).reshape(-1, 3); off += 12 * nM
        dl = np.frombuffer(buf, "i1", 3 * (nMB - nM), off).reshape(-1, 3); off += 3 * (nMB - nM); off = (off + 3) & ~3
        k = 0
        for i in range(nM):
            m = int(lens[i]) - 1
            mrna.append(lo + np.vstack([st[i][None], st[i] + np.cumsum(dl[k:k + m].astype(np.int64), 0)]) * 0.005); k += m
    assert nL == 0, "linker polylines present: not decoded"
    verts = np.frombuffer(buf, "<u2", 3 * nV, off).reshape(-1, 3).astype(float) * QSTEP + (lo - Q_ORG); off += (6 * nV + 3) & ~3
    big = fl & 1
    tris = np.frombuffer(buf, "<u4" if big else "<u2", 3 * nT, off).reshape(-1, 3); off += ((4 if big else 2) * 3 * nT + 3) & ~3
    fmem = np.frombuffer(buf, "u1", nT, off); off += (nT + 3) & ~3
    assert nX == 0
    return dict(t=t, lo=lo, inst=inst, ghost=ghost, mrna=mrna, verts=verts, tris=tris, fmem=fmem, rest=len(buf) - off)


def decode_inst(w, lo):
    w0, w1, w2, w3 = (w[:, i].astype(np.int64) for i in range(4))
    typ = w0 & 0x3FFF; qi = (w0 >> 14) & 3; reg = (w0 >> 16) & 31; flg = w0 >> 21
    p = np.c_[lo[0] - Q_ORG + (w1 & 0xFFFF) * QSTEP, lo[1] - Q_ORG + (w1 >> 16) * QSTEP, lo[2] - Q_ORG + (w2 & 0xFFFF) * QSTEP]
    i16 = lambda x: ((x & 0xFFFF) ^ 0x8000) - 0x8000
    abc = np.c_[i16(w2 >> 16), i16(w3), i16(w3 >> 16)] / 32767 / math.sqrt(2)
    big = np.sqrt(np.maximum(0, 1 - (abc ** 2).sum(1)))
    return typ, qi, reg, flg, p, big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=f"{V}/jurkat/viewer/web/data")
    ap.add_argument("--tile", type=int, nargs=3)
    ap.add_argument("--ref", action="store_true", help="compare labels with cell3d on the full-resolution fields")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--sample", type=int, default=300, help="tiles for the label_mask check when the index is large")
    ap.add_argument("--geom", default=f"{V}/data/derived/jgeom_cell1.npz", help="pipeline geometry for --ref")
    ap.add_argument("--fields-only", action="store_true", help="only geometry.json + geometry_fields.bin (no tiles/library)")
    ap.add_argument("--strat", type=int, default=3000, help="--ref: extra points per rare label (nucleolus, IMS, ...) near them")
    a = ap.parse_args(); d = a.data.rstrip("/")
    try:
        SPLIT.update(json.loads(_raw(f"{d}/split.json"))); SPLIT["_base"] = [d + "/"]
        if len(SPLIT) > 1: print(f"split.json: {({k: v for k, v in SPLIT.items() if k != '_base'})}")
    except Exception:
        pass
    if not a.fields_only:
        # ---- index
        J = json.loads(read(f"{d}/tiles/index.json")); cols = J["tile_columns"]; T = np.array(J["tiles"], np.int64)
        c = {k: i for i, k in enumerate(cols)}
        print(f"index.json: {len(T)} tiles, ix {T[:, 0].min()}..{T[:, 0].max()}, iy {T[:, 1].min()}..{T[:, 1].max()}, iz {T[:, 2].min()}..{T[:, 2].max()}; "
              f"{T[:, c['n_inst']].sum():,} instances, {T[:, c['n_ghost']].sum():,} ghosts, {T[:, c['n_tri']].sum():,} mesh triangles, "
              f"{T[:, c['gz_bytes']].sum() / 1e6:.1f} MB gz; 'wall' key: {'wall' in J}")
        for key, want in (("region_names", REGION_NAMES), ("mem_names", MEM_NAMES), ("label_names", LABEL_NAMES)):
            ok_ = J[key] == want or J[key] == want[:len(J[key])]          # older builds (v4/v5): a prefix of the v6 lists
            print(f"  {key} {('match the app' + ('' if J[key] == want else f' (older build: first {len(J[key])})')) if ok_ else 'DIFFER from the app: ' + str(J[key])}")
        assert T[:, :3].min() >= -64 and T[:, :3].max() <= 63, "tiles outside the app's tile-state grid (data.TGRID 128, -64..63)"
        whole = bool(J.get("whole_cell")); sh = J.get("shard")
        if whole:
            print(f"  whole-cell build: shards {sh['path']} ({sh['block_tiles']}^3 tiles per block), overview {len(J['overview']['blocks'])} blocks")
        # ---- library
        tj = json.loads(read(f"{d}/library/types.json")); types = tj["types"]; nT = len(types)
        beads = np.frombuffer(read(f"{d}/library/beads.bin"), dtype=[("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("k", "u1"), ("p", "u1")])
        st = np.frombuffer(read(f"{d}/library/spheretree.bin"), "<f4").reshape(-1, 4)
        bo = np.array([t["bo"] for t in types]); nb = np.array([t["nb"] for t in types]); so = np.array([t["so"] for t in types]); sn = np.array([t["sn"] for t in types])
        print(f"types.json: {nT} types (type id field 14 bits: {'ok' if nT < 16384 else 'TOO MANY'}), {len(beads):,} beads "
              f"(n_beads {tj['n_beads']:,}), {len(st):,} spheres (n_spheres {tj['n_spheres']:,}); offsets {'fit' if max(len(beads), len(st)) < 2 ** OFF_BITS else 'EXCEED'} 26 bits")
        assert np.all(np.r_[0, np.cumsum(nb)[:-1]] == bo) and np.all(np.r_[0, np.cumsum(sn)[:-1]] == so), "bo/so not cumulative"
        print(f"  beads/type {nb.min()}..{nb.max()}, nucleotide beads {int((beads['k'] == 1).sum()):,}, pLDDT byte values {np.unique(beads['p'])[:5]}, "
              f"plddt_mean non-null: {sum(t['plddt_mean'] is not None for t in types)}; library GPU size {len(beads) * 16 / 2**20:.0f} MB (16 B/bead)")
        ids = [t["id"] for t in types]
        for pat in ("RIBO", "NUCLEOSOME", "H1-", "NPC", "MT_", "F_"):
            m = [i for i in ids if (i == pat if pat in ("NPC", "NUCLEOSOME") else i.startswith(pat))]
            print(f"  special '{pat}': {m[:6]}{' ...' if len(m) > 6 else ''}")
        print(f"  src values: {sorted(set(t['src'] for t in types))}; copies total {sum(t['copies'] for t in types):,}")
        tt = read(f"{d}/types_tiles.bin"); toff = np.frombuffer(tt, "<u4", nT + 1); tlist = np.frombuffer(tt, "<u4" if J.get("types_tiles_u32") else "<u2", offset=4 * (nT + 1))
        assert toff[-1] == len(tlist) and tlist.max(initial=0) < len(T), "types_tiles.bin inconsistent"
        print(f"types_tiles.bin: {len(tlist):,} (type, tile) entries, consistent")
        # ---- one tile
        if a.tile: ti = int(np.where(np.all(T[:, :3] == a.tile, axis=1))[0][0])
        else: ti = int(np.argmin(np.abs(T[:, :3] + 0.5).sum(1)))            # a tile at the cube centre
        tl = T[ti]
        if sh:                                            # worker.ts getTileBytes: byte range of the block shard
            b = sh["block_tiles"]; sp = sh["path"].format(bx=tl[0] // b, by=tl[1] // b, bz=tl[2] // b)
            D = decode_tile(read(f"{d}/{sp}", (int(tl[c["off"]]), int(tl[c["gz_bytes"]]))))
            print(f"tile {tuple(tl[:3])} read as bytes {tl[c['off']]}+{tl[c['gz_bytes']]} of {sp}")
        else:
            D = decode_tile(read(f"{d}/tiles/{tl[0]}_{tl[1]}_{tl[2]}.bin.gz"))
        typ, qi, reg, flg, p, big = decode_inst(D["inst"], D["lo"])
        print(f"tile {tuple(tl[:3])}: {len(D['inst']):,} instances (index {tl[c['n_inst']]:,}), {len(D['ghost'])} ghosts, {len(D['verts'])} verts, "
              f"{len(D['tris'])} tris, {D['rest']} trailing bytes")
        if len(typ):
            print(f"  type ids {typ.min()}..{typ.max()} (of {nT}), near-membrane flag {int((flg & 1).sum())}, "
                  f"positions in tile {np.mean(np.all((p >= D['lo']) & (p < D['lo'] + TILE), axis=1)) * 100:.1f}%")
            h = np.bincount(reg, minlength=len(REGION_NAMES))
            print("  regions: " + ", ".join(f"{REGION_NAMES[i] if i < len(REGION_NAMES) else i}={h[i]}" for i in np.nonzero(h)[0]))
            top = np.bincount(typ, minlength=nT).argsort()[::-1][:5]
            print("  most frequent: " + ", ".join(f"{types[i]['id']}({types[i]['gene']})x{int((typ == i).sum())}" for i in top))
        if len(D["tris"]):
            fm = np.bincount(D["fmem"], minlength=7)
            print("  face_mem: " + ", ".join(f"{MEM_NAMES[i]}={fm[i]}" for i in np.nonzero(fm)[0]) + f"; tri index max {D['tris'].max()} < {len(D['verts'])}")
        if whole:
            tot = sum(t["copies"] for t in types)
            print(f"  index n_inst total {T[:, c['n_inst']].sum():,} vs library copies {tot:,} "
                  f"({'equal' if T[:, c['n_inst']].sum() == tot else 'DIFFERENT'})")
            if D["mrna"]: print(f"  mRNA: {len(D['mrna'])} pieces, max step {max(np.linalg.norm(np.diff(m, axis=0), axis=1).max() for m in D['mrna']):.3f} nm")
            # one overview block (tiles.ts Overview.onLoaded)
            ov = J["overview"]; oc = ov["columns"]; ob = np.array(ov["blocks"])
            k = int(np.argmax(ob[:, oc.index("n_proxy")]))
            B = ob[k]; buf = read(f"{d}/" + ov["path"].format(bx=B[0], by=B[1], bz=B[2]))
            hd = np.frombuffer(buf, "<u4", 2); assert hd[0] == 0x314F4359, "bad overview magic"
            tl_ = np.frombuffer(buf, "<i4", 4 * int(hd[1]), 8).reshape(-1, 4); rec = np.frombuffer(buf, "<u4", offset=8 + 16 * int(hd[1])).reshape(-1, 4)
            assert len(rec) == tl_[:, 3].sum() == B[oc.index("n_proxy")], "overview record count"
            keys = {tuple(r[:3]) for r in T[:, :3].tolist()}
            miss = sum(tuple(r[:3]) not in keys for r in tl_.tolist())
            w = rec[:, 3]; xyz = rec[:, :3].copy().view("<f4")
            inside = np.all((xyz >= B[:3] * ov["block_nm"]) & (xyz < (B[:3] + 1) * ov["block_nm"]), axis=1).mean()
            print(f"overview block {tuple(B[:3])}: {int(hd[1])} tiles ({miss} not in index), {len(rec):,} stand-ins, kind {np.unique(w & 15)}, "
                  f"regions {np.unique((w >> 4) & 31).tolist()}, radius {((w >> 9) & 1023).min() / 10}..{((w >> 9) & 1023).max() / 10} nm "
                  f"(median {np.median((w >> 9) & 1023) / 10}), inside block {inside * 100:.1f}%")
    # ---- geometry
    geo = json.loads(read(f"{d}/geometry.json")); F = load_fields(geo, d); G = Geo(geo, F)
    names = geo["fields"]["names"]; half = geo["cube_nm"] / 2
    print(f"geometry.json: kind {geo['kind']}, cube {geo['cube_nm']} nm at model {geo['centre_nm']}, R {geo['R']:.0f} nm, fields {names}, "
          f"{G.n_pores_all} pores ({len(G.P)} near the cube)")
    for k, hd in enumerate(geo["fields"]["head"]):
        print(f"  {names[k]:7s} origin {np.round(hd[:3], 1).tolist()} h {hd[3]} bricks {[int(x) for x in hd[4:7]]} B {int(hd[7])} "
              f"clamp -{hd[8]}/+{hd[9]} idx@{int(hd[10])} data@{int(hd[11])} ({geo['fields']['n_data'][k] // (int(hd[7]) + 1) ** 3} stored bricks)")
    for q in ([0.0, 0.0, 0.0], [half - 1, half - 1, half - 1], [-half + 1, -half + 1, -half + 1]):
        dm, mx, hp = G.mem_sdfs(*q)
        fv = {names[k]: round(field_sdf(F, *q, k), 2) for k in range(len(names))}
        print(f"  at {q}: fields {fv}; mem sdf {dict(zip(MEM_NAMES, np.round(dm, 1).tolist()))}; label {LABEL_NAMES[G.label(*q)]}")
    # label histogram on a grid through the cube
    s = (np.arange(16) + 0.5) / 16 * 2 * half - half
    lab = [G.label(x, y, z) for x in s for y in s for z in s]
    h = np.bincount(lab, minlength=len(LABEL_NAMES))
    print("  label fractions (16^3 grid): " + ", ".join(f"{LABEL_NAMES[i]} {h[i] / len(lab) * 100:.1f}%" for i in np.nonzero(h)[0]))
    if G.face_on: print(f"  open face at z = {G.zf:.1f} nm (label {LABEL_NAMES[OUTV]} below)")
    if G.spOff: print(f"  speckles: {int(F[G.spOff])} spheres (r {F[G.spOff + 4]:.0f} nm ...) at F[{G.spOff}]")
    else: print("  speckles: none in geometry.json")
    if G.crOff: print(f"  cristae: {int(F[G.crOff])} mitochondria, id grid {[int(v) for v in F[G.crOff + 5:G.crOff + 8]]} at {F[G.crOff + 4]} nm, "
                      f"{geo['fields']['cristae']['n_data'] // int(F[G.crOff + 8]) ** 3} stored id bricks")
    if not a.fields_only:
        # ---- label_mask of every tile re-derived with the app's rule (jenc samples 5^3 points per tile)
        s5 = (np.arange(5) + 0.5) * TILE / 5; ok = 0; bad = []
        TS = T if len(T) <= 2000 else T[np.random.default_rng(1).choice(len(T), a.sample, replace=False)]
        for r in TS:
            lo = r[:3] * TILE; m = 0
            for x in s5:
                for y in s5:
                    for z in s5: m |= 1 << G.label(lo[0] + x, lo[1] + y, lo[2] + z)
            if m == r[c["label_mask"]]: ok += 1
            else: bad.append((tuple(r[:3]), int(r[c["label_mask"]]), m))
        print(f"label_mask re-derived with the app rule: {ok}/{len(TS)} tiles identical" + (f"; differing (tile, index, app): {bad[:5]}" if bad else ""))
        # ---- open face: no membrane triangle of a tile at the face lies below it (jenc.clip_mesh)
        if G.face_on:
            at = T[(T[:, 2] * TILE <= G.zf + TILE) & ((T[:, 2] + 1) * TILE > G.zf - TILE) & (T[:, c["n_tri"]] > 0)]
            worst, ntri = np.inf, 0
            for r in at[np.random.default_rng(2).permutation(len(at))[:20]]:
                if sh:
                    b = sh["block_tiles"]; sp = sh["path"].format(bx=r[0] // b, by=r[1] // b, bz=r[2] // b)
                    Dt = decode_tile(read(f"{d}/{sp}", (int(r[c["off"]]), int(r[c["gz_bytes"]]))))
                else:
                    Dt = decode_tile(read(f"{d}/tiles/{r[0]}_{r[1]}_{r[2]}.bin.gz"))
                if len(Dt["tris"]):
                    cz = Dt["verts"][Dt["tris"].astype(np.int64)].mean(1)[:, 2]; worst = min(worst, cz.min() - G.zf); ntri += len(cz)
            print(f"open face: {min(len(at), 20)} tiles at the face decoded, {ntri:,} triangles, lowest centroid {worst:+.2f} nm above the face "
                  f"({'ok' if worst >= -0.01 else 'MEMBRANE BELOW THE FACE'})")
    # ---- optional: compare with the pipeline (cell3d on the full-resolution fields)
    if a.ref:
        sys.path.insert(0, f"{V}/jurkat/yc/pipeline"); import cell3d as c3
        z = np.load(a.geom); cell = c3.Cell(z["G"], z["pores"])
        C = np.array(geo["centre_nm"]); rng = np.random.default_rng(0)
        lo_, hi_ = -half, half
        if geo.get("whole_cell"):                                      # sample the cell's bounding box, not the whole cube
            h0 = geo["fields"]["head"][0]; lo_ = np.array(h0[:3]); hi_ = lo_ + np.array(h0[4:7]) * h0[3] * h0[7]
        P = rng.uniform(lo_, hi_, (a.n, 3)); ref = cell.labels(P + C)
        # stratified extra points for small compartments: random candidates labelled by cell3d, up to --strat per label
        Q = rng.uniform(lo_, hi_, (4_000_000, 3)); rq = cell.labels(Q + C)
        # crista lumens: IMS points inside the boundary inner membrane (matrix SDF of the boundary < 0)
        mq = cell.mems(Q + C); lumen = (rq == c3.IMS) & (mq[:, c3.M_OM] + c3.IMS_W < 0)
        extra, tag = [], []
        for L in (c3.NUCL, getattr(c3, "SPK", -99), c3.IMS, c3.MATRIX, c3.ERL, c3.GOLGI, c3.ENDO, c3.PERI, c3.PORE, c3.OUTV):
            k = np.flatnonzero(rq == L)[:a.strat]; extra.append(Q[k]); tag += [LABEL_NAMES[L]] * len(k)
        k = np.flatnonzero(lumen)[:a.strat]; extra.append(Q[k]); tag += ["IMS: crista lumen"] * len(k)
        # v6 nuclear speckles: points in balls of 1.25 r around the geometry.json spheres (viewer frame), split by the
        # cell3d label into inside (speckle) and the shell just outside (nucleoplasm, nucleolus, ...)
        spr = np.array((geo.get("speckles") or {}).get("spheres", []), float).reshape(-1, 4)
        if len(spr):
            m = 8 * a.strat; j = rng.integers(0, len(spr), m)
            v = rng.normal(size=(m, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
            Qs = spr[j, :3] + v * (spr[j, 3] * 1.25 * rng.uniform(0, 1, m) ** (1 / 3))[:, None]; rs = cell.labels(Qs + C)
            k = np.flatnonzero(rs == SPK)[:a.strat]; extra.append(Qs[k]); tag += ["speckle (ball)"] * len(k)
            k = np.flatnonzero(rs != SPK)[:a.strat]; extra.append(Qs[k]); tag += ["near speckle (outside)"] * len(k)
        P = np.r_[P, np.concatenate(extra)]; ref = np.r_[ref, np.concatenate([cell.labels(e + C) if len(e) else np.zeros(0, np.int8) for e in extra])]
        tag = np.array(["random"] * a.n + tag)
        app = np.array([G.label(*q) for q in P])
        rnd = tag == "random"; ins = rnd & (ref != c3.EXT) & (ref != c3.OUTV)
        print(f"labels vs cell3d ({a.geom}): random points {rnd.sum():,}: {(ref == app)[rnd].mean() * 100:.2f}% agree; "
              f"inside the cell {ins.sum():,}: {(ref == app)[ins].mean() * 100:.2f}%")
        print("  per cell3d label (random + stratified points): " + "; ".join(
            f"{LABEL_NAMES[L]} {(app[ref == L] == L).mean() * 100:.1f}% of {(ref == L).sum():,}" for L in np.unique(ref)))
        k = tag == "IMS: crista lumen"
        if k.any(): print(f"  crista lumens (IMS inside the boundary IM): {(app[k] == c3.IMS).mean() * 100:.1f}% of {k.sum():,} labelled IMS by the app "
                          f"(matrix {(app[k] == c3.MATRIX).mean() * 100:.1f}%)")
        for t_ in ("speckle (ball)", "near speckle (outside)"):
            k = tag == t_
            if k.any(): print(f"  {t_}: {(app[k] == ref[k]).mean() * 100:.2f}% of {k.sum():,} agree "
                              f"(app speckle {(app[k] == SPK).mean() * 100:.1f}%, cell3d speckle {(ref[k] == SPK).mean() * 100:.1f}%)")
        dis = ref != app
        if dis.any():
            mm = cell.mems(P + C); dmin = np.abs(mm[:, :7]).min(1)
            dn = np.array([abs(c3.field_sdf(*(q + C), cell.G, c3.F_NUCLEOLUS)) if c3.F_NUCLEOLUS < int(cell.G[c3.G_N]) else 1e9 for q in P[dis]])
            spr = np.array(geo.get("speckles", {}).get("spheres", [])).reshape(-1, 4)
            ds = (np.abs(np.linalg.norm(P[dis][:, None, :] - spr[None, :, :3], axis=2) - spr[None, :, 3]).min(1) if len(spr)
                  else np.full(dis.sum(), 1e9))
            print(f"  disagreeing points: median distance to the nearest membrane {np.median(dmin[dis]):.2f} nm (all points {np.median(dmin):.1f} nm), "
                  f"to the nucleolus surface {np.median(dn):.1f} nm, to a speckle surface {np.median(ds):.1f} nm")
            pairs, cnts = np.unique(np.c_[ref[dis], app[dis]], axis=0, return_counts=True)
            o = np.argsort(-cnts)[:12]
            print("  pairs (cell3d -> app): " + ", ".join(f"{LABEL_NAMES[pairs[i][0]]}->{LABEL_NAMES[pairs[i][1]]} {cnts[i]}" for i in o))

if __name__ == "__main__":
    main()
