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
                "mem_ER", "mem_MITO_OM", "mem_MITO_IM", "mem_GOLGI", "mem_ENDO", "chromatin", "polysome_ribosome", "cytoskeleton", "NPC"]
MEM_NAMES = ["PM", "NE", "ER", "MITO_OM", "MITO_IM", "GOLGI", "ENDO"]
LABEL_NAMES = ["exterior", "cytosol", "perinuclear", "nucleoplasm", "nucleolus", "mito_IMS", "mito_matrix", "ER_lumen",
               "golgi_lumen", "endolysosomal_lumen", "pore_channel"]
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
def load_fields(geo, d):
    fd = geo["fields"]; buf = read(os.path.join(d, fd["file"])); n = len(fd["head"])
    total = 1 + 12 * n + sum(fd["n_index"]) + sum(fd["n_data"])
    F = np.zeros(total, np.float32); F[0] = n; off = 0
    for k in range(n):
        F[1 + 12 * k:13 + 12 * k] = fd["head"][k]
        ni, nd = fd["n_index"][k], fd["n_data"][k]
        idx = np.frombuffer(buf, "<i4", ni, off); off += 4 * ni
        F[int(fd["head"][k][10]):int(fd["head"][k][10]) + ni] = idx
        dat = np.frombuffer(buf, "<i2", nd, off); off += 2 * nd; off = (off + 3) // 4 * 4
        d0 = int(fd["head"][k][11]); F[d0:d0 + nd] = dat.astype(np.float32) * np.float32(fd["q"])
    assert off == len(buf), f"geometry_fields.bin: decoded {off} of {len(buf)} bytes"
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

    def mem_sdfs(self, x, y, z):
        d = [0.0] * 7; rc = self.PORE_R + 0.5 * self.PERI
        d[0] = -field_sdf(self.F, x, y, z, 0)
        hp = field_sdf(self.F, x, y, z, 1); dne = abs(hp) - 0.5 * self.PERI
        if abs(hp) < rc + self.PERI:
            rho = self.in_pore(x, y, z, rc)
            if rho >= 0: dne = math.sqrt((rho - rc) ** 2 + hp * hp) - 0.5 * self.PERI
        d[1] = dne
        om = field_sdf(self.F, x, y, z, 2); d[3] = om; matrix = om + self.IMS; d[4] = -matrix
        d[2] = field_sdf(self.F, x, y, z, 3); d[5] = field_sdf(self.F, x, y, z, 4); d[6] = field_sdf(self.F, x, y, z, 5)
        return d, matrix, hp

    def label(self, x, y, z):
        d, matrix, hp = self.mem_sdfs(x, y, z); PERI = self.PERI
        if d[0] < 0: return 0
        if d[1] < 0: return 2
        if hp < -0.5 * PERI:
            if hp > -PERI - 8 and self.in_pore(x, y, z, self.PORE_R + 0.5 * PERI) >= 0: return 10
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
    a = ap.parse_args(); d = a.data.rstrip("/")
    try:
        SPLIT.update(json.loads(_raw(f"{d}/split.json"))); SPLIT["_base"] = [d + "/"]
        if len(SPLIT) > 1: print(f"split.json: {({k: v for k, v in SPLIT.items() if k != '_base'})}")
    except Exception:
        pass
    # ---- index
    J = json.loads(read(f"{d}/tiles/index.json")); cols = J["tile_columns"]; T = np.array(J["tiles"], np.int64)
    c = {k: i for i, k in enumerate(cols)}
    print(f"index.json: {len(T)} tiles, ix {T[:, 0].min()}..{T[:, 0].max()}, iy {T[:, 1].min()}..{T[:, 1].max()}, iz {T[:, 2].min()}..{T[:, 2].max()}; "
          f"{T[:, c['n_inst']].sum():,} instances, {T[:, c['n_ghost']].sum():,} ghosts, {T[:, c['n_tri']].sum():,} mesh triangles, "
          f"{T[:, c['gz_bytes']].sum() / 1e6:.1f} MB gz; 'wall' key: {'wall' in J}")
    for key, want in (("region_names", REGION_NAMES), ("mem_names", MEM_NAMES), ("label_names", LABEL_NAMES)):
        print(f"  {key} {'match the app' if J[key] == want else 'DIFFER from the app: ' + str(J[key])}")
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
    h = np.bincount(lab, minlength=11)
    print("  label fractions (16^3 grid): " + ", ".join(f"{LABEL_NAMES[i]} {h[i] / len(lab) * 100:.1f}%" for i in np.nonzero(h)[0]))
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
    # ---- optional: compare with the pipeline (cell3d on the full-resolution fields)
    if a.ref:
        sys.path.insert(0, f"{V}/jurkat/yc/pipeline"); import cell3d as c3
        z = np.load(f"{V}/data/derived/jgeom_cell1.npz"); cell = c3.Cell(z["G"], z["pores"])
        C = np.array(geo["centre_nm"]); rng = np.random.default_rng(0)
        P = rng.uniform(-half, half, (a.n, 3)); ref = cell.labels(P + C)
        app = np.array([G.label(*q) for q in P])
        agree = (ref == app).mean()
        print(f"labels vs cell3d at {a.n} random points: {agree * 100:.2f}% agree")
        if agree < 1:
            mm = cell.mems(P + C); dmin = np.abs(mm[:, :7]).min(1)
            dis = ref != app
            print(f"  disagreeing points: median distance to the nearest membrane {np.median(dmin[dis]):.2f} nm (all points {np.median(dmin):.1f} nm); "
                  f"pairs (cell3d->app): {sorted(set(zip(ref[dis].tolist(), app[dis].tolist())))[:10]}")


if __name__ == "__main__":
    main()
