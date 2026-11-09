"""Streaming whole-cell encoder for the Jurkat WebGPU viewer (whole-cell mode; jenc.py encodes one cube).

The packed cell (~900 M instances in ~7,500 packing tiles of 500 nm) is re-binned into the viewer's 200 nm tiles (same
YCT1 tile binary as jenc.py / webenc.py) without ever holding the cell in memory:

* Work unit = one 1 um x-slab of the cell. A slab is encoded block by block (1 um cubes = 5x5x5 viewer tiles = 2x2x2
  packing tiles; blocks are aligned with both grids). Slabs run in a fork pool (--workers, default 6); each slab reads
  the packing tiles it covers plus an 80 nm margin (the ghost range: ghost centres must lie within the 360 nm
  quantisation window that starts 80 nm below the tile origin), so memory is bounded by one slab per worker.
* Ghosts (near-membrane instances of neighbouring tiles) are taken from the block + margin set, so they cross block and
  slab borders exactly as in jenc.py.
* mRNA threads (prepass/mrna_cell1.npz) are resampled to 0.6 nm and cut into 64-point pieces counted from the thread
  start, so every slab cuts a thread identically; a piece belongs to the viewer tile of its midpoint and is written only
  by the slab owning that tile (no piece is lost or duplicated at borders).
* Output tiles are gzip members concatenated into one shard file per block (tiles/s_BX_BY_BZ.bin, ~1-30 MB);
  index.json gives each tile's byte offset and length, and the viewer fetches a tile with an HTTP Range request
  (falling back to the whole shard when a server ignores Range). This keeps the file count near 2,000 (Hugging Face
  advises < 10k files per folder, < 100k per repo) while keeping per-tile streaming.
* Overview level: per viewer tile up to --proxies stand-in spheres (instances sampled by mass, radius scaled so the
  stand-ins fill about half the tile's molecular envelope volume, coloured by region), one gzip file per block
  (overview/BX_BY_BZ.bin.gz). The viewer shows them on the cut face until the tile's molecules are loaded.
* Library: every type present anywhere in the cell. If the total bead count exceeds --bead-budget, the largest
  libraries get coarser beads (stride subsampling of their Morton-ordered beads) until the total fits; see
  encode_meta.json 'coarsened'. The default budget (2^24) leaves the v3 library whole (10.4 M beads = 166 MB on the
  GPU at 16 B per bead: fine on desktop GPUs, which expose >= 1 GB storage bindings, but above the WebGPU minimum of
  128 MiB). Since the renderer keeps the library resident, residue detail of every type is available anywhere in the
  cell without per-region library shards (types occur cell-wide, so shards would mostly duplicate it).
* Distance fields (cut-face fill) are resampled from 16 nm to --field-sp (default 48 nm) for the whole cell; the
  per-tile membrane meshes (8 nm) carry the fine membrane detail.
* Resumable: each block writes its shard then a small JSON record in OUT_work/; finished blocks are skipped on rerun.

usage: jenc_full.py --tiles DIR --libs libs.json --meta tiles_meta.json --parquet tiles.parquet --geom jgeom.npz
                    --out DIR [--workers 6] [--only-slab BX] [--only-block BX BY BZ] [--proxies 128]
Outputs (OUT): tiles/index.json, tiles/s_*.bin, overview/*.bin.gz, library/{types.json, beads.bin(.NNN),
spheretree.bin}, types_tiles.bin (uint32 tile indices, <= 256 tiles per type), geometry.json, geometry_fields.bin,
organelle_meshes.bin.gz, bookmarks.json, split.json, version.json, encode_meta.json.
"""
import os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "1")            # one core per worker process
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # OpenMP aborts after fork
import json, gzip, time, argparse, math, shutil, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd, numba as nb
from skimage.measure import marching_cubes

HERE = Path(__file__).resolve().parent
V = os.environ.get("VCELL_ROOT", str(HERE.parents[1]))      # workspace root (jurkat/viewer/../..)
sys.path.insert(0, str(HERE)); sys.path.insert(0, f"{V}/jurkat/yc/pipeline")
import jenc as J                                              # shared YCT1 helpers, organelle meshes, bookmarks
import cell3d as c3
from config import LIB

D = f"{V}/data/derived"
VT, BLK, MARGIN = 200.0, 1000.0, J.Q_ORG                      # viewer tile, block, ghost margin (nm)
NB = int(BLK / VT)                                            # viewer tiles per block edge (5)
PIECE, STEP = 64, 0.6                                         # mRNA piece length (points) and resampling step (nm)
MR_Q = 0.005
SPLIT_MB = 45.0                                               # files above this are split into parts (split.json)
PRE_REG_OF = lambda n: (J.REGION_NAMES.index("chromatin") if (n == "NUCLEOSOME" or n.startswith("H1-")) else
                        J.REGION_NAMES.index("polysome_ribosome") if n.startswith("RIBO") else
                        J.REGION_NAMES.index("NPC") if n.startswith("NPC") else J.REGION_NAMES.index("cytoskeleton"))
log = J.log
S = {}                                                        # state shared with forked workers (read-only)


# ------------------------------------------------------------------ library
def load_type(lid):
    z = np.load(f"{LIB}/{lid}.npz"); b = z["beads"].astype(np.float64)
    k = z["kind"].astype(np.uint8) if "kind" in z.files else np.zeros(len(b), np.uint8)
    st = np.c_[z["st_c"], z["st_r"]].astype(np.float32)
    br = float((np.linalg.norm(z["st_c"], axis=1) + z["st_r"]).max())
    return b, k, st, br


def coarsen_plan(nbeads, budget):
    """stride per type (1 = full) so that sum(ceil(nb / stride)) <= budget, coarsening the largest libraries first"""
    stride = np.ones(len(nbeads), np.int64)
    tot = lambda: int(np.ceil(nbeads / stride).sum())
    if tot() <= budget: return stride
    order = np.argsort(-nbeads)
    for s in (2, 3, 4, 6, 8, 12, 16):
        for i in order:
            if tot() <= budget: return stride
            if nbeads[i] < 20000: break                     # only big assemblies are coarsened
            stride[i] = max(stride[i], s)
    return stride


# ------------------------------------------------------------------ fields resampled for the whole cell
@nb.njit(cache=False)
def _eval_field(P, G, k):
    out = np.empty(len(P), np.float32)
    for i in range(len(P)): out[i] = c3.field_sdf(P[i, 0], P[i, 1], P[i, 2], G, k)
    return out


def web_fields_whole(cell, out, sp=48.0, sp_thin=32.0, B=4):
    """jenc.web_fields layout for the whole cell, resampled at sp nm (bricks of B voxels, (B+1)^3 nodes each); the
    fields of thin compartments (ER, Golgi, endo-lysosomal: sheets and vesicles of 30-100 nm) at sp_thin nm"""
    G = cell.G; heads, n_index, n_data, parts = [], [], [], []
    off = 1 + 12 * len(J.BANDS); n1 = B + 1
    for k, (bin_, bout) in enumerate(J.BANDS):
        sp_k = sp_thin if c3.FIELD_NAMES[k] in ("er", "golgi", "endo") else sp
        g = np.arange(n1) * sp_k; node = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        o = c3.G_N + 1 + 12 * k; h = G[o:o + 12]
        org, sp0, nb0, B0 = h[0:3], h[3], h[4:7].astype(int), int(h[7])
        ext = nb0 * sp0 * B0; nbc = np.ceil(ext / (sp_k * B)).astype(int)
        idx = np.empty(int(np.prod(nbc)), np.int64); keep = []
        bricks = np.stack(np.meshgrid(*[np.arange(n) for n in nbc], indexing="ij"), -1).reshape(-1, 3)
        for c0 in range(0, len(bricks), 20000):
            bb = bricks[c0:c0 + 20000]
            P = (org + bb[:, None, :] * sp_k * B + node[None]).reshape(-1, 3)
            vals = _eval_field(P, G, k).reshape(len(bb), -1)
            mx, mn = vals.max(1), vals.min(1)
            for j in range(len(bb)):
                if mx[j] <= -bin_: idx[c0 + j] = -1
                elif mn[j] >= bout: idx[c0 + j] = -2
                else: idx[c0 + j] = len(keep); keep.append(np.clip(vals[j], -bin_, bout))
        dat = np.round(np.concatenate(keep) / J.FIELD_Q).astype(np.int16) if keep else np.zeros(0, np.int16)
        heads.append([float(x) for x in org] + [float(sp_k), float(nbc[0]), float(nbc[1]), float(nbc[2]), float(B), bin_, bout,
                                                float(off), float(off + len(idx))])
        off += len(idx) + len(dat); n_index.append(int(len(idx))); n_data.append(int(len(dat)))
        parts += [idx.astype("<i4").tobytes(), dat.astype("<i2").tobytes() + b"\0" * ((-2 * len(dat)) % 4)]
        log(f"field {c3.FIELD_NAMES[k]}: {len(keep):,} of {len(idx):,} bricks stored at {sp_k:.0f} nm")
    open(f"{out}/geometry_fields.bin", "wb").write(b"".join(parts))
    return dict(file="geometry_fields.bin", names=c3.FIELD_NAMES, head=heads, n_index=n_index, n_data=n_data, q=J.FIELD_Q,
                layout="as jenc.py web_fields (whole cell, resampled at %g nm; er, golgi, endo at %g nm)" % (sp, sp_thin))


# ------------------------------------------------------------------ mRNA pieces of one slab
def mrna_slab_pieces(x0, x1):
    """pieces (64 points at 0.6 nm, counted from each thread start) whose midpoint lies in x0 <= x < x1;
    returns list of (tile key (3,), (n,3) array)"""
    off, pts, xmin, xmax = S["mr_off"], S["mr_pts"], S["mr_xmin"], S["mr_xmax"]
    reach = PIECE * STEP
    out = []
    for i in np.flatnonzero((xmax >= x0 - reach) & (xmin < x1 + reach)):
        P = np.asarray(pts[off[i]:off[i + 1]], np.float64)
        if len(P) < 2: continue
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]
        t = np.arange(0, s[-1], STEP)
        if len(t) < 2: continue
        R = np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)
        for p0 in range(0, len(R) - 1, PIECE - 1):
            q = R[p0:p0 + PIECE]
            if len(q) < 2: continue
            mid = q[len(q) // 2]
            if x0 <= mid[0] < x1: out.append((np.floor(mid / VT).astype(np.int64), q))
    return out


# ------------------------------------------------------------------ one slab
def load_slab(bx):
    """instances whose centre lies in the slab bx (+- MARGIN in x), from packed tiles and the pre-pass"""
    tij, lo0, PT = S["tij"], S["lo0"], S["PT"]
    x0, x1 = bx * BLK - MARGIN, (bx + 1) * BLK + MARGIN
    tlo = tij[:, 0] * PT + lo0[0]
    sel = np.flatnonzero((tlo < x1 + 50) & (tlo + PT > x0 - 50))       # (instances lie in their packing tile; 50 nm slack)
    T, P, Q, R = [], [], [], []
    for ti in sel:
        f = f"{S['tiles']}/{ti}.npz"
        if os.path.exists(f):
            z = np.load(f); pos = z["pos"]; k = (pos[:, 0] >= x0) & (pos[:, 0] < x1)
            if k.any():
                reg = np.repeat(z["job_region"], z["job_placed"]).astype(np.int8)
                T.append(S["vt_of"][z["types"][k]]); P.append(pos[k]); Q.append(z["quat"][k]); R.append(reg[k])
        p0, p1 = S["poff"][ti], S["poff"][ti + 1]
        if p1 > p0:
            pp = S["ppos"][p0:p1]; k = (pp[:, 0] >= x0) & (pp[:, 0] < x1)
            if k.any():
                ty = S["ptypes"][p0:p1][k]
                T.append(S["vt_of"][S["pre_glob"][ty]]); P.append(pp[k]); Q.append(S["pquat"][p0:p1][k]); R.append(S["pre_reg"][ty])
    if not T: return None
    T = np.concatenate(T).astype(np.int32); assert (T >= 0).all()
    # float32 / int8 while the slab is held (about 33 B per instance); float64 per block only
    return T, np.concatenate(P).astype(np.float32), np.concatenate(Q).astype(np.float32), np.concatenate(R).astype(np.int8)


def encode_slab(bx):
    t0 = time.time(); work, out = S["work"], S["out"]
    todo_only = S.get("only_block")
    data = load_slab(bx); t1 = time.time()
    pieces = mrna_slab_pieces(bx * BLK, (bx + 1) * BLK); t2 = time.time()
    log(f"slab {bx}: {0 if data is None else len(data[0]):,} instances loaded in {t1 - t0:.0f}s, {len(pieces):,} mRNA pieces in {t2 - t1:.0f}s")
    if data is None and not pieces: return bx, 0, time.time() - t0
    if data is None: data = (np.zeros(0, np.int32), np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32), np.zeros(0, np.int8))
    T, P, Q, R = data
    # group by the (by, bz) block of the centre: one int32 code per instance, argsort (cheap at ~1e8 instances)
    OFFB = 64
    code = ((np.floor(P[:, 1] / BLK).astype(np.int32) + OFFB) * 128 + (np.floor(P[:, 2] / BLK).astype(np.int32) + OFFB))
    order = np.argsort(code, kind="stable"); cs = code[order]; del code
    ucode, first = np.unique(cs, return_index=True); last = np.r_[first[1:], len(cs)]; del cs
    grp = {(int(u) // 128 - OFFB, int(u) % 128 - OFFB): order[f:l] for u, f, l in zip(ucode, first, last)}
    pk = {}
    for key, q in pieces: pk.setdefault((int(key[1]) // NB, int(key[2]) // NB), []).append((key, q))
    blocks = set(grp.keys()) | set(pk.keys())
    # candidate blocks also include those reached only by a membrane (no instance): every block whose cube meets the cell
    cb = S["cell_blocks"]; blocks |= {(b[1], b[2]) for b in cb if b[0] == bx}
    n_done = 0
    for (by, bz) in sorted(blocks):
        if todo_only is not None and (bx, by, bz) != todo_only: continue
        name = f"{bx}_{by}_{bz}"
        if os.path.exists(f"{work}/{name}.json"): continue
        lo_b = np.array([bx, by, bz], float) * BLK
        ii = [grp[(by + a, bz + c)] for a in (-1, 0, 1) for c in (-1, 0, 1) if (by + a, bz + c) in grp]
        ii = np.concatenate(ii) if ii else np.zeros(0, np.int64)
        if len(ii):
            m = np.all((P[ii] >= lo_b - MARGIN) & (P[ii] < lo_b + BLK + MARGIN), axis=1); ii = ii[m]
        tb = time.time()
        encode_block((bx, by, bz), T[ii], P[ii].astype(np.float64), Q[ii].astype(np.float64), R[ii].astype(np.int64),
                     pk.get((by, bz), []))
        n_done += 1
        if n_done <= 3 or n_done % 20 == 0: log(f"  block {name}: {len(ii):,} instances (with margin) in {time.time() - tb:.0f}s")
    return bx, n_done, time.time() - t0


def tile_mesh(cell, lo):
    """membrane meshes of one viewer tile (jenc.py: marching cubes at 8 nm); a 9^3 pre-test skips tiles no membrane reaches"""
    s9 = np.linspace(0, VT, 9)
    X = np.stack(np.meshgrid(lo[0] + s9, lo[1] + s9, lo[2] + s9, indexing="ij"), -1).reshape(-1, 3)
    m9 = cell.mems(X)[:, J.MEMS]
    near = (m9.min(0) < 0) & (m9.max(0) > 0) | (np.abs(m9).min(0) < 30)
    if not near.any(): return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.uint8)
    m_ = int(VT / J.MESH_H) + 1; gm = np.arange(m_) * J.MESH_H
    X = np.stack(np.meshgrid(lo[0] + gm, lo[1] + gm, lo[2] + gm, indexing="ij"), -1).reshape(-1, 3)
    Sd = cell.mems(X).reshape(m_, m_, m_, c3.NMEM)
    Vs, Fs, FM, nv = [], [], [], 0
    for mi, mm in enumerate(J.MEMS):
        s = Sd[..., mm]
        if s.min() < 0 < s.max():
            v, fc, _, _ = marching_cubes(s, 0.0, spacing=(J.MESH_H,) * 3)
            Vs.append(v + lo); Fs.append(fc + nv); FM.append(np.full(len(fc), mi, np.uint8)); nv += len(v)
    if not Vs: return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.uint8)
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(FM)


def proxies(vt, P, R, key, K):
    """K stand-in spheres for one tile: instances sampled by mass; record = xyz (f32), w = 7 | region << 4 | r/0.1nm << 9"""
    n = len(vt)
    if n == 0: return np.zeros((0, 4), np.uint32)
    mass, env = S["mass"][vt], S["env"][vt]
    rng = np.random.default_rng([int(key[0]) + 4096, int(key[1]) + 4096, int(key[2]) + 4096])
    k = min(K, n)
    sel = rng.choice(n, k, replace=False, p=mass / mass.sum()) if k < n else np.arange(n)
    r = env[sel]
    s = np.cbrt(0.5 * (env ** 3).sum() / max((r ** 3).sum(), 1e-9)); s = float(np.clip(s, 1.0, 6.0))
    rq = np.clip(np.round(r * s / 0.1), 1, 1023).astype(np.uint32)
    rec = np.zeros((k, 4), np.uint32); rec[:, :3] = P[sel].astype(np.float32).view(np.uint32)
    rec[:, 3] = np.uint32(7) | (R[sel].astype(np.uint32) << 4) | (rq << 9)
    return rec


def encode_block(b, T, P, Q, R, pieces):
    cell, out, work = S["cell"], S["out"], S["work"]
    bx, by, bz = b; lo_b = np.array(b, float) * BLK; name = f"{bx}_{by}_{bz}"
    BR = S["br"]
    own = np.all((P >= lo_b) & (P < lo_b + BLK), axis=1) if len(P) else np.zeros(0, bool)
    # near-membrane flags (jenc.py) for own + margin instances (margin ones become ghosts of this block's tiles)
    if len(P):
        mm = cell.mems(P); dmin = np.abs(mm[:, J.MEMS]).min(1); flags = (dmin < BR[T] + 3.0).astype(np.uint8)
    else:
        flags = np.zeros(0, np.uint8)
    key = np.floor(P / VT).astype(np.int64)
    fl_i = np.flatnonzero(flags == 1)
    oi = np.flatnonzero(own)
    korder = oi[np.lexsort((R[oi], T[oi], key[oi, 2], key[oi, 1], key[oi, 0]))]
    kl = key[korder] - np.array(b) * NB                     # tile within the block, 0..NB-1 per axis
    code = (kl[:, 0] * NB + kl[:, 1]) * NB + kl[:, 2]        # sorted (korder is x, y, z major)
    pkey = {}
    for kk, q in pieces: pkey.setdefault(tuple(int(x) for x in kk), []).append(q)
    shard = bytearray(); rows, ov_tiles, ov_recs, type_tile = [], [], [], []
    for a in range(NB):
        for c in range(NB):
            for e in range(NB):
                t = (bx * NB + a, by * NB + c, bz * NB + e); lo = np.array(t, float) * VT; hi = lo + VT
                c_ = (a * NB + c) * NB + e
                sel = korder[np.searchsorted(code, c_, "left"):np.searchsorted(code, c_, "right")]
                Vm, Fm, FMm = tile_mesh(cell, lo)
                mpieces = pkey.get(t, [])
                if not len(sel) and not len(Vm) and not mpieces: continue
                inst = J.pack_instances(T[sel], P[sel], Q[sel], R[sel], flags[sel], lo)
                # ghosts: flagged instances of other tiles whose bounding sphere reaches this tile
                if len(fl_i):
                    p = P[fl_i]; br = BR[T[fl_i]] + 3.0
                    ok = np.all((p > lo - br[:, None]) & (p < hi + br[:, None]), axis=1) & ~np.all(key[fl_i] == t, axis=1)
                    ok &= np.all((p > lo - J.Q_ORG + 1) & (p < lo - J.Q_ORG + J.Q_RANGE - 1), axis=1)
                    gh = fl_i[ok]
                else:
                    gh = np.zeros(0, np.int64)
                ghost = J.pack_instances(T[gh], P[gh], Q[gh], R[gh], flags[gh], lo)
                vbytes = J.pad4(J.quant_pos(Vm, lo).astype("<u2").tobytes()) if len(Vm) else b""
                big = len(Vm) > 65535
                fbytes = J.pad4(Fm.astype("<u4" if big else "<u2").tobytes()) if len(Fm) else b""
                mbytes = J.pad4(FMm.astype("u1").tobytes()) if len(Fm) else b""
                mrb, nm, nmb = J.pack_polylines(mpieces, lo, MR_Q)
                hdr = np.zeros(16, np.uint32); hdr[0] = int.from_bytes(b"YCT1", "little"); hdr[1:4] = np.array(t, np.int32).view(np.uint32)
                hdr[4:15] = [len(inst), len(ghost), nm, nmb, 0, 0, len(Vm), len(Fm), 1 if big else 0, 0, 0]
                blob = hdr.tobytes() + inst.astype("<u4").tobytes() + ghost.astype("<u4").tobytes() + mrb + vbytes + fbytes + mbytes
                gzb = gzip.compress(blob, compresslevel=6)
                s5 = (np.arange(5) + 0.5) * VT / 5
                lab = cell.labels(np.stack(np.meshgrid(lo[0] + s5, lo[1] + s5, lo[2] + s5, indexing="ij"), -1).reshape(-1, 3))
                lmask = int(sum(1 << int(l) for l in np.unique(lab)))
                rows.append([*t, int(len(inst)), int(len(ghost)), int(nmb), 0, int(len(Vm)), int(len(Fm)), int(len(gzb)), lmask, len(shard)])
                shard += gzb
                pr = proxies(T[sel], P[sel], R[sel], t, S["K"])
                ov_tiles.append([*t, len(pr)]); ov_recs.append(pr)
                for ty in np.unique(T[sel]): type_tile.append((int(ty), *t))
    if rows:
        tmp = f"{out}/tiles/s_{name}.bin.tmp"; open(tmp, "wb").write(bytes(shard)); os.replace(tmp, f"{out}/tiles/s_{name}.bin")
        hdr = np.array([int.from_bytes(b"YCO1", "little"), len(ov_tiles)], np.uint32).tobytes() + np.array(ov_tiles, "<i4").tobytes()
        recs = np.concatenate(ov_recs).astype("<u4") if ov_recs else np.zeros((0, 4), "<u4")
        ob = gzip.compress(hdr + recs.tobytes(), compresslevel=6)
        tmp = f"{out}/overview/{name}.bin.gz.tmp"; open(tmp, "wb").write(ob); os.replace(tmp, f"{out}/overview/{name}.bin.gz")
    else:
        ob = b""
    # one tile per type per block for types_tiles (fly-to-nearest); keep the first tile of each type
    tt = {}
    for ty, *t in type_tile: tt.setdefault(ty, t)
    rec = dict(block=list(b), rows=rows, shard_bytes=len(shard), n_proxy=int(sum(x[3] for x in ov_tiles)), ov_bytes=len(ob),
               own=int(own.sum()), types=[[ty, *t] for ty, t in tt.items()])
    tmp = f"{work}/{name}.json.tmp"; json.dump(rec, open(tmp, "w"), separators=(",", ":")); os.replace(tmp, f"{work}/{name}.json")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tiles", required=True); ap.add_argument("--libs", required=True)
    ap.add_argument("--meta", required=True); ap.add_argument("--parquet", required=True)
    ap.add_argument("--geom", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prepass", default=f"{D}/wc", help="folder with prepass_{off,types,pos,quat}.npy and prepass_libs.json")
    ap.add_argument("--mrna", default=f"{D}/prepass/mrna_cell1.npz")
    ap.add_argument("--ingredients", default=f"{D}/ingredients_v0.tsv")
    ap.add_argument("--uniprot", default=f"{V}/data/reference/uniprot_human_full.tsv.gz")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--proxies", type=int, default=128)
    ap.add_argument("--bead-budget", type=int, default=1 << 24,
                    help="max library beads; above it the largest assemblies get coarser beads (8e6 fits a 128 MiB binding)")
    ap.add_argument("--field-sp", type=float, default=48.0)
    ap.add_argument("--field-sp-thin", type=float, default=32.0, help="spacing of the ER, Golgi and endo fields")
    ap.add_argument("--fields-only", action="store_true", help="only rewrite geometry_fields.bin and geometry.json 'fields'")
    ap.add_argument("--only-slab", type=int, nargs="*"); ap.add_argument("--only-block", type=int, nargs=3)
    ap.add_argument("--bookmark-centre", type=float, nargs=3, default=[200.0, -3000.0, -1400.0],
                    help="centre of the region where the named bookmarks are picked (default: the jenc.py detail cube)")
    ap.add_argument("--assemble-only", action="store_true", help="only (re)build index.json etc. from finished blocks")
    ap.add_argument("--skip-static", action="store_true", help="skip library / fields / meshes (already written)")
    a = ap.parse_args()
    out = a.out; work = out.rstrip("/") + "_work"
    for d in (out, f"{out}/tiles", f"{out}/overview", f"{out}/library", work): os.makedirs(d, exist_ok=True)
    T0 = time.time()
    z = np.load(a.geom); cell = c3.Cell(z["G"], z["pores"]); del z
    R_cell = float(cell.G[0]); log(f"geometry {a.geom}: R {R_cell:.0f} nm, {len(cell.pores)} pores")
    if a.fields_only:
        gj = json.load(open(f"{out}/geometry.json"))
        gj["fields"] = web_fields_whole(cell, out, sp=a.field_sp, sp_thin=a.field_sp_thin)
        json.dump(gj, open(f"{out}/geometry.json", "w"), separators=(",", ":"))
        sp = split_large(out); json.dump(dict(v=time.strftime("%Y%m%d%H%M%S")), open(f"{out}/version.json", "w"))
        if os.path.exists(f"{out}/encode_meta.json"):
            m = json.load(open(f"{out}/encode_meta.json")); m.update(split=sp, field_sp_nm=a.field_sp, field_sp_thin_nm=a.field_sp_thin)
            json.dump(m, open(f"{out}/encode_meta.json", "w"), indent=1)
        log("fields rewritten"); return
    if a.assemble_only:
        tj = json.load(open(f"{out}/library/types.json"))
        coarsened = {t["id"]: dict(stride=t["bead_stride"]) for t in tj["types"] if "bead_stride" in t}
        assemble(a, out, work, len(tj["types"]), half_from_geo(out), time.time() - T0,
                 sum(t["copies"] for t in tj["types"]), coarsened, cell)
        return
    M = pd.read_parquet(a.parquet); meta = json.load(open(a.meta))
    tij = M[["ti", "tj", "tk"]].values.astype(np.int64); lo0 = np.array(meta["lo0"], float); PT = float(meta["TILE"])
    libs = json.load(open(a.libs))["libs"]; PL = json.load(open(f"{a.prepass}/prepass_libs.json"))
    names = list(libs); gi = {n: i for i, n in enumerate(names)}
    for n in PL:
        if n not in gi: gi[n] = len(names); names.append(n)
    pre_glob = np.array([gi[n] for n in PL], np.int64)
    # ---- pass 0: copies per (type, region) from the job tables and the pre-pass (no instance arrays needed)
    NR = len(J.REGION_NAMES); cnt = np.zeros((len(names), NR), np.int64)
    ptypes = np.load(f"{a.prepass}/prepass_types.npy"); poff = np.load(f"{a.prepass}/prepass_off.npy")
    assert len(poff) == len(tij) + 1, "pre-pass offsets do not match the tile table"
    pre_reg = np.array([PRE_REG_OF(n) for n in PL], np.int64)
    pc = np.bincount(ptypes, minlength=len(PL))
    for j, n in enumerate(PL): cnt[pre_glob[j], pre_reg[j]] += pc[j]
    with mp.get_context("fork").Pool(a.workers) as pool:
        for jl, jr, jp in pool.imap_unordered(_jobs, [f"{a.tiles}/{ti}.npz" for ti in range(len(tij))], chunksize=32):
            np.add.at(cnt, (jl, jr), jp)
    tot = cnt.sum(1); present = np.flatnonzero(tot > 0)
    vt_of = np.full(len(names), -1, np.int32); vt_of[present] = np.arange(len(present))
    ids = [names[i] for i in present]; NT = len(ids)
    assert NT < 16384, NT
    log(f"{int(tot.sum()):,} instances ({int(pc.sum()):,} pre-pass) of {NT:,} types; pass 0 {time.time() - T0:.0f}s")
    # ---- library (+ per-type radius, envelope, mass)
    BR = np.zeros(NT); ENV = np.zeros(NT); MASS = np.zeros(NT)
    cache = f"{work}/typeinfo.npz"; lib = None
    if a.skip_static and os.path.exists(cache) and list(np.load(cache)["ids"]) == ids:
        zc = np.load(cache); BR, ENV, MASS, nbeads = zc["br"], zc["env"], zc["mass"], zc["nb"]
    else:
        lib = [load_type(l) for l in ids]
        nbeads = np.array([len(x[0]) for x in lib])
        for i, (b, k, st, br) in enumerate(lib):
            BR[i] = br
            rg = float(np.sqrt(((b - b.mean(0)) ** 2).sum(1).mean())); ENV[i] = max(rg * math.sqrt(5 / 3), 1.0)
            MASS[i] = float(np.where(k == 0, 110.0, 330.0).sum())
        np.savez(cache, ids=np.array(ids), br=BR, env=ENV, mass=MASS, nb=nbeads)
    stride = coarsen_plan(nbeads, a.bead_budget)
    coarsened = {ids[i]: dict(stride=int(stride[i]), beads=int(nbeads[i]), kept=int(math.ceil(nbeads[i] / stride[i])))
                 for i in np.flatnonzero(stride > 1)}
    log(f"library: {int(nbeads.sum()):,} beads; budget {a.bead_budget:,}; coarsened {len(coarsened)} types -> "
        f"{int(np.ceil(nbeads / stride).sum()):,} beads")
    if not a.skip_static:
        assert lib is not None
        write_library(a, out, ids, lib, stride, cnt[present], BR, MASS)
        geo_fields = web_fields_whole(cell, out, sp=a.field_sp, sp_thin=a.field_sp_thin)
        half = math.ceil(max(np.abs(lo0).max(), R_cell) / VT) * VT
        json.dump(dict(kind="jurkat_fields", whole_cell=True, centre_nm=[0.0, 0.0, 0.0], cube_nm=2 * half, R=R_cell,
                       fields=geo_fields, pores=cell.pores.round(4).ravel().tolist(),
                       consts=dict(PERI_W=c3.PERI_W, IMS_W=c3.IMS_W, PORE_R=c3.PORE_R, HALF=c3.HALF),
                       label_names=c3.LABEL_NAMES, mem_names=J.MEM_NAMES,
                       label_rule="cell3d.label_from (no cristae: geometry before jcristae)"),
                  open(f"{out}/geometry.json", "w"), separators=(",", ":"))
        J.organelle_meshes(cell, np.zeros(3), out)
    lib = None
    # ---- blocks that meet the cell (so membrane-only blocks are encoded too): cell label at 100 nm samples per block
    g = np.arange(lo0[0], -lo0[0], 100.0) + 50
    Xg = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    labs = np.concatenate([cell.labels(Xg[i:i + 2_000_000]) for i in range(0, len(Xg), 2_000_000)])
    inside = Xg[labs != c3.EXT]
    cell_blocks = np.unique(np.floor(inside / BLK).astype(np.int64), axis=0)
    log(f"{len(cell_blocks)} blocks meet the cell")
    # ---- shared state for the fork pool (pre-pass and mRNA in RAM: random reads on Lustre are slow)
    pm = np.load(a.mrna); mr_off = pm["off"].astype(np.int64); mr_pts = pm["pts"].astype(np.float32); del pm
    nthr = len(mr_off) - 1; lens = np.diff(mr_off); nz = lens > 0
    mr_xmin = np.full(nthr, np.inf); mr_xmax = np.full(nthr, -np.inf)
    st = mr_off[:-1][nz]
    mr_xmin[nz] = np.minimum.reduceat(mr_pts[:, 0], st); mr_xmax[nz] = np.maximum.reduceat(mr_pts[:, 0], st)
    log(f"mRNA: {nthr:,} threads, {len(mr_pts):,} points")
    S.update(cell=cell, tij=tij, lo0=lo0, PT=PT, tiles=a.tiles, vt_of=vt_of, pre_glob=pre_glob, pre_reg=pre_reg,
             poff=poff, ptypes=ptypes, ppos=np.load(f"{a.prepass}/prepass_pos.npy"), pquat=np.load(f"{a.prepass}/prepass_quat.npy"),
             mr_off=mr_off, mr_pts=mr_pts, mr_xmin=mr_xmin, mr_xmax=mr_xmax, br=BR, env=ENV, mass=MASS, K=a.proxies,
             out=out, work=work, cell_blocks=[tuple(map(int, b)) for b in cell_blocks],
             only_block=tuple(a.only_block) if a.only_block else None)
    xs = np.unique(np.r_[cell_blocks[:, 0], np.floor((tij[:, 0] * PT + lo0[0]) / BLK).astype(np.int64),
                         np.floor((tij[:, 0] * PT + lo0[0] + PT - 1) / BLK).astype(np.int64)])
    if a.only_block: xs = [a.only_block[0]]
    elif a.only_slab: xs = a.only_slab
    # biggest slabs first (centre of the cell) for load balance
    xs = sorted(map(int, xs), key=lambda x: abs(x + 0.5))
    log(f"encoding {len(xs)} slabs with {a.workers} workers")
    T1 = time.time()
    with mp.get_context("fork").Pool(min(a.workers, len(xs))) as pool:
        for bx, n, dt in pool.imap_unordered(encode_slab, xs):
            log(f"slab {bx}: {n} blocks in {dt:.0f}s")
    log(f"blocks done in {time.time() - T1:.0f}s")
    if a.only_block or a.only_slab:
        log("partial run: index not assembled"); return
    assemble(a, out, work, NT, half_from_geo(out), time.time() - T0, int(tot.sum()), coarsened, cell)


def half_from_geo(out):
    return json.load(open(f"{out}/geometry.json"))["cube_nm"] / 2


def _jobs(f):
    if not os.path.exists(f): return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    z = np.load(f)
    return z["job_lib"].astype(np.int64), z["job_region"].astype(np.int64), z["job_placed"].astype(np.int64)


def write_library(a, out, ids, lib, stride, cnt, BR, MASS):
    I = pd.read_csv(a.ingredients, sep="\t").set_index("ing_id")
    up = pd.read_csv(a.uniprot, sep="\t", usecols=["Entry", "Protein names"]).set_index("Entry")["Protein names"]
    B_, K_, ST, types = [], [], [], []; bo = so = 0
    for i, lid in enumerate(ids):
        b, k, st, br = lib[i]; o = J.morton_order(b); b, k = b[o], k[o]
        if stride[i] > 1: b, k = b[::stride[i]], k[::stride[i]]
        qb = np.round(b / J.BEAD_Q); assert np.abs(qb).max() < 32767, lid
        B_.append(qb.astype(np.int16)); K_.append(k); ST.append(st)
        row = I.loc[lid] if lid in I.index else None
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        acc = row["uniprot"] if row is not None and isinstance(row["uniprot"], str) else ""
        genes = row["genes"] if row is not None and isinstance(row["genes"], str) else lid
        nm = up.get(acc, None) if acc else None
        nm = nm.split(" (")[0] if isinstance(nm, str) else lid
        src = (row["struct_src"].split(":")[0] if row is not None and isinstance(row["struct_src"], str) else "prepass")
        reg = {str(r): int(c) for r, c in enumerate(cnt[i]) if c}
        types.append(dict(id=lid, gene=genes.split(";")[0] if row is not None and row["kind"] == "protein" else lid, genes=genes,
                          uniprot=acc, name=nm, src="AF" if src.startswith("afdb") else ("PDB" if src == "pdb" else src),
                          pdb=row["struct_src"].split(":")[1] if row is not None and src == "pdb" else "",
                          oligomer=1, nb=int(len(b)), nnt=int((k == 1).sum()), bo=int(bo), so=int(so), sn=int(len(st)),
                          br=round(float(BR[i]), 3), rg=round(float(np.sqrt(((b - b.mean(0)) ** 2).sum(1).mean())), 3),
                          mass=round(float(MASS[i])), plddt_mean=None, plddt_frac_kept=None,
                          copies=int(cnt[i].sum()), regions=reg, **({"bead_stride": int(stride[i])} if stride[i] > 1 else {})))
        bo += len(b); so += len(st)
    B_ = np.concatenate(B_); K_ = np.concatenate(K_)
    rec = np.zeros(len(B_), dtype=[("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("k", "u1"), ("p", "u1")])
    rec["x"], rec["y"], rec["z"], rec["k"], rec["p"] = B_[:, 0], B_[:, 1], B_[:, 2], K_, 255
    rec.tofile(f"{out}/library/beads.bin"); np.concatenate(ST).astype("<f4").tofile(f"{out}/library/spheretree.bin")
    json.dump(dict(bead_q=J.BEAD_Q, n_beads=int(len(B_)), n_spheres=int(so), types=types, whole_cell=True),
              open(f"{out}/library/types.json", "w"), separators=(",", ":"))
    log(f"library: {len(ids)} types, {len(B_):,} beads, {so:,} spheres")


def split_large(out):
    """split files above SPLIT_MB into parts path.000, path.001, ... (data.ts / worker.ts read split.json); files that
    were split by an earlier run (whole file gone) keep their entry"""
    old = json.load(open(f"{out}/split.json")) if os.path.exists(f"{out}/split.json") else {}
    sp = {}
    for p in ["library/beads.bin", "library/spheretree.bin", "geometry_fields.bin", "types_tiles.bin", "organelle_meshes.bin.gz"]:
        f = f"{out}/{p}"
        if not os.path.exists(f):
            if p in old: sp[p] = old[p]
            continue
        for q in [x for x in os.listdir(os.path.dirname(f)) if x.startswith(os.path.basename(f) + ".")]:
            os.remove(os.path.join(os.path.dirname(f), q))            # stale parts of an earlier run
        if os.path.getsize(f) <= SPLIT_MB * 1e6: continue
        part = int(32e6); k = 0
        with open(f, "rb") as fh:
            while True:
                b = fh.read(part)
                if not b: break
                open(f"{f}.{k:03d}", "wb").write(b); k += 1
        os.remove(f); sp[p] = k
    json.dump(sp, open(f"{out}/split.json", "w"))
    return sp


def assemble(a, out, work, NT, half, dt_total, n_total, coarsened, cell):
    recs = [json.load(open(f"{work}/{f}")) for f in sorted(os.listdir(work)) if f.endswith(".json")]
    rows, blocks, tpairs, own = [], [], [], 0
    for r in recs:
        rows += r["rows"]; own += r["own"]
        if r["rows"]: blocks.append([*r["block"], r["n_proxy"], r["ov_bytes"], r["shard_bytes"]])
        tpairs += r["types"]
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    tid = {(x[0], x[1], x[2]): i for i, x in enumerate(rows)}
    log(f"index: {len(rows):,} tiles in {len(blocks)} blocks; {own:,} instances encoded of {n_total:,}")
    json.dump(dict(tile_nm=VT, q_org=J.Q_ORG, q_range=J.Q_RANGE, mrna_q=MR_Q, link_q=0.01, label_names=c3.LABEL_NAMES,
                   region_names=J.REGION_NAMES, mem_names=J.MEM_NAMES, label_columns=c3.LABEL_NAMES, whole_cell=True,
                   shard=dict(block_tiles=NB, path="tiles/s_{bx}_{by}_{bz}.bin",
                              note="tile = gzip member at bytes [off, off + gz_bytes) of the shard of block floor(i / block_tiles)"),
                   overview=dict(path="overview/{bx}_{by}_{bz}.bin.gz", block_nm=BLK,
                                 columns=["bx", "by", "bz", "n_proxy", "gz_bytes", "shard_bytes"], blocks=blocks,
                                 layout="gzip: u32 magic 'YCO1', u32 n_tiles, n_tiles x i32 (ix, iy, iz, count), then count x "
                                        "(f32 x, y, z, u32 w = 7 | region << 4 | radius/0.1nm << 9) per tile in that order"),
                   tiles=rows, types_tiles_u32=True,
                   tile_columns=["ix", "iy", "iz", "n_inst", "n_ghost", "n_mrna_beads", "n_link_beads", "n_vert", "n_tri",
                                 "gz_bytes", "label_mask", "off"]),
              open(f"{out}/tiles/index.json", "w"), separators=(",", ":"))
    # types_tiles: per type up to 256 tiles (one per block where it occurs, random subset), uint32 tile indices
    per = [[] for _ in range(NT)]
    for ty, x, y, z in tpairs: per[ty].append(tid[(x, y, z)])
    rng = np.random.default_rng(0); offs = np.zeros(NT + 1, np.uint32); lists = []
    for i, l in enumerate(per):
        l = np.array(sorted(l), np.uint32)
        if len(l) > 256: l = np.sort(rng.choice(l, 256, replace=False))
        lists.append(l); offs[i + 1] = offs[i] + len(l)
    with open(f"{out}/types_tiles.bin", "wb") as f:
        f.write(offs.astype("<u4").tobytes()); f.write(np.concatenate(lists).astype("<u4").tobytes())
    # named views: jenc.py's picks inside the 1.6 um detail cube (where every compartment occurs), shifted back to cell coordinates
    Cc = np.array(a.bookmark_centre, float)
    J.bookmarks(cell, Cc, 1600.0, out, n_total)
    bm = json.load(open(f"{out}/bookmarks.json"))
    for x in bm: x["target"] = [round(float(v + c), 3) for v, c in zip(x["target"], Cc)]
    bm[0].update(target=[0.0, 0.0, 0.0], normal=[0, 0, 1], up=[0, 1, 0], name="Whole cell", caption=f"Section through the whole Jurkat T cell model ({n_total / 1e6:.0f} million macromolecules). "
                 "Molecules stream in around the view centre; further out, stand-in spheres and compartment colours.",
                 width=2 * float(cell.G[0]) * 0.75, slab=60)
    json.dump(bm, open(f"{out}/bookmarks.json", "w"), indent=1)
    sp = split_large(out)
    json.dump(dict(v=time.strftime("%Y%m%d%H%M%S")), open(f"{out}/version.json", "w"))
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(out) for f in fs]
    sizes = np.array([os.path.getsize(f) for f in files])
    big = files[int(sizes.argmax())]
    meta = dict(mode="whole_cell", tiles_dir=os.path.relpath(a.tiles, V), libs=os.path.relpath(a.libs, V),
                geom=os.path.relpath(a.geom, V), n_instances=n_total, n_encoded=own, n_types=NT, tiles=len(rows),
                blocks=len(blocks), files=len(files), total_gb=round(float(sizes.sum()) / 1e9, 2),
                largest=[os.path.relpath(big, out), int(sizes.max())], split=sp, coarsened=coarsened,
                proxies_per_tile=a.proxies, field_sp_nm=a.field_sp, field_sp_thin_nm=a.field_sp_thin, encode_s=round(dt_total))
    json.dump(meta, open(f"{out}/encode_meta.json", "w"), indent=1)
    log(f"done: {meta['files']} files, {meta['total_gb']} GB, largest {meta['largest']}")


if __name__ == "__main__":
    main()
