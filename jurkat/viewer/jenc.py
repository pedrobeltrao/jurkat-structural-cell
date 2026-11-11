"""Encode a sub-volume of the packed Jurkat cell for the (yeast) WebGPU viewer.

The yeast viewer streams 200 nm tiles in a +-2.6 um grid and keeps at most ~8 M instances on the GPU, so the whole
Jurkat cell (~1e9 instances) does not fit. This encoder takes a CUBE nm cube that contains plasma membrane, cytosol,
nuclear envelope, nucleoplasm, mitochondria and ER (chosen automatically, or --centre x y z in model nm), re-centres
it on the origin and re-bins its instances into the viewer's 200 nm tiles (same tile binary 'YCT1' as webenc.py).
Whole-cell context: decimated-resolution organelle meshes of the entire cell (organelle_meshes.bin.gz), shifted the
same way. Compartment fill for cut faces: the six distance fields cropped to the cube (geometry.json 'jurkat_fields').

Outputs (OUT, default jurkat/viewer/web/data): tiles/*.bin.gz, tiles/index.json, library/{types.json,beads.bin,
spheretree.bin}, types_tiles.bin, geometry.json, geometry_fields.bin, organelle_meshes.bin.gz, bookmarks.json,
version.json, encode_meta.json.  mRNA threads are encoded as polylines (resampled to 0.6 nm). Not encoded yet: linker DNA, per-membrane lipid compositions.
usage: jenc.py [--cube 1600] [--centre x y z] [--tiles DIR] [--out DIR]"""
import os, sys, json, gzip, time, argparse, math, shutil
import numpy as np, pandas as pd, numba as nb
from skimage.measure import marching_cubes

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
sys.path.insert(0, f"{V}/jurkat/yc/pipeline")
import cell3d as c3
from config import LIB
D = f"{V}/data/derived"; W = f"{D}/wc"
VT = 200.0                      # viewer tile
WT = 500.0                      # packing tile
Q_ORG, Q_RANGE = 80.0, 360.0; Q_STEP = Q_RANGE / 65535.0
BEAD_Q = 0.005; MESH_H = 8.0; FIELD_Q = 0.01
RNAMES = ["cytosol", "nucleoplasm", "matrix", "ER_lumen", "golgi_lumen", "endo_lumen", "perinuclear",
          "mem_PM", "mem_NE", "mem_ER", "mem_MITO_OM", "mem_MITO_IM", "mem_GOLGI", "mem_ENDO"]
PRE_REG = ["chromatin", "polysome_ribosome", "cytoskeleton", "NPC"]
# v5 (jpack_test.RNAMES: + nucleolus volume region and 12 membrane-associated shells) appended AFTER the v4 ids, so viewer
# region ids 0-17 keep their v4 meaning (v4 data stay decodable; membrane-embedded = 7..13 as before) and 18-30 are new.
# 31 ids <= 32: fits the 5-bit region field of the instance word and of the overview stand-in word (w = 7 | region << 4 | ...).
V5_EXTRA = ["nucleolus", "sh_PM_cyto", "sh_PM_exo", "sh_PM_cortex", "sh_NE_lamina", "sh_NE_nuc", "sh_ER_cyto", "sh_ER_lumen",
            "sh_GOLGI_cyto", "sh_GOLGI_lumen", "sh_ENDO_cyto", "sh_ENDO_lumen", "sh_OM_cyto"]
# v6 (jpack_test.RNAMES: + speckle volume region after the nucleolus, + sh_NPC shell): speckle = viewer id 31, the last id of
# the 5-bit field. sh_NPC (NPC-proximal shell, both faces of the NE next to pores: transport receptors, basket proteins)
# shares the existing viewer region NPC (17) (PACK_ALIAS), since no id is left.
V6_EXTRA = ["speckle"]
REGION_NAMES = RNAMES + PRE_REG + V5_EXTRA + V6_EXTRA
assert len(REGION_NAMES) <= 32
PACK_ALIAS = {"sh_NPC": "NPC", "sh_PM_periph": "sh_PM_cyto", "sh_ENDO_periph": "sh_ENDO_cyto", "sh_GOLGI_periph": "sh_GOLGI_cyto",
              # v7 (jpack_test.RNAMES_V7): virtual cytosol regions (MT bands, PCM, vimentin cage) are cytosol at viewer level (no
              # region id is left in the 5-bit field; the molecules keep their own types), the NPC channel shell is NPC
              "mt_lattice": "cytosol", "mt_plus": "cytosol", "pcm": "cytosol", "vim_cage": "cytosol", "sh_NPC_fg": "NPC"}  # pack region -> viewer region of another name (v6: NPC shell -> NPC; peripheral shells share the anchored-shell ids)
V4_PACK_RNAMES = RNAMES                                        # job_region numbering of packs v1-v4


def region_map(pack_rnames):
    """viewer region id of each pack region index (tile job_region): names as jpack_test.RNAMES (v5: 'PM', 'NE', ... for
    membranes, mapped to 'mem_PM' ...) or the v4 list above (identity)"""
    m = []
    for n in pack_rnames:
        v = PACK_ALIAS.get(n, n if n in REGION_NAMES else f"mem_{n}")
        assert v in REGION_NAMES, f"pack region {n} has no viewer region id"
        m.append(REGION_NAMES.index(v))
    return np.array(m, np.int64)


PACK_SCHEMES = ["v4", "v5", "v6", "v7"]
PACK_DEFAULT = "v6"


def pack_rnames(which):
    """pack region list (tile job_region numbering): 'v4' (jenc.RNAMES, packs v1-v4), 'v5' (jpack_test.RNAMES_V5, frozen:
    the archived v5 tiles), 'v6'/'current' (jpack_test.RNAMES of the pipeline: + speckle at 8, shells shifted by one, + sh_NPC)"""
    if which == "v4": return list(V4_PACK_RNAMES)
    import jpack_test
    if which == "v5": return list(jpack_test.RNAMES_V5)
    if which == "v6": return list(jpack_test.RNAMES_V6)
    if which == "v7": return list(jpack_test.RNAMES_V7)                # v7 pack (JWC_REGIONS=v7 numbering)
    assert which == "current", which
    return list(jpack_test.RNAMES)


def face_z(cell):
    """z (model nm) of the open boundary (EM-volume truncation face, cell3d v5), or None"""
    G = cell.G
    return float(G[c3.G_ZFACE]) if G.shape[0] > c3.G_FACE and G[c3.G_FACE] > 0 else None


def clip_mesh(V_, F_, FM_, zf):
    """drop triangles whose centroid lies below the open face (z < zf; no membrane outside the imaged volume) and the
    vertices no triangle uses any more; V_ in the frame where the face is at z = zf"""
    if zf is None or not len(F_): return V_, F_, FM_
    keep = V_[F_].mean(1)[:, 2] >= zf
    F_, FM_ = F_[keep], FM_[keep]
    used = np.zeros(len(V_), bool); used[F_.ravel()] = True
    remap = np.cumsum(used) - 1
    return V_[used], remap[F_].astype(F_.dtype), FM_
MEMS = [c3.M_PM, c3.M_NE, c3.M_ER, c3.M_OM, c3.M_IM, c3.M_GOLGI, c3.M_ENDO]      # face_mem = index in this list
MEM_NAMES = ["PM", "NE", "ER", "MITO_OM", "MITO_IM", "GOLGI", "ENDO"]
GOLGI_DEEP_BAND = 320.0         # v7: inside band of the exported deep Golgi field (cisternal depth)
BANDS = [(180.0, 40.0), (100.0, 100.0), (40.0, 40.0), (40.0, 40.0), (40.0, 40.0), (40.0, 40.0), (40.0, 40.0)]   # per field (in, out) nm
AL = 16                         # F offsets (geometry.json heads) are multiples of AL: float32 represents them exactly up to 2^28


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ encoding helpers (webenc.py format)
def quant_pos(p, lo):
    u = np.round((p - (lo - Q_ORG)) / Q_STEP)
    assert u.min(initial=0) >= 0 and u.max(initial=0) <= 65535, (u.min(), u.max())
    return u.astype(np.uint32)


def quat_encode(q):
    q = np.asarray(q, np.float64); q = q / np.linalg.norm(q, axis=1, keepdims=True)
    idx = np.argmax(np.abs(q), axis=1); sgn = np.sign(q[np.arange(len(q)), idx]); sgn[sgn == 0] = 1
    q = q * sgn[:, None]
    rest = np.array([[j for j in range(4) if j != i] for i in range(4)])[idx]
    v = np.take_along_axis(q, rest, 1)
    iv = np.round(v / (1.0 / math.sqrt(2.0)) * 32767).astype(np.int16)
    return idx.astype(np.uint32), iv


def pack_instances(types, pos, quat, region, flags, lo):
    n = len(types)
    if n == 0:
        return np.zeros((0, 4), np.uint32)
    u = quant_pos(pos, lo); qi, qv = quat_encode(quat); qv = qv.astype(np.int32) & 0xFFFF
    assert types.max() < 16384 and region.max() < 32
    w = np.zeros((n, 4), np.uint32)
    w[:, 0] = types.astype(np.uint32) | (qi << 14) | (region.astype(np.uint32) << 16) | (flags.astype(np.uint32) << 21)
    w[:, 1] = u[:, 0] | (u[:, 1] << 16); w[:, 2] = u[:, 2] | (qv[:, 0].astype(np.uint32) << 16)
    w[:, 3] = qv[:, 1].astype(np.uint32) | (qv[:, 2].astype(np.uint32) << 16)
    return w


def pad4(b):
    return b + b"\0" * ((-len(b)) % 4)


def morton_order(b):
    lo, hi = b.min(0), b.max(0); g = ((b - lo) / np.maximum(hi - lo, 1e-6) * 1023).astype(np.uint64)
    def spread(x):
        x = (x | (x << 16)) & 0x030000FF; x = (x | (x << 8)) & 0x0300F00F; x = (x | (x << 4)) & 0x030C30C3; x = (x | (x << 2)) & 0x09249249
        return x
    code = spread(g[:, 0]) | (spread(g[:, 1]) << np.uint64(1)) | (spread(g[:, 2]) << np.uint64(2))
    return np.argsort(code, kind="stable")


def pack_polylines(chains, lo, q):
    """webenc.pack_polylines: exact delta coding of quantised coordinates (int8 deltas)"""
    if not chains:
        return b"", 0, 0
    lens = np.array([len(c) for c in chains], np.uint32); starts, deltas = [], []
    for c in chains:
        P = np.round((np.asarray(c, np.float64) - lo) / q).astype(np.int64)
        starts.append(P[0]); d = np.diff(P, axis=0)
        assert np.abs(d).max(initial=0) <= 127, np.abs(d).max()
        deltas.append(d.astype(np.int8))
    b = lens.tobytes() + np.array(starts, np.int32).tobytes() + np.concatenate(deltas).tobytes()
    return pad4(b), len(chains), int(lens.sum())


def mrna_pieces(C, clo, chi, step=0.6, piece=64):
    """mRNA threads (prepass/mrna_cell1.npz, 3 nm beads) inside the cube, resampled to ~1 nt (0.6 nm) like the yeast
    threads, cut into pieces of `piece` points; returns list of (n,3) arrays in the shifted frame"""
    z = np.load(f"{D}/prepass/mrna_cell1.npz", mmap_mode="r"); off = z["off"]; pts = z["pts"]
    out = []
    for i in range(len(off) - 1):
        P = np.asarray(pts[off[i]:off[i + 1]], np.float64)
        inside = np.all((P >= clo) & (P < chi), axis=1)
        if not inside.any(): continue
        # contiguous runs inside the cube
        e = np.flatnonzero(np.diff(np.r_[0, inside.astype(np.int8), 0]))
        for a0, a1 in zip(e[::2], e[1::2]):
            R = P[a0:a1]
            if len(R) < 2: continue
            seg = np.linalg.norm(np.diff(R, axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]
            t = np.arange(0, s[-1], step)
            Rr = np.stack([np.interp(t, s, R[:, k]) for k in range(3)], 1) - C
            for p0 in range(0, len(Rr) - 1, piece - 1):
                q = Rr[p0:p0 + piece]
                if len(q) >= 2: out.append(q)
    return out


# ------------------------------------------------------------------ sub-volume choice
def choose_centre(cell, cube, step=400.0):
    R = cell.G[0]; g = np.arange(-R, R + 1, step)
    C = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    m = cell.mems(C); lab = cell.labels(C)
    C = C[(lab != c3.EXT) & (lab != c3.OUTV) & (m[:, c3.M_PM] < 0.45 * cube) & (np.abs(m[:, c3.M_NE]) < 0.45 * cube)]
    want = [c3.CYTO, c3.NUCP, c3.MATRIX, c3.ERL, c3.EXT]
    s = (np.arange(16) + 0.5) / 16 * cube - cube / 2
    S = np.stack(np.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3)
    best, bc = -1, None
    for c in C:
        l = cell.labels(S + c); f = np.array([(l == w).mean() for w in want])
        score = (f > 0.01).sum() + min(f[2], 0.08) * 10 + min(f[3], 0.05) * 10 + min(f[1], 0.3) + min(f[0], 0.3)
        if score > best: best, bc = score, c
    log(f"sub-volume centre {bc.round(0).tolist()} (score {best:.2f})")
    return bc


# ------------------------------------------------------------------ fields cropped to the cube
def crop_field(G, k, lo, hi, band, o=None):
    o = c3.G_N + 1 + 12 * k if o is None else o; h = G[o:o + 12]        # o: a 12-value header anywhere in G (v7 cisternae)
    org, sp = h[0:3], h[3]; nb = h[4:7].astype(int); B = int(h[7]); n1 = B + 1
    idx = G[int(h[10]):int(h[10]) + int(np.prod(nb))].astype(np.int64).reshape(nb)
    data = G[int(h[11]):]
    b0 = np.clip(np.floor((lo - org) / (sp * B)).astype(int), 0, nb); b1 = np.clip(np.ceil((hi - org) / (sp * B)).astype(int), 0, nb)
    sub = idx[b0[0]:b1[0], b0[1]:b1[1], b0[2]:b1[2]]; snb = np.array(sub.shape)
    bin_, bout = band; keep = []; new = np.empty(sub.shape, np.int64)
    for i in np.ndindex(*sub.shape):
        bi = sub[i]
        if bi < 0:
            new[i] = bi; continue
        blk = data[bi * n1 ** 3:(bi + 1) * n1 ** 3]
        if blk.max() <= -bin_: new[i] = -1
        elif blk.min() >= bout: new[i] = -2
        else: new[i] = len(keep); keep.append(np.clip(blk, -bin_, bout))
    dat = np.round(np.concatenate(keep) / FIELD_Q).astype(np.int16) if keep else np.zeros(0, np.int16)
    org2 = org + b0 * sp * B
    return [float(x) for x in org2], float(sp), snb.tolist(), B, new.ravel(), dat, float(h[8]), float(h[9])


def n_fields_of(G):
    return int(G[c3.G_N]) if G.shape[0] > c3.G_N else 0


def _al(x):
    return -(-int(x) // AL) * AL


def cristae_section(G, C=np.zeros(3), Bc=4):
    """the cristae section of G (cell3d.crista_lumen; jcristae.py) for the viewer, shifted by -C: per-mitochondrion
    parameters (n x 12: axis xyz, phase, spacing S, junction direction xyz, centroid xyz, enabled) and the dense 48 nm
    id grid stored in bricks of Bc^3 cells (index -1 = brick without any mitochondrion id, else slot; slot data = ids,
    -1 = none). None when G has no cristae."""
    if G.shape[0] <= c3.G_CRISTAE or G[c3.G_CRISTAE] <= 0: return None
    o = int(G[c3.G_CRISTAE]); n = int(G[o]); org = G[o + 1:o + 4] - C; h = float(G[o + 4]); dims = G[o + 5:o + 8].astype(int)
    par = G[o + 8:o + 8 + 12 * n].reshape(n, 12).copy()
    ids = G[o + 8 + 12 * n:o + 8 + 12 * n + int(np.prod(dims))].reshape(dims)
    par[:, 3] -= par[:, 0:3] @ C; par[:, 8:11] -= C                 # t = a.(p' + C) - phase = a.p' - (phase - a.C)
    nb = -(-dims // Bc); pad = np.full(nb * Bc, -1.0); pad[:dims[0], :dims[1], :dims[2]] = ids
    bl = pad.reshape(nb[0], Bc, nb[1], Bc, nb[2], Bc).transpose(0, 2, 4, 1, 3, 5).reshape(-1, Bc ** 3)
    full = (bl >= 0).any(1); idx = np.full(len(bl), -1, np.int64); idx[full] = np.arange(full.sum())
    return dict(n=n, origin=[float(x) for x in org], h=h, dims=dims.tolist(), B=Bc, nb=nb.tolist(),
                par=par.astype(np.float32), idx=idx, dat=bl[full].astype(np.int16),
                consts=dict(CR_W=c3.CR_W, CR_GAP=c3.CR_GAP, CR_JW=c3.CR_JW))


def write_fields_bin(out, items, cr):
    """geometry_fields.bin + the geometry.json 'fields' dict. items: per field (head[:10] = origin xyz, spacing, nbx, nby,
    nbz, B, clamp_in, clamp_out; int brick index; int16 node data in FIELD_Q). Offsets into the app's Float32Array F are
    rounded up to multiples of AL (exact in float32: v4 builds had odd offsets > 2^25, e.g. the Golgi index, which F
    shifted by one word). The cristae section (cristae_section) follows the fields."""
    n = len(items); off = _al(1 + 12 * n); heads, n_index, n_data, parts = [], [], [], []
    for h10, idx, dat in items:
        io = off; off = _al(off + len(idx)); do = off; off = _al(off + len(dat))
        heads.append([float(x) for x in h10] + [float(io), float(do)])
        n_index.append(int(len(idx))); n_data.append(int(len(dat)))
        parts += [idx.astype("<i4").tobytes(), dat.astype("<i2").tobytes() + b"\0" * ((-2 * len(dat)) % 4)]
    fd = dict(file="geometry_fields.bin", names=(c3.FIELD_NAMES + ["golgi_deep"])[:n], head=heads, n_index=n_index, n_data=n_data, q=FIELD_Q,
              layout="per field: int32 brick index (nbx*nby*nbz, x-major; -1 inside, -2 outside, else brick id), then int16 node "
                     "values ((B+1)^3 per brick, x-major) in q nm, padded to 4 bytes; head = origin xyz, spacing, nbx, nby, nbz, B, "
                     "clamp_in, clamp_out, idx_off, data_off (word offsets of the index and data in the app's Float32Array F, "
                     "multiples of 16; the file itself is the sections back to back). Then the optional cristae section: "
                     "float32 params (n x 12), int32 brick index, int16 ids (padded), placed at cristae.off_par/off_idx/off_dat of F.")
    if cr is not None:
        po = off; off = _al(off + cr["par"].size); xo = off; off = _al(off + len(cr["idx"])); do = off; off = _al(off + cr["dat"].size)
        parts += [cr["par"].astype("<f4").tobytes(), cr["idx"].astype("<i4").tobytes(),
                  cr["dat"].astype("<i2").tobytes() + b"\0" * ((-2 * cr["dat"].size) % 4)]
        fd["cristae"] = dict(n=cr["n"], origin=cr["origin"], h=cr["h"], dims=cr["dims"], B=cr["B"], nb=cr["nb"],
                             n_par=int(cr["par"].size), n_index=int(len(cr["idx"])), n_data=int(cr["dat"].size),
                             off_par=po, off_idx=xo, off_dat=do, consts=cr["consts"],
                             rule="cell3d.crista_lumen: params p = F[off_par + 12 k ..]; cell c = floor((x - origin) / h); brick b = c // B; "
                                  "slot = F[off_idx + (bx nby + by) nbz + bz]; id k = F[off_dat + slot B^3 + (lx B + ly) B + lz]")
    fd["total"] = int(off)
    open(f"{out}/geometry_fields.bin", "wb").write(b"".join(parts))
    return fd


def speckles_json(cell, C=np.zeros(3)):
    """nuclear speckles (cell3d v6 G[G_SPECKLE] section: n, then n x (cx, cy, cz, r) nm) shifted by -C, for geometry.json
    'speckles'; n = 0 when the geometry has none (G[4] == 0, v5 and older)"""
    G = cell.G
    sp = np.zeros((0, 4))
    if G.shape[0] > c3.G_SPECKLE and G[c3.G_SPECKLE] > 0:
        o = int(G[c3.G_SPECKLE]); n = int(G[o]); sp = G[o + 1:o + 1 + 4 * n].reshape(n, 4).copy(); sp[:, :3] -= C
    return dict(n=int(len(sp)), spheres=sp.round(3).ravel().tolist(),
                note="nuclear speckles (cell3d v6 in_speckle): label speckle inside a sphere (cx, cy, cz, r nm; model frame of "
                     "the viewer), within the nucleoplasm branch after the nucleolus test; no membrane")


def face_json(cell, C=np.zeros(3)):
    zf = face_z(cell)
    return dict(on=zf is not None, z=(zf - float(C[2])) if zf is not None else 0.0,
                note="open boundary: cell 1 is truncated by the EM volume's z = 0 face; below z the label is outside_EM_volume, "
                     "no membrane on the face (cell3d v5 G_FACE / G_ZFACE)")


def web_fields(cell, C, cube, out):
    lo, hi = C - cube / 2 - 200, C + cube / 2 + 200
    items = []
    for k, band in enumerate(BANDS[:n_fields_of(cell.G)]):
        org, sp, nb, B, idx, dat, cin, cout = crop_field(cell.G, k, lo, hi, band)
        org = [org[i] - C[i] for i in range(3)]
        items.append((org + [sp, float(nb[0]), float(nb[1]), float(nb[2]), float(B), band[0], band[1]], idx, dat))
    if cell.G.shape[0] > c3.G_GOLGI_CIS and cell.G[c3.G_GOLGI_CIS] > 0 and len(items) == 7:
        # v7 cisternal Golgi: the deep Golgi SDF of the cisternae section as field 7 ("golgi_deep"); the app evaluates
        # cell3d.golgi_sdf (layers W/S, CIS_CLR) and the intercisternal label GIC (13) when the fields hold 8 entries
        o = int(cell.G[c3.G_GOLGI_CIS]); band = (min(GOLGI_DEEP_BAND, float(cell.G[o + 2 + 8])), 40.0)
        org, sp, nb, B, idx, dat, cin, cout = crop_field(cell.G, None, lo, hi, band, o=o + 2)
        org = [org[i] - C[i] for i in range(3)]
        items.append((org + [sp, float(nb[0]), float(nb[1]), float(nb[2]), float(B), band[0], band[1]], idx, dat))
    return write_fields_bin(out, items, cristae_section(cell.G, C))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=float, default=1600.0)
    ap.add_argument("--centre", type=float, nargs=3)
    ap.add_argument("--tiles", default=f"{W}/tiles")
    ap.add_argument("--out", default=f"{V}/jurkat/viewer/web/data")
    ap.add_argument("--regions", default=PACK_DEFAULT, choices=PACK_SCHEMES, help="job_region numbering of the packing tiles")
    a = ap.parse_args()
    out = a.out; shutil.rmtree(out, ignore_errors=True); os.makedirs(f"{out}/tiles"); os.makedirs(f"{out}/library")
    z = np.load(f"{D}/jgeom_cell1.npz"); cell = c3.Cell(z["G"], z["pores"])
    rmap = region_map(pack_rnames(a.regions)); zf = face_z(cell)
    C = np.array(a.centre) if a.centre else choose_centre(cell, a.cube)
    C = np.round(C / VT) * VT                                 # cube faces on viewer tile boundaries
    half = a.cube / 2; clo, chi = C - half, C + half
    M = pd.read_parquet(f"{W}/tiles.parquet"); meta = json.load(open(f"{W}/tiles_meta.json")); lo0 = np.array(meta["lo0"])
    L = json.load(open(f"{W}/libs.json")); libs = L["libs"]; PL = json.load(open(f"{W}/prepass_libs.json"))
    poff = np.load(f"{W}/prepass_off.npy")
    ptypes = np.load(f"{W}/prepass_types.npy", mmap_mode="r"); ppos = np.load(f"{W}/prepass_pos.npy", mmap_mode="r"); pquat = np.load(f"{W}/prepass_quat.npy", mmap_mode="r")
    # ---- gather instances in the cube (global ids: packed libs 0..len(libs)-1, pre-pass libs after)
    T, P, Q, R = [], [], [], []
    pre_reg = np.array([REGION_NAMES.index("chromatin") if (n == "NUCLEOSOME" or n.startswith("H1-")) else
                        REGION_NAMES.index("polysome_ribosome") if n.startswith("RIBO") else
                        REGION_NAMES.index("NPC") if n.startswith("NPC") else REGION_NAMES.index("cytoskeleton") for n in PL])
    tij = M[["ti", "tj", "tk"]].values; tlo = tij * WT + lo0
    sel = np.where(np.all((tlo < chi) & (tlo + WT > clo), axis=1))[0]
    for ti in sel:
        f = f"{a.tiles}/{ti}.npz"
        if os.path.exists(f):
            t = np.load(f); reg = rmap[np.repeat(t["job_region"], t["job_placed"]).astype(np.int64)]
            k = np.all((t["pos"] >= clo) & (t["pos"] < chi), axis=1)
            T.append(t["types"][k].astype(np.int64)); P.append(t["pos"][k].astype(np.float64)); Q.append(t["quat"][k].astype(np.float64)); R.append(reg[k])
        p0, p1 = poff[ti], poff[ti + 1]
        if p1 > p0:
            pp = np.asarray(ppos[p0:p1], np.float64); k = np.all((pp >= clo) & (pp < chi), axis=1)
            ty = np.asarray(ptypes[p0:p1])[k].astype(np.int64)
            T.append(ty + len(libs)); P.append(pp[k]); Q.append(np.asarray(pquat[p0:p1], np.float64)[k]); R.append(pre_reg[ty])
    T, P, Q, R = (np.concatenate(x) for x in (T, P, Q, R))
    log(f"cube {a.cube:.0f} nm at {C.tolist()}: {len(T):,} instances from {len(sel)} packing tiles")
    # ---- types present -> viewer type ids
    allnames = libs + PL
    uniq, inv = np.unique(T, return_inverse=True); vt = inv.astype(np.int64)
    ids = [allnames[u] for u in uniq]
    BR = np.zeros(len(ids))
    beads, kinds, sts = [], [], []
    for i, lid in enumerate(ids):
        zz = np.load(f"{LIB}/{lid}.npz"); b = zz["beads"].astype(np.float64)
        beads.append(b); kinds.append(zz["kind"].astype(np.uint8) if "kind" in zz.files else np.zeros(len(b), np.uint8))
        sts.append(np.c_[zz["st_c"], zz["st_r"]].astype(np.float32))
        BR[i] = float((np.linalg.norm(zz["st_c"], axis=1) + zz["st_r"]).max())
    nbeads = sum(len(b) for b in beads); log(f"{len(ids)} types, {nbeads:,} library beads (viewer offset limit 2^26)")
    # ---- near-membrane flags
    mm = cell.mems(P); dmin = np.abs(mm[:, MEMS]).min(1); flags = (dmin < BR[vt] + 3.0).astype(np.uint8)
    # ---- shift + bin into viewer tiles
    Ps = P - C; key = np.floor(Ps / VT).astype(np.int64)
    order = np.lexsort((R, vt, key[:, 2], key[:, 1], key[:, 0]))
    vt, Ps, Q, R, flags, key, Porig = vt[order], Ps[order], Q[order], R[order], flags[order], key[order], P[order]
    tk, first = np.unique(key, axis=0, return_index=True); last = np.r_[first[1:], len(key)]
    n1 = int(round(a.cube / VT)); g = np.arange(n1) - n1 // 2
    tiles = [tuple(int(x) for x in t) for t in np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)]
    rng_ = {tuple(int(x) for x in t): (f0, f1) for t, f0, f1 in zip(tk, first, last)}
    lab_names = c3.LABEL_NAMES
    pieces = mrna_pieces(C, clo, chi)
    pkey = {}
    for j, q in enumerate(pieces):
        pkey.setdefault(tuple(int(x) for x in np.floor(q[len(q) // 2] / VT)), []).append(j)
    log(f"mRNA: {len(pieces):,} pieces, {sum(len(q) for q in pieces):,} points in the cube")
    rows, types_tiles = [], [[] for _ in ids]
    t0 = time.time()
    for it, t in enumerate(tiles):
        lo = np.array(t, float) * VT; hi = lo + VT
        f0, f1 = rng_.get(t, (0, 0))
        inst = pack_instances(vt[f0:f1], Ps[f0:f1], Q[f0:f1], R[f0:f1], flags[f0:f1], lo)
        for ty in np.unique(vt[f0:f1]): types_tiles[ty].append(it)
        # ghosts: near-membrane instances of neighbour tiles overlapping this tile
        gh = []
        for d in np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 3):
            if not d.any(): continue
            nt = tuple(int(x) for x in np.array(t) + d)
            if nt not in rng_: continue
            n0, nn1 = rng_[nt]; m = flags[n0:nn1] == 1
            if not m.any(): continue
            p = Ps[n0:nn1][m]; br = BR[vt[n0:nn1][m]] + 3.0
            ok = np.all((p > lo - br[:, None]) & (p < hi + br[:, None]), axis=1)
            if ok.any():
                ii = np.arange(n0, nn1)[m][ok]; gh.append(ii)
        gh = np.concatenate(gh) if gh else np.zeros(0, np.int64)
        gh = gh[np.all((Ps[gh] > lo - Q_ORG + 1) & (Ps[gh] < lo - Q_ORG + Q_RANGE - 1), axis=1)]
        ghost = pack_instances(vt[gh], Ps[gh], Q[gh], R[gh], flags[gh], lo)
        # membrane mesh at MESH_H (original frame), shifted
        m_ = int(VT / MESH_H) + 1; gm = np.arange(m_) * MESH_H
        X = np.stack(np.meshgrid(lo[0] + C[0] + gm, lo[1] + C[1] + gm, lo[2] + C[2] + gm, indexing="ij"), -1).reshape(-1, 3)
        S = cell.mems(X).reshape(m_, m_, m_, c3.NMEM)
        Vs, Fs, FM, nv = [], [], [], 0
        for mi, m in enumerate(MEMS):
            s = S[..., m]
            if s.min() < 0 < s.max():
                v, fc, _, _ = marching_cubes(s, 0.0, spacing=(MESH_H,) * 3)
                Vs.append(v + lo); Fs.append(fc + nv); FM.append(np.full(len(fc), mi, np.uint8)); nv += len(v)
        Vm = np.concatenate(Vs) if Vs else np.zeros((0, 3)); Fm = np.concatenate(Fs) if Fs else np.zeros((0, 3), np.int64)
        FMm = np.concatenate(FM) if FM else np.zeros(0, np.uint8)
        Vm, Fm, FMm = clip_mesh(Vm, Fm, FMm, None if zf is None else zf - C[2])
        vbytes = pad4(quant_pos(Vm, lo).astype("<u2").tobytes()) if len(Vm) else b""
        big = len(Vm) > 65535
        fbytes = pad4(Fm.astype("<u4" if big else "<u2").tobytes()) if len(Fm) else b""
        mbytes = pad4(FMm.astype("u1").tobytes()) if len(Fm) else b""
        mrb, nm, nmb = pack_polylines([pieces[j] for j in pkey.get(t, [])], lo, 0.005)
        hdr = np.zeros(16, np.uint32); hdr[0] = int.from_bytes(b"YCT1", "little"); hdr[1:4] = np.array(t, np.int32).view(np.uint32)
        hdr[4:15] = [len(inst), len(ghost), nm, nmb, 0, 0, len(Vm), len(Fm), 1 if big else 0, 0, 0]
        blob = hdr.tobytes() + inst.astype("<u4").tobytes() + ghost.astype("<u4").tobytes() + mrb + vbytes + fbytes + mbytes
        gzb = gzip.compress(blob, compresslevel=6); open(f"{out}/tiles/{t[0]}_{t[1]}_{t[2]}.bin.gz", "wb").write(gzb)
        # label mask from a coarse sample
        s5 = (np.arange(5) + 0.5) * VT / 5
        lab = cell.labels(np.stack(np.meshgrid(lo[0] + C[0] + s5, lo[1] + C[1] + s5, lo[2] + C[2] + s5, indexing="ij"), -1).reshape(-1, 3))
        lmask = int(sum(1 << int(l) for l in np.unique(lab)))
        rows.append([*t, int(len(inst)), int(len(ghost)), int(nmb), 0, int(len(Vm)), int(len(Fm)), int(len(gzb)), lmask])
        if it % 100 == 0: log(f"tile {it}/{len(tiles)} {time.time() - t0:.0f}s")
    json.dump(dict(tile_nm=VT, q_org=Q_ORG, q_range=Q_RANGE, mrna_q=0.005, link_q=0.01, label_names=lab_names,
                   region_names=REGION_NAMES, mem_names=MEM_NAMES, label_columns=lab_names, tiles=rows,
                   tile_columns=["ix", "iy", "iz", "n_inst", "n_ghost", "n_mrna_beads", "n_link_beads", "n_vert", "n_tri", "gz_bytes", "label_mask"]),
              open(f"{out}/tiles/index.json", "w"), separators=(",", ":"))
    offs = np.zeros(len(ids) + 1, np.uint32); offs[1:] = np.cumsum([len(x) for x in types_tiles])
    with open(f"{out}/types_tiles.bin", "wb") as f:
        f.write(offs.astype("<u4").tobytes()); f.write(np.concatenate([np.array(x, "<u2") for x in types_tiles]).tobytes())
    # ---- library
    I = pd.read_csv(f"{D}/ingredients_v0.tsv", sep="\t").set_index("ing_id")
    up = pd.read_csv(f"{V}/data/reference/uniprot_human_full.tsv.gz", sep="\t", usecols=["Entry", "Protein names"]).set_index("Entry")["Protein names"]
    cnt = np.bincount(vt, minlength=len(ids)); rc = {}
    pr = pd.DataFrame(dict(t=vt, r=R)).value_counts()
    for (t_, r_), c_ in pr.items(): rc.setdefault(int(t_), {})[str(int(r_))] = int(c_)
    B, K, ST, types = [], [], [], []; bo = so = 0
    for i, lid in enumerate(ids):
        b, k = beads[i], kinds[i]; o = morton_order(b); b, k = b[o], k[o]
        qb = np.round(b / BEAD_Q); assert np.abs(qb).max() < 32767, lid
        B.append(qb.astype(np.int16)); K.append(k); ST.append(sts[i])
        row = I.loc[lid] if lid in I.index else None
        acc = row["uniprot"] if row is not None and isinstance(row["uniprot"], str) else ""
        genes = row["genes"] if row is not None and isinstance(row["genes"], str) else lid
        name = up.get(acc, None) if acc else None
        name = name.split(" (")[0] if isinstance(name, str) else lid
        src = (row["struct_src"].split(":")[0] if row is not None else "prepass")
        types.append(dict(id=lid, gene=genes.split(";")[0] if row is not None and row["kind"] == "protein" else lid, genes=genes, uniprot=acc,
                          name=name, src="AF" if src.startswith("afdb") else ("PDB" if src == "pdb" else src),
                          pdb=row["struct_src"].split(":")[1] if row is not None and src == "pdb" else "",
                          oligomer=1, nb=int(len(b)), nnt=int((k == 1).sum()), bo=int(bo), so=int(so), sn=int(len(sts[i])),
                          br=round(float(BR[i]), 3), rg=round(float(np.sqrt(((b - b.mean(0)) ** 2).sum(1).mean())), 3),
                          mass=round(float(np.where(k == 0, 110.0, 330.0).sum())), plddt_mean=None, plddt_frac_kept=None,
                          copies=int(cnt[i]), regions=rc.get(i, {})))
        bo += len(b); so += len(sts[i])
    B = np.concatenate(B); K = np.concatenate(K)
    rec = np.zeros(len(B), dtype=[("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("k", "u1"), ("p", "u1")])
    rec["x"], rec["y"], rec["z"], rec["k"], rec["p"] = B[:, 0], B[:, 1], B[:, 2], K, 255
    rec.tofile(f"{out}/library/beads.bin"); np.concatenate(ST).astype("<f4").tofile(f"{out}/library/spheretree.bin")
    json.dump(dict(bead_q=BEAD_Q, n_beads=int(len(B)), n_spheres=int(so), types=types), open(f"{out}/library/types.json", "w"), separators=(",", ":"))
    log(f"library: {len(ids)} types, {len(B):,} beads, {so:,} spheres")
    # ---- geometry (fields cropped to the cube) + whole-cell organelle meshes
    fields = web_fields(cell, C, a.cube, out)
    pores = cell.pores.copy(); pores[:, :3] -= C
    json.dump(dict(kind="jurkat_fields", centre_nm=C.tolist(), cube_nm=a.cube, R=float(cell.G[0]), fields=fields,
                   pores=pores.round(4).ravel().tolist(), consts=dict(PERI_W=c3.PERI_W, IMS_W=c3.IMS_W, PORE_R=c3.PORE_R, HALF=c3.HALF,
                               GOLGI_W=float(cell.G[int(cell.G[c3.G_GOLGI_CIS])]) if cell.G[c3.G_GOLGI_CIS] > 0 else 0.0,
                               GOLGI_S=float(cell.G[int(cell.G[c3.G_GOLGI_CIS]) + 1]) if cell.G[c3.G_GOLGI_CIS] > 0 else 0.0,
                               CIS_CLR=c3.CIS_CLR),
                   face=face_json(cell, C), speckles=speckles_json(cell, C), label_names=lab_names, mem_names=MEM_NAMES, region_names=REGION_NAMES,
                   label_rule=LABEL_RULE), open(f"{out}/geometry.json", "w"), separators=(",", ":"))
    organelle_meshes(cell, C, out)
    bookmarks(cell, C, a.cube, out, len(T))
    json.dump(dict(v=time.strftime("%Y%m%d%H%M%S")), open(f"{out}/version.json", "w"))
    json.dump(dict(centre_nm=C.tolist(), cube_nm=a.cube, n_instances=int(len(T)), n_types=len(ids), n_beads=int(len(B)),
                   tiles=len(tiles), gz_mb=round(sum(r[-2] for r in rows) / 1e6, 1)), open(f"{out}/encode_meta.json", "w"), indent=1)
    log("done", out)


LABEL_RULE = ("cell3d.label_from (v6): outside_EM_volume below the open face (face.z); exterior if -field(cell) < 0; perinuclear if "
              "|field(ne_mid)| < PERI_W/2 (pore torus); nucleoplasm side: pore channel, nucleolus if field(nucleolus) < 0, speckle "
              "if inside a speckles sphere, else nucleoplasm; mito: matrix if max(mito + IMS_W, -crista_lumen) < 0 else IMS; then ER, golgi, endo; else cytosol")


@nb.njit(cache=False)
def _field_pts(P, G, k):
    out = np.empty(len(P), np.float32)
    for i in range(len(P)): out[i] = c3.field_sdf(P[i, 0], P[i, 1], P[i, 2], G, k)
    return out


def organelle_meshes(cell, C, out, h=64.0):
    """whole-cell membrane meshes at h nm (context in the overview), shifted by -C; clipped at the open face"""
    R = cell.G[0]; g = np.arange(-R, R + h, h)
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    S = np.empty((len(X), c3.NMEM), np.float32)
    for i in range(0, len(X), 4_000_000): S[i:i + 4_000_000] = cell.mems(X[i:i + 4_000_000])
    if cell.G.shape[0] > c3.G_GOLGI_CIS and cell.G[c3.G_GOLGI_CIS] > 0:
        # v7 cisternal Golgi: 20 nm cisternae alias at h = 64 nm; the context mesh shows the Golgi envelope (blob field = outer
        # membrane of the outermost cisterna) instead
        S[:, c3.M_GOLGI] = _field_pts(X, cell.G, c3.F_GOLGI)
    S = S.reshape(len(g), len(g), len(g), c3.NMEM); res = []
    for mi, m in enumerate(MEMS):
        if m == c3.M_IM: continue                              # at 64 nm the IM coincides with the OM
        s = S[..., m] if m != c3.M_NE else np.minimum(S[..., m], 1e3)
        if not (s.min() < 0 < s.max()): continue
        v, fc, _, _ = marching_cubes(s, 0.0, spacing=(h,) * 3)
        v, fc, _ = clip_mesh(v - R, fc, np.zeros(len(fc), np.uint8), face_z(cell))
        if not len(fc): continue
        res.append((mi, (v - C).astype("<f4"), fc.astype("<u4")))
    with gzip.open(f"{out}/organelle_meshes.bin.gz", "wb", compresslevel=6) as f:
        hdr = np.zeros(1 + 3 * len(res), np.uint32); hdr[0] = len(res)
        for i, (m, V_, F_) in enumerate(res): hdr[1 + 3 * i:4 + 3 * i] = [m, len(V_), len(F_)]
        f.write(hdr.tobytes())
        for m, V_, F_ in res: f.write(V_.tobytes()); f.write(F_.tobytes())
    log("organelle meshes:", [(MEM_NAMES[m], len(F_)) for m, _, F_ in res])


def bookmarks(cell, C, cube, out, n_inst):
    unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)
    s = (np.arange(24) + 0.5) / 24 * cube - cube / 2
    S = np.stack(np.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3)
    lab = cell.labels(S + C); mm = cell.mems(S + C)
    def pick(mask, score):
        i = np.where(mask)[0]
        return S[i[np.argmax(score[i])]] if len(i) else np.zeros(3)
    def grad(p, m, e=4.0):
        q = p + C; gg = np.array([cell.mems((q + d)[None])[0, m] - cell.mems((q - d)[None])[0, m] for d in np.eye(3) * e])
        return unit(gg)
    bm = [dict(name="Sub-volume", caption=f"A {cube / 1000:.1f} um cube of the Jurkat T cell model ({n_inst / 1e6:.1f} million macromolecules): "
               "plasma membrane, cytoplasm, nuclear envelope and nucleus.", target=[0, 0, 0], normal=[0, 0, 1], up=[0, 1, 0], width=cube * 1.1, slab=60)]
    p = pick(np.abs(mm[:, c3.M_PM]) < 30, -np.linalg.norm(S, axis=1)); u = grad(p, c3.M_PM)
    bm.append(dict(name="Plasma membrane", caption="Plasma membrane with T-cell surface receptors (cortex below).", target=p, normal=unit(np.cross(u, [0.3, 0.5, 0.8])), up=u, width=300, slab=40))
    p = pick(lab == c3.CYTO, mm[:, c3.M_NE].clip(max=300) + np.minimum(mm[:, c3.M_PM], 300)); bm.append(dict(name="Cytoplasm", caption="Crowded cytoplasm: polysomes (ribosomes on mRNA), cytoskeleton and soluble proteins.", target=p, normal=[0, 0, 1], up=[0, 1, 0], width=200, slab=30))
    p = pick(np.abs(mm[:, c3.M_NE]) < 20, -np.linalg.norm(S, axis=1)); u = grad(p, c3.M_NE)
    bm.append(dict(name="Nuclear envelope", caption="Outer and inner nuclear membranes with nuclear pore complexes.", target=p, normal=unit(np.cross(u, [0.3, 0.5, 0.8])), up=u, width=400, slab=50))
    pr = cell.pores[:, :3] - C; ins = np.all(np.abs(pr) < cube / 2 - 150, axis=1)
    if ins.any():                                          # NPC (pre-pass at the pores): view down the pore axis
        k = np.flatnonzero(ins)[np.argmin(np.linalg.norm(pr[ins], axis=1))]; u = unit(cell.pores[k, 3:6])
        bm.append(dict(name="Nuclear pore", caption="Nuclear pore complex (PDB 7R5K scaffold) seen from the cytoplasm, set in a pore of the nuclear envelope.",
                       target=pr[k], normal=-u, up=unit(np.cross(u, [0.3, 0.5, 0.8])), width=250, slab=120))   # normal = camera forward
    p = pick(lab == c3.NUCP, -mm[:, c3.M_NE].clip(-400, 0) * 0 + (-np.abs(mm[:, c3.M_NE] + 300)))
    bm.append(dict(name="Chromatin", caption="Nucleoplasm: nucleosome chains with linker histone H1 and nuclear proteins.", target=p, normal=[0, 0, 1], up=[0, 1, 0], width=200, slab=40))
    if (lab == c3.MATRIX).any():
        p = pick(lab == c3.MATRIX, -mm[:, c3.M_OM]); bm.append(dict(name="Mitochondrion", caption="Mitochondrion: outer and inner membranes and the dense matrix.", target=p, normal=[0, 0, 1], up=[0, 1, 0], width=400, slab=40))
    if (lab == c3.ERL).any():
        p = pick(lab == c3.ERL, -mm[:, c3.M_ER]); bm.append(dict(name="Endoplasmic reticulum", caption="ER sheet with translocons and ER-bound polysomes.", target=p, normal=[0, 0, 1], up=[0, 1, 0], width=300, slab=40))
    for b in bm:
        for k in ("target", "normal", "up"): b[k] = [round(float(x), 3) for x in b[k]]
    json.dump(bm, open(f"{out}/bookmarks.json", "w"), indent=1)


if __name__ == "__main__":
    main()
