"""Encode the 3D whole-cell model for the WebGPU viewer (static files, streamed per 200 nm tile).

Outputs (OUT = config.WEB_OUT):
  library/types.json        per-type metadata (gene, UniProt, structure source, counts, offsets)
  library/beads.bin         per residue/nucleotide bead: int16 x,y,z (0.005 nm units), uint8 kind, uint8 pLDDT (255 = experimental)
  library/spheretree.bin    float32 (x,y,z,r) sphere-tree spheres (L1 level of detail)
  tiles/<ix>_<iy>_<iz>.bin.gz  per-tile binary (format below)
  tiles/index.json          tile table
  types_tiles.bin           per type: sorted uint16 tile indices (for fly-to-nearest-copy)
  geometry.json             analytic SDF primitives (exact compartment / membrane evaluation in WGSL)

Tile binary (little endian, sections 4-byte aligned):
  header 16 x u32: magic 'YCT1', ix, iy, iz (i32), nInst, nGhost, nMrna, nMrnaBeads, nLink, nLinkBeads,
                   nVert, nTri, flags(bit0 = 32-bit triangle indices), 3 reserved
  instances nInst x 4 u32, ghosts nGhost x 4 u32:
      w0 = type(14 bits) | qidx<<14 (2 bits) | region<<16 (5 bits) | flags<<21 (bit0 near membrane)
      w1 = px | py<<16 ; w2 = pz | qa<<16 ; w3 = qb | qc<<16
      position p = lo - 80 + u16 * (360/65535) nm ; quaternion smallest-three, int16/32767 * (1/sqrt2)
  mRNA: u32 len[nMrna]; i32 start[nMrna*3] (0.005 nm units from lo); i8 delta[(nMrnaBeads-nMrna)*3]
  linker DNA: same with 0.01 nm units
  mesh: u16 vert[nVert*3] (same quantisation as positions); u16|u32 tri[nTri*3]; u8 mem[nTri]
"""
import os, json, gzip, re, tarfile, math
import numpy as np
import pandas as pd
import numba as nb

import cell3d as c3
import tiles3d as t3

from config import LIB, RAW, GDIR, WEB_OUT as OUT, W3D, DATA, AFDB_TAR, NPROC
TILE = 200.0
Q_ORG = 80.0
Q_RANGE = 360.0
Q_STEP = Q_RANGE / 65535.0
BEAD_Q = 0.005
MR_Q = 0.005
LK_Q = 0.01
R_NUCLEOSOME, R_NPC = 22, 23
INV_SQRT2 = 1.0 / math.sqrt(2.0)


# ------------------------------------------------------------------ quantisation helpers
def quant_pos(p, lo):
    u = np.round((p - (lo - Q_ORG)) / Q_STEP)
    assert u.min() >= 0 and u.max() <= 65535, (u.min(), u.max())
    return u.astype(np.uint32)


def dequant_pos(u, lo):
    return (lo - Q_ORG) + u.astype(np.float64) * Q_STEP


def quat_encode(q):
    """q (n,4) w,x,y,z -> (idx, a, b, c) smallest three as int16."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    idx = np.argmax(np.abs(q), axis=1)
    sgn = np.sign(q[np.arange(len(q)), idx]); sgn[sgn == 0] = 1
    q = q * sgn[:, None]
    others = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])[idx]
    v = np.take_along_axis(q, others, axis=1)
    iv = np.clip(np.round(v / INV_SQRT2 * 32767), -32767, 32767).astype(np.int16)
    return idx.astype(np.uint32), iv


def quat_decode(idx, iv):
    v = iv.astype(np.float64) / 32767 * INV_SQRT2
    big = np.sqrt(np.clip(1 - (v ** 2).sum(1), 0, 1))
    q = np.zeros((len(idx), 4))
    others = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])[idx]
    np.put_along_axis(q, others, v, axis=1)
    q[np.arange(len(idx)), idx] = big
    return q


def pack_instances(types, pos, quat, region, flags, lo):
    n = len(types)
    if n == 0:
        return np.zeros((0, 4), np.uint32)
    u = quant_pos(pos, lo)
    qi, qv = quat_encode(quat)
    qv = qv.astype(np.int32) & 0xFFFF
    w = np.zeros((n, 4), np.uint32)
    assert types.max() < 16384 and region.max() < 32
    w[:, 0] = types.astype(np.uint32) | (qi << 14) | (region.astype(np.uint32) << 16) | (flags.astype(np.uint32) << 21)
    w[:, 1] = u[:, 0] | (u[:, 1] << 16)
    w[:, 2] = u[:, 2] | (qv[:, 0].astype(np.uint32) << 16)
    w[:, 3] = qv[:, 1].astype(np.uint32) | (qv[:, 2].astype(np.uint32) << 16)
    return w


def unpack_instances(w, lo):
    w = w.astype(np.uint32)
    types = w[:, 0] & 0x3FFF; qi = (w[:, 0] >> 14) & 3; region = (w[:, 0] >> 16) & 31; flags = w[:, 0] >> 21
    u = np.stack([w[:, 1] & 0xFFFF, w[:, 1] >> 16, w[:, 2] & 0xFFFF], 1)
    qv = np.stack([w[:, 2] >> 16, w[:, 3] & 0xFFFF, w[:, 3] >> 16], 1).astype(np.uint16).view(np.int16)
    return types, dequant_pos(u, lo), quat_decode(qi, qv), region, flags


def pack_polylines(chains, lo, q):
    """chains: list of (n,3) arrays. Exact delta coding of quantised coordinates."""
    if not chains:
        return b"", 0, 0
    lens = np.array([len(c) for c in chains], np.uint32)
    starts, deltas = [], []
    for c in chains:
        P = np.round((np.asarray(c, np.float64) - lo) / q).astype(np.int64)
        starts.append(P[0]); d = np.diff(P, axis=0)
        assert np.abs(d).max(initial=0) <= 127, np.abs(d).max()
        deltas.append(d.astype(np.int8))
    b = lens.tobytes() + np.array(starts, np.int32).tobytes() + np.concatenate(deltas).tobytes()
    return pad4(b), len(chains), int(lens.sum())


def unpack_polylines(buf, off, n, nb_, lo, q):
    lens = np.frombuffer(buf, np.uint32, n, off); off += 4 * n
    st = np.frombuffer(buf, np.int32, 3 * n, off).reshape(n, 3); off += 12 * n
    d = np.frombuffer(buf, np.int8, 3 * (nb_ - n), off).reshape(-1, 3); off += 3 * (nb_ - n)
    off = (off + 3) & ~3
    out, k = [], 0
    for i in range(n):
        m = lens[i] - 1
        P = np.vstack([st[i][None], st[i] + np.cumsum(d[k:k + m].astype(np.int64), 0)])
        out.append(lo + P * q); k += m
    return out, off


def pad4(b):
    return b + b"\0" * ((-len(b)) % 4)


# ------------------------------------------------------------------ library
def morton_order(b):
    lo, hi = b.min(0), b.max(0)
    g = ((b - lo) / np.maximum(hi - lo, 1e-6) * 1023).astype(np.uint64)

    def spread(x):
        x = (x | (x << 16)) & 0x030000FF; x = (x | (x << 8)) & 0x0300F00F
        x = (x | (x << 4)) & 0x030C30C3; x = (x | (x << 2)) & 0x09249249
        return x
    code = spread(g[:, 0]) | (spread(g[:, 1]) << np.uint64(1)) | (spread(g[:, 2]) << np.uint64(2))
    return np.argsort(code, kind="stable")


def af_plddt_job(args):
    acc, lid, data, tm, nbeads = args
    import gemmi
    st = gemmi.make_structure_from_block(gemmi.cif.read_string(data)[0])
    rows = []
    for ch in st[0]:
        for res in ch:
            a = res.find_atom("CA", "*")
            if a is not None:
                rows.append((res.seqid.num, a.b_iso))
    arr = np.array(rows); seq, pl = arr[:, 0].astype(int), arr[:, 1]
    keep = pl >= 50
    tmm = np.zeros(len(seq), bool)
    for a, b, _ in tm:
        tmm |= (seq >= a) & (seq <= b)
    keep |= tmm
    if keep.sum() < 10:
        keep[:] = True
    if keep.sum() != nbeads:
        return lid, None, float(pl.mean())
    return lid, pl[keep].astype(np.float32), float(pl.mean())


def af_plddt(ids, nbeads, ing, n_jobs=NPROC):
    from joblib import Parallel, delayed
    from build_structs import parse_ranges
    topo = pd.read_csv(f"{DATA}/uniprot_topology.tsv", sep="\t").set_index("Entry")
    i2 = ing.set_index("ing_id")
    want = {}
    for lid in ids:
        if lid in i2.index and i2.loc[lid, "kind"] == "protein" and isinstance(i2.loc[lid, "uniprot"], str):
            want[i2.loc[lid, "uniprot"]] = lid
    jobs = []
    with tarfile.open(AFDB_TAR) as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if not name.endswith(".cif.gz") or "-F1-" not in name:
                continue
            acc = name.split("-")[1]
            if acc in want:
                tm = parse_ranges(topo.loc[acc, "Transmembrane"], "TRANSMEM") if acc in topo.index else []
                jobs.append((acc, want[acc], gzip.decompress(tf.extractfile(m).read()).decode(), tm, nbeads[want[acc]]))
    res = Parallel(n_jobs=n_jobs, batch_size=16)(delayed(af_plddt_job)(j) for j in jobs)
    return {lid: (pl, mean) for lid, pl, mean in res}


# library types placed by the pre-pass that have no inventory row
PREPASS_META = {"F_actin": dict(uniprot="P60010", label="Actin filament subunit (ADP-F-actin, cables)", genes="ACT1", gene="ACT1 (F-actin)",
                                kind="complex", oligomer_n=1),
                "MT_tubulin": dict(uniprot="", label="Microtubule alpha/beta-tubulin dimer (13-protofilament lattice)", genes="TUB1;TUB2",
                                   gene="TUB1\u2013TUB2 (microtubule)", kind="complex", oligomer_n=2),
                "gTuRC_nuc": dict(uniprot="", label="gamma-tubulin ring complex with SPC110 (nuclear MT minus end)",
                                  genes="TUB4;SPC97;SPC98;SPC110", gene="\u03b3-TuRC (nuclear)", kind="complex", oligomer_n=1),
                "TPM1_coil": dict(uniprot="P17536", label="Tropomyosin TPM1 coiled-coil dimer on an actin cable (path from 5JLF)",
                                  genes="TPM1", gene="TPM1 (on cables)", kind="complex", oligomer_n=2),
                "CP_cap": dict(uniprot="", label="Capping protein at an actin barbed end (CAP1/CAP2 on the 8F8Q template)",
                               genes="CAP1;CAP2", gene="CAP1\u2013CAP2 (capping)", kind="complex", oligomer_n=2),
                "gTuRC_cyto": dict(uniprot="", label="gamma-tubulin ring complex (cytoplasmic MT minus end)",
                                   genes="TUB4;SPC97;SPC98", gene="\u03b3-TuRC (cytoplasmic)", kind="complex", oligomer_n=1)}


def encode_library(meta, ing, counts_by_type, region_counts, out=OUT):
    ids = [str(x) for x in meta["ids"]]
    os.makedirs(f"{out}/library", exist_ok=True)
    raw = {lid: np.load(f"{LIB}/{lid}.npz") for lid in ids}
    nbeads = {lid: len(raw[lid]["beads"]) for lid in ids}
    plmap = af_plddt([l for l in ids if meta["src"][ids.index(l)] == "AF"], nbeads, ing)
    i2 = ing.set_index("ing_id")
    uni = pd.read_csv(f"{DATA}/uniprot_yeast.tsv", sep="\t").set_index("Entry")
    B, K, PL, ST, types = [], [], [], [], []
    bo = so = 0
    n_pl_ok = 0
    for t, lid in enumerate(ids):
        z = raw[lid]
        b = z["beads"].astype(np.float64); k = z["kind"].astype(np.uint8)
        src = str(meta["src"][t]); pdb = str(meta["pdb"][t])
        if "plddt" in z.files:   # predicted assemblies carry their own per-bead pLDDT
            pl = np.clip(np.round(z["plddt"]), 0, 100).astype(np.uint8); n_pl_ok += 1
            plmean = float(pl.mean())
        elif src == "AF" and lid in plmap and plmap[lid][0] is not None:
            pl = np.clip(np.round(plmap[lid][0]), 0, 100).astype(np.uint8); n_pl_ok += 1
            plmean = float(pl.mean())
        else:
            pl = np.full(len(b), 255, np.uint8); plmean = None
        o = morton_order(b)
        b, k, pl = b[o], k[o], pl[o]
        qb = np.round(b / BEAD_Q)
        assert np.abs(qb).max() < 32767
        B.append(qb.astype(np.int16)); K.append(k); PL.append(pl)
        stc, str_ = z["st_c"].astype(np.float32), z["st_r"].astype(np.float32)
        ST.append(np.c_[stc, str_])
        base = lid[:-6] if lid.endswith("_oligo") else lid
        row = i2.loc[base] if base in i2.index else PREPASS_META.get(lid)
        acc = row["uniprot"] if row is not None and isinstance(row["uniprot"], str) else ""
        name = row["label"] if row is not None else lid
        if acc and acc in uni.index and isinstance(uni.loc[acc, "Protein names"], str):
            name = uni.loc[acc, "Protein names"].split(" (")[0]
        genes = row["genes"] if row is not None and isinstance(row["genes"], str) else lid
        rg = float(np.sqrt(((b - b.mean(0)) ** 2).sum(1).mean()))
        mass = float(np.where(k == 0, 110.0, 330.0).sum())
        nchain = 1
        if "nchain" in z.files:
            nchain = int(z["nchain"])
        elif lid.endswith("_oligo"):
            nchain = int(row["oligomer_n"]) if row is not None else 1
        types.append(dict(
            id=lid, gene=base if lid.endswith("_oligo") or (row is not None and row["kind"] == "protein")
            else (genes.replace(";", "\u2013") if lid.endswith("_het") else PREPASS_META.get(lid, {}).get("gene", lid)),
            genes=genes, uniprot=acc, name=name, src=src, pdb=pdb if src == "PDB" else "",
            oligomer=nchain, nb=int(len(b)), nnt=int((k == 1).sum()), bo=int(bo), so=int(so), sn=int(len(stc)),
            br=round(float(meta["BR"][t]), 3), rg=round(rg, 3), mass=round(mass),
            plddt_mean=None if plmean is None else round(plmean, 1),
            plddt_frac_kept=round(float(z["plddt_frac_kept"]), 3) if "plddt_frac_kept" in z.files else None,
            copies=int(counts_by_type[t]), regions={str(r): int(c) for r, c in region_counts.get(t, {}).items()},
            **({"model": str(z["model"])} if "model" in z.files else {})))
        bo += len(b); so += len(stc)
    B = np.concatenate(B); K = np.concatenate(K); PL = np.concatenate(PL)
    rec = np.zeros(len(B), dtype=[("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("k", "u1"), ("p", "u1")])
    rec["x"], rec["y"], rec["z"], rec["k"], rec["p"] = B[:, 0], B[:, 1], B[:, 2], K, PL
    rec.tofile(f"{out}/library/beads.bin")
    np.concatenate(ST).astype("<f4").tofile(f"{out}/library/spheretree.bin")
    json.dump(dict(bead_q=BEAD_Q, n_beads=int(len(B)), n_spheres=int(so), types=types),
              open(f"{out}/library/types.json", "w"), separators=(",", ":"))
    return dict(n_types=len(ids), n_beads=int(len(B)), n_st=int(so), n_af_plddt=n_pl_ok,
                n_af=int(sum(1 for s in meta["src"] if s == "AF")))


# ------------------------------------------------------------------ tiles
def tile_index_list():
    names = pd.read_parquet(f"{W3D}/tile_measure.parquet").index
    tl = [tuple(int(x) for x in re.findall(r"[+-]\d+", n)) for n in names]
    return tl


def linker_chains(pre, nuc_beads, nuc_kind, step=0.68):
    """Linker DNA between consecutive nucleosomes of the same chain fragment (cubic Hermite curves from the
    exit end of one nucleosome to the entry end of the next). Schematic geometry (no clash check)."""
    from scipy.spatial.transform import Rotation
    d = nuc_beads[nuc_kind == 1]; n = len(d) // 2; I, J = d[:n], d[n:]
    e_in, e_out = 0.5 * (I[0] + J[-1]), 0.5 * (I[-1] + J[0])
    t_in = -(0.5 * (I[3] + J[-4]) - e_in); t_out = e_out - 0.5 * (I[-4] + J[3])
    t_in /= np.linalg.norm(t_in); t_out /= np.linalg.norm(t_out)
    pos, q, ch, brk = pre["nuc_pos"], pre["nuc_q"], pre["nuc_chain"], pre["nuc_brk"]
    R = Rotation.from_quat(np.c_[q[:, 1:], q[:, :1]]).as_matrix()
    chains, mids = [], []
    for i in range(len(pos) - 1):
        if ch[i + 1] != ch[i] or brk[i + 1]:
            continue
        a = R[i] @ e_out + pos[i]; ta = R[i] @ t_out
        b = R[i + 1] @ e_in + pos[i + 1]; tb = R[i + 1] @ t_in
        L = np.linalg.norm(b - a)
        if L > 40:
            continue
        m = max(2, int(np.ceil(L * 1.15 / step)))
        s = np.linspace(0, 1, m)[:, None]
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
        P = h00 * a + h10 * (ta * L) + h01 * b + h11 * (tb * L)
        chains.append(P); mids.append(P[len(P) // 2])
    return chains, np.array(mids)


@nb.njit(cache=True)
def near_membrane_flags(pos, br, G, pores, pidx, PT, PQ, PC, margin):
    n = pos.shape[0]
    f = np.zeros(n, np.uint8)
    out = np.zeros(c3.NMEM)
    for i in range(n):
        c3.mem_sdfs(pos[i, 0], pos[i, 1], pos[i, 2], G, pores, pidx, PT, PQ, PC, out)
        d = 1e9
        for m in range(8):
            d = min(d, abs(out[m]))
        if d < br[i] + margin:
            f[i] = 1
    return f


_S = {}


def _stage1(ti):
    """per tile: load raw + assigned pre-pass, region codes, near-membrane flags -> tmp npz"""
    S = _S
    t = S["tiles"][ti]; lo = np.array(t, float) * TILE; name = t3.tile_name(t)
    r = np.load(f"{RAW}/{name}.npz")
    types = r["types"].astype(np.int64); pos = r["pos"].astype(np.float64); quat = r["quat"].astype(np.float64)
    region = np.repeat(r["job_region"], r["job_placed"]).astype(np.int64)
    assert len(region) == len(types)
    k = S["pre_tile"] == ti
    if k.any():
        types = np.concatenate([types, S["pre_type"][k]]); pos = np.concatenate([pos, S["pre_pos"][k]])
        quat = np.concatenate([quat, S["pre_q"][k]]); region = np.concatenate([region, S["pre_reg"][k]])
    cell = S["cell"]
    a = cell.args(lo, lo + TILE, margin=80.0)
    flags = near_membrane_flags(pos, S["BR"][types], *a, 3.0) if len(pos) else np.zeros(0, np.uint8)
    np.savez(f"{S['tmp']}/{name}.npz", types=types, pos=pos, quat=quat, region=region, flags=flags)
    pk, pc = np.unique(types * 32 + region, return_counts=True)
    return ti, len(types), int((flags == 1).sum()), pk, pc


def _stage2(ti):
    S = _S
    tiles, tset = S["tiles"], S["tset"]
    t = tiles[ti]; lo = np.array(t, float) * TILE; hi = lo + TILE; name = t3.tile_name(t)
    z = np.load(f"{S['tmp']}/{name}.npz")
    g = np.load(f"{GDIR}/{name}.npz")
    V, F, FM = g["verts"].astype(np.float64), g["faces"], g["face_mem"]
    # ghosts: near-membrane instances of the 26 neighbours that may overlap this tile's lipids
    gh = []
    if len(F):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    nt = (t[0] + dx, t[1] + dy, t[2] + dz)
                    if nt not in tset:
                        continue
                    zz = np.load(f"{S['tmp']}/{t3.tile_name(nt)}.npz")
                    m = zz["flags"] == 1
                    if not m.any():
                        continue
                    p = zz["pos"][m]; br = S["BR"][zz["types"][m]] + 3.0
                    ok = np.all((p > lo - br[:, None]) & (p < hi + br[:, None]), axis=1)
                    if ok.any():
                        gh.append((zz["types"][m][ok], p[ok], zz["quat"][m][ok], zz["region"][m][ok], zz["flags"][m][ok]))
    order = np.lexsort((z["region"], z["types"]))   # group by type for compression and coherent drawing
    inst = pack_instances(z["types"][order], z["pos"][order], z["quat"][order], z["region"][order], z["flags"][order], lo)
    if gh:
        gt, gp, gq, gr, gf = (np.concatenate(x) for x in zip(*gh))
        ghost = pack_instances(gt, gp, gq, gr, gf, lo)
    else:
        ghost = np.zeros((0, 4), np.uint32)
    r = np.load(f"{RAW}/{name}.npz")
    mo, mp = r["mrna_off"], r["mrna_pts"].astype(np.float64)
    mchains = [mp[mo[i]:mo[i + 1]] for i in range(len(mo) - 1) if mo[i + 1] - mo[i] >= 2]
    if len(S.get("poly_mrna_tile", ())):      # polysome mRNA pieces (pre-pass), by midpoint tile
        mchains += [S["poly_mrna"][i] for i in np.where(S["poly_mrna_tile"] == ti)[0]]
    mbytes, nm, nmb = pack_polylines(mchains, lo, MR_Q)
    li = np.where(S["link_tile"] == ti)[0]
    lbytes, nl, nlb = pack_polylines([S["links"][i] for i in li], lo, LK_Q)
    vbytes = pad4(quant_pos(V, lo).astype("<u2").tobytes()) if len(V) else b""
    big = len(V) > 65535
    fbytes = pad4(F.astype("<u4" if big else "<u2").tobytes()) if len(F) else b""
    mbytes2 = pad4(FM.astype("u1").tobytes()) if len(F) else b""
    # extra polylines with a kind per chain (4 pre-rRNA, 5 disordered protein / FG repeats), appended after the meshes
    xi = np.where(S["extra_tile"] == ti)[0] if len(S.get("extra_tile", ())) else np.zeros(0, np.int64)
    xbytes, nx, nxb = pack_polylines([S["extra"][i] for i in xi], lo, MR_Q)
    if nx:
        xbytes = pad4(S["extra_kind"][xi].astype("u1").tobytes()) + xbytes
    hdr = np.zeros(16, np.uint32)
    hdr[0] = int.from_bytes(b"YCT1", "little")
    hdr[1:4] = np.array(t, np.int32).view(np.uint32)
    hdr[4:15] = [len(inst), len(ghost), nm, nmb, nl, nlb, len(V), len(F), 1 if big else 0, nx, nxb]
    blob = (hdr.tobytes() + inst.astype("<u4").tobytes() + ghost.astype("<u4").tobytes() + mbytes + lbytes + vbytes + fbytes
            + mbytes2 + xbytes)
    fn = f"{S['out']}/tiles/{t[0]}_{t[1]}_{t[2]}.bin.gz"
    gzb = gzip.compress(blob, compresslevel=9)
    open(fn, "wb").write(gzb)
    return ti, len(inst), len(ghost), nm, nmb, nl, nlb, len(V), len(F), len(blob), len(gzb)


def decode_tile(blob):
    """Python reference decoder (used for round-trip tests)."""
    h = np.frombuffer(blob, np.uint32, 16)
    assert h[0] == int.from_bytes(b"YCT1", "little")
    t = h[1:4].view(np.int32); lo = t.astype(float) * TILE
    nI, nG, nM, nMB, nL, nLB, nV, nT, fl = (int(x) for x in h[4:13])
    off = 64
    inst = np.frombuffer(blob, np.uint32, nI * 4, off).reshape(-1, 4); off += 16 * nI
    ghost = np.frombuffer(blob, np.uint32, nG * 4, off).reshape(-1, 4); off += 16 * nG
    mr, off = unpack_polylines(blob, off, nM, nMB, lo, MR_Q) if nM else ([], off)
    lk, off = unpack_polylines(blob, off, nL, nLB, lo, LK_Q) if nL else ([], off)
    V = dequant_pos(np.frombuffer(blob, np.uint16, nV * 3, off).reshape(-1, 3), lo) if nV else np.zeros((0, 3)); off += (6 * nV + 3) & ~3
    F = np.frombuffer(blob, np.uint32 if fl & 1 else np.uint16, nT * 3, off).reshape(-1, 3) if nT else np.zeros((0, 3), int)
    off += ((4 if fl & 1 else 2) * 3 * nT + 3) & ~3
    FM = np.frombuffer(blob, np.uint8, nT, off) if nT else np.zeros(0, np.uint8)
    return dict(tile=tuple(t), lo=lo, inst=unpack_instances(inst, lo), ghost=unpack_instances(ghost, lo) if nG else None,
                mrna=mr, linker=lk, verts=V, faces=F, face_mem=FM)
