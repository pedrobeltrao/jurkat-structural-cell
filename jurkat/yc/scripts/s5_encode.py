"""Stage 5: encode the packed cell for the WebGPU viewer -> config.WEB_OUT. Notebook cells 38-45.

  python s5_encode.py               # full encode (library, tiles, index, geometry, meshes, bookmarks)
  python s5_encode.py --tiles-only  # tiles only (baseline reproduction check; needs no inventory)
"""
import argparse, gzip, shutil, multiprocessing as mp
from _common import *
import cell3d as c3, tiles3d as t3, webenc as we, wc3d, organisation as og

MRNA_PIECE = 64          # nt per encoded mRNA piece (pieces are assigned to tiles by their midpoint)
IDX_COLS = ["ti", "n_inst", "n_ghost", "n_mrna", "n_mrna_beads", "n_link", "n_link_beads", "n_vert", "n_tri", "raw_bytes", "gz_bytes"]


def pre_regions(pre):
    """region code per pre-placed instance, in wc3d.pre_instances order"""
    fixed = {"npc": we.R_NPC, "nuc": we.R_NUCLEOSOME, "poly": t3.R_CYTO, "act": t3.R_CYTO, "tpm": t3.R_CYTO, "pat": t3.R_CYTO,
             "atp": t3.REGION_NAMES.index("mem_MITO_IM")}
    return np.concatenate([pre[f"{k}_reg"].astype(np.int64) if f"{k}_reg" in pre else np.full(len(pre[f"{k}_pos"]), fixed[k])
                           for k, _ in wc3d.PRE_GROUPS if f"{k}_pos" in pre])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-only", action="store_true")
    args = ap.parse_args()
    meta = dict(np.load(f"{C.W3D}/libpack_meta.npz"))
    ids = [str(x) for x in meta["ids"]]; NT = len(ids)
    cell = c3.Cell.load(f"{C.W3D}/geometry.json")
    tiles = we.tile_index_list(); tindex = {t: i for i, t in enumerate(tiles)}
    pre = dict(np.load(f"{C.W3D}/prepass.npz"))
    gt, pp, gq = wc3d.pre_instances(pre)        # NPCs, nucleosomes, polysome ribosomes, actin, microtubules
    n_poly = len(pre["poly_pos"]) if "poly_pos" in pre else 0
    n_act = len(pre["act_pos"]) if "act_pos" in pre else 0
    pre_tile = np.array([tindex[tuple(int(v) for v in np.floor(p / we.TILE))] for p in pp])
    # polysome mRNA: one piece per ribosome (channel + linker to the next), assigned to the tile of its midpoint
    pieces, piece_tile = [], np.zeros(0, np.int64)
    if n_poly:
        off = pre["poly_mrna_off"]
        pieces = [pre["poly_mrna"][off[i]:off[i + 1]].astype(np.float64) for i in range(len(off) - 1)]
        piece_tile = np.array([tindex.get(tuple(int(v) for v in np.floor(p[len(p) // 2] / we.TILE)), -1) for p in pieces])
    if "mrna_pts" in pre:   # full-length mRNA threads: cut into MRNA_PIECE-nt pieces, each assigned to its midpoint's tile
        mpts, mo = pre["mrna_pts"].astype(np.float64), pre["mrna_off"]
        extra = []
        for i in range(len(mo) - 1):
            for a in range(mo[i], mo[i + 1], MRNA_PIECE):
                b = min(mo[i + 1], a + MRNA_PIECE)
                if b - a >= 2:
                    extra.append(mpts[a:b])
        mids = np.array([p[len(p) // 2] for p in extra])
        keys = np.floor(mids / we.TILE).astype(np.int64)
        et = np.array([tindex.get(tuple(k), -1) for k in map(tuple, keys)])
        pieces = list(pieces) + extra; piece_tile = np.r_[piece_tile, et].astype(np.int64)
        log(f"mRNA threads: {len(mo) - 1} threads -> {len(extra)} pieces ({(et < 0).sum()} outside tiles)")
    # FG-repeat domains (kind 5): cut into MRNA_PIECE-residue pieces, by midpoint tile
    extra, extra_kind = [], []
    for key, kind in (("fg", 5),):
        if f"{key}_pts" in pre:
            P_, O_ = pre[f"{key}_pts"].astype(np.float64), pre[f"{key}_off"]
            for i in range(len(O_) - 1):
                for a in range(O_[i], O_[i + 1], MRNA_PIECE):
                    b = min(O_[i + 1], a + MRNA_PIECE)
                    if b - a >= 2:
                        extra.append(P_[a:b]); extra_kind.append(kind)
    extra_tile = np.array([tindex.get(tuple(int(v) for v in np.floor(p[len(p) // 2] / we.TILE)), -1) for p in extra], np.int64)
    log(f"extra polylines: {len(extra)} pieces ({(extra_tile < 0).sum()} outside tiles)")
    shutil.rmtree(C.WEB_TMP, ignore_errors=True); os.makedirs(C.WEB_TMP)
    shutil.rmtree(we.OUT, ignore_errors=True); os.makedirs(f"{we.OUT}/tiles")
    we._S.update(tiles=tiles, tset=set(tiles), cell=cell, BR=meta["BR"].astype(np.float64), pre_tile=pre_tile,
                 pre_type=gt.astype(np.int64), pre_pos=pp.astype(np.float64), pre_q=gq.astype(np.float64),
                 pre_reg=pre_regions(pre),
                 poly_mrna=pieces, poly_mrna_tile=piece_tile, extra=extra, extra_tile=extra_tile,
                 extra_kind=np.array(extra_kind, np.int64), tmp=C.WEB_TMP, out=we.OUT)
    t0 = time.time()
    with mp.get_context("fork").Pool(C.NPROC) as pool:
        st1 = pool.map(we._stage1, range(len(tiles)), chunksize=8)
    counts_by_type = np.zeros(NT, np.int64); region_counts = {}
    for x in st1:
        for k, c in zip(x[3], x[4]):
            t, rg = divmod(int(k), 32); counts_by_type[t] += c; region_counts.setdefault(t, {})[rg] = region_counts.get(t, {}).get(rg, 0) + int(c)
    log(f"stage1 {time.time() - t0:.0f}s; instances {counts_by_type.sum() / 1e6:.3f}M")

    zn = np.load(f"{we.LIB}/nucleosome.npz")
    links, mids = we.linker_chains(pre, zn["beads"].astype(np.float64), zn["kind"])
    link_tile = np.array([tindex.get(tuple(int(v) for v in np.floor(p / we.TILE)), -1) for p in mids])
    we._S.update(links=links, link_tile=link_tile)
    t0 = time.time()
    with mp.get_context("fork").Pool(C.NPROC) as pool:
        st2 = pool.map(we._stage2, range(len(tiles)), chunksize=4)
    st2 = pd.DataFrame(st2, columns=IDX_COLS).sort_values("ti").reset_index(drop=True)
    log(f"stage2 {time.time() - t0:.0f}s; gz {st2.gz_bytes.sum() / 1e6:.0f} MB")
    roundtrip(tiles, links, link_tile)
    if args.tiles_only:
        return

    ing, _ = load_inventory()
    log("library", we.encode_library(meta, ing, counts_by_type, region_counts))
    if os.path.exists(f"{og.ODIR}/modules.json"):          # organisation layer of s3 (module colouring in the viewer)
        og.encode_viewer(we.OUT, log=log)
    labvol = pd.read_parquet(f"{C.W3D}/tile_labvol.parquet")
    index = tile_index(tiles, st2, labvol)
    types_tiles(tiles, NT)
    web_geometry(cell)
    organelle_meshes(tiles)
    bookmarks(tiles, ids, cell, labvol)
    index["wall"] = wall_budget(tiles, cell, ing, labvol)
    for i, row in enumerate(index["tiles"]):
        row += [round(float(labvol.iloc[i]["wall_glucan"]), 1), round(float(labvol.iloc[i]["wall_mannan"]), 1)]
    index["tile_columns"] += ["vol_glucan_nm3", "vol_mannan_nm3"]
    json.dump(index, open(f"{we.OUT}/tiles/index.json", "w"), separators=(",", ":"))
    log("done ->", we.OUT)


def roundtrip(tiles, links, link_tile):
    """decode 40 tiles with the reference decoder and compare against the raw packing output"""
    rng = np.random.default_rng(3); perr, aerr = [0.0], [0.0]
    for ti in rng.choice(len(tiles), 40, replace=False):
        t = tiles[ti]; name = t3.tile_name(t)
        d = we.decode_tile(gzip.decompress(open(f"{we.OUT}/tiles/{t[0]}_{t[1]}_{t[2]}.bin.gz", "rb").read()))
        z = np.load(f"{C.WEB_TMP}/{name}.npz"); o = np.lexsort((z["region"], z["types"]))
        ty, po, qu, rg, fl = d["inst"]
        assert (ty == z["types"][o]).all() and (rg == z["region"][o]).all() and (fl == z["flags"][o]).all()
        perr.append(np.abs(po - z["pos"][o]).max(initial=0))
        if len(qu):
            q0 = z["quat"][o]; dots = np.abs((qu * q0 / np.linalg.norm(q0, axis=1, keepdims=True)).sum(1))
            aerr.append(np.degrees(2 * np.arccos(np.clip(dots, 0, 1))).max())
    log(f"round trip: max pos err {max(perr):.4f} nm, max rot err {max(aerr):.4f} deg")
    assert max(perr) < 0.01 and max(aerr) < 0.1


def tile_index(tiles, st2, labvol):
    labmask = ((labvol.values > 0).astype(np.int64) << np.arange(labvol.shape[1])).sum(1)
    return dict(tile_nm=we.TILE, q_org=we.Q_ORG, q_range=we.Q_RANGE, mrna_q=we.MR_Q, link_q=we.LK_Q,
                label_names=c3.LABEL_NAMES, region_names=t3.REGION_NAMES + ["chromatin_nucleosome", "NE_pore_NPC"],
                mem_names=c3.MEM_NAMES, label_columns=list(labvol.columns),
                tiles=[[*tiles[i], int(r.n_inst), int(r.n_ghost), int(r.n_mrna_beads), int(r.n_link_beads), int(r.n_vert),
                        int(r.n_tri), int(r.gz_bytes), int(labmask[i])] for i, r in st2.iterrows()],
                tile_columns=["ix", "iy", "iz", "n_inst", "n_ghost", "n_mrna_beads", "n_link_beads", "n_vert", "n_tri", "gz_bytes", "label_mask"])


def types_tiles(tiles, NT):
    tt = [[] for _ in range(NT)]
    for i, t in enumerate(tiles):
        for ty in np.unique(np.load(f"{C.WEB_TMP}/{t3.tile_name(t)}.npz")["types"]): tt[ty].append(i)
    offs = np.zeros(NT + 1, np.uint32); offs[1:] = np.cumsum([len(x) for x in tt])
    with open(f"{we.OUT}/types_tiles.bin", "wb") as f:
        f.write(offs.astype("<u4").tobytes()); f.write(np.concatenate([np.array(x, "<u2") for x in tt]).tobytes())


WEB_BANDS = [(180.0, 40.0), (40.0, 40.0), (100.0, 100.0)]   # viewer bands (in, out) of the cell, vacuole, nucleus fields
FIELD_Q = 0.01                                               # nm per int16 step


def web_fields(cell):
    """the real-shape distance fields for the viewer (geometry_fields.bin; layout in web/app/src/fields.ts):
    split into bricks of half the edge (less padding around the thin bands), re-classified for narrower bands, node
    values as int16 in FIELD_Q steps. None for an analytic cell."""
    if not cell.fields:
        return None
    heads, n_index, n_data, parts = [], [], [], []
    off = 1 + 12 * len(cell.fields)
    for (h, idx, data), (bin_, bout) in zip(cell.fields, WEB_BANDS):
        B = int(h[7]); n1 = B + 1; nbx, nby, nbz = int(h[4]), int(h[5]), int(h[6]); b2 = B // 2
        blocks = data.reshape(-1, n1, n1, n1)
        old = idx.astype(np.int64).reshape(nbx, nby, nbz)
        new = np.empty((2 * nbx, 2 * nby, 2 * nbz), np.int64); keep = []
        for bx in range(2 * nbx):
            for by in range(2 * nby):
                for bz in range(2 * nbz):
                    o = old[bx // 2, by // 2, bz // 2]
                    if o < 0:
                        new[bx, by, bz] = o; continue
                    i0, j0, k0 = (bx % 2) * b2, (by % 2) * b2, (bz % 2) * b2
                    blk = blocks[o, i0:i0 + b2 + 1, j0:j0 + b2 + 1, k0:k0 + b2 + 1]
                    if blk.max() <= -bin_:
                        new[bx, by, bz] = -1
                    elif blk.min() >= bout:
                        new[bx, by, bz] = -2
                    else:
                        new[bx, by, bz] = len(keep); keep.append(np.clip(blk, -bin_, bout).ravel())
        idx = new.ravel()
        dat = np.round(np.concatenate(keep) / FIELD_Q).astype(np.int16) if keep else np.zeros(0, np.int16)
        hd = [float(h[0]), float(h[1]), float(h[2]), float(h[3]), 2.0 * nbx, 2.0 * nby, 2.0 * nbz, float(b2),
              bin_, bout, float(off), float(off + len(idx))]
        off += len(idx) + len(dat)
        heads.append(hd); n_index.append(int(len(idx))); n_data.append(int(len(dat)))
        pad = (-2 * len(dat)) % 4
        parts += [idx.astype(np.int32).tobytes(), dat.tobytes() + b"\0" * pad]
    open(f"{we.OUT}/geometry_fields.bin", "wb").write(b"".join(parts))
    log(f"geometry fields for the viewer: {sum(n_data) / 1e6:.1f} M nodes, "
        f"{os.path.getsize(f'{we.OUT}/geometry_fields.bin') / 1e6:.1f} MB")
    return dict(file="geometry_fields.bin", head=heads, n_index=n_index, n_data=n_data, q=FIELD_Q)


def web_geometry(cell):
    """primitives + uniform acceleration grid for the WGSL SDF"""
    geo = json.load(open(f"{C.W3D}/geometry.json"))
    GN, EXT = 36, 2600.0; h = 2 * EXT / GN       # encloses the real cell outline (viewer tile grid: +-2.6 um)
    bc, br = cell.bc, cell.br
    pc = cell.pores[:, :3]; pr = c3.PORE_R + c3.PERI_W + 12
    cells_p, cells_q = [], []
    for ix in range(GN):
        for iy in range(GN):
            for iz in range(GN):
                lo = -EXT + h * np.array([ix, iy, iz]); hi = lo + h
                q = np.clip(bc, lo, hi); cells_p.append(np.where(((q - bc) ** 2).sum(1) <= (br + 12) ** 2)[0])
                q = np.clip(pc, lo, hi); cells_q.append(np.where(((q - pc) ** 2).sum(1) <= pr ** 2)[0])
    goff = np.zeros(GN ** 3 + 1, np.int64); goff[1:] = np.cumsum([len(a) for a in cells_p])
    qoff = np.zeros(GN ** 3 + 1, np.int64); qoff[1:] = np.cumsum([len(a) for a in cells_q])
    fields = web_fields(cell)
    gweb = dict(G=geo["G"], fields=fields, pores=cell.pores.round(6).ravel().tolist(), PT=geo["PT"], PQ=np.round(cell.PQ, 4).ravel().tolist(), PC=geo["PC"],
                consts=geo["consts"], label_names=c3.LABEL_NAMES, mem_names=c3.MEM_NAMES, npc_pos=geo["npc_pos"], mito_segments=geo["mito_segments"],
                grid=dict(n=GN, ext=EXT, prim_off=goff.tolist(), prim_idx=np.concatenate(cells_p).tolist(),
                          pore_off=qoff.tolist(), pore_idx=np.concatenate(cells_q).tolist()))
    json.dump(gweb, open(f"{we.OUT}/geometry.json", "w"), separators=(",", ":"))


def organelle_meshes(tiles):
    import fast_simplification as fsimp
    Vs = {m: [] for m in range(10)}; Fs = {m: [] for m in range(10)}; nv = {m: 0 for m in range(10)}
    for t in tiles:
        g = np.load(f"{we.GDIR}/{t3.tile_name(t)}.npz")
        if not len(g["faces"]): continue
        V, F, FM = g["verts"], g["faces"], g["face_mem"]
        for m in np.unique(FM):
            f = F[FM == m]; u, inv = np.unique(f.ravel(), return_inverse=True)
            Vs[m].append(V[u]); Fs[m].append(inv.reshape(-1, 3) + nv[m]); nv[m] += len(u)
    mesh_out = []
    for m in [m for m in range(10) if Fs[m]]:
        V = np.concatenate(Vs[m]).astype(np.float64); F = np.concatenate(Fs[m])
        _, first, inv = np.unique(np.round(V / 0.01).astype(np.int64), axis=0, return_index=True, return_inverse=True)
        V2 = V[first]; F2 = inv.ravel()[F]
        F2 = F2[(F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])]
        Vd, Fd = fsimp.simplify(V2.astype(np.float32), F2.astype(np.int32), target_reduction=0.95 if m in (0, 3) else 0.9)
        mesh_out.append((m, Vd.astype("<f4"), Fd.astype("<u4")))
    with gzip.open(f"{we.OUT}/organelle_meshes.bin.gz", "wb", compresslevel=9) as f:
        hdr = np.zeros(1 + 3 * len(mesh_out), np.uint32); hdr[0] = len(mesh_out)
        for i, (m, Vd, Fd) in enumerate(mesh_out): hdr[1 + 3 * i: 4 + 3 * i] = [m, len(Vd), len(Fd)]
        f.write(hdr.tobytes())
        for m, Vd, Fd in mesh_out: f.write(Vd.tobytes()); f.write(Fd.tobytes())


def bookmarks(tiles, ids, cell, labvol):
    """hand-placed guided views (tied to geometry seed 11)"""
    unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)
    perp = lambda a, hint=(0.3, 0.5, 0.81): unit(np.cross(a, hint))
    geo = json.load(open(f"{C.W3D}/geometry.json"))
    li = {k: ids.index(k) for k in ["Pma1_hexamer", "ribosome_80S", "nucleosome", "NPC"]}
    tcount = {k: np.zeros(len(tiles), int) for k in li}
    for i, t in enumerate(tiles):
        ty = np.load(f"{C.WEB_TMP}/{t3.tile_name(t)}.npz")["types"]
        for k, v in li.items(): tcount[k][i] = (ty == v).sum()
    lv = labvol.values; lcol = list(labvol.columns)
    tc = lambda i: np.array(tiles[i], float) * 200 + 100
    G = cell.G; cn = G[3:6]
    vac = [i for i, c in enumerate(cell.PC) if c == c3.C_VAC]
    if vac:                                                   # analytic cell: the largest vacuole lobe
        big = max(vac, key=lambda i: cell.PQ[i, 3]); vc = cell.PQ[big, :3]; rv = cell.PQ[big, 3]
        vac_target, vac_u = vc + unit(-vc) * rv, unit(-vc)
    else:                                                     # real shapes: vacuole membrane point nearest the centre
        Pr = np.random.default_rng(3).uniform(-G[0], G[0], (400000, 3))
        dv = np.array([c3.field_sdf(p[0], p[1], p[2], G, c3.F_VAC) for p in Pr])
        vc = Pr[dv < 0].mean(0)
        near = Pr[(np.abs(dv) < 4) & (np.array([c3.outer_depth(*p, G) for p in Pr]) > 250)]     # (cell field clamped at 260)
        vac_target = near[np.argmin(np.linalg.norm(near, axis=1))]
        vac_u = -c3._grad(lambda x, y, z: c3.field_sdf(x, y, z, G, c3.F_VAC), vac_target)
    n_inst = sum(t["copies"] for t in json.load(open(f"{we.OUT}/library/types.json"))["types"])
    bm = [dict(name="Whole cell", caption=f"Equatorial section through nucleus and vacuole. {n_inst / 1e6:.1f} million macromolecules, every one a real PDB or AlphaFold structure.",
               target=[0, 0, 0], normal=unit(np.cross(cn, vc)), up=unit(cn), width=5200, slab=60)]
    u = unit(np.cross(cn, vc))
    bm.append(dict(name="Cell wall", caption="Mannoprotein brush (outer 35 nm) over the beta-glucan layer, plasma membrane below.", target=u * c3.ray_depth(G, u, 55.0), normal=perp(u), up=u, width=260, slab=40))
    cer_col = lcol.index("cER_lumen")
    cand = [i for i in np.argsort(-tcount["Pma1_hexamer"]) if lv[i, cer_col] == 0][:1]
    u = unit(tc(cand[0]))
    bm.append(dict(name="Plasma membrane (face-on)", caption="Cytoplasmic face of the plasma membrane: Pma1 H+-ATPase hexamers among ~1,000 other membrane proteins; lipids generated around them.",
                   target=u * c3.ray_depth(G, u, G[0] - G[2] + 18.0), normal=u, up=perp(u), width=220, slab=30))
    cer = [i for i, c in enumerate(cell.PC) if c == c3.C_CER]
    q = cell.PQ[cer[0]]; u = unit(q[:3])
    tgt = u * c3.ray_depth(G, u, G[0] - G[2] + c3.CER_GAP + 0.5 * c3.CER_LUMEN)    # on the sheet, under the real outline
    bm.append(dict(name="Cortical ER", caption="Cortical ER sheet 27 nm beneath the plasma membrane, with Sec61 translocons and ER-bound ribosomes.", target=tgt, normal=perp(u), up=u, width=320, slab=40))
    cyto = lcol.index("cytosol")
    i = [i for i in np.argsort(-tcount["ribosome_80S"]) if lv[i, cyto] > 0.99 * lv[i].sum()][0]
    bm.append(dict(name="Cytoplasm", caption="Crowded cytoplasm (~93 mg/mL): 80S ribosomes (pink), most of them threaded on mRNAs as polysomes, with glycolytic enzymes and tRNAs.", target=tc(i), normal=unit([0.2, -0.4, 0.89]), up=unit([0, 1, 0.45]), width=160, slab=30))
    pre = dict(np.load(f"{C.W3D}/prepass.npz"))
    if "act_pos" in pre:   # longest actin filament: view the cortex from inside, cut plane through the cable
        f = np.bincount(pre["act_fil"]).argmax(); P = pre["act_pos"][pre["act_fil"] == f]
        mid = P[len(P) // 2]; u = unit(mid); tdir = unit(P[min(len(P) - 1, len(P) // 2 + 20)] - P[max(0, len(P) // 2 - 20)])
        bm.append(dict(name="Actin cable", caption="Actin cable beneath the cortex: a bundle of F-actin filaments (orange; 9GO5 subunit and helix: 167 deg, 2.76 nm per subunit) coated by tropomyosin TPM1 (teal) along both strands, running past polysomes and cytosolic proteins.",
                       target=mid, normal=u, up=tdir, width=400, slab=40))
    if "mt_pos" in pre:    # SPB: section through the plaque axis, nuclear MT bouquet below, cytoplasmic MTs above
        spb, sd = c3.nuc_surface(G, cn - G[7:10])
        nuc_mt = pre["mt_reg"] == t3.R_NUCP
        side = unit(np.cross(sd, unit(pre["mt_pos"][nuc_mt].mean(0) - spb) + [0.1, 0.2, 0.3]))
        bm.append(dict(name="Spindle pole body", caption="Microtubules (green; 8QV3 lattice, 13 protofilaments) grow from gamma-tubulin rings on the spindle pole body: "
                       "a bouquet of short nuclear microtubules (16 reach the centromeres, Rabl configuration) and a few long cytoplasmic ones.",
                       target=spb, normal=side, up=sd, width=900, slab=60))
        cyt = pre["mt_kind"] == "cMT"
        if cyt.any():
            j = int(np.argmax(np.where(cyt, pre["mt_len"], 0))); sub = pre["mt_id"] == j
            P = pre["mt_pos"][sub]; mid = P[len(P) // 2]; tdir = unit(P[-1] - P[0])
            bm.append(dict(name="Cytoplasmic microtubule", caption="A cytoplasmic microtubule, 25 nm wide and about a micrometre long, curving through the crowded cytoplasm.",
                           target=mid, normal=unit(np.cross(tdir, unit(mid))), up=tdir, width=700, slab=40))
    if "pat_pos" in pre and len(pre["pat_pos"]):   # largest actin patch, seen from the side (membrane at the top)
        j = int(np.bincount(pre["pat_id"]).argmax()); P = pre["pat_pos"][pre["pat_id"] == j]; c = P.mean(0); u = -c3._grad(lambda x, y, z: c3.outer_depth(x, y, z, G), c)
        bm.append(dict(name="Actin patch", caption="Endocytic actin patch under the plasma membrane: short actin filaments (orange) branching at 73 deg from "
                       "Arp2/3 complexes (red; 8UXW junction) and capped by capping protein (purple; 8F8Q), barbed ends toward the membrane.",
                       target=c, normal=perp(u), up=u, width=360, slab=50))
    pcen = cell.pores[:, :3]
    k = int(np.argmax(np.linalg.norm(pcen, axis=1))); ax = unit(cell.pores[k, 3:6])
    bm.append(dict(name="Nuclear pore complex", caption="NPC (PDB 7N9F) seated in a toroidal pore membrane joining the outer and inner nuclear membranes.", target=pcen[k], normal=perp(ax), up=ax, width=260, slab=50))
    nucp = lcol.index("nucleoplasm")
    i = [i for i in np.argsort(-tcount["nucleosome"]) if lv[i, nucp] > 0.95 * lv[i].sum()][0]
    bm.append(dict(name="Chromatin", caption="Nucleoplasm: nucleosome chains (73,150 nucleosomes, linker DNA) with RNA polymerase II and nuclear proteins.", target=tc(i), normal=unit([0.5, 0.3, -0.81]), up=unit([0, 1, 0.37]), width=200, slab=40))
    nucl = lcol.index("nucleolus")
    i = [i for i in range(len(tiles)) if lv[i, nucl] > 0.3 * lv[i].sum() and lv[i, nucp] > 0.3 * lv[i].sum()][0]; uu = unit(G[7:10] - cn)
    bm.append(dict(name="Nucleolus", caption="Nucleolar crescent (ribosome assembly factors, snoRNPs, Pol I) bordering the nucleoplasm.", target=tc(i), normal=perp(uu), up=uu, width=450, slab=40))
    seg = np.array(geo["mito_segments"][len(geo["mito_segments"]) // 2]); a = unit(seg[1] - seg[0])
    bm.append(dict(name="Mitochondrion", caption="Mitochondrial tubule: outer and inner membranes, lamellar cristae lined at their rims by rows of ATP synthase dimers (8F39 monomers in the 4B2Q dimer arrangement), and the protein-dense matrix (~110 mg/mL).", target=seg.mean(0), normal=perp(a), up=a, width=420, slab=40))
    u = vac_u
    bm.append(dict(name="Vacuole", caption="Vacuolar membrane with V-ATPases; the vacuole lumen is dilute (~4 mg/mL). The vacuole is a cluster of lobes, as in cryo-FIB/SEM of log-phase cells.", target=vac_target, normal=perp(u), up=u, width=300, slab=40))
    q = cell.PQ[[i for i, c in enumerate(cell.PC) if c == c3.C_LD][0]]
    bm.append(dict(name="Lipid droplet", caption="Lipid droplet: neutral-lipid core with a phospholipid monolayer and peripheral proteins, touching ER.", target=q[:3], normal=unit([0, 0, 1]), up=unit([0, 1, 0]), width=560, slab=40))
    q = cell.PQ[[i for i, c in enumerate(cell.PC) if c == c3.C_GOLGI][0]]
    bm.append(dict(name="Golgi / endosomes", caption="Golgi cisternae and endosomes with their membrane and lumenal proteins.", target=q[:3], normal=unit([0, 0, 1]), up=unit([0, 1, 0]), width=500, slab=40))
    for b in bm:
        for k in ("target", "normal", "up"): b[k] = [round(float(x), 3) for x in b[k]]
    json.dump(bm, open(f"{we.OUT}/bookmarks.json", "w"), indent=1)


def wall_budget(tiles, cell, ing, labvol):
    """wall polysaccharide budgets (protein 45% of dry mass, wall 20% of dry mass, Klis 2002 composition)"""
    prot_da = float((ing.copies_per_cell * ing.mass_da).sum())
    dry = prot_da / 0.45; wall = dry * 0.20
    glc_res = wall * (0.525 + 0.075) / 162.14; man_res = wall * 0.375 * 0.85 / 162.14
    vol_glucan = labvol["wall_glucan"].sum(); vol_mannan = labvol["wall_mannan"].sum()
    lv = labvol.values; lcol = list(labvol.columns); n_wp = 0
    for i, t in enumerate(tiles):
        if not (lv[i, lcol.index("wall_mannan")] > 0): continue
        z = np.load(f"{C.WEB_TMP}/{t3.tile_name(t)}.npz")
        k = z["region"] == t3.REGION_NAMES.index("cell_wall")
        n_wp += int(sum(c3.outer_depth(p[0], p[1], p[2], cell.G) < cell.G[0] - cell.G[1] + 5 for p in z["pos"][k]))
    return dict(dry_mass_pg=dry * 1.66054e-12, wall_mass_pg=wall * 1.66054e-12, glucan_glucose=glc_res, mannose=man_res,
                glucan_beads_per_nm3=glc_res / 36 / vol_glucan, glucan_layer_um3=vol_glucan / 1e9, mannan_layer_um3=vol_mannan / 1e9,
                wall_proteins_in_mannan_layer=n_wp, mannan_beads_per_protein=man_res / 3 / max(n_wp, 1))


if __name__ == "__main__":
    main()
