"""Stage 4: validation of the packed cell + libpack_meta.npz (type order, bounding radii, structure source)
used by the encoder. Notebook cells 31-36."""
from _common import *
from scipy.spatial import cKDTree
import cell3d as c3, tiles3d as t3, pack3d as p3, val3d as v3, wc3d, ingredients as ingm

CURATED_PDB = {c[0]: c[3] for c in ingm.COMPLEXES}
FIXED_PDB = {"NPC": "7N9F", "nucleosome": "1ID3", "tRNA": "1EHZ"}


def src_of(lid):
    """(source, pdb id) of a library type: 'PDB', 'AF' (AlphaFold DB monomer) or 'AF-multimer' (predicted assembly)."""
    z = np.load(f"{C.LIB}/{lid}.npz")
    if "src" in z.files:
        return str(z["src"]), str(z["pdb"]) if "pdb" in z.files else ""
    if lid in FIXED_PDB: return "PDB", FIXED_PDB[lid]
    if lid in CURATED_PDB: return "PDB", CURATED_PDB[lid]
    if "pdb" in z.files: return "PDB", str(z["pdb"])
    return "AF", ""


def atp_check(pre, lp, cell):
    """ATP synthase dimers on cristae rims: TM-residue distance to the inner-membrane midsurface"""
    from scipy.spatial.transform import Rotation
    z = np.load(f"{C.LIB}/ATP_dimer.npz"); B = z["beads"][z["tm"].astype(bool)].astype(np.float64)
    out = np.zeros(c3.NMEM); a = cell.args(np.full(3, -3000.0), np.full(3, 3000.0), margin=50.0); d = []
    for p, q in list(zip(pre["atp_pos"], pre["atp_q"]))[:200]:
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        for x in B[::7] @ R.T + p:
            c3.mem_sdfs(x[0], x[1], x[2], *a, out); d.append(abs(out[c3.M_IM]))
    d = np.array(d)
    return dict(n=int(len(pre["atp_pos"])), tm_median_nm=float(np.median(d)) if len(d) else None,
                tm_frac_within_2nm=float((d < 2.0).mean()) if len(d) else None)


def main():
    tiles = canonical_tiles()
    cell = c3.Cell.load(f"{C.W3D}/geometry.json")
    lib_ids = [str(x) for x in np.load(f"{C.W3D}/lib_ids.npy")]
    lp = wc3d.build_libpack(lib_ids)
    pre = dict(np.load(f"{C.W3D}/prepass.npz"))
    labvol = pd.read_parquet(f"{C.W3D}/tile_labvol.parquet")
    psum = pd.read_parquet(f"{C.W3D}/pack_summary.parquet")
    exp3 = pd.read_parquet(f"{C.W3D}/expected_totals.parquet") if os.path.exists(f"{C.W3D}/expected_totals.parquet") else None

    TY, PO, QU, TI, JOB, MRP, MRT = v3.load_instances(tiles, wc3d.RAW)
    for k in ("mrna_pts", "fg_pts"):   # pre-pass threads (s3): full-length mRNA, FG-repeat domains
        if k in pre:
            MRP = np.concatenate([MRP, pre[k].astype(MRP.dtype)])
    REG = np.concatenate([np.repeat(g.region.values, g.placed.values) for _, g in JOB.groupby("tile", sort=True)]).astype(np.int16)
    assert len(REG) == len(TY)
    li_npc, li_nuc = int(pre["li_npc"]), int(pre["li_nuc"])
    gt, gp, gq = wc3d.pre_instances(pre)          # NPCs, nucleosomes, polysome ribosomes
    TYa = np.concatenate([TY, gt]); POa = np.concatenate([PO, gp]); QUa = np.concatenate([QU, gq])
    TIa = np.concatenate([TI, np.full(len(gt), -1, np.int32)])
    boff, BB, BK, LMASS, _ = v3.lib_beads(lp["ids"])
    log(f"instances {len(TYa) / 1e6:.2f}M; residue/nt beads {np.diff(boff)[TYa].sum() / 1e9:.2f}G; mRNA beads {len(MRP) / 1e6:.1f}M")

    # ---- placed vs requested (vs expected) per region
    reqreg = JOB.groupby("region")[["n", "placed"]].sum()
    reg_tab = pd.DataFrame({"requested": reqreg.n.reindex(range(t3.NREG)).values,
                            "placed": reqreg.placed.reindex(range(t3.NREG)).values}, index=t3.REGION_NAMES).fillna(0)
    if exp3 is not None:
        reg_tab.insert(0, "expected", exp3[exp3.region.isin(t3.REGION_NAMES)].groupby("region").expected.sum().reindex(t3.REGION_NAMES).fillna(0).values)
    reg_tab["frac_placed"] = reg_tab.placed / reg_tab.requested.clip(lower=1)
    reg_tab["unplaced"] = reg_tab.requested - reg_tab.placed
    log("\n" + reg_tab.round(3).to_string())

    # ---- residue-level clashes in 64 cubes of 50 nm (half on tile corners)
    t0 = time.time()
    ctree = cKDTree(POa); mtree = cKDTree(MRP); BRa = lp["BR"]
    rng = np.random.default_rng(42)
    corners = [np.array(tiles[i], float) * t3.TILE for i in rng.choice(len(tiles), 400)]
    pm_depth = lambda p: c3.outer_depth(p[0], p[1], p[2], cell.G) - (cell.G[0] - cell.G[2])
    corners = [c for c in corners if pm_depth(c) > 60][:32]
    randc = []
    while len(randc) < 32:
        p = rng.uniform(-2000, 2000, 3)
        if pm_depth(p) > 40: randc.append(p)
    half = 25.0; n_beads = n_clash = n_clash_x = 0
    for cc in corners + randc:
        cand = np.array(ctree.query_ball_point(cc, half * np.sqrt(3) + 66.0), np.int64)
        if len(cand) == 0: continue
        cand = cand[np.linalg.norm(POa[cand] - cc, axis=1) < half * np.sqrt(3) + BRa[TYa[cand]] + 0.5]
        nb_ = np.diff(boff)[TYa[cand]].sum()
        pts = np.empty((nb_, 3)); inst = np.empty(nb_, np.int64)
        v3.beads_of(cand, TYa, POa.astype(np.float64), QUa.astype(np.float64), boff, BB, pts, inst)
        k = np.all(np.abs(pts - cc) < half + 0.5, axis=1); pts, inst = pts[k], inst[k]
        mi = np.array(mtree.query_ball_point(cc, half * np.sqrt(3) + 1), np.int64)
        if len(mi):
            mp_ = MRP[mi].astype(np.float64); kk = np.all(np.abs(mp_ - cc) < half + 0.5, axis=1)
            pts = np.vstack([pts, mp_[kk]]); inst = np.concatenate([inst, -2 - mi[kk]])
        tr = cKDTree(pts); core = np.all(np.abs(pts - cc) < half, axis=1)
        pairs = tr.query_pairs(0.3, output_type="ndarray")
        a, b = inst[pairs[:, 0]], inst[pairs[:, 1]]
        diff = a != b
        bad = np.zeros(len(pts), bool); bad[pairs[diff, 0]] = True; bad[pairs[diff, 1]] = True
        tile_of = lambda x: np.where(x >= 0, TIa[np.clip(x, 0, None)], -9)
        xt = diff & (tile_of(a) != tile_of(b)) & (a >= 0) & (b >= 0)
        badx = np.zeros(len(pts), bool); badx[pairs[xt, 0]] = True; badx[pairs[xt, 1]] = True
        n_beads += core.sum(); n_clash += (bad & core).sum(); n_clash_x += (badx & core).sum()
    clash = dict(beads_sampled=int(n_beads), beads_clashing_lt_0p3nm=int(n_clash), clash_rate=n_clash / n_beads,
                 cross_tile_clashing_beads=int(n_clash_x), cubes=len(corners) + len(randc))
    log("clash", clash, f"{time.time() - t0:.0f}s")

    # ---- containment and membrane orientation
    allp = np.arange(len(cell.PT)); rng = np.random.default_rng(7)
    PO64, QU64 = PO.astype(np.float64), QU.astype(np.float64)
    volidx = np.where(REG < t3.MEM_REGION0)[0]; s = rng.choice(volidx, 30000, replace=False)
    nb_ = np.diff(boff)[TY[s]].sum(); pts = np.empty((nb_, 3)); inst = np.empty(nb_, np.int64)
    v3.beads_of(s, TY, PO64, QU64, boff, BB, pts, inst)
    labb = c3.label_points(pts, cell.G, cell.pores, allp, cell.PT, cell.PQ, cell.PC)
    soft = {c3.NUCP: {c3.NUCL}, c3.NUCL: {c3.NUCP}, c3.MANNAN: {c3.GLUCAN}, c3.GLUCAN: {c3.MANNAN}}
    mask_b = t3.REG_MASK[REG[inst]]
    ok = ((mask_b >> labb) & 1).astype(bool)
    for l, alts in soft.items():
        for a_ in alts:
            ok |= (((mask_b >> l) & 1).astype(bool)) & (labb == a_)
    contain = dict(volume_instances_sampled=30000, beads=int(nb_), frac_beads_outside_region=float((~ok).mean()),
                   frac_instances_any_bead_outside=float(len(np.unique(inst[~ok])) / 30000))
    memidx = np.where(REG >= t3.MEM_REGION0)[0]; s = rng.choice(memidx, min(10000, len(memidx)), replace=False)
    ang = []; out = np.zeros(c3.NMEM); n_ = np.empty(3)
    for i in s:
        m = REG[i] - t3.MEM_REGION0; M = np.empty((3, 3)); p3.quat_mat(QU64[i], M)
        p = PO64[i].copy(); p3.project_to_membrane(p, m, n_, cell.G, cell.pores, allp, cell.PT, cell.PQ, cell.PC, out)
        ang.append(np.degrees(np.arccos(np.clip(M[:, 2] @ n_, -1, 1))))
    ang = np.array(ang)
    orient = dict(membrane_instances_sampled=len(s), median_angle_deg=float(np.median(ang)), frac_angle_lt_1deg=float((ang < 1).mean()))
    # TM residues of automatically oriented membrane assemblies: distance to their membrane's midplane
    tmmask = {i: np.load(f"{C.LIB}/{lid}.npz")["tm"] for i, lid in enumerate(lp["ids"])
              if "tm" in np.load(f"{C.LIB}/{lid}.npz").files}
    sel = memidx[np.isin(TY[memidx], list(tmmask))]
    tmdepth = None
    if len(sel):
        s = rng.choice(sel, min(3000, len(sel)), replace=False)
        nb_ = np.diff(boff)[TY[s]].sum(); pts = np.empty((nb_, 3)); inst = np.empty(nb_, np.int64)
        v3.beads_of(s, TY, PO64, QU64, boff, BB, pts, inst)
        keep = np.concatenate([tmmask[TY[i]] for i in s])
        d = np.array([(c3.mem_sdfs(*p, cell.G, cell.pores, allp, cell.PT, cell.PQ, cell.PC, out), abs(out[REG[i] - t3.MEM_REGION0]))[1]
                      for p, i in zip(pts[keep], inst[keep])])
        tmdepth = dict(instances=len(s), tm_residues=int(keep.sum()), median_nm=float(np.median(d)),
                       frac_within_2nm=float((d < 2.0).mean()))
    log("containment", contain); log("orientation", orient); log("TM residues of oriented assemblies", tmdepth)

    # ---- density per compartment (mg/mL)
    reg2comp = {"cytosol": "cytosol", "PM_peripheral": "cytosol", "NE_peripheral": "cytosol", "ER_peripheral": "cytosol",
                "LD_peripheral": "cytosol", "nucleoplasm": "nucleoplasm", "nucleolus": "nucleolus", "ER_lumen": "ER_lumen",
                "vacuole": "vacuole", "mito_matrix": "mito_matrix", "mito_IMS": "mito_IMS", "golgi_endosome": "golgi_endosome",
                "peroxisome": "peroxisome", "cell_wall": "cell_wall"}
    dfm = pd.DataFrame(dict(comp=pd.Series(np.array(t3.REGION_NAMES)[REG]).map(reg2comp), mass=LMASS[TY]))
    cm = dfm.groupby("comp").mass.sum()
    cm["nucleoplasm"] += len(pre["nuc_pos"]) * LMASS[li_nuc]
    if "poly_pos" in pre:
        cm["cytosol"] += len(pre["poly_pos"]) * LMASS[int(pre["li_ribo"])]
    if "act_pos" in pre:
        cm["cytosol"] += len(pre["act_pos"]) * LMASS[int(pre["li_actin"])]
    if "mt_pos" in pre:
        for rg, comp in ((t3.R_CYTO, "cytosol"), (t3.R_NUCP, "nucleoplasm")):
            cm[comp] += LMASS[pre["mt_type"][pre["mt_reg"] == rg]].sum()
    if "pat_pos" in pre:
        cm["cytosol"] += LMASS[pre["pat_type"]].sum()
    if "tpm_pos" in pre:
        cm["cytosol"] += len(pre["tpm_pos"]) * LMASS[int(pre["li_tpm"])]
    lv = labvol.sum()
    cvol = {"cytosol": lv.cytosol, "nucleoplasm": lv.nucleoplasm, "nucleolus": lv.nucleolus, "ER_lumen": lv.cER_lumen + lv.perinuclear,
            "vacuole": lv.vacuole, "mito_matrix": lv.mito_matrix, "mito_IMS": lv.mito_IMS, "golgi_endosome": lv.golgi_endosome,
            "peroxisome": lv.peroxisome, "cell_wall": lv.wall_mannan + lv.wall_glucan}
    mgml = {k: float(cm.get(k, 0) * 1.66054e-24 / (v * 1e-21) * 1000) for k, v in cvol.items()}
    log("density mg/mL", {k: round(v, 1) for k, v in mgml.items()})

    # ---- structure source per type
    SP = [src_of(str(i)) for i in lp["ids"]]
    SRC = np.array([s for s, _ in SP]); PDBID = np.array([p for _, p in SP])
    src_all = SRC[TYa]
    mass_all = LMASS[TYa]
    srcm = (pd.Series(mass_all).groupby(src_all).sum() / mass_all.sum()).round(4).to_dict()
    srcn = pd.Series(src_all).value_counts(normalize=True).round(4).to_dict()
    log("mass by source", srcm, "| PDB types", int((SRC == "PDB").sum()), "| AF-multimer types", int((SRC == "AF-multimer").sum()))

    nonmito = ~reg_tab.index.isin(["mito_matrix", "mito_IMS", "mem_MITO_OM", "mem_MITO_IM"])
    frac_nonmito = float(reg_tab[nonmito].placed.sum() / reg_tab[nonmito].requested.sum())
    val = dict(
        model="S. cerevisiae BY4741 haploid, YPD exponential; whole-cell 3D packing in 200 nm tiles (8 checkerboard phases)",
        n_tiles=len(tiles), library_types=len(lp["ids"]), instances_total=int(len(TYa)),
        residue_nt_beads_total=int(np.diff(boff)[TYa].sum()),
        mrna_threads=int(psum.n_mrna.sum() + (len(pre["mrna_off"]) - 1 if "mrna_off" in pre else 0)), mrna_beads=int(len(MRP)),
        mrna_full_length=dict(threads_reaching_target=float(np.mean(np.diff(pre["mrna_off"]) >= pre["mrna_target"])),
                              nt_fraction_of_target=float(len(pre["mrna_pts"]) / pre["mrna_target"].sum()),
                              total_nt_incl_ribosome_channels=int(len(pre["mrna_pts"]) + len(pre.get("poly_mrna", ())) * 0.5 / 0.6))
                         if "mrna_pts" in pre else None,
        chromatin=dict(nucleosomes_placed=int(len(pre["nuc_pos"])), nucleosomes_requested=int(pre["chain_target"].sum())),
        npc=int(len(pre["npc_pos"])),
        polysomes=dict(chains=int(pre["poly_chains"]), ribosomes=int(len(pre["poly_pos"])),
                       mrna_beads=int(len(pre["poly_mrna"]))) if "poly_pos" in pre else None,
        actin_cables=dict(filaments=int(len(np.unique(pre["act_fil"]))), subunits=int(len(pre["act_pos"])),
                          length_um=float(len(pre["act_pos"]) * 2.76 / 1000)) if "act_pos" in pre else None,
        microtubules=dict(n=int(len(pre["mt_len"])), by_kind={k: int((pre["mt_kind"] == k).sum()) for k in ("kMT", "nMT", "cMT")},
                          tubulin_dimers=int((pre["mt_type"] == list(lp["ids"]).index("MT_tubulin")).sum()),
                          mean_length_nm={k: float(pre["mt_len"][pre["mt_kind"] == k].mean()) for k in ("kMT", "nMT", "cMT")
                                          if (pre["mt_kind"] == k).any()}) if "mt_pos" in pre else None,
        atp_synthase_dimers=atp_check(pre, lp, cell) if "atp_pos" in pre else None,
        fg_mesh=dict(domains=int(len(pre["fg_off"]) - 1), residues=int(len(pre["fg_pts"]))) if "fg_pts" in pre else None,
        tropomyosin_dimers=int(len(pre["tpm_pos"])) if "tpm_pos" in pre else None,
        actin_patches=dict(n=int(len(np.unique(pre["pat_id"]))), by_type={str(lp["ids"][t]): int((pre["pat_type"] == t).sum()) for t in np.unique(pre["pat_type"])})
                      if "pat_pos" in pre else None,
        er_ribosomes={str(lp["ids"][t]): int((TY == t).sum()) for t in range(len(lp["ids"])) if str(lp["ids"][t]).startswith("ribosome_80S_")},
        placed_fraction_total=float(reg_tab.placed.sum() / reg_tab.requested.sum()),
        placed_fraction_outside_mitochondria=frac_nonmito,
        regions=reg_tab.reset_index().rename(columns={"index": "region"}).round(4).to_dict(orient="records"),
        clash=clash, containment=contain, membrane_orientation=orient, tm_residue_depth=tmdepth, density_mg_per_ml=mgml,
        mass_fraction_by_structure_source=srcm, instance_fraction_by_structure_source=srcn,
        acceptance={"placed_outside_mito_ge_0p99": frac_nonmito >= 0.99, "clash_rate_lt_1e-5": bool(clash["clash_rate"] < 1e-5)})
    json.dump(val, open(f"{C.RESULTS}/wholecell3d_validation.json", "w"), indent=1, cls=NpEnc)
    np.savez(f"{C.W3D}/libpack_meta.npz", ids=np.array([str(i) for i in lp["ids"]]), BR=lp["BR"], src=SRC, pdb=PDBID)
    log("acceptance", val["acceptance"])


if __name__ == "__main__":
    main()
