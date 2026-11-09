"""Stage 3: expected totals per (library type, region), NPC + microtubule + chromatin + actin cable + polysome pre-pass, distribution over tiles,
checkerboard-phased parallel tile packing. Notebook cells 27-29.

  python s3_pack.py                      # from s1 inventory + s2 geometry, with the organisation layer (organisation.py)
  python s3_pack.py --random             # the same with uniform random placement
  python s3_pack.py --from-intermediates DIR
        # re-pack the shipped model: reuse DIR/{tile_counts.parquet, prepass.npz, mrna_counts.npy, libpack_meta.npz,
        # partners.npz (organisation layer; absent: uniform placement)}
"""
import argparse, shutil
from _common import *
import organisation as og
import cell3d as c3, tiles3d as t3, pack3d as p3, wc3d, polysome as ps, cytoskeleton as cs, microtubules as mt, actin_patches as patches, mrna, atp_dimers as atp, fg_mesh as fgm, inventory as inv


def consume(exp3, ing_ids, n):
    """remove n instances from the expected totals of these ingredients (all regions, proportionally); returns n taken"""
    r = exp3.ing_id.isin(ing_ids); tot = float(exp3.loc[r, "expected"].sum()); take = min(max(n, 0), tot)
    if tot > 0:
        exp3.loc[r, "expected"] *= 1 - take / tot
    log(f"  {'/'.join(ing_ids)}: {take:.0f} of {tot:.0f} used ({n:.0f} needed)")
    return take


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-intermediates")
    ap.add_argument("--random", action="store_true", help="uniform random placement (no organisation layer)")
    args = ap.parse_args()
    org = org_prm = None
    cell = c3.Cell.load(f"{C.W3D}/geometry.json")
    tiles = canonical_tiles()
    measure = pd.read_parquet(f"{C.W3D}/tile_measure.parquet")
    assert list(measure.index) == [t3.tile_name(t) for t in tiles]

    if args.from_intermediates:
        d = args.from_intermediates
        lib_ids = [str(x) for x in np.load(f"{d}/libpack_meta.npz")["ids"]]
        lp = wc3d.build_libpack(lib_ids)
        counts = pd.read_parquet(f"{d}/tile_counts.parquet")
        mrna_counts = np.load(f"{d}/mrna_counts.npy")
        pre = dict(np.load(f"{d}/prepass.npz"))
        for k in ("li_npc", "li_nuc"): pre[k] = int(pre[k])
        exp3 = None
        if os.path.exists(f"{d}/partners.npz") and not args.random:      # organisation layer of the shipped build
            z = np.load(f"{d}/partners.npz")
            assert [str(x) for x in z["ids"]] == [str(x) for x in lp["ids"]]
            org = {k: z[k] for k in ("off", "idx", "strength", "kind")}; org_prm = og.params()
    else:
        ing, oligo_map = load_inventory()
        exp3 = wc3d.expected_totals(ing, oligo_map)
        log("expected per region:", exp3.groupby("region").expected.sum().round(0).to_dict())
        dropped = exp3[~exp3.region.isin(t3.REGION_NAMES)].groupby("region").expected.sum()
        log("not packed:", dropped.round(0).to_dict())
        cs.filament_model(); cs.tropomyosin_model(log=log); mt.model(); atp.model(log=log)
        lib_ids = sorted(set(exp3[exp3.region.isin(t3.REGION_NAMES)].lib_id) | {"NPC", "nucleosome", cs.LIB_ID, cs.TPM_ID, atp.LIB_ID, *mt.LIB_IDS, *patches.LIB_IDS})
        lp = wc3d.build_libpack(lib_ids)
        n_nuc = int(round(ing.set_index("ing_id").loc["nucleosome", "copies_per_cell"]))
        # microtubules from the SPB (alpha/beta-tubulin heterodimer pool), placed before chromatin: kinetochore-MT
        # plus ends seed the chromosome walks
        tub = (exp3.ing_id == mt.TUBULIN_ING) & (exp3.region == "cytosol")
        mtd = {}

        def microtubules(npc_spheres):
            mtd.update(mt.place(cell, lp, float(exp3.loc[tub, "expected"].sum()), obstacles=npc_spheres, log=log))
            nucl = mtd["mt_reg"] == t3.R_NUCP
            C, R = p3.emit_spheres(mtd["mt_type"][nucl], mtd["mt_pos"][nucl], mtd["mt_q"][nucl], lp["st_off"], lp["STC"],
                                   lp["STR"], lp["BR"], np.full(3, -1e4), np.full(3, 1e4))
            return dict(obstacles=(C, R), starts=mtd["kmt_tips"])
        pre = wc3d.prepass(cell, lp, n_nuc, log=log, before_chromatin=microtubules)
        pre.update({k: v for k, v in mtd.items() if k.startswith("mt_")})
        # FG-repeat domains of the nucleoporins, threaded into the pores (avoiding the NPC scaffold, chromatin, MTs)
        gt0, gp0, gq0 = wc3d.pre_instances(pre)
        FC, FR = p3.emit_spheres(gt0, gp0, gq0, lp["st_off"], lp["STC"], lp["STR"], lp["BR"], np.full(3, -1e4), np.full(3, 1e4))
        pre.update(fgm.place(cell, pre, (FC, FR), log=log))
        FGC = pre["fg_pts"].astype(np.float64); FGR = np.full(len(FGC), fgm.BEAD_R)
        exp3.loc[tub, "expected"] = (exp3.loc[tub, "expected"] - mtd["n_tub"]).clip(lower=0)
        # ring subunits: gamma-TuSC complexes first, then the free subunit pools (SPC110 from its CMD1 heterodimer)
        n_all = mtd["n_rings_nuc"] + mtd["n_rings_cyto"]
        m = consume(exp3, [mt.TUSC_ING], 6 * n_all)
        for g, k in mt.RING_GENES.items():
            consume(exp3, [g], k * n_all - (2 if g == "TUB4" else 1) * m)
        consume(exp3, ["SPC110", "CMD1_SPC110_het"], 14 * mtd["n_rings_nuc"])
        cyto_mt = mtd["mt_reg"] == t3.R_CYTO
        MC, MR = p3.emit_spheres(mtd["mt_type"][cyto_mt], mtd["mt_pos"][cyto_mt], mtd["mt_q"][cyto_mt], lp["st_off"], lp["STC"],
                                 lp["STR"], lp["BR"], np.full(3, -1e4), np.full(3, 1e4))
        # F-actin reserved in s1: cables, then Arp2/3-branched patches; whatever does not fit returns to the monomer pool
        fa = json.load(open(F_ACTIN_JSON))
        cab = cs.place(cell, lp, int(fa["cables"]), obstacles=(MC, MR), log=log)
        pre.update(cab)
        # tropomyosin (TPM1 coiled coils) along both long-pitch strands of the cable filaments
        tpm = cs.decorate(cell, lp, cab, obstacles=(MC, MR), log=log)
        pre.update(tpm)
        consume(exp3, [cs.TPM_ING], 2 * len(tpm["tpm_pos"]))
        CC, CR = p3.emit_spheres(np.r_[np.full(len(cab["act_pos"]), cab["li_actin"]), np.full(len(tpm["tpm_pos"]), tpm["li_tpm"])].astype(np.int32),
                                 np.r_[cab["act_pos"], tpm["tpm_pos"]].astype(np.float32), np.r_[cab["act_q"], tpm["tpm_q"]].astype(np.float32),
                                 lp["st_off"], lp["STC"], lp["STR"], lp["BR"], np.full(3, -1e4), np.full(3, 1e4))
        patches.model(dict(zip(*[inv.build()[0][c] for c in ("gene", "uniprot")])))
        avail = lambda i: float(exp3.loc[(exp3.ing_id == i) & (exp3.region == "cytosol"), "expected"].sum())
        pat, used = patches.place(cell, lp, fa["patches"], avail(patches.ARP23), avail(patches.CP_ING),
                             obstacles=(np.concatenate([MC, CC]), np.concatenate([MR, CR])), log=log)
        pre.update(pat)
        consume(exp3, [patches.ARP23], used[patches.ARP23]); consume(exp3, [patches.CP_ING], used[patches.CP])
        back = fa["cables"] + fa["patches"] - len(cab["act_pos"]) - used[patches.ACTIN]
        act = (exp3.ing_id == "ACT1") & (exp3.region == "cytosol")
        if act.any():
            exp3.loc[act, "expected"] += back
        else:
            exp3.loc[len(exp3)] = ["ACT1", "ACT1", "cytosol", back]
        log(f"  F-actin: {len(cab['act_pos'])} in cables, {used[patches.ACTIN]} in patches, {back:.0f} returned to the monomer pool")
        PC_, PR_ = p3.emit_spheres(pat["pat_type"], pat["pat_pos"], pat["pat_q"], lp["st_off"], lp["STC"], lp["STR"], lp["BR"],
                                   np.full(3, -1e4), np.full(3, 1e4))
        OC, OR = np.concatenate([CC, PC_]), np.concatenate([CR, PR_])
        # ATP synthase dimer rows on cristae rims (from the monomer pool)
        im = (exp3.ing_id == atp.MONOMER) & (exp3.region == "mem_MITO_IM")
        dim = atp.place(cell, lp, int(exp3.loc[im, "expected"].sum() // 2), log=log)
        pre.update(dim)
        consume(exp3, [atp.MONOMER], 2 * len(dim["atp_pos"]))
        # polysomes: a share of the cytosolic ribosomes, threaded on mRNAs, placed before packing
        rib = (exp3.lib_id == ps.RIBO) & (exp3.region == "cytosol")
        poly = ps.place(cell, lp, ps.POLY_FRACTION * exp3.loc[rib, "expected"].sum(),
                        obstacles=(np.concatenate([OC, MC, FGC]), np.concatenate([OR, MR, FGR])), log=log)
        pre.update(poly)
        exp3.loc[rib, "expected"] = exp3.loc[rib, "expected"] - len(poly["poly_pos"])
        logmod = None
        if not args.random:
            centres = (np.array(tiles, float) + 0.5) * t3.TILE
            logmod, org = og.build(exp3, ing, [str(x) for x in lp["ids"]], centres, log=log)
            org_prm = og.params()
        vol = {str(k): float((4 / 3 * np.pi * lp["STR"][lp["st_off"][i]:lp["st_off"][i + 1]] ** 3).sum()) for i, k in enumerate(lp["ids"])}
        counts = wc3d.distribute(exp3, measure, {k: i for i, k in enumerate(lp["ids"])}, logmod=logmod, vol=vol)
        # full-length mRNAs, threaded before packing: polysome tails + free transcripts (one mRNA per polysome)
        from scipy.spatial.transform import Rotation
        import structlib as sl
        zr = np.load(f"{C.LIB}/{ps.RIBO}.npz")["beads"].astype(np.float64)
        rc, rr = sl.sphere_tree(zr, per=100)
        Rm = Rotation.from_quat(np.c_[poly["poly_q"][:, 1:], poly["poly_q"][:, :1]].astype(np.float64)).as_matrix()
        RC = (np.einsum("nij,sj->nsi", Rm, rc) + poly["poly_pos"][:, None, :]).reshape(-1, 3)
        npc_t = np.full(len(pre["npc_pos"]), pre["li_npc"], np.int32)
        NC, NR = p3.emit_spheres(npc_t, pre["npc_pos"], pre["npc_q"], lp["st_off"], lp["STC"], lp["STR"], lp["BR"], np.full(3, -1e4), np.full(3, 1e4))
        mr = mrna.place(cell, poly, ps.channel_path().astype(np.float64), int(wc3d.N_MRNA) - poly["poly_chains"],
                        (np.concatenate([OC, MC, NC, RC, poly["poly_link"], FGC]),
                         np.concatenate([OR, MR, NR, np.tile(rr, len(Rm)), np.full(len(poly["poly_link"]), 0.5), FGR])), log=log)
        pre.update(mr)
        mrna_counts = np.zeros(len(tiles), np.int64)          # no in-tile threads any more

    halo = float(max(lp["BR"][i] for i, k in enumerate(lp["ids"]) if k != "NPC")) + 1.0
    log(f"{len(lp['ids'])} library types, {len(lp['STR'])} sphere-tree spheres, halo {halo:.1f} nm; "
        f"{counts.n.sum() / 1e6:.2f}M instances to place")
    counts.to_parquet(f"{C.W3D}/tile_counts.parquet")
    np.save(f"{C.W3D}/mrna_counts.npy", mrna_counts)
    np.savez(f"{C.W3D}/prepass.npz", **pre)
    np.save(f"{C.W3D}/lib_ids.npy", np.array([str(i) for i in lp["ids"]]))
    if exp3 is not None:
        exp3.to_parquet(f"{C.W3D}/expected_totals.parquet")

    wc3d.setup_shared(cell, lp, counts, tiles, pre, halo, partners=org, prm=org_prm)
    shutil.rmtree(wc3d.RAW, ignore_errors=True); os.makedirs(wc3d.RAW)
    phase = np.array([wc3d.phase_of(t) for t in tiles])
    allres = []; T0 = time.time()
    for ph in range(8):
        ids = list(np.where(phase == ph)[0]); t0 = time.time()
        r = wc3d.run_phase(ids, mrna_counts)
        allres += r
        log(f"phase {ph}: {len(ids)} tiles, {time.time() - t0:.0f}s, placed {sum(x[2] for x in r)}/{sum(x[1] for x in r)}, "
            f"overflow {sum(x[3] for x in r)}")
    res = pd.DataFrame(allres, columns=["ti", "requested", "placed", "overflow", "sec", "n_mrna", "attract_tried", "attract_placed"])
    res.to_parquet(f"{C.W3D}/pack_summary.parquet")
    log(f"total {time.time() - T0:.0f}s; placed {res.placed.sum() / 1e6:.2f}M / {res.requested.sum() / 1e6:.2f}M "
        f"= {res.placed.sum() / res.requested.sum():.4f}; next to a partner {res.attract_placed.sum() / 1e6:.2f}M "
        f"of {res.attract_tried.sum() / 1e6:.2f}M that tried")


if __name__ == "__main__":
    main()
