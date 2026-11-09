"""Stage 1: inventory -> packing ingredients (inventory v3). Notebook cells 1-4, 22, plus assemblies.py.

Order: curated complexes (ingredients.COMPLEXES) -> Complex Portal complexes with RCSB assemblies ->
ER-bound ribosome-translocon complexes -> Arp2/3 complex -> F-actin reservation -> AlphaFold heterodimers with interaction evidence -> homo-oligomers of the residual monomer pool. Each step consumes subunit copies from the pool.
Outputs: results/inventory/{ingredients.csv, oligo_map.json, complexes_tried.csv, complexes_added.csv, heterodimers.csv,
oligomers.csv}
"""
from _common import *
import inventory as inv, ingredients as ingm, extend_complexes as xc, assemblies as asm, heterodimers as hd, er_ribosomes as er, actin_patches as ap, cytoskeleton as cs, supercomplexes as sc

# curated PDB homo-oligomers from inventory v2 (paralogs reuse the same entry)
LEGACY_OLIGO = ["TDH3", "PDC1", "CDC19", "ADH1", "TPI1", "GPM1", "ENO1", "AHP1", "SOD1", "HSP82", "IPP1", "GND1", "TRR1",
                "HOM6", "YDJ1", "CYS3", "TKL1", "KRS1", "TDH1", "TDH2", "ENO2", "ADH2", "PDC5"]


def mass_in_assemblies(ing, oligo_map):
    m = ing.copies_per_cell * ing.mass_da
    asmb = (ing.kind == "complex") | ing.ing_id.isin(list(oligo_map))
    return float(m[asmb].sum() / m.sum())


def main():
    df, geo = inv.build()
    asm.build_barrel_monomers(df, log=log)
    ing, _ = ingm.build_ingredients(df)
    i1 = ing.set_index("ing_id")
    legacy = {g: (g + "_oligo", int(i1.loc[g, "oligomer_n"])) for g in LEGACY_OLIGO if os.path.exists(f"{C.LIB}/{g}_oligo.npz")}
    modeled = set(g for gs in ing[ing.kind == "complex"].genes for g in gs.split(";")) | set(legacy)
    log(f"curated complexes {int((ing.kind == 'complex').sum())}, legacy homo-oligomers {len(legacy)}; "
        f"mass in assemblies {mass_in_assemblies(ing, legacy):.3f}")

    chosen, tried = asm.extend_complexes(df, ing, modeled, log=log)
    ing2 = asm.apply_complexes(ing, df, chosen)
    tried.to_csv(f"{INV}/complexes_tried.csv", index=False)
    pd.DataFrame(chosen).drop(columns=["sub"]).assign(subunits=[json.dumps(c["sub"]) for c in chosen]).to_csv(f"{INV}/complexes_added.csv", index=False)
    log(f"{len(chosen)} Complex Portal complexes added; outcomes {tried.outcome.value_counts().to_dict()}")

    # the excess free ribosomal-protein pool (MS noise above the ribosome count) is not packed
    ing2 = ing2[~((ing2.kind == "protein") & ing2.ing_id.str.match(r"^RP[LS]\d"))].reset_index(drop=True)
    # ribosomes docked on free translocons (SSH1 complex, SEC61 trimers) in the ER membrane
    ing2, er_info = er.assign(ing2, df, log=log)
    # respiratory supercomplexes III2IV2 from the complex III and IV pools
    ing2 = sc.add(ing2, df, log=log)
    # Arp2/3 as one 7-subunit complex (AlphaFold subunits on the 8UXW branch-junction template)
    ing2 = ap.add_arp23(ing2, df, log=log)
    # F-actin (cables + patches) is reserved before the heterodimer step, which then pairs only monomeric actin
    total = float(df.set_index("gene").loc["ACT1", "copies_per_cell"])
    r = (ing2.ing_id == "ACT1") & (ing2.kind == "protein"); free = float(ing2.loc[r, "copies_per_cell"].sum())
    want = {"cables": cs.CABLE_FRACTION * total, "patches": ap.PATCH_FRACTION * total}
    f = min(1.0, free / sum(want.values())); res = {k: v * f for k, v in want.items()}
    ing2.loc[r, "copies_per_cell"] = free - sum(res.values())
    json.dump(dict(res, total_act1=total), open(F_ACTIN_JSON, "w"), indent=1)
    log(f"F-actin reserved: cables {res['cables']:.0f}, patches {res['patches']:.0f} of {total:.0f} ACT1; monomer pool {free - sum(res.values()):.0f}")
    complexes = [{g.strip() for g in gs.split(";")} for gs in ing2[ing2.kind == "complex"].genes]
    ing2, htab = hd.assign(ing2, df, complexes, legacy, log=log)
    htab.to_csv(f"{INV}/heterodimers.csv", index=False)
    oligo_map, otab = asm.homo_oligomers(ing2, df, legacy, log=log)
    otab.to_csv(f"{INV}/oligomers.csv", index=False)
    log("homo-oligomer decisions:", otab.decision.str.replace(r"\([^)]*\)|AF-\S+|\b[0-9][0-9A-Z]{3}\b", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip().value_counts().to_dict())
    log(f"mass in assemblies {mass_in_assemblies(ing2, oligo_map):.3f}")
    ing2.to_csv(ING_CSV, index=False)
    json.dump({k: list(v) for k, v in oligo_map.items()}, open(OLIGO_JSON, "w"), indent=0)
    asm.save_cache()


if __name__ == "__main__":
    main()
