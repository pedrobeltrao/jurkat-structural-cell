"""Respiratory supercomplexes (inventory v3, item B4): yeast complex III dimer with two complex IV, III2IV2.

Structure: PDB 6HU9 (S. cerevisiae III2IV2, cryo-EM 3.35 A; Hartley et al. 2019), oriented in the inner membrane like
the other membrane assemblies. Copies: min(III2, IV / 2) taken from the Complex Portal complexes III (CPX-567, a
III2 dimer) and IV (CPX-1721); the remaining III2 stays free. In yeast nearly all complex IV is in III2IV1/III2IV2
supercomplexes; III2IV1 is not modelled.
"""
import json
import numpy as np
import pandas as pd
import gemmi
import extend_complexes as xc
import assemblies as asm

SC_ID = "SC_III2IV2"
PDB = "6HU9"
III, IV = "CPX_567_Mitochondrial_respiratory_chain_", "CPX_1721_Mitochondrial_respiratory_chain_"


def add(ing, df, log=print):
    i = ing.set_index("ing_id")
    if III not in i.index or IV not in i.index:
        log("supercomplexes: complex III or IV missing"); return ing
    acc_of = dict(zip(df.gene, df.uniprot))
    genes = sorted(set(i.loc[III, "genes"].split(";")) | set(i.loc[IV, "genes"].split(";")))
    st = gemmi.read_structure(xc.fetch_assembly(PDB)); st.setup_entities()
    res, info = asm.build_complex(SC_ID, "III2IV2 supercomplex", PDB, st, genes, acc_of, False, tm=True)
    if res is None:
        log(f"supercomplexes: orientation failed {info}"); return ing
    n = float(min(i.loc[III, "copies_per_cell"], i.loc[IV, "copies_per_cell"] / 2))
    ing = ing.copy()
    ing.loc[ing.ing_id == III, "copies_per_cell"] -= n
    ing.loc[ing.ing_id == IV, "copies_per_cell"] -= 2 * n
    row = dict(ing_id=SC_ID, label="Respiratory supercomplex III2IV2", kind="complex", genes=";".join(genes),
               n_subunit_types=len(genes), copies_per_cell=n, comp_weights=json.dumps({"mito_IM": 1.0}),
               mass_da=float(i.loc[III, "mass_da"] + 2 * i.loc[IV, "mass_da"]), pdb=PDB, membrane="TM", oligomer_n=1,
               uniprot="", n_tm=int(i.loc[III, "n_tm"] + 2 * i.loc[IV, "n_tm"]))
    ing = pd.concat([ing[ing.copies_per_cell > 0.5], pd.DataFrame([row])], ignore_index=True)
    log(f"supercomplexes: {n:.0f} III2IV2 ({res[0]} residues; membrane {info}); free III2 {float(i.loc[III, 'copies_per_cell']) - n:.0f}, "
        f"free IV {float(i.loc[IV, 'copies_per_cell']) - 2 * n:.0f}")
    return ing
