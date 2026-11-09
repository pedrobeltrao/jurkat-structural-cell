"""Stage 1 adapter (Jurkat): inventory + assembly tables -> packing ingredients.

Inputs (data/derived): inventory_v0_complexed.tsv (copies at the working scale, compartments "comp:w;...", transmembrane),
complexes_v0.tsv (Complex Portal + curated nucleosome, copies_built, best_pdb, pdb_mass_cov), homo_oligomers_v0.tsv,
afdb_homodimers_v0.tsv, afdb_heterodimers_v0.tsv, structures/afdb_complexes/fetch_status.tsv.
Output: data/derived/ingredients_v0.tsv with columns
  ing_id, kind (complex|oligomer|afdb_homodimer|afdb_heterodimer|protein), genes (";"), stoich (";"), copies, mass_da,
  comp_weights (JSON, Jurkat compartment keys), transmembrane (bool), n_tm, struct_src (pdb:<id> | afdb_complex:<model>
  | afdb_monomer:<uniprot>), uniprot (monomers)
Rules:
  - complexes: kept as one species if a local PDB assembly covers >= 50% of subunit mass; otherwise their copies are
    returned to the subunit pools (packed as monomers). 60S + 40S are merged into 80S (PDB 6QZP) at min(60S, 40S);
    the remainder stays as free subunits. NPC copies are set by geometry (pores) and use PDB 7R5K.
  - AFDB dimers only when the model file is local (fetch_status ok/cached); otherwise copies return to the pools.
  - comp_weights of an assembly = mass-weighted mean of its subunits' weights; transmembrane if any subunit is TM.
  - free monomers = inventory copies - copies consumed by all assemblies (clipped at 0).
  - signal_frac / mito_om_frac: subunit-mass fraction with a UniProt signal peptide / 'Mitochondrion outer membrane'
    (used by the region mapping: lumenal only with a signal peptide; mito TM on OM vs IM).
"""
import json, numpy as np, pandas as pd

D = (__import__("os").environ.get("VCELL_ROOT", ".") + "/data/derived")
S = (__import__("os").environ.get("VCELL_ROOT", ".") + "/data/structures")
RIBO80 = "6qzp"
NPC_PDB = "7r5k"


def parse_w(s):
    if not isinstance(s, str) or not s:
        return {"cytosol": 1.0}
    out = {}
    for t in s.split(";"):
        k, v = t.split(":"); out[k] = float(v)
    return out


def main():
    inv = pd.read_csv(f"{D}/inventory_v0_complexed.tsv", sep="\t", index_col="gene")
    inv["w"] = inv.compartments.map(parse_w)
    up = pd.read_csv((__import__("os").environ.get("VCELL_ROOT", ".") + "/data/reference/uniprot_human_full.tsv.gz"), sep="\t",
                     usecols=["Entry", "Signal peptide", "Subcellular location [CC]"]).set_index("Entry")
    sig = inv.uniprot.map(lambda a: isinstance(up["Signal peptide"].get(a), str)).fillna(False).astype(bool)
    om = inv.uniprot.map(lambda a: "Mitochondrion outer membrane" in str(up["Subcellular location [CC]"].get(a, ""))).fillna(False).astype(bool)
    tm = inv.transmembrane.fillna(False).astype(bool)
    mass = inv.mass_da.fillna(0.0)
    used = pd.Series(0.0, index=inv.index)
    rows = []

    def asm_weights(sub):
        tot = sum(mass[g] * s for g, s in sub.items()) or 1.0
        w = {}
        for g, s in sub.items():
            for k, v in inv.w[g].items():
                w[k] = w.get(k, 0.0) + v * mass[g] * s / tot
        return {k: round(v, 3) for k, v in sorted(w.items(), key=lambda kv: -kv[1]) if v >= 0.01}

    def mfrac(sub, flag):
        tot = sum(mass[g] * s for g, s in sub.items()) or 1.0
        return round(sum(mass[g] * s for g, s in sub.items() if flag[g]) / tot, 3)

    def add(ing_id, kind, sub, copies, src):
        if copies <= 0:
            return
        for g, s in sub.items():
            used[g] += copies * s
        rows.append(dict(ing_id=ing_id, kind=kind, genes=";".join(sub), stoich=";".join(str(s) for s in sub.values()),
                         copies=int(copies), mass_da=float(sum(mass[g] * s for g, s in sub.items())),
                         comp_weights=json.dumps(asm_weights(sub)), transmembrane=bool(any(tm[g] for g in sub)),
                         n_tm=int(sum(tm[g] for g in sub)), struct_src=src, uniprot="",
                         signal_frac=mfrac(sub, sig), mito_om_frac=mfrac(sub, om)))

    # ---- complexes
    C = pd.read_csv(f"{D}/complexes_v0.tsv", sep="\t")
    C = C[C.copies_built > 0].set_index("cpx")
    fam = {"H2A": r"^(H2AC\d+|H2AZ[12]|H2AX|H2AJ|MACROH2A[12])$", "H2B": r"^H2BC\d+$", "H3": r"^(H3C\d+|H3-3[AB]|H3-5)$", "H4": r"^H4C\d+$"}
    for cpx, r in C.iterrows():
        if cpx == "CURATED-NUCLEOSOME":
            # generic octamer: consume each family proportionally to its gene pools
            n = int(r.copies_built)
            for f, pat in fam.items():
                gs = inv.index[inv.index.str.match(pat)]
                tot = inv.copies[gs].sum()
                for g in gs:
                    used[g] += 2 * n * inv.copies[g] / tot
            rows.append(dict(ing_id="NUCLEOSOME", kind="complex", genes="H2A;H2B;H3;H4", stoich="2;2;2;2", copies=n,
                             mass_da=108_500.0, comp_weights=json.dumps({"nucleus": 1.0}), transmembrane=False, n_tm=0,
                             struct_src="pdb:1kx5", uniprot="", signal_frac=0.0, mito_om_frac=0.0))
            continue
        if cpx in ("CPX-5183", "CPX-5223"):
            continue                                            # handled as 80S below
        sub = {x.split(":")[0]: int(x.split(":")[1]) for x in r.subunits.split(";")}
        sub = {g: s for g, s in sub.items() if g in inv.index and inv.copies[g] > 0}
        if cpx == "CPX-873":
            add("NPC", "complex", sub, int(r.copies_built), f"pdb:{NPC_PDB}")
            continue
        if isinstance(r.best_pdb, str) and r.pdb_mass_cov >= 0.5:
            add(cpx, "complex", sub, int(r.copies_built), f"pdb:{r.best_pdb}")
    # 80S = min(60S, 40S); leftovers as free subunits
    s60 = {x.split(":")[0]: int(x.split(":")[1]) for x in C.loc["CPX-5183", "subunits"].split(";")}
    s40 = {x.split(":")[0]: int(x.split(":")[1]) for x in C.loc["CPX-5223", "subunits"].split(";")}
    n60, n40 = int(C.loc["CPX-5183", "copies_built"]), int(C.loc["CPX-5223", "copies_built"])
    n80 = min(n60, n40)
    sub80 = {g: s for g, s in {**s60, **s40}.items() if g in inv.index}
    add("RIBO80S", "complex", sub80, n80, f"pdb:{RIBO80}")
    add("CPX-5183", "complex", {g: s for g, s in s60.items() if g in inv.index}, n60 - n80, f"pdb:{C.loc['CPX-5183', 'best_pdb']}")
    add("CPX-5223", "complex", {g: s for g, s in s40.items() if g in inv.index}, n40 - n80, f"pdb:{C.loc['CPX-5223', 'best_pdb']}")

    # ---- PDB homo-oligomers
    H = pd.read_csv(f"{D}/homo_oligomers_v0.tsv", sep="\t")
    for r in H.itertuples():
        add(f"{r.gene}_x{r.state}", "oligomer", {r.gene: int(r.state)}, int(r.n_oligomers), f"pdb:{r.example_pdb}")

    # ---- AFDB dimers (local models only)
    st = pd.read_csv(f"{S}/afdb_complexes/fetch_status.tsv", sep="\t").set_index("model").status
    okm = set(st.index[st.isin(["ok", "cached"])])
    for r in pd.read_csv(f"{D}/afdb_homodimers_v0.tsv", sep="\t").itertuples():
        if r.model in okm:
            add(f"{r.gene}_afx2", "afdb_homodimer", {r.gene: 2}, int(r.n_dimers), f"afdb_complex:{r.model}")
    for r in pd.read_csv(f"{D}/afdb_heterodimers_v0.tsv", sep="\t").itertuples():
        if r.model in okm and r.gene1 != r.gene2:
            add(f"{r.gene1}-{r.gene2}", "afdb_heterodimer", {r.gene1: 1, r.gene2: 1}, int(r.n), f"afdb_complex:{r.model}")

    # ---- free monomers
    free = (inv.copies - used).clip(lower=0).round().astype(np.int64)
    over = (used - inv.copies)[used > inv.copies * 1.02 + 10]
    for g in inv.index[free > 0]:
        rows.append(dict(ing_id=g, kind="protein", genes=g, stoich="1", copies=int(free[g]), mass_da=float(mass[g]),
                         comp_weights=json.dumps(inv.w[g]), transmembrane=bool(tm[g]), n_tm=int(tm[g]),
                         struct_src=f"afdb_monomer:{inv.uniprot[g]}", uniprot=inv.uniprot[g],
                         signal_frac=float(sig[g]), mito_om_frac=float(om[g])))
    I = pd.DataFrame(rows)
    # mature 80S are cytoplasmic (r-proteins carry nucleolar annotations from biogenesis); free subunits keep a small
    # nuclear (pre-ribosome) pool
    RIBO_W = {"RIBO80S": {"cytosol": 1.0}, "CPX-5183": {"cytosol": 0.9, "nucleus": 0.1}, "CPX-5223": {"cytosol": 0.9, "nucleus": 0.1}}
    for k, w in RIBO_W.items():
        I.loc[I.ing_id == k, "comp_weights"] = json.dumps(w)
    I["mass_pg"] = I.copies * I.mass_da / 6.022e23 * 1e12
    I.to_csv(f"{D}/ingredients_v0.tsv", sep="\t", index=False)
    tot = I.mass_pg.sum()
    print(f"ingredients: {len(I)} species, {I.copies.sum():.3g} copies, {tot:.1f} pg (inventory {(inv.copies * mass).sum() / 6.022e23 * 1e12:.1f} pg)")
    print(I.groupby("kind").agg(species=("ing_id", "size"), copies=("copies", "sum"), mass_pg=("mass_pg", "sum")).round(2).to_string())
    print(f"transmembrane species {I.transmembrane.sum()} ({I.mass_pg[I.transmembrane].sum():.1f} pg)")
    print(f"genes overdrawn by assemblies (>2%): {len(over)}; largest: {over.sort_values().tail(5).round(0).to_dict()}")
    print(I.sort_values("mass_pg", ascending=False).head(12)[["ing_id", "kind", "copies", "mass_pg", "struct_src"]].to_string(index=False))


if __name__ == "__main__":
    main()
