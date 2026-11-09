# Complexes v0 (yeast rule, external/yeast_cell_viewer source/pipeline/extend_complexes.py + MODEL.md):
#   copies = median over subunits of pool/stoichiometry, capped at 2x the limiting subunit (2x the 25th percentile
#   subunit for complexes with >= 4 subunit types). Complex Portal human (expanded participant list, so sub-complexes
#   are resolved). Structure candidates: local PDB entries (SIFTS chain->UniProt) covering the most subunit mass.
# Pools are consumed greedily by complex mass (largest first) to report how much of each protein is in complexes;
# overlapping complexes compete for shared subunits. Output: data/derived/complexes_v0.tsv, inventory_v0_complexed.tsv.
import re, os, numpy as np, pandas as pd
from collections import defaultdict

inv = pd.read_csv("data/derived/inventory_v0.tsv", sep="\t", index_col="gene")
acc2gene = dict(zip(inv.uniprot, inv.index)); pool = inv.copies.to_dict(); mass = inv.mass_da.fillna(0).to_dict()
cp = pd.read_csv("data/complexes/complex_portal/9606.tsv", sep="\t")
sifts = pd.read_csv("data/structures/pdb/sifts/pdb_chain_uniprot.tsv.gz", sep="\t", comment="#", usecols=["PDB", "CHAIN", "SP_PRIMARY"])
local = {f.split("-")[0] for f in os.listdir("data/structures/pdb/assemblies")}
sifts = sifts[sifts.PDB.isin(local)]
pdb_up = sifts.groupby("PDB").SP_PRIMARY.agg(lambda s: set(s)).to_dict()
up_pdb = defaultdict(set)
for p, ups in pdb_up.items():
    for u in ups: up_pdb[u].add(p)

rows = []
for r in cp.itertuples(index=False):
    parts = re.findall(r"([A-Z][A-Z0-9]{5,9})(?:-\w+)?\((\d+)\)", str(r[18]))
    sub, unmapped = {}, 0
    for a, s in parts:
        g = acc2gene.get(a)
        if g is None: unmapped += 1; continue
        sub[g] = sub.get(g, 0) + max(1, int(s))          # stoichiometry 0 = unknown -> 1
    if len(sub) < 1 or (len(sub) == 1 and list(sub.values())[0] < 2): continue
    ratios = np.array([pool[g] / s for g, s in sub.items()])
    med = float(np.median(ratios))
    lim = float(np.percentile(ratios, 25)) if len(sub) >= 4 else float(ratios.min())
    copies = min(med, 2 * lim)
    m = sum(mass[g] * s for g, s in sub.items())
    accs = {inv.uniprot[g] for g in sub}
    cand = set().union(*[up_pdb.get(a, set()) for a in accs]) if accs else set()
    best, bcov = None, 0.0
    for p in cand:
        cov = sum(mass[g] * s for g, s in sub.items() if inv.uniprot[g] in pdb_up[p]) / max(m, 1)
        if cov > bcov + 1e-9 or (abs(cov - bcov) < 1e-9 and best and len(pdb_up[p]) < len(pdb_up[best])):
            best, bcov = p, cov
    rows.append(dict(cpx=r[0], name=r[1], n_types=len(sub), unmapped=unmapped,
                     subunits=";".join(f"{g}:{s}" for g, s in sub.items()), copies_rule=round(copies),
                     limiting=min(sub, key=lambda g: pool[g] / sub[g]), mass_kda=round(m / 1e3, 1),
                     n_zero_subunits=int(sum(pool[g] == 0 for g in sub)), best_pdb=best, pdb_mass_cov=round(bcov, 2),
                     cp_pdb=";".join(re.findall(r"wwpdb:(\w{4})", str(r[8])))))
C = pd.DataFrame(rows)
C["mass_pg"] = C.copies_rule * C.mass_kda * 1e3 / 6.022e23 * 1e12

# greedy pool consumption: curated core machines first, then by complex mass. The yeast rule is re-evaluated on the
# REMAINING pools. Subunits with zero copies are tolerated (ignored) if they are < 10% of complex mass (minor paralogs,
# undetected small subunits); otherwise the complex is not built.
PRIORITY = ["CPX-5183", "CPX-5223", "CPX-5993", "CPX-25754", "CPX-873", "CPX-6030", "CPX-6151", "CPX-2940"]
def rule(vals, ntypes):
    vals = np.asarray(vals, float); med = np.median(vals)
    lim = np.percentile(vals, 25) if ntypes >= 4 else vals.min()
    return max(min(med, 2 * lim), 0)
order = sorted(C.index, key=lambda i: (PRIORITY.index(C.cpx[i]) if C.cpx[i] in PRIORITY else len(PRIORITY), -C.mass_pg[i]))
left = dict(pool); built = {}
# curated special case 1, nucleosomes: Complex Portal lists gene-specific histone variants, but histone pools are spread
# over ~50 genes, so build one generic nucleosome species from family totals (2 x H2A, H2B, H3, H4 per octamer) and
# consume all core histones proportionally. Linker histone H1 stays free (v0).
fam = {"H2A": r"^(H2AC\d+|H2AZ[12]|H2AX|H2AJ|MACROH2A[12])$", "H2B": r"^H2BC\d+$", "H3": r"^(H3C\d+|H3-3[AB]|H3-5)$", "H4": r"^H4C\d+$"}
fam_genes = {f: [g for g in pool if re.match(pat, g)] for f, pat in fam.items()}
fam_tot = {f: sum(pool[g] for g in gs) for f, gs in fam_genes.items()}
NUC_IBAQ = int(min(fam_tot.values()) // 2)
# DECISION 2026-11-08: nucleosome count from DNA, not histone iBAQ (which gives only ~45% of a 2n genome at 80 pg).
# Jurkat is near-diploid (DepMap ABSOLUTE median CN 2) and male: G1 2n ~ 6.2 Gbp / 200 bp per nucleosome.
NUC = int(6.2e9 / 200)
for f, gs in fam_genes.items():                      # scale core histone pools up so families supply 2 x NUC
    k = 2 * NUC / fam_tot[f]
    for g in gs: pool[g] = int(round(pool[g] * k)); left[g] = pool[g]
    fam_tot[f] = 2 * NUC
print(f"histone iBAQ supports {NUC_IBAQ:.3g} nucleosomes; DNA-based value used: {NUC:.3g}")
for f, gs in fam_genes.items():
    for g in gs: left[g] = max(left[g] - round(2 * NUC * pool[g] / fam_tot[f]), 0)
print("nucleosomes (family totals):", {f: f"{v:.3g}" for f, v in fam_tot.items()}, "->", f"{NUC:.3g} octamers",
      f"(~{NUC*200/1e9:.2f} Gbp at 200 bp/nucleosome; 2n genome 6.4 Gbp)")
nuc_rows = C.name.str.contains(r"^Nucleosome|CENP-A nucleosome", regex=True)
# curated special case 2, NPC count from geometry: NE area (~258 um2, HMFF pilot baseline) x ~4 NPC/um2
NPC_COPIES = 1000
for i in order:
    r = C.loc[i]; sub = {x.split(":")[0]: int(x.split(":")[1]) for x in r.subunits.split(";")}
    zero_mass = sum(mass[g] * s for g, s in sub.items() if pool[g] == 0) / max(sum(mass[g] * s for g, s in sub.items()), 1)
    present = {g: s for g, s in sub.items() if pool[g] > 0}
    if zero_mass >= 0.10 or not present or (len(sub) == 1 and list(sub.values())[0] < 2): built[i] = 0; continue
    if nuc_rows[i]: built[i] = 0; continue
    n = int(rule([left[g] / s for g, s in present.items()], len(present)))
    if r.cpx == "CPX-873": n = min(n, NPC_COPIES)
    for g, s in present.items(): left[g] = max(left[g] - n * s, 0)
    built[i] = n
C["copies_built"] = pd.Series(built)
C = pd.concat([C, pd.DataFrame([dict(cpx="CURATED-NUCLEOSOME", name="Nucleosome core particle (generic, family totals)", n_types=4,
               subunits="H2A:2;H2B:2;H3:2;H4:2", copies_rule=NUC, copies_built=NUC, mass_kda=108.5, n_zero_subunits=0,
               best_pdb="1kx5", pdb_mass_cov=1.0, mass_pg=NUC * 108.5e3 / 6.022e23 * 1e12)])], ignore_index=True)
C["copies_built"] = C.copies_built.astype(int)
C["mass_built_pg"] = C.copies_built * C.mass_kda * 1e3 / 6.022e23 * 1e12
C.sort_values("mass_built_pg", ascending=False).to_csv("data/derived/complexes_v0.tsv", sep="\t", index=False)
inv["copies"] = pd.Series(pool).reindex(inv.index).fillna(inv.copies).astype(np.int64)   # histone pools rescaled
inv["mass_pg"] = inv.copies * inv.mass_da.fillna(0) / 6.022e23 * 1e12
inv["copies_in_complexes"] = inv.copies - pd.Series(left).reindex(inv.index).fillna(inv.copies)
inv["frac_in_complexes"] = (inv.copies_in_complexes / inv.copies.replace(0, np.nan)).round(3)
inv.to_csv("data/derived/inventory_v0_complexed.tsv", sep="\t")

tot = inv.mass_pg.sum(); inc = (inv.copies_in_complexes * inv.mass_da / 6.022e23 * 1e12).sum()
b = C[C.copies_built > 0]
print(f"Complex Portal human: {len(cp)} | parsed {len(C)} | built {len(b)} | with absent subunits {(C.n_zero_subunits>0).sum()}")
print(f"protein mass in complexes: {inc:.1f} pg of {tot:.1f} pg ({inc/tot:.1%})")
print(f"built complexes with a local PDB covering >=80% / >=50% of subunit mass: {(b.pdb_mass_cov>=0.8).sum()} / {(b.pdb_mass_cov>=0.5).sum()}; "
      f"mass share of built-complex mass with >=80% cov: {b.loc[b.pdb_mass_cov>=0.8,'mass_built_pg'].sum()/b.mass_built_pg.sum():.1%}")
print(b.sort_values("mass_built_pg", ascending=False).head(25)[["cpx", "name", "n_types", "copies_built", "mass_kda", "mass_built_pg", "best_pdb", "pdb_mass_cov"]].to_string(index=False))
