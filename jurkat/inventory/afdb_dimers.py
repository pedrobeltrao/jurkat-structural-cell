# AFDB (NVIDIA) dimers for proteins still free after Complex Portal complexes + PDB homo-oligomers.
#  homodimers: AFDB human homodimer with ipSAE (max of AB/BA) > 0.6; share of the free pool = SEC fraction at >= 2x
#              monomer (a2) if in SEC, else 0.3 (AF over-predicts homodimers; conservative without data)
#  heterodimers: AFDB pair passes_quality_threshold AND (SEC co-elution r >= 0.6 OR STRING combined >= 700); both partners
#              expressed and with free copies; binds 50% of the limiting free pool (yeast rule: supported pairs bind all,
#              others 50% -- here all are only computationally supported, so 50%). Greedy by interface confidence.
# Output: data/derived/afdb_homodimers_v0.tsv, afdb_heterodimers_v0.tsv; prints the complexed fraction.
import numpy as np, pandas as pd, gzip
inv = pd.read_csv("data/derived/inventory_v0_complexed.tsv", sep="\t", index_col="gene")
ho = pd.read_csv("data/derived/homo_oligomers_v0.tsv", sep="\t").set_index("gene")
sec = pd.read_csv("data/derived/sec_assembly_thresholds.tsv", sep="\t").set_index("uniprot")
MDA = inv.mass_da.fillna(0)
used = inv.copies_in_complexes.fillna(0) + ho.copies_used.reindex(inv.index).fillna(0)
free = (inv.copies - used).clip(lower=0)
acc2g = dict(zip(inv.uniprot, inv.index))
pg = lambda n, g: n * MDA[g] / 6.022e23 * 1e12

# ---- homodimers
H = pd.read_csv("data/structures/afdb_complexes/homodimer_human.tsv.gz", sep="\t")
H["ipsae"] = H[["ipSAE_AB", "ipSAE_BA"]].max(axis=1)
H = H[H.ipsae > 0.6].sort_values("ipsae", ascending=False).drop_duplicates("uniprotAccession")
rows = []
for r in H.itertuples():
    g = acc2g.get(r.uniprotAccession)
    if g is None or g in ho.index or free[g] <= 0: continue
    share = float(sec.a2[r.uniprotAccession]) if r.uniprotAccession in sec.index else 0.3
    n = int(free[g] * share // 2)
    if n <= 0: continue
    free[g] -= 2 * n
    rows.append(dict(gene=g, uniprot=r.uniprotAccession, model=r.modelEntityId, ipsae=round(r.ipsae, 3), share=round(share, 2),
                     n_dimers=n, mass_pg=pg(2 * n, g)))
HD = pd.DataFrame(rows).sort_values("mass_pg", ascending=False); HD.to_csv("data/derived/afdb_homodimers_v0.tsv", sep="\t", index=False)

# ---- heterodimers
X = pd.read_csv("data/structures/afdb_complexes/heterodimer_human.tsv.gz", sep="\t")
X = X[X.passes_quality_threshold.astype(str).str.lower() == "true"]
X = X[X.uniprot_ac_1.isin(acc2g) & X.uniprot_ac_2.isin(acc2g)]
# SEC co-elution correlation
D = "data/proteomics/fossati2021_jurkat_SEC_PXD018033/"
import importlib.util
prof = None
try:
    exec(open("jurkat/inventory/sec_assembly.py").read().split("det = set(prof.index)")[0])
except Exception as e:
    print("SEC profiles unavailable:", e)
def sec_r(a, b):
    if prof is None or a not in prof.index or b not in prof.index: return np.nan
    return float(np.corrcoef(prof.loc[a].values, prof.loc[b].values)[0, 1])
# STRING combined score via UniProt STRING xrefs
up = pd.read_csv("data/reference/uniprot_human_full.tsv.gz", sep="\t", usecols=["Entry", "STRING"]).dropna()
up["ensp"] = up.STRING.str.split(";").str[0]
acc2ensp = dict(zip(up.Entry, up.ensp))
need = {acc2ensp.get(a) for a in pd.concat([X.uniprot_ac_1, X.uniprot_ac_2]) if a in acc2ensp}
st = {}
with gzip.open("data/interactions/string/9606.protein.links.v12.0.txt.gz", "rt") as f:
    next(f)
    for line in f:
        p1, p2, sc = line.split()
        if p1 in need and p2 in need: st[(p1, p2)] = int(sc)
X["sec_r"] = [sec_r(a, b) for a, b in zip(X.uniprot_ac_1, X.uniprot_ac_2)]
X["string"] = [st.get((acc2ensp.get(a), acc2ensp.get(b)), 0) for a, b in zip(X.uniprot_ac_1, X.uniprot_ac_2)]
X = X[(X.sec_r >= 0.6) | (X.string >= 700)].sort_values("max_ipSAE", ascending=False)
# yeast rule: skip pairs whose partners are both subunits of one already-built complex (leftover subunits, e.g. RPL4-RPL18)
C = pd.read_csv("data/derived/complexes_v0.tsv", sep="\t"); C = C[C.copies_built > 0]
same = set()
for sub in C.subunits.dropna():
    gs = [x.split(":")[0] for x in sub.split(";")]
    same |= {(a, b) for a in gs for b in gs if a != b}
X = X[[ (acc2g[a], acc2g[b]) not in same for a, b in zip(X.uniprot_ac_1, X.uniprot_ac_2)]]
rows = []
for r in X.itertuples():
    g1, g2 = acc2g[r.uniprot_ac_1], acc2g[r.uniprot_ac_2]
    n = int(0.5 * min(free[g1], free[g2]))
    if n <= 0: continue
    free[g1] -= n; free[g2] -= n
    rows.append(dict(gene1=g1, gene2=g2, model=r.modelEntityId, ipsae=round(r.max_ipSAE, 3), sec_r=round(r.sec_r, 2) if r.sec_r == r.sec_r else None,
                     string=r.string, n=n, mass_pg=pg(n, g1) + pg(n, g2)))
XD = pd.DataFrame(rows).sort_values("mass_pg", ascending=False); XD.to_csv("data/derived/afdb_heterodimers_v0.tsv", sep="\t", index=False)

tot = (inv.copies * MDA).sum() / 6.022e23 * 1e12
cp = (inv.copies_in_complexes.fillna(0) * MDA).sum() / 6.022e23 * 1e12
print(f"AFDB homodimers: {len(HD)} proteins, {HD.mass_pg.sum():.2f} pg | heterodimers: {len(XD)} pairs (of {len(X)} supported), {XD.mass_pg.sum():.2f} pg")
print(f"complexed fraction: CP {cp/tot:.1%} + PDB homo {ho.mass_pg.sum()/tot:.1%} + AFDB homo {HD.mass_pg.sum()/tot:.1%} + AFDB hetero {XD.mass_pg.sum()/tot:.1%} "
      f"= {(cp + ho.mass_pg.sum() + HD.mass_pg.sum() + XD.mass_pg.sum())/tot:.1%} (SEC range 48-66%)")
print(HD.head(8).to_string(index=False)); print(XD.head(10).to_string(index=False))
