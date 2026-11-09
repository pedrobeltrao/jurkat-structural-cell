# Homo-oligomer states for proteins not (fully) in Complex Portal complexes (v1 step, PLANNING_NOTES §4c SEC DECISION).
# Candidates: inventory proteins with frac_in_complexes < 0.5. Source: local PDB assembly1 files of homomeric entries
# (SIFTS: every chain maps to the same UniProt). State = most common protein-chain count over up to 8 entries.
# Share of the free pool that oligomerises = SEC intensity fraction above a state-matched apparent-MW threshold
# (2x monomer for dimers/trimers, 3x for 4-5mers, 4x above); 50% if the protein is not in the SEC data. Output: data/derived/homo_oligomers_v0.tsv.
import os, gzip, collections, numpy as np, pandas as pd, gemmi
from concurrent.futures import ProcessPoolExecutor

A = "data/structures/pdb/assemblies"
inv = pd.read_csv("data/derived/inventory_v0_complexed.tsv", sep="\t", index_col="gene")
sec = pd.read_csv("data/derived/sec_assembly_thresholds.tsv", sep="\t").set_index("uniprot")
sifts = pd.read_csv("data/structures/pdb/sifts/pdb_chain_uniprot.tsv.gz", sep="\t", comment="#", usecols=["PDB", "CHAIN", "SP_PRIMARY", "SP_BEG", "SP_END"])
local = {f.split("-")[0] for f in os.listdir(A)}
sifts = sifts[sifts.PDB.isin(local)]
ent = sifts.groupby("PDB").SP_PRIMARY.agg(lambda s: set(s))
homo = ent[ent.map(len) == 1].map(lambda s: next(iter(s)))
# coverage filter: the entry's chains must cover >= 50% of the UniProt sequence (drops domain fragments)
L = pd.read_csv("data/reference/uniprot_human_reviewed.tsv.gz", sep="\t", usecols=["Entry", "Length"]).set_index("Entry").Length
sifts["span"] = pd.to_numeric(sifts.SP_END, errors="coerce") - pd.to_numeric(sifts.SP_BEG, errors="coerce") + 1
cov = sifts.groupby(["PDB", "CHAIN", "SP_PRIMARY"]).span.sum().groupby(level=[0, 2]).max().reset_index()
cov["frac"] = cov.span / cov.SP_PRIMARY.map(L)
good = set(cov.loc[cov.frac >= 0.5, "PDB"])
homo = homo[homo.index.isin(good)]
by_acc = homo.groupby(homo).apply(lambda s: sorted(s.index)[:8]).to_dict()

def n_chains(pid):
    try:
        st = gemmi.read_structure(f"{A}/{pid}-assembly1.cif.gz"); st.setup_entities()
        return sum(1 for ch in st[0] if ch.get_polymer().check_polymer_type() in (gemmi.PolymerType.PeptideL,) and len(ch.get_polymer()) >= 30)
    except Exception:
        return 0

POLYMER = r"^(ACT[BG]1|ACTA[12]|ACTC1|TUB[AB]\w*|SPT[AB]\w*|LMN[AB]\w*|VIM|KRT\d+\w*|SEPTIN\d+|NEFL|DES)$"   # built procedurally
cand = inv[(inv.copies > 0) & (inv.frac_in_complexes.fillna(0) < 0.5) & inv.uniprot.isin(by_acc.keys()) & ~inv.index.str.match(POLYMER)]
sa = pd.read_csv("data/derived/sec_assembly.tsv", sep="\t").set_index("uniprot")
jobs = {g: by_acc[a] for g, a in zip(cand.index, cand.uniprot)}
allp = sorted({p for v in jobs.values() for p in v})
with ProcessPoolExecutor(16) as ex:
    nc = dict(zip(allp, ex.map(n_chains, allp, chunksize=50)))
rows = []
for g, pids in jobs.items():
    counts = [nc[p] for p in pids if nc[p] >= 1]                        # monomeric entries vote for state 1
    if not counts: continue
    state, support = collections.Counter(counts).most_common(1)[0]
    if state < 2 or support / len(counts) < 0.5: continue               # majority must be one oligomeric state
    if state > 4 and support < 2: continue                              # large states need >= 2 supporting entries
    a = inv.uniprot[g]; a4 = sec.a4.get(a, np.nan)
    if a in sa.index:                                                   # SEC consistency of the apex with the state
        ratio = sa.apex_app_kda[a] / (state * sa.mono_kda[a])
        if not (0.4 <= ratio <= 2.5) and a4 > 0.2: continue
    k = "a2" if state <= 3 else ("a3" if state <= 5 else "a4")          # state-matched SEC threshold (calibration x1.8)
    share = float(sec[k][a]) if a in sec.index else 0.5                 # continuous oligomerised share; 0.5 if no SEC
    free = inv.copies[g] - inv.copies_in_complexes.fillna(0)[g]
    n_olig = int(free * share // state)
    rows.append(dict(gene=g, uniprot=a, state=state, support=f"{support}/{len(pids)}", example_pdb=pids[0], sec_a4=a4,
                     share=share, n_oligomers=n_olig, copies_used=n_olig * state,
                     mass_pg=n_olig * state * inv.mass_da[g] / 6.022e23 * 1e12))
H = pd.DataFrame(rows).sort_values("mass_pg", ascending=False)
H.to_csv("data/derived/homo_oligomers_v0.tsv", sep="\t", index=False)
tot = inv.mass_pg.sum(); in_cpx = (inv.copies_in_complexes.fillna(0) * inv.mass_da / 6.022e23 * 1e12).sum()
print(f"candidates {len(cand)} with a local homomer PDB; assigned a state: {len(H)}; states: {H.state.value_counts().head(6).to_dict()}")
print(f"mass newly oligomerised: {H.mass_pg.sum():.1f} pg -> complexed fraction {in_cpx/tot:.1%} -> {(in_cpx + H.mass_pg.sum())/tot:.1%}")
print(H.head(20)[["gene", "state", "support", "example_pdb", "sec_a4", "n_oligomers", "mass_pg"]].to_string(index=False))
