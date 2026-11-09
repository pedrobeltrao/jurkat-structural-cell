# Inventory v0 for target cell 1 (PLANNING_NOTES §4-results, §4b): per-protein copy numbers and compartments.
#   abundance  = median Jurkat Geiger 2012 iBAQ ppb (E-PROT-20 linear reprocessing, gene-mapped) over detected replicates
#   imputation = expressed (E6-1 RNA call) but undetected genes: log-log fit ppb ~ TPM on detected genes, capped at the
#                5th percentile of detected ppb (undetected proteins sit below the detection limit)
#   zeros      = RNA 'off' and undetected; manual overrides (verified losses, homozygous deletions)
#   scale      = sum(copies * mass) = TOTAL_PG (80 pg, DECISION 2026-11-08)
#   location   = UniProt 'Subcellular location [CC]' mapped to model compartments, copies split equally (v0)
# Output: data/derived/inventory_v0.tsv and a printed report.
import re, sys, numpy as np, pandas as pd

TOTAL_PG = float(sys.argv[1]) if len(sys.argv) > 1 else 80.0
AVOGADRO = 6.02214076e23
G = "data/proteomics/geiger2012_PXD002395/proteinGroups_Geiger_MCP_2012_ppb-MappedToGeneID.txt"

# manual zero overrides: literature-verified losses and DepMap homozygous deletions (§4b)
ZERO = {"PTEN": "absent protein (Shan 2000)", "INPP5D": "absent protein, biallelic mutation (Freeburn 2002)",
        "CDKN2A": "9p21 homozygous deletion", "CDKN2B": "9p21 homozygous deletion", "MTAP": "9p21 homozygous deletion",
        "C1GALT1C1": "COSMC loss (DepMap)"}
ZERO_PATTERNS = [r"^IFNA\d+$", r"^IFNB1$", r"^IFNE$", r"^IFNW1$"]          # 9p21 interferon cluster

# ---- UniProt reference (reviewed human)
up = pd.read_csv("data/reference/uniprot_human_full.tsv.gz", sep="\t")
up = up[up.Reviewed == "reviewed"].rename(columns={"Gene Names (primary)": "gene"})
up = up.dropna(subset=["gene"]).drop_duplicates("gene")
up["gene"] = up.gene.str.split("; ").str[0]
up = up.drop_duplicates("gene").set_index("gene")

# ---- proteomics: MaxQuant protein groups (linear ppb). The gene-mapped table drops shared groups (histones,
# PTPRC, TUBB, ...), so map Majority IDs ourselves: reviewed accessions -> genes; split a group's ppb equally among
# its genes that share the first gene's 3-letter prefix (histone/tubulin families), ignore other members.
acc2gene = dict(zip(up.Entry, up.index))
g = pd.read_csv(G.replace("-MappedToGeneID", ""), sep="\t")
cols = [c for c in g.columns if "Jurkat" in c]
rows = []
for ids, vals in zip(g["Majority.protein.IDs"], g[cols].apply(pd.to_numeric, errors="coerce").values):
    if np.all(np.isnan(vals)) or np.nansum(vals) <= 0: continue
    gs = list(dict.fromkeys(acc2gene[a.split("-")[0]] for a in str(ids).split(";") if a.split("-")[0] in acc2gene))
    if not gs: continue
    fam = [x for x in gs if x[:3] == gs[0][:3]]
    for x in fam: rows.append((x, *(vals / len(fam))))
P = pd.DataFrame(rows, columns=["gene"] + cols).groupby("gene").sum(min_count=1)
ppb = P.median(axis=1, skipna=True)                     # median over detected replicates
nrep = P.notna().sum(axis=1)
ppb = ppb[ppb > 0]
print("protein groups mapped:", len(rows), "genes:", len(ppb), "| per-replicate ppb sums:", (P.sum() / 1e9).round(3).tolist())

# ---- RNA (E6-1 consensus; jurkat/inventory/rna_e61.py)
rna = pd.read_csv("data/derived/rna_e61.tsv", sep="\t", index_col="gene")

genes = sorted(set(up.index) & (set(ppb.index) | set(rna.index)))
inv = pd.DataFrame(index=pd.Index(genes, name="gene"))
inv["uniprot"] = up.Entry.reindex(inv.index)
inv["mass_da"] = up.Mass.reindex(inv.index).astype(float)
inv["ppb_geiger"] = ppb.reindex(inv.index)
inv["n_rep"] = nrep.reindex(inv.index).fillna(0).astype(int)
inv["tpm"] = rna.tpm_median.reindex(inv.index)
inv["rna_call"] = rna.call.reindex(inv.index).fillna("no_rna")

# imputation fit on detected genes with TPM > 0
d = inv.dropna(subset=["ppb_geiger"]).query("tpm > 0")
b, a = np.polyfit(np.log10(d.tpm), np.log10(d.ppb_geiger), 1)
r = np.corrcoef(np.log10(d.tpm), np.log10(d.ppb_geiger))[0, 1]
floor = np.percentile(inv.ppb_geiger.dropna(), 5)
fit = 10 ** (a + b * np.log10(inv.tpm.clip(lower=1e-3)))

inv["source"] = "geiger"
inv["ppb"] = inv.ppb_geiger
und = inv.ppb.isna()
inv.loc[und & (inv.rna_call == "expressed"), "source"] = "imputed_rna"
inv.loc[und & (inv.rna_call == "low"), "source"] = "imputed_rna_low"
inv.loc[und & inv.rna_call.isin(["off", "no_rna"]), "source"] = "zero_off"
m = inv.source == "imputed_rna"; inv.loc[m, "ppb"] = np.minimum(fit[m], floor)
m = inv.source == "imputed_rna_low"; inv.loc[m, "ppb"] = np.minimum(fit[m], floor) * 0.25
inv["ppb"] = inv.ppb.fillna(0.0)
zero_hit = inv.index.isin(list(ZERO)) | inv.index.to_series().str.match("|".join(ZERO_PATTERNS)).values
inv.loc[zero_hit, ["ppb", "source"]] = [0.0, "zero_override"]
inv["note"] = inv.index.map(ZERO).fillna("")
inv.loc[(inv.source == "geiger") & (inv.rna_call == "off"), "note"] = "detected despite RNA off"

# ---- absolute scale: molar fractions -> copies with total mass TOTAL_PG
frac = inv.ppb / inv.ppb.sum()
mean_mass = (frac * inv.mass_da.fillna(inv.mass_da.median())).sum()
N = TOTAL_PG * 1e-12 / (mean_mass / AVOGADRO)
inv["copies"] = (frac * N).round().astype(np.int64)

# ---- compartments: HPA subcellular (main locations weight 1, additional 0.3); UniProt text as fallback
HPA_MAP = {"nucleus": ["Nucleoplasm", "Nuclear bodies", "Nuclear speckles", "Nucleoli", "Nucleoli fibrillar center",
                       "Nucleoli rim", "Mitotic chromosome", "Kinetochore", "Perinuclear theca"],
           "nuclear_envelope": ["Nuclear membrane"],
           "cytosol": ["Cytosol", "Cytoplasmic bodies", "Rods & Rings", "Aggresome"],
           "cytoskeleton": ["Actin filaments", "Microtubules", "Microtubule ends", "Intermediate filaments", "Centrosome",
                            "Centriolar satellite", "Basal body", "Mitotic spindle", "Midbody", "Midbody ring",
                            "Cytokinetic bridge", "Cleavage furrow"],
           "mitochondrion": ["Mitochondria"], "er": ["Endoplasmic reticulum"], "golgi": ["Golgi apparatus"],
           "plasma_membrane": ["Plasma membrane", "Cell Junctions", "Focal adhesion sites"],
           "endolysosomal": ["Vesicles", "Endosomes", "Lysosomes"], "peroxisome": ["Peroxisomes"],
           "lipid_droplet": ["Lipid droplets"]}
term2comp = {t: c for c, ts in HPA_MAP.items() for t in ts}
hpa = pd.read_csv("data/reference/hpa_subcellular/subcellular_location.tsv", sep="\t").drop_duplicates("Gene name").set_index("Gene name")
def hpa_w(gname):
    if gname not in hpa.index: return None
    w = {}
    for col, wt in (("Main location", 1.0), ("Additional location", 0.3)):
        for t in str(hpa.at[gname, col]).split(";"):
            c = term2comp.get(t.strip())
            if c: w[c] = max(w.get(c, 0), wt)
    return w or None
COMP = [("nucleus", r"nucle(us|olus|oplasm)|chromosome|nuclear speck|PML body|Cajal"),
        ("nuclear_envelope", r"nuclear envelope|nuclear membrane|nuclear pore"),
        ("mitochondrion", r"mitochondri"), ("er", r"endoplasmic reticulum|microsome"), ("golgi", r"golgi"),
        ("plasma_membrane", r"cell membrane|plasma membrane|cell surface|cell projection|cell junction"),
        ("endolysosomal", r"endosome|lysosome|vesicle|vacuole|autophagosome|phagosome"),
        ("peroxisome", r"peroxisom"), ("lipid_droplet", r"lipid droplet"),
        ("cytoskeleton", r"cytoskeleton|centrosome|centriole|microtubule|spindle"),
        ("secreted", r"\bsecreted\b|extracellular"), ("cytosol", r"cytoplasm|cytosol")]
loc_txt = up["Subcellular location [CC]"].reindex(inv.index).fillna("")
loc_txt = loc_txt.str.replace(r"\{[^}]*\}", "", regex=True).str.replace("SUBCELLULAR LOCATION:", "", regex=False).str.split("Note=").str[0]
def up_w(t):
    hit = [c for c, pat in COMP if re.search(pat, t.lower())]
    if hit and set(hit) - {"secreted"}: hit = [c for c in hit if c != "secreted"]   # intracellular pools only
    return {c: 1.0 for c in hit} or {"unassigned": 1.0}
tm = up.Transmembrane.reindex(inv.index).notna()
MEMB = {"nuclear_envelope", "mitochondrion", "er", "golgi", "plasma_membrane", "endolysosomal", "peroxisome", "lipid_droplet"}
W, SRC = {}, {}
for gname in inv.index:
    uw = up_w(loc_txt[gname]); hw = hpa_w(gname)
    if hw is not None and hpa.at[gname, "Reliability"] == "Uncertain": hw = None
    if tm[gname]:                                    # membrane proteins: no soluble pools; distrust HPA if no membrane hit
        uw = {k: v for k, v in uw.items() if k in MEMB} or uw
        if hw is not None: hw = {k: v for k, v in hw.items() if k in MEMB} or None
    norm = lambda w: {k: v / sum(w.values()) for k, v in w.items()}
    if hw is None: w, SRC[gname] = norm(uw), "uniprot"
    elif "unassigned" in uw: w, SRC[gname] = norm(hw), "hpa"
    else:                                            # blend: half HPA (cell-line imaging), half UniProt (curated)
        hw, uw = norm(hw), norm(uw); w = {k: 0.5 * hw.get(k, 0) + 0.5 * uw.get(k, 0) for k in set(hw) | set(uw)}
        SRC[gname] = "hpa+uniprot"
    W[gname] = {k: round(v, 3) for k, v in w.items() if v >= 0.01}
# ---- nuclear share of soluble proteins from fractionation proteomics (Itzhak 2016 HeLa: 1K nuclear pellet vs cytosol).
# Annotations (HPA/UniProt) only say *where* a protein is seen; equal splitting over-filled the nucleus (glycolytic enzymes,
# 80S, actin). Only the nucleo-cytoplasmic part S = w_nucleus + w_cytosol + w_cytoskeleton is re-split; organelle weights
# are kept (the 1K pellet also collects mitochondria/ER, so the data cannot judge organelle pools). Used for proteins the
# data classify as N, C or N/C with >= 3 good profiles: fn = nuc / (nuc + cyt) minus a 0.05 baseline (cytosolic markers
# UBA1/PKM/ENO1 sit at 0.03-0.06 in the 1K pellet). No nuclear annotation -> a nuclear pool only if fn >= 0.5. TM untouched.
ITZ = pd.read_csv("data/literature/localisation/itzhak2016_partition.tsv", sep="\t")
ITZ = ITZ[ITZ.primary_for_gene & (ITZ.n_good_mnc_profiles >= 3) & ITZ.mnc_class.isin(["N", "C", "N/C"])].drop_duplicates("gene").set_index("gene")
NUC_BASE = 0.05; NC = ("nucleus", "cytosol", "cytoskeleton"); n_loc = 0
for gname in inv.index:
    if tm[gname] or gname not in ITZ.index: continue
    nc, cy = ITZ.at[gname, "frac_nuclear"], ITZ.at[gname, "frac_cytosol"]
    fn = max(nc / (nc + cy) - NUC_BASE, 0.0) / (1 - NUC_BASE)
    w = dict(W[gname]); a_ = w.get("nucleus", 0.0); S = sum(w.get(k, 0.0) for k in NC)
    if S == 0 or (a_ == 0 and fn < 0.5): continue
    cyt = {k: w[k] for k in ("cytosol", "cytoskeleton") if w.get(k, 0) > 0} or {"cytosol": 1.0}; cs = sum(cyt.values())
    for k in NC: w.pop(k, None)
    for k, v in cyt.items(): w[k] = S * (1 - fn) * v / cs
    w["nucleus"] = S * fn
    W[gname] = {k: round(v, 3) for k, v in w.items() if v >= 0.01}; SRC[gname] += "+itzhak"; n_loc += 1
print(f"nuclear share from Itzhak 2016 fractionation: {n_loc} soluble proteins")
inv["compartments"] = [";".join(f"{k}:{v}" for k, v in sorted(W[x].items(), key=lambda kv: -kv[1])) for x in inv.index]
inv["loc_source"] = pd.Series(SRC)
inv["transmembrane"] = up.Transmembrane.reindex(inv.index).notna().values
inv["mass_pg"] = inv.copies * inv.mass_da / AVOGADRO * 1e12
inv.to_csv("data/derived/inventory_v0.tsv", sep="\t")

# ---- report
print(f"genes {len(inv)} | Geiger detected {(inv.source=='geiger').sum()} | imputed {inv.source.str.startswith('imputed').sum()} "
      f"| zero (off) {(inv.source=='zero_off').sum()} | zero (override) {(inv.source=='zero_override').sum()}")
print(f"impute fit: log10 ppb = {a:.2f} + {b:.2f} log10 TPM (r = {r:.2f}); floor {floor:.0f} ppb")
print(f"total {TOTAL_PG} pg -> {N:.3g} molecules; mean mass {mean_mass/1e3:.1f} kDa")
print("mass share by source:", (inv.groupby("source").mass_pg.sum() / inv.mass_pg.sum()).round(4).to_dict())
cm = pd.Series({c: sum(inv.mass_pg[x] * W[x].get(c, 0) for x in inv.index) for c in set().union(*W.values())}).sort_values(ascending=False)
print("mass by compartment (pg, HPA-weighted):", cm.round(2).to_dict())
rp = inv[inv.index.str.match(r"^RP[SL]\d+[AXY]?\d?$|^RPLP[012]$|^RPSA$") & ~inv.index.str.contains("L$")]
print(f"cytosolic r-proteins: {len(rp)} genes, median copies {rp.copies.median():.3g} (EM target ~1.5e6)")
hist = inv[inv.index.str.match(r"^H(2A|2B|3|4)C?\d|^H4C|^H3-|^H2AZ|^H1-")]
print(f"histones: {len(hist)} genes, {hist.copies.sum():.3g} copies, {hist.mass_pg.sum():.1f} pg")
for gname in ["ACTB", "TUBB", "TUBA1B", "GAPDH", "PTPRC", "CD3E", "LCK", "ZAP70", "LAT", "CD28", "NUP93", "TFRC", "PTEN", "INPP5D"]:
    if gname in inv.index:
        x = inv.loc[gname]; print(f"  {gname:7s} {x.copies:>12,d}  {x.source:15s} {x.compartments}")
print("top 15 by copies:"); print(inv.sort_values("copies", ascending=False).head(15)[["copies", "mass_da", "source", "compartments"]].to_string())
