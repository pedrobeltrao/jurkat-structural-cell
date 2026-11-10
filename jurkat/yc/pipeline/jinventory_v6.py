"""Inventory revision v6 (PLANNING_NOTES "v6 plan: imaging-driven systematic rules" + "Correction to the v6 plan";
log in "v6 implementation"): systematic localisation rules from imaging categories plus mechanism features.

usage: jinventory_v6.py            dry run: rule tables, data/derived/inventory_v6_changes.tsv, inventory_v6_report.txt and a
                                   dry-run localisation score (predicted intended shares vs references, v5 -> v6)
       jinventory_v6.py --apply    as above, then rewrites data/derived/ingredients_v0.tsv (snapshot before the change:
                                   ingredients_v0_pre_v6.tsv; every run starts from that snapshot when it exists: idempotent).

Hold-out (DECISION, PLANNING_NOTES): OpenCell is never read here. The HPA-JURKAT genes are split in a seeded random half
(validation/jurkat_holdout_genes.txt, tier2.py split); for held-out genes the HPA consensus is replaced by a consensus
rebuilt from the per-line annotations of all other cell lines (validation/hpa_celllines.tsv, JURKAT and 'uncertain'
antibodies dropped; a term is 'main' when >= 50% of the (antibody, line) annotations carry it, 'additional' when >= 25%),
because the HPA consensus pools JURKAT. JURKAT images (Tier 2 metrics) are used only for fit-half genes, for calibration.
Antibodies shared by a fit and a held-out gene (e.g. LCP1/PLS1/PLS3) are excluded from calibration.

Rules (applied in this order; every changed field logged with version, gene, ingredient, rule_id, field, old/new value,
evidence and the validation check that motivated the class):
 R4  TM steady-state pools: TM ingredients >= 70% PM whose HPA main location includes ER or Golgi: PM -> 0.5 (HPA main
     also PM) or 0.2 (not), the rest to er/golgi by the HPA main terms.
 R2  actin-binding -> cortex (sh_PM_cortex): HPA (main/add) Actin filaments / Plasma membrane / Cell Junctions / Focal
     adhesion sites AND a mechanism (GO MF actin(-filament) binding, GO CC actin cytoskeleton / actin filament / cortical
     actin cytoskeleton, UniProt keyword Actin-binding, or an actin-binding InterPro domain). cortex = S_CX x cytoplasmic
     pool (S_CX calibrated on the fit half, clipped 0.3-0.7).
 R3  peripheral membrane binders: a membrane-binding InterPro domain (PH, C1, C2, BAR/F-BAR/I-BAR, FERM, PX, FYVE,
     ENTH/ANTH, GRAM, annexin, tubby) or a GO/UniProt peripheral-membrane term, AND an HPA membrane-type location:
     Plasma membrane / Cell Junctions / Focal adhesion sites -> pm_periph_cyto, Vesicles / Endosomes / Lysosomes ->
     endo_periph_cyto, Golgi apparatus -> golgi_periph_cyto (peripheral shells: centre within BR + 20 nm of the membrane) (main 1, additional 0.3). Shell = S_PER x cytoplasmic pool.
 R5  Golgi binders: GO CC Golgi membrane / COPI vesicle coat / COPI-coated vesicle / cis-Golgi network or GO BP Golgi
     organization, AND HPA Golgi apparatus (main/add) or inventory golgi share >= 0.2: golgi_periph_cyto = 0.5 x
     cytoplasmic pool. (R5a, the Golgi area factor on Golgi shells, is in jwc.step_counts.)
 R6a NPC-proximal shell ("npc" key -> sh_NPC): GO CC nuclear pore (incl. basket / cytoplasmic filaments): 0.3 of the
     soluble nucleus + cytoplasmic pool; nuclear import/export receptors (GO MF signal receptor activity): 0.15 (test tiles:
     0.5 / 0.25 placed 66%, the 7R5K cytoplasmic ring blocks the cytoplasmic annulus).
 R1  nucleus/cytosol from imaging: soluble genes whose model share is >= 80% nucleus + cytoplasm and whose HPA terms are
     nuclear or cytoplasmic only (membranes other than PM/junctions excluded): target N/C concentration ratio by category
     N-only 10, N main + C additional 3, N + C main 1 (Itzhak 2016 fractionation as tie-breaker, clipped 1/3-3),
     C main + N additional 0.3, C-only 0.02 (calibrated on the fit-half Tier 2 N/C); shares from Vn = 254.5, Vc = 385.5
     um3; applied when the model ratio is > 2-fold off the target.
 R6b nuclear speckles ('speckle' key -> label SPK): HPA main Nuclear speckles: 0.3 of the nuclear pool (0.6 for SR-type
     SRSF*/SON/SRRM1/2/TRA2A/B/PNN); all speckle shares scaled down to SPK_MAX mg/ml in the speckle volume.
Scope: single-gene ingredients and complexes / dimers with >= 50% of their mass from qualifying genes (v5 mechanism);
excluded: TM ingredients (except R4), v4-curated genes, actin/tubulin, genes that v5 already placed in the cortex or an
anchored shell (R2/R3/R5), lamina/nucleolus keys are kept."""
import os, re, sys, json, shutil, numpy as np, pandas as pd

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"; VAL = f"{D}/validation"; REF = f"{V}/data/reference"
ING = f"{D}/ingredients_v0.tsv"; BAK = f"{D}/ingredients_v0_pre_v6.tsv"
CHG = f"{D}/inventory_v6_changes.tsv"; REP = f"{D}/inventory_v6_report.txt"; CAL = f"{D}/v6/calibration_v6.json"
DRY = f"{D}/v6/dryrun_v6.json"
VN, VC = 254.5, 385.5
VER = "v6"
CYTO_KEYS = {"cytosol", "cytoskeleton", "peroxisome", "lipid_droplet", "unassigned", "plasma_membrane"}
SP_KEYS = {"er", "golgi", "endolysosomal", "secreted"}           # cytoplasmic when the ingredient has no signal peptide
ANCHOR_KEYS = {"pm_anchor_cyto", "pm_anchor_exo", "er_anchor_cyto", "er_anchor_lumen", "golgi_anchor_cyto", "golgi_anchor_lumen",
               "endo_anchor_cyto", "endo_anchor_lumen", "om_anchor_cyto", "ne_anchor_nuc", "cortex", "lamina",
               "pm_periph_cyto", "endo_periph_cyto", "golgi_periph_cyto"}
ACT_TUB = re.compile(r"^(ACT[BG]\d?|ACTG1|TUB[AB]\w*)$")
NUC_T = {"Nucleoplasm", "Nuclear bodies", "Nuclear speckles", "Nucleoli", "Nucleoli fibrillar center", "Nucleoli rim",
         "Mitotic chromosome", "Kinetochore"}
SPERM = {"Acrosome", "Annulus", "Calyx", "Connecting piece", "End piece", "Equatorial segment", "Flagellar centriole", "Mid piece",
         "Perinuclear theca", "Principal piece"}
ORG_T = {"Mitochondria", "Endoplasmic reticulum", "Golgi apparatus", "Vesicles", "Endosomes", "Lysosomes", "Peroxisomes",
         "Lipid droplets", "Nuclear membrane"}                 # organelle terms: R1 skips genes with these as main
PM_T = {"Plasma membrane", "Cell Junctions", "Focal adhesion sites"}
R2_T = PM_T | {"Actin filaments"}
MEMB_TGT = {"Plasma membrane": "pm", "Cell Junctions": "pm", "Focal adhesion sites": "pm", "Vesicles": "endo", "Endosomes": "endo",
            "Lysosomes": "endo", "Golgi apparatus": "golgi"}
R_CAT = {"N": 10.0, "Nc": 3.0, "NC": 1.0, "Cn": 0.3, "C": 0.02}
S_DEFAULT = 0.5
MIN_CONTRAST = 0.25           # log2 rim contrast (PM-TM vs cytosol references, HPA JURKAT fit half) needed to calibrate shares
SPK_MAX = 150.0               # mg/ml cap on speckle protein (moderate density; nucleolus was capped at 110 mg/ml, placed 88%)
# InterPro: membrane-binding domains (R3) and actin-binding domains (R2)
MEMB_IPR = {"IPR001849": "PH", "IPR002219": "C1", "IPR000008": "C2", "IPR004148": "BAR", "IPR027267": "AH/BAR",
            "IPR031160": "F-BAR", "IPR001060": "FCH", "IPR013606": "I-BAR (IMD)", "IPR000299": "FERM", "IPR019749": "FERM (band 4.1)",
            "IPR018979": "FERM N", "IPR001683": "PX", "IPR000306": "FYVE", "IPR017455": "FYVE-related", "IPR013809": "ENTH",
            "IPR011417": "ANTH", "IPR004182": "GRAM", "IPR001464": "annexin", "IPR000007": "tubby"}
ACT_IPR = {"IPR001715": "CH", "IPR002108": "ADF-H", "IPR007123": "gelsolin", "IPR003124": "WH2", "IPR015425": "FH2",
           "IPR000533": "tropomyosin", "IPR003128": "villin headpiece", "IPR002558": "I/LWEQ", "IPR001609": "myosin motor",
           "IPR011259": "ERM C-terminal (actin binding)", "IPR001997": "calponin/LIMCH1 repeat", "IPR000048": "IQ? no",
           "IPR030379": "septin-type G (cortical septin filaments, actin-associated)"}
ACT_IPR.pop("IPR000048")
ACT_GO = re.compile(r"actin filament binding \[GO:0051015\]|actin binding \[GO:0003779\]")
ACT_CC = re.compile(r"actin cytoskeleton \[GO:0015629\]|actin filament \[GO:0005884\]|cortical actin cytoskeleton \[GO:0030864\]")
PERI_CC = re.compile(r"cytoplasmic side of plasma membrane|extrinsic component of (cytoplasmic side of )?(plasma |endosome |golgi )?membrane|"
                     r"cytoplasmic side of (endosome|early endosome|late endosome|golgi) membrane", re.I)
GOLGI_CC = re.compile(r"Golgi membrane \[GO:0000139\]|COPI vesicle coat \[GO:0030126\]|COPI-coated vesicle|cis-Golgi network \[GO:0005801\]|"
                      r"Golgi stack \[GO:0005795\]")
GOLGI_BP = re.compile(r"Golgi organization \[GO:0007030\]")
NPC_CC = re.compile(r"nuclear pore( nuclear basket| cytoplasmic filaments| central transport channel)? \[GO:00")
NPC_MF = re.compile(r"nuclear (import|export) signal receptor activity|nuclear localization sequence binding \[GO:0008139\]")
SR_RE = re.compile(r"^(SRSF\d+|SON|SRRM[12]|TRA2[AB]|PNN)$")
CHECK = {"R1": "loc_compare class i (fractionation N/C, 409 genes) + HPA-JURKAT gallery (ITK nucleus too bright) + Tier 2 N/C",
         "R2": "loc_compare classes a/g (actin-binding at PM, packed soluble) + HPA-JURKAT gallery (LCP1/MYO1G cortex match) + Tier 2 rim",
         "R3": "HPA-JURKAT gallery (ARHGAP15/45, ADD3 PM rims; STAM vesicles) + loc_compare class g (873 genes) + Tier 2 rim",
         "R4": "HPA-JURKAT gallery (CD3E/CD3D ER/perinuclear pool, model PM rim) + Tier 2 image metric",
         "R5": "v5 pack (Golgi-cyto shell 35% placed) + loc_compare class g (COPI coats, golgins, NSF)",
         "R6a": "v6 plan R6 (transport receptors / basket annotated at the nuclear membrane, no NPC-proximal compartment)",
         "R6b": "v6 plan R6 (HPA main Nuclear speckles, 343 genes, no speckle compartment)"}
sys.path.insert(0, f"{V}/jurkat/validate"); sys.path.insert(0, V)


def main(apply):
    os.environ["LOC_VERSION"] = VER
    import loc_compare as lc
    os.makedirs(f"{D}/v6", exist_ok=True)
    src = BAK if os.path.exists(BAK) else ING
    I0 = pd.read_csv(src, sep="\t"); I = I0.copy()
    T = pd.read_csv(f"{VAL}/loc_compare_v5.tsv", sep="\t", index_col=0, low_memory=False); T = T[~T.index.duplicated()]
    H = pd.read_csv(f"{REF}/hpa_subcellular/subcellular_location.tsv", sep="\t").drop_duplicates("Gene name").set_index("Gene name")
    C = pd.read_csv(f"{VAL}/hpa_celllines.tsv", sep="\t", keep_default_na=False)
    U = pd.read_csv(f"{REF}/uniprot_human_domains.tsv.gz", sep="\t").set_index("Entry")
    inv = pd.read_csv(f"{D}/inventory_v0.tsv", sep="\t").drop_duplicates("gene").set_index("gene")
    gmass = inv.mass_da.to_dict(); gacc = inv.uniprot.to_dict()
    rd = lambda f: {l.strip() for l in open(f) if l.strip() and not l.startswith("#")}
    HOLD, FIT = rd(f"{VAL}/jurkat_holdout_genes.txt"), rd(f"{VAL}/jurkat_fit_genes.txt")
    ch = pd.read_csv(f"{D}/inventory_v4_changes.tsv", sep="\t")
    cur = ch[ch.field.isin(["comp_weights", "transmembrane"]) & ch.reason.astype(str).str.match(r'^"?[bcf]')]
    CUR = {g for x in cur.gene.astype(str) for g in x.split(";")}
    # v4 curations that only removed a mitochondrial share (f/drop_mito) never set the nucleus/cytosol split: R1 may act on them
    CUR_MITO = {g for x in cur[cur.reason.astype(str).str.contains("drop_mito")].gene.astype(str) for g in x.split(";")}
    c5 = pd.read_csv(f"{D}/inventory_v5_changes.tsv", sep="\t")
    V5CX = {g for x in c5[c5.rule == "C_cortex"].gene.astype(str) for g in x.split(";")}
    log, out, moved = [], [], {}
    gene_rows = []                                            # per-gene rule table (gene, rule, evidence, ...)

    def say(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

    # ------------------------------------------------------------------ per-gene inputs
    def hpa_cons(g):
        """(main set, additional set, reliability, source): consensus; held-out JURKAT genes: non-JURKAT per-line consensus"""
        if g in HOLD:
            sub = C[(C.gene == g) & (C.cell_line.str.upper() != "JURKAT") & (C.ab_reliability != "uncertain") & (C.locations != "")]
            if not len(sub): return set(), set(), "", "non-JURKAT per-line consensus (none)"
            locs = [set(x.lower().split(";")) for x in sub.locations]
            fr = {t: sum(t in l for l in locs) / len(locs) for t in set().union(*locs)}
            fix = lambda t: NORM.get(t.lower(), t)
            main = {fix(t) for t, f in fr.items() if f >= 0.5}; add = {fix(t) for t, f in fr.items() if 0.25 <= f < 0.5}
            rel = H.at[g, "Reliability"] if g in H.index else ""
            return main - SPERM, add - SPERM, rel, f"non-JURKAT per-line consensus ({len(sub)} annotations)"
        if g not in H.index: return set(), set(), "", "none"
        h = H.loc[g]; sp = lambda x: set(str(x).split(";")) - {"nan", ""} - SPERM
        return sp(h["Main location"]), sp(h["Additional location"]), h["Reliability"], "HPA consensus"

    NORM = {t.lower(): t for col in ("Main location", "Additional location") for x in H[col].dropna() for t in x.split(";")}
    HC = {}
    def hc(g):
        if g not in HC: HC[g] = hpa_cons(g)
        return HC[g]

    def feat(g):
        a = gacc.get(g)
        if not isinstance(a, str) or a not in U.index: return None
        r = U.loc[a]
        if isinstance(r, pd.DataFrame): r = r.iloc[0]
        ipr = set(str(r["InterPro"]).rstrip(";").split(";")) if isinstance(r["InterPro"], str) else set()
        return dict(ipr=ipr, kw=str(r["Keywords"]), mf=str(r["Gene Ontology (molecular function)"]),
                    cc=str(r["Gene Ontology (cellular component)"]), bp=str(r["Gene Ontology (biological process)"]),
                    up=str(r["Subcellular location [CC]"]))
    FE = {}
    def fe(g):
        if g not in FE: FE[g] = feat(g)
        return FE[g]

    up_tm = {}
    UPF = pd.read_csv(f"{REF}/uniprot_human_full.tsv.gz", sep="\t", usecols=["Entry", "Transmembrane", "Intramembrane"]).set_index("Entry")
    def is_tm(g):
        a = gacc.get(g)
        if not isinstance(a, str) or a not in UPF.index: return False
        r = UPF.loc[a]; return isinstance(r["Transmembrane"], str) and "TRANSMEM" in r["Transmembrane"] or \
            isinstance(r["Intramembrane"], str) and "INTRAMEM" in r["Intramembrane"]

    # ------------------------------------------------------------------ ingredient helpers
    def W(i): return json.loads(I.at[i, "comp_weights"]) if isinstance(I.at[i, "comp_weights"], str) else {}
    def genes(i): return str(I.at[i, "genes"]).split(";")
    def stoich(i): return [float(x) for x in str(I.at[i, "stoich"]).split(";")]
    def single(i): return len(set(genes(i))) == 1
    def frac_mass(i, S):
        m = [(g in S) * gmass.get(g, 0) * s for g, s in zip(genes(i), stoich(i))]; t = sum(gmass.get(g, 0) * s for g, s in zip(genes(i), stoich(i)))
        return sum(m) / t if t > 0 else 0.0
    def cyto_keys(i, w):
        sp = I.at[i, "signal_frac"] >= 0.5
        return [k for k in w if k in CYTO_KEYS or (k in SP_KEYS and not sp)]
    def wjson(w): return json.dumps({k: round(float(v), 4) for k, v in sorted(w.items(), key=lambda kv: -kv[1]) if v >= 1e-4})
    EV = {}                                                    # (rule, gene) -> evidence text

    def setw(i, nw, rule, gs):
        ow = W(i); d = 0.5 * sum(abs(nw.get(k, 0) - ow.get(k, 0)) for k in set(nw) | set(ow))
        if d < 1e-4: return False
        r = moved.setdefault(rule, [0.0, set(), set()]); r[0] += d * I.at[i, "mass_pg"]; r[1] |= set(gs); r[2].add(I.at[i, "ing_id"])
        ev = " | ".join(f"{g}: {EV.get((rule.split('_')[0], g), '')}" for g in sorted(gs))
        log.append(dict(version=VER, gene=I.at[i, "genes"], ingredient=I.at[i, "ing_id"], rule_id=rule, field="comp_weights",
                        old_value=I.at[i, "comp_weights"], new_value=wjson(nw), evidence=ev, motivating_check=CHECK[rule.split("_")[0]],
                        mass_moved_pg=round(d * I.at[i, "mass_pg"], 6)))
        I.at[i, "comp_weights"] = wjson(nw); return True

    def targets(S, allow_complex=True, allow_tm=False):
        """ingredient indices for gene set S: single-gene ingredients, or (allow_complex) >= 50% of the mass from S"""
        for i in range(len(I)):
            if bool(I.at[i, "transmembrane"]) != allow_tm and not (allow_tm is None): continue
            gs = set(genes(i)) & S
            if not gs: continue
            if single(i) or (allow_complex and frac_mass(i, S) >= 0.5): yield i, gs

    excl = lambda g: g in CUR or ACT_TUB.match(g) is not None
    rel_ok = lambda r: r in ("Enhanced", "Supported", "Approved")
    model_genes = [g for g in T.index if isinstance(g, str)]
    say(f"source table: {src} ({len(I0)} rows, {I0.mass_pg.sum():.3f} pg); model genes {len(model_genes)}; v4-curated {len(CUR)}; "
        f"JURKAT hold-out {len(HOLD)} / fit {len(FIT)}")

    # ------------------------------------------------------------------ calibration (Tier 2, JURKAT fit half only)
    T2 = pd.read_csv(f"{VAL}/tier2_v5.tsv", sep="\t", index_col=0)
    man = pd.read_csv(f"{REF}/hpa_subcellular/jurkat_images/manifest.tsv", sep="\t")
    ab_genes = man.groupby("antibody").gene.agg(set)
    shared = {g for gs in ab_genes for g in gs if len(gs) > 1 and (gs & HOLD) and (gs & FIT)}
    T2f = T2[(T2.split == "fit") & ~T2.index.isin(shared)]
    jl = lambda g: set(str(T2.at[g, "hpa_jurkat"]).split(";")) if g in T2.index and isinstance(T2.at[g, "hpa_jurkat"], str) else set()
    regs = lambda g: dict((k, float(v)) for k, v in (x.split(":") for x in str(T2.at[g, "model_regions"]).split(";") if ":" in x))
    cyto_ref = [g for g in T2f.index if jl(g) and jl(g) <= {"cytosol"} and regs(g).get("cytosol", 0) >= 0.9]
    pm_ref = [g for g in T2f.index if "plasma membrane" in jl(g) and regs(g).get("PM", 0) >= 0.8]
    Rc = float(np.median(2 ** T2f.loc[cyto_ref, "rim_hpa"])) if cyto_ref else np.nan
    Rp = float(np.median(2 ** T2f.loc[pm_ref, "rim_hpa"])) if pm_ref else np.nan
    cal = dict(note="Tier 2 HPA rim (linear) on JURKAT fit-half genes; share s = (R_gene - R_cytosol) / (R_PM_TM - R_cytosol), "
                    "clipped to 0.3-0.7; antibodies shared with held-out genes excluded",
               shared_antibody_genes_excluded=sorted(shared), cyto_ref=cyto_ref, pm_ref=pm_ref, R_cyto=Rc, R_pm=Rp)
    say(f"\ncalibration: cytosol reference {len(cyto_ref)} genes (R {Rc:.2f}), PM-TM reference {len(pm_ref)} genes (R {Rp:.2f}); "
        f"shared-antibody genes excluded: {sorted(shared)}")

    def share_from_rim(gs):
        gs = [g for g in gs if g in T2f.index and T2f.at[g, "rim_hpa"] == T2f.at[g, "rim_hpa"]]
        if not gs or not (np.log2(Rp / Rc) >= MIN_CONTRAST): return S_DEFAULT, gs, None   # PM vs cytosol rim contrast too small
        Rx = float(np.median(2 ** T2f.loc[gs, "rim_hpa"])); s = (Rx - Rc) / (Rp - Rc)
        return float(np.clip(s, 0.3, 0.7)), gs, s

    # N/C calibration: HPA image N/C by JURKAT category in the fit half vs the model's composite transfer
    tr = json.load(open(f"{VAL}/tier2_transfer.json"))
    def jcat(g):
        t = {x.lower() for x in jl(g)}; nt = {x.lower() for x in NUC_T}; n = bool(t & nt); c = bool(t - nt - {"nuclear membrane"})
        return "N" if n and not c else "NC" if n and c else "C" if c else None
    catc = {}
    for g in T2f.index:
        k = jcat(g)
        if k and T2f.at[g, "nc_hpa"] == T2f.at[g, "nc_hpa"]: catc.setdefault(k, []).append(float(T2f.at[g, "nc_hpa"]))
    cal["nc_by_jurkat_category"] = {k: dict(n=len(v), median_hpa_nc=round(float(np.median(v)), 3)) for k, v in catc.items()}
    # composite transfer: HPA nc = a + b (c0 + d log2 r) with the model image transfer c0 + d log2 r fitted in tier2 (v5)
    c0, d0 = 0.29, 0.566
    for k, v in cal["nc_by_jurkat_category"].items():
        v["implied_log2_r"] = round(((v["median_hpa_nc"] - tr["nc"]["a"]) / tr["nc"]["b"] - c0) / d0, 2)
    cal["r_by_category"] = R_CAT
    cal["nc_note"] = ("HPA N/C is compressed ~4x (transfer slope b) and JURKAT-category medians imply strong exclusion for "
                      "C-only (log2 r << -3) and strong enrichment for N-only; mapping set to N 10, Nc 3, NC 1, Cn 0.3, C 0.02")
    say("N/C by JURKAT category (fit half):", json.dumps(cal["nc_by_jurkat_category"]))

    # ------------------------------------------------------------------ R4: TM steady-state pools
    R4 = {}
    for g in model_genes:
        if excl(g): continue
        m_, a_, rel, srcs = hc(g)
        if not rel_ok(rel) or not ({"Endoplasmic reticulum", "Golgi apparatus"} & m_): continue
        R4[g] = (m_, a_, rel, srcs)
    n4 = 0
    for i, gs in targets(set(R4), allow_complex=True, allow_tm=True):
        w = W(i); pm = sum(v for k, v in w.items() if k == "plasma_membrane" or k.startswith("pm_"))
        tot = sum(w.values())
        if tot <= 0 or pm / tot < 0.7: continue
        g0 = sorted(gs)[0]; m_ = R4[g0][0]
        keep = 0.5 if (m_ & PM_T) else 0.2
        tg = {k: 1.0 for k, t in (("er", "Endoplasmic reticulum"), ("golgi", "Golgi apparatus")) if t in m_}
        mv = pm - keep * tot
        if mv <= 0: continue
        nw = {k: v * (1 - mv / pm) if (k == "plasma_membrane" or k.startswith("pm_")) else v for k, v in w.items()}
        for k in tg: nw[k] = nw.get(k, 0) + mv / len(tg)
        for g in gs: EV[("R4", g)] = f"TM; HPA main {';'.join(sorted(R4[g][0]))} ({R4[g][2]}, {R4[g][3]}); model PM {pm / tot:.2f} -> {keep}"
        n4 += setw(i, nw, "R4_tm_pool", gs)
    say(f"\nR4: candidates (HPA main ER/Golgi, Approved+) {len(R4)}; ingredients changed {n4}")
    for g in ("CD3E", "CD3D", "CD3G", "CD247"):
        m_, a_, rel, srcs = hc(g)
        say(f"  {g}: split {'holdout' if g in HOLD else 'fit' if g in FIT else 'not JURKAT-imaged'}; {srcs}: main {sorted(m_)} add {sorted(a_)} ({rel})"
            f" -> {'R4 candidate' if g in R4 else 'not changed (no ER/Golgi main support outside JURKAT)'}")

    # ------------------------------------------------------------------ R2: actin-binding -> cortex
    def actin_mech(g):
        f = fe(g)
        if not f: return None
        ev = []
        if ACT_GO.search(f["mf"]): ev.append("GO MF " + ";".join(sorted(set(m.group(0) for m in ACT_GO.finditer(f["mf"])))))
        if ACT_CC.search(f["cc"]): ev.append("GO CC " + ";".join(sorted(set(m.group(0) for m in ACT_CC.finditer(f["cc"])))))
        if "Actin-binding" in f["kw"]: ev.append("UniProt KW Actin-binding")
        d = sorted(ACT_IPR[x] + f" ({x})" for x in f["ipr"] & set(ACT_IPR))
        if d: ev.append("InterPro " + ", ".join(d))
        # DECISION: a GO term alone is not enough (EEF2 carries GO MF actin filament binding); need an actin-binding domain,
        # the curated UniProt keyword, or GO MF together with GO CC actin cytoskeleton
        strong = bool(d) or "Actin-binding" in f["kw"] or (ACT_GO.search(f["mf"]) and ACT_CC.search(f["cc"]))
        return "; ".join(ev) if strong else None
    R2 = {}
    for g in model_genes:
        if excl(g) or g in V5CX or is_tm(g): continue
        m_, a_, rel, srcs = hc(g)
        hit = (m_ | a_) & R2_T
        if not rel_ok(rel) or not hit: continue
        mech = actin_mech(g)
        if not mech: continue
        R2[g] = f"HPA {';'.join(sorted(hit))} ({'main' if m_ & R2_T else 'add'}, {rel}, {srcs}); mechanism {mech}"
    s2, s2g, s2raw = share_from_rim(set(R2) & FIT)
    cal["R2"] = dict(share=s2, fit_genes=s2g, raw=s2raw)
    say(f"\nR2: actin-binding genes with HPA actin/PM annotation {len(R2)} (v5 cortex genes kept as they are: {len(V5CX)}); "
        f"cortex share {s2:.2f} (fit-half genes {s2g}, raw {s2raw})")
    R2done = set()
    for i, gs in targets(set(R2)):
        w = W(i)
        if set(w) & ANCHOR_KEYS: continue
        ck = cyto_keys(i, w); pool = sum(w[k] for k in ck)
        if pool <= 0: continue
        nw = {k: (v * (1 - s2) if k in ck else v) for k, v in w.items()}; nw["cortex"] = nw.get("cortex", 0) + s2 * pool
        for g in gs: EV[("R2", g)] = R2[g] + f"; cortex = {s2:.2f} x cytoplasmic pool"
        if setw(i, nw, "R2_cortex", gs): R2done |= gs

    # ------------------------------------------------------------------ R3: peripheral membrane binders
    def memb_mech(g, main_t=None):
        """(evidence, allowed target membranes). A membrane-binding domain allows every HPA membrane target; a GO CC
        cytoplasmic-side/extrinsic term or a UniProt 'X; Peripheral membrane protein; Cytoplasmic side' statement allows
        only the membrane it names, and only when HPA has that membrane as a MAIN location (DECISION: generic or
        side-less 'Peripheral membrane protein' statements, e.g. GDI2, USP14, KARS1, and extracellular-side ones, e.g. ADA,
        do not count)."""
        f = fe(g)
        if not f: return None, set()
        d = sorted(MEMB_IPR[x] + f" ({x})" for x in f["ipr"] & set(MEMB_IPR))
        if d: return "InterPro " + ", ".join(d), {"pm", "endo", "golgi"}
        ev, ok = [], set()
        for m in PERI_CC.finditer(f["cc"]):
            t = memtgt(m.group(0))
            if t and main_t and t in main_t: ok.add(t); ev.append(f"GO CC {m.group(0)}")
        if "Chaperone" in f["kw"]: return None, set()   # chaperones meet membranes via clients (HSPA8 lysosomal CMA pool): no
        up = re.sub(r"\{[^}]*\}", "", f["up"]).replace("SUBCELLULAR LOCATION:", " ")
        for st in re.split(r"\.\s+", up):
            if "Peripheral membrane protein" in st and "Cytoplasmic side" in st:
                t = memtgt(st.split(";")[0])
                if t and main_t and t in main_t: ok.add(t); ev.append(f"UniProt '{st.strip()}'")
        return ("; ".join(ev) if ok else None), ok

    def memtgt(txt):
        t = txt.lower()
        if re.search(r"golgi|copi", t): return "golgi"
        if re.search(r"endosom|lysosom|clathrin-coated vesicle|multivesicular", t): return "endo"
        if re.search(r"cell membrane|plasma membrane|coated pit|cell junction|ruffle|lamellipodium|cell projection|focal adhesion", t): return "pm"
        return None
    R3 = {}
    for g in model_genes:
        if excl(g) or g in V5CX or g in R2 or is_tm(g): continue
        m_, a_, rel, srcs = hc(g)
        tg = {}
        for t in m_ | a_:
            if t in MEMB_TGT: tg[MEMB_TGT[t]] = max(tg.get(MEMB_TGT[t], 0), 1.0 if t in m_ else 0.3)
        if not rel_ok(rel) or not tg: continue
        mech, ok = memb_mech(g, {MEMB_TGT[t] for t in m_ if t in MEMB_TGT})
        tg = {k: v for k, v in tg.items() if k in ok}
        if not mech or not tg: continue
        R3[g] = (tg, f"HPA {';'.join(sorted((m_ | a_) & set(MEMB_TGT)))} ({rel}, {srcs}); mechanism {mech}")
    s3, s3g, s3raw = share_from_rim({g for g in set(R3) & FIT if "pm" in R3[g][0]})
    cal["R3"] = dict(share=s3, fit_genes=s3g, raw=s3raw)
    say(f"\nR3: peripheral binders with an HPA membrane location {len(R3)}; membrane share {s3:.2f} (fit-half PM genes {s3g}, raw {s3raw})")
    for g in ("HSP90AB1", "GAPDH", "ENO1", "EEF2"):
        m_, a_, rel, srcs = hc(g)
        say(f"  check {g}: HPA main {sorted(m_)} ({rel}); membrane mechanism: {memb_mech(g)}; actin mechanism: {actin_mech(g)}; "
            f"selected R2 {g in R2}, R3 {g in R3}")
    for g in ("HSPA8", "ADA", "GDI2", "USP14", "KARS1", "FKBP1A", "PKM", "ALDOA", "HSPA5", "HSP90AA1"):
        say(f"  check {g}: selected R2 {g in R2}, R3 {g in R3}" + (f" ({R3[g][1]})" if g in R3 else ""))
    R3done = set()
    for i, gs in targets(set(R3)):
        w = W(i)
        if set(w) & ANCHOR_KEYS: continue
        ck = cyto_keys(i, w); pool = sum(w[k] for k in ck)
        if pool <= 0: continue
        tg = {}
        for g in gs:
            for k, v in R3[g][0].items(): tg[k] = tg.get(k, 0) + v
        st = sum(tg.values())
        nw = {k: (v * (1 - s3) if k in ck else v) for k, v in w.items()}
        for k, v in tg.items(): nw[f"{k}_periph_cyto"] = nw.get(f"{k}_periph_cyto", 0) + s3 * pool * v / st
        for g in gs: EV[("R3", g)] = R3[g][1] + f"; membrane shells {', '.join(f'{k} {v:.1f}' for k, v in R3[g][0].items())} x {s3:.2f}"
        if setw(i, nw, "R3_peripheral", gs): R3done |= gs

    # ------------------------------------------------------------------ R5: Golgi binders
    R5 = {}
    for g in model_genes:
        if excl(g) or g in V5CX or g in R2 or g in R3 or is_tm(g): continue
        f = fe(g)
        if not f: continue
        mech = [m.group(0) for m in GOLGI_CC.finditer(f["cc"])] + [m.group(0) for m in GOLGI_BP.finditer(f["bp"])]
        if not mech: continue
        m_, a_, rel, srcs = hc(g)
        hg = "Golgi apparatus" in (m_ | a_) and rel_ok(rel)
        ig = 0.0
        if g in T.index:
            ig = float(str(T.at[g, "inv_compartments"]).count("golgi:") and
                       sum(float(x.split(":")[1]) for x in str(T.at[g, "inv_compartments"]).split(";") if x.startswith("golgi:")))
        if not (hg or ig >= 0.2): continue
        R5[g] = f"GO {';'.join(sorted(set(mech)))}; " + (f"HPA Golgi ({rel}, {srcs})" if hg else f"inventory golgi share {ig:.2f}")
    say(f"\nR5: Golgi binders (GO Golgi membrane / COPI coat / Golgi organization + HPA or inventory Golgi) {len(R5)}")
    for i, gs in targets(set(R5)):
        w = W(i)
        if set(w) & ANCHOR_KEYS: continue
        ck = cyto_keys(i, w); pool = sum(w[k] for k in ck)
        if pool <= 0: continue
        nw = {k: (v * 0.5 if k in ck else v) for k, v in w.items()}; nw["golgi_periph_cyto"] = nw.get("golgi_periph_cyto", 0) + 0.5 * pool
        for g in gs: EV[("R5", g)] = R5[g] + "; golgi_periph_cyto = 0.5 x cytoplasmic pool"
        setw(i, nw, "R5_golgi", gs)

    # ------------------------------------------------------------------ R6a: NPC-proximal shell
    R6a = {}
    for g in model_genes:
        if excl(g) or is_tm(g): continue
        f = fe(g)
        if not f: continue
        cc = [m.group(0) for m in NPC_CC.finditer(f["cc"])]; mf = [m.group(0) for m in NPC_MF.finditer(f["mf"])]
        if cc: R6a[g] = (0.3, f"GO CC {';'.join(sorted(set(cc)))}")
        elif mf: R6a[g] = (0.15, f"GO MF {';'.join(sorted(set(mf)))}")
    say(f"\nR6a: NPC-proximal genes {len(R6a)} (nuclear pore CC {sum(v[0] == 0.3 for v in R6a.values())}, "
        f"transport receptors {sum(v[0] == 0.15 for v in R6a.values())})")
    R6adone = set()
    for i, gs in targets(set(R6a)):
        w = W(i)
        ck = cyto_keys(i, w) + [k for k in ("nucleus", "nuclear_envelope") if k in w]; pool = sum(w[k] for k in ck)
        if pool <= 0: continue
        f = float(np.average([R6a[g][0] for g in gs]))
        nw = {k: (v * (1 - f) if k in ck else v) for k, v in w.items()}; nw["npc"] = nw.get("npc", 0) + f * pool
        for g in gs: EV[("R6a", g)] = R6a[g][1] + f"; npc = {f:.2f} x (nucleus + cytoplasmic) pool"
        if setw(i, nw, "R6a_npc", gs): R6adone |= gs

    # ------------------------------------------------------------------ R1: nucleus / cytosol from imaging
    def r_itz():
        f = f"{V}/data/literature/localisation/itzhak2016_partition.tsv"
        Z = pd.read_csv(f, sep="\t"); Z = Z[Z.primary_for_gene & (Z.n_good_mnc_profiles >= 3)].drop_duplicates("gene").set_index("gene")
        out_ = {}
        for g, r in Z.iterrows():
            fn = max(r.frac_nuclear / max(r.frac_nuclear + r.frac_cytosol, 1e-9) - 0.05, 0.0) / 0.95
            out_[g] = float(np.clip((fn / VN) / max((1 - fn) / VC, 1e-9), 1 / 3, 3)) if 0 < fn < 1 else (3.0 if fn >= 1 else 1 / 3)
        return out_
    ITZ = r_itz()
    R1 = {}
    for g in model_genes:
        if (excl(g) and g not in CUR_MITO) or ACT_TUB.match(g) or g in R2done | R3done | R6adone or is_tm(g): continue
        if g not in T.index or T.at[g, "m_nucleus"] + T.at[g, "m_cytosol"] < 0.8: continue
        m_, a_, rel, srcs = hc(g)
        if not rel_ok(rel) or (m_ & ORG_T): continue
        nm, cm = bool(m_ & NUC_T), bool(m_ - NUC_T - ORG_T - PM_T)       # PM / junction calls are not cytosol calls
        na, ca = bool(a_ & NUC_T), bool(a_ - NUC_T - ORG_T - PM_T)
        cat = "N" if nm and not cm and not ca else "Nc" if nm and not cm else "NC" if nm and cm else "Cn" if cm and na else "C" if cm else None
        if not cat: continue
        r = R_CAT[cat]; tb = ""
        if cat == "NC" and g in ITZ: r = ITZ[g]; tb = f"; Itzhak 2016 tie-breaker r {r:.2f}"
        R1[g] = (cat, r, f"HPA main {';'.join(sorted(m_))} add {';'.join(sorted(a_)) or '-'} ({rel}, {srcs}); category {cat} -> target N/C {r:.3g}{tb}")
    say(f"\nR1: genes with an imaging N/C category {len(R1)}: " + json.dumps(pd.Series([v[0] for v in R1.values()]).value_counts().to_dict()))
    nA = nB = 0
    for i, gs in targets(set(R1)):
        w = W(i)
        if set(w) & {"cortex", "lamina", "npc"}: continue
        ck = cyto_keys(i, w); wc = sum(w[k] for k in ck); wn = w.get("nucleus", 0.0); wl = w.get("nucleolus", 0.0) + w.get("speckle", 0.0)
        pool = wc + wn + wl
        if pool <= 0: continue
        cats = {R1[g][0] for g in gs}
        fam = {"N": "n", "Nc": "n", "C": "c", "Cn": "c", "NC": "nc"}
        if len({fam[c] for c in cats}) > 1: continue  # assemblies: every annotated member in the same family (N/Nc, C/Cn, NC)
        lr = [np.log2(R1[g][1]) for g in gs]; r = float(2 ** np.average(lr, weights=[gmass.get(g, 1) for g in gs]))
        rm = ((wn + wl) / VN) / max(wc / VC, 1e-12)
        if abs(np.log2(max(rm, 1e-6) / r)) <= 1.0: continue
        # DECISION: one-sided for all but NC: N and Nc targets are lower bounds (never make a nuclear protein less nuclear on
        # the strength of an additional cytosol call), C and Cn targets are upper bounds
        if cats <= {"N", "Nc"} and rm > r: continue
        if cats <= {"C", "Cn"} and rm < r: continue
        fn = r * VN / (r * VN + VC); new_n = max(fn * pool - wl, 0.0); new_c = pool - new_n - wl
        nw = {k: v for k, v in w.items() if k not in ck and k != "nucleus"}
        if new_n > 0: nw["nucleus"] = new_n
        if wc > 0:
            for k in ck: nw[k] = w[k] * new_c / wc
        elif new_c > 0: nw["cytosol"] = new_c
        for g in gs: EV[("R1", g)] = R1[g][2] + f"; model N/C {rm:.3g} -> {r:.3g} (nuclear share of the soluble pool {(wn + wl) / pool:.2f} -> {fn:.2f})"
        if setw(i, nw, "R1_nc", gs):
            if r > rm: nA += 1
            else: nB += 1
    say(f"R1: ingredients raised (more nuclear) {nA}, lowered {nB}")

    # ------------------------------------------------------------------ R6b: nuclear speckles
    R6b = {}
    for g in model_genes:
        if excl(g) or is_tm(g): continue
        m_, a_, rel, srcs = hc(g)
        if rel_ok(rel) and "Nuclear speckles" in m_:
            R6b[g] = (0.6 if SR_RE.match(g) else 0.3, f"HPA main Nuclear speckles ({rel}, {srcs})" + ("; SR-type" if SR_RE.match(g) else ""))
    geo = json.loads(str(np.load(f"{D}/jgeom_cell1.npz")["summary"])).get("speckles")
    vspk = geo["volume_um3"] if geo else 30 * 4 / 3 * np.pi * 0.25 ** 3
    plan = []
    for i, gs in targets(set(R6b)):
        w = W(i)
        if w.get("nucleus", 0) <= 0: continue
        plan.append((i, gs, float(np.average([R6b[g][0] for g in gs])), w))
    m_spk = sum(f * w["nucleus"] * I.at[i, "mass_pg"] for i, gs, f, w in plan)
    dens = m_spk / vspk * 1e3; sc = min(1.0, SPK_MAX / dens) if dens > 0 else 1.0
    say(f"\nR6b: speckle genes {len(R6b)} (SR-type {sum(v[0] > 0.5 for v in R6b.values())}); speckle volume {vspk:.2f} um3; "
        f"requested {m_spk:.3f} pg = {dens:.0f} mg/ml -> scale {sc:.3f} (cap {SPK_MAX} mg/ml)")
    for i, gs, f, w in plan:
        x = sc * f * w["nucleus"]; nw = dict(w); nw["nucleus"] = w["nucleus"] - x; nw["speckle"] = nw.get("speckle", 0) + x
        for g in gs: EV[("R6b", g)] = R6b[g][1] + f"; speckle = {R6b[g][0]} x nuclear pool x {sc:.3f} (capacity)"
        setw(i, nw, "R6b_speckle", gs)
    cal["R6b"] = dict(volume_um3=vspk, requested_pg=m_spk, scale=sc, cap_mg_ml=SPK_MAX)

    # ------------------------------------------------------------------ report + per-gene rule table
    say("\nrule | ingredients | genes | mass moved (pg)")
    for k, (m, g, ii) in moved.items(): say(f"  {k} | {len(ii)} | {len(g)} | {m:.3f}")
    for k in ("cortex", "pm_anchor_cyto", "endo_anchor_cyto", "golgi_anchor_cyto", "pm_periph_cyto", "endo_periph_cyto", "golgi_periph_cyto",
              "npc", "speckle", "nucleus", "cytosol", "er", "golgi"):
        a = sum(json.loads(w).get(k, 0) * p for w, p in zip(I0.comp_weights, I0.mass_pg)); b = sum(json.loads(w).get(k, 0) * p for w, p in zip(I.comp_weights, I.mass_pg))
        say(f"  key {k}: {a:.3f} -> {b:.3f} pg")
    L = pd.DataFrame(log, columns=["version", "gene", "ingredient", "rule_id", "field", "old_value", "new_value", "evidence",
                                   "motivating_check", "mass_moved_pg"])
    L.to_csv(CHG, sep="\t", index=False)
    G = []
    for rule, dct in (("R1", {g: v[2] for g, v in R1.items()}), ("R2", R2), ("R3", {g: v[1] for g, v in R3.items()}),
                      ("R4", {g: f"HPA main {';'.join(sorted(v[0]))} ({v[2]}, {v[3]})" for g, v in R4.items()}), ("R5", R5),
                      ("R6a", {g: v[1] for g, v in R6a.items()}), ("R6b", {g: v[1] for g, v in R6b.items()})):
        changed = moved.get({"R1": "R1_nc", "R2": "R2_cortex", "R3": "R3_peripheral", "R4": "R4_tm_pool", "R5": "R5_golgi",
                             "R6a": "R6a_npc", "R6b": "R6b_speckle"}[rule], [0, set()])[1]
        for g, e in dct.items():
            G.append(dict(version=VER, gene=g, rule_id=rule, selected_evidence=e, changed=g in changed,
                          jurkat_half="holdout" if g in HOLD else "fit" if g in FIT else "", motivating_check=CHECK[rule]))
    pd.DataFrame(G).to_csv(f"{D}/v6/inventory_v6_rule_genes.tsv", sep="\t", index=False)
    json.dump(cal, open(CAL, "w"), indent=1, default=float)
    say(f"\ntotal mass {I0.mass_pg.sum():.4f} -> {I.mass_pg.sum():.4f} pg; {len(L)} field changes in {L.ingredient.nunique()} ingredients -> {CHG}")
    dry = dryrun(I0, I, lc, HOLD, say)
    json.dump(dry, open(DRY, "w"), indent=1)
    if apply:
        for c in I0.columns:
            if c != "transmembrane": I[c] = I[c].astype(I0[c].dtype)
        I["transmembrane"] = I.transmembrane.astype(bool)
        tmp = ING + ".tmp"; I.to_csv(tmp, sep="\t", index=False)
        if not os.path.exists(BAK):
            shutil.copy2(ING, BAK); say(f"snapshot written: {BAK}")
        os.replace(tmp, ING); say(f"APPLIED: {ING} rewritten")
    else:
        say("dry run: ingredients_v0.tsv not modified (use --apply)")
    open(REP, "w").write("\n".join(out) + "\n")


# ---------------------------------------------------------------------- dry-run localisation score
def intended(Iv, lc, memb, consumed, pre):
    """per-gene comparison-compartment distribution (intended copies) of an ingredient table, as loc_compare.model"""
    V0 = pd.read_csv(f"{D}/inventory_v0.tsv", sep="\t").set_index("gene")
    _, fam = lc.model_genes(); famw = {f: (V0.loc[m, "copies"] / V0.loc[m, "copies"].sum()) for f, m in fam.items()}
    acc = {}
    for r in Iv.itertuples():
        if r.copies <= 0: continue
        cw = json.loads(r.comp_weights) if isinstance(r.comp_weights, str) else {"unassigned": 1.0}
        rest = max(r.copies - float(consumed.get(r.ing_id, 0)), 0.0); tm = bool(r.transmembrane) and memb.get(r.ing_id, False)
        ent = {}
        for comp, wt in cw.items():
            if tm:
                reg = lc.TMR.get(comp, "ER")
                if reg == "MITO_IM" and r.mito_om_frac >= 0.5: reg = "MITO_OM"
            else:
                reg = lc.SOL.get(comp, "cytosol")
                if reg in ("ER_lumen", "golgi_lumen", "endo_lumen") and r.signal_frac < 0.5: reg = "cytosol"
            c = lc.REG2COMP[reg]; ent[c] = ent.get(c, 0) + rest * wt
        for reg, c in pre.get(r.ing_id, {}).items():
            cc = lc.REG2COMP[reg]; ent[cc] = ent.get(cc, 0) + c
        for g, s in zip(str(r.genes).split(";"), [float(x) for x in str(r.stoich).split(";")]):
            for gg, fw in (famw[g].items() if g in famw else [(g, 1.0)]):
                d = acc.setdefault(gg, {})
                for c, x in ent.items(): d[c] = d.get(c, 0) + x * s * fw
    return {g: {c: v / sum(d.values()) for c, v in d.items() if v > 0} for g, d in acc.items() if sum(d.values()) > 0}


def dryrun(I0, I1, lc, HOLD, say):
    W = f"{D}/wc"
    L = json.load(open(f"{W}/libs.json")); consumed = L["consumed"]
    z = np.load(f"{D}/validation/cache/loc_pack_v5.npz", allow_pickle=True); memb = dict(zip(z["ids"], z["memb"]))
    PL = json.load(open(f"{W}/prepass_libs.json")); pc = np.bincount(np.load(f"{W}/prepass_types.npy", mmap_mode="r"), minlength=len(PL))
    pre = {}
    for l, c in zip(PL, pc):
        ing = lc.PRE_ING.get(l, l); pre.setdefault(ing, {}); pre[ing][lc.pre_region(l)] = pre[ing].get(lc.pre_region(l), 0) + float(c)
    P0 = intended(I0, lc, memb, consumed, pre); P1 = intended(I1, lc, memb, consumed, pre)
    T = pd.read_csv(f"{VAL}/loc_compare_v5.tsv", sep="\t", index_col=0, low_memory=False); T = T[~T.index.duplicated()]
    genes = [g for g in T.index if g in P0 and g in P1]
    S = lc.sources(T.loc[genes]); J = lc.jurkat_refs()
    mass = T.mass_pg
    res = {}
    def agg(rows):
        if not rows: return None
        R = pd.DataFrame(rows, columns=["g", "a0", "a1", "m"]); w = R.m
        return dict(n=len(R), v5_unweighted=round(R.a0.mean(), 4), v6_unweighted=round(R.a1.mean(), 4),
                    v5_mass_weighted=round(float((R.a0 * w).sum() / w.sum()), 4), v6_mass_weighted=round(float((R.a1 * w).sum() / w.sum()), 4),
                    improved=int((R.a1 > R.a0 + 1e-6).sum()), worsened=int((R.a1 < R.a0 - 1e-6).sum()),
                    worst=R.assign(d=R.a1 - R.a0).sort_values("d").head(8)[["g", "a0", "a1"]].round(3).values.tolist())
    sets = {"opencell (clean)": [], "hpa_jurkat_holdout_half (clean)": [], "hpa_jurkat_fit_half": [], "hpa_test_half (not clean)": [],
            "fit sources combined (circular)": []}
    for g in genes:
        oc = S[g].get("opencell", {}).get("q")
        if oc: sets["opencell (clean)"].append((g, 1 - lc.score(P0[g], oc)["score"], 1 - lc.score(P1[g], oc)["score"], mass[g]))
        jq = J.get(g, {}).get("q")
        if jq:
            k = "hpa_jurkat_holdout_half (clean)" if g in HOLD else "hpa_jurkat_fit_half"
            sets[k].append((g, 1 - lc.score(P0[g], jq)["score"], 1 - lc.score(P1[g], jq)["score"], mass[g]))
        hq = S[g].get("hpa", {})
        if hq.get("q") and hq.get("w", 0) > 0 and lc.hpa_split(g) == "test":
            sets["hpa_test_half (not clean)"].append((g, 1 - lc.score(P0[g], hq["q"])["score"], 1 - lc.score(P1[g], hq["q"])["score"], mass[g]))
        c0, c1 = lc.combine(P0[g], S[g])[0], lc.combine(P1[g], S[g])[0]
        if c0 == c0 and c1 == c1: sets["fit sources combined (circular)"].append((g, 1 - c0, 1 - c1, mass[g]))
    say("\nDRY RUN: predicted intended localisation agreement (1 - score), v5 inventory -> v6 inventory")
    for k, rows in sets.items():
        a = agg(rows); res[k] = a
        if a: say(f"  {k}: n {a['n']}; unweighted {a['v5_unweighted']:.4f} -> {a['v6_unweighted']:.4f}; mass-weighted "
                  f"{a['v5_mass_weighted']:.4f} -> {a['v6_mass_weighted']:.4f}; improved {a['improved']}, worsened {a['worsened']}; "
                  f"worst: {a['worst'][:5]}")
    return res


if __name__ == "__main__":
    main("--apply" in sys.argv[1:])
