"""Inventory revision v4: curated overrides on the packing ingredient table (PLANNING_NOTES §8 "Issues found", §7f NPC caveat).

usage: jinventory_v4.py            dry run: prints the summary, writes data/derived/inventory_v4_changes.tsv (+ _report.txt)
       jinventory_v4.py --apply    as above, then rewrites data/derived/ingredients_v0.tsv in place (same columns/dtypes).
                                   The first --apply copies the table to ingredients_v0_pre_v4.tsv. Every run computes
                                   from that backup when it exists, so applying twice gives the same table.
                                   It refuses (unless --force) if ingredients_v0.tsv was regenerated since the backup.

Overrides (each change logged as ing_id, gene, field, old, new, reason):
 a. Single-pass TM proteins without a usable membrane localisation (comp_weights 'unassigned' >= 0.5, or only
    cytosol/cytoskeleton/unassigned, all of which the packer sends to the ER membrane), of type I (signal peptide, or
    UniProt 'Single-pass type I') or type II (UniProt 'type II'/'signal-anchor', or no signal peptide and the TM starts
    within the first N_TERM_TM residues), with plasma-membrane support in UniProt CC ('Cell membrane', 'cell surface', ...)
    or GO CC (plasma membrane / cell surface / external side of PM): the unassigned/cytosol/cytoskeleton share goes to
    plasma_membrane. Single-gene ingredients only (monomers, homo-oligomers, AFDB homodimers). Type III/IV (C-terminal
    tail anchors, which are mostly ER/mito/Golgi) are left alone.
    Localisation source: the local UniProt dump data/reference/uniprot_human_full.tsv.gz (CC text + GO CC); TM count and
    first TM start from jurkat/yc/pipeline/data/uniprot_topology.tsv.
 b. Free CD3 chains (CD3E, CD3D, CD3G, CD247 monomers outside CPX-6482): capped at CD3_CAP x TCR-CD3 complex copies and
    routed to the ER membrane (unassembled chains are ER-retained and degraded). The surplus is removed.
 c. Curated PM routing: CD7 -> plasma_membrane 1.0; CD99, SELPLG -> plasma_membrane 0.9 + 0.1 of their HPA compartment.
 d. NPC stoichiometry from PDB 7R5K (the placed library): per-gene chain counts of assembly 1 (Nup96 and Nup98 both come
    from the NUP98 precursor and are counted as mass-weighted precursor equivalents). Nups that are not in 7R5K
    (basket/FG: TPR, NUP153, NUP50, NUP42, GLE1, RAE1, POM121) are dropped from the NPC ingredient. The free monomer
    pool of each affected gene is recomputed as inventory copies - copies used by every other ingredient - 1000 x stoich
    (clipped at 0), so the extra 7R5K copies come out of the free pools and the dropped Nups return to them.
 e. RANBP2 overdraw: CPX-4747 (RanGAP1-SUMO1/RanBP2/UBC9, 99k copies) used about twice the RANBP2 inventory before the
    NPC took its 40 copies per pore. Its copies are capped at the RANBP2 left after the NPC; RANGAP1 and UBE2I return
    to their free pools.
 f. Curated localisation fixes from a review of the top 200 ingredients by mass (CURATED_W, DROP_MITO, SOLUBLE_NOT_TM
    below), and soluble signal-peptide proteins with a C-terminal ER-retrieval motif (PROSITE PS00014 [KRHQSA][DENQ]EL)
    set to ER lumen only (HPA 'plasma membrane'/'cytosol' weights pack them in the cytosol).
 g. Obligate OXPHOS complexes (OBLIGATE): the complex-copy rule (median capped at 2x the 25th-percentile subunit) is set by
    small hydrophobic subunits that iBAQ under-detects, leaving e.g. 1.5 M ATP5F1A + 2.1 M ATP5F1B as free matrix
    monomers. Copies are raised to the subunit-mass-weighted median of (N + free/stoich) over dedicated subunits (in no
    other assembly), capped at the lowest value among major subunits (>= MAJOR_F of complex mass, so a
    well-quantified core subunit is never imputed); free subunits are consumed (clipped at 0) and the deficit of under-detected subunits is imputed
    (added mass, reported). Applied to the whitelist only: most other free surpluses (ACTB, HSPA8, RAN, GAPDH, PPP1CA,
    RBBP4/7, paralog-alternative complexes) are genuine free pools or shared subunits.
 h. AFDB heterodimers whose partners have incompatible dominant compartments (soluble classes: lumen = signal peptide,
    mito = mitochondrion weight > 0.5, cyto = cytosol/nucleus/cytoskeleton/PM > 0.5; lumen-cyto, lumen-mito, mito-cyto;
    TM partners and mixed classes are not judged) are dissolved: the row is deleted and its copies are added to each
    partner's monomer row (nothing is created; partners without a row are reported and their copies dropped).
    NOTE: deleting rows breaks JWC_LIBS_FROM-style index reuse; recount without it after applying.
"""
import os, re, sys, json, shutil, filecmp, numpy as np, pandas as pd

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"
ING = f"{D}/ingredients_v0.tsv"
BAK = f"{D}/ingredients_v0_pre_v4.tsv"
CHG = f"{D}/inventory_v4_changes.tsv"
REP = f"{D}/inventory_v4_report.txt"
LIB = f"{V}/jurkat/yc/pipeline/lib"
UP = f"{V}/data/reference/uniprot_human_full.tsv.gz"
TOPO = f"{V}/jurkat/yc/pipeline/data/uniprot_topology.tsv"
NPC_CIF = f"{V}/data/structures/pdb/assemblies/7r5k-assembly1.cif.gz"
NPC_COPIES = 1000
CD3_CAP = 0.2
N_TERM_TM = 60
AV = 6.022e23

# 7R5K entity description -> gene (assembly 1 = full cytoplasmic+inner+nuclear rings, 8 spokes)
NPC_ENTITY_GENE = {
    "E3 SUMO-protein ligase RanBP2": "RANBP2", "Nuclear pore membrane glycoprotein 210": "NUP210", "Aladin": "AAAS",
    "Nuclear pore complex protein Nup93": "NUP93", "Nucleoporin NUP188 homolog": "NUP188",
    "Nuclear pore complex protein Nup205": "NUP205", "Nuclear pore complex protein Nup155": "NUP155",
    "Nucleoporin NDC1": "NDC1", "Nucleoporin NUP35": "NUP35", "Nucleoporin p54": "NUP54", "Nucleoporin p58/p45": "NUP58",
    "Nuclear pore glycoprotein p62": "NUP62", "Nuclear pore complex protein Nup133": "NUP133",
    "Nuclear pore complex protein Nup107": "NUP107", "Nuclear pore complex protein Nup96": "NUP98",
    "Protein SEC13 homolog": "SEC13", "Nucleoporin SEH1": "SEH1L", "Nuclear pore complex protein Nup85": "NUP85",
    "Nucleoporin Nup43": "NUP43", "Nuclear pore complex protein Nup160": "NUP160", "Nucleoporin Nup37": "NUP37",
    "Protein ELYS": "AHCTF1", "Nuclear pore complex protein Nup98": "NUP98", "Nuclear pore complex protein Nup214": "NUP214",
    "Nuclear pore complex protein Nup88": "NUP88"}

PM_CC = re.compile(r"cell membrane|plasma membrane|cell surface|cell projection|cell junction|synapse|"
                   r"apical cell membrane|basolateral cell membrane", re.I)
PM_GO = re.compile(r"plasma membrane \[GO:0005886\]|cell surface \[GO:0009986\]|external side of plasma membrane|"
                   r"immunological synapse|plasma membrane raft|cell-cell junction|uropod", re.I)
NO_LOC = {"unassigned", "cytosol", "cytoskeleton"}


# f. curated fixes found in the top-200-by-mass review (key = ing_id for complexes, else gene: all single-gene ingredients)
CURATED_W = [
    ("VDAC1", {"mitochondrion": 1.0}, "outer mitochondrial membrane porin; the PM annotation put 1.4 M copies on the PM"),
    ("CPX-6151", {"mitochondrion": 1.0}, "F1Fo ATP synthase is an inner-membrane complex (ecto-ATP synthase negligible)"),
    ("ATP5F1A", {"mitochondrion": 1.0}, "ATP synthase alpha (matrix F1); PM weight packed it in the cytosol"),
    ("HSPD1", {"mitochondrion": 1.0}, "HSP60 is a matrix chaperone; PM weight packed 25% in the cytosol"),
    ("GOT2", {"mitochondrion": 1.0}, "mitochondrial aspartate aminotransferase (matrix)"),
    ("ESYT1", {"er": 1.0}, "extended synaptotagmins are ER TM proteins tethering ER to the PM"),
    ("ESYT2", {"er": 1.0}, "extended synaptotagmins are ER TM proteins tethering ER to the PM"),
    ("RTN4", {"er": 1.0}, "reticulon: ER tubule-shaping membrane protein"),
    ("CPX-2375", {"er": 1.0}, "TAPBP-PDIA3 (peptide-loading complex) is ER-resident"),
    ("CPX-5621", {"er": 1.0}, "oligosaccharyltransferase (OST-A) is ER-resident"),
    ("CPX-5622", {"er": 1.0}, "oligosaccharyltransferase (OST-B) is ER-resident"),
    ("CPX-7669", {"nuclear_envelope": 1.0}, "SUN2-SYNE2 LINC complex spans the nuclear envelope"),
    ("ATP2B4", {"plasma_membrane": 1.0}, "PMCA4 is a plasma-membrane Ca2+ pump")]
DROP_MITO = ["PGK1", "CPX-6598", "GSTP1", "VAT1", "PRDX1", "UBA1", "ILF3", "HSP90AA1", "CPX-3288"]
SOLUBLE_NOT_TM = ["CLIC1", "CLIC4"]
# g. obligate complexes (no functional free pool of their core subunits): ATP synthase, complexes I, II, III. Complex IV
# (CPX-6123) has no ingredient (no PDB >= 50% coverage), so its subunits stay free monomers (suggestion: build it).
OBLIGATE = ["CPX-6151", "CPX-577", "CPX-561", "CPX-560"]
# h. genuine cross-compartment interactions (OM BCL2 proteins + BH3-only partners, E3 adaptors, IMS/IM pairs) kept
KEEP_HET = {"SKP1-FBXL12", "ASB9-ELOC", "WSB1-ELOC", "RTKN-RHOC", "RTKN-CDC42", "UQCC1-UQCC2", "BCL2L2-BMF",
            "BCL2A1-BCL2L11", "BCL2A1-PMAIP1", "PRELID3B-TRIAP1", "CIBAR1-CBY1"}
MAJOR_F = 0.05      # a subunit carrying >= 5% of the complex mass counts as reliably quantified
SOL = {"cytosol": "cytosol", "cytoskeleton": "cytosol", "peroxisome": "cytosol", "lipid_droplet": "cytosol", "unassigned": "cytosol",
       "plasma_membrane": "cytosol", "nucleus": "nucleoplasm", "nuclear_envelope": "nucleoplasm", "mitochondrion": "matrix",
       "er": "ER_lumen", "secreted": "ER_lumen", "golgi": "golgi_lumen", "endolysosomal": "endo_lumen"}   # = jpack_test.SOL
TMR = {"plasma_membrane": "PM", "nucleus": "NE", "nuclear_envelope": "NE", "mitochondrion": "MITO_IM", "er": "ER", "golgi": "GOLGI",
       "endolysosomal": "ENDO"}                                                                         # = jpack_test.TMR (else ER)


def region_mass(T, memlib):
    """expected mass (pg) per pack region with the packer's mapping (jwc.step_counts), before pre-pass consumption"""
    out = {}
    for r in T.itertuples():
        for comp, wt in json.loads(r.comp_weights).items():
            if r.transmembrane and r.ing_id in memlib:
                reg = TMR.get(comp, "ER")
                if reg == "MITO_IM" and r.mito_om_frac >= 0.5: reg = "MITO_OM"
            else:
                reg = SOL.get(comp, "cytosol")
                if reg in ("ER_lumen", "golgi_lumen", "endo_lumen") and r.signal_frac < 0.5: reg = "cytosol"
            out[reg] = out.get(reg, 0.0) + wt * r.mass_pg
    return out


def wjson(w):
    return json.dumps({k: round(float(v), 3) for k, v in sorted(w.items(), key=lambda kv: -kv[1]) if v >= 0.001})


def pg(copies, mass):
    return copies * mass / AV * 1e12


def npc_7r5k():
    """per-gene chain counts and modelled entity masses in 7R5K assembly 1 (read from the mmCIF categories)."""
    import gemmi
    st = gemmi.read_structure(NPC_CIF)
    blk = gemmi.cif.read(NPC_CIF).sole_block()
    ent = {r[0]: (gemmi.cif.as_string(r[1]), float(r[2])) for r in blk.find("_entity.", ["id", "pdbx_description", "formula_weight"])}
    st.setup_entities()
    n = {}
    for ch in st[0]:
        sub = ch.get_polymer()
        if not len(sub): continue
        eid = st.get_entity_of(sub).name
        n[eid] = n.get(eid, 0) + 1
    rows = [(eid, ent[eid][0], NPC_ENTITY_GENE[ent[eid][0]], c, ent[eid][1]) for eid, c in n.items()]
    return pd.DataFrame(rows, columns=["entity", "description", "gene", "chains", "entity_fw"])


def main(apply):
    src = BAK if os.path.exists(BAK) else ING
    I0 = pd.read_csv(src, sep="\t")
    I = I0.copy()
    inv = pd.read_csv(f"{D}/inventory_v0_complexed.tsv", sep="\t", index_col="gene")
    up = pd.read_csv(UP, sep="\t", usecols=["Entry", "Subcellular location [CC]", "Gene Ontology (cellular component)",
                                            "Signal peptide", "Length", "Sequence"]).set_index("Entry")
    topo = pd.read_csv(TOPO, sep="\t").set_index("Entry")
    memlib = set(open(f"{D}/membrane_lib_ids.txt").read().split())
    log = []
    out = [f"source table: {src}  ({len(I0)} rows, {I0.mass_pg.sum():.3f} pg)"]

    def say(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

    def chg(i, field, new, reason):
        old = I.at[i, field]
        if str(old) == str(new): return
        log.append(dict(ing_id=I.at[i, "ing_id"], gene=I.at[i, "genes"], field=field, old=old, new=new, reason=reason))
        I.at[i, field] = new

    def row(ing):
        m = np.nonzero((I.ing_id == ing).values)[0]
        return int(m[0]) if len(m) else None

    # ------------------------------------------------------------------ a. single-pass type I/II -> PM
    single = I.genes.map(lambda s: len(set(s.split(";"))) == 1) & I.transmembrane
    gene_up = inv.uniprot
    cand = []; fun = {k: [0, 0.0] for k in ("no_loc", "single_pass", "type_I_II", "PM_support")}
    for i in np.nonzero(single.values)[0]:
        r = I.iloc[i]; g = r.genes.split(";")[0]; w = json.loads(r.comp_weights)
        if g in ("CD3E", "CD3D", "CD3G", "CD247", "CD7", "CD99", "SELPLG"): continue   # handled in b/c
        if not (w.get("unassigned", 0) >= 0.5 or set(w) <= NO_LOC): continue
        fun["no_loc"][0] += 1; fun["no_loc"][1] += r.mass_pg
        acc = r.uniprot if isinstance(r.uniprot, str) and r.uniprot else gene_up.get(g)
        if not isinstance(acc, str) or acc not in up.index: continue
        tmtxt = topo.Transmembrane.get(acc, np.nan)
        tms = re.findall(r"TRANSMEM (\d+)\.\.(\d+)", tmtxt) if isinstance(tmtxt, str) else []
        if len(tms) != 1: continue
        fun["single_pass"][0] += 1; fun["single_pass"][1] += r.mass_pg
        cc = str(up.at[acc, "Subcellular location [CC]"]); go = str(up.at[acc, "Gene Ontology (cellular component)"])
        sig = isinstance(up.at[acc, "Signal peptide"], str)
        tm0 = int(tms[0][0]); L = int(up.at[acc, "Length"])
        cc1, cc2 = "type I membrane" in cc, re.search(r"type II membrane|signal-anchor", cc, re.I) is not None
        if re.search(r"type III membrane|type IV membrane|tail-anchored", cc, re.I) and not (cc1 or cc2): continue
        t1 = cc1 or (sig and not cc2)
        t2 = cc2 or (not sig and tm0 <= N_TERM_TM)
        if not (t1 or t2): continue
        fun["type_I_II"][0] += 1; fun["type_I_II"][1] += r.mass_pg
        ev_cc = PM_CC.search(re.sub(r"\{[^}]*\}", "", cc)) is not None
        ev_go = PM_GO.search(go) is not None
        if not (ev_cc or ev_go): continue
        fun["PM_support"][0] += 1; fun["PM_support"][1] += r.mass_pg
        mv = sum(v for k, v in w.items() if k in NO_LOC)
        nw = {k: v for k, v in w.items() if k not in NO_LOC}
        nw["plasma_membrane"] = nw.get("plasma_membrane", 0) + mv
        why = (f"a: single-pass type {'I' if t1 else 'II'} (sig={int(sig)}, TM@{tm0}/{L}); PM support "
               f"{'+'.join(x for x, f in (('UniProtCC', ev_cc), ('GO_CC', ev_go)) if f)}; was {'unassigned' if 'unassigned' in w else 'cytosol/cytoskeleton'} -> ER default")
        cand.append((r.ing_id, g, r.copies, mv * r.mass_pg, r.mass_pg, 'I' if t1 else 'II', ev_cc, ev_go, r.ing_id in memlib, r.comp_weights))
        chg(i, "comp_weights", wjson(nw), why)
    A = pd.DataFrame(cand, columns=["ing_id", "gene", "copies", "moved_pg", "mass_pg", "type", "cc_PM", "go_PM", "memb_lib", "old_w"])
    say(f"\n[a] single-pass type I/II TM proteins with no usable localisation routed to PM: {len(A)} ingredients, "
        f"{A.moved_pg.sum():.4f} pg moved ({A.copies.sum():,} copies; membrane library for {A.memb_lib.sum()} of them, "
        f"{A.moved_pg[A.memb_lib].sum():.4f} pg; the rest have no oriented library and pack as soluble cytosol either way)")
    say("    funnel (single-gene TM ingredients; n, pg): " + "; ".join(f"{k} {v[0]} ({v[1]:.4f} pg)" for k, v in fun.items()))
    say(f"    evidence: UniProt CC {int(A.cc_PM.sum())}, GO only {int((~A.cc_PM & A.go_PM).sum())}; type I {int((A.type == 'I').sum())}, type II {int((A.type == 'II').sum())}")
    say("    top 30 by mass moved:\n" + A.sort_values("moved_pg", ascending=False).head(30)[
        ["ing_id", "copies", "moved_pg", "type", "cc_PM", "go_PM", "memb_lib", "old_w"]].to_string(index=False))

    # ------------------------------------------------------------------ b. free CD3 chains
    tcr = row("CPX-6482"); ntcr = int(I.at[tcr, "copies"]); cap = int(round(CD3_CAP * ntcr))
    say(f"\n[b] TCR-CD3 complex CPX-6482: {ntcr:,} copies ({I.at[tcr, 'genes']} x {I.at[tcr, 'stoich']}); free-chain cap {cap:,}")
    for g in ("CD3E", "CD3D", "CD3G", "CD247"):
        i = row(g)
        if i is None:
            say(f"    {g}: no free monomer ingredient (all copies in the complex)"); continue
        c0 = int(I.at[i, "copies"]); c1 = min(c0, cap)
        chg(i, "copies", c1, f"b: free {g} capped at {CD3_CAP} x TCR-CD3 ({ntcr}); unassembled chains are ER-retained/degraded; surplus removed")
        chg(i, "comp_weights", wjson({"er": 1.0}), "b: unassembled CD3 chains are ER-retained")
        say(f"    {g}: {c0:,} -> {c1:,} copies on the ER membrane (was {I0.at[i, 'comp_weights']}); removed {pg(c0 - c1, I.at[i, 'mass_da']):.4f} pg; "
            f"PM copies {c0 * json.loads(I0.at[i, 'comp_weights']).get('plasma_membrane', 0):,.0f} -> 0")

    # ------------------------------------------------------------------ c. curated PM proteins
    say("\n[c] curated PM routing")
    for g, keep in (("CD7", 0.0), ("CD99", 0.1), ("SELPLG", 0.1)):
        for i in np.nonzero(I.genes.map(lambda s: g in s.split(";")).values)[0]:
            if set(I.at[i, "genes"].split(";")) != {g}:
                say(f"    {g}: also in {I.at[i, 'ing_id']} (left as is)"); continue
            w = json.loads(I0.at[i, "comp_weights"])
            oth = {k: v for k, v in w.items() if k not in NO_LOC and k != "plasma_membrane"}
            nw = {"plasma_membrane": 1.0}
            if oth and keep > 0:
                nw = {"plasma_membrane": 1.0 - keep, **{k: keep * v / sum(oth.values()) for k, v in oth.items()}}
            chg(i, "comp_weights", wjson(nw), f"c: {g} is a T-cell surface protein; HPA/UniProt localisation was {I0.at[i, 'comp_weights']}")
            say(f"    {I.at[i, 'ing_id']}: {I0.at[i, 'comp_weights']} -> {I.at[i, 'comp_weights']} ({I.at[i, 'mass_pg']:.4f} pg, "
                f"{(1 - w.get('plasma_membrane', 0)) * nw['plasma_membrane'] * I.at[i, 'mass_pg']:.4f} pg to PM)")

    # ------------------------------------------------------------------ d. NPC stoichiometry (7R5K) + e. CPX-4747
    say("\n[d] NPC stoichiometry from 7R5K assembly 1")
    E = npc_7r5k()
    say(E.sort_values("entity", key=lambda s: s.astype(int)).to_string(index=False))
    gm = inv.mass_da
    st = E.groupby("gene").chains.sum().to_dict()
    # NUP98 precursor (Nup98 + Nup96 after autoproteolysis): mass-weighted precursor equivalents
    e98 = E[E.gene == "NUP98"]
    st["NUP98"] = int(round((e98.chains * e98.entity_fw).sum() / gm["NUP98"]))
    say(f"    NUP98 precursor equivalents: {' + '.join(f'{c} x {d}' for c, d in zip(e98.chains, e98.description))} "
        f"-> {st['NUP98']} x {gm['NUP98']:.0f} Da")
    npc = row("NPC")
    old_st = dict(zip(I.at[npc, "genes"].split(";"), map(int, I.at[npc, "stoich"].split(";"))))
    genes = sorted(st, key=lambda g: -st[g] * gm[g])
    new_mass = float(sum(st[g] * gm[g] for g in genes))
    chg(npc, "genes", ";".join(genes), "d: NPC composition = PDB 7R5K (the placed library); basket/FG Nups not in 7R5K dropped")
    chg(npc, "stoich", ";".join(str(st[g]) for g in genes), "d: per-gene chain counts in 7R5K assembly 1 (NUP98 = Nup98+Nup96 precursor equivalents)")
    chg(npc, "mass_da", new_mass, "d: sum of stoich x full-length gene mass")
    upsig = up["Signal peptide"].notna(); uptm = topo.Transmembrane.notna()
    acc = {g: gm_acc for g, gm_acc in inv.uniprot.items()}
    sf = round(sum(st[g] * gm[g] for g in genes if upsig.get(acc.get(g), False)) / new_mass, 3)
    chg(npc, "signal_frac", sf, "d: recomputed for the 7R5K composition")
    chg(npc, "n_tm", int(sum(bool(uptm.get(acc.get(g), False)) for g in genes)), "d: recomputed for the 7R5K composition")
    chg(npc, "mass_pg", pg(I.at[npc, "copies"], new_mass), "d: mass_da changed")
    say(f"    NPC mass {I0.at[npc, 'mass_da'] / 1e6:.2f} -> {new_mass / 1e6:.2f} MDa per pore; x{I.at[npc, 'copies']} = "
        f"{I0.at[npc, 'mass_pg']:.4f} -> {I.at[npc, 'mass_pg']:.4f} pg")

    # e. CPX-4747 cap (uses RANBP2 alongside the NPC)
    cx = row("CPX-4747")
    if cx is not None and "RANBP2" in I.at[cx, "genes"].split(";"):
        avail = int(inv.copies["RANBP2"]) - NPC_COPIES * st.get("RANBP2", 0)
        c0 = int(I.at[cx, "copies"]); c1 = max(0, min(c0, avail))
        chg(cx, "copies", c1, f"e: RANBP2 inventory {int(inv.copies['RANBP2']):,} - {NPC_COPIES * st.get('RANBP2', 0):,} in NPCs; complex was overdrawn")
        chg(cx, "mass_pg", pg(c1, I.at[cx, "mass_da"]), "e: copies changed")
        say(f"\n[e] CPX-4747 ({I.at[cx, 'genes']}): {c0:,} -> {c1:,} copies (RANBP2 inventory {int(inv.copies['RANBP2']):,}, "
            f"NPC needs {NPC_COPIES * st.get('RANBP2', 0):,}); {pg(c0 - c1, I.at[cx, 'mass_da']):.4f} pg; RANGAP1/UBE2I return to free pools")

    # free pools of every gene touched by d/e: inventory - other ingredients - (NPC), clipped at 0
    touched = set(old_st) | set(st) | (set(I.at[cx, "genes"].split(";")) if cx is not None else set())
    other = {g: 0.0 for g in touched}
    for i, r in I.iterrows():
        if r.kind == "protein" or r.ing_id == "NUCLEOSOME": continue
        for g, s in zip(r.genes.split(";"), r.stoich.split(";")):
            if g in other: other[g] += r.copies * float(s)
    rows = []
    for g in sorted(touched, key=lambda g: -st.get(g, 0) * gm.get(g, 0)):
        need = NPC_COPIES * st.get(g, 0); tot = int(inv.copies.get(g, 0))
        i = row(g); f0 = int(I.at[i, "copies"]) if i is not None else 0
        f1 = int(max(0, round(tot - other[g])))
        short = max(0, round(other[g] - tot))
        if i is not None:
            chg(i, "copies", f1, f"d/e: free pool = inventory {tot:,} - assemblies {int(other[g]):,} (NPC 7R5K stoich {st.get(g, 0)}, was {old_st.get(g, 0)})")
            chg(i, "mass_pg", pg(f1, I.at[i, "mass_da"]), "d/e: copies changed")
        rows.append((g, old_st.get(g, 0), st.get(g, 0), need, tot, int(other[g] - need), f0, f1 if i is not None else None, short))
    N = pd.DataFrame(rows, columns=["gene", "old_stoich", "7R5K", "NPC_need", "inventory", "other_assemblies", "free_before", "free_after", "shortfall"])
    say("    per-gene pools (free_after None = no free-monomer ingredient; shortfall = copies the inventory lacks):\n" + N.to_string(index=False))
    say(f"    genes whose inventory cannot cover NPC + other assemblies: {list(N.gene[N.shortfall > 0])}")

    # ------------------------------------------------------------------ f. curated localisation fixes (top-200 review)
    say("\n[f] curated localisation fixes")
    fmv = []

    def setw(i, nw, why, tag):
        w0 = json.loads(I.at[i, "comp_weights"]); nw = {k: v / sum(nw.values()) for k, v in nw.items()}
        moved = 0.5 * sum(abs(nw.get(k, 0) - w0.get(k, 0)) for k in set(nw) | set(w0)) * I.at[i, "mass_pg"]
        if moved < 1e-12: return
        chg(i, "comp_weights", wjson(nw), f"f/{tag}: {why}")
        fmv.append((tag, I.at[i, "ing_id"], moved, I.at[i, "comp_weights"], wjson(w0)))

    def rows_of(key):
        """an ing_id (complexes) or every single-gene ingredient of a gene"""
        if key.startswith("CPX-"): return [row(key)] if row(key) is not None else []
        return list(np.nonzero(I.genes.map(lambda x: set(x.split(";")) == {key}).values)[0])

    for key, nw, why in CURATED_W:
        for i in rows_of(key): setw(i, dict(nw), why, "curated")
    for key in DROP_MITO:
        for i in rows_of(key):
            w = {k: v for k, v in json.loads(I.at[i, "comp_weights"]).items() if k != "mitochondrion"} or {"cytosol": 1.0}
            setw(i, w, "cytosolic protein without a mitochondrial presequence; HPA/UniProt mito share would pack it into the matrix", "drop_mito")
    for g in SOLUBLE_NOT_TM:
        for i in rows_of(g):
            chg(i, "transmembrane", False, "f/clic: CLIC proteins are mostly soluble (metamorphic; the membrane form is minor)")
            chg(i, "n_tm", 0, "f/clic: soluble")
            setw(i, {"cytosol": 0.8, "nucleus": 0.2}, "CLIC soluble pool (cytosol/nucleus)", "clic")
    # soluble signal-peptide proteins with a C-terminal ER-retrieval signal (PROSITE PS00014): ER lumen only
    kdel = re.compile(r"[KRHQSA][DENQ]EL$"); LUM = {"er", "golgi", "endolysosomal", "secreted"}
    for i in np.nonzero(((~I.transmembrane) & (I.signal_frac >= 0.5)).values)[0]:
        gs = set(I.at[i, "genes"].split(";"))
        if len(gs) != 1: continue
        a = inv.uniprot.get(next(iter(gs)))
        sq = up.Sequence.get(a) if isinstance(a, str) else None
        if not isinstance(sq, str) or not kdel.search(sq): continue
        w = json.loads(I.at[i, "comp_weights"])
        if set(w) <= {"er"}: continue
        setw(i, {"er": 1.0}, f"ER-retrieval C-terminus {sq[-4:]} + signal peptide: ER lumen resident (non-lumen weights pack it in the cytosol)", "kdel")
    F = pd.DataFrame(fmv, columns=["tag", "ing_id", "moved_pg", "new_w", "old_w"])
    for tag, x in F.groupby("tag", sort=False):
        say(f"    {tag}: {len(x)} ingredients, {x.moved_pg.sum():.4f} pg re-routed")
        say("      " + x.sort_values("moved_pg", ascending=False).head(15).to_string(index=False).replace("\n", "\n      "))

    # ------------------------------------------------------------------ g. obligate OXPHOS complexes absorb free core subunits
    say("\n[g] obligate complexes: copies raised toward the subunit-implied count (mass-weighted median, capped at the lowest major subunit)")
    nasm = {}
    for r in I[I.kind != "protein"].itertuples():
        for g in set(r.genes.split(";")): nasm[g] = nasm.get(g, 0) + 1
    gm_ = inv.mass_da
    for cid in OBLIGATE:
        ci = row(cid)
        if ci is None: continue
        gs = I.at[ci, "genes"].split(";"); ss = [float(x) for x in I.at[ci, "stoich"].split(";")]; N = int(I.at[ci, "copies"])
        fr = {g: (int(I.at[row(g), "copies"]) if row(g) is not None else 0) for g in gs}
        ded = [(g, s) for g, s in zip(gs, ss) if nasm.get(g, 0) == 1]           # subunits found in no other assembly
        if len(ded) < 2: continue
        imp = np.array([N + fr[g] / s for g, s in ded]); w = np.array([gm_[g] * s for g, s in ded])
        o = np.argsort(imp); Nmed = imp[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)]
        tot_m = sum(gm_[g] * s for g, s in zip(gs, ss))
        cap = min((N + fr[g] / s for g, s in zip(gs, ss) if gm_[g] * s >= MAJOR_F * tot_m), default=None)
        if cap is None: continue                                            # major (reliably quantified) subunits,
        Ns = int(round(min(Nmed, cap)))                                     # shared or not, are never imputed
        if Ns <= N: continue
        absorbed = imputed = 0.0
        chg(ci, "copies", Ns, f"g: obligate complex; N = min(subunit-mass-weighted median, lowest major subunit) of (N + free/stoich) over {len(ded)} dedicated subunits (was {N:,})")
        for g, s in zip(gs, ss):
            need = (Ns - N) * s; take = min(fr[g], need)
            absorbed += take * gm_[g]; imputed += (need - take) * gm_[g]
            if row(g) is not None and take > 0:
                chg(row(g), "copies", int(round(fr[g] - take)), f"g: free subunit absorbed into {cid} ({N:,} -> {Ns:,} copies)")
        say(f"    {cid} ({len(gs)} subunit types, {len(ded)} dedicated): {N:,} -> {Ns:,} copies; free subunits absorbed "
            f"{absorbed / AV * 1e12:.4f} pg; under-detected subunits imputed {imputed / AV * 1e12:.4f} pg; "
            f"largest remaining free: " + ", ".join(f"{g} {int(I.at[row(g), 'copies']):,}" for g in sorted(
                [g for g in gs if row(g) is not None], key=lambda g: -I.at[row(g), 'copies'] * gm_[g])[:3]))

    # ------------------------------------------------------------------ h. dissolve cross-compartment AFDB heterodimers
    say("\n[h] AFDB heterodimers whose partners have incompatible dominant compartments: dissolved")
    sigs = up["Signal peptide"].notna()

    def cls(g):
        """dominant compartment class of a soluble gene: lumen (signal peptide), mito, cyto; None = TM or mixed"""
        if bool(topo.Transmembrane.notna().get(inv.uniprot.get(g), False)): return None
        if bool(sigs.get(inv.uniprot.get(g), False)): return "lumen"
        i = row(g)
        w = json.loads(I.at[i, "comp_weights"]) if i is not None and I.at[i, "kind"] == "protein" else {
            k: float(v) for k, v in (t.split(":") for t in str(inv.compartments.get(g, "cytosol:1")).split(";"))}
        tot = sum(w.values()) or 1.0
        mi = w.get("mitochondrion", 0) / tot
        cy = sum(v for k, v in w.items() if k in ("cytosol", "nucleus", "cytoskeleton", "unassigned", "plasma_membrane")) / tot
        return "mito" if mi > 0.5 else "cyto" if cy > 0.5 else None
    BAD = {frozenset(("lumen", "cyto")), frozenset(("lumen", "mito")), frozenset(("mito", "cyto"))}
    drop, hrows = [], []
    for i in np.nonzero((I.kind == "afdb_heterodimer").values)[0]:
        g1, g2 = I.at[i, "genes"].split(";")
        c1, c2 = cls(g1), cls(g2)
        if None in (c1, c2) or frozenset((c1, c2)) not in BAD or I.at[i, "ing_id"] in KEEP_HET: continue
        n = int(I.at[i, "copies"]); note = []
        for g, c in ((g1, c1), (g2, c2)):
            j = row(g)
            if j is not None and I.at[j, "kind"] == "protein":
                chg(j, "copies", int(I.at[j, "copies"]) + n, f"h: + {n:,} copies from dissolved heterodimer {I.at[i, 'ing_id']} ({c1} vs {c2})")
                if not os.path.exists(f"{LIB}/{g}.npz"): note.append(f"{g}: monomer row but no library")
            else:
                note.append(f"{g}: no monomer row, {n:,} copies dropped")
        log.append(dict(ing_id=I.at[i, "ing_id"], gene=I.at[i, "genes"], field="row", old=f"copies={n}", new="deleted",
                        reason=f"h: dissolved, partners in incompatible compartments ({g1}={c1}, {g2}={c2})" + (f"; {'; '.join(note)}" if note else "")))
        drop.append(i); hrows.append((I.at[i, "ing_id"], c1, c2, n, I0.at[i, "mass_pg"], "; ".join(note)))
    Hd = pd.DataFrame(hrows, columns=["ing_id", "class1", "class2", "copies", "mass_pg", "note"])
    say(f"    {len(Hd)} heterodimers dissolved, {Hd.mass_pg.sum():.4f} pg, {Hd.copies.sum():,} dimers")
    if len(Hd): say("    " + Hd.sort_values("mass_pg", ascending=False).to_string(index=False).replace("\n", "\n    "))

    # ------------------------------------------------------------------ bookkeeping
    I["copies"] = I.copies.astype(np.int64)
    for i in set(np.nonzero((I.copies.values != I0.copies.values) | (I.mass_da.values != I0.mass_da.values))[0]):
        I.at[i, "mass_pg"] = pg(I.at[i, "copies"], I.at[i, "mass_da"])
    I = I.drop(index=I.index[drop]).reset_index(drop=True)          # h: dissolved heterodimer rows
    rb, ra = region_mass(I0, memlib), region_mass(I, memlib)
    say("\nexpected mass by pack region (pg, before -> after; packer mapping, pre-pass consumption not subtracted):")
    say("    " + "; ".join(f"{k} {rb.get(k, 0):.3f} -> {ra.get(k, 0):.3f}" for k in sorted(set(rb) | set(ra), key=lambda k: -ra.get(k, 0))))
    L = pd.DataFrame(log, columns=["ing_id", "gene", "field", "old", "new", "reason"])
    L.to_csv(CHG, sep="\t", index=False)
    say(f"\ntotal mass {I0.mass_pg.sum():.4f} -> {I.mass_pg.sum():.4f} pg; {len(L)} field changes in {L.ing_id.nunique()} ingredients -> {CHG}")
    if apply:
        for c in I0.columns:
            I[c] = I[c].astype(I0[c].dtype)
        assert list(I.columns) == list(I0.columns) and len(I) == len(I0) - len(drop)
        tmp = ING + ".tmp"; I.to_csv(tmp, sep="\t", index=False)
        if os.path.exists(BAK) and not (filecmp.cmp(ING, BAK, shallow=False) or filecmp.cmp(ING, tmp, shallow=False)) \
                and "--force" not in sys.argv:
            os.remove(tmp)
            sys.exit(f"ABORT: {ING} differs from both the v4 backup and the v4 result (regenerated by jingredients.py?). "
                     f"Remove {BAK} to rebase v4 on the new table, or pass --force to overwrite from the old backup.")
        if not os.path.exists(BAK):
            shutil.copy2(ING, BAK); say(f"backup written: {BAK}")
        os.replace(tmp, ING)
        say(f"APPLIED: {ING} rewritten")
    else:
        say("dry run: ingredients_v0.tsv not modified (use --apply)")
    open(REP, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main("--apply" in sys.argv[1:])
