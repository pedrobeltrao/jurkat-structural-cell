"""Inventory revision v5 (PLANNING_NOTES §7g "v5 scope — DECISION"): placement classes and compartments from the Tier-1
localisation review (data/derived/validation/loc_rules.md, loc_compare.tsv; v4 model table cache/loc_model.parquet).

usage: jinventory_v5.py            dry run: summary + data/derived/inventory_v5_changes.tsv (+ inventory_v5_report.txt)
       jinventory_v5.py --apply    as above, then rewrites data/derived/ingredients_v0.tsv (backup ingredients_v0_pre_v5.tsv;
                                   every run computes from that backup when it exists, so applying twice is idempotent).

New comp_weights keys (jpack_test.SOL maps them to regions): nucleolus; pm_anchor_cyto / pm_anchor_exo, er_anchor_cyto /
er_anchor_lumen, golgi_anchor_cyto / golgi_anchor_lumen, endo_anchor_cyto / endo_anchor_lumen, om_anchor_cyto, ne_anchor_nuc
(lipid-anchored shells, centre within BR + 2 nm of the bilayer surface); cortex (0-150 nm under the PM); lamina (0-25 nm
under the INM).

Rules (applied in this order; every change logged with its rule tag):
 T  TM routing (R3): ingredients for which jlib_tm_route.py built a membrane library (data/derived/v5/tm_route_libs.tsv,
    method != skipped) get transmembrane = True (n_tm from UniProt TRANSMEM + INTRAMEM if 0), so jwc routes them to the
    membrane of each compartment (jpack_test.TMR). mito_om_frac = 1 when at least half of the mitochondrial members annotated by UniProt are
    outer-membrane only (not also inner membrane/matrix), else 0 (TOM complex, BAK1, BAX, BCL2 family, MAVS, MFN1; not HADHB).
 L  Lamina: LMNB1, LMNB2: the nucleus + nuclear_envelope share -> lamina; LMNA 0.8 of it (0.2 nucleoplasmic lamin A/C pool);
    TMPO (P42166, LAP2alpha, non-TM in the model): 0.5 of it (LAP2alpha is largely nucleoplasmic). LBR is TM (membrane).
 A  Lipid anchors (R1_lipid_only): non-TM ingredients (< 1 MDa) containing a gene with a UniProt LIPID feature (myristoyl,
    palmitoyl, farnesyl/geranylgeranyl, GPI-anchor amidated): the inventory's membrane shares go to the anchored shell of
    that membrane: plasma_membrane -> pm_anchor, er -> er_anchor, golgi -> golgi_anchor, endolysosomal -> endo_anchor,
    nuclear_envelope -> ne_anchor_nuc, mitochondrion -> om_anchor_cyto only when UniProt says outer membrane.
    Side: GPI -> exoplasmic / lumenal; acyl/prenyl -> cytosolic. v4-curated compartments are kept (only the placement within
    the compartment changes). UniProt-driven, so it also applies to HPA-JURKAT genes.
 C  Cortex (R6): soluble actin/cortex-annotated genes whose HPA or UniProt main location is the PM (loc_compare.rules R6
    selection; HPA-JURKAT genes only when UniProt alone qualifies; v4-curated excluded): cortex = plasma_membrane share +
    0.5 x the other cytosol-region shares. Single-gene ingredients and complexes with >= 50% of their mass from such genes.
 N  Nucleolus: HPA main location Nucleoli / Nucleoli fibrillar center / Nucleoli rim, reliability Approved or better,
    not HPA-JURKAT, not v4-curated: nucleolar share = F of the nucleus share (F = 0.7; 0.85 for Enhanced/Supported when the
    nucleoli terms are the only nuclear main location). Single-gene ingredients and complexes with >= 50% of their mass
    from such genes (e.g. the SSU processome CPX-2511, the nucleolar pre-40S; the model has no free 40S/60S ingredients,
    mature 80S are cytosolic).  Capacity: if the nucleolar protein density would exceed NUCL_MAX mg/ml, all nucleolar
    shares are scaled down uniformly.
 P  Nuclear pool (R2b): genes with packed unit < 60 kDa, model nucleus < 5% and cytosol >= 80%, no TM / signal peptide,
    HPA or UniProt listing the nucleus; not HPA-JURKAT, not v4-curated, not actin/tubulin: nuclear share of the soluble
    pool (cytosol + unassigned + nucleus keys) = Vn / (Vn + Vc) (free diffusion, equal concentration). Single-gene
    ingredients only.
"""
import os, re, sys, json, shutil, numpy as np, pandas as pd

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"
VAL = f"{D}/validation"
ING = f"{D}/ingredients_v0.tsv"
BAK = f"{D}/ingredients_v0_pre_v5.tsv"
CHG = f"{D}/inventory_v5_changes.tsv"
REP = f"{D}/inventory_v5_report.txt"
UP = f"{V}/data/reference/uniprot_human_full.tsv.gz"
TMLIB = f"{D}/v5/tm_route_libs.tsv"
AV = 6.02214076e23
VN, VC = 254.5, 385.5        # um3: v4 nucleoplasm minus the nucleolus (5.9) and cytosol (tile measures, §7f)
VNUCL = 5.9                  # um3 nucleolus (EM nucleolus_seg in cell 1)
NUCL_MAX = 110.0             # mg/ml cap on nucleolar protein (v5 test tiles: 168 mg/ml requested jammed at 91 mg/ml placed, 54%;
                             # IDR-rich nucleolar proteins; rRNA not modelled). The rest of each nucleolar share stays nucleoplasmic.
MAX_ANCHOR_DA = 1e6
CYTO_KEYS = {"cytosol", "cytoskeleton", "peroxisome", "lipid_droplet", "unassigned", "plasma_membrane"}
ACT_TUB = re.compile(r"^(ACT[BG]\d?|ACTG1|TUB[AB]\w*)$")
sys.path.insert(0, f"{V}/jurkat/validate"); sys.path.insert(0, V)


def wjson(w):
    return json.dumps({k: round(float(v), 4) for k, v in sorted(w.items(), key=lambda kv: -kv[1]) if v >= 1e-4})


def main(apply):
    import loc_compare as lc
    src = BAK if os.path.exists(BAK) else ING
    I0 = pd.read_csv(src, sep="\t"); I = I0.copy()
    T = pd.read_csv(f"{VAL}/loc_compare.tsv", sep="\t", index_col=0, low_memory=False)
    up = pd.read_csv(UP, sep="\t", usecols=["Entry", "Gene Names (primary)", "Subcellular location [CC]", "Lipidation", "Transmembrane",
                                            "Intramembrane"]).set_index("Entry")
    C = pd.read_csv(f"{VAL}/hpa_celllines.tsv", sep="\t", keep_default_na=False)
    JUR = set(C[C.cell_line.str.upper() == "JURKAT"].gene)
    ch = pd.read_csv(f"{D}/inventory_v4_changes.tsv", sep="\t")
    cur = ch[ch.field.isin(["comp_weights", "transmembrane"]) & ch.reason.astype(str).str.match(r'^"?[bcf]')]
    CUR = {g for x in cur.gene.astype(str) for g in x.split(";")}
    inv = pd.read_csv(f"{D}/inventory_v0.tsv", sep="\t").drop_duplicates("gene").set_index("gene")
    gmass = inv.mass_da.to_dict(); gacc = inv.uniprot.to_dict()
    log, out = [], [f"source table: {src} ({len(I0)} rows, {I0.mass_pg.sum():.3f} pg); HPA-JURKAT genes {len(JUR)}; v4-curated {len(CUR)}"]

    def say(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

    def chg(i, field, new, rule):
        old = I.at[i, field]
        if str(old) == str(new): return
        log.append(dict(ing_id=I.at[i, "ing_id"], gene=I.at[i, "genes"], field=field, old=old, new=new, rule=rule))
        I.at[i, field] = new

    def W(i): return json.loads(I.at[i, "comp_weights"]) if isinstance(I.at[i, "comp_weights"], str) else {}

    def genes(i): return str(I.at[i, "genes"]).split(";")

    def stoich(i): return [float(x) for x in str(I.at[i, "stoich"]).split(";")]

    def frac_mass(i, S):
        """mass fraction of ingredient i contributed by genes in S"""
        m = [(g in S) * gmass.get(g, 0) * s for g, s in zip(genes(i), stoich(i))]; t = sum(gmass.get(g, 0) * s for g, s in zip(genes(i), stoich(i)))
        return sum(m) / t if t > 0 else 0.0

    def single(i): return len(set(genes(i))) == 1

    moved = {}                                                 # rule -> [mass pg moved, set of genes, set of ings]

    def setw(i, nw, rule):
        ow = W(i)
        d = 0.5 * sum(abs(nw.get(k, 0) - ow.get(k, 0)) for k in set(nw) | set(ow))
        if d < 1e-6: return
        r = moved.setdefault(rule, [0.0, set(), set()])
        r[0] += d * I.at[i, "mass_pg"]; r[1] |= set(genes(i)); r[2].add(I.at[i, "ing_id"])
        chg(i, "comp_weights", wjson(nw), rule)

    # ------------------------------------------------------------------ T. TM routing
    if os.path.exists(TMLIB):
        R = pd.read_csv(TMLIB, sep="\t")
        ok = set(R[R.method.astype(str) != "skipped"].ing_id)
        say(f"\nT: membrane libraries built for {len(ok)} of {len(R)} ingredients ({TMLIB})")
        for i in np.nonzero(I.ing_id.isin(ok).values)[0]:
            r = moved.setdefault("T_tm_routing", [0.0, set(), set()]); r[0] += I.at[i, "mass_pg"]; r[1] |= set(genes(i)); r[2].add(I.at[i, "ing_id"])
            chg(i, "transmembrane", True, "T_tm_routing")
            if int(I.at[i, "n_tm"]) == 0:
                n = 0
                for g in set(genes(i)):
                    a = gacc.get(g)
                    if isinstance(a, str) and a in up.index:
                        n += sum(str(up.at[a, c]).count(k) for c, k in (("Transmembrane", "TRANSMEM"), ("Intramembrane", "INTRAMEM")))
                chg(i, "n_tm", max(n, 1), "T_tm_routing")
            w = W(i)
            if w.get("mitochondrion", 0) > 0:
                def _om(t): t = t.lower(); return "outer membrane" in t and "inner membrane" not in t and "matrix" not in t
                om = [_om(str(up.at[gacc[g], "Subcellular location [CC]"])) for g in set(genes(i))
                      if isinstance(gacc.get(g), str) and gacc[g] in up.index and "mitochondri" in str(up.at[gacc[g], "Subcellular location [CC]"]).lower()]
                if om: chg(i, "mito_om_frac", 1.0 if np.mean(om) >= 0.5 else 0.0, "T_tm_routing_OM")
    else:
        say(f"\nT: {TMLIB} missing: TM routing skipped")

    # ------------------------------------------------------------------ L. lamina
    LAM = {"LMNB1": 1.0, "LMNB2": 1.0, "LMNA": 0.8, "TMPO": 0.5}
    for i in range(len(I)):
        g = genes(i)
        if not single(i) or g[0] not in LAM or bool(I.at[i, "transmembrane"]): continue
        w = W(i); f = LAM[g[0]]; pool = w.get("nucleus", 0) + w.get("nuclear_envelope", 0)
        nw = {k: v for k, v in w.items() if k not in ("nucleus", "nuclear_envelope")}
        nw["lamina"] = f * pool
        if 1 - f > 0: nw["nucleus"] = (1 - f) * pool
        setw(i, nw, "L_lamina")

    # ------------------------------------------------------------------ A. lipid anchors
    def lipid_kind(g):
        a = gacc.get(g)
        if not isinstance(a, str) or a not in up.index: return None
        t = str(up.at[a, "Lipidation"])
        if "GPI-anchor" in t: return "gpi"
        if re.search(r"myristoyl|palmitoyl|farnesyl|geranylgeranyl", t): return "cyto"
        return None
    LK = {g: lipid_kind(g) for g in inv.index}
    LK = {g: k for g, k in LK.items() if k}
    say(f"\nA: genes with a UniProt lipid anchor: {len(LK)} ({sum(k == 'gpi' for k in LK.values())} GPI)")
    MAPA = {"plasma_membrane": "pm", "er": "er", "golgi": "golgi", "endolysosomal": "endo", "nuclear_envelope": "ne"}
    lam_genes = set(LAM)
    for i in range(len(I)):
        if bool(I.at[i, "transmembrane"]) or I.at[i, "mass_da"] > MAX_ANCHOR_DA: continue
        ks = {LK[g] for g in set(genes(i)) if g in LK}
        if not ks or (single(i) and genes(i)[0] in lam_genes): continue
        side = "gpi" if ks == {"gpi"} else "cyto"
        w = W(i); nw = dict(w); hit = False
        for comp, pre in MAPA.items():
            if w.get(comp, 0) <= 0: continue
            if pre == "ne":
                if side == "gpi": continue
                key = "ne_anchor_nuc"
            else:
                key = f"{pre}_anchor_" + ({"pm": "exo"}.get(pre, "lumen") if side == "gpi" else "cyto")
            nw[key] = nw.get(key, 0) + nw.pop(comp); hit = True
        if w.get("mitochondrion", 0) > 0 and side == "cyto":
            om = any("mitochondrion outer membrane" in str(up.at[gacc[g], "Subcellular location [CC]"]).lower() for g in set(genes(i))
                     if g in LK and isinstance(gacc.get(g), str) and gacc[g] in up.index)
            if om: nw["om_anchor_cyto"] = nw.pop("mitochondrion"); hit = True
        if hit: setw(i, nw, f"A_lipid_{side}")

    # ------------------------------------------------------------------ C. cortex (R6 selection as loc_compare.rules)
    cand = T.index[T.cause.isin(["a_peripheral_or_lipid_anchored", "g_membrane_main_no_anchor_packed_soluble"])
                   & T.actin_cortex_annot.astype(bool) & (T.m_PM < 0.05)]
    S = lc.sources(T.loc[cand])
    R6 = set()
    for g in cand:
        if g in CUR: continue
        srcs = ("uniprot",) if g in JUR else ("hpa", "uniprot")
        if any(S[g].get(k, {}).get("q", {}).get("PM", 0) >= 0.999 and S[g].get(k, {}).get("w", 0) > 0 for k in srcs): R6.add(g)
    say(f"\nC: cortical genes {len(R6)} (of {len(cand)} R6 candidates; JURKAT genes via UniProt only)")
    for i in range(len(I)):
        if bool(I.at[i, "transmembrane"]): continue
        gs = set(genes(i)) & R6
        if not gs or not (single(i) or frac_mass(i, R6) >= 0.5): continue
        w = W(i); nw = {}; cx = 0.0
        for k, v in w.items():
            if k == "plasma_membrane": cx += v
            elif k in CYTO_KEYS: cx += 0.5 * v; nw[k] = 0.5 * v
            else: nw[k] = v
        if cx <= 0: continue
        nw["cortex"] = nw.get("cortex", 0) + cx
        setw(i, nw, "C_cortex")

    # ------------------------------------------------------------------ N. nucleolus
    hm = T.hpa_main.fillna(""); rel = T.hpa_reliability.fillna("")
    NU = {}
    for g in T.index[hm.str.contains("Nucleoli") & rel.isin(["Enhanced", "Supported", "Approved"])]:
        if g in JUR or g in CUR: continue
        terms = [t.strip() for t in hm[g].split(";")]
        nuc_other = [t for t in terms if t in ("Nucleoplasm", "Nuclear speckles", "Nuclear bodies", "Nuclear membrane", "Kinetochore",
                                               "Mitotic chromosome")]
        NU[g] = 0.85 if (rel[g] in ("Enhanced", "Supported") and not nuc_other) else 0.7
    say(f"\nN: nucleolar genes {len(NU)} (HPA main Nucleoli*, Approved+, not JURKAT/curated); "
        f"F = 0.85 for {sum(v > 0.8 for v in NU.values())}, 0.7 for the rest")
    nset = set(NU); plan = []
    for i in range(len(I)):
        if bool(I.at[i, "transmembrane"]): continue
        gs = set(genes(i)) & nset
        if not gs: continue
        if not single(i) and frac_mass(i, nset) < 0.5: continue
        w = W(i)
        if w.get("nucleus", 0) <= 0: continue
        f = np.average([NU[g] for g in gs])
        plan.append((i, f, w))
    m_nucl = sum(f * w["nucleus"] * I.at[i, "mass_pg"] for i, f, w in plan)
    dens = m_nucl / VNUCL * 1e3                                  # pg/um3 -> mg/ml
    sc = min(1.0, NUCL_MAX / dens) if dens > 0 else 1.0
    say(f"N: nucleolar protein {m_nucl:.3f} pg in {VNUCL} um3 = {dens:.0f} mg/ml before the cap; scale {sc:.3f} "
        f"(cap {NUCL_MAX} mg/ml; before pre-pass / open-face scaling)")
    for i, f, w in plan:
        nw = dict(w); x = sc * f * w["nucleus"]; nw["nucleus"] = w["nucleus"] - x; nw["nucleolus"] = nw.get("nucleolus", 0) + x
        setw(i, nw, "N_nucleolus")
    out_dens = sum(sc * f * w["nucleus"] * I.at[i, "mass_pg"] for i, f, w in plan) / VNUCL * 1e3

    # ------------------------------------------------------------------ P. nuclear pool (R2b)
    share = VN / (VN + VC)
    cand = T[(T.m_nucleus < 0.05) & (T.m_cytosol >= 0.8) & (T.unit_kda.fillna(T.mw_kda) < 60) & (T.tm_ing_soluble_frac == 0)
             & ~(T.uniprot_tm.fillna(False).astype(bool)) & ~(T.uniprot_sp.fillna(False).astype(bool))].index
    S2 = lc.sources(T.loc[cand])
    R2 = {g for g in cand if any(S2[g].get(k, {}).get("q", {}).get("nucleus", 0) > 0 for k in ("hpa", "uniprot"))
          and g not in JUR and g not in CUR and not ACT_TUB.match(g)}
    say(f"\nP: nuclear-pool genes {len(R2)} (R2b candidates {len(cand)}); nuclear share Vn/(Vn+Vc) = {share:.3f}")
    for i in range(len(I)):
        if not single(i) or genes(i)[0] not in R2 or bool(I.at[i, "transmembrane"]) or I.at[i, "mass_da"] >= 60000: continue
        w = W(i); pool = w.get("cytosol", 0) + w.get("unassigned", 0) + w.get("nucleus", 0)
        tgt = share * pool
        if pool <= 0 or w.get("nucleus", 0) >= tgt: continue
        take = tgt - w.get("nucleus", 0); cy = w.get("cytosol", 0) + w.get("unassigned", 0)
        nw = dict(w)
        for k in ("cytosol", "unassigned"):
            if k in nw: nw[k] = nw[k] * (1 - take / cy)
        nw["nucleus"] = tgt
        setw(i, nw, "P_nuclear_pool")

    # ------------------------------------------------------------------ report
    say("\nrule | ingredients | genes | mass moved (pg; TM routing: ingredient mass)")
    for k, (m, g, ii) in moved.items():
        say(f"  {k} | {len(ii)} | {len(g)} | {m:.3f}")
    say(f"nucleolus: {out_dens:.0f} mg/ml after the cap (inventory, before pre-pass / open-face scaling)")
    for k in ("pm_anchor_cyto", "pm_anchor_exo", "cortex", "lamina", "nucleolus", "er_anchor_cyto", "er_anchor_lumen", "golgi_anchor_cyto",
              "golgi_anchor_lumen", "endo_anchor_cyto", "endo_anchor_lumen", "om_anchor_cyto", "ne_anchor_nuc"):
        m = sum(json.loads(w).get(k, 0) * p for w, p in zip(I.comp_weights, I.mass_pg) if isinstance(w, str))
        if m > 0: say(f"  key {k}: {m:.3f} pg")
    L = pd.DataFrame(log, columns=["ing_id", "gene", "field", "old", "new", "rule"]); L.to_csv(CHG, sep="\t", index=False)
    say(f"\ntotal mass {I0.mass_pg.sum():.4f} -> {I.mass_pg.sum():.4f} pg; {len(L)} field changes in {L.ing_id.nunique()} ingredients -> {CHG}")
    if apply:
        for c in I0.columns:
            if c != "transmembrane": I[c] = I[c].astype(I0[c].dtype)
        I["transmembrane"] = I.transmembrane.astype(bool)
        tmp = ING + ".tmp"; I.to_csv(tmp, sep="\t", index=False)
        if not os.path.exists(BAK):
            shutil.copy2(ING, BAK); say(f"backup written: {BAK}")
        os.replace(tmp, ING); say(f"APPLIED: {ING} rewritten")
    else:
        say("dry run: ingredients_v0.tsv not modified (use --apply)")
    open(REP, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main("--apply" in sys.argv[1:])
