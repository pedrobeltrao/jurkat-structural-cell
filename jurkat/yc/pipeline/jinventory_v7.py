"""Inventory revision v7 (PLANNING_NOTES "v7 prep"; Pedro 2026-11-10: v7 = these changes + the genome-mapped chromatin,
packed together later).

usage: jinventory_v7.py            dry run: rule tables, data/derived/inventory_v7_changes.tsv, inventory_v7_report.txt
       jinventory_v7.py --apply    as above, then writes data/derived/ingredients_v7.tsv (new file; the v6 inventory
                                   ingredients_v0.tsv is the input and is never modified; snapshot provenance/snapshots/
                                   ingredients_v6.tsv)

Hold-out (DECISION v6, unchanged): OpenCell is never read; for the held-out HPA-JURKAT genes (validation/
jurkat_holdout_genes.txt) HPA terms come from a consensus rebuilt from the per-line annotations of all other lines
(JURKAT and 'uncertain' antibodies dropped). No Tier 2 / JURKAT image metric is used by any v7 rule.

Rules (scope as v6: single-gene ingredients, or complexes / oligomers with >= 50% of the mass from qualifying genes;
TM ingredients only in R4b; v4-curated genes excluded; shares move from the cytoplasmic keys, proportionally):
 R4b TCR/CD3 steady-state pools (complex CPX-6482: TRAC, TRBC2, CD3D, CD3E x2, CD3G, CD247 x2): PM 0.75, endosome 0.17
     (constitutive internalisation and recycling of the TCR:CD3 complex in resting T cells and Jurkat: Liu et al. 2000
     Immunity 13:665, PMID 11114379; recycling-endosome pool), ER 0.06 + Golgi 0.02 (assembly / export pool). Judgement
     values from the literature mechanism, not fitted. Rule input: literature only (no HPA, no images). The free CD3D /
     CD3E monomers (proteomic excess over the complex, ER 1.0 since v0) are unchanged.
 R6c pericentriolar material / centriolar satellites ("pcm" key -> virtual region pcm, a 400 nm sphere around the
     centriole pair at the stage-3 MTOC candidate): HPA main Centrosome / Centriolar satellite / Microtubule organizing
     center AND GO CC centrosome / centriole / pericentriolar material / centriolar satellite / gamma-tubulin complex:
     S_PCM x cytoplasmic pool; all pcm shares scaled to PCM_MAX mg/ml in the cytosolic part of the sphere.
 R7  MT-associated proteins: plus-end trackers (InterPro EB1 C-terminal IPR004953, CAP-Gly IPR000938, TOG IPR034085)
     -> "mt_plus" (band 14-26 nm from the MT axis in the last 400 nm of MTs >= 1 um), S_PLUS x cytoplasmic pool; lattice
     MAPs (MAP tubulin-binding repeat IPR001084, MAP7 IPR008604, doublecortin IPR003533) -> "mt_lattice" (the band along
     all MTs), S_LAT; both need HPA (main/additional) Microtubules / Microtubule ends / Centrosome / Mitotic spindle /
     Cytokinetic bridge or an inventory 'cytoskeleton' share >= 0.2. Capped at MT_MAX mg/ml in the band volumes.
 R6d vimentin cage: cytoplasmic intermediate filaments (InterPro IF rod IPR039008, HPA Intermediate filaments main;
     nuclear lamins excluded): S_VIM x cytoplasmic pool -> "vim_cage" (cytosol within 300 nm of the ONM). Lymphocyte
     vimentin forms a perinuclear cage (Brown et al. 2001 J Immunol 166:6640). Filaments are not built.
 R6e NPC central-channel docking ("npc_fg" key -> shell sh_NPC_fg, centre within 25 nm of a pore axis, pore channel
     and the axial cytoplasmic / basket zone): nuclear import/export receptors (GO MF as v6 R6a): N_FG receptors per
     pore in total, split by copies, taken from their v6 "npc" share (v6 placed 99% on the nuclear face).
Lamina (lattice pre-pass jlamina.py), cisternal Golgi (jgolgi.py), ENDO anchored band (jpack_test ENDO_AH) and the
cytoskeleton pre-pass (jcytoskel_v7.py) are not inventory changes; they are logged in the v7 change table as
non-inventory rows (field = 'prepass' / 'geometry' / 'packer') so that the per-gene history is complete."""
import os, re, sys, json, shutil, numpy as np, pandas as pd

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"; VAL = f"{D}/validation"; REF = f"{V}/data/reference"
SRC = f"{D}/ingredients_v0.tsv"; OUT = f"{D}/ingredients_v7.tsv"; SNAP = f"{D}/provenance/snapshots/ingredients_v6.tsv"
CHG = f"{D}/inventory_v7_changes.tsv"; REP = f"{D}/inventory_v7_report.txt"; GENES_T = f"{D}/v7/inventory_v7_rule_genes.tsv"
VER = "v7"
CYTO_KEYS = {"cytosol", "cytoskeleton", "peroxisome", "lipid_droplet", "unassigned", "plasma_membrane"}
SP_KEYS = {"er", "golgi", "endolysosomal", "secreted"}
SPERM = {"Acrosome", "Annulus", "Calyx", "Connecting piece", "End piece", "Equatorial segment", "Flagellar centriole", "Mid piece",
         "Perinuclear theca", "Principal piece"}
TCR = "CPX-6482"
TCR_W = {"plasma_membrane": 0.75, "endolysosomal": 0.17, "er": 0.06, "golgi": 0.02}
S_PCM, PCM_MAX = 0.2, 100.0          # share; mg/ml cap in the PCM sphere (cytosolic part)
R_PCM = 400.0
S_PLUS, S_LAT, MT_MAX = 0.15, 0.4, 150.0
S_VIM = 0.7
N_FG = 20                            # receptors per pore in the central channel (order of tens per NPC: Kapinos et al.
                                     # 2017 Biophys J 112:1215; Lowe et al. 2015 eLife 4:e04052); judgement
N_PORES = 1000
PCM_T = {"Centrosome", "Centriolar satellite", "Microtubule organizing center"}
PCM_CC = re.compile(r"centrosome \[GO:0005813\]|centriole \[GO:0005814\]|pericentriolar material \[GO:0000242\]|"
                    r"centriolar satellite \[GO:0034451\]|gamma-tubulin (ring |small )?complex")
PLUS_IPR = {"IPR004953": "EB1 C-terminal", "IPR000938": "CAP-Gly", "IPR034085": "TOG"}
LAT_IPR = {"IPR001084": "MAP tubulin-binding repeat", "IPR008604": "MAP7", "IPR003533": "doublecortin"}
MT_T = {"Microtubules", "Microtubule ends", "Centrosome", "Mitotic spindle", "Cytokinetic bridge"}
IF_IPR = "IPR039008"
NPC_MF = re.compile(r"nuclear (import|export) signal receptor activity|nuclear localization sequence binding \[GO:0008139\]")
CHECK = {"R4b": "v6 held-out disagreement (HPA-JURKAT CD3E/CD3D ER-dominant vs model PM rim); rule input literature only: "
                "CD3E/CD3D JURKAT scores are no longer clean after v7",
         "R6c": "v6 plan R6 (centrosome / PCM / centriolar satellites ~260 genes, no compartment) + HPA JURKAT tubulin asters",
         "R7": "v6 plan R7 (MAPs on traced MTs; stage 3 MT network)",
         "R6d": "v6 plan R6 (vimentin perinuclear cage, no compartment)",
         "L3": "v7 prep: lamina lattice holds 43% of the lamins; DECISION Pedro 2026-11-10 (rest = nucleoplasmic pool)",
         "F1": "v7 test pack: CPX-2511 (SSU processome) placed as an NE TM ingredient, blocked by the lamina",
         "R6e": "v6 pack (sh_NPC 99% on the nuclear face; cytoplasmic annulus blocked by the 7R5K ring)"}
AV = 6.022e23


def main(apply):
    os.makedirs(f"{D}/v7", exist_ok=True)
    I = pd.read_csv(SRC, sep="\t"); I0 = I.copy()
    H = pd.read_csv(f"{REF}/hpa_subcellular/subcellular_location.tsv", sep="\t").drop_duplicates("Gene name").set_index("Gene name")
    C = pd.read_csv(f"{VAL}/hpa_celllines.tsv", sep="\t", keep_default_na=False)
    U = pd.read_csv(f"{REF}/uniprot_human_domains.tsv.gz", sep="\t").set_index("Entry")
    inv = pd.read_csv(f"{D}/inventory_v0.tsv", sep="\t").drop_duplicates("gene").set_index("gene")
    gmass = inv.mass_da.to_dict(); gacc = inv.uniprot.to_dict()
    rd = lambda f: {l.strip() for l in open(f) if l.strip() and not l.startswith("#")}
    HOLD = rd(f"{VAL}/jurkat_holdout_genes.txt")
    ch = pd.read_csv(f"{D}/inventory_v4_changes.tsv", sep="\t")
    cur = ch[ch.field.isin(["comp_weights", "transmembrane"]) & ch.reason.astype(str).str.match(r'^"?[bcf]')]
    CUR = {g for x in cur.gene.astype(str) for g in x.split(";")}
    NORM = {t.lower(): t for col in ("Main location", "Additional location") for x in H[col].dropna() for t in x.split(";")}
    out, log, moved, grow = [], [], {}, []

    def say(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

    HC = {}
    def hc(g):
        if g in HC: return HC[g]
        if g in HOLD:
            sub = C[(C.gene == g) & (C.cell_line.str.upper() != "JURKAT") & (C.ab_reliability != "uncertain") & (C.locations != "")]
            if not len(sub): r = (set(), set(), "non-JURKAT per-line consensus (none)")
            else:
                locs = [set(x.lower().split(";")) for x in sub.locations]
                fr = {t: sum(t in l for l in locs) / len(locs) for t in set().union(*locs)}
                fx = lambda t: NORM.get(t.lower(), t)
                r = ({fx(t) for t, f in fr.items() if f >= 0.5} - SPERM, {fx(t) for t, f in fr.items() if 0.25 <= f < 0.5} - SPERM,
                     f"non-JURKAT per-line consensus ({len(sub)})")
        elif g in H.index:
            h = H.loc[g]; sp = lambda x: set(str(x).split(";")) - {"nan", ""} - SPERM
            r = (sp(h["Main location"]), sp(h["Additional location"]), "HPA consensus")
        else: r = (set(), set(), "none")
        HC[g] = r; return r

    FE = {}
    def fe(g):
        if g in FE: return FE[g]
        a = gacc.get(g); r = None
        if isinstance(a, str) and a in U.index:
            x = U.loc[a]; x = x.iloc[0] if isinstance(x, pd.DataFrame) else x
            r = dict(ipr=set(str(x["InterPro"]).rstrip(";").split(";")) if isinstance(x["InterPro"], str) else set(),
                     mf=str(x["Gene Ontology (molecular function)"]), cc=str(x["Gene Ontology (cellular component)"]), kw=str(x["Keywords"]))
        FE[g] = r; return r

    def Wt(i): return json.loads(I.at[i, "comp_weights"]) if isinstance(I.at[i, "comp_weights"], str) else {}
    def genes(i): return str(I.at[i, "genes"]).split(";")
    def stoich(i): return [float(x) for x in str(I.at[i, "stoich"]).split(";")]
    def frac_mass(i, S):
        gs, st = genes(i), stoich(i); t = sum(gmass.get(g, 0) * s for g, s in zip(gs, st))
        return sum((g in S) * gmass.get(g, 0) * s for g, s in zip(gs, st)) / t if t > 0 else 0.0
    def cyto_keys(i, w):
        sp = I.at[i, "signal_frac"] >= 0.5
        return [k for k in w if k in CYTO_KEYS or (k in SP_KEYS and not sp)]
    def wjson(w): return json.dumps({k: round(float(v), 4) for k, v in sorted(w.items(), key=lambda kv: -kv[1]) if v >= 1e-4})
    EV = {}

    def setw(i, nw, rule, gs):
        ow = Wt(i); d = 0.5 * sum(abs(nw.get(k, 0) - ow.get(k, 0)) for k in set(nw) | set(ow))
        if d < 1e-4: return False
        r = moved.setdefault(rule, [0.0, set(), set()]); r[0] += d * I.at[i, "mass_pg"]; r[1] |= set(gs); r[2].add(I.at[i, "ing_id"])
        log.append(dict(version=VER, gene=I.at[i, "genes"], ingredient=I.at[i, "ing_id"], rule_id=rule, field="comp_weights",
                        old_value=I.at[i, "comp_weights"], new_value=wjson(nw),
                        evidence=" | ".join(f"{g}: {EV.get((rule, g), '')}" for g in sorted(gs)),
                        motivating_check=CHECK[rule.split("_")[0]], mass_moved_pg=round(d * I.at[i, "mass_pg"], 6)))
        I.at[i, "comp_weights"] = wjson(nw); return True

    def targets(S):
        for i in range(len(I)):
            if bool(I.at[i, "transmembrane"]): continue
            gs = set(genes(i)) & S
            if not gs: continue
            if len(set(genes(i))) == 1 or frac_mass(i, S) >= 0.5: yield i, gs

    def move(i, key, share, rule, gs):
        w = Wt(i); ck = cyto_keys(i, w); pool = sum(w[k] for k in ck)
        if pool <= 0 or share <= 0: return 0.0
        x = share * pool; nw = dict(w)
        for k in ck: nw[k] = w[k] * (1 - share)
        nw[key] = nw.get(key, 0) + x
        return x if setw(i, nw, rule, gs) else 0.0

    model_genes = sorted({g for i in range(len(I)) for g in genes(i)})
    say(f"source {SRC} ({len(I)} rows, {I.mass_pg.sum():.3f} pg); JURKAT hold-out {len(HOLD)} genes (non-JURKAT consensus)")

    # ---------------------------------------------------------------- R4b TCR/CD3
    k = I.index[I.ing_id == TCR]
    if len(k):
        i = k[0]; gs = set(genes(i))
        for g in gs: EV[("R4b", g)] = "TCR:CD3 complex member; literature recycling/ER-assembly pools (Liu et al. 2000 Immunity 13:665)"
        setw(i, dict(TCR_W), "R4b", gs)
        say(f"R4b {TCR}: {I0.at[i, 'comp_weights']} -> {I.at[i, 'comp_weights']} ({I.at[i, 'copies']:,} copies)")

    # ---------------------------------------------------------------- R6c PCM
    pcm = []
    for g in model_genes:
        if g in CUR: continue
        m, a, src = hc(g); f = fe(g)
        if not (m & PCM_T) or f is None or not PCM_CC.search(f["cc"]): continue
        pcm.append(g); EV[("R6c", g)] = f"HPA main {';'.join(sorted(m & PCM_T))} ({src}); GO CC {PCM_CC.search(f['cc']).group(0)}"
    req = {}
    for i, gs in targets(set(pcm)):
        w = Wt(i); pool = sum(w[k] for k in cyto_keys(i, w)); req[i] = (gs, S_PCM * pool * I.at[i, "copies"] * I.at[i, "mass_da"])
    vol_pcm = pcm_volume()
    mass_pcm = sum(v[1] for v in req.values()) / AV * 1e12
    cap = PCM_MAX * vol_pcm * 1e-3                            # mg/ml x um3 -> pg (1 mg/ml = 1e-3 pg/um3)
    sc = min(1.0, cap / mass_pcm) if mass_pcm > 0 else 1.0
    for i, (gs, _) in req.items(): move(i, "pcm", S_PCM * sc, "R6c", gs)
    say(f"R6c PCM: {len(pcm)} genes, {len(req)} ingredients; requested {mass_pcm:.4f} pg in {vol_pcm:.3f} um3 cytosolic PCM sphere "
        f"-> scale {sc:.3f} (cap {PCM_MAX} mg/ml)")
    grow += [dict(gene=g, rule="R6c", evidence=EV[("R6c", g)]) for g in pcm]

    # ---------------------------------------------------------------- R7 MAPs
    plus, latt = [], []
    for g in model_genes:
        if g in CUR or re.match(r"^(TUB[AB]|TBC[A-E]$|DNAAF)", g): continue   # tubulin-folding cofactors / dynein assembly
                                                                           # factors: CAP-Gly / TOG-like but not MT-bound
        f = fe(g)
        if f is None: continue
        m, a, src = hc(g)
        loc_ok = bool((m | a) & MT_T)
        ii = [i for i in I.index[I.genes == g]]
        cs = max([Wt(i).get("cytoskeleton", 0) for i in ii] or [0])
        if not (loc_ok or cs >= 0.2): continue
        why = f"HPA {';'.join(sorted((m | a) & MT_T))} ({src})" if loc_ok else f"inventory cytoskeleton share {cs:.2f}"
        dp = [PLUS_IPR[x] for x in f["ipr"] if x in PLUS_IPR]; dl = [LAT_IPR[x] for x in f["ipr"] if x in LAT_IPR]
        if dp: plus.append(g); EV[("R7_plus", g)] = f"InterPro {', '.join(dp)}; {why}"
        elif dl: latt.append(g); EV[("R7_lattice", g)] = f"InterPro {', '.join(dl)}; {why}"
    ctx = mt_volumes()
    for key, gl, s0, rule, vol in (("mt_plus", plus, S_PLUS, "R7_plus", ctx["plus_um3"]), ("mt_lattice", latt, S_LAT, "R7_lattice", ctx["lat_um3"])):
        req = {}
        for i, gs in targets(set(gl)):
            w = Wt(i); pool = sum(w[k] for k in cyto_keys(i, w)); req[i] = (gs, s0 * pool * I.at[i, "copies"] * I.at[i, "mass_da"])
        mass = sum(v[1] for v in req.values()) / AV * 1e12; cap = MT_MAX * vol * 1e-3
        sc = min(1.0, cap / mass) if mass > 0 else 1.0
        for i, (gs, _) in req.items(): move(i, key, s0 * sc, rule, gs)
        say(f"{rule}: {len(gl)} genes ({', '.join(gl[:12])}{' ...' if len(gl) > 12 else ''}), {len(req)} ingredients; requested {mass:.4f} pg "
            f"in {vol:.3f} um3 -> scale {sc:.3f}")
        grow += [dict(gene=g, rule=rule, evidence=EV[(rule, g)]) for g in gl]

    # ---------------------------------------------------------------- R6d vimentin
    vim = []
    for g in model_genes:
        f = fe(g)
        if f is None or IF_IPR not in f["ipr"] or g.startswith("LMN") or g in CUR: continue
        m, a, src = hc(g)
        if "Intermediate filaments" not in m: continue
        vim.append(g); EV[("R6d", g)] = f"InterPro IF rod; HPA main Intermediate filaments ({src})"
    for i, gs in targets(set(vim)): move(i, "vim_cage", S_VIM, "R6d", gs)
    say(f"R6d vimentin cage: {vim}")
    grow += [dict(gene=g, rule="R6d", evidence=EV[("R6d", g)]) for g in vim]

    # ---------------------------------------------------------------- R6e NPC FG docking
    rec = [g for g in model_genes if fe(g) is not None and NPC_MF.search(fe(g)["mf"])]
    tl = [(i, gs) for i, gs in targets(set(rec)) if Wt(i).get("npc", 0) > 0]
    tot = sum(I.at[i, "copies"] for i, _ in tl); want = N_FG * N_PORES
    for i, gs in tl:
        w = Wt(i); x = min(w["npc"], want / tot)               # fraction of copies to the channel
        nw = dict(w); nw["npc"] = w["npc"] - x; nw["npc_fg"] = nw.get("npc_fg", 0) + x
        for g in gs: EV[("R6e", g)] = f"GO MF nuclear import/export receptor; {N_FG} receptors per pore in total, split by copies"
        setw(i, nw, "R6e", gs)
    say(f"R6e NPC channel: {len(tl)} receptor ingredients, {tot:,} copies, {want:,} to the channel ({want / max(tot, 1):.4f} of copies)")
    grow += [dict(gene=g, rule="R6e", evidence=EV.get(("R6e", g), "")) for g in rec]

    # ---------------------------------------------------------------- L3 nucleoplasmic lamin pool (DECISION Pedro 2026-11-10)
    # LMNB1 stays at 2.06 M copies; the lamins the lattice pre-pass (jlamina.py) cannot hold go to the nucleoplasm instead of
    # the random lamina shell: lamina share = lattice copies / copies, the rest of the old lamina share -> nucleus. jwc takes
    # the lattice consumption out of the lamina key (prepass json consumed_key), so the random lamina shell is ~empty.
    lj0 = json.load(open(f"{D}/prepass_v7/lamina_cell1.json"))
    for g, P_ in lj0["consumed"].items():
        k = I.index[I.ing_id == g]
        if not len(k): continue
        i = k[0]; w = Wt(i); C_ = float(I.at[i, "copies"]); wl = w.get("lamina", 0.0); s_ = min(wl, P_ / C_)
        nw = dict(w); nw["lamina"] = s_; nw["nucleus"] = nw.get("nucleus", 0.0) + (wl - s_)
        EV[("L3", g)] = f"lattice holds {P_:,} of {int(round(wl * C_)):,} lamina copies (jlamina.py); rest -> nucleoplasmic lamin pool"
        setw(i, nw, "L3", {g})
    say(f"L3 lamin pool: {lj0['consumed']} in the lattice; rest -> nucleus")
    # ---------------------------------------------------------------- F1 CPX-2511 (SSU processome) compartment
    # v0 typed it as a TM ingredient (1 member TM annotation) on the NE; the SSU processome is nucleolar. Nucleolar share x0.338
    # as every v5 nucleolar share (110 mg/ml cap, rule N), rest nucleoplasm.
    k = I.index[I.ing_id == "CPX-2511"]
    if len(k):
        i = k[0]; oldtm = I.at[i, "transmembrane"]
        EV[("F1", "CPX-2511")] = "SSU processome (Complex Portal CPX-2511): nucleolus; v5 nucleolar cap factor 0.338"
        setw(i, {"nucleolus": 0.338, "nucleus": 0.662}, "F1", {"CPX-2511"})
        I.at[i, "transmembrane"] = False; I.at[i, "n_tm"] = 0
        log.append(dict(version=VER, gene=I.at[i, "genes"], ingredient="CPX-2511", rule_id="F1", field="transmembrane", old_value=oldtm,
                        new_value=False, evidence=EV[("F1", "CPX-2511")], motivating_check=CHECK["F1"], mass_moved_pg=0.0))
        say("F1 CPX-2511: -> nucleolus 0.338 / nucleus 0.662, transmembrane False")

    # ---------------------------------------------------------------- N8 nucleosome copies = placed genome-mapped chromatin
    # (coordinator DECISION 2026-11-10): the inventory had 31 M octamers (6.2 Gbp / 200 bp); the v8 chromatin places
    # 28.19 M on 5.935 Gbp of GRCh38 2n DNA (karyotype deletions applied) at 210.5 bp per nucleosome (NRL 195/208/210 for
    # A/B/H, NFRs, Pol II depletion). The 2.81 M gap is documented here, not kept as 'used' histones.
    cj8 = os.environ.get("V7_CHROMATIN_JSON", f"{D}/chromatin/v8/prepass/chromatin_cell1.json")
    if os.path.exists(cj8):
        n8 = int(json.load(open(cj8))["n_nucleosomes"]); k = I.index[I.ing_id == "NUCLEOSOME"]
        if len(k) and n8 > 0:
            i = k[0]; old = int(I.at[i, "copies"])
            I.at[i, "copies"] = n8; I.at[i, "mass_pg"] = n8 * I.at[i, "mass_da"] / AV * 1e12
            log.append(dict(version=VER, gene=I.at[i, "genes"], ingredient="NUCLEOSOME", rule_id="N8", field="copies", old_value=old, new_value=n8,
                            evidence=f"genome-mapped chromatin v8 places {n8:,} nucleosomes (5.935 Gbp 2n with the Jurkat deletions, "
                                     f"210.5 bp/nucleosome); gap {old - n8:,} = 6.2 Gbp/200 bp estimate vs GRCh38 + NRL 203 + NFRs",
                            motivating_check="chromatin v8 report (inventory 31 M vs 28.19 M placed); coordinator DECISION",
                            mass_moved_pg=round((old - n8) * I.at[i, "mass_da"] / AV * 1e12, 4)))
            say(f"N8 NUCLEOSOME copies {old:,} -> {n8:,} ({(old - n8) * I.at[i, 'mass_da'] / AV * 1e12:.2f} pg less)")

    # ---------------------------------------------------------------- non-inventory rows (per-gene history)
    nonint = []
    def nrow(gene, ing, rule, field, old, new, ev, chk, m=0.0):
        nonint.append(dict(version=VER, gene=gene, ingredient=ing, rule_id=rule, field=field, old_value=old, new_value=new,
                           evidence=ev, motivating_check=chk, mass_moved_pg=m))
    P7 = f"{D}/prepass_v7"
    cj = json.load(open(f"{P7}/cytoskeleton_cell1.json")) if os.path.exists(f"{P7}/cytoskeleton_cell1.json") else {}
    c6 = json.load(open(f"{D}/prepass/cytoskeleton_cell1.json"))
    if cj:
        nrow("TUBA1B;TUBB4B", "TUBA1B-TUBB4B", "MT3b", "prepass", f"consumed {c6['consumed']['TUBA1B-TUBB4B']:,} (procedural MTs + centrioles)",
             f"consumed {cj['consumed']['TUBA1B-TUBB4B']:,} (stage-3 b MTs + centrioles at the MTOC candidate)",
             "cytoskel/stage3 variant b (EM-scaffolded, 1.456 mm); DECISION stage 3 -> v7", "stage 3 report: old pre-pass radial, 2.3 mm; HPA aster ratio 3.14 vs 2.20")
        nrow("ACTB", "ACTB", "FA_trim", "prepass", f"consumed {c6['consumed']['ACTB']:,}", f"consumed {cj['consumed']['ACTB']:,}",
             "F-actin subunits clashing with the new MTs / centrioles or the v7 geometry removed", "stage 3 clash counts (16k subunits within 16 nm of the new axes)")
    lj = json.load(open(f"{P7}/lamina_cell1.json")) if os.path.exists(f"{P7}/lamina_cell1.json") else {}
    for g in ["LMNB1", "LMNB2", "LMNA"]:
        if lj: nrow(g, g, "L2_lattice", "prepass", "random lamina shell (v6 0.68 placed)", f"lattice pre-pass {lj['consumed'][g]:,} of {lj['target'][g]:,} (rest: L3 nucleoplasm)",
                    "jlamina.py two-layer rod lattice under the INM (Turgay et al. 2017)", "v6 pack: lamina shell 0.683 placed")
    nrow("(Golgi TM + Golgi shells)", "*", "R5b_cisternae", "geometry", "EM blob, area factor 4.5",
         "juxta-centrosomal procedural Golgi (jgolgi_ribbon.py) with cisternal layers (jgolgi.py); factor = 4.5 x EM blob area / measured",
         "Ladinsky et al. 1999 cisternal spacing; DECISION Pedro 2026-11-10 (Golgi around the centrosome)",
         "v6 plan R5 + v7 prep (EM blob 6.4 um from the MTOC, overlapping mito/ER)")
    nrow("(ENDO anchored)", "*", "E_band", "packer", "sh_ENDO_cyto BR + 2 nm", "BR + 8 nm", "Rab hypervariable C-terminal linker",
         "v6 pack: ENDO-cyto 0.747 placed")

    # ---------------------------------------------------------------- outputs
    L = pd.DataFrame(log + nonint)
    L.to_csv(CHG, sep="\t", index=False)
    pd.DataFrame(grow).to_csv(GENES_T, sep="\t", index=False)
    say("\nper rule (ingredients / genes / mass moved pg):")
    for r, (m, g, ing) in moved.items(): say(f"  {r}: {len(ing)} / {len(g)} / {m:.4f}")
    say(f"mass check (N8 changes the nucleosome copies): {I0.mass_pg.sum():.4f} -> {I.mass_pg.sum():.4f} pg; field changes {len(log)}; non-inventory rows {len(nonint)}")
    tot_key = lambda X, key: sum(json.loads(w).get(key, 0) * c * m for w, c, m in zip(X.comp_weights, X.copies, X.mass_da)) / AV * 1e12
    for key in ("pcm", "mt_plus", "mt_lattice", "vim_cage", "npc_fg", "npc"):
        say(f"  key {key}: {tot_key(I0, key):.4f} -> {tot_key(I, key):.4f} pg")
    if apply:
        os.makedirs(os.path.dirname(SNAP), exist_ok=True)
        if not os.path.exists(SNAP): shutil.copy(SRC, SNAP)
        I.to_csv(OUT, sep="\t", index=False); say(f"written {OUT}")
    open(REP, "w").write("\n".join(out) + "\n")


def pcm_volume(n=400_000):
    """cytosolic volume (um3) of the PCM sphere (Monte Carlo on the v7 geometry)"""
    sys.path.insert(0, f"{V}/jurkat/yc/pipeline"); import cell3d as c3
    z = np.load(f"{D}/jgeom_cell1_v7.npz"); cell = c3.Cell(z["G"], z["pores"])
    J = json.load(open(f"{D}/prepass_v7/cytoskeleton_cell1.json")); cen = np.array(J["log"]["centrioles"]["new_centre_nm"])
    rng = np.random.default_rng(3); u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    P = cen + u * R_PCM * np.cbrt(rng.random((n, 1)))
    return float((cell.labels(P) == c3.CYTO).mean() * 4 / 3 * np.pi * (R_PCM / 1e3) ** 3)


def mt_volumes():
    """band volumes (um3) of mt_lattice / mt_plus from the centerline lengths (annulus 14-26 nm; no membrane correction)"""
    sys.path.insert(0, f"{V}/jurkat/yc/pipeline"); import jvirtual as jv
    ctx = jv.load_context(f"{D}/cytoskel/stage3/mt_stage3_b_centerlines.npz", f"{D}/prepass_v7/cytoskeleton_cell1.json")
    ann = np.pi * (jv.BAND[1] ** 2 - jv.BAND[0] ** 2) / 1e6          # um2
    L_all = float(np.linalg.norm(ctx["ax"].B - ctx["ax"].A, axis=1).sum()) / 1e3
    return dict(lat_um3=ann * L_all, plus_um3=ann * ctx["plus_um"], L_um=L_all, plus_um=ctx["plus_um"], n_plus=ctx["n_plus"])


if __name__ == "__main__":
    main("--apply" in sys.argv)
