"""Inventory v3: multi-chain assemblies for soluble complexes and homo-oligomers.

1. Complex Portal complexes (all of them, not only the 40 most abundant): highest-resolution RCSB assembly 1 in which
   the subunits present make up >= 80% of subunit mass and >= 60% of their residues are modeled.
   RNA chains are kept when Complex Portal lists an RNA component (snoRNPs, RNase P/MRP, SRP, telomerase), and only
   the complex's own RNA (never rRNA, spacers, tRNA or mRNA of a larger particle the complex was solved in).
   Membrane complexes and membrane homo-oligomers are oriented automatically (membrane_orient.py).
2. Homo-oligomers of the residual soluble monomer pool:
   a. UniProt-annotated n-mers with a PDB assembly 1 that is a homo-n-mer of the full-length protein;
   b. otherwise a predicted AlphaFold DB homodimer (YC_DIMERS/homodimers) if confident:
      ipTM >= 0.5 (the dimer set is already selected for confident interfaces, ipSAE >= 0.6).
All RCSB responses are cached in WORK/rcsb_cache.json, so re-runs are deterministic and offline.
"""
import os, re, json, collections
import numpy as np, pandas as pd, requests, gemmi
import structlib as sl
import build_structs as bs
import extend_complexes as xc
import membrane_orient as mo
from inventory import MITO_SUB
from config import DATA, PDB, WORK, DIMERS, LIB

CACHE_F = f"{WORK}/rcsb_cache.json"
_cache = json.load(open(CACHE_F)) if os.path.exists(CACHE_F) else {}
RNA_TERMS = re.compile(r"URS[0-9A-F]+|CHEBI:13333[56]")          # RNAcentral ids; box C/D, H/ACA snoRNA classes
RNA_NAMES = re.compile(r"snoRNP|ribonucleoprotein|ribonuclease P|ribonuclease MRP|recognition particle|telomerase|snRNP|spliceosom", re.I)
# RNA chains kept inside a complex: the complex's own RNA, never rRNA / spacers / tRNA / mRNA from a larger particle
RNA_KEEP = re.compile(r"sno|snR|\bU\d+\b|RPR1|NME1|SCR1|7SL|SRP RNA|TLC1|telomerase|RNase (P|MRP) RNA|guide", re.I)
RNA_SKIP = re.compile(r"rRNA|\bETS\b|\bITS|tRNA|mRNA|messenger|ribosomal", re.I)
RIBO_SUBUNIT_RNA = [(re.compile(r"large ribosomal subunit", re.I), re.compile(r"\b(21S|25S|5\.8S|5S)\b.*rRNA|large", re.I)),
                    (re.compile(r"small ribosomal subunit", re.I), re.compile(r"\b(15S|18S)\b.*rRNA|small", re.I))]


def rna_chain_belongs(cname, desc, n_rna_entities):
    """Is an RNA entity with this description part of the complex named cname?"""
    for cpat, rpat in RIBO_SUBUNIT_RNA:        # ribosomal subunits keep their own rRNA
        if cpat.search(cname):
            return bool(rpat.search(desc))
    u = re.search(r"\bU(\d+)\b", cname)        # snRNPs keep only the snRNA they are named after
    if u and re.search(r"\bU\d+\b", desc):
        return re.search(rf"\bU{u.group(1)}\b", desc) is not None
    if RNA_SKIP.search(desc):
        return False
    return bool(RNA_KEEP.search(desc)) or n_rna_entities == 1
HOMO_ANNOT_IPTM = 0.5
HOMO_DENOVO_IPTM, HOMO_DENOVO_IPSAE = 0.5, 0.6     # the dimer set is already selected for ipSAE >= 0.6
MIN_COPIES = 20          # do not search structures for very rare proteins
# complexes: subunits present in the assembly >= 80% of subunit mass, and >= 60% of their residues modeled
# (unmodeled loops/termini are disordered; AF models are trimmed the same way at pLDDT < 50)
MIN_RESIDUE_COV = 0.6


def save_cache():
    json.dump(_cache, open(CACHE_F, "w"))


def _cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def entries_with(acc):
    """RCSB entries containing a polymer entity with this UniProt accession."""
    def go():
        q = {"query": {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
            "operator": "exact_match", "value": acc}}, "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 1000}}}
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=60)
        return [x["identifier"] for x in r.json().get("result_set", [])] if r.status_code == 200 and r.text else []
    return _cached(f"acc:{acc}", go)


def entry_info(ids):
    """{id: {uniprot: [...], res: float}} for a batch of entries."""
    need = [i for i in ids if f"entry:{i}" not in _cache]
    for k in range(0, len(need), 50):
        q = ('{ entries(entry_ids: %s) { rcsb_id rcsb_entry_info { resolution_combined } '
             'polymer_entities { rcsb_polymer_entity_container_identifiers { uniprot_ids } } } }') % json.dumps(need[k:k + 50])
        r = requests.post("https://data.rcsb.org/graphql", json={"query": q}, timeout=120).json()
        for e in r["data"]["entries"] or []:
            acc = sorted({a for pe in e["polymer_entities"] for a in (pe["rcsb_polymer_entity_container_identifiers"]["uniprot_ids"] or [])})
            _cache[f"entry:{e['rcsb_id']}"] = dict(uniprot=acc, res=(e["rcsb_entry_info"]["resolution_combined"] or [99.0])[0])
    return {i: _cache.get(f"entry:{i}") for i in ids}


def entity_uniprot(pid):
    return {k: set(v) for k, v in _cached(f"ent:{pid}", lambda: {k: sorted(v) for k, v in xc.entity_uniprot(pid).items()}).items()}


def entity_align(pid):
    """entity_id -> [(UniProt accession, entity begin, UniProt begin, length)] (label_seq -> UniProt numbering)"""
    def go():
        q = ('{ entry(entry_id:"%s"){ polymer_entities { rcsb_polymer_entity_container_identifiers { entity_id } '
             'rcsb_polymer_entity_align { reference_database_name reference_database_accession '
             'aligned_regions { entity_beg_seq_id ref_beg_seq_id length } } } } }') % pid
        r = requests.post("https://data.rcsb.org/graphql", json={"query": q}, timeout=60).json()
        out = {}
        for pe in (r["data"]["entry"]["polymer_entities"] or []):
            eid = pe["rcsb_polymer_entity_container_identifiers"]["entity_id"]
            out[eid] = [(a["reference_database_accession"], g["entity_beg_seq_id"], g["ref_beg_seq_id"], g["length"])
                        for a in (pe["rcsb_polymer_entity_align"] or []) if a["reference_database_name"] == "UniProt"
                        for g in a["aligned_regions"]]
        return out
    return {k: [tuple(x) for x in v] for k, v in _cached(f"align:{pid}", go).items()}


def entity_desc(pid):
    """entity_id -> description (used to decide which RNA chains belong to a complex)"""
    def go():
        q = '{ entry(entry_id:"%s"){ polymer_entities { rcsb_polymer_entity { pdbx_description } rcsb_polymer_entity_container_identifiers { entity_id } } } }' % pid
        r = requests.post("https://data.rcsb.org/graphql", json={"query": q}, timeout=60).json()
        return {pe["rcsb_polymer_entity_container_identifiers"]["entity_id"]: pe["rcsb_polymer_entity"]["pdbx_description"] or ""
                for pe in (r["data"]["entry"]["polymer_entities"] or [])}
    return _cached(f"desc:{pid}", go)


# ------------------------------------------------------------------ 1. complexes
def complex_table(cp, df):
    acc2gene = dict(zip(df.uniprot, df.gene)); ntm = dict(zip(df.gene, df.n_tm))
    rows = []
    for r in cp.itertuples(index=False):
        sub = {}
        for a, s in re.findall(r"([A-Z0-9]{6,10})(?:-\w+)?\((\d+)\)", r[4]):
            g = acc2gene.get(a)
            if g:
                sub[g] = max(1, int(s))
        if len(sub) < 2:
            continue
        rows.append(dict(cpx=r[0], name=r[1], sub=sub, tm=any(ntm.get(g, 0) > 0 for g in sub),
                         rna=bool(RNA_TERMS.search(r[4]) or RNA_NAMES.search(r[1]) or re.search(r"ribosomal subunit", r[1], re.I))))
    return rows


def best_assembly(sub, acc_of, mass, log=print):
    """Highest-resolution assembly 1 covering >= 80% of subunit mass; returns (pid, res, cov, st, psub) or None."""
    hits = collections.Counter()
    for g in sub:
        for e in entries_with(acc_of[g]):
            hits[e] += mass[g] * sub[g]
    cand = [e for e, _ in hits.most_common(40)]
    info = entry_info(cand)
    tot = sum(mass[g] * sub[g] for g in sub)
    scored = []
    for e in cand:
        inf = info.get(e)
        if not inf: continue
        cov = sum(mass[g] * sub[g] for g in sub if acc_of[g] in inf["uniprot"]) / tot
        if cov >= 0.8:
            scored.append((inf["res"], e))
    best = None
    for res, pid in sorted(scored)[:4]:          # verify the best few on the actual assembly file
        fn = xc.fetch_assembly(pid)
        if not fn: continue
        try:
            xc._ENT_CACHE[pid] = entity_uniprot(pid)
            cov, r_, nres, st, psub = xc.coverage(fn, sub, acc_of, mass, pid)
        except Exception as ex:
            log(f"  {pid}: {ex}"); continue
        present = sum(mass[g] * sub[g] for g in psub) / tot
        if present >= 0.8 and cov >= MIN_RESIDUE_COV and nres < 60000:
            best = (pid, res, cov, st, psub); break
    return best


def build_complex(ing_id, cname, pid, st, psub, acc_of, keep_rna, tm=False):
    """Keep the complex's chains (and its own RNA), orient membrane complexes; returns (n_beads, n_nt, n_rna, info)
    or None (with info) when a membrane complex cannot be oriented."""
    st.remove_ligands_and_waters()
    keep_acc = {acc_of[g] for g in psub}
    ea = entity_uniprot(pid); ed = entity_desc(pid) if keep_rna else {}
    n_rna_ent = sum(1 for e in st.entities if e.polymer_type == gemmi.PolymerType.Rna)
    drop, n_rna = [], 0
    for ch in st[0]:
        ent = st.get_entity_of(ch.get_polymer()) if len(ch) else None
        if ent is None:
            drop.append(ch.name); continue
        if keep_rna and ent.polymer_type == gemmi.PolymerType.Rna and rna_chain_belongs(cname, ed.get(ent.name, ""), n_rna_ent):
            n_rna += 1; continue
        if not (ea.get(ent.name, set()) & keep_acc):
            drop.append(ch.name)
    for nm in drop:
        st[0].remove_chain(nm)
    st.remove_empty_chains()
    xyz, kind, acc, pos = mo.beads_with_positions(st, align=entity_align(pid))
    extra, info = {"pdb": pid}, {}
    if tm:
        tmm, ns, ps = mo.masks(acc, pos, xyz)
        xyz, info = mo.orient(xyz, tmm, ns, ps, min_score=mo.min_score_for(acc))
        if xyz is None:
            return None, info
        extra.update(membrane="TM", tm=tmm, memb_score=np.float32(info["score"]))
    else:
        xyz = xyz - xyz.mean(0)
    bs.save(ing_id, xyz, kind, extra=extra, per=30)
    return (len(xyz), int((kind == 1).sum()), n_rna), info


# mtDNA-encoded subunits: hydrophobic, translated in the matrix and strongly under-detected by MS (COB 40, COX3 18,
# ATP8 0 copies in PaxDb vs thousands of their nuclear partners); they do not inform complex copy numbers
MTDNA_ENCODED = {"COB", "COX1", "COX2", "COX3", "ATP6", "ATP8", "OLI1", "VAR1"}


def complex_copies(pool, sub):
    """median over (nuclear-encoded) subunits of pool/stoichiometry, robust to single-subunit MS noise, capped at 2x the
    limiting subunit (25th percentile for >= 4 subunit types) so a scarce subunit is never inflated many-fold"""
    per = [pool.get(g, 0) / s for g, s in sub.items() if g not in MTDNA_ENCODED] or [pool.get(g, 0) / s for g, s in sub.items()]
    lim = min(per) if len(per) < 4 else float(np.percentile(per, 25))
    return float(min(np.median(per), 2 * lim))


def extend_complexes(df, ing, modeled, log=print):
    cp = pd.read_csv(f"{DATA}/complexportal_559292.tsv", sep="\t")
    acc_of = dict(zip(df.gene, df.uniprot)); mass = dict(zip(df.gene, df.mass_da)); ntm = dict(zip(df.gene, df.n_tm))
    pool = {r.ing_id: r.copies_per_cell for r in ing[ing.kind == "protein"].itertuples()}
    rows = complex_table(cp, df)
    for r in rows:
        r["copies"] = complex_copies(pool, r["sub"])
        r["mass_per_cell"] = r["copies"] * sum(mass.get(g, 0) * s for g, s in r["sub"].items())
    rows.sort(key=lambda r: -r["mass_per_cell"])
    chosen, tried, used = [], [], set(modeled)
    for r in rows:
        if set(r["sub"]) & used:
            tried.append((r["cpx"], r["name"], "overlaps modeled complex", None, None)); continue
        if r["copies"] < 1:
            tried.append((r["cpx"], r["name"], "< 1 copy", None, None)); continue
        b = best_assembly(r["sub"], acc_of, mass, log)
        save_cache()
        if b is None:
            tried.append((r["cpx"], r["name"], "no assembly >= 80% coverage", None, None)); continue
        pid, res, cov, st, psub = b
        copies = complex_copies(pool, psub)
        if copies < 1:
            tried.append((r["cpx"], r["name"], "< 1 copy", pid, cov)); continue
        ing_id = r["cpx"].replace("-", "_") + "_" + re.sub(r"\W+", "_", r["name"])[:32]
        tm = any(ntm.get(g, 0) > 0 for g in psub)
        built, info = build_complex(ing_id, r["name"], pid, st, psub, acc_of, r["rna"], tm)
        if built is None:
            tried.append((r["cpx"], r["name"], f"membrane orientation failed ({info.get('fail')})", pid, cov)); continue
        nb_, nnt, n_rna = built
        for g, s in psub.items():
            pool[g] = max(0.0, pool.get(g, 0) - copies * s)
        chosen.append(dict(ing_id=ing_id, cpx=r["cpx"], label=r["name"], sub=psub, copies=copies,
                           mass_da=sum(mass[g] * s for g, s in psub.items()) + nnt * 330.0,
                           pdb=pid, resolution=res, coverage=cov, n_beads=nb_, n_nt=nnt,
                           n_tm=sum(ntm.get(g, 0) * s for g, s in psub.items()), memb_score=info.get("score")))
        used |= set(psub)
        tried.append((r["cpx"], r["name"], "added", pid, cov))
        log(f"{r['cpx']:10s} {r['name'][:42]:42s} {pid} res={res} cov={cov:.2f} copies={copies:.0f} nt={nnt}"
            f"{' membrane ' + str(info) if tm else ''} sub={psub}")
    return chosen, pd.DataFrame(tried, columns=["complex_ac", "name", "outcome", "pdb", "coverage"])


# ------------------------------------------------------------------ 2. homo-oligomers
def homo_assembly(acc, n, length):
    """PDB entry whose assembly 1 is a homo-n-mer of this accession, each chain >= 60% of the full length."""
    def go():
        ids = entries_with(acc)
        info = entry_info(ids)
        for res, pid in sorted((info[i]["res"], i) for i in ids if info.get(i) and info[i]["uniprot"] == [acc]):
            a = requests.get(f"https://data.rcsb.org/rest/v1/core/assembly/{pid}/1", timeout=60)
            if a.status_code != 200: continue
            ai = a.json().get("rcsb_assembly_info", {})
            if ai.get("polymer_entity_count") == 1 and ai.get("polymer_entity_instance_count") == n \
                    and ai.get("modeled_polymer_monomer_count", 0) >= 0.6 * n * length:
                return pid
        return None
    return _cached(f"homo:{acc}:{n}", go)


def af_dimer_table():
    h = pd.read_csv(f"{DIMERS}/homodimers/candidates.csv")
    return {r.uniprotAccession_1: r for r in h.itertuples()}


def build_af_dimer(lib_id, model_id, acc, tm=False):
    """pLDDT < 50 residues trimmed (TM segments always kept, as for monomers); membrane dimers are oriented.
    Returns (n_beads, info) or (None, info) if a membrane dimer cannot be oriented."""
    st = gemmi.read_structure(f"{DIMERS}/homodimers/{model_id}.pdb")
    st.remove_hydrogens()
    xyz, kind, accs, pos = mo.beads_with_positions(st, acc_of_chain={ch.name: acc for ch in st[0]})
    pl = np.array([a.b_iso for ch in st[0] for res in ch for a in [res.find_atom("CA", "*")] if a is not None])
    tmm, ns, ps = mo.masks(accs, pos, xyz)
    keep = (pl >= 50) | tmm
    if keep.sum() < 20:
        keep[:] = True
    extra = dict(src="AF-multimer", model=model_id, nchain=2, plddt=pl[keep].astype(np.float32),
                 plddt_frac_kept=np.float32(keep.mean()))
    info = {}
    if tm:
        out, info = mo.orient(xyz[keep], tmm[keep], ns[keep], ps[keep], min_score=mo.min_score_for(accs))
        if out is None:
            return None, info
        extra.update(membrane="TM", tm=tmm[keep], memb_score=np.float32(info["score"]))
    else:
        out = xyz[keep] - xyz[keep].mean(0)
    bs.save(lib_id, out, np.zeros(len(out), np.int8), extra=extra)
    return len(out), info


def build_pdb_homo(lib_id, pid, n, tm=False):
    st = gemmi.read_structure(xc.fetch_assembly(pid))
    st.remove_ligands_and_waters()
    xyz, kind, acc, pos = mo.beads_with_positions(st, align=entity_align(pid))
    p = kind == 0; xyz, acc, pos = xyz[p], acc[p], pos[p]
    extra, info = dict(pdb=pid, src="PDB", nchain=n), {}
    if tm:
        tmm, ns, ps = mo.masks(acc, pos, xyz)
        xyz, info = mo.orient(xyz, tmm, ns, ps, min_score=mo.min_score_for(acc))
        if xyz is None:
            return None, info
        extra.update(membrane="TM", tm=tmm, memb_score=np.float32(info["score"]))
    else:
        xyz = xyz - xyz.mean(0)
    bs.save(lib_id, xyz, np.zeros(len(xyz), np.int8), extra=extra)
    return len(xyz), info


def homo_oligomers(ing, df, legacy, log=print):
    """legacy: {gene: (lib_id, n)} existing curated PDB homo-oligomers (kept as they are).
    Returns oligo_map {ing_id: (lib_id, n)} and a table of decisions."""
    length = dict(zip(df.uniprot, df.mass_da / 110.0))
    afd = af_dimer_table()
    oligo, rows = dict(legacy), []
    cand = ing[(ing.kind == "protein") & (ing.copies_per_cell >= MIN_COPIES) & ~ing.ing_id.isin(list(legacy))]
    for r in cand.sort_values("copies_per_cell", ascending=False).itertuples():
        n_uni = int(r.oligomer_n); lib_id = f"{r.ing_id}_oligo"; how = None; tm = r.n_tm > 0
        if n_uni > 1:
            pid = homo_assembly(r.uniprot, n_uni, length.get(r.uniprot, 0))
            if pid:
                built, info = build_pdb_homo(lib_id, pid, n_uni, tm)
                if built is not None:
                    oligo[r.ing_id] = (lib_id, n_uni); how = f"PDB {pid} ({n_uni}-mer){' membrane' if tm else ''}"
                else:
                    how = f"PDB {pid} membrane orientation failed ({info.get('fail')})"
        if r.ing_id not in oligo and r.uniprot in afd:
            d = afd[r.uniprot]
            ok = d.ipTM >= HOMO_ANNOT_IPTM if n_uni > 1 else (d.ipTM >= HOMO_DENOVO_IPTM and d.max_ipSAE >= HOMO_DENOVO_IPSAE)
            if ok:
                built, info = build_af_dimer(lib_id, d.modelEntityId, r.uniprot, tm)
                if built is not None:
                    oligo[r.ing_id] = (lib_id, 2); how = f"AF dimer {d.modelEntityId} (ipTM {d.ipTM:.2f}){' membrane' if tm else ''}"
                else:
                    how = f"AF dimer membrane orientation failed ({info.get('fail')})"
            else:
                how = f"AF dimer rejected (ipTM {d.ipTM:.2f}, ipSAE {d.max_ipSAE:.2f})"
        rows.append(dict(ing_id=r.ing_id, uniprot=r.uniprot, copies=r.copies_per_cell, uniprot_oligomer=n_uni,
                         decision=how or "monomer", lib_id=oligo.get(r.ing_id, ("", 0))[0]))
        save_cache()
    return oligo, pd.DataFrame(rows)


# ------------------------------------------------------------------ inventory update
def apply_complexes(ing, df, chosen):
    """Add chosen complexes as ingredients and subtract their subunits from the monomer pool (extend_complexes.apply
    plus: n_tm for membrane complexes, and a generic 'mitochondrion' weight resolved to the mitochondrial
    sub-compartments named by the other subunits, e.g. ATP synthase -> inner membrane only)."""
    ing = ing.copy()
    cw_of = dict(zip(df.gene, df.comp_weights))
    new = []
    for c in chosen:
        w = {}
        for g in c["sub"]:
            for k, v in json.loads(cw_of[g]).items():
                w[k] = w.get(k, 0) + v / len(c["sub"])
        spec = {k: v for k, v in w.items() if k in MITO_SUB}
        if "mitochondrion" in w and spec:
            gen = w.pop("mitochondrion"); z = sum(spec.values())
            for k, v in spec.items():
                w[k] += gen * v / z
        for g, s in c["sub"].items():
            m = (ing.kind == "protein") & (ing.ing_id == g)
            ing.loc[m, "copies_per_cell"] = np.maximum(0, ing.loc[m, "copies_per_cell"] - c["copies"] * s)
        new.append(dict(ing_id=c["ing_id"], label=c["label"], kind="complex", genes=";".join(sorted(c["sub"])),
                        n_subunit_types=len(c["sub"]), copies_per_cell=c["copies"], comp_weights=json.dumps(w),
                        mass_da=c["mass_da"], pdb=c["pdb"], membrane="", oligomer_n=1, uniprot="", n_tm=int(c.get("n_tm", 0))))
    ing = pd.concat([ing, pd.DataFrame(new)], ignore_index=True)
    return ing[(ing.kind != "protein") | (ing.copies_per_cell >= 1)].reset_index(drop=True)


# ------------------------------------------------------------------ beta-barrel monomers
def build_barrel_monomers(df, log=print):
    """AlphaFold monomers of outer-membrane beta-barrels (no TRANSMEM annotation), oriented by their strands and saved
    over the soluble-centred library entries lib/<gene>.npz"""
    import tarfile, gzip
    from config import AFDB_TAR
    want = {a: g for a, g in zip(df.uniprot, df.gene) if a in mo.barrels()}
    data = {}
    with tarfile.open(AFDB_TAR) as tf:
        for m in tf:
            n = os.path.basename(m.name)
            if n.endswith(".cif.gz") and "-F1-" in n and n.split("-")[1] in want:
                data[n.split("-")[1]] = gzip.decompress(tf.extractfile(m).read()).decode()
                if len(data) == len(want):
                    break
    for acc, txt in data.items():
        st = gemmi.make_structure_from_block(gemmi.cif.read_string(txt)[0])
        xyz, kind, accs, pos = mo.beads_with_positions(st, acc_of_chain={ch.name: acc for ch in st[0]})
        pl = np.array([a.b_iso for ch in st[0] for res in ch for a in [res.find_atom("CA", "*")] if a is not None])
        tmm, ns, ps = mo.masks(accs, pos, xyz)
        keep = (pl >= 50) | tmm
        out, info = mo.orient(xyz[keep], tmm[keep], ns[keep], ps[keep], min_score=mo.BARREL_MIN_SCORE)
        if out is None:   # annotated integral: take the best slab even when the score is low
            out, info = mo.orient(xyz[keep], tmm[keep], ns[keep], ps[keep], min_score=0.0, min_in=0.0)
            info["low_confidence"] = True
        bs.save(want[acc], out, np.zeros(len(out), np.int8),
                extra=dict(membrane="TM", tm=tmm[keep], plddt=pl[keep].astype(np.float32), plddt_frac_kept=np.float32(keep.mean()),
                           memb_score=np.float32(info["score"])))
        log(f"barrel {want[acc]}: {info}")
