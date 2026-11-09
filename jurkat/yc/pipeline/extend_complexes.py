"""Inventory v2: add abundant soluble Complex Portal complexes that have an S. cerevisiae PDB assembly
covering >=80% of subunit mass. Membrane complexes (any TM subunit) are skipped because their
multi-chain membrane orientation is not reliably derivable automatically."""
import re, os, json, gzip
import numpy as np, pandas as pd, requests, gemmi
import structlib as sl
import build_structs as bs

from config import DATA as D, PDB


def candidates(cp, df, modeled_genes, top=40):
    acc2gene = dict(zip(df.uniprot, df.gene)); pool = dict(zip(df.gene, df.copies_per_cell))
    mass = dict(zip(df.gene, df.mass_da)); ntm = dict(zip(df.gene, df.n_tm))
    rows = []
    for r in cp.itertuples(index=False):
        toks = re.findall(r"([A-Z0-9]{6,10})(?:-\w+)?\((\d+)\)", r[4])
        sub = {}
        for a, s in toks:
            g = acc2gene.get(a)
            if g:
                sub[g] = max(1, int(s))
        if len(sub) < 2 or set(sub) & modeled_genes:
            continue
        if any(ntm.get(g, 0) > 0 for g in sub):
            continue
        copies = float(np.median([pool.get(g, 0) / s for g, s in sub.items()]))
        m = sum(mass.get(g, 0) * s for g, s in sub.items())
        pdbs = re.findall(r"wwpdb:(\w{4})", str(r[8]))
        rows.append(dict(cpx=r[0], name=r[1], sub=sub, copies=copies, mass_da=m, mass_per_cell=copies * m, pdbs=pdbs))
    c = pd.DataFrame(rows).sort_values("mass_per_cell", ascending=False)
    return c.head(top)


def rcsb_search(accs, max_n=25):
    """S. cerevisiae entries containing all given UniProt accessions (best resolution first)."""
    nodes = [{"type": "terminal", "service": "text", "parameters": {
        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
        "operator": "exact_match", "value": a}} for a in accs]
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": nodes},
         "return_type": "entry", "request_options": {"paginate": {"start": 0, "rows": max_n},
         "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}]}}
    try:
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=60)
        return [x["identifier"] for x in r.json().get("result_set", [])] if r.status_code == 200 else []
    except Exception:
        return []


def fetch_assembly(pid):
    fn = f"{PDB}/{pid}-assembly1.cif.gz"
    if not os.path.exists(fn):
        r = requests.get(f"https://files.rcsb.org/download/{pid}-assembly1.cif.gz", timeout=120)
        if r.status_code != 200:
            return None
        open(fn, "wb").write(r.content)
    return fn


_ENT_CACHE = {}


def entity_uniprot(pid):
    """entity_id -> set(UniProt) from the RCSB GraphQL API (assembly files lack _struct_ref)."""
    if pid in _ENT_CACHE:
        return _ENT_CACHE[pid]
    q = '{ entry(entry_id:"%s"){ polymer_entities { rcsb_polymer_entity_container_identifiers { entity_id uniprot_ids } } } }' % pid
    out = {}
    try:
        r = requests.post("https://data.rcsb.org/graphql", json={"query": q}, timeout=60).json()
        for pe in (r["data"]["entry"]["polymer_entities"] or []):
            ci = pe["rcsb_polymer_entity_container_identifiers"]
            out[ci["entity_id"]] = set(ci.get("uniprot_ids") or [])
    except Exception:
        pass
    _ENT_CACHE[pid] = out
    return out


def coverage(fn, sub, acc_of, mass, pid=None):
    st = gemmi.read_structure(fn); st.setup_entities()
    b = gemmi.cif.read(fn).sole_block()
    ent_acc = entity_uniprot(pid or os.path.basename(fn)[:4].upper())
    present, obs = {}, {}
    nres = 0
    for ch in st[0]:
        for span in ch.subchains():
            ent = st.get_entity_of(span)
            if ent is None or ent.entity_type != gemmi.EntityType.Polymer:
                continue
            nres += len(span)
            for a in ent_acc.get(ent.name, ()):
                present[a] = present.get(a, 0) + 1
                obs[a] = max(obs.get(a, 0), len(span))
    # coverage = mass fraction of subunit *types* present (Complex Portal stoichiometries are often
    # unknown/0); the stoichiometry actually used is taken from the PDB assembly itself
    tot = sum(mass[g] for g in sub)
    # residue-level: observed residues of the best chain / sequence length (mass/110 Da)
    cov = sum(mass[g] * min(1.0, obs.get(acc_of[g], 0) / (mass[g] / 110.0)) for g in sub) / tot
    pdb_sub = {g: present[acc_of[g]] for g in sub if present.get(acc_of[g], 0) > 0}
    res = None
    for tag in ["_refine.ls_d_res_high", "_em_3d_reconstruction.resolution", "_reflns.d_resolution_high"]:
        v = b.find_value(tag)
        try:
            res = float(v); break
        except (TypeError, ValueError):
            pass
    return cov, res, nres, st, pdb_sub


def run(df, modeled_genes, log=print):
    cp = pd.read_csv(f"{D}/complexportal_559292.tsv", sep="\t")
    acc_of = dict(zip(df.gene, df.uniprot)); mass = dict(zip(df.gene, df.mass_da))
    cand = candidates(cp, df, modeled_genes)
    chosen, tried, used = [], [], set(modeled_genes)
    for c in cand.itertuples():
        if set(c.sub) & used:
            tried.append((c.cpx, c.name, "overlaps chosen complex", None, None)); continue
        ids = list(dict.fromkeys([p.upper() for p in c.pdbs] + rcsb_search([acc_of[g] for g in c.sub])))[:12]
        best = None
        for pid in ids:
            fn = fetch_assembly(pid)
            if not fn:
                continue
            try:
                cov, res, nres, st, psub = coverage(fn, c.sub, acc_of, mass, pid)
            except Exception as e:
                continue
            if cov >= 0.8 and nres < 60000 and (best is None or (res or 99) < (best[1] or 99)):
                best = (pid, res, cov, st, psub)
        if best is None:
            tried.append((c.cpx, c.name, "no PDB assembly with >=80% subunit-mass coverage", None, None)); continue
        pid, res, cov, st, psub = best
        pool = dict(zip(df.gene, df.copies_per_cell))
        copies = float(np.median([pool.get(g, 0) / s for g, s in psub.items()]))
        mda = sum(mass[g] * s for g, s in psub.items())
        st.remove_ligands_and_waters()
        keep_acc = {acc_of[g] for g in psub}
        ea = entity_uniprot(pid)
        drop = []
        for ch in st[0]:
            ent = st.get_entity_of(ch.get_polymer()) if len(ch) else None
            if ent is None or not (ea.get(ent.name, set()) & keep_acc):
                drop.append(ch.name)   # non-subunit proteins and all nucleic acids (e.g. processome rRNA)
        for nm in drop:
            st[0].remove_chain(nm)
        st.remove_empty_chains()
        xyz, kind, _ = sl.beads_from_structure(st)
        xyz = xyz - xyz.mean(0)
        ing_id = c.cpx.replace("-", "_") + "_" + re.sub(r"\W+", "_", c.name)[:32]
        bs.save(ing_id, xyz, kind, extra={"pdb": pid}, per=30)
        chosen.append(dict(ing_id=ing_id, cpx=c.cpx, label=c.name, sub=psub, copies=copies, mass_da=mda,
                           pdb=pid, resolution=res, coverage=cov, n_beads=len(xyz)))
        used |= set(psub)
        tried.append((c.cpx, c.name, "added", pid, cov))
        log(f"{c.cpx} {c.name[:40]:40s} {pid} cov={cov:.2f} res={res} copies={copies:.0f} sub={psub}")
    return chosen, pd.DataFrame(tried, columns=["complex_ac", "name", "outcome", "pdb", "coverage"])


def apply(ing, df, chosen):
    """Add chosen complexes; subtract their subunits from residual monomer ingredients."""
    ing = ing.copy()
    cw_of = dict(zip(df.gene, df.comp_weights))
    new = []
    for c in chosen:
        # compartment weights: copy-weighted average of subunit weights
        w = {}
        for g, s in c["sub"].items():
            for k, v in json.loads(cw_of[g]).items():
                w[k] = w.get(k, 0) + v / len(c["sub"])
        for g, s in c["sub"].items():
            m = (ing.kind == "protein") & (ing.ing_id == g)
            ing.loc[m, "copies_per_cell"] = np.maximum(0, ing.loc[m, "copies_per_cell"] - c["copies"] * s)
        new.append(dict(ing_id=c["ing_id"], label=c["label"], kind="complex", genes=";".join(sorted(c["sub"])),
                        n_subunit_types=len(c["sub"]), copies_per_cell=c["copies"], comp_weights=json.dumps(w),
                        mass_da=c["mass_da"], pdb=c["pdb"], membrane="", oligomer_n=1, uniprot="", n_tm=0))
    ing = pd.concat([ing, pd.DataFrame(new)], ignore_index=True)
    ing = ing[(ing.kind != "protein") | (ing.copies_per_cell >= 1)].reset_index(drop=True)
    return ing
