"""Structure library for the Jurkat ingredients (lib/<ing_id>.npz: residue beads nm, kind, sphere tree; membrane key
for oriented TM species). Library convention as yeast (build_structs / membrane_orient): membrane normal +z,
midplane z = 0, cytoplasmic (N) side +z.

  assemblies (pdb:<id>, afdb_complex:<model>): CA/P beads of the whole assembly; TM assemblies oriented with
    membrane_orient.orient (TRANSMEM/TOPO_DOM residues mapped through SIFTS for PDB entries, chain order for AFDB dimers)
  monomers (afdb_monomer:<acc>): build_structs.process_af on the human AFDB v6 F1 model (pLDDT < 50 trimmed, TM ranges
    kept, TM orientation from UniProt topology), one pass over the shared proteome tar
usage: jlib_build.py [assemblies|monomers|all]; writes lib/*.npz and data/derived/lib_build_log.tsv
"""
import os, sys, gzip, json, tarfile, numpy as np, pandas as pd, gemmi, requests
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_structs as bs, membrane_orient as mo
from config import LIB

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
PDBDIR = f"{V}/data/structures/pdb/assemblies"
AFDIR = f"{V}/data/structures/afdb_complexes/models"
TAR = __import__("os").environ.get("AFDB_HUMAN_TAR", "UP000005640_9606_HUMAN_v6.tar")
_SIFTS = None


def sifts():
    global _SIFTS
    if _SIFTS is None:
        _SIFTS = pd.read_csv(f"{V}/data/structures/pdb/sifts/pdb_chain_uniprot.tsv.gz", sep="\t", comment="#",
                             usecols=["PDB", "CHAIN", "SP_PRIMARY", "RES_BEG", "RES_END", "SP_BEG", "SP_END"], low_memory=False)
    return _SIFTS


def pdb_file(pid):
    fn = f"{PDBDIR}/{pid}-assembly1.cif.gz"
    if not os.path.exists(fn):
        r = requests.get(f"https://files.rcsb.org/download/{pid}-assembly1.cif.gz", timeout=300)
        if r.status_code != 200:
            return None
        open(fn, "wb").write(r.content)
    return fn


def beads_pdb(pid, return_chain=False):
    """CA/P beads of the assembly with (UniProt accession, UniProt position) per bead. Mapping via SIFTS SEQRES ranges
    (RES_BEG..RES_END = label_seq; PDB_BEG is empty when the first residues are unobserved, so it is not used)."""
    st = gemmi.read_structure(pdb_file(pid)); st.remove_hydrogens(); st.setup_entities()
    s = sifts(); s = s[s.PDB == pid]
    seg = {}
    for r in s.itertuples():
        try:
            seg.setdefault(r.CHAIN, []).append((int(r.RES_BEG), int(r.RES_END), r.SP_PRIMARY, int(r.SP_BEG)))
        except (ValueError, TypeError):
            pass
    xyz, kind, acc, pos, chn = [], [], [], [], []
    for ch in st[0]:
        segs = seg.get(ch.name.split("-")[0], [])
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            if info is None:
                continue
            if info.is_amino_acid():
                a = res.find_atom("CA", "*"); k = 0
            elif info.is_nucleic_acid():
                a = res.find_atom("P", "*") or res.find_atom("C4'", "*"); k = 1
            else:
                continue
            if a is None:
                continue
            ac, p = "", -1
            n = res.label_seq
            if n is not None:
                for b, e, sp, spb in segs:
                    if b <= n <= e:
                        ac, p = sp, spb + n - b; break
            xyz.append([a.pos.x / 10, a.pos.y / 10, a.pos.z / 10]); kind.append(k); acc.append(ac); pos.append(p); chn.append(ch.name)
    out = (np.array(xyz), np.array(kind, np.int8), np.array(acc), np.array(pos))
    return out + (np.array(chn),) if return_chain else out


def beads_afdb(model, accs):
    st = gemmi.read_structure(f"{AFDIR}/{model}.cif.gz"); st.remove_hydrogens()
    chains = [ch.name for ch in st[0]]
    return mo.beads_with_positions(st, acc_of_chain={c: a for c, a in zip(chains, accs)})


def build_assembly(row):
    ing, src, tm = row["ing_id"], row["struct_src"], row["transmembrane"]
    try:
        kind_, ident = src.split(":", 1)
        graft_note = ""
        if kind_ == "pdb":
            if pdb_file(ident) is None:
                return ing, "no_file", ""
            xyz, kind, acc, pos, chn = beads_pdb(ident, return_chain=True)
            if tm and ing != "NPC":
                import tm_graft
                xyz, kind, acc, pos, graft_note = tm_graft.graft_missing_tm(xyz, kind, acc, pos, chn)
        else:
            xyz, kind, acc, pos = beads_afdb(ident, row["accs"])
        if len(xyz) < 10:
            return ing, "too_few_beads", ""
        extra, note = {"src": src}, ""
        if tm or ing == "NPC":
            if ing == "NPC":                                         # ring axis = smallest variance; midplane at median z
                X = xyz - xyz.mean(0); w, Vv = np.linalg.eigh(np.cov(X.T)); out = X @ bs.rot_to_z(Vv[:, 0]).T
                out[:, 2] -= np.median(out[:, 2]); xyz = out; extra["membrane"] = "NE"; note = "npc ring axis"
            else:
                tmm, ns, ps = mo.masks(acc, pos, xyz)
                out, info = mo.orient(xyz, tmm, ns, ps, min_score=mo.min_score_for(acc))
                if out is None:
                    xyz = xyz - xyz.mean(0); note = "orient_failed:" + info.get("fail", "")
                else:
                    xyz = out; extra["membrane"] = "TM"; note = f"oriented {info.get('sign_from')} score {info.get('score')}"
            if graft_note:
                note += "; " + graft_note
        else:
            xyz = xyz - xyz.mean(0)
        n, k, rmax = bs.save(ing, xyz, kind, extra, per=40 if len(xyz) > 3000 else 30)
        return ing, "ok", f"{n} beads, {k} spheres, rmax {rmax:.1f} nm; {note}"
    except Exception as e:
        return ing, "error", str(e)[:200]


def assemblies(I, nproc=24):
    acc = pd.read_csv(f"{V}/data/derived/inventory_v0_complexed.tsv", sep="\t", usecols=["gene", "uniprot"]).set_index("gene").uniprot
    A = I[I.kind != "protein"].copy()
    A["accs"] = A.genes.map(lambda gs: [acc.get(g, "") for g in gs.split(";")])
    rows = A.to_dict("records")
    for pid in {r["struct_src"].split(":")[1] for r in rows if r["struct_src"].startswith("pdb:")}:
        pdb_file(pid)                                                 # fetch missing entries serially first
    with ProcessPoolExecutor(nproc) as ex:
        res = list(ex.map(build_assembly, rows, chunksize=4))
    return pd.DataFrame(res, columns=["ing_id", "status", "note"])


def monomers(I, nproc=24):
    P = I[I.kind == "protein"]
    topo = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/data/uniprot_topology.tsv", sep="\t").set_index("Entry")
    topo = topo.rename(columns={"Transmembrane": "tm", "Topological domain": "topo"})
    ing = pd.DataFrame({"ing_id": P.ing_id, "kind": "protein", "uniprot": P.uniprot.fillna("")})
    done, missing = bs.process_all_af(ing, topo, tar_path=TAR, n_jobs=nproc)
    rows = [(k, "ok" if not (isinstance(v, tuple) and v and v[0] == "ERR") else "error", str(v)[:150]) for k, v in done.items()]
    rows += [(m, "no_afdb_model", "") for m in missing]
    return pd.DataFrame(rows, columns=["ing_id", "status", "note"])


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    I = pd.read_csv(f"{V}/data/derived/ingredients_v0.tsv", sep="\t")
    logs = []
    if what in ("assemblies", "all"):
        L = assemblies(I); logs.append(L); print("assemblies:", L.status.value_counts().to_dict(), flush=True)
        print(L[L.note.str.contains("orient_failed")].head(10).to_string(index=False))
    if what in ("monomers", "all"):
        L = monomers(I); logs.append(L); print("monomers:", L.status.value_counts().to_dict(), flush=True)
    fn = f"{V}/data/derived/lib_build_log.tsv"
    old = pd.read_csv(fn, sep="\t") if os.path.exists(fn) else pd.DataFrame(columns=["ing_id", "status", "note"])
    new = pd.concat([old[~old.ing_id.isin(pd.concat(logs).ing_id)]] + logs)
    new.to_csv(fn, sep="\t", index=False)
