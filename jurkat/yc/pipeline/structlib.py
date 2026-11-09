"""Layer 2: structure library. Residue-level beads (CA for protein, P for nucleic acid) in nm,
plus coarse sphere-trees for collision detection (cellPACK convention)."""
import gzip, io, json, tarfile, os
import numpy as np
import gemmi
import requests
from sklearn.cluster import KMeans

BEAD_R = 0.42  # nm, residue bead radius used for rendering / sphere-tree padding


def beads_from_structure(st, chains=None, plddt_min=None):
    st.setup_entities()
    model = st[0]
    xyz, kind, chain_ids = [], [], []
    for ch in model:
        if chains is not None and ch.name not in chains:
            continue
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            atom = None
            if info is not None and info.is_amino_acid():
                atom = res.find_atom("CA", "*"); k = 0
            elif info is not None and info.is_nucleic_acid():
                atom = res.find_atom("P", "*") or res.find_atom("C4'", "*"); k = 1
            if atom is None:
                continue
            if plddt_min is not None and atom.b_iso < plddt_min:
                continue
            xyz.append([atom.pos.x, atom.pos.y, atom.pos.z]); kind.append(k); chain_ids.append(ch.name)
    return np.asarray(xyz, float) / 10.0, np.asarray(kind, np.int8), np.asarray(chain_ids)


def sphere_tree(xyz, per=30, pad=BEAD_R, seed=0):
    n = len(xyz)
    if n == 0:
        return np.zeros((0, 3)), np.zeros(0)
    k = max(1, int(np.ceil(n / per)))
    if k == 1:
        c = xyz.mean(0, keepdims=True)
        r = np.array([np.percentile(np.linalg.norm(xyz - c, axis=1), 95) + pad])
        return c, r
    km = KMeans(n_clusters=k, n_init=1, random_state=seed, max_iter=50).fit(xyz)
    cs, rs = [], []
    for i in range(k):
        p = xyz[km.labels_ == i]
        if len(p) == 0:
            continue
        c = p.mean(0)
        cs.append(c); rs.append(np.percentile(np.linalg.norm(p - c, axis=1), 95) + pad)
    return np.array(cs), np.array(rs)


def center(xyz):
    return xyz - xyz.mean(0)


def afdb_iter(tar_path, wanted):
    """Yield (uniprot, structure) for AF models of wanted accessions (fragment F1 only)."""
    with tarfile.open(tar_path) as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if not name.startswith("AF-") or "-F1-" not in name:
                continue
            if not (name.endswith(".cif.gz") or name.endswith(".pdb.gz")):
                continue
            acc = name.split("-")[1]
            if acc not in wanted:
                continue
            data = gzip.decompress(tf.extractfile(m).read()).decode()
            if name.endswith(".cif.gz"):
                st = gemmi.make_structure_from_block(gemmi.cif.read_string(data)[0])
            else:
                st = gemmi.read_pdb_string(data)
            yield acc, name, st


def rcsb_homo_assembly(acc, n, max_entries=15):
    """Find best-resolution PDB entry whose assembly 1 is a homo-n-mer of this UniProt accession."""
    q = {"query": {"type": "terminal", "service": "text", "parameters": {
        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
        "operator": "exact_match", "value": acc}},
        "return_type": "entry",
        "request_options": {"sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
                            "paginate": {"start": 0, "rows": max_entries}}}
    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=60)
    if r.status_code != 200:
        return None
    for hit in r.json().get("result_set", []):
        pid = hit["identifier"]
        a = requests.get(f"https://data.rcsb.org/rest/v1/core/assembly/{pid}/1", timeout=60)
        if a.status_code != 200:
            continue
        info = a.json().get("rcsb_assembly_info", {})
        if info.get("polymer_entity_count") == 1 and info.get("polymer_entity_instance_count") == n:
            return pid
    return None
