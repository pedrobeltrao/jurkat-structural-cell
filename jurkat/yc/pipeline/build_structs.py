"""Process structures into residue-bead coordinates + sphere-trees, with membrane orientation.
Output: config.LIB/<ing_id>.npz  (beads[n,3] nm centered, kind[n], st_c[k,3], st_r[k], memb_axis info)
Membrane proteins are rotated so that membrane normal = +z, membrane midplane at z=0, cytoplasmic side = +z."""
import os, re, json, gzip
import numpy as np
import pandas as pd
import gemmi
import structlib as sl

from config import LIB, PDB, AFDB_TAR, NPROC
os.makedirs(LIB, exist_ok=True)


def rot_to_z(v):
    v = v / np.linalg.norm(v)
    z = np.array([0, 0, 1.0])
    c = np.dot(v, z)
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        return np.diag([1, -1, -1.0])
    ax = np.cross(v, z); s = np.linalg.norm(ax); ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def parse_ranges(s, key):
    if not isinstance(s, str):
        return []
    out = []
    for m in re.finditer(key + r" (\d+)\.\.(\d+);\s*/note=\"([^\"]*)\"", s):
        out.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    return out


def save(ing_id, beads, kind, extra=None, per=30):
    beads = np.asarray(beads, float)
    c, r = sl.sphere_tree(beads, per=per)
    d = dict(beads=beads.astype(np.float32), kind=kind.astype(np.int8), st_c=c.astype(np.float32), st_r=r.astype(np.float32))
    if extra:
        d.update(extra)
    np.savez_compressed(f"{LIB}/{ing_id}.npz", **d)
    return len(beads), len(r), float(np.linalg.norm(beads, axis=1).max())


def residue_beads_with_numbers(st):
    """beads + (chain, seqnum, entity description) for TM mapping."""
    st.setup_entities()
    ent_desc = {}
    for e in st.entities:
        for sub in e.subchains:
            ent_desc[sub] = (e.name, getattr(e, "pdbx_description", []) or [])
    out = []
    for ch in st[0]:
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
            out.append((a.pos.x / 10, a.pos.y / 10, a.pos.z / 10, k, ch.name, res.seqid.num, res.subchain))
    return out


def process_complex(ing_id, pdb, pma1_tm=None):
    st = gemmi.read_structure(f"{PDB}/{pdb}-assembly1.cif.gz")
    st.remove_hydrogens()
    rows = residue_beads_with_numbers(st)
    xyz = np.array([r[:3] for r in rows]); kind = np.array([r[3] for r in rows])
    xyz -= xyz.mean(0)
    extra = {}
    if ing_id == "NPC":
        w, V = np.linalg.eigh(np.cov(xyz.T)); ax = V[:, 0]  # smallest variance = ring axis
        R = rot_to_z(ax); xyz = xyz @ R.T
        xyz[:, 2] -= np.median(xyz[:, 2])
        extra["membrane"] = "NE"
    elif ing_id == "Pma1_hexamer":
        seq = np.array([r[5] for r in rows])
        tm = np.zeros(len(rows), bool)
        for a, b, _ in pma1_tm:
            tm |= (seq >= a) & (seq <= b)
        p = xyz[tm]; w, V = np.linalg.eigh(np.cov(p.T)); ax = V[:, 0]  # TM layer = slab -> smallest variance
        R = rot_to_z(ax); xyz = xyz @ R.T
        mid = xyz[tm, 2].mean(); xyz[:, 2] -= mid
        if xyz[~tm, 2].mean() < 0:  # cytoplasmic domains -> +z
            xyz[:, 1:] *= -1
        extra["membrane"] = "PM"
    elif ing_id == "V-ATPase":
        w, V = np.linalg.eigh(np.cov(xyz.T)); ax = V[:, 2]
        R = rot_to_z(ax); xyz = xyz @ R.T
        # c-ring: chains with the shortest length occurring >= 8 times -> use lowest-density end; simpler:
        # Vo membrane sector is at the end with the narrower radial profile; find end with fewer residues
        z = xyz[:, 2]; lo, hi = np.percentile(z, 5), np.percentile(z, 95)
        n_lo = (z < lo + 6).sum(); n_hi = (z > hi - 6).sum()
        chains = np.array([r[4] for r in rows])
        # identify c-ring by chain length ~160 aa
        lens = pd.Series(chains).value_counts()
        cchains = set(lens[(lens > 140) & (lens < 220)].index)
        cmask = np.isin(chains, list(cchains))
        mid = np.median(z[cmask]) if cmask.sum() > 300 else (lo + 2 if n_lo < n_hi else hi - 2)
        xyz[:, 2] -= mid
        if np.median(xyz[~cmask, 2]) < 0:  # V1 head is cytoplasmic (+z)
            xyz[:, 1:] *= -1
        extra["membrane"] = "VAC"
    return save(ing_id, xyz, kind, extra, per=40)


def process_af(acc, ing_id, st, tm_ranges=None, topo=None, plddt_min=50):
    st.remove_hydrogens()
    rows = []
    for ch in st[0]:
        for res in ch:
            a = res.find_atom("CA", "*")
            if a is not None:
                rows.append((a.pos.x / 10, a.pos.y / 10, a.pos.z / 10, res.seqid.num, a.b_iso))
    arr = np.array(rows)
    xyz, seq, pl = arr[:, :3], arr[:, 3].astype(int), arr[:, 4]
    keep = pl >= plddt_min
    if tm_ranges:
        tm = np.zeros(len(seq), bool)
        for a, b, _ in tm_ranges:
            tm |= (seq >= a) & (seq <= b)
        keep |= tm
    if keep.sum() < 10:  # mostly disordered: keep everything (rendered as compact blob)
        keep[:] = True
    xyz, seq = xyz[keep], seq[keep]
    extra = {"plddt_frac_kept": np.float32(keep.mean())}
    if tm_ranges:
        tm = np.zeros(len(seq), bool); vecs = []
        for a, b, _ in tm_ranges:
            m = (seq >= a) & (seq <= b); tm |= m
            if m.sum() >= 5:
                p = xyz[m]; vecs.append(p[-1] - p[0])
        vecs = np.array(vecs)
        ref = vecs[0]
        n = sum(v if np.dot(v, ref) >= 0 else -v for v in vecs)
        R = rot_to_z(n); xyz = (xyz - xyz[tm].mean(0)) @ R.T
        # cytoplasmic side -> +z using topology annotation
        cyt = np.zeros(len(seq), bool)
        for a, b, note in (topo or []):
            if "ytoplasm" in note:
                cyt |= (seq >= a) & (seq <= b)
        if cyt.sum() > 0 and xyz[cyt, 2].mean() < 0:
            xyz[:, 1:] *= -1
        extra["membrane"] = "TM"
    else:
        xyz = xyz - xyz.mean(0)
    return save(ing_id, xyz, np.zeros(len(xyz), np.int8), extra)


def process_homo_pdb(ing_id, pdb, acc):
    path = f"{PDB}/{pdb}-assembly1.cif.gz"
    if not os.path.exists(path):
        import requests
        open(path, "wb").write(requests.get(f"https://files.rcsb.org/download/{pdb}-assembly1.cif.gz", timeout=120).content)
    st = gemmi.read_structure(path)
    xyz, kind, _ = sl.beads_from_structure(st)
    xyz = xyz[kind == 0]
    return save(ing_id, xyz - xyz.mean(0), np.zeros(len(xyz), np.int8), {"pdb": pdb})


def _af_worker(args):
    acc, ing_id, name, data, tm, tp = args
    import gemmi
    if name.endswith(".cif.gz"):
        st = gemmi.make_structure_from_block(gemmi.cif.read_string(data)[0])
    else:
        st = gemmi.read_pdb_string(data)
    try:
        return ing_id, process_af(acc, ing_id, st, tm, tp)
    except Exception as e:
        return ing_id, ("ERR", str(e))


def process_all_af(ing, topo, tar_path=AFDB_TAR, n_jobs=NPROC):
    import tarfile, os
    from joblib import Parallel, delayed
    want = {r.uniprot: r.ing_id for r in ing[(ing.kind == "protein") & (ing.uniprot != "")].itertuples()}
    jobs = []
    with tarfile.open(tar_path) as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if not name.endswith(".cif.gz") or "-F1-" not in name:
                continue
            acc = name.split("-")[1]
            if acc in want:
                data = gzip.decompress(tf.extractfile(m).read()).decode()
                tm = parse_ranges(topo.loc[acc, "tm"], "TRANSMEM") if acc in topo.index else []
                tp = parse_ranges(topo.loc[acc, "topo"], "TOPO_DOM") if acc in topo.index else []
                jobs.append((acc, want[acc], name, data, tm, tp))
    res = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(_af_worker)(j) for j in jobs)
    return dict(res), set(want.values()) - {j[1] for j in jobs}
