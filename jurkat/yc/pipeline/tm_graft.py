"""Graft missing transmembrane segments into experimental assemblies from AlphaFold DB monomer models.

For every chain of a UniProt TM protein (TRANSMEM ranges) whose TM residues are mostly absent from the deposited
structure (< 50% present):
  1. AFDB v6 F1 model of that accession (shared human proteome tar, random access via data/derived/afdb_human_tar_index.json)
  2. Kabsch superposition of AF CA onto the chain's CA over residues present in both with AF pLDDT >= 70
     (>= 20 residues and RMSD <= 3 A required)
  3. add the AF residues absent from the chain: all TM residues, plus other residues with pLDDT >= 50 (same trimming as
     the monomer library)
  4. clash check (CA-CA < 3.0 A to another chain): reject if > 15% of added beads or > 2% of added TM beads clash;
     otherwise drop the clashing non-TM beads (AF monomer conformation differs at subunit interfaces)
The linker confidence (mean pLDDT of the AF residues between the superposed region and the nearest TM segment) is
reported: low values mean the TM placement relative to the resolved domain is uncertain.
Returns the extended bead arrays and a note; the caller then orients the assembly (membrane_orient.orient), whose
slab score checks that the grafted TM segments of different chains are coplanar."""
import gzip, json, numpy as np, gemmi
from scipy.spatial import cKDTree
import membrane_orient as mo

TAR = __import__("os").environ.get("AFDB_HUMAN_TAR", "UP000005640_9606_HUMAN_v6.tar")
IDX = (__import__("os").environ.get("VCELL_ROOT", ".") + "/data/derived/afdb_human_tar_index.json")
_IDX = None


def af_model(acc):
    """(UniProt positions, CA xyz nm, pLDDT) of the AFDB F1 model, or None"""
    global _IDX
    if _IDX is None:
        _IDX = json.load(open(IDX))
    if acc not in _IDX:
        return None
    off, size = _IDX[acc]
    with open(TAR, "rb") as f:
        f.seek(off); data = gzip.decompress(f.read(size)).decode()
    st = gemmi.make_structure_from_block(gemmi.cif.read_string(data)[0])
    pos, xyz, pl = [], [], []
    for ch in st[0]:
        for res in ch:
            a = res.find_atom("CA", "*")
            if a is not None:
                pos.append(res.seqid.num); xyz.append([a.pos.x / 10, a.pos.y / 10, a.pos.z / 10]); pl.append(a.b_iso)
    return np.array(pos), np.array(xyz), np.array(pl)


def kabsch(P, Q):
    """R, t minimising |R P + t - Q|"""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc); U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def graft_missing_tm(xyz, kind, acc, pos, chn):
    topo = mo.topology()
    add_xyz, add_acc, add_pos, add_chn, notes = [], [], [], [], []
    n_ok = n_try = 0
    for c in np.unique(chn):
        sel = chn == c
        accs = set(acc[sel]) - {""}
        if len(accs) != 1:
            continue
        a = accs.pop()
        if a not in topo or not topo[a][0]:
            continue
        tm_ranges = topo[a][0]
        tm_pos = np.concatenate([np.arange(lo, hi + 1) for lo, hi in tm_ranges])
        present = set(pos[sel & (acc == a)].tolist())
        if np.mean([p in present for p in tm_pos]) >= 0.5:
            continue
        n_try += 1
        m = af_model(a)
        if m is None:
            notes.append(f"{c}:{a} no AF model"); continue
        apos, axyz, apl = m
        amap = {p: i for i, p in enumerate(apos)}
        ci = np.where(sel & (acc == a))[0]
        common = [(i, amap[pos[i]]) for i in ci if pos[i] in amap and apl[amap[pos[i]]] >= 70]
        if len(common) < 20:
            notes.append(f"{c}:{a} <20 confident common residues"); continue
        Pi = np.array([j for _, j in common]); Qi = np.array([i for i, _ in common])
        R, t = kabsch(axyz[Pi], xyz[Qi])
        rmsd = np.sqrt(((axyz[Pi] @ R.T + t - xyz[Qi]) ** 2).sum(1).mean()) * 10
        if rmsd > 3.0:
            notes.append(f"{c}:{a} superposition RMSD {rmsd:.1f} A"); continue
        tmset = set(tm_pos.tolist())
        new = [j for j, p in enumerate(apos) if p not in present and (p in tmset or apl[j] >= 50)]
        if not new:
            continue
        nx = axyz[new] @ R.T + t
        others = np.where(~sel)[0]
        clash_note = ""
        if len(others):
            d, _ = cKDTree(xyz[others]).query(nx)
            cl = d < 0.30; is_tm = np.array([apos[j] in tmset for j in new])
            if cl.mean() > 0.15 or (cl & is_tm).mean() > 0.02:
                notes.append(f"{c}:{a} graft clashes ({cl.mean():.0%})"); continue
            if cl.any():                                     # drop clashing non-TM grafted beads (AF monomer conformation)
                keep = ~cl; new = [j for j, k in zip(new, keep) if k]; nx = nx[keep]; clash_note = f", {int(cl.sum())} clashing beads dropped"
        # linker confidence: AF residues between the superposed region and the closest TM segment
        sup_lo, sup_hi = apos[Pi].min(), apos[Pi].max()
        gaps = [(hi, sup_lo) if hi < sup_lo else (sup_hi, lo) for lo, hi in tm_ranges if hi < sup_lo or lo > sup_hi]
        if gaps:
            g0, g1 = min(gaps, key=lambda g: g[1] - g[0])
            lk = [apl[j] for j, p in enumerate(apos) if g0 < p < g1]
            link = f"linker pLDDT {np.mean(lk):.0f}" if lk else "no linker"
        else:
            link = "TM within superposed span"
        add_xyz.append(nx); add_acc += [a] * len(new); add_pos += list(apos[new]); add_chn += [c] * len(new)
        n_ok += 1
        notes.append(f"{c}:{a} +{len(new)} beads (RMSD {rmsd:.1f} A, {link}{clash_note})")
    if not add_xyz:
        return xyz, kind, acc, pos, (f"graft: none of {n_try} chains ({'; '.join(notes[:3])})" if n_try else "")
    xyz = np.vstack([xyz] + add_xyz)
    kind = np.r_[kind, np.zeros(len(add_acc), np.int8)]
    acc = np.r_[acc, np.array(add_acc)]; pos = np.r_[pos, np.array(add_pos)]
    return xyz, kind, acc, pos, f"graft: {n_ok}/{n_try} chains [{'; '.join(notes[:4])}{' ...' if len(notes) > 4 else ''}]"
