"""Membrane orientation of multi-chain structures (complexes, homo-oligomers), inventory v3 item B1.

Library convention (pack3d): membrane normal = +z, membrane midplane at z = 0, N side (cytoplasm; matrix for the
mitochondrial inner membrane) at +z. The normal is the direction that puts the most UniProt TRANSMEM residues inside a
+-1.6 nm slab while keeping other residues out of it (all directions on a hemisphere, then local refinement). The sign
comes from UniProt TOPO_DOM annotations (N-side vs P-side domains), else the side carrying more soluble mass is
taken as the N side (true for peripheral cytosolic complexes on organelle membranes). Generalises the per-structure code that oriented
NPC, Pma1 and V-ATPase in build_structs.process_complex.
"""
import re
import numpy as np
import pandas as pd
import gemmi
from build_structs import parse_ranges, rot_to_z
from config import DATA

HALF = 1.6                 # nm, half-thickness of the hydrophobic slab scored for TM residues
MIN_TM_BEADS = 10
P_SIDE = re.compile(r"lumen|extracellular|vacuol|intermembrane|perinuclear|periplasm|peroxisom", re.I)
N_SIDE = re.compile(r"cytoplasm|mitochondrial matrix|nuclear|nucleoplasm", re.I)
_TOPO = None
_BARREL = None
BARREL_MIN_SCORE = 0.3      # all-beta barrels: loops also sit near the slab, so the slab score is lower


def barrels():
    """UniProt accessions annotated as integral (multi-pass) membrane proteins but without TRANSMEM ranges:
    outer-membrane beta-barrels (TOM40, SAM50, MDM10, MDM34). Their beta-strands stand in for TM segments."""
    global _BARREL
    if _BARREL is None:
        u = pd.read_csv(f"{DATA}/uniprot_yeast.tsv", sep="\t")
        integral = u["Subcellular location [CC]"].fillna("").str.contains("Multi-pass membrane protein")
        _BARREL = set(u.Entry[integral & u["Transmembrane"].isna()])
    return _BARREL


def min_score_for(acc):
    """slab-score threshold: lower when beta-barrel strands define the membrane segment"""
    return BARREL_MIN_SCORE if set(acc) & barrels() else 0.45


def strands(xyz, min_len=5):
    """beta-strand residues from Calpha geometry: runs of >= min_len residues with Ca(i-1)-Ca(i+1) > 0.6 nm"""
    d = np.r_[0, np.linalg.norm(xyz[2:] - xyz[:-2], axis=1), 0]
    out = np.zeros(len(xyz), bool)
    i = 0
    while i < len(d):
        j = i
        while j < len(d) and d[j] > 0.60:
            j += 1
        if j - i >= min_len:
            out[i:j] = True
        i = max(j, i + 1)
    return out


def topology():
    """UniProt accession -> (TM ranges, N-side ranges, P-side ranges), residue numbers in UniProt numbering"""
    global _TOPO
    if _TOPO is None:
        t = pd.read_csv(f"{DATA}/uniprot_topology.tsv", sep="\t")
        t.columns = ["acc", "topo", "tm", "intramem"]
        _TOPO = {}
        for r in t.itertuples():
            tm = [(a, b) for a, b, _ in parse_ranges(r.tm, "TRANSMEM")]
            dom = parse_ranges(r.topo, "TOPO_DOM")
            ps = [(a, b) for a, b, n in dom if P_SIDE.search(n)]
            ns = [(a, b) for a, b, n in dom if N_SIDE.search(n) and not P_SIDE.search(n)]
            _TOPO[r.acc] = (tm, ns, ps)
    return _TOPO


HYDROPHOBIC = set("AILMFVWC")
CHARGED = set("DEKR")
HYD_MIN_SCORE = 0.3


def beads_with_positions(st, align=None, acc_of_chain=None, with_aa=False):
    """Residue beads in the same order as structlib.beads_from_structure, plus (UniProt accession, UniProt position).
    align: {entity_id: [(acc, entity_beg, ref_beg, length)]} for PDB assemblies (label_seq numbering);
    acc_of_chain: {chain: acc} for AlphaFold models (author numbering = UniProt numbering)."""
    st.setup_entities()
    sub2ent = {s: e.name for e in st.entities for s in e.subchains}
    xyz, kind, acc, pos, aa = [], [], [], [], []
    for ch in st[0]:
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            atom = None
            if info is not None and info.is_amino_acid():
                atom = res.find_atom("CA", "*"); k = 0
            elif info is not None and info.is_nucleic_acid():
                atom = res.find_atom("P", "*") or res.find_atom("C4'", "*"); k = 1
            if atom is None:
                continue
            a, p = "", -1
            if acc_of_chain is not None:
                a, p = acc_of_chain.get(ch.name, ""), res.seqid.num
            elif align is not None and res.label_seq is not None:
                for (ac, eb, rb, ln) in align.get(sub2ent.get(res.subchain, ""), []):
                    if eb <= res.label_seq < eb + ln:
                        a, p = ac, rb + res.label_seq - eb; break
            xyz.append([atom.pos.x / 10, atom.pos.y / 10, atom.pos.z / 10]); kind.append(k); acc.append(a); pos.append(p)
            aa.append(info.one_letter_code.upper())
    out = (np.array(xyz), np.array(kind, np.int8), np.array(acc), np.array(pos))
    return out + (np.array(aa),) if with_aa else out


def masks(acc, pos, xyz=None):
    """TM / N-side / P-side residue masks from UniProt; with xyz, strands of beta-barrel proteins count as TM"""
    topo = topology()
    tm = np.zeros(len(acc), bool); ns = np.zeros(len(acc), bool); ps = np.zeros(len(acc), bool)
    if xyz is not None:
        for a in set(acc) & barrels():
            sel = np.where(acc == a)[0]
            tm[sel[strands(xyz[sel])]] = True
    for a in set(acc) - {""}:
        if a not in topo:
            continue
        sel = acc == a; p = pos[sel]
        t, n, q = topo[a]
        for arr, rng in ((tm, t), (ns, n), (ps, q)):
            m = np.zeros(sel.sum(), bool)
            for lo, hi in rng:
                m |= (p >= lo) & (p <= hi)
            arr[np.where(sel)[0][m]] = True
    return tm, ns, ps


def _hemisphere(n):
    i = np.arange(n) + 0.5
    z = i / n                                   # upper hemisphere (the normal's sign is set later)
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.c_[r * np.cos(phi), r * np.sin(phi), z]


def _score(X, tm, dirs, out=None):
    """slab score per direction: fraction of tm beads in the slab minus fraction of 'out' beads (default: all others)"""
    out = ~tm if out is None else out
    z = X @ dirs.T                               # beads x directions
    zc = np.median(z[tm], axis=0)
    inside = np.abs(z - zc) < HALF
    s_tm = inside[tm].mean(0)
    s_out = inside[out].mean(0) if out.any() else np.zeros(len(dirs))
    return s_tm - s_out, s_tm, zc


def orient_hydrophobic(xyz, aa, ns, ps, seed=0):
    """Fallback for integral proteins without TRANSMEM annotation (outer-membrane beta-barrels): the slab is the belt
    that holds the most hydrophobic residues and the fewest charged ones. Returns (xyz, info, slab mask) or (None, info, None)."""
    from scipy.spatial import cKDTree
    nbr = np.array([len(x) for x in cKDTree(xyz).query_ball_point(xyz, 1.0)])
    exposed = nbr <= np.percentile(nbr, 50)        # lipid-facing residues are on the surface
    hyd, chg = np.isin(aa, list(HYDROPHOBIC)) & exposed, np.isin(aa, list(CHARGED)) & exposed
    out, info = orient(xyz, hyd, ns, ps, seed, out=chg, min_score=HYD_MIN_SCORE, min_in=0.35)
    if out is None:
        return None, info, None
    info["method"] = "hydrophobic belt"
    return out, info, np.abs(out[:, 2]) < HALF


def orient(xyz, tm, ns, ps, seed=0, out=None, min_score=0.45, min_in=0.6):
    """Rotate/translate xyz into the membrane frame. Returns (xyz_oriented, info) or (None, info) on failure.
    out: beads that should stay out of the slab (default: all non-tm beads)."""
    info = dict(n_tm=int(tm.sum()))
    if tm.sum() < MIN_TM_BEADS:
        info["fail"] = "too few TM residues in the structure"; return None, info
    outm = ~tm if out is None else out
    c0 = xyz[tm].mean(0); X = xyz - c0
    best = None
    for chunk in np.array_split(_hemisphere(4000), max(8, len(X) * 4000 // 20_000_000 + 1)):   # <= ~160 MB per chunk
        sc, stm, zc = _score(X, tm, chunk, outm)
        k = int(np.argmax(sc))
        if best is None or sc[k] > best[0]:
            best = (sc[k], chunk[k], stm[k])
    rng = np.random.default_rng(seed)
    for step in (0.08, 0.03, 0.01):              # local refinement (radians)
        cand = best[1] + rng.normal(0, step, (400, 3)); cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        sc, stm, zc = _score(X, tm, cand, outm)
        k = int(np.argmax(sc))
        if sc[k] > best[0]:
            best = (sc[k], cand[k], stm[k])
    n = best[1]
    out = X @ rot_to_z(n).T
    out[:, 2] -= np.median(out[tm, 2])
    out[:, :2] -= out[tm, :2].mean(0)            # origin at the TM centroid in the membrane plane
    if ns.any() or ps.any():
        zn = out[ns, 2].mean() if ns.any() else 0.0
        zp = out[ps, 2].mean() if ps.any() else 0.0
        sign = zn - zp
        if sign < 0:
            out[:, 1:] *= -1                     # 180 deg about x: N side -> +z
        sign_from = "topology"
    else:                                        # no topology: the side carrying more soluble mass is cytoplasmic
        sol = ~tm
        if (out[sol, 2] > HALF).sum() < (out[sol, 2] < -HALF).sum():
            out[:, 1:] *= -1
        sign_from = "soluble mass"
    info.update(score=round(float(best[0]), 3), frac_tm_in_slab=round(float(best[2]), 3), sign_from=sign_from)
    if best[2] < min_in or best[0] < min_score:
        info["fail"] = "no convincing membrane slab"; return None, info
    return out, info
