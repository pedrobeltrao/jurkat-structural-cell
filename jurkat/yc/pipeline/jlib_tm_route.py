"""Membrane libraries for transmembrane-routed ingredients whose library had no membrane frame (inventory v5 TM routing).

Input: data/derived/v5/tm_route_ids.tsv (ing_id, group):
  tm_no_memlib            flagged transmembrane, but the library lacks the "membrane" key (mostly PDB structures of the
                          soluble domains only, so membrane_orient.orient failed), or no library at all (multi-fragment
                          AFDB proteins where process_af errored; proteins without an AFDB model)
  uniprot_tm_not_flagged  ingredient contains a gene with a UniProt TRANSMEM/INTRAMEM feature (monotopic / hairpin proteins)
Library convention (as jlib_build / build_structs): membrane normal +z, midplane z = 0, +z = cytoplasmic side for
PM/ER/Golgi/endo/mito OM, nucleoplasm/cytoplasm (non-perinuclear) for NE, matrix for the mito IM; lumenal /
extracellular / IMS / perinuclear domains at -z.

Topology: UniProt REST features (TRANSMEM, INTRAMEM, TOPO_DOM, signal) cached in data/derived/v5/tm_route_uniprot.json
(the local uniprot_topology.tsv lacks INTRAMEM-only entries such as SQLE, STOM, HADHB). Per-residue side: TOPO_DOM
(Cytoplasmic / Nuclear / Mitochondrial matrix -> +1; Extracellular / Lumenal / intermembrane / Perinuclear -> -1);
unannotated soluble blocks (sequence between TRANSMEM segments; INTRAMEM does not cross) take the opposite side of an
annotated neighbour block; no TOPO_DOM at all: N-terminal block lumenal if the protein has a signal peptide, else the
largest block cytosolic.

Methods (per ingredient, beads re-read with UniProt accession/position as jlib_build did):
  orient          the structure holds >= 50% of its chains' TRANSMEM residues: membrane_orient.orient (default
                  thresholds); accepted if soluble beads inside |z| < 2 nm <= 0.5 x membrane residues + 30 and >= 80% of
                  annotated soluble beads outside the slab lie on their side. CPX-6121 (TOM, 8xva): normal = principal
                  direction of the TOMM40 beta strands (barrel axis), midplane at their median, TOMM22 cytosolic domain +z.
  orient_relaxed  as orient with min_score 0.25, min_in 0.45; assemblies spanning two membranes (gap junctions GJA1/GJB2
                  x12, VANGL1/2 x6 dimers of trimers): slab from one layer of chains, the partner layer protrudes
  anchored        soluble part kept rigid (structure residues of TRANSMEM/INTRAMEM segments are removed and rebuilt
                  ideally, so their bead count is preserved). Soluble beads are split by side; each side group is turned
                  so that the vector from its centroid to its membrane anchors (the residues sequence-adjacent to the
                  chain's TM/IM segments) points at the membrane (smallest principal axis when the anchors are central or
                  absent), then shifted so its nearest bead is 3.5 nm from the midplane. Ideal helices (CA beads, 0.15 nm
                  rise, radius 0.23 nm, 100 deg/residue) are added under the footprint, near the chain's anchor, axes
                  >= 1 nm apart: TRANSMEM centred on z = 0 (one per segment per chain copy; if no TM-bearing chain is in the
                  structure, per inventory stoichiometry); INTRAMEM as hairpins of <= 11-residue half-depth helices
                  (antiparallel, z from 1.9 to ~0.3 nm) in the leaflet of the protein's side.
  af_fragment     monomers without a library: AFDB v6 fragment that contains all TRANSMEM segments (most central);
                  1-2 TM: helix-axis orientation + soluble blocks (>= 30 residues) shifted to |z| >= 2.3 nm on their side
                  (as jlib_reorient_tm); >= 3 TM: orient (relaxed fallback); the rest of the protein is not modelled
                  (structural mass undercount, noted). Falls back to anchored when orientation fails.
                  GSDMB_x24 (8et1, pore without its modelled barrel): ring normal, ring on the cytosolic face, unmodelled
                  TRANSMEM residues built as 24 beta-hairpins forming the barrel.
  skipped         no structure (no AFDB model: selenoproteins, NRXN2/3, ERVK-24)
npz keys as before (beads, kind, st_c, st_r, src, plddt_frac_kept for monomers) + membrane="TM", tm_route=<method>,
tm_beads (int8 per bead, 1 = TRANSMEM/INTRAMEM residue, from the structure or built).
Mass: structural residues are unchanged except for TM/IM helix beads (new_beads column = beads - old library beads).
When the inventory struct_src differs from the library's src (12 complexes), the library's structure is kept (noted),
except USE_INVENTORY_SRC (CPX-6121 TOM 8xva, CPX-3073 hERG 8io4).
Backups: data/derived/lib_bak_v4_tmroute/<id>.npz (first copy kept). Sphere trees: per 30 (40 above 3000 beads) here;
then lib_refine_st.py <ids> with INCLUDE_MEMBRANE=1 (run by `jlib_tm_route.py refine`).
usage: jlib_tm_route.py build [ids]   -> lib/<id>.npz, data/derived/v5/tm_route_libs.tsv
       jlib_tm_route.py refine        -> adaptive sphere trees for the built ids, fills beads/spheres columns
       jlib_tm_route.py verify        -> prints checks
OUT_LIB=<dir> writes libraries elsewhere (dry run)."""
import os, sys, re, json, gzip, shutil, subprocess, numpy as np, pandas as pd, gemmi, requests
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_structs as bs, membrane_orient as mo, jlib_build as jb, tm_graft, structlib as sl
from config import LIB

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D5 = f"{V}/data/derived/v5"
IDS = f"{D5}/tm_route_ids.tsv"
UPJ = f"{D5}/tm_route_uniprot.json"
FRAG = f"{D5}/afdb_fragment_index_tmroute.json"
LOG = f"{D5}/tm_route_libs.tsv"
BAK = f"{V}/data/derived/lib_bak_v4_tmroute"
TAR = jb.TAR
OUT = os.environ.get("OUT_LIB", LIB)
HALF = 2.0            # nm, slab half-thickness used for the in-slab checks (cell3d.HALF)
GAP = 3.5             # nm, nearest soluble bead of an anchored part from the midplane
RISE, HRAD, HSEP = 0.15, 0.23, 1.0
P_RE = re.compile(r"extracell|lumen|intermembrane|perinuclear|vacuol|peroxisom|periplasm", re.I)
N_RE = re.compile(r"cytoplasm|nuclear|nucleoplasm|matrix", re.I)
_UP = None


# ---------------------------------------------------------------- UniProt topology
def fetch_uniprot(accs):
    out = json.load(open(UPJ)) if os.path.exists(UPJ) else {}
    for a in sorted(set(accs) - set(out) - {""}):
        j = requests.get(f"https://rest.uniprot.org/uniprotkb/{a}.json?fields=ft_transmem,ft_intramem,ft_topo_dom,ft_signal,"
                         "ft_transit,cc_subcellular_location,length,gene_names", timeout=60).json()
        f = [(x["type"], x["location"]["start"]["value"], x["location"]["end"]["value"], x.get("description", ""))
             for x in j.get("features", [])]
        loc = [s.get("location", {}).get("value", "") for c in j.get("comments", []) for s in c.get("subcellularLocations", [])]
        out[a] = dict(features=f, loc=loc, length=j.get("sequence", {}).get("length"),
                      gene=(j.get("genes") or [{}])[0].get("geneName", {}).get("value", ""))
    json.dump(out, open(UPJ, "w"), indent=0)


def up(a):
    """{'tm': [(s, e)], 'im': [(s, e)], 'side': per-residue array (index = UniProt position), 'L'}"""
    global _UP
    if _UP is None:
        _UP = json.load(open(UPJ))
    v = _UP.get(a)
    if v is None:
        return None
    F = v["features"]
    tm = [(s, e) for t, s, e, _ in F if t == "Transmembrane" and s and e]
    im = [(s, e) for t, s, e, _ in F if t == "Intramembrane" and s and e]
    td = [(s, e, -1 if P_RE.search(n) else (1 if N_RE.search(n) else 0)) for t, s, e, n in F if t == "Topological domain"]
    sig = any(t == "Signal" for t, *_ in F)
    L = max([v["length"] or 0] + [e for t, s, e, _ in F if e]) + 2
    side = np.zeros(L + 1, np.int8)
    for s, e, g in td:
        side[s:e + 1] = g
    ann = side.copy()
    # soluble blocks between TRANSMEM segments
    cuts = sorted(tm); blocks, p = [], 1
    for s, e in cuts:
        blocks.append((p, s - 1)); p = e + 1
    blocks.append((p, L))
    sg = []
    for s, e in blocks:
        x = side[max(s, 0):e + 1]; x = x[x != 0]
        sg.append(int(np.sign(x.sum())) if len(x) else 0)
    if not any(sg):                                    # no TOPO_DOM: signal peptide -> N-terminal block lumenal,
        if sig:                                        # otherwise every soluble block cytosolic
            sg[0] = -1
        else:
            sg = [1] * len(sg)
    for _ in range(len(sg)):
        for k in range(len(sg)):
            if sg[k] == 0:
                nb = [sg[j] for j in (k - 1, k + 1) if 0 <= j < len(sg) and sg[j] != 0]
                if nb: sg[k] = -nb[0]
    for (s, e), g in zip(blocks, sg):
        seg = side[max(s, 0):e + 1]
        seg[seg == 0] = g
    for s, e in tm:
        side[s:e + 1] = 0; ann[s:e + 1] = 0
    return dict(tm=tm, im=im, side=side, ann=ann, L=L, loc=v["loc"], gene=v["gene"])


def masks(acc, pos):
    """per bead: TRANSMEM, INTRAMEM, side (inferred), side (TOPO_DOM annotation only)"""
    tmb = np.zeros(len(acc), bool); imb = np.zeros(len(acc), bool); sd = np.zeros(len(acc), np.int8); sa = np.zeros(len(acc), np.int8)
    for a in set(acc) - {""}:
        u = up(a)
        if u is None:
            continue
        sel = np.where(acc == a)[0]; p = pos[sel]
        for s, e in u["tm"]:
            tmb[sel[(p >= s) & (p <= e)]] = True
        for s, e in u["im"]:
            imb[sel[(p >= s) & (p <= e)]] = True
        ok = (p >= 0) & (p < len(u["side"]))
        sd[sel[ok]] = u["side"][p[ok]]; sa[sel[ok]] = u["ann"][p[ok]]
    return tmb, imb, sd, sa


# ---------------------------------------------------------------- bead loading
def af_entry(name_or_acc, frag=None):
    """(pos UniProt numbering, xyz nm, plddt) of an AFDB v6 model (F1 via tm_graft's index, or a fragment entry name)"""
    if frag is None:
        return tm_graft.af_model(name_or_acc)
    off, size = json.load(open(FRAG))[frag]
    with open(TAR, "rb") as f:
        f.seek(off); data = gzip.decompress(f.read(size)).decode()
    b = gemmi.cif.read_string(data)[0]
    beg = int(b.find_value("_ma_target_ref_db_details.seq_db_align_begin"))
    st = gemmi.make_structure_from_block(b)
    pos, xyz, pl = [], [], []
    for ch in st[0]:
        for res in ch:
            a = res.find_atom("CA", "*")
            if a is not None:
                pos.append(res.seqid.num + beg - 1); xyz.append([a.pos.x / 10, a.pos.y / 10, a.pos.z / 10]); pl.append(a.b_iso)
    return np.array(pos), np.array(xyz), np.array(pl)


def beads_afdb_chains(model, accs):
    """CA beads of an AFDB complex model with (accession, UniProt position, chain) per bead (as jlib_build.beads_afdb)"""
    st = gemmi.read_structure(f"{jb.AFDIR}/{model}.cif.gz"); st.remove_hydrogens()
    if len(accs) < len(st[0]):                     # homodimer models list the gene once
        accs = [accs[i % len(accs)] for i in range(len(st[0]))]
    xyz, acc, pos, chn = [], [], [], []
    for ch, a in zip(st[0], accs):
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            at = res.find_atom("CA", "*") if info is not None and info.is_amino_acid() else None
            if at is None: continue
            xyz.append([at.pos.x / 10, at.pos.y / 10, at.pos.z / 10]); acc.append(a); pos.append(res.seqid.num); chn.append(ch.name)
    return np.array(xyz), np.zeros(len(xyz), np.int8), np.array(acc), np.array(pos), np.array(chn)


def local_tm(acc):
    t = mo.topology()
    return t[acc][0] if acc in t else []


def load(ing, row, group, old):
    src = row["struct_src"]; kind_, ident = src.split(":", 1); notes = []
    if kind_ == "pdb":
        if jb.pdb_file(ident) is None:
            raise RuntimeError("no pdb file")
        xyz, kind, acc, pos, chn = jb.beads_pdb(ident, return_chain=True)
        if group == "tm_no_memlib":                # the existing library was built with the AF graft (jlib_build)
            xyz, kind, acc, pos, g = tm_graft.graft_missing_tm(xyz, kind, acc, pos, chn)
            if g: notes.append(g[:120])
            n0 = len(chn)
            if len(xyz) > n0:                      # grafted beads: chain of the nearest bead of the same accession
                chn = _graft_chains(xyz, acc, pos, np.r_[chn, np.array(["G"] * (len(xyz) - n0))], n0)
        return xyz, kind, acc, pos, chn, {}, notes
    if kind_ == "afdb_complex":
        xyz, kind, acc, pos, chn = beads_afdb_chains(ident, row["accs"])
        return xyz, kind, acc, pos, chn, {}, notes
    # afdb monomer
    a = row["uniprot"]
    m = af_entry(a)
    if m is None:
        return None
    p, x, pl = m
    tm = np.zeros(len(p), bool)
    for s, e in local_tm(a):
        tm |= (p >= s) & (p <= e)
    keep = (pl >= 50) | tm
    if keep.sum() < 10: keep[:] = True
    return x[keep], np.zeros(keep.sum(), np.int8), np.array([a] * keep.sum()), p[keep], np.array(["A"] * keep.sum()), \
        {"plddt_frac_kept": np.float32(keep.mean())}, notes


def _graft_chains(xyz, acc, pos, chn, n0):
    """assign grafted beads to the nearest original chain of the same accession (spatially)"""
    from scipy.spatial import cKDTree
    out = chn.copy()
    for i in range(n0, len(xyz)):
        sel = np.where((acc[:n0] == acc[i]))[0]
        if len(sel):
            out[i] = chn[sel[np.argmin(((xyz[sel] - xyz[i]) ** 2).sum(1))]]
    return out


def load_fragment(a):
    """monomer without library: AFDB fragment containing the TRANSMEM segments (most central)"""
    u = up(a); idx = json.load(open(FRAG))
    names = sorted(k for k in idx if k.split("-")[1] == a)
    if not names:
        return None
    tms = u["tm"] or u["im"]
    lo, hi = min(s for s, e in tms), max(e for s, e in tms)
    best = None
    for n in names:
        fr = int(n.split("-")[2][1:]); beg = 200 * (fr - 1) + 1; end = beg + 1399
        if len(names) == 1: end = 10 ** 6
        if beg <= lo and hi <= end:
            c = min(lo - beg, end - hi)
            if best is None or c > best[0]: best = (c, n)
    if best is None:
        return None
    p, x, pl = af_entry(a, best[1])
    tm = np.zeros(len(p), bool)
    for s, e in u["tm"] + u["im"]:
        tm |= (p >= s) & (p <= e)
    keep = (pl >= 50) | tm
    if keep.sum() < 10: keep[:] = True
    note = f"fragment {best[1].split('-')[2]} res {p.min()}-{p.max()} of {u['L'] - 2}; {keep.sum()} beads kept (pLDDT>=50 + TM)"
    return x[keep], np.zeros(keep.sum(), np.int8), np.array([a] * keep.sum()), p[keep], np.array(["A"] * keep.sum()), \
        {"plddt_frac_kept": np.float32(keep.mean())}, [note]


# ---------------------------------------------------------------- geometry
def helix(n, z0, rise, phase=0.0):
    k = np.arange(n); t = np.deg2rad(100.0) * k + phase
    return np.c_[HRAD * np.cos(t), HRAD * np.sin(t), z0 + rise * k]


def tm_helix(n):
    rise = min(RISE, 6.0 / max(n - 1, 1))
    return helix(n, -rise * (n - 1) / 2, rise)


def im_helices(n, s):
    """INTRAMEM segment of n residues: hairpin(s) of <= 11-residue helices from the surface (|z| 1.9) to ~0.3 nm"""
    k = max(2, int(np.ceil(n / 11))); parts = np.array_split(np.arange(n), k); out = []
    for j, p in enumerate(parts):
        m = len(p); rise = min(RISE, 1.6 / max(m - 1, 1))
        h = helix(m, 0, rise); z = 1.9 - h[:, 2]          # going down from the surface
        if j % 2: z = z[::-1]                             # antiparallel partner comes back up
        h[:, 2] = s * z; out.append(h)
    return out


def hex_slot(p0, placed):
    if not placed:
        return p0
    P = np.array(placed)
    for r in range(0, 40):
        cand = [p0] if r == 0 else [p0 + r * HSEP * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 6 * r, endpoint=False)]
        cand = sorted(cand, key=lambda c: np.linalg.norm(c - p0))
        for c in cand:
            if np.min(np.linalg.norm(P - c, axis=1)) >= HSEP - 1e-6:
                return c
    return p0 + np.array([100.0, 0])


def in_slab_soluble(xyz, memb):
    return int(((np.abs(xyz[:, 2]) < HALF) & ~memb).sum())


def side_ok(xyz, sd, memb):
    m = (sd != 0) & ~memb & (np.abs(xyz[:, 2]) >= HALF)
    return float((np.sign(xyz[m, 2]) == sd[m]).mean()) if m.any() else 1.0


def try_orient(xyz, tmb, imb, sa, relaxed=False, sub=None):
    """membrane_orient.orient with UniProt TOPO_DOM sides; sub: TM beads that define the slab (default all TRANSMEM).
    Accepted if soluble beads in the slab <= membrane residues + 30 (UniProt TM boundaries are shorter than the
    helices; loops, re-entrant and amphipathic helices of multi-pass domains sit in the slab) and >= 80% of the annotated
    soluble beads outside the slab are on their side."""
    ns, ps = sa > 0, sa < 0
    kw = dict(min_score=0.25, min_in=0.45) if relaxed else {}
    out, info = mo.orient(xyz, tmb if sub is None else sub, ns, ps, **kw)
    if out is None:
        return None, info.get("fail", "")
    memb = tmb | imb; nm = int(memb.sum())
    nin = in_slab_soluble(out, memb); so = side_ok(out, sa, memb)
    if nin > nm + 30 or so < 0.8:
        return None, f"rejected (soluble in slab {nin}, membrane residues {nm}, side ok {so:.2f})"
    return out, f"score {info.get('score')} sign from {info.get('sign_from')}; soluble in slab {nin}; side ok {so:.2f}"


USE_INVENTORY_SRC = {"CPX-6121": "8xva has TOMM20 as well; barrel orientation as requested",
                     "CPX-3073": "8io4 resolves the hERG TM domain (orients), 9chp does not"}
BARREL = {"CPX-6121": ("O96008", "Q9NS69")}      # TOM: TOMM40 barrels, TOMM22 cytosolic domain -> +z
RING_PORE = {"GSDMB_x24": "Q8TAX9"}              # 8et1 "GSDMB pore without transmembrane beta-barrel": flat 24-mer ring


def ring_pore(xyz, kind, acc, pos, chn, tmb, barrel_acc):
    """gasdermin pore without its modelled barrel: ring normal (smallest variance) = membrane normal, ring on the cytosolic
    face (nearest bead 2.3 nm from the midplane, strand bases facing the membrane); each chain's unmodelled TRANSMEM
    (beta-stranded) residues are added as one beta-hairpin (two antiparallel CA strands 0.48 nm apart, from z = +2 down to
    -2 and back) below the chain's strand bases, so the 24 hairpins form the transmembrane barrel."""
    X = xyz - xyz.mean(0); n = np.linalg.eigh(np.cov(X.T))[1][:, 0]; X = X @ bs.rot_to_z(n).T
    if X[tmb, 2].mean() > X[:, 2].mean():
        X[:, 1:] *= -1
    X[:, 2] += 2.3 - X[:, 2].min()
    tm_pos = np.concatenate([np.arange(s_, e + 1) for s_, e in up(barrel_acc)["tm"]])
    H = []
    for c in np.unique(chn):
        sel = (chn == c) & (acc == barrel_acc)
        if not sel.any(): continue
        miss = len(set(tm_pos.tolist()) - set(pos[sel].tolist()))
        if miss < 4: continue
        base = X[sel & tmb][:, :2].mean(0) if (sel & tmb).any() else X[sel][:, :2].mean(0)
        t = np.array([-base[1], base[0]]) / max(np.linalg.norm(base), 1e-6)
        for j, m in enumerate((miss - miss // 2, miss // 2)):
            z = np.linspace(2.0, -2.0, m)
            if j: z = z[::-1]
            xy = base + (j - 0.5) * 0.48 * t
            H.append(np.c_[np.full(m, xy[0]), np.full(m, xy[1]), z])
    Y = np.vstack([X] + H); K = np.r_[kind, np.zeros(len(Y) - len(X), np.int8)]
    M = np.r_[np.zeros(len(X), bool), np.ones(len(Y) - len(X), bool)]
    return Y, K, M, len(H), f"ring normal; ring on the cytosolic face; {len(H)} beta strands ({int(M.sum())} unmodelled TRANSMEM residues) built as the barrel"


def orient_barrel(xyz, acc, pos, chn, tmb, sd, barrel_acc, cyto_acc):
    """beta-barrel assemblies: normal = principal direction of the barrel strands (TRANSMEM 'Beta stranded' segments, sign-free
    eigenvector of sum v v^T), midplane at the strands' median height, sign from cyto_acc's cytosolic beads (else all
    cytosol-side beads)"""
    vecs, sb = [], np.zeros(len(acc), bool)
    for c in np.unique(chn):
        sel = np.where((chn == c) & (acc == barrel_acc))[0]
        for s_, e in up(barrel_acc)["tm"]:
            m = sel[(pos[sel] >= s_) & (pos[sel] <= e)]
            if len(m) >= 4:
                v = xyz[m[-1]] - xyz[m[0]]; vecs.append(v / np.linalg.norm(v)); sb[m] = True
    w, Vv = np.linalg.eigh(sum(np.outer(v, v) for v in vecs))
    c0 = xyz[sb].mean(0)
    cands = {"strand direction": Vv[:, -1],                                  # narrow barrels (TOM40)
             "barrel-ring normal": np.linalg.eigh(np.cov((xyz[sb] - c0).T))[1][:, 0],   # wide flat rings (gasdermin pore)
             "assembly flat axis": np.linalg.eigh(np.cov((xyz - xyz.mean(0)).T))[1][:, 0]}
    best = None
    for how, n in cands.items():
        X = (xyz - c0) @ bs.rot_to_z(n).T; X[:, 2] -= np.median(X[sb, 2])
        sc = (np.abs(X[sb, 2]) < HALF).mean() - (np.abs(X[~tmb, 2]) < HALF).mean()
        if best is None or sc > best[0]: best = (sc, how, X)
    sc, how, X = best
    cyt = ((acc == cyto_acc) if cyto_acc else np.ones(len(acc), bool)) & (sd > 0) & ~tmb
    if cyt.any() and X[cyt, 2].mean() < 0:
        X[:, 1:] *= -1
    return X, f"barrel axis ({how}, slab score {sc:.2f}) from {len(vecs)} {barrel_acc} strands"


def two_membranes(xyz, chn, tmb):
    """chains whose TM centroids form two layers > 5 nm apart along the principal axis of the centroids: TM beads of one layer"""
    cs = [c for c in np.unique(chn) if (tmb & (chn == c)).sum() >= 10]
    if len(cs) < 4:
        return None, ""
    C = np.array([xyz[tmb & (chn == c)].mean(0) for c in cs]); C0 = C - C.mean(0)
    w, Vv = np.linalg.eigh(np.cov(C0.T)); p = C0 @ Vv[:, -1]
    o = np.sort(p); gaps = np.diff(o); k = int(np.argmax(gaps))
    if gaps[k] < 5.0:
        return None, ""
    lay = [c for c, q in zip(cs, p) if q <= o[k]]
    if len(lay) * 2 != len(cs):
        return None, ""
    return np.isin(chn, lay), f"{len(lay)} of {len(cs)} chains, layers {gaps[k]:.1f} nm apart"


def reorient_blocks(xyz, pos, tmb, sd):
    """1-2 TM monomer (jlib_reorient_tm): helix-axis normal, soluble blocks >= 30 residues shifted to |z| >= 2.3 on their side"""
    vecs = []
    edges = np.flatnonzero(np.diff(np.r_[0, tmb.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 5: vecs.append(xyz[b - 1] - xyz[a])
    if not vecs:
        return None, "no TM helix"
    n = sum(v if v @ vecs[0] >= 0 else -v for v in vecs)
    X = (xyz - xyz[tmb].mean(0)) @ bs.rot_to_z(n).T
    if (sd > 0).any() and X[sd > 0, 2].mean() < 0:
        X[:, 1:] *= -1
    edges = np.flatnonzero(np.diff(np.r_[0, (~tmb).astype(int), 0])); moved = 0
    for a, b in zip(edges[::2], edges[1::2]):
        bl = np.arange(a, b)
        if len(bl) < 30: continue
        x = sd[bl]; s = int(np.sign(x.sum())) if (x != 0).any() else (1 if X[bl, 2].mean() >= 0 else -1)
        lo = (s * X[bl, 2]).min()
        if lo < 2.3:
            X[bl, 2] += s * (2.3 - lo); moved += len(bl)
    return X, f"helix-axis orientation, {moved} soluble beads shifted out of the slab"


def anchored(xyz, kind, acc, pos, chn, tmb, imb, sd, row):
    memb = tmb | imb
    sol = np.where(~memb)[0]
    # side of every soluble bead: annotation, else chain majority, else global majority, else +1
    s = sd[sol].astype(int).copy()
    glob = int(np.sign(s.sum())) or 1
    for c in np.unique(chn[sol]):
        m = chn[sol] == c; x = s[m]
        cm = int(np.sign(x.sum())) if (x != 0).any() else glob
        x[x == 0] = cm or glob; s[m] = x
    s[s == 0] = glob
    notes = []
    for g in (1, -1):                                         # tiny side groups join the other side
        if 0 < (s == g).sum() < max(20, 0.05 * len(s)):
            notes.append(f"{(s == g).sum()} beads of side {g:+d} kept with the {-g:+d} part"); s[s == g] = -g
    # helices to build: (chain, acc, type, start, end)
    segs = []
    accs_struct = set(acc) - {""}
    for c in np.unique(chn):
        ac = pd.Series(acc[chn == c]); ac = ac[ac != ""]
        if not len(ac): continue
        a = ac.mode()[0]; u = up(a)
        if u is None: continue
        for t, L in (("TM", u["tm"]), ("IM", u["im"])):
            for st_, en in L:
                segs.append((c, a, t, st_, en))
    if not segs:
        # no TM/IM-bearing chain in the structure: helices from the inventory stoichiometry of the TM genes
        st = str(row.get("stoich", "")).split(";")
        for k, (gname, a) in enumerate(zip(row["genes"].split(";"), row["accs"])):
            u = up(a) if a else None
            if u is None or not (u["tm"] or u["im"]): continue
            try: n = int(float(st[k]))
            except (ValueError, IndexError): n = 1
            for j in range(max(n, 1)):
                for t, L in (("TM", u["tm"]), ("IM", u["im"])):
                    for st_, en in L: segs.append((f"absent{k}_{j}", a, t, st_, en))
        notes.append(f"TM subunit(s) absent from the structure: helices from stoichiometry")
    # anchors: the soluble beads of the same chain sequence-adjacent to the segment, before and after it (up to 3 each)
    anc = []
    for c, a, t, st_, en in segs:
        m = sol[(chn[sol] == c) & (acc[sol] == a)]
        A = []
        for mm, d in ((m[pos[m] < st_], st_ - pos[m[pos[m] < st_]]), (m[pos[m] > en], pos[m[pos[m] > en]] - en)):
            if len(mm) and d.min() <= 60:
                k = mm[np.argsort(d)[:3]]; A.append((k, int(s[np.searchsorted(sol, k[0])])))
        if not A and len(m):
            d = np.minimum(np.abs(pos[m] - st_), np.abs(pos[m] - en)); k = m[np.argsort(d)[:3]]
            A.append((k, int(s[np.searchsorted(sol, k[0])])))
        anc.append(A)
    # place each side group
    out = np.zeros_like(xyz); groups = {}
    for g in (1, -1):
        gi = sol[s == g]
        if not len(gi): continue
        A = [xyz[k].mean(0) for AA in anc for (k, sg) in AA if sg == g]
        X = xyz[gi]; cen = X.mean(0); rg = np.sqrt(((X - cen) ** 2).sum(1).mean())
        if A:
            am = np.mean(A, 0); u_ = am - cen
            if np.linalg.norm(u_) > 0.3 * rg:
                n = u_ / np.linalg.norm(u_); how = "anchor"
            else:
                w, Vv = np.linalg.eigh(np.cov((X - cen).T)); n = Vv[:, 0] * (1 if Vv[:, 0] @ u_ >= 0 else -1); how = "flat face"
        else:
            am = cen; w, Vv = np.linalg.eigh(np.cov((X - cen).T)); n = Vv[:, 0]; how = "flat face (no anchor)"
        R = bs.rot_to_z(-g * n)
        Y = (X - am) @ R.T
        Y[:, 2] += g * GAP - (Y[:, 2].min() if g > 0 else Y[:, 2].max())
        out[gi] = Y; groups[g] = (R, am, how)
        notes.append(f"{'+z' if g > 0 else '-z'} part {len(gi)} beads ({how})")
    # helices
    placed, H, nadd_tm, nadd_im, nnew = [], [], 0, 0, 0
    main = 1 if (s == 1).sum() >= (s == -1).sum() else -1
    for (c, a, t, st_, en), AA in zip(segs, anc):
        n = en - st_ + 1
        AA = sorted([x for x in AA if x[1] in groups], key=lambda x: x[1] != main)
        if AA:
            k, sg = AA[0]; R, am, _ = groups[sg]; p0 = ((xyz[k].mean(0) - am) @ R.T)[:2]
        else:
            p0 = np.zeros(2); sg = main
        present = int(((chn == c) & (acc == a) & (pos >= st_) & (pos <= en)).sum())
        nnew += n - present
        if t == "TM":
            q = hex_slot(p0, placed); placed.append(q)
            h = tm_helix(n); h[:, :2] += q; H.append(h); nadd_tm += 1
        else:
            for h in im_helices(n, sg if sg else main):
                q = hex_slot(p0, placed); placed.append(q); h[:, :2] += q; H.append(h)
            nadd_im += 1
    X = np.vstack([out[sol]] + H) if H else out[sol]
    K = np.r_[kind[sol], np.zeros(sum(len(h) for h in H), np.int8)]
    M = np.r_[np.zeros(len(sol), bool), np.ones(len(X) - len(sol), bool)]
    side = "both" if len(groups) == 2 else ("+z" if 1 in groups else "-z")
    notes.append(f"{nadd_tm} TM + {nadd_im} IM segments built ideal ({int(memb.sum())} structure residues remodelled, {nnew} new)")
    return X, K, M, side, nadd_tm + nadd_im, "; ".join(notes)


# ---------------------------------------------------------------- driver
_CHN = None


def save(ing, xyz, kind, extra):
    per = 40 if len(xyz) > 3000 else 30
    c, r = sl.sphere_tree(np.asarray(xyz, float), per=per)
    d = dict(beads=np.asarray(xyz, np.float32), kind=np.asarray(kind, np.int8), st_c=c.astype(np.float32), st_r=r.astype(np.float32))
    d.update(extra)
    os.makedirs(OUT, exist_ok=True)
    if OUT == LIB and os.path.exists(f"{LIB}/{ing}.npz") and not os.path.exists(f"{BAK}/{ing}.npz"):
        os.makedirs(BAK, exist_ok=True); shutil.copy2(f"{LIB}/{ing}.npz", f"{BAK}/{ing}.npz")
    tmp = f"{OUT}/.{ing}.tmroute.tmp.npz"
    np.savez_compressed(tmp, **d); os.replace(tmp, f"{OUT}/{ing}.npz")
    return len(xyz), len(r)


def build(args):
    global _CHN
    ing, group, row = args
    f = f"{LIB}/{ing}.npz"
    srcf = f"{BAK}/{ing}.npz" if os.path.exists(f"{BAK}/{ing}.npz") else f
    old = dict(np.load(srcf)) if os.path.exists(srcf) else None
    nold = len(old["beads"]) if old else 0
    rec = dict(ing_id=ing, group=group, method="skipped", side="", n_tm_helices_added=0, beads=0, spheres=0, old_beads=nold,
               new_beads=0, membrane_residues=0, soluble_in_slab=0, note="")
    try:
        notes = []
        if old is not None and "src" in old and str(old["src"]) != row["struct_src"]:
            if ing in USE_INVENTORY_SRC:
                notes.append(f"library src {old['src']} replaced by inventory struct_src {row['struct_src']} ({USE_INVENTORY_SRC[ing]})")
            else:                                  # keep the library's structure (protein content unchanged)
                notes.append(f"library src {old['src']} kept (inventory struct_src {row['struct_src']})")
                row = dict(row, struct_src=str(old["src"]))
        if row["kind"] == "protein" and old is None:
            L = load_fragment(row["uniprot"])
            if L is None:
                rec["note"] = "no AFDB model" if not any(k.split("-")[1] == row["uniprot"] for k in json.load(open(FRAG))) \
                    and tm_graft.af_model(row["uniprot"]) is None else "no AFDB fragment covers the TM segments"
                return rec
            frag = True
        else:
            L = load(ing, row, group, old); frag = False
            if L is None:
                rec["note"] = "no AFDB model"; return rec
        xyz, kind, acc, pos, chn, extra, n2 = L; notes += n2; _CHN = chn
        tmb, imb, sd, sa = masks(acc, pos); memb = tmb | imb
        # expected TRANSMEM residues of the chains present
        exp = 0
        for c in np.unique(chn):
            ac = pd.Series(acc[chn == c]); ac = ac[ac != ""]
            if len(ac) and up(ac.mode()[0]): exp += sum(e - s + 1 for s, e in up(ac.mode()[0])["tm"])
        frac = tmb.sum() / exp if exp else 0.0
        X = None; method = ""
        if ing in RING_PORE:
            X, K, M, nh, msg = ring_pore(xyz, kind, acc, pos, chn, tmb, RING_PORE[ing]); method = "anchored"; notes.append(msg)
            side = "+z"
        elif ing in BARREL:
            X, msg = orient_barrel(xyz, acc, pos, chn, tmb, sd, *BARREL[ing]); method = "orient"
            nin = in_slab_soluble(X, memb); msg += f"; soluble in slab {nin}; side ok {side_ok(X, sa, memb):.2f}"
            notes.append(msg); M = memb; K = kind; nh = 0
        elif tmb.sum() >= 10 and frac >= 0.3:
            ntm_seg = len(up(acc[tmb][0])["tm"]) if row["kind"] == "protein" else 99
            if row["kind"] == "protein" and ntm_seg <= 2:
                X, msg = reorient_blocks(xyz, pos, tmb, sd); method = "af_fragment" if frag else "orient"
                if X is not None:
                    nin = in_slab_soluble(X, memb); msg += f"; soluble in slab {nin}"
                notes.append(msg)
            else:
                for relaxed in (False, True):
                    X, msg = try_orient(xyz, tmb, imb, sa, relaxed)
                    notes.append(("relaxed: " if relaxed else "orient: ") + msg)
                    if X is not None:
                        method = "af_fragment" if frag else ("orient_relaxed" if relaxed else "orient"); break
                if X is None:                      # assemblies spanning two membranes (gap junctions): slab from one half
                    lay, how = two_membranes(xyz, chn, tmb)
                    if lay is not None:
                        X, msg = try_orient(xyz, tmb, imb, np.where(lay, sa, 0), True, sub=tmb & lay)
                        notes.append(f"one-membrane slab ({how}): " + msg)
                        if X is not None: method = "orient_relaxed"
            M = memb; K = kind; nh = 0
        else:
            notes.append(f"TRANSMEM residues present {int(tmb.sum())}/{exp}")
        if ing in RING_PORE:
            pass
        elif X is None:
            X, K, M, side, nh, msg = anchored(xyz, kind, acc, pos, chn, tmb, imb, sd, row)
            method = "af_fragment" if frag else "anchored"; notes.append(msg)
            if frag: notes.append("anchored construction")
        else:
            m = (sa != 0) & ~M[:len(sa)] & (np.abs(X[:, 2]) >= HALF)
            up_, dn = int((X[m, 2] > 0).sum()), int((X[m, 2] < 0).sum())
            side = "both" if min(up_, dn) > 0.15 * max(up_ + dn, 1) else ("+z" if up_ >= dn else "-z")
        ex = {k: v for k, v in (old or {}).items() if k not in ("beads", "kind", "st_c", "st_r", "st_c30", "st_r30", "membrane", "reoriented")}
        ex.update(extra); ex["src"] = np.array(row["struct_src"]); ex["membrane"] = np.array("TM"); ex["tm_route"] = np.array(method)
        ex["tm_beads"] = M.astype(np.int8)                 # 1 = TRANSMEM / INTRAMEM residue (structure or built)
        nb, ns = save(ing, X, K, ex)
        rec.update(method=method, side=side, n_tm_helices_added=nh, beads=nb, spheres=ns, new_beads=nb - nold if nold else nb,
                   membrane_residues=int(M.sum()), soluble_in_slab=in_slab_soluble(X, M), note="; ".join(notes))
    except Exception as e:
        import traceback
        rec["note"] = "ERROR " + repr(e)[:200] + " " + traceback.format_exc().splitlines()[-3][:150]
    return rec


def main_build(only=None):
    T = pd.read_csv(IDS, sep="\t")
    if only: T = T[T.ing_id.isin(only)]
    I = pd.read_csv(f"{V}/data/derived/ingredients_v0.tsv", sep="\t").set_index("ing_id")
    acc = pd.read_csv(f"{V}/data/derived/inventory_v0_complexed.tsv", sep="\t", usecols=["gene", "uniprot"]).set_index("gene").uniprot
    jobs, allacc = [], set()
    for r in T.itertuples():
        row = I.loc[r.ing_id].to_dict()
        row["accs"] = [acc.get(g, "") if isinstance(acc.get(g, ""), str) else "" for g in row["genes"].split(";")]
        if row["kind"] == "protein" and isinstance(row["uniprot"], str):
            row["accs"] = [row["uniprot"]]
        allacc |= set(row["accs"]); jobs.append((r.ing_id, r.group, row))
    fetch_uniprot(allacc)
    for src in {j[2]["struct_src"] for j in jobs if j[2]["struct_src"].startswith("pdb:")}:
        jb.pdb_file(src.split(":")[1])
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 8))) as ex:
        res = list(ex.map(build, jobs, chunksize=1))
    R = pd.DataFrame(res)
    if only and os.path.exists(LOG) and OUT == LIB:
        old = pd.read_csv(LOG, sep="\t"); R = pd.concat([old[~old.ing_id.isin(R.ing_id)], R])
    R.to_csv(LOG if OUT == LIB else f"{OUT}/tm_route_libs.tsv", sep="\t", index=False)
    print(R.method.value_counts().to_string())
    print(R[R.note.str.startswith("ERROR")].to_string())


def main_refine():
    R = pd.read_csv(LOG, sep="\t")
    ids = list(R.ing_id[R.method != "skipped"])
    env = dict(os.environ, INCLUDE_MEMBRANE="1", NPROC="8", REFINE_LOG=f"{D5}/tm_route_refine_st_log.tsv")
    subprocess.run([sys.executable, f"{os.path.dirname(os.path.abspath(__file__))}/lib_refine_st.py"] + ids, env=env, check=True)
    for i in ids:
        z = np.load(f"{LIB}/{i}.npz"); R.loc[R.ing_id == i, "spheres"] = len(z["st_r"]); R.loc[R.ing_id == i, "beads"] = len(z["beads"])
    R.to_csv(LOG, sep="\t", index=False)


def main_verify():
    R = pd.read_csv(LOG, sep="\t"); rows = []
    for r in R[R.method != "skipped"].itertuples():
        z = np.load(f"{OUT}/{r.ing_id}.npz"); B = z["beads"]; M = z["tm_beads"].astype(bool)
        zt = B[M, 2] if M.any() else np.zeros(1)
        rows.append(dict(ing_id=r.ing_id, method=r.method, membrane="membrane" in z.files, beads=len(B), spheres=len(z["st_r"]),
                         sol_in_slab=int(((np.abs(B[:, 2]) < HALF) & ~M).sum()), tm_zmin=round(float(zt.min()), 1),
                         tm_zmax=round(float(zt.max()), 1), sol_up=int((B[~M, 2] >= HALF).sum()), sol_dn=int((B[~M, 2] <= -HALF).sum()),
                         st_zmin=round(float((z["st_c"][:, 2] - z["st_r"]).min()), 1), st_zmax=round(float((z["st_c"][:, 2] + z["st_r"]).max()), 1)))
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(T.to_string(index=False))
    print("all membrane key:", bool(T.membrane.all()), "| libraries with soluble beads in the slab:", int((T.sol_in_slab > 0).sum()),
          "(by method:", T[T.sol_in_slab > 0].method.value_counts().to_dict(), ")")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "build"
    if what == "build":
        main_build(sys.argv[2:] or None)
    elif what == "refine":
        main_refine()
    elif what == "verify":
        main_verify()
