"""Pull soluble domains of single/double-pass TM monomers out of the bilayer.

build_structs.process_af orients a TM monomer by its helix axis alone. For single-pass proteins AlphaFold often packs the
TM helix flat against the soluble domain, so after orientation the domain lies beside the helix inside the bilayer slab
(FDFT1, CLIC1, TMX1, SLC3A2, VAPA, LMAN1, ...: 708 monomers, 34% of the membrane-monomer mass had in-slab beads > 2.5x their
TM residues + 30). Such proteins took up bilayer area that real ones do not (ER membrane placed 83%, ENDO 68% in pack v2),
and the cytosolic/luminal domains were buried in the membrane.
Here monomers with 1-2 UniProt TRANSMEM segments are rebuilt from the AFDB model with the same bead selection and
orientation, then every soluble block of >= MIN_BLOCK consecutive kept residues is translated rigidly along the normal
to its topological side (UniProt TOPO_DOM; for proteins without any: N-terminal block lumenal if a signal peptide, else
the largest block cytoplasmic; remaining blocks opposite to their neighbour across a TM) until all its beads sit at |z| >= SLAB. The flexible linker to the helix is stretched (not modelled).
Multi-pass proteins (>= 3 TM) are left as built: many (sterol enzymes, transporters) are largely embedded.
Backup of the replaced libraries: data/derived/lib_bak_membrane_st30 (copied before this step).
usage: jlib_reorient_tm.py [ing_ids]           Output: rewrites lib/<id>.npz; data/derived/lib_reorient_tm_log.tsv"""
import os, sys, gzip, tarfile, numpy as np, pandas as pd, gemmi
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_structs as bs
from config import LIB

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
TAR = __import__("os").environ.get("AFDB_HUMAN_TAR", "UP000005640_9606_HUMAN_v6.tar")
SLAB = 2.3          # nm: bilayer half-thickness (2.0) + clearance
MIN_BLOCK = 30      # residues; shorter loops/tails stay where AlphaFold put them
PLDDT_MIN = 50      # as build_structs.process_af
HALF = 2.0          # nm, cell3d.HALF (for the in-slab audit only)


def side_of(seq, topo):
    s = np.zeros(len(seq), np.int8)
    for a, b, note in topo:
        m = (seq >= a) & (seq <= b)
        s[m] = 1 if ("ytoplasm" in note or "atrix" in note) else -1   # +z = N side (cytoplasm; matrix for the IM, whose
                                                         # SDF rises into the matrix); lumenal / extracellular / IMS / nucleoplasm
                                                         # (NE is one surface whose gradient points to the cytoplasm) -> -z
    return s


def rebuild(args):
    acc, ing, data, tm_ranges, topo, signal = args
    st = gemmi.make_structure_from_block(gemmi.cif.read_string(data)[0]); st.remove_hydrogens()
    rows = [(a.pos.x / 10, a.pos.y / 10, a.pos.z / 10, r.seqid.num, a.b_iso)
            for ch in st[0] for r in ch for a in [r.find_atom("CA", "*")] if a is not None]
    arr = np.array(rows); xyz, seq, pl = arr[:, :3], arr[:, 3].astype(int), arr[:, 4]
    tm = np.zeros(len(seq), bool)
    for a, b, _ in tm_ranges:
        tm |= (seq >= a) & (seq <= b)
    keep = (pl >= PLDDT_MIN) | tm
    if keep.sum() < 10: keep[:] = True
    xyz, seq, tm = xyz[keep], seq[keep], tm[keep]
    vecs = [xyz[(seq >= a) & (seq <= b)][[0, -1]] for a, b, _ in tm_ranges if ((seq >= a) & (seq <= b)).sum() >= 5]
    if not vecs:
        return ing, "no_tm_beads", 0, 0, 0
    vecs = np.array([v[1] - v[0] for v in vecs]); n = sum(v if v @ vecs[0] >= 0 else -v for v in vecs)
    xyz = (xyz - xyz[tm].mean(0)) @ bs.rot_to_z(n).T
    sd = side_of(seq, topo)
    if (sd > 0).any() and xyz[sd > 0, 2].mean() < 0:
        xyz[:, 1:] *= -1
    in0 = int((np.abs(xyz[~tm, 2]) < HALF).sum())
    # soluble blocks: runs of consecutive non-TM beads (in chain order)
    edges = np.flatnonzero(np.diff(np.r_[0, (~tm).astype(int), 0]))
    blocks = [np.arange(a, b) for a, b in zip(edges[::2], edges[1::2])]
    signs = []
    for bl in blocks:
        ann = sd[bl]; s = int(np.sign(ann.sum())) if (ann != 0).any() else 0
        signs.append(s)
    if not any(signs) and blocks:                        # no TOPO_DOM at all: signal peptide -> N-terminal block lumenal,
        if signal and tm[0] == False:                    # else the largest soluble block cytoplasmic
            signs[0] = -1
        else:
            signs[int(np.argmax([len(b) for b in blocks]))] = 1
    for _ in range(len(blocks)):                         # unannotated blocks: opposite to a neighbour across a TM
        for k in range(len(blocks)):
            if signs[k] == 0:
                nb = [signs[j] for j in (k - 1, k + 1) if 0 <= j < len(blocks) and signs[j] != 0]
                if nb: signs[k] = -nb[0]
    if not (sd != 0).any() and sum(len(b) * s * xyz[b, 2].mean() for b, s in zip(blocks, signs)) < 0:
        xyz[:, 1:] *= -1                                 # inferred sides: turn the protein over rather than drag domains across
    moved = 0
    for bl, s in zip(blocks, signs):
        if len(bl) < MIN_BLOCK: continue
        if s == 0:
            s = 1 if xyz[bl, 2].mean() >= 0 else -1
        lo = (s * xyz[bl, 2]).min()
        if lo < SLAB:
            xyz[bl, 2] += s * (SLAB - lo); moved += len(bl)
    in1 = int((np.abs(xyz[~tm, 2]) < HALF).sum())
    old = dict(np.load(f"{LIB}/{ing}.npz"))
    extra = {k: v for k, v in old.items() if k not in ("beads", "kind", "st_c", "st_r", "st_c30", "st_r30")}
    extra["membrane"] = "TM"; extra["reoriented"] = np.int8(1)
    tmp = f".{ing}.tmp"
    bs.save(tmp, xyz, np.zeros(len(xyz), np.int8), extra)
    os.replace(f"{LIB}/{tmp}.npz", f"{LIB}/{ing}.npz")
    return ing, "ok", in0, in1, moved


def main():
    I = pd.read_csv(f"{V}/data/derived/ingredients_v0.tsv", sep="\t")
    I = I[(I.kind == "protein") & I.transmembrane.astype(bool) & I.uniprot.notna()]
    topo = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/data/uniprot_topology.tsv", sep="\t").set_index("Entry")
    want = {}
    for r in I.itertuples():
        if r.uniprot not in topo.index or not os.path.exists(f"{LIB}/{r.ing_id}.npz"): continue
        tmr = bs.parse_ranges(topo.loc[r.uniprot, "Transmembrane"], "TRANSMEM")
        if 1 <= len(tmr) <= 2:
            want[r.uniprot] = (r.ing_id, tmr, bs.parse_ranges(topo.loc[r.uniprot, "Topological domain"], "TOPO_DOM"), r.signal_frac >= 0.5)
    if len(sys.argv) > 1:                                # restrict to the given ing_ids (tests)
        want = {k: v for k, v in want.items() if v[0] in sys.argv[1:]}
    print("single/double-pass TM monomers:", len(want), flush=True)
    jobs = []
    with tarfile.open(TAR) as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if name.endswith(".cif.gz") and "-F1-" in name and name.split("-")[1] in want:
                acc = name.split("-")[1]; ing, tmr, tp, sig = want[acc]
                jobs.append((acc, ing, gzip.decompress(tf.extractfile(m).read()).decode(), tmr, tp, sig))
    print("models found:", len(jobs), flush=True)
    with ProcessPoolExecutor(24) as ex:
        res = list(ex.map(rebuild, jobs, chunksize=4))
    T = pd.DataFrame(res, columns=["ing_id", "status", "in_slab_soluble_before", "in_slab_soluble_after", "beads_moved"])
    T.to_csv(f"{V}/data/derived/lib_reorient_tm_log" + (os.environ.get("LOG_SUFFIX", "_test") if len(sys.argv) > 1 else "") + ".tsv", sep="\t", index=False)
    print(T.status.value_counts().to_string())
    print(f"soluble beads inside the slab: {T.in_slab_soluble_before.sum():,} -> {T.in_slab_soluble_after.sum():,}; "
          f"libraries with blocks moved: {(T.beads_moved > 0).sum()}")


if __name__ == "__main__":
    main()
