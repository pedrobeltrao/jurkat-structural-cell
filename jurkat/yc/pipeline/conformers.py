"""Conformational motion modes from the pool predictions (exploration branch).

Single-chain AlphaFold library types are animated in the viewer along their principal motions in the pool ensembles
(AF3 predictions of each protein in many random multi-protein pools, about 600 per protein):

  1. kept_residues()   this module, pipeline env: for each AF monomer type, the residue numbers its library beads
                       come from (AFDB model, pLDDT >= 50 or TM, as in build_structs.process_af) -> RES_F
  2. pool_modes.py     standalone, needs foldcomp (prot_dashboard env): samples pool conformers, superposes their
                       kept residues on the library beads, PCA -> modes and a conformer path in mode space -> MODES_F
  3. webenc.encode_modes  modes into the viewer library (library/modes.bin + per-type fields in types.json)
"""
import os, gzip, tarfile
import numpy as np
import pandas as pd
from config import LIB, DATA, AFDB_TAR, WORK, NPROC

CDIR = f"{WORK}/conformers"
RES_F = f"{CDIR}/kept_residues.npz"
MODES_F = f"{CDIR}/pool_modes.npz"


def _job(args):
    acc, lid, data, tm, beads = args
    import gemmi
    st = gemmi.make_structure_from_block(gemmi.cif.read_string(data)[0])
    rows = [(res.seqid.num, res.find_atom("CA", "*").b_iso) for ch in st[0] for res in ch if res.find_atom("CA", "*")]
    arr = np.array(rows); seq, pl = arr[:, 0].astype(int), arr[:, 1]
    keep = pl >= 50
    for a, b, _ in tm:
        keep |= (seq >= a) & (seq <= b)
    if keep.sum() < 10:
        keep[:] = True
    return lid, acc, (seq[keep] if keep.sum() == len(beads) else None)


def kept_residues(types, log=print):
    """types: viewer types.json rows. Writes RES_F: per AF monomer type its accession, kept residue numbers and the
    library beads (nm, library frame, residue order)."""
    from joblib import Parallel, delayed
    from build_structs import parse_ranges
    topo = pd.read_csv(f"{DATA}/uniprot_topology.tsv", sep="\t").set_index("Entry")
    want = {t["uniprot"]: t["id"] for t in types
            if t["src"] == "AF" and (t.get("oligomer") or 1) <= 1 and t.get("uniprot") and ";" not in t["uniprot"]}
    jobs = []
    with tarfile.open(AFDB_TAR) as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if not name.endswith(".cif.gz") or "-F1-" not in name:
                continue
            acc = name.split("-")[1]
            if acc in want:
                lid = want[acc]
                tm = parse_ranges(topo.loc[acc, "Transmembrane"], "TRANSMEM") if acc in topo.index else []
                beads = np.load(f"{LIB}/{lid}.npz")["beads"]
                jobs.append((acc, lid, gzip.decompress(tf.extractfile(m).read()).decode(), tm, beads))
    res = Parallel(n_jobs=NPROC, batch_size=16)(delayed(_job)(j) for j in jobs)
    ok = [(lid, acc, seq) for lid, acc, seq in res if seq is not None]
    os.makedirs(CDIR, exist_ok=True)
    np.savez_compressed(RES_F, lid=np.array([r[0] for r in ok]), acc=np.array([r[1] for r in ok]),
                        off=np.cumsum([0] + [len(r[2]) for r in ok]), res=np.concatenate([r[2] for r in ok]),
                        beads=np.concatenate([np.load(f"{LIB}/{r[0]}.npz")["beads"] for r in ok]).astype(np.float32))
    log(f"kept residues: {len(ok)} of {len(want)} AF monomer types ({len(want) - len(jobs)} not in the AFDB tar, "
        f"{len(jobs) - len(ok)} bead-count mismatches)")
    return RES_F


CAP_NM = float(os.environ.get("MOTION_CAP", 0))
                # nm: ensembles whose 95th-percentile bead excursion is larger are scaled down to it; the viewer also
                # clamps every bead to this. 0 = no cap (the shipped modes; swinging termini then move several nm)


def encode_modes(modes_f, out_dir, log=print):
    """Viewer files from pool_modes.py output, for an already encoded library in out_dir:
      library/modes.bin   u32 words: per bead of each animated type, N_MODES words of int8 x, y, z (mode vectors,
                          per-mode scale folded into the coefficients), in the library's Morton bead order;
                          then per type its conformer path, nc x N_MODES f32 coefficients (nm per int8 unit)
      library/modes.json  {"n_modes", "cap" (nm, 0 = none), "types": {type index: [mode word offset, coef word offset, n conformers, max nm]}}
    """
    import json
    from webenc import morton_order
    z = dict(np.load(modes_f))       # npz members decompress on every access
    m = int(z["n_modes"])
    tj = json.load(open(f"{out_dir}/library/types.json"))
    index = {t["id"]: i for i, t in enumerate(tj["types"])}
    words, coefs, entries, capped, stats = [], [], {}, 0, []
    nmw = 0
    for k, lid in enumerate(z["lid"]):
        ti = index.get(str(lid))
        if ti is None:
            continue
        V = z["V"][:, z["boff"][k]:z["boff"][k + 1]].astype(np.float64)        # m, n, 3 (residue order)
        C = z["C"][z["coff"][k]:z["coff"][k + 1]].astype(np.float64)            # nc, m
        b = np.load(f"{LIB}/{lid}.npz")["beads"].astype(np.float64)
        if len(b) != V.shape[1] or len(b) != tj["types"][ti]["nb"]:
            continue
        V = V[:, morton_order(b)]
        disp = np.einsum("km,mnc->knc", C, V)
        ex = np.linalg.norm(disp, axis=2).max(0)                                  # per bead, over the path
        p95 = float(np.percentile(ex, 95))
        if CAP_NM and p95 > CAP_NM:
            C *= CAP_NM / p95; ex *= CAP_NM / p95; p95 = CAP_NM; capped += 1
        mx = float(min(ex.max(), CAP_NM) if CAP_NM else ex.max()); stats.append((p95, float(np.median(ex)), float(ex.max())))
        s = np.abs(V).reshape(m, -1).max(1); s[s == 0] = 1.0
        q = np.clip(np.round(V / s[:, None, None] * 127), -127, 127).astype(np.int8)    # m, n, 3
        w = np.zeros((V.shape[1], m, 4), np.int8); w[:, :, :3] = q.transpose(1, 0, 2)
        words.append(w.view(np.uint32).reshape(-1))
        coefs.append((C * s[None, :] / 127.0).astype(np.float32).reshape(-1))
        entries[ti] = [nmw, None, len(C), round(mx, 3)]
        nmw += w.shape[0] * m
    cw = nmw
    for (ti, e), c in zip(entries.items(), coefs):
        e[1] = cw; cw += len(c)
    blob = np.concatenate(words + [c.view(np.uint32) for c in coefs]).astype(np.uint32)
    blob.tofile(f"{out_dir}/library/modes.bin")
    json.dump({"n_modes": m, "cap": CAP_NM, "types": {str(k): v for k, v in entries.items()}}, open(f"{out_dir}/library/modes.json", "w"))
    st = np.array(stats)
    log(f"modes: {len(entries)} animated types, {blob.nbytes / 1e6:.1f} MB; bead excursion along the path: median per "
        f"type {np.median(st[:, 1]):.2f} nm, 95th percentile {np.median(st[:, 0]):.2f} nm; {capped} types scaled to {CAP_NM} nm; max excursion median {np.median(st[:, 2]):.2f} nm, largest {st[:, 2].max():.1f} nm")
