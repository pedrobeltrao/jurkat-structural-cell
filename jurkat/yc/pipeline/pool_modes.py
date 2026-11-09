"""Principal motion modes of each protein's pool ensemble, on the residues the viewer draws (exploration branch).

Standalone: needs only numpy and foldcomp, so it runs in an environment that has foldcomp (e.g. prot_dashboard).

For every protein in RES_F (conformers.kept_residues):
  - sample N_SAMPLE pool structures (fixed per-protein seed), Cα of the kept residues, Å -> nm
  - drop conformers whose mean pLDDT over these residues is below MIN_PLDDT; skip the protein if fewer than MIN_CONF
  - superpose each conformer on the library beads (Kabsch, all kept residues)
  - PCA of the displacements from the ensemble mean -> N_MODES modes (unit vectors over all bead coordinates) and
    each conformer's coefficients; the conformers are ordered into a closed nearest-neighbour path in mode space,
    which the viewer follows (smooth interpolation from one sampled conformer to the next)

Usage: python pool_modes.py RES_F OUT_F [--procs N] [--limit N]
"""
import argparse, os, time, zlib
from multiprocessing import Pool
import numpy as np

DB = os.environ.get("POOL_DB", "")      # foldcomp database of the pool predictions
N_SAMPLE = 64
MIN_PLDDT = 60.0      # mean pool pLDDT over the drawn residues; well-folded proteins sit at 65-85
MIN_CONF = 16
N_MODES = 4
_R = {}


def records():
    names = [ln.split("\t")[1] for ln in open(f"{DB}.lookup")]
    out = {}
    for n in names:
        out.setdefault(n.rsplit("_", 1)[-1], []).append(n)
    return {u: sorted(v) for u, v in out.items()}


def kabsch(X, Y):
    """X superposed on Y (both n x 3)"""
    xc, yc = X.mean(0), Y.mean(0)
    U, S, Vt = np.linalg.svd((X - xc).T @ (Y - yc))
    d = np.sign(np.linalg.det(U @ Vt))
    return (X - xc) @ (U @ np.diag([1, 1, d]) @ Vt) + yc


def path_order(C):
    """closed nearest-neighbour tour through the conformers, starting nearest the ensemble mean"""
    left = list(range(len(C))); cur = int(np.argmin((C ** 2).sum(1))); order = [cur]; left.remove(cur)
    while left:
        d = ((C[left] - C[cur]) ** 2).sum(1); cur = left[int(np.argmin(d))]; order.append(cur); left.remove(cur)
    return np.array(order)


def _init(rec, R):
    _R.update(rec=rec, **R)


def process(i):
    import foldcomp
    acc, res, Y = _R["acc"][i], _R["res"][_R["off"][i]:_R["off"][i + 1]], _R["beads"][_R["off"][i]:_R["off"][i + 1]].astype(np.float64)
    names = _R["rec"].get(acc.lower(), [])
    if len(names) < MIN_CONF:
        return i, None, f"{len(names)} pool structures"
    rng = np.random.default_rng(zlib.crc32(acc.encode()))
    pick = names if len(names) <= N_SAMPLE else [names[k] for k in sorted(rng.choice(len(names), N_SAMPLE, replace=False))]
    X, P = [], []
    with foldcomp.open(DB, ids=pick) as db:
        for _, pdb in db:
            ca = {int(l[22:26]): (float(l[30:38]), float(l[38:46]), float(l[46:54]), float(l[60:66]))
                  for l in pdb.splitlines() if l[:4] == "ATOM" and l[12:16] == " CA "}
            if not all(r in ca for r in res):
                continue
            a = np.array([ca[r] for r in res])
            X.append(a[:, :3] / 10.0); P.append(a[:, 3].mean())
    P = np.array(P)
    X = [x for x, p in zip(X, P) if p >= MIN_PLDDT]
    if len(X) < MIN_CONF:
        return i, None, f"{len(X)} conformers with pLDDT >= {MIN_PLDDT:g}"
    X = np.stack([kabsch(x, Y) for x in X])
    D = (X - X.mean(0)).reshape(len(X), -1)
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    m = min(N_MODES, len(S))
    C = U[:, :m] * S[:m]
    o = path_order(C)
    tot = (S ** 2).sum()
    return i, dict(V=Vt[:m].reshape(m, -1, 3).astype(np.float32), C=C[o].astype(np.float32),
                   var=(S[:m] ** 2 / tot).astype(np.float32) if tot > 0 else np.zeros(m, np.float32),
                   rms=float(np.sqrt(tot / len(X) / len(res))), n=len(X)), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res_f"); ap.add_argument("out_f")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    z = np.load(a.res_f)
    R = dict(acc=z["acc"], off=z["off"], res=z["res"], beads=z["beads"])
    n = len(R["acc"]) if not a.limit else min(a.limit, len(R["acc"]))
    t0 = time.time()
    out, notes = {}, {}
    with Pool(a.procs, initializer=_init, initargs=(records(), R)) as pool:
        for k, (i, r, note) in enumerate(pool.imap_unordered(process, range(n), chunksize=4), 1):
            if r is None:
                notes[note.split(" ", 1)[1] if note[0].isdigit() else note] = notes.get(note.split(" ", 1)[1] if note[0].isdigit() else note, 0) + 1
            else:
                out[i] = r
            if k % 500 == 0:
                print(f"  {k}/{n} ({time.time() - t0:.0f} s)", flush=True)
    idx = sorted(out)
    nb = [len(out[i]["V"][0]) for i in idx]; nc = [len(out[i]["C"]) for i in idx]
    np.savez_compressed(a.out_f, lid=z["lid"][idx], n_modes=N_MODES,
                        boff=np.cumsum([0] + nb), coff=np.cumsum([0] + nc),
                        V=np.concatenate([np.pad(out[i]["V"], ((0, N_MODES - len(out[i]["V"])), (0, 0), (0, 0))) for i in idx], axis=1).astype(np.float16),
                        C=np.concatenate([np.pad(out[i]["C"], ((0, 0), (0, N_MODES - out[i]["C"].shape[1]))) for i in idx]),
                        var=np.stack([np.pad(out[i]["var"], (0, N_MODES - len(out[i]["var"]))) for i in idx]),
                        rms=np.array([out[i]["rms"] for i in idx]), n_conf=np.array([out[i]["n"] for i in idx]))
    v = np.stack([out[i]["var"].sum() for i in idx]) if idx else np.zeros(0)
    print(f"{len(idx)} of {n} proteins with modes in {time.time() - t0:.0f} s; variance in {N_MODES} modes: median "
          f"{np.median(v):.2f}; per-residue rms spread median {np.median([out[i]['rms'] for i in idx]):.3f} nm; skipped: {notes}")


if __name__ == "__main__":
    main()
