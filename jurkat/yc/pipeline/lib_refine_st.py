"""Adaptive sphere trees for the packing libraries.

build_structs.save uses one sphere per ~30 beads. That suits compact domains but inflates extended or partly disordered
chains: their sphere-tree volume reached 15-25x the protein volume (HMGs, hnRNPs, SR proteins, H1 750 nm3 for 22 kDa;
mass-weighted mean 3.6x over the top-300 soluble species), which jammed the nucleoplasm in the first whole-cell pack.
Here each library keeps per=30 if its sphere-tree volume is <= MAX_RATIO x the protein volume (1.21 nm3/kDa); otherwise
per = 15, 8, 5 is tried and the first that meets the bound (else the finest) is kept. The original tree is kept in the
npz as st_c30/st_r30. Membrane-oriented libraries are skipped unless INCLUDE_MEMBRANE=1 (their beads are stored in the
membrane frame, so the refined tree keeps z along the normal); pre-pass polymer libraries are skipped unless listed explicitly.
usage: lib_refine_st.py [ids...]   (default: every ingredient library in data/derived/ingredients_v0.tsv)
Output: rewrites lib/<id>.npz; data/derived/lib_refine_st_log.tsv"""
import os, sys, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LIB
import structlib as sl

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
MAX_RATIO = 2.5
PERS = (15, 8, 5)


def st_volume(c, r, h=0.5):
    lo = (c - r[:, None]).min(0); hi = (c + r[:, None]).max(0)
    g = np.stack(np.meshgrid(*[np.arange(a, b, h) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    inside = np.zeros(len(g), bool)
    for cc, rr in zip(c, r):
        inside |= ((g - cc) ** 2).sum(1) < rr * rr
    return inside.sum() * h ** 3


def refine(args):
    ing, kda = args
    f = f"{LIB}/{ing}.npz"
    try:
        z = dict(np.load(f))
    except Exception as e:
        return ing, kda, np.nan, np.nan, 0, 0, f"error {e}"
    if "membrane" in z and not os.environ.get("INCLUDE_MEMBRANE"):   # membrane beads share the oriented frame of st_c (z = normal)
        return ing, kda, np.nan, np.nan, len(z["st_r"]), len(z["st_r"]), "skip_membrane"
    if "st_c30" in z:                                   # refined by an earlier run
        return ing, kda, np.nan, np.nan, len(z["st_r30"]), len(z["st_r"]), "done_before"
    if len(z["beads"]) > 20000:                         # large assemblies (ribosome, NPC, ...) are compact; the volume grid is too costly
        return ing, kda, np.nan, np.nan, len(z["st_r"]), len(z["st_r"]), "skip_large"
    c0, r0 = (z["st_c30"], z["st_r30"]) if "st_c30" in z else (z["st_c"], z["st_r"])
    vp = max(kda * 1.21, 1.0)
    v0 = st_volume(c0.astype(float), r0.astype(float)); rat0 = v0 / vp
    if rat0 <= MAX_RATIO:
        return ing, kda, rat0, rat0, len(r0), len(r0), "kept"
    best = (c0, r0, rat0)
    for per in PERS:
        c, r = sl.sphere_tree(z["beads"].astype(float), per=per)
        rat = st_volume(c, r) / vp
        best = (c, r, rat)
        if rat <= MAX_RATIO:
            break
    z["st_c30"], z["st_r30"] = c0.astype(np.float32), r0.astype(np.float32)
    z["st_c"], z["st_r"] = best[0].astype(np.float32), best[1].astype(np.float32)
    tmp = f"{LIB}/.{ing}.tmp.npz"
    np.savez_compressed(tmp, **z); os.replace(tmp, f)
    return ing, kda, rat0, best[2], len(r0), len(best[1]), f"per{per}"


def main():
    if len(sys.argv) > 1:
        I = pd.read_csv(f"{V}/data/derived/ingredients_v0.tsv", sep="\t").set_index("ing_id")
        jobs = [(i, float(I.mass_da.get(i, 22000.0)) / 1e3) for i in sys.argv[1:]]
    else:
        I = pd.read_csv(f"{V}/data/derived/ingredients_v0.tsv", sep="\t")
        I = I[I.ing_id.map(lambda i: os.path.exists(f"{LIB}/{i}.npz"))]
        jobs = list(zip(I.ing_id, I.mass_da / 1e3))
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 24))) as ex:
        rows = list(ex.map(refine, jobs, chunksize=16))
    T = pd.DataFrame(rows, columns=["ing_id", "kda", "ratio30", "ratio_new", "nsph30", "nsph_new", "action"])
    T.to_csv(os.environ.get("REFINE_LOG", f"{V}/data/derived/lib_refine_st_log.tsv"), sep="\t", index=False)
    print(T.action.value_counts().to_string())
    ok = T.ratio30.notna()
    print(f"ratio median {T.ratio30[ok].median():.2f} -> {T.ratio_new[ok].median():.2f}; "
          f"mean {T.ratio30[ok].mean():.2f} -> {T.ratio_new[ok].mean():.2f}; spheres {T.nsph30.sum():,} -> {T.nsph_new.sum():,}")


if __name__ == "__main__":
    main()
