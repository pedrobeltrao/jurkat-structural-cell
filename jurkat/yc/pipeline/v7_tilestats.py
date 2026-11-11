"""Per-region placed fractions of a set of packed tiles (test packs): python v7_tilestats.py <out_dir> [regions v6|v7]"""
import os, sys, json, numpy as np
W = sys.argv[1]
os.environ["JWC_REGIONS"] = sys.argv[2] if len(sys.argv) > 2 else "v6"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jpack_test as jp
L = json.load(open(f"{W}/libs.json")); mass = np.array(L["mass_da"]); RN = L.get("rnames", jp.RNAMES)
e, p, ne, np_ = {}, {}, {}, {}
fs = [f for f in os.listdir(f"{W}/tiles") if f.endswith(".npz") and not f.startswith(".")]
for f in fs:
    z = np.load(f"{W}/tiles/{f}")
    for l, r, n, k in zip(z["job_lib"], z["job_region"], z["job_n"], z["job_placed"]):
        e[r] = e.get(r, 0) + n * mass[l]; p[r] = p.get(r, 0) + k * mass[l]; ne[r] = ne.get(r, 0) + n; np_[r] = np_.get(r, 0) + k
out = {RN[r]: dict(exp_pg=round(e[r] / 6.022e11, 4), frac=round(p[r] / max(e[r], 1), 3), copies=int(ne[r]), copy_frac=round(np_[r] / max(ne[r], 1), 3)) for r in sorted(e)}
tot = dict(tiles=len(fs), frac=round(sum(p.values()) / sum(e.values()), 4))
json.dump(dict(total=tot, regions=out), open(f"{W}/tilestats.json", "w"), indent=1)
print(tot); [print(f"  {k:16s} {v['frac']:.3f}  ({v['exp_pg']:.4f} pg, {v['copies']:,} copies, copy frac {v['copy_frac']:.3f})") for k, v in out.items()]
