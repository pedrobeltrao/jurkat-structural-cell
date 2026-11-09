"""Pack the N tiles with the most matrix volume (final v4 settings, written to wc/tiles for reuse by the full pack) and
report placed mass fraction per region plus the top unplaced matrix species.  usage: matrix_probe.py pack|report [N]"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jwc, jpack_test as jp
W = jwc.W; N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
M = pd.read_parquet(f"{W}/tiles.parquet"); sel = np.argsort(-M["matrix"].values)[:N]
if sys.argv[1] == "pack":
    jwc.step_pack(sel)
else:
    L = json.load(open(f"{W}/libs.json")); mass = np.array(L["mass_da"]); e = {}; p = {}; miss = {}
    for ti in sel:
        z = np.load(f"{W}/tiles/{ti}.npz")
        for li, r, n, k in zip(z["job_lib"], z["job_region"], z["job_n"], z["job_placed"]):
            e[r] = e.get(r, 0) + n * mass[li]; p[r] = p.get(r, 0) + k * mass[li]
            if jp.RNAMES[r] == "matrix": miss[li] = miss.get(li, 0) + (n - k) * mass[li]
    print({jp.RNAMES[r]: round(p[r] / e[r], 3) for r in sorted(e)})
    print("matrix volume in tiles (um3):", round(M["matrix"].values[sel].sum() / 1e9, 2))
    for li, m in sorted(miss.items(), key=lambda x: -x[1])[:10]: print(L["libs"][li], round(m / e[jp.RNAMES.index("matrix")], 3))
