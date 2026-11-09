"""A/B test of lumen-aligned orientations: pack the N tiles with the most ER lumen with JWC_NO_LUMEN_ALIGN set (copied to
wc/tiles_ab_off) and unset (left in wc/tiles for the full pack); report placed mass fraction per volume region.
usage: lumen_ab.py off|on|report [N]"""
import os, sys, json, shutil, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jwc, jpack_test as jp
W = jwc.W; N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
M = pd.read_parquet(f"{W}/tiles.parquet"); sel = np.argsort(-M["ER_lumen"].values)[:N]
if sys.argv[1] in ("off", "on"):
    jwc.step_pack(sel)
    if sys.argv[1] == "off":
        os.makedirs(f"{W}/tiles_ab_off", exist_ok=True)
        for ti in sel: shutil.move(f"{W}/tiles/{ti}.npz", f"{W}/tiles_ab_off/{ti}.npz")
else:
    mass = np.array(json.load(open(f"{W}/libs.json"))["mass_da"])
    for d in ("tiles_ab_off", "tiles"):
        e = {}; p = {}
        for ti in sel:
            z = np.load(f"{W}/{d}/{ti}.npz")
            for li, r, n, k in zip(z["job_lib"], z["job_region"], z["job_n"], z["job_placed"]):
                e[r] = e.get(r, 0) + n * mass[li]; p[r] = p.get(r, 0) + k * mass[li]
        print(d, {jp.RNAMES[r]: round(p[r] / e[r], 3) for r in sorted(e)})
