"""Requested load per region, v6 pack counts vs v7 dry-run counts: volume regions mg/ml, membranes / shells pg per um2.
usage: v7_capacity.py  -> data/derived/v7/capacity_v6_v7.tsv"""
import os, sys, json, numpy as np, pandas as pd
V = (__import__("os").environ.get("VCELL_ROOT", ".") + ""); D = f"{V}/data/derived"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["JWC_REGIONS"] = "v6"
import importlib, jpack_test as jp6
RN6, NV6, NSH6, SM6 = list(jp6.RNAMES_V6), 9, 16, dict(jp6.SHELL_MEMNAME)
def load(W, RN, NV, NSH0, SM):
    M = pd.read_parquet(f"{W}/tiles.parquet"); L = json.load(open(f"{W}/libs.json")); mass = np.array(L["mass_da"])
    c = np.load(f"{W}/counts.npz"); m = np.bincount(c["region"], weights=c["n"] * mass[c["lib"]], minlength=len(RN)) / 6.022e11
    out = {}
    for r, name in enumerate(RN):
        if r < NV: meas = M[name].sum() / 1e9; unit = "mg/ml"; val = m[r] / max(meas, 1e-9) * 1e3
        else:
            mem = name if r < NSH0 else SM[name]; meas = M["mem_" + mem].sum() / 1e6; unit = "fg/um2"; val = m[r] / max(meas, 1e-9) * 1e3
        out[name] = (round(float(m[r]), 4), round(float(meas), 2), unit, round(float(val), 2))
    return out
a = load(f"{D}/wc", RN6, NV6, NSH6, SM6)
S6 = json.load(open(f"{D}/wc/summary_v6.json"))["regions"] if os.path.exists(f"{D}/wc/summary_v6.json") else json.load(open(f"{D}/wc/summary.json"))["regions"]
os.environ["JWC_REGIONS"] = "v7"; jp7 = importlib.reload(jp6)
b = load(f"{D}/v7/wc_v7", list(jp7.RNAMES), jp7.NV, jp7.NSH0, dict(jp7.SHELL_MEMNAME))
rows = []
for n in list(dict.fromkeys(list(a) + list(b))):
    x, y = a.get(n), b.get(n)
    rows.append(dict(region=n, v6_req_pg=x[0] if x else None, v6_measure=x[1] if x else None, v6_load=x[3] if x else None,
                     v6_placed_frac=S6.get(n, {}).get("frac"), v7_req_pg=y[0] if y else None, v7_measure=y[1] if y else None,
                     unit=(y or x)[2], v7_load=y[3] if y else None))
T = pd.DataFrame(rows); T.to_csv(f"{D}/v7/capacity_v6_v7.tsv", sep="\t", index=False); print(T.to_string(index=False))
