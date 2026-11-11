"""Capacity-aware redistribution of unplaced cytosolic copies to neighbouring tiles (v7; DECISION 2026-11-10: if the
cell-wide cytosol placement of the full pack is below v6, move cytosolic copies that a crowded tile could not place to
its neighbours with spare capacity, instead of an exclusion rule).

Source tiles: cytosol jobs placed < SRC_MAX of their mass. Their unplaced cytosol copies (per library) are split over the
26-neighbour tiles whose cytosol placement is >= RCV_MIN, in proportion to the neighbour's cytosol volume (tile measure).
Bookkeeping: in the source tiles the cytosol job_n is lowered to the placed count (the copies moved away); counts.npz gets
the extra copies on the receivers; only the receivers are repacked (jwc pack on that tile list, their neighbours as
fixed obstacles). Pass-1 files are kept: wc/counts_v7_pass1.npz and wc/tiles_v7_pass1/<i>.npz for every changed tile.
usage: JWC_CONFIG=... python jwc_redistribute.py plan|apply      (then: JWC_CONFIG=... JWC_TILES=<W>/redistribute_tiles.npy jwc.py pack)"""
import os, sys, json, shutil
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jwc

SRC_MAX = 0.97
RCV_MIN = 0.995
W = jwc.W


def plan():
    M = pd.read_parquet(f"{W}/tiles.parquet"); T = M[["ti", "tj", "tk"]].values; key = {tuple(t): i for i, t in enumerate(T)}
    L = json.load(open(f"{W}/libs.json")); mass = np.array(L["mass_da"]); ci = jwc.RNAMES.index("cytosol")
    frac = np.ones(len(M)); miss = {}
    for ti in range(len(M)):
        f = f"{W}/tiles/{ti}.npz"
        if not os.path.exists(f): continue
        z = np.load(f); m = z["job_region"] == ci
        if not m.any(): continue
        n, p, l = z["job_n"][m], z["job_placed"][m], z["job_lib"][m]
        e = (n * mass[l]).sum(); frac[ti] = (p * mass[l]).sum() / e if e > 0 else 1.0
        if frac[ti] < SRC_MAX: miss[ti] = {int(a): int(b) for a, b in zip(l, n - p) if b > 0}
    vol = M["cytosol"].values; add = {}
    for ti, d in miss.items():
        nb = [j for j in jwc._neigh_of(T[ti], key) if frac[j] >= RCV_MIN and vol[j] > 0]
        if not nb: continue
        w = vol[nb] / vol[nb].sum()
        rng = np.random.default_rng(ti)
        for li, k in d.items():
            for j, x in zip(nb, rng.multinomial(k, w)):
                if x: add.setdefault(j, {}); add[j][li] = add[j].get(li, 0) + int(x)
    moved = sum(sum(d.values()) for d in miss.values())
    S = dict(src_tiles=len(miss), receivers=len(add), copies_moved=int(moved), src_max=SRC_MAX, rcv_min=RCV_MIN,
             cytosol_frac_tiles_below_src=int((frac < SRC_MAX).sum()))
    json.dump(dict(summary=S, miss={str(k): v for k, v in miss.items()}, add={str(k): v for k, v in add.items()}),
              open(f"{W}/redistribute_plan.json", "w"))
    print(S, flush=True)
    return miss, add


def apply():
    P = json.load(open(f"{W}/redistribute_plan.json")); miss = {int(k): v for k, v in P["miss"].items()}; add = {int(k): v for k, v in P["add"].items()}
    ci = jwc.RNAMES.index("cytosol")
    arch = f"{W}/tiles_v7_pass1"; os.makedirs(arch, exist_ok=True)
    if not os.path.exists(f"{W}/counts_v7_pass1.npz"): shutil.copy(f"{W}/counts.npz", f"{W}/counts_v7_pass1.npz")
    for ti in set(miss) | set(add):
        if not os.path.exists(f"{arch}/{ti}.npz"): shutil.copy(f"{W}/tiles/{ti}.npz", f"{arch}/{ti}.npz")
    # sources: job_n -> placed for the moved cytosol copies (tile files and counts)
    c = dict(np.load(f"{W}/counts_v7_pass1.npz")); t, l, r, n = c["tile"].astype(np.int64), c["lib"].astype(np.int64), c["region"].astype(np.int64), c["n"].astype(np.int64)
    idx = {(a, b): i for i, (a, b, rr) in enumerate(zip(t, l, r)) if rr == ci}
    for ti, d in miss.items():
        if ti in add: continue                                  # a tile is never both (receivers have spare capacity)
        z = dict(np.load(f"{arch}/{ti}.npz")); m = z["job_region"] == ci
        z["job_n"] = np.where(m, z["job_placed"], z["job_n"]).astype(np.int32)
        np.savez(f"{W}/tiles/.{ti}.tmp.npz", **z); os.replace(f"{W}/tiles/.{ti}.tmp.npz", f"{W}/tiles/{ti}.npz")
        for li, k in d.items():
            i = idx.get((ti, int(li)))
            if i is not None: n[i] -= int(k)
    extra = []
    for ti, d in add.items():
        for li, k in d.items():
            i = idx.get((ti, int(li)))
            if i is not None: n[i] += int(k)
            else: extra.append((ti, int(li), ci, int(k)))
        os.remove(f"{W}/tiles/{ti}.npz")                        # archived in tiles_v7_pass1; repacked with the extra copies
    if extra:
        E = np.array(extra, np.int64); t, l, r, n = (np.r_[t, E[:, 0]], np.r_[l, E[:, 1]], np.r_[r, E[:, 2]], np.r_[n, E[:, 3]])
    o = np.lexsort((l, t))
    np.savez_compressed(f"{W}/counts.npz", tile=t[o].astype(np.int32), lib=l[o].astype(np.int32), region=r[o].astype(np.int16), n=n[o].astype(np.int32))
    np.save(f"{W}/redistribute_tiles.npy", np.array(sorted(add), np.int64))
    print(f"applied: {len(miss)} sources lowered, {len(add)} receivers to repack ({W}/redistribute_tiles.npy)", flush=True)


if __name__ == "__main__":
    {"plan": plan, "apply": apply}[sys.argv[1]]()
