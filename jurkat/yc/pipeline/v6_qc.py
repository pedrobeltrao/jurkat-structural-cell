"""v6 test tiles: placement of the new and enlarged regions before the full pack.

usage: v6_qc.py select        -> data/derived/v6/test_tiles.npy (10 tiles: 2 speckle, 2 NE/pores, 2 Golgi, 2 side-wall PM,
                                 2 endosome-rich)
       v6_qc.py pack          -> packs them with jwc.step_pack into wc/tiles (no packed neighbours), log wc/pack_log.tsv
       v6_qc.py check         -> per-region placement (mass) and band checks -> data/derived/v6/qc_tiles.json
       v6_qc.py clean         -> moves the test tiles to data/derived/v6/test_tiles_pack/ (before the full pack)
"""
import os, sys, json, shutil, numpy as np, pandas as pd
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3, jpack_test as jp

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"; W = f"{D}/wc"; O = f"{D}/v6"
RN = jp.RNAMES


def select():
    M = pd.read_parquet(f"{W}/tiles.parquet"); pick = []
    def take(order, k):
        for t in order:
            if t not in pick and k > 0: pick.append(int(t)); k -= 1
    take(M.sort_values("speckle", ascending=False).tile.values, 2)
    take(M.sort_values("mem_NE", ascending=False).tile.values, 2)
    take(M.sort_values("mem_GOLGI", ascending=False).tile.values, 2)
    meta = json.load(open(f"{W}/tiles_meta.json")); zlo = M.tk.values * 500.0 + meta["lo0"][2]
    take(M[zlo > -5000].sort_values("mem_PM", ascending=False).tile.values, 2)
    take(M.sort_values("mem_ENDO", ascending=False).tile.values, 2)
    os.makedirs(O, exist_ok=True); np.save(f"{O}/test_tiles.npy", np.array(pick))
    print("test tiles:", pick)
    print(M.loc[pick, ["tile", "ti", "tj", "tk", "cytosol", "nucleoplasm", "speckle", "mem_NE", "mem_GOLGI", "mem_PM", "mem_ENDO"]].to_string())


def pack():
    import jwc
    jwc.step_pack(np.load(f"{O}/test_tiles.npy"))


def check():
    tl = list(np.load(f"{O}/test_tiles.npy"))
    L = json.load(open(f"{W}/libs.json")); mass = np.array(L["mass_da"])
    z = np.load(f"{D}/jgeom_cell1.npz"); cell = c3.Cell(z["G"], z["pores"])
    exp, plc = {}, {}; out = dict(tiles=[int(t) for t in tl])
    npc_pos = []
    for t in tl:
        f = f"{W}/tiles/{t}.npz"
        if not os.path.exists(f): continue
        q = np.load(f)
        for l, r, n, p in zip(q["job_lib"], q["job_region"], q["job_n"], q["job_placed"]):
            exp[r] = exp.get(r, 0) + n * mass[l]; plc[r] = plc.get(r, 0) + p * mass[l]
        reg = np.repeat(q["job_region"], q["job_placed"])
        npc_pos.append(q["pos"][reg == RN.index("sh_NPC")])
    out["regions"] = {RN[r]: dict(expected_fg=round(exp[r] / 6.022e23 * 1e15, 3), placed_frac=round(plc[r] / max(exp[r], 1), 3))
                      for r in sorted(exp)}
    P = np.concatenate(npc_pos) if npc_pos else np.zeros((0, 3))
    if len(P):
        k = np.array([c3.in_pore(p[0], p[1], p[2], cell.pores, 150.5) for p in P.astype(np.float64)])
        lab = cell.labels(P.astype(np.float64)); m = cell.mems(P.astype(np.float64))[:, c3.M_NE]
        out["sh_NPC"] = dict(n=int(len(P)), within_150nm_of_pore_axis=round(float((k >= 0).mean()), 4),
                             cytoplasmic_face=round(float((lab == c3.CYTO).mean()), 3), nuclear_face=round(float((lab == c3.NUCP).mean()), 3),
                             ne_dist_nm_median=round(float(np.median(m)), 1), ne_dist_nm_p95=round(float(np.percentile(m, 95)), 1))
    os.makedirs(O, exist_ok=True); json.dump(out, open(f"{O}/qc_tiles.json", "w"), indent=1); print(json.dumps(out, indent=1))


def clean():
    d = f"{O}/test_tiles_pack"; os.makedirs(d, exist_ok=True)
    for t in np.load(f"{O}/test_tiles.npy"):
        f = f"{W}/tiles/{t}.npz"
        if os.path.exists(f): shutil.move(f, f"{d}/{t}.npz")
    if os.path.exists(f"{W}/pack_log.tsv"): shutil.move(f"{W}/pack_log.tsv", f"{d}/pack_log_test.tsv")
    print("test tiles moved to", d)


if __name__ == "__main__":
    {"select": select, "pack": pack, "check": check, "clean": clean}[sys.argv[1]]()
