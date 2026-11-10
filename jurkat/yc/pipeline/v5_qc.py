"""v5 QC on packed tiles (test tiles or the full pack): shell bands, open face, nucleolus fill, placement fractions, slices.

usage: v5_qc.py select            -> prints and saves data/derived/v5/test_tiles.npy (8 tiles: nucleolus, face, mito, PM)
       v5_qc.py check [tiles...]  -> checks on the given (default: test) tiles; images data/derived/qc/v5_*.png,
                                     json data/derived/v5/qc_tiles.json
"""
import os, sys, json, numpy as np, pandas as pd
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3, jpack_test as jp
from config import LIB

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"; W = f"{D}/wc"; Q = f"{D}/qc"; O = f"{D}/v5"
RN = jp.RNAMES


def select():
    M = pd.read_parquet(f"{W}/tiles.parquet"); z = np.load(f"{D}/jgeom_cell1.npz"); G = z["G"]
    pick = []
    def take(order, k):
        for t in order:
            if t not in pick and k > 0: pick.append(int(t)); k -= 1
    take(M.sort_values("nucleolus", ascending=False).tile.values, 2)
    zf = G[c3.G_ZFACE]; meta = json.load(open(f"{W}/tiles_meta.json")); lo0 = np.array(meta["lo0"])
    zlo = M.tk.values * 500.0 + lo0[2]
    face = M[(zlo <= zf) & (zlo + 500 > zf)]
    take(face.sort_values("cytosol", ascending=False).tile.values[:1], 1)
    take(face.sort_values("mem_PM", ascending=False).tile.values, 1)            # face edge: side-wall PM meets the face
    take(M.sort_values("matrix", ascending=False).tile.values, 2)
    take(M[zlo > zf + 2000].sort_values("mem_PM", ascending=False).tile.values, 2)
    os.makedirs(O, exist_ok=True); np.save(f"{O}/test_tiles.npy", np.array(pick))
    print("test tiles:", pick); print(M.loc[pick, ["tile", "ti", "tj", "tk", "cytosol", "nucleoplasm", "nucleolus", "matrix", "mem_PM"]].to_string())
    return pick


def check(tiles):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    z = np.load(f"{D}/jgeom_cell1.npz"); cell = c3.Cell(z["G"], z["pores"]); G = cell.G; zf = G[c3.G_ZFACE] if G[c3.G_FACE] > 0 else -1e9
    L = json.load(open(f"{W}/libs.json")); libs = L["libs"]; mass = np.array(L["mass_da"])
    M = pd.read_parquet(f"{W}/tiles.parquet"); meta = json.load(open(f"{W}/tiles_meta.json")); lo0 = np.array(meta["lo0"])
    BR = {}
    def br(l):
        if l not in BR:
            zz = np.load(f"{LIB}/{libs[l]}.npz"); BR[l] = float((np.linalg.norm(zz["st_c"], axis=1) + zz["st_r"]).max())
        return BR[l]
    exp = {}; plc = {}; band = []; face = dict(pm_below_face=0, any_centre_below_face=0, pm_on_face=0)
    allP, allR = [], []
    for ti in tiles:
        f = f"{W}/tiles/{ti}.npz"
        if not os.path.exists(f): continue
        t = np.load(f)
        for l, r, n, p in zip(t["job_lib"], t["job_region"], t["job_n"], t["job_placed"]):
            exp[RN[r]] = exp.get(RN[r], 0) + n * mass[l]; plc[RN[r]] = plc.get(RN[r], 0) + p * mass[l]
        reg = np.repeat(t["job_region"].astype(int), t["job_placed"]); pos = t["pos"].astype(np.float64); typ = t["types"]
        allP.append(pos); allR.append(reg)
        S = cell.mems(pos)
        face["any_centre_below_face"] += int((pos[:, 2] < zf).sum())
        pm = reg == RN.index("PM")
        face["pm_below_face"] += int((pm & (pos[:, 2] < zf)).sum())
        face["pm_on_face"] += int((pm & (pos[:, 2] < zf + 20) & (S[:, c3.M_PM] > 40)).sum())   # PM protein near the face, away from the side wall
        for k, s_ in enumerate(jp.SHELL_REG):
            m = reg == jp.NSH0 + k
            if not m.any(): continue
            u = s_[3] * S[m, s_[1]]
            hi = np.array([s_[5] + (br(l) if s_[6] else 0) for l in typ[m]])
            band.append(dict(shell=s_[0], n=int(m.sum()), in_band=float(((u >= s_[4] - 0.5) & (u <= hi + 0.5)).mean()),
                             u_median=float(np.median(u)), u_p95=float(np.percentile(u, 95)), hi_median=float(np.median(hi))))
    fr = {r: dict(expected_pg=round(exp[r] / 6.022e23 * 1e12 * 1e3, 3), frac=round(plc[r] / max(exp[r], 1), 3)) for r in exp}
    B = pd.DataFrame(band).groupby("shell").apply(lambda d: pd.Series(dict(n=d.n.sum(), in_band=(d.in_band * d.n).sum() / d.n.sum(),
                                                                            u_median=d.u_median.median(), u_p95=d.u_p95.max())), include_groups=False) if band else pd.DataFrame()
    print("placed mass fraction per region (expected in fg):"); print(json.dumps(fr, indent=1))
    print("shell bands:"); print(B.to_string() if len(B) else "none")
    print("open face:", face)
    # nucleolus density on the tiles
    P = np.concatenate(allP); R = np.concatenate(allR)
    vnl = M.loc[[t for t in tiles if os.path.exists(f"{W}/tiles/{t}.npz")], "nucleolus"].sum() / 1e9
    nl_mass = plc.get("nucleolus", 0) / 6.022e23 * 1e12
    nld = nl_mass / vnl * 1e3 if vnl > 0 else float("nan")
    print(f"nucleolus on these tiles: {vnl:.3f} um3, placed {nl_mass * 1e3:.2f} fg -> {nld:.0f} mg/ml (packed protein only)")
    json.dump(dict(tiles=[int(t) for t in tiles], regions=fr, shells=B.reset_index().to_dict("records") if len(B) else [], face=face,
                   nucleolus_mg_per_ml=nld), open(f"{O}/qc_tiles.json", "w"), indent=1, default=float)
    # slices: one per tile, 30 nm slab through the tile centre (x-z), instances coloured by region class, labels underneath
    os.makedirs(Q, exist_ok=True)
    cols = {"cytosol": "#bbbbbb", "nucleoplasm": "#6b8fd6", "nucleolus": "#d62728", "matrix": "#2ca02c", "PM": "#000000",
            "sh_PM_cyto": "#ff7f0e", "sh_PM_cortex": "#e377c2", "sh_PM_exo": "#8c564b", "sh_NE_lamina": "#9467bd", "MITO_IM": "#17becf",
            "MITO_OM": "#bcbd22", "NE": "#1f77b4"}
    for ti in tiles:
        f = f"{W}/tiles/{ti}.npz"
        if not os.path.exists(f): continue
        t = np.load(f); reg = np.repeat(t["job_region"].astype(int), t["job_placed"]); pos = t["pos"]
        lo = M.loc[ti, ["ti", "tj", "tk"]].values * 500.0 + lo0; yc = lo[1] + 250
        m = np.abs(pos[:, 1] - yc) < 15
        g = np.arange(0, 500, 4.0) + 2; X, Z = np.meshgrid(lo[0] + g, lo[2] + g, indexing="ij")
        lab = cell.labels(np.stack([X.ravel(), np.full(X.size, yc), Z.ravel()], 1)).reshape(X.shape)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(lab.T, origin="lower", extent=[lo[0], lo[0] + 500, lo[2], lo[2] + 500], cmap="tab20", vmin=0, vmax=19, alpha=0.35)
        for r in np.unique(reg[m]):
            k = m & (reg == r); nm = RN[r]
            ax.scatter(pos[k, 0], pos[k, 2], s=3, c=cols.get(nm, "#444444"), label=f"{nm} ({k.sum()})")
        if lo[2] <= zf < lo[2] + 500: ax.axhline(zf, color="red", lw=0.8, ls="--", label="open face (edge of EM data)")
        ax.set_xlim(lo[0], lo[0] + 500); ax.set_ylim(lo[2], lo[2] + 500); ax.set_xlabel("x (nm)"); ax.set_ylabel("z (nm)")
        ax.set_title(f"v5 tile {ti}: x-z slab |y - {yc:.0f}| < 15 nm"); ax.legend(fontsize=6, markerscale=3, loc="upper right")
        fig.savefig(f"{Q}/v5_tile{ti}_xz.png", dpi=130); plt.close(fig)
    print("images:", f"{Q}/v5_tile*_xz.png")


if __name__ == "__main__":
    if sys.argv[1] == "select": select()
    else:
        tl = [int(x) for x in sys.argv[2:]] or list(np.load(f"{O}/test_tiles.npy"))
        check(tl)
