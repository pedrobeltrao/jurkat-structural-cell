"""Insert the NPC pre-pass (jnpc.py) into a finished whole-cell pack without repacking the whole cell.

Steps (run after `jwc.py counts` and `jwc.py index`, which pick up prepass/npc_cell1.npz and drop NPC from the random jobs):
tiles whose box comes within R_NPC + MARGIN of an NPC centre are moved to wc/tiles_pre_npc/ and repacked by
jwc.step_pack with the NPCs as fixed pre-pass obstacles; all other packed tiles stay and act as neighbour obstacles
(none of them reaches an NPC, so nothing packed earlier can overlap one). The repacked tiles use the new counts.npz draw
(the multinomial split changes when a library row is dropped; per-tile totals are the same in expectation).
Optionally (PATCH_LIBS=<file of ing_ids>, PATCH_OLD_COUNTS=<counts used for the pack>) every tile holding a library
rebuilt after the pack is repacked too, so no stored instance refers to changed geometry. Run counts with
JWC_LIBS_FROM=<libs.json of the pack> so that library indices stored in the kept tiles stay valid.
usage: jwc_patch_npc.py [--dry]"""
import os, sys, json, shutil, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jwc

R_NPC = 75.0 + 36.0     # nm: library ring radius 75, half height 36 -> bounding radius ~83; use the safe sum
MARGIN = 30.0           # largest packed sphere-tree reach beyond its centre tile is covered by the tile halo (40 nm)


def main(dry=False):
    W = jwc.W
    M = pd.read_parquet(f"{W}/tiles.parquet"); meta = json.load(open(f"{W}/tiles_meta.json"))
    lo = M[["ti", "tj", "tk"]].values * jwc.TILE + np.array(meta["lo0"]); hi = lo + jwc.TILE
    P = np.load(f"{jwc.D}/prepass/npc_cell1.npz")["pos"].astype(float)
    hit = np.zeros(len(M), bool)
    for p in P:
        d = np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=1)
        hit |= d < R_NPC + MARGIN
    print(f"{len(P)} NPCs touch {int(hit.sum())} of {len(M)} tiles", flush=True)
    if os.environ.get("PATCH_LIBS"):                     # libraries rebuilt after the pack: repack every tile that holds one
        ids = set(open(os.environ["PATCH_LIBS"]).read().split())
        L = json.load(open(f"{W}/libs.json"))["libs"]; li = [i for i, l in enumerate(L) if l in ids]
        for cf in (f"{W}/counts.npz", os.environ.get("PATCH_OLD_COUNTS", f"{W}/counts.npz")):
            c = np.load(cf); hit[np.unique(c["tile"][np.isin(c["lib"], li)])] = True
        print(f"+ tiles holding the {len(li)} rebuilt libraries: {int(hit.sum())} tiles in total", flush=True)
    sel = np.flatnonzero(hit)
    if dry: return
    bak = f"{W}/tiles_pre_npc"; os.makedirs(bak, exist_ok=True)
    for ti in sel:
        f = f"{W}/tiles/{ti}.npz"
        if os.path.exists(f): shutil.move(f, f"{bak}/{ti}.npz")
    np.save(f"{W}/npc_patch_tiles.npy", sel)
    jwc.step_pack(sel)


if __name__ == "__main__":
    main("--dry" in sys.argv)
