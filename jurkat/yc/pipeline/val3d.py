"""Validation of the packed 3D whole cell."""
import glob, os, json
import numpy as np, pandas as pd
import numba as nb
from scipy.spatial import cKDTree
import cell3d as c3, tiles3d as t3, pack3d as p3

from config import LIB


def load_instances(tiles, raw):
    T, P, Q, TI, JOB = [], [], [], [], []
    mr_pts, mr_tile = [], []
    for i, t in enumerate(tiles):
        z = np.load(f"{raw}/{t3.tile_name(t)}.npz")
        T.append(z["types"]); P.append(z["pos"]); Q.append(z["quat"]); TI.append(np.full(len(z["types"]), i, np.int32))
        JOB.append(pd.DataFrame(dict(tile=i, lib=z["job_lib"], region=z["job_region"], n=z["job_n"], placed=z["job_placed"])))
        if len(z["mrna_pts"]):
            mr_pts.append(z["mrna_pts"]); mr_tile.append(np.full(len(z["mrna_pts"]), i, np.int32))
    return (np.concatenate(T), np.concatenate(P), np.concatenate(Q), np.concatenate(TI), pd.concat(JOB, ignore_index=True),
            np.concatenate(mr_pts) if mr_pts else np.zeros((0, 3), np.float32),
            np.concatenate(mr_tile) if mr_tile else np.zeros(0, np.int32))


def lib_beads(ids):
    off = [0]; B, K, MASS, SRC = [], [], [], []
    for i in ids:
        z = np.load(f"{LIB}/{i}.npz")
        b = z["beads"].astype(np.float64); k = z["kind"].astype(np.int8)
        B.append(b); K.append(k); off.append(off[-1] + len(b))
        MASS.append(float((np.where(k == 0, 110.0, 330.0)).sum()))
        SRC.append("PDB" if ("pdb" in z.files or i in ("NPC", "nucleosome", "tRNA")) else "AF")
    return np.array(off, np.int64), np.concatenate(B), np.concatenate(K), np.array(MASS), np.array(SRC)


@nb.njit(cache=True)
def beads_of(idx, types, pos, quat, boff, B, out_pts, out_inst):
    M = np.empty((3, 3)); q = np.empty(4); t = np.empty(3)
    k = 0
    for ii in range(idx.shape[0]):
        i = idx[ii]
        for a in range(4):
            q[a] = quat[i, a]
        p3.quat_mat(q, M)
        for a in range(3):
            t[a] = pos[i, a]
        a0 = boff[types[i]]; a1 = boff[types[i] + 1]
        p3.transform(B[a0:a1], M, t, out_pts[k:k + a1 - a0])
        out_inst[k:k + a1 - a0] = i
        k += a1 - a0
    return k
