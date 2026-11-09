"""Crowding and tracer diffusion predicted from the packed Jurkat cell (validation backlog item 1a).

For sample boxes (BOX nm cubes at tile centres that lie fully inside one compartment):
  1. occupancy at residue level on a VOX nm grid: every placed instance (random-packed tiles, pre-pass polymers, mRNA
     beads) is expanded to its residue/nucleotide beads (library beads rotated by the instance quaternion, w-x-y-z);
     a voxel is occupied if within R_BEAD of a bead. R_BEAD is calibrated so that the bead union of a library sample
     has the protein's volume (1.21 nm3/kDa), i.e. the occupied fraction phi is a macromolecular volume fraction.
  2. probe-accessible volume fraction f(a): voxels farther than a from any occupied voxel (probe centre positions).
  3. obstruction-limited long-time diffusion D_obs/D0 for a probe of radius a: lattice random walk of probe centres in
     the accessible space ("blind ant"); D from the MSD slope at long lag (box edges reflect).
  4. hydrodynamic factor: Tokuyama-Oppenheim short-time self-diffusion for hard spheres at the box's phi.
  Predicted D/D0 = (D_obs/D0) x (D_hyd/D0). Nothing here includes binding or the fluid-phase viscosity of cytosol
  (about 1.2-1.4x water), so it is an upper bound for an inert tracer.
Experimental GFP references (D0 in water about 87 um2/s): cytoplasm about 3-4x slower (Swaminathan 1997 Biophys J),
nucleus similar to cytoplasm, mitochondrial matrix 20-30 um2/s (Partikian 1998 J Cell Biol), ER lumen 5-10 um2/s
(Dayel 1999 Biophys J).
usage: crowding.py [n_boxes_per_region]   Output: data/derived/validation/crowding.tsv (+ json)"""
import os, sys, json, time, numpy as np, pandas as pd, numba as nb
from scipy import ndimage

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
sys.path.insert(0, f"{V}/jurkat/yc/pipeline")
from config import LIB
W = f"{V}/data/derived/wc"; TILES = os.environ.get("TILES_DIR", f"{W}/tiles"); OUT = f"{V}/data/derived/validation"
TILE = 500.0; BOX = 150.0; VOX = 0.5; MARGIN = 25.0
PROBES = [0.0, 1.0, 1.6, 2.3, 3.0]     # nm; 2.3 ~ GFP (Rh 2.4); frozen obstacles overstate blocking of larger probes
REGIONS = ["cytosol", "nucleoplasm", "matrix", "ER_lumen"]
_beads = {}


def lib_beads(lid):
    if lid not in _beads:
        z = np.load(f"{LIB}/{lid}.npz"); _beads[lid] = z["beads"].astype(np.float32)
    return _beads[lid]


def qmat(q):
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def calibrate_r(names, mass_kda, n=40, seed=0):
    """bead radius r such that the bead union volume matches 1.21 nm3/kDa (median over a library sample)"""
    rng = np.random.default_rng(seed); idx = rng.choice(len(names), min(n, len(names)), replace=False)
    rs = np.arange(0.0, 1.5, 0.05); ratios = []
    for i in idx:
        b = lib_beads(names[i]); lo = b.min(0) - 2; dims = np.ceil((b.max(0) + 2 - lo) / VOX).astype(int) + 1
        if np.prod(dims) > 4e7: continue
        g = np.ones(dims, bool); ij = np.floor((b - lo) / VOX).astype(int); g[ij[:, 0], ij[:, 1], ij[:, 2]] = False
        d = ndimage.distance_transform_edt(g) * VOX                 # same grid and rule as the analysis
        ratios.append([(d <= r).sum() * VOX ** 3 / (mass_kda[i] * 1.21) for r in rs])
    med = np.median(np.array(ratios), 0)
    return float(np.interp(1.0, med, rs)), dict(zip(np.round(rs, 3).tolist(), np.round(med, 3).tolist()))


def gather(box_lo, M, key, L, PL):
    """world-space beads of every instance near the box"""
    lo, hi = box_lo - MARGIN, box_lo + BOX + MARGIN
    pts = []
    t = np.floor((box_lo + BOX / 2 - M.attrs["lo0"]) / TILE).astype(int)
    for d in np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 3):
        j = key.get(tuple(t + d))
        if j is None: continue
        srcs = []
        f = f"{TILES}/{j}.npz"
        if os.path.exists(f):
            z = np.load(f); srcs.append((z["types"], z["pos"], z["quat"], L))
        p0, p1 = M.attrs["poff"][j], M.attrs["poff"][j + 1]
        if p1 > p0:
            srcs.append((M.attrs["ptypes"][p0:p1], M.attrs["ppos"][p0:p1], M.attrs["pquat"][p0:p1], PL))
        for ty, pos, qu, names in srcs:
            pos = np.asarray(pos); k = np.where(np.all((pos > lo - 30) & (pos < hi + 30), axis=1))[0]
            for i in k:
                b = lib_beads(names[ty[i]]) @ qmat(np.asarray(qu[i], float)).T + pos[i]
                b = b[np.all((b > lo) & (b < hi), axis=1)]
                if len(b): pts.append(b)
        m0, m1 = M.attrs["moff"][j], M.attrs["moff"][j + 1]
        if m1 > m0:
            mp = np.asarray(M.attrs["mpts"][m0:m1], np.float32); mp = mp[np.all((mp > lo) & (mp < hi), axis=1)]
            if len(mp): pts.append(("mrna", mp))
    prot = [p for p in pts if not isinstance(p, tuple)]; mr = [p[1] for p in pts if isinstance(p, tuple)]
    return (np.concatenate(prot) if prot else np.zeros((0, 3), np.float32)), (np.concatenate(mr) if mr else np.zeros((0, 3), np.float32))


@nb.njit(cache=True)
def walk(acc, starts, nsteps, rec_every, seed):
    np.random.seed(seed)
    nx, ny, nz = acc.shape; nw = starts.shape[0]; nrec = nsteps // rec_every
    pos = starts.copy(); disp = np.zeros((nw, 3), np.int64); msd = np.zeros(nrec)
    D = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    for s in range(nsteps):
        for w in range(nw):
            k = np.random.randint(6)
            x = pos[w, 0] + D[k, 0]; y = pos[w, 1] + D[k, 1]; z = pos[w, 2] + D[k, 2]
            if x < 0 or y < 0 or z < 0 or x >= nx or y >= ny or z >= nz: continue
            if acc[x, y, z]:
                pos[w, 0] = x; pos[w, 1] = y; pos[w, 2] = z
                disp[w, 0] += D[k, 0]; disp[w, 1] += D[k, 1]; disp[w, 2] += D[k, 2]
        if (s + 1) % rec_every == 0:
            m = 0.0
            for w in range(nw): m += disp[w, 0] ** 2 + disp[w, 1] ** 2 + disp[w, 2] ** 2
            msd[(s + 1) // rec_every - 1] = m / nw
    return msd


def tokuyama(phi):
    b = np.sqrt(9 * phi / 8); c = 11 * phi / 16
    H = 2 * b * b / (1 - b) - c / (1 + 2 * c) - b * c * (2 + c) / ((1 + c) * (1 - b + c))
    return 1.0 / (1.0 + H)


def main():
    nbox = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    os.makedirs(OUT, exist_ok=True)
    M = pd.read_parquet(f"{W}/tiles.parquet"); meta = json.load(open(f"{W}/tiles_meta.json"))
    L = json.load(open(f"{W}/libs.json")); names, mass = L["libs"], np.array(L["mass_da"]) / 1e3
    PL = json.load(open(f"{W}/prepass_libs.json"))
    M.attrs.update(lo0=np.array(meta["lo0"]), poff=np.load(f"{W}/prepass_off.npy"), ptypes=np.load(f"{W}/prepass_types.npy", mmap_mode="r"),
                   ppos=np.load(f"{W}/prepass_pos.npy", mmap_mode="r"), pquat=np.load(f"{W}/prepass_quat.npy", mmap_mode="r"),
                   moff=np.load(f"{W}/mrna_off.npy"), mpts=np.load(f"{W}/mrna_pts.npy", mmap_mode="r"))
    mrad = json.load(open(f"{W}/mrna_meta.json"))["bead_radius"]
    key = {tuple(t): i for i, t in enumerate(M[["ti", "tj", "tk"]].values)}
    r_bead, curve = calibrate_r(names, mass); print(f"calibrated bead radius {r_bead:.3f} nm", flush=True)
    rng = np.random.default_rng(1); rows = []
    for reg in REGIONS:
        full = np.where(M[reg].values >= 0.98 * TILE ** 3)[0]
        if len(full) == 0:                                   # thin compartments: most-filled tiles
            full = np.argsort(-M[reg].values)[:50]
        for ti in rng.choice(full, min(nbox, len(full)), replace=False):
            t0 = time.time()
            box_lo = M.loc[ti, ["ti", "tj", "tk"]].values * TILE + M.attrs["lo0"] + (TILE - BOX) / 2
            P, Rm = gather(box_lo, M, key, L["libs"], PL)
            n = int(round((BOX + 2 * 10) / VOX)); glo = box_lo - 10
            g = np.ones((n, n, n), bool)
            for pts in (P, Rm):
                ij = np.floor((pts - glo) / VOX).astype(int); ok = np.all((ij >= 0) & (ij < n), axis=1); ij = ij[ok]
                g[ij[:, 0], ij[:, 1], ij[:, 2]] = False
            d_b = ndimage.distance_transform_edt(g) * VOX               # distance to nearest bead centre
            occ = d_b <= r_bead
            if len(Rm):                                                 # mRNA beads are 1 nm spheres
                gm = np.ones((n, n, n), bool); ij = np.floor((Rm - glo) / VOX).astype(int); ok = np.all((ij >= 0) & (ij < n), axis=1)
                gm[ij[ok, 0], ij[ok, 1], ij[ok, 2]] = False; occ |= ndimage.distance_transform_edt(gm) * VOX <= mrad
            c = slice(int(10 / VOX), int((10 + BOX) / VOX)); phi = float(occ[c, c, c].mean())
            d_o = ndimage.distance_transform_edt(~occ) * VOX            # probe-centre clearance
            row = dict(region=reg, tile=int(ti), box_lo=box_lo.round(1).tolist(), phi=round(phi, 4), n_beads=int(len(P)),
                       D_hyd=round(float(tokuyama(phi)), 3))
            for a in PROBES:
                acc = d_o > a; f = float(acc[c, c, c].mean()); row[f"f_acc_{a}"] = round(f, 4)
                cand = np.argwhere(acc[c, c, c]) + int(10 / VOX)
                if len(cand) < 50: row[f"Dobs_{a}"] = 0.0; continue
                st = cand[rng.choice(len(cand), 1500, replace=len(cand) < 1500)].astype(np.int64)
                nsteps, rec = 16000, 400
                msd = walk(acc, st, nsteps, rec, int(ti))
                tt = np.arange(1, len(msd) + 1) * rec; h = len(msd) // 2
                slope = np.polyfit(tt[h:], msd[h:], 1)[0]             # free lattice walk: msd = 1 x steps
                row[f"Dobs_{a}"] = round(float(slope), 3); row[f"D_{a}"] = round(float(slope * row["D_hyd"]), 3)
            row["s"] = round(time.time() - t0, 1); rows.append(row); print(row, flush=True)
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/crowding.tsv", sep="\t", index=False)
    S = T.groupby("region")[[c for c in T.columns if c.startswith(("phi", "f_acc", "Dobs", "D_"))]].mean().round(3)
    print(S.to_string())
    json.dump(dict(r_bead=r_bead, calib=curve, summary=S.reset_index().to_dict("records")), open(f"{OUT}/crowding.json", "w"), indent=1)


if __name__ == "__main__":
    main()
