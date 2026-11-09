"""Example dynamics on the packed Jurkat cell: Brownian dynamics of inert tracers in the frozen macromolecular crowd.

For one cytosol box and one nucleoplasm box (BOX nm cubes inside the viewer sub-volume, jurkat/viewer/web/data):
  1. the obstacle field as in crowding.py (residue/nucleotide beads of every placed instance, bead radius calibrated
     to the protein volume, 0.5 nm voxels) and the probe-centre clearance map d_o;
  2. off-lattice Brownian dynamics of N tracers per probe size: Gaussian steps with D = D0 x Tokuyama-Oppenheim
     hydrodynamic factor at the box's volume fraction; a step is rejected if it ends where d_o <= probe radius
     (hard obstacles) or outside the box (reflecting walls); dt = DT ns;
  3. MSD(t), the anomalous exponent alpha (log-log slope) and D(t)/D0 = MSD / (6 D0 t) at short and long lag,
     compared with GFP measurements (D/D0 ~ 0.25-0.33 in cytoplasm and nucleus; Swaminathan 1997, Seksek 1997, Dayel 1999);
  4. viewer export: tracer positions every FRAME_EVERY steps as uint16 in the box (web/data/dynamics/), coordinates
     relative to the sub-volume centre like the bookmarks.
Caveats: crowders are frozen (no obstacle motion, so cages never open: long-lag D is a lower bound for large probes),
no binding or attraction, no fluid-phase viscosity above water, tracers do not see each other.
usage: dynamics.py   Output: data/derived/validation/dynamics.json, dynamics_msd.tsv; jurkat/viewer/web/data/dynamics/"""
import os, sys, json, time, numpy as np, pandas as pd, numba as nb
from scipy import ndimage

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crowding as cw
import cell3d as c3

OUT = f"{V}/data/derived/validation"; WEB = f"{V}/jurkat/viewer/web/data"
BOX = 150.0; VOX = cw.VOX; PAD = 10.0
DT = 1.0                    # ns
NSTEP = 20000               # 20 us
FRAME_EVERY = 20            # 20 ns per viewer frame -> 1000 frames
N_TRACER = 400              # per probe for statistics; the first N_SHOW are exported
N_SHOW = 150
# probe radius nm, D0 in water nm2/us (Stokes-Einstein, 20 C; GFP 87 um2/s); 1 um2/s = 1 nm2/us
PROBES = {"GFP": (2.3, 87.0), "IgG-sized": (4.5, 45.0)}
REGIONS = ["cytosol", "nucleoplasm"]
LABEL = {"cytosol": c3.CYTO, "nucleoplasm": c3.NUCP}
GFP_REF = {"cytosol": (0.25, 0.33, "GFP in cytoplasm 3-4x slower than water (Swaminathan 1997; Dayel 1999)"),
           "nucleoplasm": (0.25, 0.35, "GFP in nucleoplasm similar to cytoplasm (Seksek 1997; Dayel 1999)")}


@nb.njit(cache=True)
def bd(free, glo, vox, lo, hi, x0, sig, nstep, every, seed):
    """free: bool grid of allowed probe-centre voxels; returns frames (nf, n, 3) and acceptance rate"""
    np.random.seed(seed)
    n = x0.shape[0]; nf = nstep // every; fr = np.zeros((nf, n, 3), np.float32)
    x = x0.copy(); acc = 0; nx, ny, nz = free.shape
    for s in range(nstep):
        for i in range(n):
            a = x[i, 0] + sig * np.random.randn(); b = x[i, 1] + sig * np.random.randn(); c = x[i, 2] + sig * np.random.randn()
            if a < lo[0] or b < lo[1] or c < lo[2] or a >= hi[0] or b >= hi[1] or c >= hi[2]:
                continue
            ia = int((a - glo[0]) / vox); ib = int((b - glo[1]) / vox); ic = int((c - glo[2]) / vox)
            if ia < 0 or ib < 0 or ic < 0 or ia >= nx or ib >= ny or ic >= nz or not free[ia, ib, ic]:
                continue
            x[i, 0] = a; x[i, 1] = b; x[i, 2] = c; acc += 1
        if (s + 1) % every == 0:
            fr[(s + 1) // every - 1] = x
    return fr, acc / (nstep * n)


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(f"{WEB}/dynamics", exist_ok=True)
    W = cw.W
    M = pd.read_parquet(f"{W}/tiles.parquet"); meta = json.load(open(f"{W}/tiles_meta.json"))
    L = json.load(open(f"{W}/libs.json")); names, mass = L["libs"], np.array(L["mass_da"]) / 1e3
    PL = json.load(open(f"{W}/prepass_libs.json"))
    M.attrs.update(lo0=np.array(meta["lo0"]), poff=np.load(f"{W}/prepass_off.npy"), ptypes=np.load(f"{W}/prepass_types.npy", mmap_mode="r"),
                   ppos=np.load(f"{W}/prepass_pos.npy", mmap_mode="r"), pquat=np.load(f"{W}/prepass_quat.npy", mmap_mode="r"),
                   moff=np.load(f"{W}/mrna_off.npy"), mpts=np.load(f"{W}/mrna_pts.npy", mmap_mode="r"))
    mrad = json.load(open(f"{W}/mrna_meta.json"))["bead_radius"]
    key = {tuple(t): i for i, t in enumerate(M[["ti", "tj", "tk"]].values)}
    em = json.load(open(f"{WEB}/encode_meta.json")); C = np.array(em["centre_nm"]); half = em["cube_nm"] / 2
    z = np.load(f"{V}/data/derived/jgeom_cell1.npz"); cell = c3.Cell(z["G"], z["pores"])
    r_bead, _ = cw.calibrate_r(names, mass); print(f"calibrated bead radius {r_bead:.3f} nm", flush=True)
    rng = np.random.default_rng(5); out = dict(dt_ns=DT, n_steps=NSTEP, frame_ns=DT * FRAME_EVERY, box_nm=BOX, r_bead=r_bead,
                                               probes={k: dict(radius_nm=v[0], D0_nm2_per_us=v[1]) for k, v in PROBES.items()}, regions=[])
    msd_rows = []
    for reg in REGIONS:
        # box anywhere in the viewer cube whose 6x6x6 sample points all carry the region label; nearest the cube centre
        s6 = np.linspace(0, BOX, 6); grid = np.stack(np.meshgrid(s6, s6, s6, indexing="ij"), -1).reshape(-1, 3)
        cs = C - half + 20 + rng.random((3000, 3)) * (2 * half - 40 - BOX)
        lab = cell.labels((cs[:, None, :] + grid[None]).reshape(-1, 3)).reshape(len(cs), -1)
        ok = np.where((lab == LABEL[reg]).all(1))[0]
        if len(ok) == 0:
            print(f"{reg}: no {BOX:.0f} nm box inside the viewer cube", flush=True); continue
        box_lo = cs[ok[np.argmin(np.linalg.norm(cs[ok] + BOX / 2 - C, axis=1))]]; t0 = time.time()
        ti = int(key[tuple(np.floor((box_lo + BOX / 2 - M.attrs["lo0"]) / cw.TILE).astype(int))])
        P, Rm = cw.gather(box_lo, M, key, L["libs"], PL)
        n = int(round((BOX + 2 * PAD) / VOX)); glo = box_lo - PAD
        g = np.ones((n, n, n), bool)
        ij = np.floor((P - glo) / VOX).astype(int); ok = np.all((ij >= 0) & (ij < n), axis=1); g[tuple(ij[ok].T)] = False
        occ = ndimage.distance_transform_edt(g) * VOX <= r_bead
        if len(Rm):
            gm = np.ones((n, n, n), bool); ij = np.floor((Rm - glo) / VOX).astype(int); ok = np.all((ij >= 0) & (ij < n), axis=1)
            gm[tuple(ij[ok].T)] = False; occ |= ndimage.distance_transform_edt(gm) * VOX <= mrad
        c = slice(int(PAD / VOX), int((PAD + BOX) / VOX)); phi = float(occ[c, c, c].mean())
        d_o = ndimage.distance_transform_edt(~occ) * VOX
        hyd = float(cw.tokuyama(phi))
        R = dict(region=reg, tile=ti, box_lo_world=box_lo.round(1).tolist(), box_lo_view=(box_lo - C).round(1).tolist(),
                 phi=round(phi, 4), D_hyd=round(hyd, 3), n_beads=int(len(P)), probes={})
        print(f"{reg}: tile {ti}, phi {phi:.3f}, hydrodynamic factor {hyd:.3f}, obstacle map {time.time() - t0:.0f}s", flush=True)
        for pk, (a, D0) in PROBES.items():
            free = d_o > a
            inner = np.argwhere(free[c, c, c]) + int(PAD / VOX)               # start anywhere accessible in the box
            st = inner[rng.choice(len(inner), N_TRACER, replace=False)] * VOX + glo + rng.random((N_TRACER, 3)) * VOX
            D = D0 * hyd / 1e3                                                  # nm2/ns
            sig = float(np.sqrt(2 * D * DT))
            t1 = time.time()
            fr, accr = bd(free, glo.astype(np.float64), VOX, box_lo.astype(np.float64), (box_lo + BOX).astype(np.float64),
                          st.astype(np.float64), sig, NSTEP, FRAME_EVERY, ti + int(a * 10))
            # MSD from the start (time origins: every frame of the first half, lags up to half the run)
            nf = len(fr); lags = np.unique(np.round(np.logspace(0, np.log10(nf // 2), 30)).astype(int))
            msd = np.array([((fr[l:nf // 2 + l] - fr[:nf // 2]) ** 2).sum(-1).mean() for l in lags])
            t_us = lags * FRAME_EVERY * DT / 1e3
            Dt = msd / (6 * t_us) / D0                                          # D(t)/D0, D0 in nm2/us
            h = len(lags) // 2
            alpha = float(np.polyfit(np.log(t_us[h:]), np.log(msd[h:]), 1)[0])
            frac_free = float(free[c, c, c].mean())
            R["probes"][pk] = dict(radius_nm=a, accessible_frac=round(frac_free, 4), step_accept=round(float(accr), 3),
                                   D_over_D0_20ns=round(float(Dt[0]), 3), D_over_D0_long=round(float(Dt[-1]), 3),
                                   long_lag_us=round(float(t_us[-1]), 2), alpha_long=round(alpha, 3), seconds=round(time.time() - t1, 1))
            for l, t, m, d in zip(lags, t_us, msd, Dt):
                msd_rows.append(dict(region=reg, probe=pk, t_us=round(float(t), 4), msd_nm2=round(float(m), 2), D_over_D0=round(float(d), 4)))
            q = np.clip(np.round((fr[:, :N_SHOW] - box_lo) / BOX * 65535), 0, 65535).astype(np.uint16)
            fn = f"dynamics/{reg}_{pk.split('-')[0].lower()}.bin"
            q.tofile(f"{WEB}/{fn}"); R["probes"][pk].update(file=fn, n_show=N_SHOW, n_frames=int(len(fr)))
            print(f"  {pk}: {R['probes'][pk]}", flush=True)
        lo_r, hi_r, ref = GFP_REF[reg]; g_ = R["probes"]["GFP"]["D_over_D0_long"]
        R["gfp_reference"] = dict(lo=lo_r, hi=hi_r, source=ref, within=bool(lo_r <= g_ <= hi_r))
        out["regions"].append(R)
    json.dump(out, open(f"{OUT}/dynamics.json", "w"), indent=1)
    pd.DataFrame(msd_rows).to_csv(f"{OUT}/dynamics_msd.tsv", sep="\t", index=False)
    view = dict(frame_ns=out["frame_ns"], box_nm=BOX, quant=65535,
                sets=[dict(region=R["region"], box_lo=R["box_lo_view"], phi=R["phi"],
                           tracers=[dict(name=k, radius_nm=v["radius_nm"], file=v["file"], n=v["n_show"], frames=v["n_frames"],
                                         D_over_D0=v["D_over_D0_long"]) for k, v in R["probes"].items()]) for R in out["regions"]])
    json.dump(view, open(f"{WEB}/dynamics.json", "w"), indent=1)
    bm = [b for b in json.load(open(f"{WEB}/bookmarks.json")) if not b["name"].startswith("Diffusion")]
    for R in out["regions"]:
        g_ = R["probes"]["GFP"]
        bm.append(dict(name=f"Diffusion: {R['region']}", dynamics=R["region"],
                       caption=f"Brownian dynamics of GFP-sized tracers in the frozen {R['region']} crowd ({BOX:.0f} nm box, "
                               f"{out['frame_ns']:.0f} ns per frame). Predicted D/D0 = {g_['D_over_D0_long']:.2f} "
                               f"(GFP measured about {R['gfp_reference']['lo']}-{R['gfp_reference']['hi']}).",
                       target=[round(float(x), 3) for x in np.array(R["box_lo_view"]) + BOX / 2], normal=[0, 0, 1], up=[0, 1, 0],
                       width=BOX * 1.3, slab=40))
    json.dump(bm, open(f"{WEB}/bookmarks.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
