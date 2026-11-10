"""Packing test on one block of the Jurkat cell (PLANNING_NOTES §7 step 3): how much of the inventory fits?

The block (BLOCK nm cube) is packed as a single pack3d tile (no neighbours, no halo obstacles, no organisation layer,
no pre-pass: chromatin, NPCs, microtubules, polysome chains and mRNA are not placed in this test).
Expected copies in the block = cell copies x compartment weight x (block measure / cell measure) of the target region,
with randomised rounding. Region mapping (soluble | transmembrane):
  cytosol, cytoskeleton, peroxisome, lipid_droplet, unassigned, plasma_membrane(sol) -> cytosol | PM (plasma_membrane),
  ER (others); nucleus, nuclear_envelope(sol) -> nucleoplasm | NE; mitochondrion -> matrix | mito IM (outer-membrane proteins -> OM;
  IM area = CRISTAE_F x OM area, only the inner-boundary share packed); er, secreted -> ER lumen | ER; golgi -> golgi lumen | GOLGI; endolysosomal -> lumen | ENDO;
  lumens only for species with a signal peptide (signal_frac >= 0.5), else cytosol (peripheral / ribosome-bound).
usage: jpack_test.py [block_nm=1000] [seed=1] [prepass]  (prepass: fixed obstacles from data/derived/prepass/*_cell1.npz,
  their consumed copies removed from the random-packing jobs)
Output: data/derived/packtest/block_<...>.npz (placed instances), summary json; printed placement statistics."""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3, pack3d as p3
from skimage.measure import marching_cubes
from config import LIB

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"
_A = sys.argv if __name__ == "__main__" else []          # module constants also imported by jwc.py / jlipids.py
BLOCK = float(_A[1]) if len(_A) > 1 else 1000.0
SEED = int(_A[2]) if len(_A) > 2 else 1
USE_PREPASS = len(_A) > 3 and _A[3] == 'prepass'
VOX = 4.0; MESH_H = 8.0
GOLGI_STACK_F = 4.5   # Golgi cisternal membrane area / smooth 16 nm blob area (cisternae ~25 nm thick: S/V ~80 um-1)
CRISTAE_F = 3.0   # total IM area / OM area (cristae not built yet: only the inner-boundary share of IM proteins is packed)
VOL_REG = [("cytosol", (c3.CYTO,)), ("nucleoplasm", (c3.NUCP,)), ("matrix", (c3.MATRIX,)), ("ER_lumen", (c3.ERL,)),
           ("golgi_lumen", (c3.GOLGI,)), ("endo_lumen", (c3.ENDO,)), ("perinuclear", (c3.PERI,)), ("nucleolus", (c3.NUCL,))]
MEM_REG = [("PM", c3.M_PM), ("NE", c3.M_NE), ("ER", c3.M_ER), ("MITO_OM", c3.M_OM), ("MITO_IM", c3.M_IM), ("GOLGI", c3.M_GOLGI), ("ENDO", c3.M_ENDO)]
# v5 shells: (name, membrane, allowed labels, side (+1 = cytoplasmic/nucleoplasmic side of the cell3d SDF), d0, d1 nm from the
# membrane midsurface, rad: add the library's bounding radius to d1). Anchored proteins: centre within BR + 2 nm of the bilayer
# surface (d1 = 2 + MEM_HALF); cortex 0-150 nm under the PM; lamina 0-25 nm from the INM surface on the nucleoplasmic side.
_AH = lambda m: 2.0 + float(c3.MEM_HALF[m])
SHELL_REG = [("sh_PM_cyto", c3.M_PM, (c3.CYTO,), 1.0, 0.0, _AH(c3.M_PM), 1),
             ("sh_PM_exo", c3.M_PM, (c3.EXT,), -1.0, 0.0, _AH(c3.M_PM), 1),
             ("sh_PM_cortex", c3.M_PM, (c3.CYTO,), 1.0, 0.0, 150.0, 0),
             ("sh_NE_lamina", c3.M_NE, (c3.NUCP,), 1.0, 0.0, 25.0 + float(c3.MEM_HALF[c3.M_NE]), 0),
             ("sh_NE_nuc", c3.M_NE, (c3.NUCP,), 1.0, 0.0, _AH(c3.M_NE), 1),
             ("sh_ER_cyto", c3.M_ER, (c3.CYTO,), 1.0, 0.0, _AH(c3.M_ER), 1),
             ("sh_ER_lumen", c3.M_ER, (c3.ERL,), -1.0, 0.0, _AH(c3.M_ER), 1),
             ("sh_GOLGI_cyto", c3.M_GOLGI, (c3.CYTO,), 1.0, 0.0, _AH(c3.M_GOLGI), 1),
             ("sh_GOLGI_lumen", c3.M_GOLGI, (c3.GOLGI,), -1.0, 0.0, _AH(c3.M_GOLGI), 1),
             ("sh_ENDO_cyto", c3.M_ENDO, (c3.CYTO,), 1.0, 0.0, _AH(c3.M_ENDO), 1),
             ("sh_ENDO_lumen", c3.M_ENDO, (c3.ENDO,), -1.0, 0.0, _AH(c3.M_ENDO), 1),
             ("sh_OM_cyto", c3.M_OM, (c3.CYTO,), 1.0, 0.0, _AH(c3.M_OM), 1)]
NV = len(VOL_REG)
NSH0 = NV + len(MEM_REG)                                  # first shell region index
SOL = {"cytosol": "cytosol", "cytoskeleton": "cytosol", "peroxisome": "cytosol", "lipid_droplet": "cytosol", "unassigned": "cytosol",
       "plasma_membrane": "cytosol", "nucleus": "nucleoplasm", "nuclear_envelope": "nucleoplasm", "mitochondrion": "matrix",
       "er": "ER_lumen", "secreted": "ER_lumen", "golgi": "golgi_lumen", "endolysosomal": "endo_lumen",
       # v5 compartment keys (jinventory_v5.py)
       "nucleolus": "nucleolus", "pm_anchor_cyto": "sh_PM_cyto", "pm_anchor_exo": "sh_PM_exo", "cortex": "sh_PM_cortex",
       "lamina": "sh_NE_lamina", "ne_anchor_nuc": "sh_NE_nuc", "er_anchor_cyto": "sh_ER_cyto", "er_anchor_lumen": "sh_ER_lumen",
       "golgi_anchor_cyto": "sh_GOLGI_cyto", "golgi_anchor_lumen": "sh_GOLGI_lumen", "endo_anchor_cyto": "sh_ENDO_cyto",
       "endo_anchor_lumen": "sh_ENDO_lumen", "om_anchor_cyto": "sh_OM_cyto"}
TMR = {"plasma_membrane": "PM", "nucleus": "NE", "nuclear_envelope": "NE", "mitochondrion": "MITO_IM", "er": "ER", "golgi": "GOLGI",
       "endolysosomal": "ENDO", "cytosol": "ER", "cytoskeleton": "ER", "peroxisome": "ER", "lipid_droplet": "ER", "unassigned": "ER", "secreted": "ER",
       "nucleolus": "NE", "pm_anchor_cyto": "PM", "pm_anchor_exo": "PM", "cortex": "PM", "lamina": "NE", "ne_anchor_nuc": "NE",
       "er_anchor_cyto": "ER", "er_anchor_lumen": "ER", "golgi_anchor_cyto": "GOLGI", "golgi_anchor_lumen": "GOLGI",
       "endo_anchor_cyto": "ENDO", "endo_anchor_lumen": "ENDO", "om_anchor_cyto": "MITO_IM"}
RNAMES = [r[0] for r in VOL_REG] + [m[0] for m in MEM_REG] + [s_[0] for s_ in SHELL_REG]
SHELL_MEMNAME = {s_[0]: [m[0] for m in MEM_REG][s_[1]] for s_ in SHELL_REG}   # shell -> membrane region (tile measure mem_<name>)


def shell_arrays():
    """pack3d.pack_tile shell arguments: n_shell0, SH_MEM, SH_MASK, SH_SIGN, SH_D0, SH_D1, SH_RAD"""
    msk = []
    for s_ in SHELL_REG:
        m = 0
        for l in s_[2]: m |= 1 << l
        msk.append(m)
    return (NSH0, np.array([s_[1] for s_ in SHELL_REG], np.int64), np.array(msk, np.int64), np.array([s_[3] for s_ in SHELL_REG]),
            np.array([s_[4] for s_ in SHELL_REG]), np.array([s_[5] for s_ in SHELL_REG]), np.array([s_[6] for s_ in SHELL_REG], np.int64))
LUMEN_MEM = {"ER_lumen": c3.M_ER, "golgi_lumen": c3.M_GOLGI, "perinuclear": c3.M_NE}   # thin lumens: orientation follows the membrane
if os.environ.get("JWC_MATRIX_THIN", "1") != "0":
    LUMEN_MEM["matrix"] = c3.M_IM   # v4: matrix slabs between cristae are thin too (more tries, IM-aligned half the time)


def regmem():
    return np.array([LUMEN_MEM.get(r[0], -1) for r in VOL_REG], np.int64)


def qflat(st_off, STC, STR):
    """per library: quaternion (w,x,y,z) rotating its smallest principal axis (sphere-tree centres weighted by r^3) to +z"""
    Q = np.zeros((len(st_off) - 1, 4)); Q[:, 0] = 1.0
    for i in range(len(st_off) - 1):
        c = STC[st_off[i]:st_off[i + 1]]; w = STR[st_off[i]:st_off[i + 1]] ** 3
        if len(c) < 3: continue
        m = (c * w[:, None]).sum(0) / w.sum(); X = c - m
        _, U = np.linalg.eigh((X * w[:, None]).T @ X)           # ascending: U[:, 0] = smallest axis
        R = np.stack([U[:, 1], U[:, 2], U[:, 0]])               # rows: e_mid -> x, e_big -> y, e_small -> z
        if np.linalg.det(R) < 0: R[0] *= -1
        tr = np.trace(R); w_ = np.sqrt(max(1.0 + tr, 1e-12)) / 2
        if w_ > 1e-3:
            Q[i] = [w_, (R[2, 1] - R[1, 2]) / (4 * w_), (R[0, 2] - R[2, 0]) / (4 * w_), (R[1, 0] - R[0, 1]) / (4 * w_)]
        else:
            from scipy.spatial.transform import Rotation
            x, y, z, ww = Rotation.from_matrix(R).as_quat(); Q[i] = [ww, x, y, z]
    return Q


def regmask():
    out = []
    for _, labs in VOL_REG:
        m = 0
        for l in labs: m |= 1 << l
        out.append(m)
    return np.array(out + [0] * (len(MEM_REG) + len(SHELL_REG)), np.int64)


def cell_measures(cell, summ, n=3_000_000, seed=0):
    """region volumes (nm3) by Monte Carlo in the enclosing box; membrane areas (nm2) from the geometry build summary"""
    R = cell.G[0]; rng = np.random.default_rng(seed)
    P = rng.uniform(-R, R, (n, 3)); lab = cell.labels(P)
    vol = {name: np.isin(lab, labs).mean() * (2 * R) ** 3 for name, labs in VOL_REG}
    a = {"PM": summ["cell"]["area_um2"], "NE": 2 * summ["ne_mid"]["area_um2"], "ER": summ["er"]["area_um2"], "MITO_OM": summ["mito"]["area_um2"],
         "MITO_IM": CRISTAE_F * summ["mito"]["area_um2"], "GOLGI": GOLGI_STACK_F * summ["golgi"]["area_um2"], "ENDO": summ["endo"]["area_um2"]}
    return vol, {k: v * 1e6 for k, v in a.items()}


def choose_block(cell, size, seed=0):
    """block centre: random candidates inside the cell; prefer cytosol 40-75% with nucleus, mito and ER present"""
    rng = np.random.default_rng(seed); R = cell.G[0]; best = None
    for _ in range(400):
        c = rng.uniform(-0.6 * R, 0.6 * R, 3)
        P = c + rng.uniform(-size / 2, size / 2, (4000, 3)); lab = cell.labels(P)
        f = {n: np.isin(lab, labs).mean() for n, labs in VOL_REG}; ext = (lab == c3.EXT).mean()
        if ext > 0.02: continue
        score = (0.4 <= f["cytosol"] <= 0.75) + (f["nucleoplasm"] > 0.05) + 2 * (f["matrix"] > 0.03) + (f["ER_lumen"] > 0.01)
        if best is None or score > best[0]: best = (score, c, f)
    return best[1], best[2]


def main():
    z = np.load(f"{D}/jgeom_cell1.npz"); cell = c3.Cell(z["G"], z["pores"]); summ = json.loads(str(z["summary"]))
    t0 = time.time()
    cvol, carea = cell_measures(cell, summ)
    print("cell volumes um3:", {k: round(v / 1e9, 1) for k, v in cvol.items()}, "| areas um2:", {k: round(v / 1e6, 1) for k, v in carea.items()}, flush=True)
    c, f = choose_block(cell, BLOCK, SEED)
    lo = c - BLOCK / 2
    print("block centre", c.round(0).tolist(), "fractions", {k: round(v, 3) for k, v in f.items()}, flush=True)
    # voxel seeds per volume region (4 nm)
    n = int(BLOCK / VOX); g = (np.arange(n) + 0.5) * VOX
    X, Y, Zg = np.meshgrid(lo[0] + g, lo[1] + g, lo[2] + g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Zg.ravel()], 1); lab = cell.labels(P)
    bvol, vox_off, VP = {}, [0], []
    for name, labs in VOL_REG:
        m = np.isin(lab, labs); VP.append(P[m]); vox_off.append(vox_off[-1] + int(m.sum())); bvol[name] = m.sum() * VOX ** 3
    vox_off += [vox_off[-1]] * len(MEM_REG); VOXP = np.concatenate(VP)
    del X, Y, Zg, P, lab
    # membrane triangles per membrane (8 nm marching cubes of each SDF inside the block)
    m_ = int(BLOCK / MESH_H) + 1; gm = np.arange(m_) * MESH_H
    X, Y, Zg = np.meshgrid(lo[0] + gm, lo[1] + gm, lo[2] + gm, indexing="ij")
    S = cell.mems(np.stack([X.ravel(), Y.ravel(), Zg.ravel()], 1)).reshape(m_, m_, m_, c3.NMEM)
    TRI, tri_off, barea = [], [0], {}
    for name, mi in MEM_REG:
        s = S[..., mi]
        if s.min() < 0 < s.max():
            v, fcs, _, _ = marching_cubes(s, 0.0, spacing=(MESH_H,) * 3); v += lo
            t = v[fcs].reshape(-1, 9)
        else:
            t = np.zeros((0, 9))
        ar = 0.5 * np.linalg.norm(np.cross(t[:, 3:6] - t[:, 0:3], t[:, 6:9] - t[:, 0:3]), axis=1) if len(t) else np.zeros(0)
        TRI.append(t); tri_off.append(tri_off[-1] + len(t)); barea[name] = float(ar.sum())
    TRI = np.concatenate(TRI) if TRI else np.zeros((0, 9))
    ar = 0.5 * np.linalg.norm(np.cross(TRI[:, 3:6] - TRI[:, 0:3], TRI[:, 6:9] - TRI[:, 0:3]), axis=1)
    CUMA = np.cumsum(ar) if len(ar) else np.zeros(1)
    print("block volumes um3:", {k: round(v / 1e9, 4) for k, v in bvol.items()}, "| areas um2:", {k: round(v / 1e6, 3) for k, v in barea.items()},
          f"| setup {time.time() - t0:.0f}s", flush=True)
    # expected counts
    I = pd.read_csv(f"{D}/ingredients_v0.tsv", sep="\t")
    PRE = sorted(f for f in os.listdir(f"{D}/prepass") if f.endswith("_cell1.npz")) if USE_PREPASS and os.path.isdir(f"{D}/prepass") else []
    consumed = {}
    for f in PRE:
        js = f"{D}/prepass/{f[:-4]}.json"
        if os.path.exists(js):
            for k, v in json.load(open(js)).get("consumed", {}).items(): consumed[k] = consumed.get(k, 0) + v
    if consumed:
        I["copies"] = (I.copies - I.ing_id.map(consumed).fillna(0)).clip(lower=0)
        print("prepass modules:", PRE, "| consumed:", {k: f"{v:.3g}" for k, v in consumed.items() if v > 1e5}, flush=True)
    have = {f[:-4] for f in os.listdir(LIB)}
    missing = I[~I.ing_id.isin(have)]
    I = I[I.ing_id.isin(have)].reset_index(drop=True)
    lp_ids = list(I.ing_id)
    memb = np.array([("membrane" in np.load(f"{LIB}/{i}.npz").files) for i in lp_ids])
    rng = np.random.default_rng(SEED); jobs = {}
    rows_exp = []
    for li, r in I.iterrows():
        w = json.loads(r.comp_weights)
        for comp, wt in w.items():
            if memb[li] and r.transmembrane:
                reg = TMR.get(comp, "ER")
                if reg == "MITO_IM" and r.mito_om_frac >= 0.5: reg = "MITO_OM"
                frac = barea[reg] / carea[reg] if carea[reg] > 0 else 0.0
            else:
                reg = SOL.get(comp, "cytosol")
                if reg in ("ER_lumen", "golgi_lumen", "endo_lumen") and r.signal_frac < 0.5: reg = "cytosol"   # no signal peptide: cytosolic face
                if reg in SHELL_MEMNAME:
                    mr = SHELL_MEMNAME[reg]; frac = barea[mr] / carea[mr] if carea[mr] > 0 else 0.0
                else:
                    frac = bvol[reg] / cvol[reg] if cvol[reg] > 0 else 0.0
            x = r.copies * wt * frac
            k = int(x) + (rng.random() < x - int(x))
            if k > 0:
                ri = RNAMES.index(reg); jobs[(li, ri)] = jobs.get((li, ri), 0) + k
                rows_exp.append((r.ing_id, reg, k, k * r.mass_da))
    E = pd.DataFrame(rows_exp, columns=["ing", "region", "n", "mass_da"])
    print(f"expected in block: {E.n.sum():,} instances, {E.mass_da.sum() / 6.022e23 * 1e12 * 1e3:.2f} fg; by region:",
          E.groupby("region").n.sum().to_dict(), f"| missing library: {len(missing)} species ({missing.mass_pg.sum():.2f} pg cell-wide)", flush=True)
    # library pack (only the species with jobs)
    used = sorted({li for li, _ in jobs}); remap = {li: k for k, li in enumerate(used)}
    st_off = [0]; C, Rr, BR = [], [], []
    for li in used:
        zz = np.load(f"{LIB}/{lp_ids[li]}.npz"); cc = zz["st_c"].astype(np.float64); rr = zz["st_r"].astype(np.float64)
        C.append(cc); Rr.append(rr); st_off.append(st_off[-1] + len(rr)); BR.append(float((np.linalg.norm(cc, axis=1) + rr).max()))
    st_off = np.array(st_off, np.int64); STC = np.concatenate(C); STR = np.concatenate(Rr); BR = np.array(BR)
    J = pd.DataFrame([(remap[li], ri, n_) for (li, ri), n_ in jobs.items()], columns=["lib", "region", "n"])
    J["br"] = BR[J.lib]; J["mem"] = (J.region >= NV) & (J.region < NSH0)
    J = J.sort_values(["mem", "br"], ascending=[False, False])
    jl, jr, jn = (J[c].values.astype(np.int64) for c in ("lib", "region", "n"))
    OC, OR = np.zeros((0, 3)), np.zeros(0); pre_mass = {}
    for f in PRE:
        z2 = np.load(f"{D}/prepass/{f}"); ids = list(z2["lib_ids"]); pos = z2["pos"].astype(np.float64)
        sel = np.all((pos > lo - 60) & (pos < lo + BLOCK + 60), axis=1)
        if not sel.any(): continue
        so = [0]; CC, RR, BB, MD = [], [], [], []
        for i in ids:
            zz = np.load(f"{LIB}/{i}.npz"); cc = zz["st_c"].astype(np.float64); rr = zz["st_r"].astype(np.float64)
            CC.append(cc); RR.append(rr); so.append(so[-1] + len(rr)); BB.append(float((np.linalg.norm(cc, axis=1) + rr).max()))
            MD.append(float(I.set_index("ing_id").mass_da.get(i, 0)) if i in set(I.ing_id) else 0.0)
        t2 = z2["types"][sel].astype(np.int64)
        C0, R0 = p3.emit_spheres(t2, pos[sel], z2["quat"][sel].astype(np.float64), np.array(so, np.int64), np.concatenate(CC), np.concatenate(RR), np.array(BB), lo, lo + BLOCK)
        OC = np.vstack([OC, C0]); OR = np.r_[OR, R0]
        inb = np.all((pos[sel] >= lo) & (pos[sel] < lo + BLOCK), axis=1)
        if inb.any():
            lab_c = cell.labels(pos[sel][inb]); md = np.array(MD)[t2[inb]]
            for name, labs in VOL_REG:
                pre_mass[name] = pre_mass.get(name, 0.0) + float(md[np.isin(lab_c, labs)].sum())
        print(f"  obstacles from {f}: {int(sel.sum()):,} instances, {len(R0):,} spheres", flush=True)
    nsph = int((np.diff(st_off)[jl] * jn).sum()); cap_s = nsph + len(OR) + 1000; cap_e = cap_s * 14
    print(f"jobs {len(J)}, spheres to place {nsph:,}; packing ...", flush=True)
    e = np.zeros(0, np.int64)
    t1 = time.time()
    res = p3.pack_tile(SEED, lo, BLOCK, 0.0, cell.G, cell.pores, cell.pidx, cell.PT, cell.PQ, cell.PC, st_off, STC, STR, BR, regmask(),
                       jl, jr, jn, NV, np.array(vox_off, np.int64), VOXP, VOX, np.array(tri_off, np.int64), TRI, CUMA,
                       OC, OR, 0, 1500.0, 0, 200, 25, cap_s, cap_e,
                       e, np.zeros((0, 3)), e, e, np.zeros(0), e, np.zeros(7), qflat(st_off, STC, STR), regmem(), *shell_arrays())
    types, pos, quat, placed, _, _, overflow, ns, ne, _ = res
    dt = time.time() - t1
    J["placed"] = placed; J["ing"] = [lp_ids[used[l]] for l in J.lib]; J["mass_da"] = I.mass_da.values[[used[l] for l in J.lib]]
    J["reg"] = [RNAMES[r] for r in J.region]
    out = []
    for reg, g_ in J.groupby("reg"):
        me, mp_ = (g_.n * g_.mass_da).sum(), (g_.placed * g_.mass_da).sum()
        meas = bvol.get(reg) or barea.get(reg)
        dens = mp_ / 6.022e23 * 1e12 / (meas / 1e9) * 1e3 if reg in bvol and meas > 0 else np.nan   # mg/mL = pg/um3*1e3/1e3
        pm_ = pre_mass.get(reg, 0.0)
        tot_d = (mp_ + pm_) / 6.022e23 * 1e12 / (meas / 1e9) * 1e3 if reg in bvol and meas > 0 else np.nan
        out.append(dict(region=reg, expected=int(g_.n.sum()), placed=int(g_.placed.sum()), frac_placed=round(g_.placed.sum() / max(g_.n.sum(), 1), 3),
                        prepass_mg_per_ml=round(pm_ / 6.022e23 * 1e12 / (meas / 1e9) * 1e3, 1) if reg in bvol and meas > 0 else None,
                        total_mg_per_ml=round(tot_d, 1) if tot_d == tot_d else None,
                        mass_frac_placed=round(mp_ / max(me, 1), 3), placed_mg_per_ml=round(dens, 1) if dens == dens else None,
                        expected_mg_per_ml=round(me / 6.022e23 * 1e12 / (meas / 1e9) * 1e3, 1) if reg in bvol and meas > 0 else None))
    O = pd.DataFrame(out); print(f"packed in {dt:.0f}s; overflow={overflow}; spheres {ns:,}"); print(O.to_string(index=False))
    # which species fail most (by mass not placed)
    J["miss_mass"] = (J.n - J.placed) * J.mass_da
    print("largest unplaced (by mass):"); print(J.sort_values("miss_mass", ascending=False).head(10)[["ing", "reg", "n", "placed", "br"]].to_string(index=False))
    os.makedirs(f"{D}/packtest", exist_ok=True)
    tag = f"block{int(BLOCK)}_seed{SEED}" + ("_prepass" if USE_PREPASS else "")
    np.savez_compressed(f"{D}/packtest/{tag}.npz", types=types, pos=pos, quat=quat, lib_ids=np.array([lp_ids[u] for u in used]), lo=lo, size=BLOCK)
    O.to_csv(f"{D}/packtest/{tag}_summary.tsv", sep="\t", index=False); J.to_csv(f"{D}/packtest/{tag}_jobs.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
