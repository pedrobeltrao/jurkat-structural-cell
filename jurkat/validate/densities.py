"""Macromolecular mass density per compartment of the packed cell (validation backlog item 4).

mass = random-packed protein placed per region (tiles: job_lib x job_placed x library mass) + pre-pass polymers
(nucleosomes + H1 -> nucleoplasm; polysome ribosomes and cytoskeleton -> cytosol, ER-bound ribosomes too) + mRNA
(nt x 330 Da; by bead position label); volume = compartment volumes from wc/tiles.parquet (8 nm label sampling).
Reference (hedged): optical diffraction tomography finds nuclei less dense than cytoplasm (Schuermann et al. 2016
Biophys J; refractive-index increments give roughly 0.8x), and mammalian cytoplasm dry-mass densities of roughly
0.1-0.3 g/ml from quantitative phase imaging. Water/solute mass in membranes (lipids) is not included here.
Output: data/derived/validation/densities.json"""
import os, json, glob, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

V = (__import__("os").environ.get("VCELL_ROOT", ".") + ""); D = f"{V}/data/derived"; W = f"{D}/wc"
OUT = f"{D}/validation"; AV = 6.022e23
import sys
sys.path.insert(0, f"{V}/jurkat/yc/pipeline")
import jpack_test as jp                                    # v5: region list incl. nucleolus and shells
RN = list(jp.RNAMES); NV = jp.NV; NSH0 = jp.NSH0
# shells contribute their mass to the volume region they occupy (exoplasmic PM shell: extracellular, not counted)
SHELL_VOL = {"sh_PM_cyto": "cytosol", "sh_PM_cortex": "cytosol", "sh_NE_lamina": "nucleoplasm", "sh_NE_nuc": "nucleoplasm",
             "sh_ER_cyto": "cytosol", "sh_ER_lumen": "ER_lumen", "sh_GOLGI_cyto": "cytosol", "sh_GOLGI_lumen": "golgi_lumen",
             "sh_ENDO_cyto": "cytosol", "sh_ENDO_lumen": "endo_lumen", "sh_OM_cyto": "cytosol"}
SHELL_VOL.update({"sh_PM_periph": "cytosol", "sh_ENDO_periph": "cytosol", "sh_GOLGI_periph": "cytosol"})   # v6
SHELL_SPLIT = {"sh_NPC": {"cytosol": 0.5, "nucleoplasm": 0.5}}   # v6 NPC-proximal shell: both NE faces (split evenly)


def tile_mass(files):
    mass = np.array(json.load(open(f"{W}/libs.json"))["mass_da"]); out = np.zeros(len(RN))
    for f in files:
        z = np.load(f)
        np.add.at(out, z["job_region"].astype(int), z["job_placed"] * mass[z["job_lib"]])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    fs = sorted(glob.glob(f"{W}/tiles/*.npz"))
    with ProcessPoolExecutor(8) as ex:
        packed = sum(ex.map(tile_mass, [fs[i::64] for i in range(64)]))
    M = pd.read_parquet(f"{W}/tiles.parquet")
    vol = {r: float(M[r].sum()) / 1e9 for r in RN[:NV]}                   # um3
    for k in range(NSH0, len(RN)):
        if RN[k] in SHELL_VOL: packed[RN.index(SHELL_VOL[RN[k]])] += packed[k]
        for r_, f_ in SHELL_SPLIT.get(RN[k], {}).items(): packed[RN.index(r_)] += f_ * packed[k]
    # pre-pass mass by library
    PL = json.load(open(f"{W}/prepass_libs.json")); pt = np.load(f"{W}/prepass_types.npy")
    cnt = np.bincount(pt, minlength=len(PL))
    import sys; sys.path.insert(0, f"{V}/jurkat/yc/pipeline"); from config import LIB
    pmass = np.array([float(np.where(np.load(f"{LIB}/{l}.npz")["kind"] == 0, 110.0, 330.0).sum()) for l in PL])
    pre = {"nucleoplasm": 0.0, "cytosol": 0.0}
    for l, c, m in zip(PL, cnt, pmass):
        pre["nucleoplasm" if (l == "NUCLEOSOME" or l.startswith("H1-")) else "cytosol"] += c * m   # (v6: none inside speckles)
    mr = np.load(f"{D}/prepass/mrna_cell1.npz", mmap_mode="r"); nt = float(np.asarray(mr["nt"]).sum())
    rows = []
    for i, r in enumerate(RN[:NV]):
        if vol[r] <= 0: continue
        prot = packed[i]; extra = pre.get(r, 0.0) + (nt * 330.0 if r == "cytosol" else 0.0)
        rows.append(dict(region=r, volume_um3=round(vol[r], 1), packed_pg=round(prot / AV * 1e12, 2), prepass_pg=round(extra / AV * 1e12, 2),
                         mg_per_ml=round((prot + extra) / AV / (vol[r] * 1e-12) * 1e3, 1), packed_only_mg_per_ml=round(prot / AV / (vol[r] * 1e-12) * 1e3, 1)))
    T = pd.DataFrame(rows); print(T.to_string(index=False))
    Ti = T.set_index("region"); c = Ti.loc["cytosol", "mg_per_ml"]
    nm = ["nucleoplasm"] + [r_ for r_ in ("nucleolus", "speckle") if r_ in Ti.index]   # whole nucleus (+ nucleolus, speckles)
    n = float((Ti.loc[nm, "packed_pg"] + Ti.loc[nm, "prepass_pg"]).sum() / Ti.loc[nm, "volume_um3"].sum() * 1e3)
    print(f"nucleus / cytosol density ratio {n / c:.2f}; membrane-protein mass (pg): " +
          ", ".join(f"{RN[i]} {packed[i] / AV * 1e12:.2f}" for i in range(NV, NSH0)))
    json.dump(dict(rows=rows, nuc_cyto_ratio=round(n / c, 3), mem_pg={RN[i]: round(packed[i] / AV * 1e12, 3) for i in range(NV, NSH0)},
                   shell_pg={RN[i]: round(packed[i] / AV * 1e12, 3) for i in range(NSH0, len(RN))},
                   mrna_pg=round(nt * 330 / AV * 1e12, 2)), open(f"{OUT}/densities.json", "w"), indent=1)


if __name__ == "__main__":
    main()
