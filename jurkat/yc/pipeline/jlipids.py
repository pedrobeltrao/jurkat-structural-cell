"""Lipid layer for the Jurkat cell (yeast approach: lipids are procedural, one bead per AREA_PER_LIPID per leaflet on
every membrane midsurface, masked around protein beads; not packed and not counted in densities), with
per-membrane, per-leaflet species-level compositions from data/lipids_jurkat.json (jlipids_table.py; Pogozheva 2022
+ Zech 2009 for the Jurkat PM); yeast used one whole-cell composition. Lipid counts use each leaflet's mean area per lipid.

This script builds the per-membrane summary used by the encoder (lipid classes per leaflet) and by the planned MD
volume export: membrane area, TM-protein footprint (2D convex hull of library beads within the bilayer slab x copies
assigned to that membrane), free lipid area, and lipid counts per class and leaflet.
Output: data/derived/lipids_cell1.tsv (+ json with the composition actually used)."""
import os, sys, json, numpy as np, pandas as pd
from scipy.spatial import ConvexHull
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LIB
import jpack_test as jp

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"
COMP = json.load(open(f"{os.path.dirname(os.path.abspath(__file__))}/data/lipids_jurkat.json"))
BEAD_R = 0.42


def footprint(ing_id):
    z = np.load(f"{LIB}/{ing_id}.npz")
    b = z["beads"]; s = b[np.abs(b[:, 2]) < 1.5]
    if len(s) < 3:
        return 0.0
    try:
        h = ConvexHull(s[:, :2]); return float(h.volume + h.area * BEAD_R + np.pi * BEAD_R ** 2)   # hull area + bead rim
    except Exception:
        return 0.0


def main():
    g = np.load(f"{D}/jgeom_cell1.npz"); summ = json.loads(str(g["summary"]))
    area = {"PM": summ["cell"]["area_um2"], "NE": 2 * summ["ne_mid"]["area_um2"], "ER": summ["er"]["area_um2"],
            "MITO_OM": summ["mito"]["area_um2"], "MITO_IM": jp.CRISTAE_F * summ["mito"]["area_um2"],
            "GOLGI": jp.GOLGI_STACK_F * summ["golgi"]["area_um2"], "ENDO": summ["endo"]["area_um2"]}
    I = pd.read_csv(f"{D}/ingredients_v0.tsv", sep="\t")
    I = I[I.transmembrane & I.ing_id.map(lambda i: os.path.exists(f"{LIB}/{i}.npz"))]
    I = I[I.ing_id.map(lambda i: "membrane" in np.load(f"{LIB}/{i}.npz").files)]
    fp = {i: footprint(i) for i in I.ing_id}
    occ = {m: 0.0 for m in area}
    for r in I.itertuples():
        for comp, w in json.loads(r.comp_weights).items():
            reg = jp.TMR.get(comp, "ER")
            if reg == "MITO_IM" and r.mito_om_frac >= 0.5: reg = "MITO_OM"
            occ[reg] += r.copies * w * fp[r.ing_id] / 1e6          # nm2 -> um2
    rows = []
    for m, a in area.items():
        spec = COMP["membranes"][m]; spec = COMP["membranes"][spec["same_as"]] if "same_as" in spec else spec
        free = max(a - occ[m], 0.0) * 1e6                           # nm2 of lipid-covered midsurface per leaflet
        row = dict(membrane=m, area_um2=round(a, 1), tm_protein_um2=round(occ[m], 1), protein_area_frac=round(occ[m] / a, 3),
                   dpp_nm=spec["dpp_nm"])
        for lf in ("cyt", "exo"):
            L = spec["leaflets"][lf]; n = free / L["mean_apl_nm2"]
            row[f"lipids_{lf}"] = round(n)
            for cls, f in L["classes"].items():
                row[f"{lf}_{cls}"] = round(n * f)
        rows.append(row)
    T = pd.DataFrame(rows).fillna(0)
    T.to_csv(f"{D}/lipids_cell1.tsv", sep="\t", index=False)
    print(T[["membrane", "area_um2", "tm_protein_um2", "protein_area_frac", "lipids_cyt", "lipids_exo", "dpp_nm"]].to_string(index=False))
    tot = T.lipids_cyt.sum() + T.lipids_exo.sum()
    print(f"total lipids {tot:.3g} (~{tot * 750 / 6.022e23 * 1e12:.1f} pg at ~750 Da); compositions {COMP['version']}")


if __name__ == "__main__":
    main()
