# Compartment volumes inside a refined cell (argv[1], default 1) (16 nm, s2; _seg layers, label frame), with GT-based volume corrections
# (§3: ER x0.75; endo/lyso/vesicle merged x2). Then protein concentration per compartment from inventory v0 (complexed).
import sys, numpy as np, pandas as pd
sys.path.insert(0, (__import__("os").environ.get("VCELL_ROOT", ".") + ""))
from jurkat.em.io import read
CELL = int(sys.argv[1]) if len(sys.argv) > 1 else 1
z = np.load(f"data/derived/cell{CELL}_s2.npz"); cell = z["cell"]; o = z["origin_s2_xyz"]
sl = tuple(slice(int(a), int(a) + n) for a, n in zip(o, cell.shape)); V3 = 16 * 16 * 13.76e-9
vol = {"cell": cell.sum() * V3}
lay = {"nucleus": "nucleus_seg", "mitochondrion": "mito_seg", "er": "er_seg", "golgi": "golgi_seg", "endo": "endo_seg",
       "lyso": "lyso_seg", "vesicle": "vesicle_seg", "lipid_droplet": "ld_seg", "nucleolus": "nucleolus_seg"}
occ = np.zeros(cell.shape, bool)
for k, l in lay.items():
    m = (read(l, "s2", sl) > 0) & cell
    vol[k] = m.sum() * V3
    if k not in ("nucleolus",): occ |= m
vol["cytosol_raw"] = (cell & ~occ).sum() * V3
corr = {"er": 0.75, "endo": 2.0, "lyso": 2.0, "vesicle": 2.0}
v = pd.Series(vol).round(2); print("raw volumes um3:"); print(v.to_string())
cv = {k: vol[k] * corr.get(k, 1.0) for k in lay}
cv["endolysosomal"] = cv.pop("endo") + cv.pop("lyso") + cv.pop("vesicle")
cv.pop("nucleolus")
cv["cytosol"] = vol["cell"] - sum(cv.values())
print("\ncorrected volumes um3 (ER x0.75, endo/lyso/vesicle x2, merged):", {k: round(x, 1) for k, x in cv.items()})
inv = pd.read_csv("data/derived/inventory_v0_complexed.tsv", sep="\t", index_col="gene")
def mass(c): return (inv.mass_pg * inv.compartments.fillna("").str.extract(rf"(?:^|;){c}:([\d.]+)")[0].astype(float).fillna(0)).sum()
rows = []
for c, vv in cv.items():
    mp = mass(c) + (mass("cytoskeleton") + mass("plasma_membrane") * 0 if c == "cytosol" else 0) + (mass("nuclear_envelope") * 0)
    rows.append((c, round(vv, 1), round(mp, 2), round(mp / vv * 1000, 0) if vv > 0 else np.nan))
print("\ncompartment, volume um3, protein pg, mg/mL (cytosol includes cytoskeleton pool; membrane pools not in lumen volumes):")
for r in rows: print("  ", r)
pd.DataFrame(rows, columns=["compartment", "volume_um3", "protein_pg", "mg_per_ml"]).to_csv(f"data/derived/cell{CELL}_compartments.tsv", sep="\t", index=False)
