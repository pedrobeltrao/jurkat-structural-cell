"""Extract real segmented meshes for the composite model cell into pipeline/cell_meshes.npz (kept in the repository).

Input: the yeast_organelle_meshes bundle (YC_ORGANELLE_MESHES), segmented from EMPIAR-11416 (Dumoux et al. 2023).
Cell 08 is the only cell that is not cut by the edge of the imaged volume. Extracted, in the cell frame (um,
origin = cell centroid), high LOD:
  - the cell outline (outer surface of the wall), 16 vacuoles and 14 lipid droplets (real segmentations)
  - the bundle's synthetic nucleus of cell 08, only as a placement hint: centroid and principal axes (cell 08's nucleus
    was not segmented; the synthetic one sits in the cell's most open region, where the unsegmented nucleus probably is)
  - the real nucleus of cell 05 (its own frame, centred on its centroid): the composite cell gets this nucleus
    (cell 04's is the other real one, but its short axis lies along the milling direction of a cut cell)
pipeline/cellshape.py turns these into smoothed signed-distance fields.

Usage: YC_ORGANELLE_MESHES=.../yeast_organelle_meshes python cell_meshes.py
"""
from _common import *
from organelle_stats import load_obj, moments

SRC = os.environ.get("YC_ORGANELLE_MESHES", "")
CELL = "cell_08"
NUC_CELL = "cell_05"
OUT = f"{C.ROOT}/cell_meshes.npz"


def main():
    if not SRC:
        raise SystemExit("set YC_ORGANELLE_MESHES to the yeast_organelle_meshes folder")
    man = json.load(open(f"{SRC}/outputs/manifest.json"))
    out = {}
    nv = nl = 0
    for p in man["parts"]:
        if p["cell_id"] == NUC_CELL and p["organelle"] == "nucleus" and p["provenance"] == "real":
            V, F = load_obj(f"{SRC}/outputs/{p['obj']['high']}")
            V = V + np.array(p["obj_offset_um"]["high"])
            vol, cen, ax = moments(V, F)
            out["nuc_V"], out["nuc_F"] = (V - cen).astype(np.float32), F.astype(np.int32)
            e = V.max(0) - V.min(0)
            log(f"{NUC_CELL} nucleus: {vol:.2f} um3, semi-axes {np.round(ax, 2)}, extent x/y/z {np.round(e, 2)}")
        if p["cell_id"] != CELL:
            continue
        V, F = load_obj(f"{SRC}/outputs/{p['obj']['high']}")
        V = V + np.array(p["obj_offset_um"]["high"])
        if p["organelle"] == "cell":
            out["cell_V"], out["cell_F"] = V.astype(np.float32), F.astype(np.int32)
        elif p["organelle"] == "vacuole" and p["provenance"] == "real":
            out[f"vac{nv}_V"], out[f"vac{nv}_F"] = V.astype(np.float32), F.astype(np.int32); nv += 1
        elif p["organelle"] == "lipid_droplet" and p["provenance"] == "real":
            out[f"ld{nl}_V"], out[f"ld{nl}_F"] = V.astype(np.float32), F.astype(np.int32); nl += 1
        elif p["organelle"] == "nucleus":
            vol, cen, ax = moments(V, F)
            a, b, c = V[F[:, 0]] - cen, V[F[:, 1]] - cen, V[F[:, 2]] - cen
            dv = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0; s = a + b + c
            M = (dv[:, None, None] * (np.einsum("ni,nj->nij", a, a) + np.einsum("ni,nj->nij", b, b)
                                      + np.einsum("ni,nj->nij", c, c) + np.einsum("ni,nj->nij", s, s))).sum(0) / 20.0
            w, U = np.linalg.eigh(M)
            out["nuc_hint_centre"] = cen; out["nuc_hint_axes"] = U[:, ::-1].T      # rows: long, middle, short
            out["nuc_hint_volume"] = np.array(vol)
    out["n_vac"] = np.array(nv); out["n_ld"] = np.array(nl)
    out["source"] = np.array(f"EMPIAR-11416 {CELL}, yeast_organelle_meshes bundle (high LOD), um, cell frame")
    np.savez_compressed(OUT, **out)
    log(f"{CELL}: cell {len(out['cell_F'])} faces, {nv} vacuoles, {nl} lipid droplets, nucleus hint at "
        f"{np.round(out['nuc_hint_centre'], 2)} um -> {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
