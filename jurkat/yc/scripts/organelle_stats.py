"""Organelle shape statistics from segmented cryo-FIB/SEM yeast cells -> pipeline/organelle_stats.json (small, kept in the repository).

Input: the yeast_organelle_meshes bundle (EMPIAR-11416, Dumoux et al. 2023: cells 04-08, watertight OBJ meshes per
part plus manifest.json), path in YC_ORGANELLE_MESHES. Only parts with provenance "real" are used: cell outlines,
vacuoles, lipid droplets (depositor labels) and the two segmented nuclei (cells 04, 05). The bundle's synthetic
mitochondria, cortical ER, peroxisomes and nuclei are ignored.

Per part the volume, centroid and principal semi-axes come from the volume moments of the closed mesh (signed
tetrahedra from the origin), so they are not biased by vertex density. cell3d.build() matches these characteristics;
it does not copy any one cell.

Usage: YC_ORGANELLE_MESHES=.../yeast_organelle_meshes python organelle_stats.py
"""
from _common import *

SRC = os.environ.get("YC_ORGANELLE_MESHES", "")
OUT = f"{C.ROOT}/organelle_stats.json"


def load_obj(fn):
    V, F = [], []
    for line in open(fn):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.array(V), np.array(F)


def moments(V, F):
    """volume, centroid and semi-axes (descending) of the solid ellipsoid with the same second moments"""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    dv = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    vol = dv.sum()
    cen = (dv[:, None] * (a + b + c) / 4.0).sum(0) / vol
    a, b, c = a - cen, b - cen, c - cen
    # second moment of each tetrahedron (origin at the centroid): vol/20 * (sum v v^T + (sum v)(sum v)^T)
    s = a + b + c
    M = (dv[:, None, None] * (np.einsum("ni,nj->nij", a, a) + np.einsum("ni,nj->nij", b, b)
                              + np.einsum("ni,nj->nij", c, c) + np.einsum("ni,nj->nij", s, s))).sum(0) / 20.0
    ev = np.sort(np.linalg.eigvalsh(M / vol))[::-1]
    return float(vol), cen, np.sqrt(5.0 * np.clip(ev, 0, None))


def main():
    if not SRC:
        raise SystemExit("set YC_ORGANELLE_MESHES to the yeast_organelle_meshes folder")
    man = json.load(open(f"{SRC}/outputs/manifest.json"))
    cells = {c["cell_id"]: c for c in man["cells"]}
    parts = {}
    for p in man["parts"]:
        if p["provenance"] != "real":
            continue
        V, F = load_obj(f"{SRC}/outputs/{p['obj']['high']}")
        V = V + np.array(p["obj_offset_um"]["high"])            # back into the cell frame
        vol, cen, ax = moments(V, F)
        parts.setdefault(p["cell_id"], []).append(dict(kind=p["organelle"], vol=vol, cen=cen, ax=ax))
    out = dict(source="EMPIAR-11416 (Dumoux et al. 2023), segmented by the yeast_organelle_meshes pipeline; real parts only",
               cells={})
    for cid, ps in sorted(parts.items()):
        cell = next(q for q in ps if q["kind"] == "cell")
        vac = [q for q in ps if q["kind"] == "vacuole"]; ld = [q for q in ps if q["kind"] == "lipid_droplet"]
        nuc = [q for q in ps if q["kind"] == "nucleus"]
        rc = (3 * cell["vol"] / (4 * np.pi)) ** (1 / 3)
        vc = np.array([q["cen"] for q in vac]); vv = np.array([q["vol"] for q in vac])
        vcen = (vc * vv[:, None]).sum(0) / vv.sum()
        # gaps between neighbouring lobes, from equivalent-sphere radii (a lower-bound-free, rough measure)
        r_eq = (3 * vv / (4 * np.pi)) ** (1 / 3)
        D = np.linalg.norm(vc[:, None] - vc[None], axis=2) - r_eq[:, None] - r_eq[None]
        np.fill_diagonal(D, np.inf)
        e = dict(truncated=cells[cid]["truncated"], cell_volume_um3=cell["vol"], cell_semi_axes_um=cell["ax"].tolist(),
                 vacuole_volumes_um3=sorted(vv.tolist(), reverse=True), vacuole_fraction=float(vv.sum() / cell["vol"]),
                 vacuole_cluster_rms_um=float(np.sqrt((vv * ((vc - vcen) ** 2).sum(1)).sum() / vv.sum())),
                 vacuole_cluster_offset_rel=float(np.linalg.norm(vcen - cell["cen"]) / rc),
                 vacuole_nn_gap_um=float(np.median(D.min(1))) if len(vac) > 1 else None,
                 vacuole_semi_axes_um=[q["ax"].tolist() for q in sorted(vac, key=lambda q: -q["vol"])[:5]],
                 lipid_droplet_volumes_um3=sorted([q["vol"] for q in ld], reverse=True),
                 lipid_droplet_fraction=float(sum(q["vol"] for q in ld) / cell["vol"]))
        if nuc:
            n = nuc[0]
            e.update(nucleus_volume_um3=n["vol"], nucleus_fraction=float(n["vol"] / cell["vol"]),
                     nucleus_semi_axes_um=n["ax"].tolist(),
                     nucleus_offset_rel=float(np.linalg.norm(n["cen"] - cell["cen"]) / rc),
                     vacuole_to_nucleus_rel=float(np.linalg.norm(vcen - n["cen"]) / rc))
        out["cells"][cid] = e
        log(f"{cid}: cell {cell['vol']:.1f} um3 axes {np.round(cell['ax'], 2)}; {len(vac)} vacuoles {e['vacuole_fraction']:.3f} "
            f"(cluster rms {e['vacuole_cluster_rms_um']:.2f} um, nn gap {e['vacuole_nn_gap_um']}); {len(ld)} LDs "
            f"{e['lipid_droplet_fraction']:.4f}" + (f"; nucleus {e['nucleus_fraction']:.3f} axes {np.round(n['ax'], 2)}" if nuc else ""))
    json.dump(out, open(OUT, "w"), indent=1, cls=NpEnc)
    log("->", OUT)


if __name__ == "__main__":
    main()
