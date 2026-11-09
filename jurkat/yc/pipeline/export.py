import json, gzip
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def placements_table(pk):
    rows = []
    for k, ((lid, Rm, t), tag) in enumerate(zip(pk.placed, pk.tags)):
        q = Rotation.from_matrix(Rm).as_quat()  # x, y, z, w
        rows.append((k, lid, tag, *np.round(t, 3), *np.round(q, 5)))
    return pd.DataFrame(rows, columns=["instance", "ingredient", "region", "x_nm", "y_nm", "z_nm", "qx", "qy", "qz", "qw"])


def recipe(ing, cexp, clog, lib, oligo, exp_ids, geo, layout, regions, wc_summary):
    pdb_of = {"ribosome_80S": "4V88", "proteasome_26S": "4CR2", "FAS": "2UV8", "CCT": "4V94", "PolII": "1WCM",
              "PolI": "4C2M", "PolIII": "5FJ8", "NPC": "7N9F", "Pma1_hexamer": "7VH5", "V-ATPase": "6O7V",
              "nucleosome": "1ID3", "tRNA": "1EHZ"}
    pdb_of.update({k: str(v[1]) for k, v in oligo.items()})
    placed = clog.groupby(["lib_id", "region"]).placed.sum()
    comps = {}
    for (lid, reg), n in placed.items():
        reg2 = "nucleoplasm" if reg == "chromatin" else reg
        d = lib.get(lid)
        ent = {"count_packed": int(n),
               "count_expected": float(cexp[(cexp.lib_id == lid) & (cexp.region == reg)].expected.sum()) if lid != "tRNA" else None,
               "packing_mode": "membrane (TM normal = +z, cytoplasmic side +z before flip)" if reg2 in regions and regions[reg2]["kind"] == "membrane" else "random",
               "structure_source": "PDB " + pdb_of[lid] if lid in pdb_of else "AlphaFold DB v6 (UniProt %s)" % (ing.set_index("ing_id").uniprot.get(lid, "")),
               "n_residue_beads": int(d["beads"].shape[0]) if d else None,
               "sphere_tree": {"n": int(d["st_r"].shape[0]), "max_radius_nm": float(d["st_r"].max())} if d else None,
               "representation": f"structure_library/{lid}.npz"}
        comps.setdefault(reg2, {})[lid] = ent
    reg_out = {}
    for k, v in regions.items():
        reg_out[k] = {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist()) for kk, vv in v.items()}
    return {"name": "Saccharomyces cerevisiae S288C (BY4741) haploid, exponential growth in YPD",
            "model": "detailed column (wall -> nucleoplasm) + coarse whole-cell equatorial slab",
            "units": "nm",
            "column": {"box_nm": [250, 250, layout["top"]], "periodic": ["x", "y"], "layout": layout,
                       "regions": reg_out, "ingredients_by_region": comps},
            "cell_geometry": geo, "whole_cell_slab": wc_summary}


def write_mmcif(path, items, title="yeast_column_subslab"):
    """items: list of (pos (n,3) nm, comp_id, atom_name, element, asym_ids (n,) str, seq (n,) int, entity_label)"""
    hdr = [f"data_{title}", "#", f"_struct.title '{title}: residue-level coarse-grained beads (1 bead per residue / nucleotide / lipid group / ~sugar unit)'", "#",
           "loop_", "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol", "_atom_site.label_atom_id", "_atom_site.label_alt_id",
           "_atom_site.label_comp_id", "_atom_site.label_asym_id", "_atom_site.label_entity_id", "_atom_site.label_seq_id", "_atom_site.pdbx_PDB_ins_code",
           "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z", "_atom_site.occupancy", "_atom_site.B_iso_or_equiv",
           "_atom_site.auth_seq_id", "_atom_site.auth_asym_id", "_atom_site.pdbx_PDB_model_num"]
    n0 = 1
    with gzip.open(path, "wt") as f:
        f.write("\n".join(hdr) + "\n")
        for ent, (pos, comp, atom, el, asym, seq) in enumerate(items, start=1):
            A = pos * 10.0  # nm -> Angstrom
            n = len(A)
            ids = np.arange(n0, n0 + n); n0 += n
            lines = [f"HETATM {i} {el} {atom} . {comp} {a} {ent} {s} ? {x:.2f} {y:.2f} {z:.2f} 1.00 0.00 {s} {a} 1"
                     for i, a, s, (x, y, z) in zip(ids, asym, seq, A)]
            f.write("\n".join(lines) + "\n")
        f.write("#\n")
    return n0 - 1
