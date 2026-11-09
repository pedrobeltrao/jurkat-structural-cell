"""NPC pre-pass: one nuclear pore complex in each nuclear-envelope pore of the cell geometry.

Packs v1-v3 drew the NPC ingredient (997 copies, PDB 7R5K) as an ordinary random membrane job (NE, also ER/ENDO through
its compartment weights); pack3d keeps NE proteins clear of the pores and the 120 nm ring does not fit on a flat patch,
so only 1 of 997 was placed (v2). Here the NPCs sit at the pore centres of jgeom_cell1.npz (`pores`: centre on the NE
midplane + unit normal pointing to the cytoplasm), ring axis along the normal (library convention: +z = cytoplasm,
midplane z = 0), random spin about the axis. The copies are taken from the inventory (ingredients NPC) and consumed
here, so the random pack no longer draws them; if there are fewer copies than pores a random subset of pores is filled.
Output: data/derived/prepass/npc_cell1.npz (lib_ids, types, pos, quat w,x,y,z) and npc_cell1.json."""
import os, sys, json, numpy as np, pandas as pd

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
D = f"{V}/data/derived"


def quat_z_to(n, spin):
    """(w, x, y, z): rotation taking +z to unit n, after a spin about z (as pack3d.quat_z_to)."""
    w = 1.0 + n[2]
    if w < 1e-8:
        q1 = np.array([0.0, 1.0, 0.0, 0.0])
    else:
        q1 = np.array([w, -n[1], n[0], 0.0]); q1 /= np.linalg.norm(q1)
    q2 = np.array([np.cos(spin / 2), 0.0, 0.0, np.sin(spin / 2)])
    a1, b1, c1, d1 = q1; a2, b2, c2, d2 = q2
    q = np.array([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                  a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])
    return q if q[0] >= 0 else -q


def main(seed=11):
    rng = np.random.default_rng(seed)
    pores = np.load(f"{D}/jgeom_cell1.npz")["pores"]
    I = pd.read_csv(f"{D}/ingredients_v0.tsv", sep="\t").set_index("ing_id")
    n = min(int(I.loc["NPC", "copies"]), len(pores))
    sel = np.sort(rng.choice(len(pores), n, replace=False))
    pos = pores[sel, :3].astype(np.float32)
    nrm = pores[sel, 3:6] / np.linalg.norm(pores[sel, 3:6], axis=1, keepdims=True)
    quat = np.array([quat_z_to(v, rng.random() * 2 * np.pi) for v in nrm], np.float32)
    os.makedirs(f"{D}/prepass", exist_ok=True)
    np.savez_compressed(f"{D}/prepass/npc_cell1.npz", lib_ids=np.array(["NPC"]), types=np.zeros(n, np.int32),
                        pos=pos, quat=quat, pore=sel.astype(np.int32))
    info = dict(n_pores=int(len(pores)), n_npc=n, consumed={"NPC": n}, seed=seed,
                caveats="ring axis = pore normal; cytoplasmic side = library +z (verified: RanBP2/NUP214/NUP88 at z +27..+31 nm, ELYS at -20 nm). The library is the full 7R5K scaffold (624k residue beads), but the inventory ingredient counts one copy of each Nup per NPC (3.9 MDa), so the mass bookkeeping undercounts the NPC and leaves surplus Nups as free monomers")
    json.dump(info, open(f"{D}/prepass/npc_cell1.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
