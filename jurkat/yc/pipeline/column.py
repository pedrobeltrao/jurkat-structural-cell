"""Detailed column model: cell wall -> PM -> cortical ER -> cytoplasm -> nuclear envelope (+NPC) -> nucleoplasm.
z axis points from the cell exterior (z=0) inward. Units: nm."""
import json
import numpy as np
import pandas as pd
from packer import Packer

L = 250.0
LAY = dict(wall_out=0.0, mannan_end=35.0, wall_in=110.0, pm=113.0,
           erA=142.0, erB=162.0, er_x=(0.0, 125.0),
           onm=592.0, inm=626.0, top=760.0)
HALF = 2.0           # bilayer half-thickness (hydrophobic core + heads) used for exclusion
NPC_HOLE = 45.0      # pore-membrane radius in the NE (nm)
LIPID_AREA = 0.65    # nm^2 per lipid
# Ejsing et al. 2009 class composition (mol%), BY4741; remainder lumped as sphingolipid/other
LIPIDS = {"PI": 20.3, "PE": 14.9, "PC": 14.3, "PA": 8.3, "PS": 1.8, "ergosterol": 12.0}
LIPIDS["sphingolipid/other"] = 100 - sum(LIPIDS.values())


def region_volumes():
    cyto = L * L * ((LAY["onm"] - HALF) - (LAY["pm"] + HALF)) - (LAY["er_x"][1] - LAY["er_x"][0]) * L * (LAY["erB"] - LAY["erA"] + 2 * HALF)
    nuc = L * L * (LAY["top"] - (LAY["inm"] + HALF))
    wall = L * L * LAY["wall_in"]
    er_area = (LAY["er_x"][1] - LAY["er_x"][0]) * L
    return dict(cytosol=cyto, nucleoplasm=nuc, cell_wall=wall, pm_area=L * L, ne_area=L * L, er_area=er_area)


def expected_counts(ing, geo):
    """Expected copies of each ingredient in each column region (data-driven by compartment concentration)."""
    rv = region_volumes()
    V = geo["volumes_um3"]; A = geo["areas_um2"]
    nm3 = 1e9  # nm^3 per um^3
    out = []
    for r in ing.itertuples():
        w = json.loads(r.comp_weights)
        if r.ing_id == "NPC":
            continue  # placed explicitly (one pore per column; expected 0.74)
        tm = (float(r.n_tm) > 0 if r.n_tm == r.n_tm else False) or (r.membrane in ("PM", "NE", "VAC"))
        n = r.copies_per_cell
        for c, f in w.items():
            cp = n * f
            if cp <= 0:
                continue
            if c == "cytosol":
                if not tm:  # TM proteins mis-assigned to cytosol by the weight split are not placed there
                    out.append((r.ing_id, "cytosol", cp * rv["cytosol"] / (V["cytosol"] * nm3)))
            elif c == "nucleoplasm":
                out.append((r.ing_id, "nucleoplasm", 0 if tm else cp * rv["nucleoplasm"] / (V["nucleoplasm"] * nm3)))
            elif c == "cell_wall":
                out.append((r.ing_id, "cell_wall", cp * rv["cell_wall"] / (V["cell_wall"] * nm3)))
            elif c == "plasma_membrane":
                lam = cp * rv["pm_area"] / (A["plasma_membrane"] * 1e6)
                out.append((r.ing_id, "PM" if tm else "PM_peripheral", lam))
            elif c == "ER":
                lam = cp * rv["er_area"] / (A["ER"] / 2 * 1e6)
                out.append((r.ing_id, "ER_membrane" if tm else "ER_lumen", lam))
            elif c == "nuclear_envelope":
                lam = cp * rv["ne_area"] / (A["nuclear_envelope"] / 2 * 1e6)
                out.append((r.ing_id, "NE" if tm else "NE_peripheral", lam))
    e = pd.DataFrame(out, columns=["ing_id", "region", "expected"])
    e = e[e.expected > 0].groupby(["ing_id", "region"], as_index=False).expected.sum()
    return e


def regions():
    rim = (LAY["erB"] - LAY["erA"]) / 2 + HALF + 1
    erbox = [LAY["er_x"][0] - rim, LAY["er_x"][1] + rim, LAY["erA"] - HALF, LAY["erB"] + HALF]
    R = {
        "cytosol": dict(kind="volume", zlo=LAY["pm"] + HALF, zhi=LAY["onm"] - HALF, boxes_out=[erbox]),
        "PM_peripheral": dict(kind="volume", zlo=LAY["pm"] + HALF, zhi=LAY["pm"] + 22, boxes_out=[erbox]),
        "NE_peripheral": dict(kind="volume", zlo=LAY["onm"] - 20, zhi=LAY["onm"] - HALF, boxes_out=[erbox]),
        "nucleoplasm": dict(kind="volume", zlo=LAY["inm"] + HALF, zhi=LAY["top"]),
        "cell_wall": dict(kind="volume", zlo=LAY["wall_out"], zhi=LAY["mannan_end"] + 10),
        "ER_lumen": dict(kind="volume", zlo=LAY["erA"] + HALF, zhi=LAY["erB"] - HALF, boxes_in=[[LAY["er_x"][0] + 3, LAY["er_x"][1] - 3, LAY["erA"] + HALF, LAY["erB"] - HALF]]),
        "PM": dict(kind="membrane", z0=LAY["pm"], flip=False),
        "ER_membrane_A": dict(kind="membrane", z0=LAY["erA"], flip=True, boxes_in=[[LAY["er_x"][0] + 4, LAY["er_x"][1] - 4, -1e9, 1e9]]),
        "ER_membrane_B": dict(kind="membrane", z0=LAY["erB"], flip=False, boxes_in=[[LAY["er_x"][0] + 4, LAY["er_x"][1] - 4, -1e9, 1e9]]),
        "NE_ONM": dict(kind="membrane", z0=LAY["onm"], flip=True, hole=(L / 2, L / 2, NPC_HOLE + (LAY["inm"] - LAY["onm"]) / 2)),
        "NE_INM": dict(kind="membrane", z0=LAY["inm"], flip=False, hole=(L / 2, L / 2, NPC_HOLE + (LAY["inm"] - LAY["onm"]) / 2)),
    }
    return R


def stochastic_round(x, rng):
    return int(np.floor(x) + (rng.random() < (x - np.floor(x))))


# ---------------------------------------------------------------- assembly
from config import LIB


def load_lib(ids):
    lib = {}
    for i in ids:
        try:
            d = np.load(f"{LIB}/{i}.npz")
            lib[i] = {k: d[k] for k in d.files}
        except FileNotFoundError:
            pass
    return lib


def chromatin_chain(pk, n_target, nuc_lib, rng, step=12.5, zlo=None, zhi=None):
    """Nucleosome chain by self-avoiding random walk with persistence (lattice-free analogue of the paper's
    lattice nucleoid). Returns placed count and linker-DNA bead coordinates."""
    zlo = zlo or LAY["inm"] + HALF + 6; zhi = zhi or LAY["top"] - 6
    st_c, st_r = nuc_lib["st_c"], nuc_lib["st_r"]
    placed, linkers = 0, []
    prev, d = None, None
    fails = 0
    while placed < n_target and fails < 4000:
        if prev is None:
            pos = np.array([rng.uniform(0, L), rng.uniform(0, L), rng.uniform(zlo, zhi)])
        else:
            nd = rng.normal(size=3); nd /= np.linalg.norm(nd)
            d = nd if d is None else (0.5 * d + nd); d /= np.linalg.norm(d)
            pos = prev + d * step
        if not (zlo < pos[2] < zhi):
            fails += 1; prev = None if fails % 50 == 0 else prev; continue
        from packer import random_rotation
        Rm = random_rotation(rng)
        P = st_c @ Rm.T + pos
        if pk.overlaps(P, st_r):
            fails += 1
            if fails % 30 == 0:
                prev = None
            continue
        pk.insert(P, st_r); pk.placed.append(("nucleosome", Rm, pos.copy()))
        if prev is not None:
            seg = np.linspace(prev, pos, 14)[2:-2]
            linkers.append(seg)
        prev = pos; placed += 1
    return placed, (np.vstack(linkers) if linkers else np.zeros((0, 3)))


def build_column(ing, geo, oligo_map, seed=7):
    rng = np.random.default_rng(seed)
    exp = expected_counts(ing, geo)
    # oligomer substitution: copies of monomer -> copies/n of oligomer ingredient
    rows = []
    for r in exp.itertuples():
        if r.ing_id in oligo_map:
            oid, n = oligo_map[r.ing_id]
            rows.append((oid, r.region, r.expected / n, r.ing_id))
        else:
            rows.append((r.ing_id, r.region, r.expected, r.ing_id))
    exp = pd.DataFrame(rows, columns=["lib_id", "region", "expected", "ing_id"])
    exp["n"] = [stochastic_round(x, rng) for x in exp.expected]
    lib = load_lib(exp.lib_id.unique().tolist() + ["NPC"])
    exp["has_struct"] = exp.lib_id.isin(lib.keys())
    pk = Packer(L, LAY["top"] + 10, h=6.0, seed=seed)
    R = regions()
    log = []
    # 1. NPC at the centre of the NE
    npc = lib["NPC"]
    pos = np.array([L / 2, L / 2, (LAY["onm"] + LAY["inm"]) / 2])
    pk.insert(npc["st_c"] + pos, npc["st_r"]); pk.placed.append(("NPC", np.eye(3), pos))
    tags = ["NPC"]
    # 2. membrane proteins
    memb = exp[exp.region.isin(["PM", "ER_membrane", "NE"]) & exp.has_struct].copy()
    memb["size"] = [lib[i]["st_r"].sum() for i in memb.lib_id]
    for r in memb.sort_values("size", ascending=False).itertuples():
        d = lib[r.lib_id]
        if r.region == "PM":
            targets = [("PM", r.n)]
        elif r.region == "ER_membrane":
            a = rng.binomial(r.n, 0.5); targets = [("ER_membrane_A", a), ("ER_membrane_B", r.n - a)]
        else:
            a = rng.binomial(r.n, 0.5); targets = [("NE_ONM", a), ("NE_INM", r.n - a)]
        for reg, n in targets:
            if n:
                ok = pk.try_place(r.lib_id, d["st_c"], d["st_r"], R[reg], n)
                log.append((r.lib_id, reg, n, ok)); tags += [reg] * ok
    # 3. chromatin
    nuc_n = int(exp[(exp.lib_id == "nucleosome")].n.sum())
    got, linker = chromatin_chain(pk, nuc_n, lib["nucleosome"], rng)
    log.append(("nucleosome", "nucleoplasm", nuc_n, got)); tags += ["chromatin"] * got
    # 4. soluble, largest first
    sol = exp[~exp.region.isin(["PM", "ER_membrane", "NE"]) & exp.has_struct & (exp.lib_id != "nucleosome") & (exp.n > 0)].copy()
    sol["size"] = [lib[i]["st_r"].max() * 0 + (lib[i]["beads"].shape[0]) for i in sol.lib_id]
    for r in sol.sort_values("size", ascending=False).itertuples():
        d = lib[r.lib_id]
        ok = pk.try_place(r.lib_id, d["st_c"], d["st_r"], R[r.region], r.n, max_tries=300)
        log.append((r.lib_id, r.region, r.n, ok)); tags += [r.region] * ok
    pk.tags = tags
    assert len(tags) == len(pk.placed)
    log = pd.DataFrame(log, columns=["lib_id", "region", "requested", "placed"])
    return pk, exp, log, linker, lib
