"""ER-bound ribosomes (inventory v3, item C1): 80S ribosomes docked on a translocon in the ER membrane.

Geometry from PDB 2WW9 (yeast 80S ribosome-Ssh1 complex, cryo-EM 8.6 A; Becker et al. 2009): its tunnel-exit
ribosomal proteins superpose the 4V88 ribosome of the library onto the translocon. Two library types:
  ribosome_80S_Ssh1    4V88 + SSH1/SSS1/SBH2 from 2WW9
  ribosome_80S_Sec61   4V88 + SEC61/SSS1/SBH1 from 7KAH (3.1 A), placed by superposing its SSS1 on the 2WW9 SSS1
Both are oriented in the membrane frame from the translocon's TM segments (ribosome on the cytosolic side) and pack
as ER-membrane (and outer nuclear membrane) assemblies.

Copies: ribosome-bound translocons are limited by the free translocon pools (SSH1 complex; SEC61 trimers not in
the SEC complex). Assumption: BOUND_FRACTION of those pools carries a ribosome. The docked ribosomes come out of the
ribosome_80S ingredient, replacing its former 3% 'ER' share (free ribosomes near the ER).
"""
import os, json
import numpy as np
import pandas as pd
import gemmi
import extend_complexes as xc
import assemblies as A
import membrane_orient as mo
import structlib as sl
import build_structs as bs
from config import LIB

RIBO = "ribosome_80S"
TYPES = {"ribosome_80S_Ssh1": ("SSH1", "SSS1", "SBH2"), "ribosome_80S_Sec61": ("SEC61", "SSS1", "SBH1")}
BOUND_FRACTION = 0.8
MAX_ER_FRACTION = 0.1        # never more than this share of ribosomes on the ER
INFO_F = f"{LIB}/er_ribosomes.json"


def _kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def _beads(pid, keep_acc=None):
    st = gemmi.read_structure(xc.fetch_assembly(pid)); st.remove_ligands_and_waters(); st.setup_entities()
    xyz, kind, acc, pos = mo.beads_with_positions(st, align=A.entity_align(pid))
    if keep_acc is not None:
        k = np.isin(acc, list(keep_acc)); xyz, kind, acc, pos = xyz[k], kind[k], acc[k], pos[k]
    return xyz, kind, acc, pos


def _fit(x0, a0, p0, x1, a1, p1, trim=True):
    """transform taking structure 1 onto structure 0 from residues with the same (UniProt, position)"""
    key0 = {}
    for i, k in enumerate(zip(a0, p0)):
        if k[0] and k[1] > 0:
            key0.setdefault(k, i)
    I, J = [], []
    for j, k in enumerate(zip(a1, p1)):
        if k in key0:
            I.append(key0[k]); J.append(j)
    P, Q = x1[J], x0[I]; R, t = _kabsch(P, Q)
    for _ in range(4 if trim else 0):
        res = np.linalg.norm(P @ R.T + t - Q, axis=1)
        keep = res < max(0.3, np.percentile(res, 75)); R, t = _kabsch(P[keep], Q[keep])
    res = np.linalg.norm(P @ R.T + t - Q, axis=1)
    return R, t, len(I), float(np.median(res))


def build(acc_of, log=print):
    """library entries for both types; returns {lib_id: info}"""
    if os.path.exists(INFO_F) and all(os.path.exists(f"{LIB}/{k}.npz") for k in TYPES):
        return json.load(open(INFO_F))
    # ribosome library frame = 4V88 beads minus their mean (as in build_structs / polysome.channel_path)
    xs, ks, _ = sl.beads_from_structure(gemmi.read_structure(xc.fetch_assembly("4V88")))
    lib = np.load(f"{LIB}/{RIBO}.npz")
    assert len(lib["beads"]) == len(xs) and np.allclose(lib["beads"], xs - xs.mean(0), atol=1e-3)
    x0, _, a0, p0 = _beads("4V88")
    ribo_ss = {a for s in A.entity_uniprot("2WW9").values() for a in s} - {acc_of[g] for g in ("SSH1", "SSS1", "SBH2")}
    x1, _, a1, p1 = _beads("2WW9", ribo_ss)
    R, t, n, rmsd = _fit(x0, a0, p0, x1, a1, p1)
    log(f"  2WW9 -> 4V88: {n} matched tunnel-exit residues, median deviation {rmsd:.2f} nm")
    trans = {}
    xt, kt, at, pt = _beads("2WW9", {acc_of[g] for g in ("SSH1", "SSS1", "SBH2")})
    trans["ribosome_80S_Ssh1"] = (xt @ R.T + t, kt, at, pt)
    # Sec61 trimer from 7KAH onto the 2WW9 translocon via SSS1 (identical subunit)
    xs7, ks7, as7, ps7 = _beads("7KAH", {acc_of[g] for g in ("SEC61", "SSS1", "SBH1")})
    k_sss = acc_of["SSS1"]
    R2, t2, n2, d2 = _fit(xt[at == k_sss], at[at == k_sss], pt[at == k_sss], xs7[as7 == k_sss], as7[as7 == k_sss], ps7[as7 == k_sss], trim=False)
    x7 = (xs7 @ R2.T + t2) @ R.T + t
    from scipy.spatial import cKDTree
    d_ch = cKDTree(trans["ribosome_80S_Ssh1"][0][at == acc_of["SSH1"]]).query(x7[as7 == acc_of["SEC61"]])[0]
    log(f"  7KAH SEC61 trimer on SSS1 ({n2} residues, deviation {d2:.2f} nm); SEC61 to SSH1 channel: median nearest {np.median(d_ch):.2f} nm")
    trans["ribosome_80S_Sec61"] = (x7, ks7, as7, ps7)
    xr = lib["beads"].astype(np.float64)
    info = {}
    for lid, (xt_, kt_, at_, pt_) in trans.items():
        # translocon beads that would sit inside the ribosome (flexible loops at the junction) are dropped
        near = cKDTree(xr).query(xt_)[0] < 0.4
        xt_, kt_, at_, pt_ = xt_[~near], kt_[~near], at_[~near], pt_[~near]
        tm, ns, ps = mo.masks(at_, pt_, xt_)
        X = np.r_[xr, xt_]; K = np.r_[lib["kind"], kt_]
        TM = np.r_[np.zeros(len(xr), bool), tm]; NS = np.r_[np.zeros(len(xr), bool), ns]; PS = np.r_[np.zeros(len(xr), bool), ps]
        Xo, oi = mo.orient(X, TM, NS, PS, min_score=0.3)
        if Xo is None:
            log(f"  {lid}: orientation failed {oi}"); continue
        zr = float(np.median(Xo[:len(xr), 2]))
        if zr < 0:
            log(f"  {lid}: ribosome came out on the lumenal side (median z {zr:.1f}); flipped")
            Xo = Xo * [1, -1, -1]
        bs.save(lid, Xo, K, extra=dict(pdb="2WW9", membrane="TM", tm=TM, memb_score=np.float32(oi["score"])), per=30)
        info[lid] = dict(score=oi["score"], sign_from=oi["sign_from"], ribosome_median_z=abs(zr), dropped_junction=int(near.sum()),
                         n_tm=int(tm.sum()))
        log(f"  {lid}: membrane score {oi['score']}, ribosome centre {abs(zr):.1f} nm above the midplane")
    json.dump(info, open(INFO_F, "w"))
    return info


def assign(ing, df, log=print):
    """add the docked-ribosome ingredients, consuming translocon subunits and ribosomes from the pools"""
    acc_of = dict(zip(df.gene, df.uniprot))
    info = build(acc_of, log=log)
    i = ing.set_index("ing_id")
    free = {g: float(i.loc[g, "copies_per_cell"]) if g in i.index and i.loc[g, "kind"] == "protein" else 0.0
            for g in {g for v in TYPES.values() for g in v}}
    n_ribo = float(i.loc[RIBO, "copies_per_cell"]); left = MAX_ER_FRACTION * n_ribo
    rows, used = [], {}
    # channel demand per type; the shared gamma subunit (SSS1) and the ER cap are split in proportion to it
    want = {lid: BOUND_FRACTION * free[TYPES[lid][0]] for lid in TYPES if lid in info}
    f = min(1.0, free["SSS1"] / max(sum(want.values()), 1e-9), left / max(sum(want.values()), 1e-9))
    for lid in ("ribosome_80S_Ssh1", "ribosome_80S_Sec61"):
        if lid not in info:
            continue
        a, b, c = TYPES[lid]
        n = want[lid] * f                                       # beta subunit optional (SBH1/2 are non-essential)
        if n < 1:
            continue
        nb = min(n, free[c])
        for g, k in ((a, n), (b, n), (c, nb)):
            free[g] -= k; used[g] = used.get(g, 0) + k
        left -= n
        genes = sorted(set(i.loc[RIBO, "genes"].split(";")) | {a, b, c})
        rows.append(dict(ing_id=lid, label=f"80S ribosome on the {a} translocon (ER)", kind="complex", genes=";".join(genes),
                         n_subunit_types=len(genes), copies_per_cell=n, comp_weights=json.dumps({"ER": 1.0}),
                         mass_da=float(i.loc[RIBO, "mass_da"]) + sum(float(df.set_index("gene").loc[g, "mass_da"]) for g in (a, b, c)
                                                                     if "mass_da" in df.columns and g in set(df.gene)),
                         pdb="2WW9", membrane="TM", oligomer_n=1, uniprot="", n_tm=int(info[lid]["n_tm"] > 0) * 11))
        log(f"  {lid}: {n:.0f} copies ({a} pool {free[a] + n:.0f}, {c} {nb:.0f})")
    n_docked = sum(r["copies_per_cell"] for r in rows)
    ing = ing.copy()
    for g, k in used.items():
        ing.loc[(ing.ing_id == g) & (ing.kind == "protein"), "copies_per_cell"] -= k
    # the docked ribosomes replace the ribosome ingredient's ER share; the rest is cytosolic
    r = ing.ing_id == RIBO
    ing.loc[r, "copies_per_cell"] -= n_docked
    ing.loc[r, "comp_weights"] = json.dumps({"cytosol": 1.0})
    ing = ing[ing.copies_per_cell > 0.5]
    log(f"ER-bound ribosomes: {n_docked:.0f} ({n_docked / n_ribo:.1%} of {n_ribo:.0f})")
    return pd.concat([ing, pd.DataFrame(rows)], ignore_index=True), dict(n_docked=n_docked, used=used, info=info)
