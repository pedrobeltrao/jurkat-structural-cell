"""Actin patches (inventory v3, item D): Arp2/3-branched actin networks at endocytic sites of the plasma membrane.

Structures (all composed in the frame of the library F-actin subunit from 9GO5, see cytoskeleton.py):
  Arp23    S. cerevisiae Arp2/3 complex: AlphaFold models of ARP2, ARP3, ARC40, ARC35, ARC18, ARC19, ARC15
           superposed on their S. pombe homologues in the branch-junction cryo-EM structure 8UXW (2.7 A).
           Also the inventory's Arp2/3 complex (free, inactive copies use the same model).
  CP_cap   capping protein (CAP1/CAP2 AlphaFold models) superposed on CapZ at a capped barbed end, 8F8Q (2.8 A).
Branch geometry: 8UXW gives the Arp2/3 pose and the first daughter actin relative to the mother-filament subunit
they sit on (73 deg branch); 8F8Q gives the capping-protein pose relative to the terminal actin. The library actin
fits every actin of both structures to 0.03 nm. +z of the actin frame points to the barbed end.

Layout (assumptions): patches sit in the cytosol directly under the plasma membrane where there is no cortical ER;
each grows from a few primer filaments whose barbed ends face the membrane, then branches (Arp2/3 on a random mother
subunit, daughters must not point away from the membrane), and every filament is capped after ~CAP_LEN subunits.
ACTIN_PER_PATCH sets the size; the number of patches follows from the F-actin reserved for patches.
"""
import os, json
import numpy as np
import gemmi
import pack3d as p3
import cell3d as c3
import build_structs as bs
from config import LIB

ARP23 = "Arp23"
CP = "CP_cap"
CP_ING = "CAP2_CAP1_het"     # inventory form of capping protein (consumed by the caps)
ACTIN = "F_actin"
LIB_IDS = (ARP23, CP)
ARP23_GENES = {"ARP3": "A", "ARP2": "B", "ARC40": "C", "ARC35": "D", "ARC18": "E", "ARC19": "F", "ARC15": "G"}   # -> 8UXW chain
CP_GENES = {"CAP1": "G", "CAP2": "H"}                                                                            # -> 8F8Q chain
PATCH_FRACTION = 0.3        # share of ACT1 in patches (assumption; cables take cytoskeleton.CABLE_FRACTION)
ACTIN_PER_PATCH = 1500
CAP_LEN = (20, 6)            # subunits per filament before capping (mean, sd): ~55 nm
N_PRIMER = 3
PATCH_R, PATCH_DEPTH = 130.0, 220.0
H = 20.0
GEOM_F = f"{LIB}/actin_branch.json"


def _ca(st, ch):
    return {r.seqid.num: np.array(r.find_atom("CA", "*").pos.tolist()) / 10 for r in st[0][ch] if r.find_atom("CA", "*")}


def _kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def _fit(P, Q, trim=3):
    R, t = _kabsch(P, Q)
    for _ in range(trim):
        r = np.linalg.norm(P @ R.T + t - Q, axis=1); m = r < max(0.2, np.percentile(r, 80)); R, t = _kabsch(P[m], Q[m])
    return R, t, float(np.median(np.linalg.norm(P @ R.T + t - Q, axis=1)))


def _fit_num(L, W):
    k = sorted(set(L) & set(W))
    return _fit(np.array([L[i] for i in k]), np.array([W[i] for i in k]))


def _seq(ch):
    return gemmi.one_letter_code([r.name for r in ch])


def _homolog_fit(af_st, tmpl_st, tmpl_chain):
    """superpose an AlphaFold monomer on a homologous template chain via a global sequence alignment"""
    a = af_st[0][0]; b = tmpl_st[0][tmpl_chain]
    al = gemmi.align_string_sequences(list(_seq(a)), list(_seq(b)), [])
    ia = ib = 0; P, Q = [], []
    # walk the aligned strings (gap '-'); well-predicted residues only
    sa, sb = al.add_gaps(_seq(a), 1), al.add_gaps(_seq(b), 2)
    for x, y in zip(sa, sb):
        if x != "-" and y != "-":
            ca, cb = a[ia].find_atom("CA", "*"), b[ib].find_atom("CA", "*")
            if ca and cb and a[ia].find_atom("CA", "*").b_iso >= 70:
                P.append(np.array(ca.pos.tolist()) / 10); Q.append(np.array(cb.pos.tolist()) / 10)
        ia += x != "-"; ib += y != "-"
    R, t, rmsd = _fit(np.array(P), np.array(Q))
    return R, t, rmsd, al.calculate_identity()


def _af_structures(accs):
    import tarfile, gzip
    from config import AFDB_TAR
    out = {}
    with tarfile.open(AFDB_TAR) as tf:
        for m in tf:
            n = os.path.basename(m.name)
            if n.endswith(".cif.gz") and "-F1-" in n and n.split("-")[1] in accs:
                out[n.split("-")[1]] = gemmi.make_structure_from_block(gemmi.cif.read_string(gzip.decompress(tf.extractfile(m).read()).decode())[0])
                if len(out) == len(accs):
                    break
    return out


def _pose_inv(R, t):
    return R.T, -R.T @ t


def _pose_mul(a, b):
    return a[0] @ b[0], a[0] @ b[1] + a[1]


def model(acc_of, log=print):
    """library entries Arp23 and CP_cap (+ F_actin) and the branch/cap geometry (poses relative to an actin subunit)"""
    if os.path.exists(GEOM_F) and all(os.path.exists(f"{LIB}/{x}.npz") for x in LIB_IDS):
        return json.load(open(GEOM_F))
    import extend_complexes as xc, cytoskeleton as cs
    cs.filament_model()
    B = np.load(f"{LIB}/{ACTIN}.npz")["beads"].astype(np.float64)
    s9 = gemmi.read_structure(xc.fetch_assembly("9GO5"))
    ch9 = next(c.name for c in s9[0] if sum(1 for r in c if r.find_atom("CA", "*")) == len(B))
    L = dict(zip([r.seqid.num for r in s9[0][ch9] if r.find_atom("CA", "*")], B))
    af = _af_structures({acc_of[g] for g in list(ARP23_GENES) + list(CP_GENES)})
    geo, info = {}, {}
    for pid, genes, lid in (("8UXW", ARP23_GENES, ARP23), ("8F8Q", CP_GENES, CP)):
        st = gemmi.read_structure(xc.fetch_assembly(pid)); st.setup_entities()
        actins = [c.name for c in st[0] if c.name not in genes.values() and sum(1 for r in c if r.find_atom("CA", "*")) > 300]
        poses = {c: _fit_num(L, _ca(st, c)) for c in actins}
        info[pid] = {"actin_fit_nm": round(max(p[2] for p in poses.values()), 3)}
        # complex beads in the template frame, from the AlphaFold models (pLDDT >= 50 kept)
        X, K, PL, fits = [], [], [], {}
        for g, ch in genes.items():
            s = af[acc_of[g]]
            R, t, rmsd, ident = _homolog_fit(s, st, ch)
            fits[g] = dict(chain=ch, identity=round(ident, 1), rmsd_nm=round(rmsd, 2))
            for r in s[0][0]:
                a = r.find_atom("CA", "*")
                if a is not None and a.b_iso >= 50:
                    X.append(R @ (np.array(a.pos.tolist()) / 10) + t); PL.append(a.b_iso)
        X = np.array(X); cx = X.mean(0)
        bs.save(lid, X - cx, np.zeros(len(X), np.int8),
                extra=dict(pdb=pid, src="AF", plddt=np.array(PL, np.float32), nchain=len(genes)), per=30)
        info[pid]["subunits"] = fits
        cpose = (np.eye(3), cx)                       # library -> template frame
        axes = {c: poses[c][0][:, 2] for c in actins}
        if pid == "8UXW":
            # mother vs daughter: two groups of parallel actins (73 deg apart); the daughter group is the smaller one
            g0 = [c for c in actins if abs(axes[c] @ axes[actins[0]]) > 0.95]; g1 = [c for c in actins if c not in g0]
            mother, daughter = (g0, g1) if len(g0) > len(g1) else (g1, g0)
            d0 = min(daughter, key=lambda c: np.linalg.norm(poses[c][1] - cx))
            m0 = min(mother, key=lambda c: np.linalg.norm(poses[c][1] - cx))
            Pm0 = (poses[m0][0], poses[m0][1]); inv = _pose_inv(*Pm0)
            geo["arp"] = [x.tolist() for x in _pose_mul(inv, cpose)]
            geo["d0"] = [x.tolist() for x in _pose_mul(inv, (poses[d0][0], poses[d0][1]))]
            geo["branch_angle_deg"] = float(np.degrees(np.arccos(abs(axes[mother[0]] @ axes[daughter[0]]))))
            geo["daughter_away"] = float((poses[d0][1] - cx) @ axes[d0])     # > 0: +z points away from Arp2/3 (barbed end)
        else:
            # terminal actin: the one whose next subunit along +z (barbed end) is nearest the capping protein
            term = min(actins, key=lambda c: np.linalg.norm(poses[c][1] + 2.76 * axes[c] - cx))
            geo["cap"] = [x.tolist() for x in _pose_mul(_pose_inv(poses[term][0], poses[term][1]), cpose)]
    geo["info"] = info
    json.dump(geo, open(GEOM_F, "w"), indent=1)
    log(f"actin branch model: {json.dumps(info)}; branch angle {geo['branch_angle_deg']:.0f} deg")
    return geo


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _quat(Rm):
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(Rm).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def place(cell, lp, n_actin, n_arp_max, n_cp_max, obstacles=None, seed=41, log=print):
    """patches using up to n_actin actin subunits (and at most n_arp_max Arp2/3 / n_cp_max capping proteins)"""
    geo = json.load(open(GEOM_F)); hel = json.load(open(f"{LIB}/{ACTIN}_helix.json"))
    phi, rise, r0 = np.radians(hel["phi_deg"]), hel["rise"], hel["r0"]
    S = (_rz(phi), _rz(phi) @ np.array([r0, 0, 0]) - np.array([r0, 0, 0]) + np.array([0, 0, rise]))   # next subunit (barbed)
    T_arp = tuple(np.array(x) for x in geo["arp"]); T_d0 = tuple(np.array(x) for x in geo["d0"])
    T_cap = tuple(np.array(x) for x in geo["cap"])
    li = {k: int(np.where(lp["ids"] == k)[0][0]) for k in (ACTIN, ARP23, CP)}
    sub = {k: (lp["STC"][lp["st_off"][v]:lp["st_off"][v + 1]], lp["STR"][lp["st_off"][v]:lp["st_off"][v + 1]], float(lp["BR"][v]))
           for k, v in li.items()}
    rng = np.random.default_rng(seed)
    G = cell.G
    glo = np.full(3, -G[0] - 10.0); ghi = -glo
    a = cell.args(glo, ghi, margin=50.0)
    cap = int(n_actin * 1.3) * len(sub[ACTIN][1]) + 200000 + (len(obstacles[1]) if obstacles is not None else 0)
    G1 = p3.grid_new(glo, ghi, H, cap, cap * 14)
    if obstacles is not None and len(obstacles[1]):
        k = np.all((obstacles[0] > glo) & (obstacles[0] < ghi), axis=1)
        p3.g_insert(glo, H, *G1, obstacles[0][k], obstacles[1][k])
    out = np.zeros(c3.NMEM); M = np.empty((3, 3))
    from scipy.spatial import cKDTree
    BEADS = {k: np.load(f"{LIB}/{k}.npz")["beads"].astype(np.float64) for k in (ACTIN, ARP23, CP)}

    def spheres(lid, pose):
        stc, str_, _ = sub[lid]
        P = np.empty((len(stc), 3)); p3.quat_mat(_quat(pose[0]), M); p3.transform(stc, M, pose[1], P)
        return P

    def ok(lid, pose, grids, site, n):
        P = spheres(lid, pose); str_, br = sub[lid][1], sub[lid][2]
        d = pose[1] - site; depth = -(d @ n); lat = np.linalg.norm(d + depth * n)
        if depth > PATCH_DEPTH or lat > PATCH_R:
            return None
        for gr in grids:
            if p3.g_overlap(glo, H, *gr[:6], P, str_, p3.TOL):
                return None
        if not p3.check_volume(P, str_, pose[1][0], pose[1][1], pose[1][2], br, 1 << c3.CYTO, *a, out):
            return None
        return P

    def find_site():
        for _ in range(200):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            r0 = c3.ray_depth(G, u, G[0] - G[2] + 17.5)           # just inside the plasma membrane
            for r in r0 - np.arange(0.0, 6.0, 0.5):
                p = u * r
                c3.mem_sdfs(p[0], p[1], p[2], *a, out)
                if abs(out[c3.M_PM]) < 18.0:
                    m = c3.mem_sdfs(p[0], p[1], p[2], *a, out); l = c3.label_from(p[0], p[1], p[2], G, cell.pores, out, m)
                    if l == c3.CYTO and c3.min_mem_dist(out, -1) >= abs(out[c3.M_PM]) - 0.5:   # PM is the nearest membrane
                        nrm = -c3._grad(lambda x, y, z: c3.outer_depth(x, y, z, G), p); return p, nrm
                    break
        return None, None

    pos, rot, typ, pid = [], [], [], []
    used = {ACTIN: 0, ARP23: 0, CP: 0}
    npatch = 0
    while used[ACTIN] + ACTIN_PER_PATCH * 0.5 <= n_actin and npatch < 400:
        site, n = find_site()
        if site is None:
            break
        target = min(ACTIN_PER_PATCH, n_actin - used[ACTIN])
        G2 = p3.grid_new(glo, ghi, H, int(target * 1.6) * len(sub[ACTIN][1]) + 50000, (int(target * 1.6) * len(sub[ACTIN][1]) + 50000) * 14)
        items = []                      # (lib, pose, spheres)
        fils = []                       # (subunit poses, filament id) for branching
        n_act = 0
        # residue-level record of the patch, for the parts that must touch their mother filament (Arp2/3, first two
        # daughter subunits): they are checked against everything in the patch except that mother
        bx = {"tree": None, "lab": None, "pend": [], "ptree": None, "plab": None, "all": [], "alab": []}

        def beads(lid, pose):
            return BEADS[lid] @ pose[0].T + pose[1]

        def commit(X, fid):
            bx["pend"].append((X, fid)); bx["ptree"] = None
            if sum(len(x) for x, _ in bx["pend"]) > 40000:
                bx["all"] += [x for x, _ in bx["pend"]]; bx["alab"] += [np.full(len(x), f) for x, f in bx["pend"]]
                X_ = np.concatenate(bx["all"]); bx["tree"] = cKDTree(X_); bx["lab"] = np.concatenate(bx["alab"]); bx["pend"] = []

        def bead_clear(Xc, excl):
            for tk, lk in (("tree", "lab"), ("ptree", "plab")):
                if tk == "ptree" and bx["ptree"] is None and bx["pend"]:
                    bx["ptree"] = cKDTree(np.concatenate([x for x, _ in bx["pend"]]))
                    bx["plab"] = np.concatenate([np.full(len(x), f) for x, f in bx["pend"]])
                if bx[tk] is None:
                    continue
                d, k = bx[tk].query(Xc, distance_upper_bound=0.35)
                hit = np.isfinite(d)
                if hit.any() and (bx[lk][k[hit]] != excl).any():
                    return False
            return True

        def grow(first, arp=None, mother=-1):
            """grow a filament from pose `first`; returns (subunit poses, id) (inserted into G2 with its Arp2/3 and cap)"""
            nonlocal n_act
            fid = len(fils)
            L = max(4, int(rng.normal(*CAP_LEN)))
            poses, Ps = [], []
            pose = first
            for j in range(L):
                if n_act + len(poses) >= target:
                    break
                near = arp is not None and j < 2
                P = ok(ACTIN, pose, (G1,) if near else (G1, G2), site, n)
                if P is None or (near and not bead_clear(beads(ACTIN, pose), mother)):
                    break
                poses.append(pose); Ps.append(P)
                pose = _pose_mul(pose, S)
            if len(poses) < (3 if arp is not None else 4):
                return None
            new = [(ACTIN, p, P) for p, P in zip(poses, Ps)]
            if arp is not None:
                new.append(arp)
            if used[CP] + sum(1 for x in items if x[0] == CP) < n_cp_max:
                cp_pose = _pose_mul(poses[-1], T_cap)
                Pc = ok(CP, cp_pose, (G1, G2), site, n)
                if Pc is not None and bead_clear(beads(CP, cp_pose), fid):
                    new.append((CP, cp_pose, Pc))
            for lid, p, P in new:
                p3.g_insert(glo, H, *G2, P, sub[lid][1])
                commit(beads(lid, p), fid if lid != CP else -2 - fid)   # a cap may never touch a new branch on its filament
            items.extend(new); n_act += len(poses)
            return poses, fid

        # primers: barbed ends toward the membrane, starting ~100-150 nm inside
        for _ in range(N_PRIMER * 20):
            if len(fils) >= N_PRIMER:
                break
            w = n + 0.8 * rng.normal(size=3); w /= np.linalg.norm(w)
            if w @ n < 0.3:
                continue
            e1 = np.cross(w, rng.normal(size=3)); e1 /= np.linalg.norm(e1)
            start = site - n * rng.uniform(100, 150) + (np.eye(3) - np.outer(n, n)) @ rng.normal(size=3) * 40
            f = grow((np.c_[e1, np.cross(w, e1), w], start))
            if f:
                fils.append(f)
        # dendritic branching
        tries = 0
        while n_act < target and fils and tries < 30 * ACTIN_PER_PATCH // CAP_LEN[0]:
            tries += 1
            if used[ARP23] + sum(1 for x in items if x[0] == ARP23) >= n_arp_max:
                break
            f, mfid = fils[rng.integers(len(fils))]
            if len(f) < 3:
                continue
            m = f[rng.integers(1, len(f) - 1)]
            arp_pose = _pose_mul(m, T_arp); d0 = _pose_mul(m, T_d0)
            if d0[0][:, 2] @ n < -0.1:                  # daughters do not grow away from the membrane
                continue
            Pa = ok(ARP23, arp_pose, (G1,), site, n)
            if Pa is None or not bead_clear(beads(ARP23, arp_pose), mfid):
                continue
            g = grow(d0, arp=(ARP23, arp_pose, Pa), mother=mfid)
            if g:
                fils.append(g)
        if n_act < 0.3 * target:
            continue                                     # crowded site: try another
        for lid, p, P in items:
            p3.g_insert(glo, H, *G1, P, sub[lid][1])
            pos.append(p[1]); rot.append(_quat(p[0])); typ.append(li[lid]); pid.append(npatch)
            used[lid] += 1
        npatch += 1
    log(f"actin patches: {npatch} patches, {used[ACTIN]} actin subunits (of {n_actin:.0f} reserved), {used[ARP23]} Arp2/3 branches, "
        f"{used[CP]} capping proteins; {used[ACTIN] / max(npatch, 1):.0f} actin per patch")
    return dict(pat_pos=np.array(pos, np.float32).reshape(-1, 3), pat_q=np.array(rot, np.float32).reshape(-1, 4),
                pat_type=np.array(typ, np.int32), pat_id=np.array(pid, np.int32)), used


def add_arp23(ing, df, log=print):
    """Arp2/3 as one 7-subunit complex (copies: median of the subunit pools, capped at 2x the limiting one)"""
    import assemblies as asm
    acc_of = dict(zip(df.gene, df.uniprot))
    model(acc_of, log=log)
    i = ing.set_index("ing_id")
    pool = {g: float(i.loc[g, "copies_per_cell"]) if g in i.index and i.loc[g, "kind"] == "protein" else 0.0 for g in ARP23_GENES}
    n = asm.complex_copies(pool, {g: 1 for g in ARP23_GENES})
    ing = ing.copy()
    for g in ARP23_GENES:
        r = (ing.ing_id == g) & (ing.kind == "protein")
        ing.loc[r, "copies_per_cell"] = (ing.loc[r, "copies_per_cell"] - n).clip(lower=0)
    m = df.set_index("gene").mass_da
    row = dict(ing_id=ARP23, label="Arp2/3 complex", kind="complex", genes=";".join(ARP23_GENES), n_subunit_types=7,
               copies_per_cell=n, comp_weights=json.dumps({"cytosol": 1.0}), mass_da=float(sum(m[g] for g in ARP23_GENES)),
               pdb="8UXW", membrane=np.nan, oligomer_n=1, uniprot="", n_tm=0)
    import pandas as pd
    ing = pd.concat([ing[ing.copies_per_cell > 0.5], pd.DataFrame([row])], ignore_index=True)
    log(f"Arp2/3 complex: {n:.0f} copies (subunit pools {', '.join(f'{g} {v:.0f}' for g, v in pool.items())})")
    return ing
