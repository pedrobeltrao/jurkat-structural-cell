"""ATP synthase dimer rows on cristae rims (inventory v3, item B4).

Structure: two copies of the yeast ATP synthase monomer of the inventory (8F39, the CPX-3281 library entry) placed
by superposition (F1, gamma, c-ring residues) on the two monomers of 4B2Q, the yeast ATP synthase dimer modelled into
a cryo-ET subtomogram average (Davies et al. 2012). The dimer angle between the monomers' membrane normals is ~84 deg,
and their membrane centres lie on a circle of radius ~8.8 nm: the curvature of a crista rim. The library frame puts
the origin on that circle's axis, +z along the bisector of the two membrane normals (into the matrix), +x across the
ridge (from monomer 2 to monomer 1), +y along the ridge.

Placement: dimers are centred on the rim tube of lamellar cristae (cell3d C_CRISTA pills, tube radius CRISTA_HALF),
one every SPACING nm along the rim, ridge along the rim tangent, as rows. TM residues must lie within TM_TOL of the
inner-membrane midsurface; other residues must stay out of the membranes and inside the mitochondrion. Copies come
out of the monomer pool (2 per dimer); the rest stay monomers in the flat inner membrane.
"""
import os, json
import numpy as np
import gemmi
import pack3d as p3
import cell3d as c3
import build_structs as bs
from config import LIB

LIB_ID = "ATP_dimer"
MONOMER = "CPX_3281_Mitochondrial_proton_transportin"
SPACING = 13.0        # nm between dimers along a row (cryo-ET of yeast cristae: rows of dimers ~12-14 nm apart; assumption)
TM_TOL = 2.5
MAX_IN_MEMBRANE = 0.02   # share of non-TM residues allowed within 0.6 nm of a membrane midsurface
H = 10.0
INFO_F = f"{LIB}/{LIB_ID}.json"


def _kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def model(log=print):
    if os.path.exists(INFO_F) and os.path.exists(f"{LIB}/{LIB_ID}.npz"):
        return json.load(open(INFO_F))
    import extend_complexes as xc, assemblies as A, membrane_orient as mo
    lib = np.load(f"{LIB}/{MONOMER}.npz"); L = lib["beads"].astype(np.float64); tm = lib["tm"].astype(bool)
    st = gemmi.read_structure(xc.fetch_assembly(str(lib["pdb"]))); st.remove_ligands_and_waters(); st.setup_entities()
    x8, k8, a8, p8 = mo.beads_with_positions(st, align=A.entity_align(str(lib["pdb"])))
    assert len(x8) == len(L), (len(x8), len(L))     # the library entry holds all chains of the entry, same bead order
    Ro, to = _kabsch(x8, L)                          # structure -> library (membrane) frame
    s4 = gemmi.read_structure(xc.fetch_assembly("4B2Q")); s4.setup_entities()
    x4, _, a4, p4 = mo.beads_with_positions(s4, align=A.entity_align("4B2Q"))
    ch4 = np.array([c.name for c in s4[0] for r in c if r.find_atom("CA", "*")])
    key = {}
    for i, k in enumerate(zip(a8, p8)):
        if k[0] and k[1] > 0:
            key.setdefault(k, i)
    T, devs = [], []
    for sel in (np.char.isupper(ch4), np.char.islower(ch4)):
        I, J = [], []
        for j in np.where(sel)[0]:
            k = (a4[j], p4[j])
            if k in key:
                I.append(key[k]); J.append(j)
        P, Q = x8[I], x4[J]; R, t = _kabsch(P, Q)
        for _ in range(3):
            r = np.linalg.norm(P @ R.T + t - Q, axis=1); m = r < max(0.5, np.percentile(r, 80)); R, t = _kabsch(P[m], Q[m])
        devs.append(float(np.median(np.linalg.norm(P @ R.T + t - Q, axis=1))))
        # library -> 4B2Q frame
        T.append((R @ Ro.T, t - R @ Ro.T @ to))
    ez = np.array([0, 0, 1.0]); ctm = L[tm].mean(0); ctm[2] = 0
    n = [Ri @ ez for Ri, _ in T]; m = [Ri @ ctm + ti for Ri, ti in T]
    alpha = float(np.arccos(np.clip(n[0] @ n[1], -1, 1)))
    rt = float(np.linalg.norm(m[0] - m[1]) / np.linalg.norm(n[0] - n[1]))
    O = 0.5 * (m[0] + m[1] - rt * (n[0] + n[1]))
    z = (n[0] + n[1]) / np.linalg.norm(n[0] + n[1]); x = n[0] - n[1]; x -= (x @ z) * z; x /= np.linalg.norm(x)
    F = np.c_[x, np.cross(z, x), z]
    X = np.concatenate([(L @ Ri.T + ti - O) @ F for Ri, ti in T])
    from scipy.spatial import cKDTree
    gap = float(cKDTree(X[:len(L)]).query(X[len(L):])[0].min())
    bs.save(LIB_ID, X, np.r_[lib["kind"], lib["kind"]], extra=dict(pdb="4B2Q", membrane="TM", tm=np.r_[tm, tm], nchain=2), per=30)
    info = dict(dimer_angle_deg=float(np.degrees(alpha)), rim_radius_nm=rt, fit_median_dev_nm=devs, monomer_gap_nm=gap)
    json.dump(info, open(INFO_F, "w"))
    log(f"ATP synthase dimer: {info}")
    return info


def place(cell, lp, n_max, seed=7, log=print):
    from scipy.spatial.transform import Rotation
    info = model(log=log)
    li = int(np.where(lp["ids"] == LIB_ID)[0][0])
    s0, s1 = lp["st_off"][li], lp["st_off"][li + 1]
    stc, str_ = lp["STC"][s0:s1], lp["STR"][s0:s1]
    z = np.load(f"{LIB}/{LIB_ID}.npz"); B = z["beads"].astype(np.float64); tm = z["tm"].astype(bool)
    G = cell.G; glo = np.full(3, -G[0] - 10.0); ghi = -glo
    a = cell.args(glo, ghi, margin=50.0)
    cap = int(n_max * len(str_) * 1.1) + 1000
    grid = p3.grid_new(glo, ghi, H, cap, cap * 14)
    rng = np.random.default_rng(seed)
    out = np.zeros(c3.NMEM); M = np.empty((3, 3)); P = np.empty((len(stc), 3))
    ok_lab = {c3.MATRIX, c3.IMS}
    cr = [i for i, c in enumerate(cell.PC) if c == c3.C_CRISTA]
    rng.shuffle(cr)
    pos, quat, n_try = [], [], 0
    for i in cr:
        if len(pos) >= n_max:
            break
        q = cell.PQ[i]; c = q[:3]; ax = q[3:6] / np.linalg.norm(q[3:6]); R = q[6]
        e1 = np.cross(ax, rng.normal(size=3)); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
        th0 = rng.uniform(0, 2 * np.pi)
        for th in th0 + np.arange(0, 2 * np.pi - SPACING / R, SPACING / R):
            if len(pos) >= n_max:
                break
            n_try += 1
            u = np.cos(th) * e1 + np.sin(th) * e2; t = np.cross(ax, u)
            Rm = np.c_[ax, t, u]                      # library x -> crista normal, y -> rim tangent, z -> outward
            if np.linalg.det(Rm) < 0:
                Rm[:, 0] *= -1
            centre = c + R * u
            X = B @ Rm.T + centre
            good = True; n_sol = n_in = 0
            for k, p in enumerate(X):
                mtx = c3.mem_sdfs(p[0], p[1], p[2], *a, out)
                if tm[k]:
                    if abs(out[c3.M_IM]) > TM_TOL:
                        good = False; break
                elif k % 3 == 0:                    # every 3rd soluble residue: compartment and membrane clearance
                    if c3.label_from(p[0], p[1], p[2], G, cell.pores, out, mtx) not in ok_lab:
                        good = False; break
                    n_sol += 1; n_in += c3.min_mem_dist(out, -1) < 0.6
            # interfacial residues of the membrane-domain subunits may dip into the bilayer (as in the flat packer)
            good = good and n_in <= MAX_IN_MEMBRANE * max(n_sol, 1)
            if not good:
                continue
            qq = Rotation.from_matrix(Rm).as_quat(); qq = np.array([qq[3], qq[0], qq[1], qq[2]])
            p3.quat_mat(qq, M); p3.transform(stc, M, centre, P)
            if p3.g_overlap(glo, H, *grid[:6], P, str_, p3.TOL):
                continue
            p3.g_insert(glo, H, *grid, P, str_)
            pos.append(centre); quat.append(qq)
    log(f"ATP synthase dimers: {len(pos)} on cristae rims ({len(pos) / max(n_try, 1):.0%} of {n_try} rim sites tried; "
        f"{2 * len(pos)} monomers of {2 * n_max})")
    return dict(atp_pos=np.array(pos, np.float32).reshape(-1, 3), atp_q=np.array(quat, np.float32).reshape(-1, 4), li_atp=li)
