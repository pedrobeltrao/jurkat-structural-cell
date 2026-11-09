"""Actin cables (inventory v3, item D): bundles of F-actin running beneath the cell cortex.

Filament geometry from PDB 9GO5 (yeast ADP-F-actin, cryo-EM 2.5 A): each subunit is the previous one rotated
167.2 deg about the filament axis and translated 2.76 nm along it (genetic helix). The library type F_actin is
the 9GO5 subunit (filament conformation) in an axis-aligned frame. Filaments follow great-circle arcs CABLE_DEPTH
below the plasma membrane (beneath the cortical ER), FILAMENTS_PER_CABLE per bundle, ~CABLE_LENGTH long.
Subunits that would leave the cytosol or clash are skipped. Placed before polysomes and packing, like chromatin.

Assumptions: CABLE_FRACTION of ACT1 copies are in cables (the rest stays monomeric or in profilin/cofilin complexes;
actin patches are not modeled yet); cables are randomly oriented (unbudded cell, no polarity).
"""
import os, json
import numpy as np
import pack3d as p3
import cell3d as c3
import build_structs as bs
from config import LIB

LIB_ID = "F_actin"
CABLE_FRACTION = 0.3
CABLE_LENGTH = 1500.0      # nm
FILAMENTS_PER_CABLE = 8
CABLE_DEPTH = 90.0         # nm beneath the plasma membrane
FIL_SPACING = 12.0         # nm between neighbouring filaments in a bundle (fimbrin-crosslinked; leaves room for tropomyosin)
H = 20.0
PARAM_F = f"{LIB}/{LIB_ID}_helix.json"


def filament_model():
    """library entry F_actin (axis frame: filament axis = +z, subunit centroid at origin) + helix parameters"""
    if os.path.exists(PARAM_F) and os.path.exists(f"{LIB}/{LIB_ID}.npz"):
        return json.load(open(PARAM_F))
    import gemmi, extend_complexes as xc
    from scipy.spatial.transform import Rotation
    st = gemmi.read_structure(xc.fetch_assembly("9GO5"))
    ch = {}
    for c in st[0]:
        d = {r.seqid.num: np.array(r.find_atom("CA", "*").pos.tolist()) / 10 for r in c if r.find_atom("CA", "*")}
        if len(d) > 300:
            ch[c.name] = d
    names = list(ch); common = sorted(set.intersection(*[set(d) for d in ch.values()]))
    X = {n: np.array([ch[n][i] for i in common]) for n in names}
    best = None
    for a in names:                       # the screw between genetic-helix neighbours (rise ~2.76 nm)
        for b in names:
            if a == b: continue
            pa, pb = X[a] - X[a].mean(0), X[b] - X[b].mean(0)
            U, S, Vt = np.linalg.svd(pa.T @ pb); d = np.sign(np.linalg.det(Vt.T @ U.T))
            R = Vt.T @ np.diag([1, 1, d]) @ U.T; t = X[b].mean(0) - R @ X[a].mean(0)
            rv = Rotation.from_matrix(R).as_rotvec(); u = rv / np.linalg.norm(rv); ang = np.linalg.norm(rv)
            if abs(abs(t @ u) - 2.76) < 0.3:
                if t @ u < 0:                 # axis along the rise; the rotation angle changes sign with it
                    u, ang = -u, -ang
                best = (a, R, t, u, ang); break
        if best: break
    a, R, t, u, ang = best
    # point on the screw axis: (I - R) p = t - (t.u) u
    p0 = np.linalg.lstsq(np.eye(3) - R, t - (t @ u) * u, rcond=None)[0]
    c = X[a].mean(0); r = (c - p0) - ((c - p0) @ u) * u; r0 = np.linalg.norm(r)
    ex = r / r0; ey = np.cross(u, ex); F = np.c_[ex, ey, u]          # axis frame (columns)
    xyz, kind, _ = __import__("structlib").beads_from_structure(st, chains=[a])
    local = (xyz - c) @ F                                             # centred on the subunit, axis-aligned
    bs.save(LIB_ID, local, kind, extra=dict(pdb="9GO5", src="PDB"), per=30)
    prm = dict(phi_deg=float(np.degrees(ang)), rise=float(t @ u), r0=float(r0))
    json.dump(prm, open(PARAM_F, "w"))
    return prm


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def place(cell, lp, n_target, seed=23, obstacles=None, log=print):
    from scipy.spatial.transform import Rotation
    prm = filament_model()
    li = int(np.where(lp["ids"] == LIB_ID)[0][0])
    s0, s1 = lp["st_off"][li], lp["st_off"][li + 1]
    stc, str_, br = lp["STC"][s0:s1], lp["STR"][s0:s1], float(lp["BR"][li])
    G = cell.G; Rc = G[2] - CABLE_DEPTH
    glo = np.full(3, -G[0] - 10.0); ghi = -glo
    a = cell.args(glo, ghi, margin=50.0)
    cap = int(n_target * 1.5) * len(str_) + 1000 + (len(obstacles[1]) if obstacles is not None else 0)
    grid = p3.grid_new(glo, ghi, H, cap, cap * 14)
    if obstacles is not None and len(obstacles[1]):
        k = np.all((obstacles[0] > glo) & (obstacles[0] < ghi), axis=1)
        p3.g_insert(glo, H, *grid, obstacles[0][k], obstacles[1][k])
    rng = np.random.default_rng(seed)
    out = np.zeros(c3.NMEM); P = np.empty((len(stc), 3)); M = np.empty((3, 3))
    pos, rot, fil, kk = [], [], [], []
    # filament offsets in the bundle cross-section: the FILAMENTS_PER_CABLE hexagonal-lattice points nearest the centre,
    # in (radial, lateral) coordinates
    hexp = [(FIL_SPACING * (i + 0.5 * j), FIL_SPACING * np.sqrt(3) / 2 * j) for i in range(-3, 4) for j in range(-3, 4)]
    offs = sorted(hexp, key=lambda o: (round(np.hypot(*o), 6), np.arctan2(o[1], o[0])))[:FILAMENTS_PER_CABLE]
    nf = 0; attempts = 0
    while len(pos) < n_target and attempts < 10 * n_target:
        n = rng.normal(size=3); n /= np.linalg.norm(n)                # great-circle plane normal
        e1 = np.cross(n, rng.normal(size=3)); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
        th0 = rng.uniform(0, 2 * np.pi)
        for dr, dl in offs:
            R_ = Rc + dr; nsub = int(CABLE_LENGTH / prm["rise"]); spin = rng.uniform(0, 2 * np.pi)
            shaped = c3.n_fields(G) > c3.F_CELL
            mine = []        # subunits of this filament touch each other: insert them only once the filament is done
            for k in range(nsub):
                attempts += 1
                s = k * prm["rise"]; th = th0 + s / R_
                radial = np.cos(th) * e1 + np.sin(th) * e2
                tang = -np.sin(th) * e1 + np.cos(th) * e2
                if shaped:    # follow the real outline at the same depth below the plasma membrane
                    R_ = c3.ray_depth(G, radial, G[0] - G[2] + CABLE_DEPTH - dr)
                F = np.c_[radial, np.cross(tang, radial), tang]            # x radial, z along the cable
                Rk = F @ _rz(spin + np.radians(prm["phi_deg"]) * k)
                centre = R_ * radial + dl * n + Rk @ np.array([prm["r0"], 0.0, 0.0])
                Rq = Rotation.from_matrix(Rk).as_quat()
                q = np.array([Rq[3], Rq[0], Rq[1], Rq[2]])
                p3.quat_mat(q, M); p3.transform(stc, M, centre, P)
                if p3.g_overlap(glo, H, *grid[:6], P, str_, p3.TOL):
                    continue
                if not p3.check_volume(P, str_, centre[0], centre[1], centre[2], br, 1 << c3.CYTO, *a, out):
                    continue
                mine.append(P.copy())
                pos.append(centre); rot.append(q); fil.append(nf); kk.append(k)
                if len(pos) >= n_target:
                    break
            for Pk in mine:
                p3.g_insert(glo, H, *grid, Pk, str_)
            nf += 1
            if len(pos) >= n_target:
                break
    log(f"actin cables: {nf // FILAMENTS_PER_CABLE + 1} cables, {nf} filaments, {len(pos)} subunits "
        f"({len(pos) * prm['rise'] / 1000:.0f} um of filament); helix {prm}")
    return dict(act_pos=np.array(pos, np.float32), act_q=np.array(rot, np.float32), act_fil=np.array(fil, np.int32),
                act_k=np.array(kk, np.int32), li_actin=li)


# ------------------------------------------------------------------ tropomyosin on cables
TPM_ID = "TPM1_coil"
TPM_ING = "TPM1"
TPM_STEP = 10              # filament subunits per tropomyosin dimer (5 along its long-pitch strand)
TPM_PARAM_F = f"{LIB}/{TPM_ID}.json"


def _ca(st, ch):
    return {r.seqid.num: np.array(r.find_atom("CA", "*").pos.tolist()) / 10 for r in st[0][ch] if r.find_atom("CA", "*")}


def _kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def tropomyosin_model(log=print):
    """library entry TPM1_coil: a TPM1 coiled-coil dimer (2 x 200 residues) along the tropomyosin path of the F-actin-
    tropomyosin cryo-EM structure 5JLF (mammalian; its deposited coiled coil spans 135 residues), extended with the
    filament's own symmetry (two subunits along a long-pitch strand: 40 residues), the copies blended into one smooth
    coiled-coil axis; the two chains are wound around that axis as an ideal coiled coil.
    Frame: the library actin subunit at which the dimer starts."""
    if os.path.exists(TPM_PARAM_F) and os.path.exists(f"{LIB}/{TPM_ID}.npz"):
        return json.load(open(TPM_PARAM_F))
    import gemmi, extend_complexes as xc
    from scipy.interpolate import CubicSpline
    prm = filament_model()
    B = np.load(f"{LIB}/{LIB_ID}.npz")["beads"].astype(np.float64)
    s9 = gemmi.read_structure(xc.fetch_assembly("9GO5"))
    ch9 = next(c.name for c in s9[0] if sum(1 for r in c if r.find_atom("CA", "*")) == len(B))
    L = dict(zip([r.seqid.num for r in s9[0][ch9] if r.find_atom("CA", "*")], B))
    st = gemmi.read_structure(xc.fetch_assembly("5JLF"))
    W = _ca(st, "A"); k = sorted(set(L) & set(W))
    R, t = _kabsch(np.array([L[i] for i in k]), np.array([W[i] for i in k]))      # library actin -> chain A
    trace = {c: (np.array([a.pos.tolist() for r in st[0][c] for a in [r.find_atom("CA", "*")] if a]) / 10 - t) @ R for c in "FG"}
    # S^2 (next subunit on the same long-pitch strand) in the actin frame; axis at (-r0, 0, z)
    phi, rise, r0 = np.radians(prm["phi_deg"]), prm["rise"], prm["r0"]
    Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    S2R = Rz(2 * phi); S2t = S2R @ np.array([r0, 0, 0]) - np.array([r0, 0, 0]) + np.array([0, 0, 2 * rise])
    spl = {c: CubicSpline(np.arange(len(x)), x, axis=0) for c, x in trace.items()}
    n = len(trace["F"])
    # residue shift D with trace(i + D) = S2(trace(i))
    best = None
    for D in np.r_[np.arange(-45, -30, 0.02), np.arange(30, 45, 0.02)]:
        i = np.arange(max(0, -D), min(n - 1, n - 1 - D), 1.0)
        if len(i) < 40:
            continue
        e = np.mean([np.linalg.norm(spl[c](i) @ S2R.T + S2t - spl[c](i + D), axis=1).mean() for c in "FG"])
        if best is None or e < best[0]:
            best = (e, D)
    err, D = best

    def pos(c, s):
        """position at residue index s: weighted blend of every symmetry copy of the trace that covers s (bell weights
        vanish at the trace ends, so the copies join smoothly despite the helix-phase mismatch)"""
        out = np.zeros((len(s), 3)); wsum = np.zeros(len(s))
        for k in range(-8, 9):
            u = s - k * D
            ok = (u > 2) & (u < n - 3)
            if not ok.any():
                continue
            p = spl[c](u[ok])
            for _ in range(abs(k)):
                p = (p @ S2R.T + S2t) if k > 0 else (p - S2t) @ S2R
            w = np.sin(np.pi * (u[ok] - 2) / (n - 5)) ** 2
            out[ok] += w[:, None] * p; wsum[ok] += w
        return out / wsum[:, None]
    nres = int(np.floor(5 * abs(D)))             # one dimer per 5 subunits of its strand, ends abutting
    s0 = 0.0 if D > 0 else float(n - 1)
    sgn = 1 if D > 0 else -1                     # walk toward the barbed end (+z): the direction S^2 moves
    # coiled-coil axis from the blended path; the two chains are wound around it as an ideal left-handed coiled coil
    # (Crick parameters: superhelix radius 0.49 nm, pitch 14 nm, 3.6 residues per alpha-helix turn, Ca radius 0.23 nm)
    sres = s0 + sgn * np.arange(nres)
    Mx = 0.5 * (pos("F", sres) + pos("G", sres))
    T = np.gradient(Mx, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
    N = np.zeros_like(T); n0 = np.cross(T[0], [0, 0, 1.0]); N[0] = n0 / np.linalg.norm(n0)
    for i in range(1, len(T)):                   # parallel transport
        v = N[i - 1] - (N[i - 1] @ T[i]) * T[i]; N[i] = v / np.linalg.norm(v)
    Bn = np.cross(T, N)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Mx, axis=0), axis=1))]
    X = []
    for c in (0, 1):
        th0 = -2 * np.pi * arc / 14.0 + c * np.pi
        u = np.cos(th0)[:, None] * N + np.sin(th0)[:, None] * Bn; v = np.cross(T, u)
        th1 = 2 * np.pi * np.arange(nres) / 3.6
        X.append(Mx + 0.49 * u + 0.23 * (np.cos(th1)[:, None] * u + np.sin(th1)[:, None] * v))
    X = np.concatenate(X)
    bs.save(TPM_ID, X, np.zeros(len(X), np.int8), extra=dict(pdb="5JLF", src="PDB", nchain=2), per=30)
    rad = np.linalg.norm(X[:, :2] + [r0, 0], axis=1)
    info = dict(residue_shift=float(D), symmetry_error_nm=float(err), residues_per_chain=nres,
                radius_nm=[float(rad.min()), float(rad.mean()), float(rad.max())], z_span_nm=float(np.ptp(X[:, 2])))
    json.dump(info, open(TPM_PARAM_F, "w"))
    log(f"tropomyosin model: {info}")
    return info


def decorate(cell, lp, cab, obstacles=None, log=print):
    """TPM1 dimers on both long-pitch strands of every cable filament, one per TPM_STEP contiguous subunits; dimers that
    would touch a neighbouring filament (residue level), an obstacle or leave the cytosol are left out."""
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    tropomyosin_model(log=log)
    li = int(np.where(lp["ids"] == TPM_ID)[0][0])
    s0, s1 = lp["st_off"][li], lp["st_off"][li + 1]
    stc, str_, br = lp["STC"][s0:s1], lp["STR"][s0:s1], float(lp["BR"][li])
    XT = np.load(f"{LIB}/{TPM_ID}.npz")["beads"].astype(np.float64)
    XA = np.load(f"{LIB}/{LIB_ID}.npz")["beads"].astype(np.float64)
    G = cell.G; glo = np.full(3, -G[0] - 10.0); ghi = -glo
    a = cell.args(glo, ghi, margin=50.0); out = np.zeros(c3.NMEM); M = np.empty((3, 3)); P = np.empty((len(stc), 3))
    grid = None
    if obstacles is not None and len(obstacles[1]):
        grid = p3.grid_new(glo, ghi, H, len(obstacles[1]) + 1000, (len(obstacles[1]) + 1000) * 14)
        k = np.all((obstacles[0] > glo) & (obstacles[0] < ghi), axis=1)
        p3.g_insert(glo, H, *grid, obstacles[0][k], obstacles[1][k])
    rot = lambda q: Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    fil, kk = cab["act_fil"], cab["act_k"]
    cable_of = fil // FILAMENTS_PER_CABLE
    pos, quat, n_try = [], [], 0
    for c in np.unique(cable_of):
        sel = np.where(cable_of == c)[0]
        beads = np.concatenate([XA @ rot(cab["act_q"][i]).T + cab["act_pos"][i] for i in sel])
        lab = np.repeat(fil[sel], len(XA)); tree = cKDTree(beads)
        idx = {(int(fil[i]), int(kk[i])): i for i in sel}
        for f in np.unique(fil[sel]):
            ks = sorted(k for (ff, k) in idx if ff == f)
            for k in ks:
                if k % TPM_STEP not in (0, 1) or not all((f, k + j) in idx for j in range(TPM_STEP)):
                    continue
                n_try += 1
                i = idx[(f, k)]; R = rot(cab["act_q"][i]); t = cab["act_pos"][i].astype(np.float64)
                X = XT @ R.T + t
                d, j = tree.query(X, distance_upper_bound=0.35)
                hit = np.isfinite(d)
                if hit.any() and (lab[j[hit]] != f).any():
                    continue
                q = cab["act_q"][i].astype(np.float64)
                p3.quat_mat(q, M); p3.transform(stc, M, t, P)
                if grid is not None and p3.g_overlap(glo, H, *grid[:6], P, str_, p3.TOL):
                    continue
                if not p3.check_volume(P, str_, t[0], t[1], t[2], br, 1 << c3.CYTO, *a, out):
                    continue
                pos.append(t); quat.append(q)
    log(f"tropomyosin: {len(pos)} TPM1 dimers on cables ({len(pos) / max(n_try, 1):.0%} of {n_try} sites; "
        f"{2 * len(pos)} TPM1 chains)")
    return dict(tpm_pos=np.array(pos, np.float32).reshape(-1, 3), tpm_q=np.array(quat, np.float32).reshape(-1, 4), li_tpm=li)
