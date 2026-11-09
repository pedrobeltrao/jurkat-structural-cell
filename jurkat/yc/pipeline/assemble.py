"""Expand packed placements to residue-level beads and add lipids, cell-wall glucan and mannan, linker DNA."""
import numpy as np
from scipy.spatial import cKDTree
from column import LAY, L, HALF, NPC_HOLE, LIPID_AREA, LIPIDS

# bead classes
B_PROT, B_NUC = 0, 1


def expand(pk, lib):
    """Return dict of arrays: pos (N,3) float32, inst (N,) int32, kind (N,) int8; plus instance table."""
    P, I, K = [], [], []
    for k, (lid, Rm, t) in enumerate(pk.placed):
        b = lib[lid]["beads"] @ Rm.T.astype(np.float32) + t.astype(np.float32)
        b[:, 0] %= L; b[:, 1] %= L
        P.append(b.astype(np.float32)); I.append(np.full(len(b), k, np.int32)); K.append(lib[lid]["kind"])
    return np.vstack(P), np.concatenate(I), np.concatenate(K)


# ---------------------------------------------------------------- lipids
LIPID_GEOM = {  # (offsets from midplane along the leaflet normal, radii)
    "lipid": ([1.95, 1.25, 0.55], [0.55, 0.45, 0.45]),
    "ergosterol": ([1.55, 0.75], [0.45, 0.45]),
}


def _flat_sites(x0, x1, rng, spacing=np.sqrt(LIPID_AREA)):
    xs = np.arange(x0 + spacing / 2, x1, spacing); ys = np.arange(spacing / 2, L, spacing)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    p = np.c_[X.ravel(), Y.ravel()] + rng.uniform(-0.18, 0.18, (X.size, 2))
    return p


def _torus_sites(rng, Rc, a, zc, area_per):
    """inner half of a torus (pore membrane joining ONM and INM); returns points and outward normals."""
    area = 2 * np.pi * a * (np.pi * Rc - 2 * a)  # inner half-torus area
    n = int(area / area_per)
    out = []
    while sum(len(o) for o in out) < n:
        th = rng.uniform(np.pi / 2, 3 * np.pi / 2, 4 * n); ph = rng.uniform(0, 2 * np.pi, 4 * n)
        w = (Rc + a * np.cos(th)) / (Rc + a)          # area-weighted rejection
        keep = rng.random(4 * n) < w
        out.append(np.c_[th[keep], ph[keep]])
    tp = np.vstack(out)[:n]
    th, ph = tp[:, 0], tp[:, 1]
    rr = Rc + a * np.cos(th)
    pts = np.c_[L / 2 + rr * np.cos(ph), L / 2 + rr * np.sin(ph), zc + a * np.sin(th)]
    nrm = np.c_[np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)]  # points away from tube centre
    return pts, nrm


def lipids(rng, prot_tree):
    classes = list(LIPIDS); probs = np.array([LIPIDS[c] for c in classes]); probs /= probs.sum()
    pos, rad, cls, inst, leaf = [], [], [], [], []
    k = [0]; lf = [0]

    def emit(center, normal, sign, ctype):
        g = LIPID_GEOM["ergosterol" if classes[ctype] == "ergosterol" else "lipid"]
        for o, r in zip(*g):
            pos.append(center + sign * o * normal); rad.append(r); cls.append(ctype); inst.append(k[0]); leaf.append(lf[0])
        k[0] += 1

    membranes = [("PM", LAY["pm"], 0, L, None), ("ER_A", LAY["erA"], *LAY["er_x"], None),
                 ("ER_B", LAY["erB"], *LAY["er_x"], None),
                 ("ONM", LAY["onm"], 0, L, "hole"), ("INM", LAY["inm"], 0, L, "hole")]
    meta = []
    for name, z0, x0, x1, hole in membranes:
        for sign in (+1, -1):
            s = _flat_sites(x0, x1, rng)
            if hole:
                rr = np.hypot(s[:, 0] - L / 2, s[:, 1] - L / 2)
                s = s[rr > NPC_HOLE + (LAY["inm"] - LAY["onm"]) / 2]
            ct = rng.choice(len(classes), len(s), p=probs)
            for (x, y), c in zip(s, ct):
                emit(np.array([x, y, z0]), np.array([0, 0, 1.0]), sign, c)
            meta.append((name, sign, len(s))); lf[0] += 1
    # cortical-ER sheet rims (half cylinders joining the two sheet membranes at both edges)
    ar = (LAY["erB"] - LAY["erA"]) / 2; zc = (LAY["erA"] + LAY["erB"]) / 2
    for xe, side in ((LAY["er_x"][1], +1), (LAY["er_x"][0], -1)):
        for sign in (+1, -1):
            rr = ar + sign * 1.2
            n_th = int(np.pi * rr / np.sqrt(LIPID_AREA)); ys = np.arange(0.4, L, np.sqrt(LIPID_AREA))
            th = np.linspace(-np.pi / 2, np.pi / 2, n_th)
            for t in th:
                nrm = np.array([side * np.cos(t), 0, np.sin(t)])
                for y in ys:
                    c = np.array([xe, y + rng.uniform(-0.15, 0.15), zc]) + ar * nrm
                    c[0] %= L
                    emit(c, nrm, sign, rng.choice(len(classes), p=probs))
            meta.append(("ER_rim", sign, n_th * len(ys))); lf[0] += 1
    # pore membrane (half torus)
    a = (LAY["inm"] - LAY["onm"]) / 2; Rc = NPC_HOLE + a
    for sign in (+1, -1):
        pts, nrm = _torus_sites(rng, Rc, a, (LAY["inm"] + LAY["onm"]) / 2, LIPID_AREA * (1 if sign > 0 else 1))
        ct = rng.choice(len(classes), len(pts), p=probs)
        # inner-leaflet points sit on a smaller tube; shift by the normal: lipid centre on the midsurface
        for p, n_, c in zip(pts, nrm, ct):
            emit(p, n_, sign, c)
        meta.append(("pore", sign, len(pts))); lf[0] += 1
    pos = np.asarray(pos, np.float32); rad = np.asarray(rad, np.float32)
    cls = np.asarray(cls, np.int8); inst = np.asarray(inst, np.int32); leaf = np.asarray(leaf, np.int16)
    # remove lipids clashing with protein residue beads
    d, _ = prot_tree.query(pos, k=1, distance_upper_bound=0.95)
    bad_inst = np.unique(inst[np.isfinite(d)])
    keep = ~np.isin(inst, bad_inst)
    return dict(pos=pos[keep], rad=rad[keep], cls=cls[keep], inst=inst[keep], leaf=leaf[keep], classes=classes,
                n_lipids_total=k[0], n_removed=len(bad_inst), meta=meta)


# ---------------------------------------------------------------- wall polysaccharides
def wall_mass_budget(total_protein_da, geo, wall_frac_dry=0.20, protein_frac_dry=0.45):
    """Column share of wall polysaccharide mass. Assumptions (flagged in report): protein = 45% of cell dry mass,
    wall = 20% of dry mass (literature range 15-30%), wall composition from Klis et al. 2002 (FEMS Microbiol Rev): beta-1,3-glucan 52.5%,
    beta-1,6-glucan 7.5%, mannoprotein 37.5% (~85% of it carbohydrate), chitin 1.5%."""
    dry = total_protein_da / protein_frac_dry
    wall = dry * wall_frac_dry
    col_frac = (L * L * LAY["wall_in"]) / (geo["volumes_um3"]["cell_wall"] * 1e9)
    w = wall * col_frac
    return dict(dry_mass_pg=dry * 1.66054e-12, wall_mass_pg=wall * 1.66054e-12, column_fraction=col_frac,
                b13_glucose=w * 0.525 / 162.14, b16_glucose=w * 0.075 / 162.14,
                mannose=w * 0.375 * 0.85 / 162.14, chitin_glcnac=w * 0.015 / 203.2)


def glucan(rng, budget, occupied_tree, bead_r=0.75, step=1.2, res_per_bead=36):
    """beta-1,3-glucan as triple-helical fibrils (~1.5 nm diameter); 1 bead = 1.2 nm of triple helix
    (~0.1 nm rise per residue across 3 chains -> ~12 residues per chain per bead; 36 residues/bead).
    Semi-flexible, mostly tangential (x-y) chains in the inner wall layer, plus short beta-1,6 branches."""
    n_beads = int((budget["b13_glucose"] + budget["b16_glucose"]) / res_per_bead)
    zlo, zhi = LAY["mannan_end"], LAY["wall_in"] - 2.0
    out, fid = [], []
    f = 0
    while sum(len(o) for o in out) < n_beads:
        nseg = rng.integers(60, 250)
        p = np.array([rng.uniform(0, L), rng.uniform(0, L), rng.uniform(zlo, zhi)])
        th = rng.uniform(0, 2 * np.pi); d = np.array([np.cos(th), np.sin(th), rng.normal(0, 0.15)]); d /= np.linalg.norm(d)
        pts = []
        for _ in range(nseg):
            d = d + rng.normal(0, 0.12, 3) * np.array([1, 1, 0.5]); d /= np.linalg.norm(d)
            p = p + step * d
            if p[2] < zlo or p[2] > zhi:
                d[2] = -d[2]; p[2] = np.clip(p[2], zlo, zhi)
            pts.append(p.copy())
        pts = np.array(pts); pts[:, :2] %= L
        dd, _ = occupied_tree.query(pts, k=1, distance_upper_bound=bead_r + 0.6)
        pts = pts[~np.isfinite(dd)]
        out.append(pts); fid.append(np.full(len(pts), f, np.int32)); f += 1
    return np.vstack(out).astype(np.float32), np.concatenate(fid)


def mannan(rng, budget, wall_prot_pos, wall_prot_inst, occupied_tree, bead_r=0.6, res_per_bead=3):
    """N/O-linked mannan as branched random-walk brushes rooted on cell-wall protein surfaces, biased outward (-z).
    1 bead ~ 3 mannose residues."""
    n_beads = int(budget["mannose"] / res_per_bead)
    insts = np.unique(wall_prot_inst)
    per = max(1, n_beads // max(len(insts), 1))
    out, owner = [], []
    for i in insts:
        sp = wall_prot_pos[wall_prot_inst == i]
        n_ch = max(2, per // 25)
        roots = sp[rng.integers(0, len(sp), n_ch)]
        for r0 in roots:
            p = r0.copy(); d = np.array([0, 0, -1.0]) + rng.normal(0, 0.6, 3); d /= np.linalg.norm(d)
            for _ in range(rng.integers(12, 38)):
                d = d + rng.normal(0, 0.55, 3) + np.array([0, 0, -0.15]); d /= np.linalg.norm(d)
                p = p + 1.05 * d
                if p[2] < 0.5:
                    break
                out.append(p.copy()); owner.append(i)
    pts = np.array(out, np.float32); pts[:, :2] %= L; owner = np.array(owner, np.int32)
    dd, _ = occupied_tree.query(pts, k=1, distance_upper_bound=0.75)
    # beads touching their own protein are kept only if not deep inside (distance > 0.5)
    keep = ~np.isfinite(dd) | (dd > 0.5)
    return pts[keep], owner[keep], n_beads


def mrna_threads(rng, n_chains, prot_tree, zlo, zhi, nt_mean=1500, step=0.6, bead_r=0.5, extra_trees=()):
    """mRNA as flexible single-strand threads (1 bead per nucleotide, 0.6 nm rise) grown through free cytosol,
    avoiding protein beads. Real mRNPs are compacted by RNA-binding proteins; this is a schematic thread."""
    from column import L
    chains = []
    for c in range(n_chains):
        n = max(300, int(rng.exponential(nt_mean)))
        for attempt in range(50):
            p = np.array([rng.uniform(0, L), rng.uniform(0, L), rng.uniform(zlo, zhi)])
            if not np.isfinite(prot_tree.query(p, distance_upper_bound=1.5)[0]):
                break
        pts = [p]; d = rng.normal(size=3); d /= np.linalg.norm(d)
        stuck = 0
        while len(pts) < n and stuck < 200:
            nd = d + rng.normal(0, 0.9, 3); nd /= np.linalg.norm(nd)
            q = pts[-1] + step * nd
            q[:2] %= L
            if q[2] < zlo or q[2] > zhi or np.isfinite(prot_tree.query(q, distance_upper_bound=1.0)[0]):
                stuck += 1
                continue
            pts.append(q); d = nd; stuck = 0
        chains.append(np.array(pts))
    return chains
