"""Coarse whole-cell model: equatorial slab (+/-15 nm) of a haploid yeast cell, one sphere per macromolecule.
Organelle volumes are data-driven (geo dict); their arrangement in the section is illustrative.
Plane coordinates (u, v) in nm, origin at cell centre. Label map at 1 nm resolution."""
import json
import numpy as np
import numba as nb
from scipy.ndimage import distance_transform_edt

RES = 1.0
SLAB = 30.0
# compartment labels
EXT, MANNAN, GLUCAN, CYTO, CER, PERI, NUCP, NUCL, VAC, IMS, MATRIX, LD, GOLGI, PEROX, PORE = range(15)
LABEL_NAMES = ["exterior", "wall_mannan", "wall_glucan", "cytosol", "cER_lumen", "perinuclear", "nucleoplasm",
               "nucleolus", "vacuole", "mito_IMS", "mito_matrix", "lipid_droplet", "golgi", "peroxisome", "pore"]
DA_PER_NM3 = 813.0  # protein density 1.35 g/cm3


def radius_from_mass(m, f=1.2):
    return f * (3 * m / DA_PER_NM3 / (4 * np.pi)) ** (1 / 3)


def build_layout(geo, rng):
    Rcell = geo.get("cell_radius_um", 2.156) * 1000
    wall = geo["wall_thickness_um"] * 1000
    Rp = Rcell - wall
    half = int(np.ceil(Rcell + 60))
    n = 2 * half
    ax = (np.arange(n) - half + 0.5) * RES
    U, V = np.meshgrid(ax, ax, indexing="xy")   # U horizontal, V vertical
    rr = np.hypot(U, V)
    lab = np.full((n, n), EXT, np.int8)
    lab[rr < Rcell] = MANNAN
    lab[rr < Rcell - 35] = GLUCAN
    lab[rr < Rp] = CYTO
    shapes = {}
    # cortical ER: sheets 23 nm below PM, 24 nm thick (lumen ~16 nm between 4-nm membranes), covering 48% of PM
    cover = geo["cer_cover"]
    th = np.arctan2(V, U)
    n_arc = 7
    starts = np.sort(rng.uniform(0, 2 * np.pi, n_arc))
    seg = 2 * np.pi * cover / n_arc
    in_arc = np.zeros_like(rr, bool)
    for s in starts:
        d = (th - s) % (2 * np.pi)
        in_arc |= d < seg
    cer = in_arc & (rr < Rp - 23 - 4) & (rr > Rp - 23 - 24 + 4)
    lab[cer] = CER
    # nucleus and vacuole (equatorial sections of spheres with the data-driven volumes)
    rn = geo["nucleus_radius_um"] * 1000
    rv = (geo["volumes_um3"]["vacuole"] * 3 / (4 * np.pi)) ** (1 / 3) * 1000
    cn = np.array([-850.0, 430.0]); cv = np.array([830.0, -420.0])
    dn = np.hypot(U - cn[0], V - cn[1]); dv = np.hypot(U - cv[0], V - cv[1])
    lab[dn < rn] = PERI
    lab[dn < rn - 34] = NUCP
    # nucleolus: crescent against the NE on the side away from the vacuole (~20% of the nuclear section area)
    dirn = (cn - cv) / np.linalg.norm(cn - cv)
    cno = cn + dirn * rn * 0.82
    lab[(np.hypot(U - cno[0], V - cno[1]) < rn * 0.62) & (dn < rn - 34 - 8)] = NUCL
    lab[dv < rv] = VAC
    shapes.update(nucleus=(cn, rn), vacuole=(cv, rv), nucleolus_c=cno)
    # mitochondria: 1 in-plane tubule (capsule) + 1 transverse section, radius 150 nm (2% volume)
    def capsule(p0, p1, r):
        p0, p1 = np.asarray(p0), np.asarray(p1)
        d = p1 - p0
        if d @ d == 0:
            return np.hypot(U - p0[0], V - p0[1])
        t = np.clip(((U - p0[0]) * d[0] + (V - p0[1]) * d[1]) / (d @ d), 0, 1)
        return np.hypot(U - (p0[0] + t * d[0]), V - (p0[1] + t * d[1]))
    mito_d = [capsule((-200, -1500), (380, -1250), 150), capsule((-1250, -820), (-1250, -820), 150)]
    for md in mito_d:
        lab[md < 150] = IMS
        lab[md < 150 - 10] = MATRIX
    # cristae: lamellar invaginations of the inner membrane (IMS-labelled, 12 nm wide) in the tubule
    p0, p1 = np.array([-200, -1500.0]), np.array([380, -1250.0])
    ax_ = (p1 - p0) / np.linalg.norm(p1 - p0); nrm = np.array([-ax_[1], ax_[0]])
    for t in np.linspace(0.12, 0.88, 6):
        c = p0 + t * (p1 - p0)
        along = (U - c[0]) * ax_[0] + (V - c[1]) * ax_[1]
        across = (U - c[0]) * nrm[0] + (V - c[1]) * nrm[1]
        side = 1 if t * 10 % 2 < 1 else -1
        cr = (np.abs(along) < 6) & (side * across > -60) & (mito_d[0] < 150 - 10)
        lab[cr] = IMS
    # lipid droplet (1% volume, r 215 nm; expected ~1 in the section), next to the nucleus/ER
    cld = np.array([-300.0, -900.0])
    lab[np.hypot(U - cld[0], V - cld[1]) < 215] = LD
    # Golgi/endosome (1%): three flattened cisternae + vesicles
    for (cx, cy, ang) in [(380, 950, 0.3), (520, 1050, 0.45), (1250, 300, 1.2)]:
        ca, sa = np.cos(ang), np.sin(ang)
        a_ = ((U - cx) * ca + (V - cy) * sa) / 230; b_ = (-(U - cx) * sa + (V - cy) * ca) / 30
        lab[(a_ ** 2 + b_ ** 2 < 1) & (lab == CYTO)] = GOLGI
    k = 0
    while k < 16:
        c = rng.uniform(-1800, 1800, 2)
        m = np.hypot(U - c[0], V - c[1]) < 25
        if np.all(lab[m] == CYTO) and np.min(np.hypot(*(c[:, None] - np.array([[400, 1300], [1000, 1250]]).T))) < 500:
            lab[m] = GOLGI; k += 1
    # peroxisome (0.2%): one profile r 90 nm
    cp = np.array([1500.0, 700.0])
    lab[(np.hypot(U - cp[0], V - cp[1]) < 90)] = PEROX
    # nuclear pores: expected NPCs intersecting the section = density x strip width x circumference
    dens = geo["n_npc"] / geo["areas_um2"]["nuclear_envelope"] * 2 / 1e6  # per nm2 of nuclear surface
    n_npc = int(round(dens * (SLAB + 100) * 2 * np.pi * rn))
    angs = []
    while len(angs) < n_npc:   # minimum centre spacing 160 nm (NPC outer diameter ~120 nm)
        a = rng.uniform(0, 2 * np.pi)
        if all(abs((a - b + np.pi) % (2 * np.pi) - np.pi) * rn > 160 for b in angs):
            angs.append(a)
    npc_pos = []
    for a in angs:
        p = cn + rn * np.array([np.cos(a), np.sin(a)]) - 17 * np.array([np.cos(a), np.sin(a)])
        m = (np.hypot(U - p[0], V - p[1]) < 45) & ((lab == PERI) | (dn < rn + 4) & (dn > rn - 38))
        lab[m] = PORE
        npc_pos.append((p, a))
    shapes.update(npc=npc_pos, mito=mito_d, cld=cld, cp=cp, Rp=Rp, Rcell=Rcell)
    return lab, ax, shapes


MEMBRANE_PAIRS = {  # name: label pairs whose boundary is a bilayer
    "PM": {(GLUCAN, CYTO)}, "cER": {(CYTO, CER)}, "ONM": {(CYTO, PERI)}, "INM": {(PERI, NUCP), (PERI, NUCL)},
    "tonoplast": {(CYTO, VAC)}, "MOM": {(CYTO, IMS)}, "MIM": {(IMS, MATRIX)}, "LD": {(CYTO, LD)},
    "golgi": {(CYTO, GOLGI)}, "perox": {(CYTO, PEROX)},
}


def membrane_pixels(lab):
    """returns dict name -> (rows, cols) of boundary pixels (on the first label's side)"""
    out = {k: [] for k in MEMBRANE_PAIRS}
    pairs = {}
    for k, ps in MEMBRANE_PAIRS.items():
        for a, b in ps:
            pairs[(a, b)] = k; pairs[(b, a)] = k
    for sh in [(0, 1), (1, 0)]:
        A = lab[:-sh[0] or None, :-sh[1] or None]; B = lab[sh[0]:, sh[1]:]
        diff = A != B
        ii, jj = np.nonzero(diff)
        for i, j in zip(ii, jj):
            k = pairs.get((A[i, j], B[i, j]))
            if k:
                out[k].append((i, j))
    return {k: np.array(v, np.int64).reshape(-1, 2) for k, v in out.items()}


def edt_by_label(lab):
    d = np.zeros(lab.shape, np.float32)
    for l in np.unique(lab):
        m = lab == l
        d[m] = distance_transform_edt(m)[m] * RES
    return d


# ---------------------------------------------------------------- packing
@nb.njit(cache=True)
def _pack(order_r, order_comp, order_memb, comp_pix_off, comp_pix, memb_pix_off, memb_pix, edt, half, h,
          nx, max_tries, seed, out_xyz, out_ok, zhalf, rmax):
    np.random.seed(seed)
    ncell = nx * nx
    head = -np.ones(ncell, np.int64)
    nxt = -np.ones(order_r.shape[0], np.int64)
    for o in range(order_r.shape[0]):
        r = order_r[o]
        c = order_comp[o]; mb = order_memb[o]
        for t in range(max_tries):
            if mb >= 0:
                a, b = memb_pix_off[mb], memb_pix_off[mb + 1]
                if b <= a:
                    break
                k = np.random.randint(a, b)
                i, j = memb_pix[k, 0], memb_pix[k, 1]
            else:
                a, b = comp_pix_off[c], comp_pix_off[c + 1]
                if b <= a:
                    break
                k = np.random.randint(a, b)
                i, j = comp_pix[k, 0], comp_pix[k, 1]
                if edt[i, j] < r + 2.5:
                    continue
            x = j + np.random.random() - half; y = i + np.random.random() - half
            z = (np.random.random() * 2 - 1) * zhalf
            ci = int((x + half) / h); cj = int((y + half) / h)
            reach = int((r + rmax) / h) + 1
            bad = False
            for di in range(-reach, reach + 1):
                if bad:
                    break
                for dj in range(-reach, reach + 1):
                    ii = ci + di; jj = cj + dj
                    if ii < 0 or jj < 0 or ii >= nx or jj >= nx:
                        continue
                    e = head[ii * nx + jj]
                    while e >= 0:
                        dx = x - out_xyz[e, 0]; dy = y - out_xyz[e, 1]; dz = z - out_xyz[e, 2]
                        rr = r + order_r[e]
                        if dx * dx + dy * dy + dz * dz < rr * rr:
                            bad = True
                            break
                        e = nxt[e]
                    if bad:
                        break
            if bad:
                continue
            out_xyz[o, 0] = x; out_xyz[o, 1] = y; out_xyz[o, 2] = z; out_ok[o] = True
            cc = ci * nx + cj
            nxt[o] = head[cc]; head[cc] = o
            break


def pack(objs, lab, mpix, edt, seed=1, max_tries=60):
    """objs: DataFrame with columns r, comp(label int or -1), memb(name or None). Sorted large->small inside."""
    n = lab.shape[0]; half = n / 2 * RES
    comps = sorted(set(int(c) for c in objs.comp if c >= 0))
    comp_pix_list, offs = [], [0]
    comp_index = {}
    for l in range(15):
        ii, jj = np.nonzero(lab == l)
        comp_pix_list.append(np.c_[ii, jj]); offs.append(offs[-1] + len(ii)); comp_index[l] = l
    comp_pix = np.vstack(comp_pix_list).astype(np.int64); comp_off = np.array(offs, np.int64)
    mnames = list(MEMBRANE_PAIRS)
    mp = [mpix[k] for k in mnames]; moff = np.cumsum([0] + [len(x) for x in mp]).astype(np.int64)
    memb_pix = np.vstack(mp).astype(np.int64)
    o = objs.sort_values("r", ascending=False).reset_index(drop=True)
    rr = o.r.values.astype(np.float64)
    cc = o.comp.values.astype(np.int64)
    mm = np.array([mnames.index(m) if isinstance(m, str) else -1 for m in o.memb], np.int64)
    xyz = np.zeros((len(o), 3)); ok = np.zeros(len(o), np.bool_)
    h = 16.0; nx = int(np.ceil(n * RES / h)) + 1
    _pack(rr, cc, mm, comp_off, comp_pix, moff, memb_pix, edt, half, h, nx, max_tries, seed, xyz, ok, SLAB / 2, rr.max())
    o["u"], o["v"], o["z"], o["ok"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], ok
    return o


# ---------------------------------------------------------------- objects, colours, background
COMP2LAB = {"cytosol": CYTO, "nucleoplasm": NUCP, "nucleolus": NUCL, "mitochondrion": MATRIX, "ER": CER,
            "golgi_endosome": GOLGI, "vacuole": VAC, "cell_wall": MANNAN, "peroxisome": PEROX,
            "plasma_membrane": CYTO, "nuclear_envelope": CYTO, "lipid_droplet": CYTO}
COMP2MEMB = {"plasma_membrane": "PM", "ER": "cER", "vacuole": "tonoplast", "mitochondrion": "MIM",
             "golgi_endosome": "golgi", "peroxisome": "perox", "lipid_droplet": "LD", "nuclear_envelope": "NE"}


def make_objects(ing_pack, oligo_map, geo, lab, mpix, n_trna, rng):
    area = np.bincount(lab.ravel(), minlength=15) * RES ** 2
    slab_vol = {l: area[l] * SLAB for l in range(15)}
    mlen = {k: len(v) * np.pi / 4 * RES for k, v in mpix.items()}
    V = geo["volumes_um3"]; A = geo["areas_um2"]
    vol_of = {"cytosol": V["cytosol"], "nucleoplasm": V["nucleoplasm"], "nucleolus": V["nucleolus"],
              "mitochondrion": V["mitochondrion"], "ER": V["ER"], "golgi_endosome": V["golgi_endosome"],
              "vacuole": V["vacuole"], "cell_wall": V["cell_wall"], "peroxisome": V["peroxisome"]}
    lab_vol = {"cytosol": slab_vol[CYTO], "nucleoplasm": slab_vol[NUCP], "nucleolus": slab_vol[NUCL],
               "mitochondrion": slab_vol[MATRIX] + slab_vol[IMS], "ER": slab_vol[CER], "golgi_endosome": slab_vol[GOLGI],
               "vacuole": slab_vol[VAC], "cell_wall": slab_vol[MANNAN] + slab_vol[GLUCAN], "peroxisome": slab_vol[PEROX]}
    memb_frac = {"PM": mlen["PM"] * SLAB / (A["plasma_membrane"] * 1e6), "cER": mlen["cER"] * SLAB / (A["ER"] * 1e6),
                 "NE": (mlen["ONM"] + mlen["INM"]) * SLAB / (A["nuclear_envelope"] * 1e6),
                 "tonoplast": mlen["tonoplast"] * SLAB / (A["vacuole"] * 1e6),
                 "MIM": lab_vol["mitochondrion"] / (V["mitochondrion"] * 1e9),
                 "golgi": lab_vol["golgi_endosome"] / (V["golgi_endosome"] * 1e9),
                 "perox": lab_vol["peroxisome"] / (V["peroxisome"] * 1e9),
                 "LD": slab_vol[LD] / (V["lipid_droplet"] * 1e9)}
    rows = []
    special_r = {"ribosome_80S": 12.5, "nucleosome": 5.0}
    for r in ing_pack.itertuples():
        if r.ing_id == "NPC":
            continue
        w = json.loads(r.comp_weights)
        tm = (float(r.n_tm) > 0 if r.n_tm == r.n_tm else False) or (r.membrane in ("PM", "NE", "VAC"))
        lid, n_ol = (oligo_map[r.ing_id] if r.ing_id in oligo_map else (r.ing_id, 1))
        rad = special_r.get(r.ing_id, radius_from_mass(r.mass_da * n_ol))
        for c, f in w.items():
            cp = r.copies_per_cell * f / n_ol
            if cp <= 0:
                continue
            if tm and c in COMP2MEMB:
                m = COMP2MEMB[c]
                lam = cp * memb_frac[m]
                if m == "NE":
                    k = rng.poisson(lam); a = rng.binomial(k, 0.5)
                    rows += [(lid, rad, -1, "ONM", c)] * a + [(lid, rad, -1, "INM", c)] * (k - a)
                    continue
                rows += [(lid, rad, -1, m, c)] * rng.poisson(lam)
            elif c in vol_of:
                lam = cp * lab_vol[c] / (vol_of[c] * 1e9)
                rows += [(lid, rad, COMP2LAB[c], None, c)] * rng.poisson(lam)
            else:  # peripheral proteins of PM/NE/LD: cytosol
                lam = cp * slab_vol[CYTO] / (V["cytosol"] * 1e9) * 0  # counted via their cytosol share only
    rows += [("tRNA", radius_from_mass(25000), CYTO, None, "cytosol")] * rng.poisson(n_trna * slab_vol[CYTO] / (V["cytosol"] * 1e9))
    import pandas as pd
    return pd.DataFrame(rows, columns=["lib_id", "r", "comp", "memb", "src_comp"]), slab_vol, mlen, memb_frac


WPAL = {"cytosol": "#7EA7D3", "ribosome": "#D9679C", "tRNA": "#E89BBE", "nucleoplasm": "#B993C9", "nucleolus": "#9461AC",
        "nucleosome": "#F0A14E", "mito": "#D9896A", "ER": "#93C383", "golgi": "#9DBF7E", "vacuole": "#79B4B0",
        "perox": "#C9A15A", "wall": "#9A8352", "PM": "#4FA370", "memb": "#5FAE75", "big": "#8B88C9", "NPC": "#3E9C9A"}
BG = {EXT: "#F6F1E7", MANNAN: "#D9D3A0", GLUCAN: "#E6D2A8", CYTO: "#E4ECF5", CER: "#E3EED9", PERI: "#E3EED9",
      NUCP: "#F1E7F1", NUCL: "#E6D6EA", VAC: "#E3EEEE", IMS: "#F2E9DF", MATRIX: "#F5E1D1", LD: "#F7DC98",
      GOLGI: "#E6EED6", PEROX: "#EFE5CC", PORE: "#E4ECF5"}


def obj_color(o, rng):
    from render import hex2rgb
    if o.lib_id == "ribosome_80S": k = "ribosome"
    elif o.lib_id == "tRNA": k = "tRNA"
    elif o.lib_id == "nucleosome": k = "nucleosome"
    elif isinstance(o.memb, str): k = "PM" if o.memb == "PM" else ("mito" if o.memb == "MIM" else "memb")
    elif o.comp == NUCP: k = "nucleoplasm"
    elif o.comp == NUCL: k = "nucleolus"
    elif o.comp == MATRIX: k = "mito"
    elif o.comp == CER: k = "ER"
    elif o.comp == GOLGI: k = "golgi"
    elif o.comp == VAC: k = "vacuole"
    elif o.comp == PEROX: k = "perox"
    elif o.comp == MANNAN: k = "wall"
    elif o.lib_id in ("proteasome_26S", "FAS", "CCT"): k = "big"
    else: k = "cytosol"
    return np.clip(hex2rgb(WPAL[k]) * (1 + rng.uniform(-0.08, 0.08)), 0, 1)


def background_image(lab, mpix, memb_rgb="#F1C232", edge_rgb="#7A6420", width=2):
    from render import hex2rgb
    from scipy.ndimage import binary_dilation
    img = np.zeros(lab.shape + (3,))
    for l, h in BG.items():
        img[lab == l] = hex2rgb(h)
    # glucan texture: faint fibrillar strokes
    m = np.zeros(lab.shape, bool)
    for k, p in mpix.items():
        m[p[:, 0], p[:, 1]] = True
    band = binary_dilation(m, iterations=width)
    edge = binary_dilation(band, iterations=1) & ~band
    # glucan texture: tangential fibrils as faint darker strokes
    rng = np.random.default_rng(4)
    n = lab.shape[0]; half = n / 2
    gl = np.argwhere(lab == GLUCAN)
    seeds = gl[rng.integers(0, len(gl), 2500)]
    for i0, j0 in seeds:
        y, x = i0 - half, j0 - half; ang = np.arctan2(y, x) + np.pi / 2 + rng.normal(0, 0.35)
        L_ = rng.uniform(40, 160); t = np.linspace(-L_ / 2, L_ / 2, int(L_ * 2))
        ii = np.clip((i0 + t * np.sin(ang)).astype(int), 0, n - 1); jj = np.clip((j0 + t * np.cos(ang)).astype(int), 0, n - 1)
        ok = lab[ii, jj] == GLUCAN
        img[ii[ok], jj[ok]] = img[ii[ok], jj[ok]] * 0.6 + hex2rgb("#B08B4F") * 0.4
    img[band] = hex2rgb(memb_rgb)
    img[edge] = img[edge] * 0.35 + hex2rgb(edge_rgb) * 0.65
    return img
