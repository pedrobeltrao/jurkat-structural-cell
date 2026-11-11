"""Tier 2 of the microscopy-driven validation: image-level metrics on the HPA JURKAT fields and on simulated
confocal sections of the model, with the same segmentation and metric code on both sides.

HPA side (data/reference/hpa_subcellular/jurkat_images/, CC BY-SA 4.0, analysis only): 244 fields, 117 genes; channels
blue = nucleus (DAPI), red = microtubules, green = target. Pixel 80 nm (HPA 63x confocal).
Model side: the packed cell (wc/tiles + pre-pass, fluor.py channel weights) rendered as confocal sections (fluor.py PSF:
FWHM 250 nm lateral, 600 nm axial) at several heights through the nucleus, 80 nm pixels, Poisson noise; blue = H2B
(nucleosomes), red = beta-tubulin (microtubules + free), green = the gene.

Segmentation (seg_cells, identical for both):
  nuclei   blue, Gaussian 1 px, Otsu, holes filled, objects < 8 um2 dropped, touching nuclei split by a distance-transform
           watershed (peaks >= 2.4 um apart); nuclei 12-160 um2 kept.
  cells    (red + green, each scaled to its in-field 99.5th percentile), Gaussian 2 px, threshold = Otsu of the field,
           union nuclei, holes filled; every cell pixel goes to the nearest nucleus within 4 um (expand_labels).
           Cells touching the field border are dropped.
  bands    rim = outer 0.4 um of the cell mask (outside the nucleus); nucleus interior = nucleus eroded by 0.24 um;
           cytoplasm interior = cell minus rim minus the nucleus dilated by 0.24 um (needs >= 20 px).
  metrics  (green, background = median outside the dilated cell masks, subtracted)
           nc        mean nucleus interior / mean cytoplasm interior
           rim       mean rim / mean cytoplasm interior
           nuc_frac  share of the cell's green signal inside the nucleus mask (scale-free, no thin-band problem)
           punct     share of the cell's green signal in white-top-hat spots (> 4 MAD, >= 3 px; top-hat 1.2 um)
           Ratios use eps = 2% of the cell's 99th-percentile green in numerator and denominator; log2 clipped at +-6.
  per gene median over cells (HPA: all fields/antibodies of the gene except 'uncertain'-only antibodies; model: sections).

Limits (honest): Jurkat cells are small and round; the cytoplasm is a 1-2 um ring around a large nucleus, so the
cytoplasm interior is a few hundred pixels at best and is partly contaminated by nuclear and rim signal through the PSF;
rim and N/C are therefore compressed towards 1 on both sides. nuc_frac is the most robust metric. HPA fields are
fixed, permeabilised cells with antibody-dependent background; the model is one cell.

Steps (conda_envs/vcell/bin/python jurkat/validate/tier2.py <step> [--ver v5]):
  split    fixed random half of the HPA-JURKAT genes (seed 20261110) -> validation/jurkat_holdout_genes.txt (+ _fit)
  collect  read the current wc/tiles once for the 117 genes + H2B + TUBB -> validation/fluor/cache_tier2_<ver>/points.npz
  hpa      segment the HPA fields -> validation/cache/tier2_hpa_cells.tsv, tier2_hpa_genes.tsv
  model    render + segment the model sections -> validation/cache/tier2_model_<ver>_cells.tsv
  compare  -> validation/tier2_<ver>.tsv (per gene) + tier2_<ver>.json (overall, fit half, held-out half)
Long runs: setsid nohup ... > logs/tier2_<step>.log 2>&1 &"""
import os, sys, glob, json, time
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
V = (__import__("os").environ.get("VCELL_ROOT", ".") + ""); D = f"{V}/data/derived"; OUT = f"{D}/validation"
REF = f"{V}/data/reference"; IMG = f"{REF}/hpa_subcellular/jurkat_images"; CA = f"{OUT}/cache"
HOLD = f"{OUT}/jurkat_holdout_genes.txt"; FIT = f"{OUT}/jurkat_fit_genes.txt"
SEED = 20261110
PX = 80.0                     # nm per pixel (HPA and model)
VER = "v5"
for i, a in enumerate(sys.argv):
    if a == "--ver": VER = sys.argv[i + 1]
NPROC = int(os.environ.get("TIER2_NPROC", 16))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jurkat_genes():
    C = pd.read_csv(f"{OUT}/hpa_celllines.tsv", sep="\t", keep_default_na=False)
    return sorted(set(C[C.cell_line.str.upper() == "JURKAT"].gene))


# ----------------------------------------------------------------- hold-out split
def split():
    """fixed random half of the HPA-JURKAT genes (DECISION: held out from all fitting/calibration; seed SEED)"""
    if os.path.exists(HOLD):
        log("split exists, not regenerated:", HOLD); return
    g = jurkat_genes(); rng = np.random.default_rng(SEED); perm = rng.permutation(len(g))
    hold = sorted(g[i] for i in perm[: len(g) // 2]); fit = sorted(set(g) - set(hold))
    hdr = (f"# HPA-JURKAT hold-out half: {len(hold)} of {len(g)} genes (numpy default_rng({SEED}).permutation, first half).\n"
           "# Never used to fit or calibrate inventory rules (v6+). The other half is in jurkat_fit_genes.txt.\n")
    open(HOLD, "w").write(hdr + "\n".join(hold) + "\n")
    open(FIT, "w").write(hdr.replace("hold-out half", "fit half").replace("Never used", "May be used") + "\n".join(fit) + "\n")
    log(f"hold-out {len(hold)}, fit {len(fit)}")


def read_split():
    rd = lambda f: [l.strip() for l in open(f) if l.strip() and not l.startswith("#")]
    return set(rd(HOLD)), set(rd(FIT))


# ----------------------------------------------------------------- shared segmentation + metrics
def seg_cells(blue, red, green, px=PX, min_nuc_um2=12.0, max_nuc_um2=160.0, drop_border=True):
    """per-cell metric dicts (see module docstring). blue/red/green: 2D float arrays of the same field."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed, expand_labels
    from skimage.feature import peak_local_max
    from skimage.morphology import remove_small_objects, white_tophat, disk
    um = 1000.0 / px                                                     # pixels per um
    b = ndi.gaussian_filter(blue.astype(np.float32), 1.0)
    if b.max() <= 0: return []
    nuc = ndi.binary_fill_holes(b > threshold_otsu(b))
    nuc = remove_small_objects(nuc, int(8 * um * um))
    dt = ndi.distance_transform_edt(nuc)
    pk = peak_local_max(dt, min_distance=int(1.2 * um), labels=ndi.label(nuc)[0], exclude_border=False)
    mk = np.zeros(nuc.shape, np.int32); mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    nl = watershed(-dt, mk, mask=nuc)
    area = np.bincount(nl.ravel()) / um ** 2
    ok = (area >= min_nuc_um2) & (area <= max_nuc_um2); ok[0] = False
    nl = np.where(ok[nl], nl, 0)
    if nl.max() == 0: return []
    sc = lambda x: x.astype(np.float32) / max(np.percentile(x, 99.5), 1e-6)
    s = ndi.gaussian_filter(sc(red) + sc(green), 2.0)
    cm = ndi.binary_fill_holes((s > threshold_otsu(s)) | (nl > 0))
    terr = expand_labels(nl, distance=int(4 * um)); cl = np.where(cm, terr, 0)
    bgm = ~ndi.binary_dilation(cm, iterations=int(1 * um))
    g = green.astype(np.float32) - (np.median(green[bgm]) if bgm.any() else 0.0)
    th = white_tophat(ndi.gaussian_filter(green.astype(np.float32), 0.7), footprint=disk(int(0.6 * um)))
    out = []
    H_, W_ = g.shape
    objs = ndi.find_objects(cl)
    r_rim, r_n = 0.4 * um, 0.24 * um
    for lab, sl in enumerate(objs, start=1):
        if sl is None: continue
        if drop_border and (sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H_ or sl[1].stop == W_): continue
        p0 = [max(sl[0].start - 3, 0), max(sl[1].start - 3, 0)]
        ss = (slice(p0[0], min(sl[0].stop + 3, H_)), slice(p0[1], min(sl[1].stop + 3, W_)))
        cell = cl[ss] == lab; nucm = (nl[ss] == lab) & cell
        if nucm.sum() < 10 or cell.sum() < nucm.sum() + 20: continue
        depth = ndi.distance_transform_edt(cell)
        own = (terr[ss] == lab) & ~cell                                   # this cell's territory outside its mask
        outd = ndi.distance_transform_edt(~cell)
        nd_out = ndi.distance_transform_edt(~nucm); nd_in = ndi.distance_transform_edt(nucm)
        rim = cell & (depth <= r_rim) & ~nucm
        n_in = nucm & (nd_in > r_n)
        c_in = cell & (depth > r_rim) & (nd_out > r_n)
        gg = g[ss]; tot = gg[cell].sum()
        p99 = np.percentile(gg[cell], 99); eps = 0.02 * max(p99, 1e-6)
        r = dict(cell_area_um2=cell.sum() / um ** 2, nuc_area_um2=nucm.sum() / um ** 2, n_cyto_px=int(c_in.sum()),
                 n_rim_px=int(rim.sum()), nuc_frac=float(gg[nucm].sum() / tot) if tot > 0 else np.nan)
        if c_in.sum() >= 20 and n_in.sum() >= 20:
            cy = gg[c_in].mean()
            r["nc"] = float(np.clip(np.log2((gg[n_in].mean() + eps) / (cy + eps)), -6, 6))
            # rim = peak of the radial profile: 1-px shells from 0.24 um outside the mask edge to 0.4 um inside (nucleus
            # excluded); the peak tracks the membrane whatever the exact position of the thresholded edge
            sh = []
            for d in range(-int(round(r_n)), int(round(r_rim)) + 1):
                m_ = (cell & (np.abs(depth - d) <= 0.5) & ~nucm) if d > 0 else (own & (np.abs(outd + d) <= 0.5)) if d < 0 else \
                     (cell & (depth <= 1.0) & ~nucm)
                if m_.sum() >= 10: sh.append(gg[m_].mean())
            r["rim"] = float(np.clip(np.log2((max(sh) + eps) / (cy + eps)), -6, 6)) if sh else np.nan
            r["rim_band"] = float(np.clip(np.log2((gg[rim].mean() + eps) / (cy + eps)), -6, 6)) if rim.sum() >= 20 else np.nan
        else:
            r["nc"] = r["rim"] = r["rim_band"] = np.nan
        t = th[ss]; v = t[cell]; med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) + 1e-9
        sp, k = ndi.label((t > med + 4 * mad) & cell)
        if k:
            a = np.bincount(sp.ravel()); keep = a >= 3; keep[0] = False
            r["punct"] = float(np.clip(t[keep[sp]].sum(), 0, None) / max(np.clip(gg[cell], 0, None).sum(), 1e-9))
        else:
            r["punct"] = 0.0
        out.append(r)
    return out


# ----------------------------------------------------------------- HPA side
def _hpa_field(base):
    from PIL import Image
    ch = {c: np.asarray(Image.open(f"{IMG}/{base}_{c}.jpg").convert("RGB")).astype(np.float32) for c in ("blue", "red", "green")}
    rows = seg_cells(ch["blue"][..., 2], ch["red"][..., 0], ch["green"][..., 1])
    return base, rows


def hpa():
    import multiprocessing as mp
    os.makedirs(CA, exist_ok=True)
    M = pd.read_csv(f"{IMG}/manifest.tsv", sep="\t")
    M = M[(M.channel == "green") & M.present]
    M["base"] = M.file.str.replace("_green.jpg", "", regex=False)
    with mp.get_context("fork").Pool(NPROC) as pool:
        res = dict(pool.imap_unordered(_hpa_field, list(M.base), chunksize=2))
    rows = []
    for r in M.itertuples():
        for k, c in enumerate(res.get(r.base, [])):
            rows.append(dict(gene=r.gene, antibody=r.antibody, field=r.base, ab_reliability=r.ab_reliability, cell=k, **c))
    C = pd.DataFrame(rows); C.to_csv(f"{CA}/tier2_hpa_cells.tsv", sep="\t", index=False)
    log(f"HPA: {M.base.nunique()} fields, {len(C)} cells, {C.gene.nunique()} genes; median cells/field "
        f"{C.groupby('field').size().median():.0f}")
    G = agg_cells(C[C.ab_reliability != "uncertain"])
    G.to_csv(f"{CA}/tier2_hpa_genes.tsv", sep="\t")
    log(G.describe().to_string())


def agg_cells(C):
    g = C.groupby("gene")
    G = pd.DataFrame(dict(n_cells=g.size(), n_fields=g.field.nunique() if "field" in C else g.size()))
    for m in ("nc", "rim", "rim_band", "nuc_frac", "punct"):
        G[m] = g[m].median(); G[f"{m}_q25"] = g[m].quantile(0.25); G[f"{m}_q75"] = g[m].quantile(0.75)
    G["cell_area_um2"] = g.cell_area_um2.median(); G["nuc_area_um2"] = g.nuc_area_um2.median(); G["n_cyto_px"] = g.n_cyto_px.median()
    return G


# ----------------------------------------------------------------- model side
def _panel():
    import fluor as F
    genes = jurkat_genes()
    F.PANEL = [(g, "jurkat", ("gene", g), g) for g in genes] + [("H2B", "ref", ("re", r"^H2B"), "H2B (all)"),
                                                                ("TUBB", "ref", ("re", r"^TUBB"), "beta-tubulin (all)")]
    F.NAMES = [p[0] for p in F.PANEL]
    F.CACHE = f"{F.FD}/cache_tier2_{VER}"
    return F, genes


TPL_P = 0.02                  # fraction of all packed instances kept for the region templates


def _tpl_one(files):
    rng = np.random.default_rng(abs(hash(files[0])) % 2 ** 32); P, R, Wm = [], [], []
    for f in files:
        z = np.load(f); n = len(z["types"])
        if n == 0: continue
        k = rng.random(n) < TPL_P
        reg = np.repeat(z["job_region"].astype(np.int16), z["job_placed"])
        P.append(z["pos"][k].astype(np.float32)); R.append(reg[k])
    if not P: return np.zeros((0, 3), np.float32), np.zeros(0, np.int16)
    return np.concatenate(P), np.concatenate(R)


def collect():
    """panel instances (fluor.collect) + a 2% random sample of every packed instance with its region (region templates)"""
    import multiprocessing as mp
    F, genes = _panel(); os.makedirs(F.CACHE, exist_ok=True)
    F.NPROC = int(os.environ.get("TIER2_NPROC", 8))
    log(f"collect {len(genes)} genes + H2B + TUBB from {F.W}/tiles into {F.CACHE}")
    F.collect()
    fs = sorted(glob.glob(f"{F.W}/tiles/*.npz")); chunks = [fs[i::96] for i in range(96)]
    with mp.get_context("fork").Pool(F.NPROC) as pool:
        res = pool.map(_tpl_one, chunks)
    P = np.concatenate([r[0] for r in res]); R = np.concatenate([r[1] for r in res])
    np.savez(f"{F.CACHE}/templates.npz", pos=P, reg=R, p=TPL_P)
    log(f"templates: {len(R):,} sampled instances, {len(np.unique(R))} regions")


_M = {}


def _section_stack(name, pw=None):
    """confocal sections (noise-free, 80 nm px) of channel `name` (or of the points/weights pw) at the heights _M['zs']"""
    from scipy import ndimage as ndi
    pos, w = pw if pw is not None else _M["pts"][name][:2]; lo, n, H = _M["lo"], _M["n"], _M["H"]; sxy, sz = _M["sxy"], _M["sz"]
    out = []
    ij = np.floor((pos[:, :2] - lo[:2]) / H).astype(np.int64)
    okxy = np.all((ij >= 0) & (ij < n[:2]), 1)
    for z0 in _M["zs"]:
        dz = pos[:, 2] - z0; sel = okxy & (np.abs(dz) < 4 * sz)
        wz = w[sel] * np.exp(-dz[sel] ** 2 / (2 * sz ** 2))
        im = np.bincount(ij[sel, 0] * n[1] + ij[sel, 1], weights=wz, minlength=n[0] * n[1]).reshape(n[0], n[1]).astype(np.float32)
        im = ndi.gaussian_filter(im, sxy / H)
        out.append(np.clip(ndi.zoom(im, H / PX, order=1), 0, None))
    return out


def _model_gene(name):
    """two model images per section: 'sampled' = the gene's own packed molecules (copy-number shot noise included);
    'expected' = sum over pack regions of (gene's share in the region) x (region template: all packed instances of that
    region, 2% sample, normalised), i.e. the infinite-copy image of the gene's localisation (pre-pass shares: H2B/
    nucleoplasm, NPC/NE, others cytosol as fluor.load_points)."""
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 32)
    pos, w, reg = _M["pts"][name]; cop = float(w.sum()); rows = []
    fr = np.bincount(reg.astype(np.int64), weights=w, minlength=len(_M["tpl"])) / max(cop, 1e-12)
    st = {"sampled": _section_stack(name)}
    st["expected"] = [sum(fr[r] * _M["tpl"][r][k] for r in np.nonzero(fr > 1e-4)[0] if _M["tpl"][r] is not None)
                      for k in range(len(_M["zs"]))]
    for mode, stack in st.items():
        for k, z0 in enumerate(_M["zs"]):
            im = stack[k]
            if not isinstance(im, np.ndarray) or im.max() <= 0: continue
            ref = np.percentile(im[_M["inside"][k]], 99.5) if _M["inside"][k].any() else im.max()
            noisy = rng.poisson(im * 200.0 / max(ref, 1e-12) + 5.0).astype(np.float32)
            for c in seg_cells(_M["blue"][k], _M["red"][k], noisy, drop_border=False):
                rows.append(dict(gene=name, mode=mode, section=k, z_nm=z0, copies=cop, **c))
    return name, rows


def model():
    import multiprocessing as mp
    F, genes = _panel(); c3, cell = F.cell_geom()
    S = json.load(open(f"{OUT}/fluor_setup_v5.json" if os.path.exists(f"{OUT}/fluor_setup_v5.json") else f"{OUT}/fluor_setup.json"))   # frozen v5 section heights
    lo, n = np.array(S["grid_lo"]), np.array(S["grid_n"]); z0 = S["z0_nm"]
    zs = [z0 + d for d in (-1500, -1000, -500, 0, 500, 1000, 1500)]
    pts = F.load_points(); log("points loaded:", sum(len(v[0]) for v in pts.values()))
    sxy, sz = F.PSF["confocal"]["sxy"], F.PSF["confocal"]["sz"]
    _M.update(pts=pts, lo=lo, n=n, H=F.H, zs=zs, sxy=sxy, sz=sz)
    rng = np.random.default_rng(1)
    blue = _section_stack("H2B"); red = _section_stack("TUBB")
    nx, ny = blue[0].shape
    inside = []
    for k, z in enumerate(zs):
        x = lo[0] + (np.arange(nx) + 0.5) * PX; y = lo[1] + (np.arange(ny) + 0.5) * PX
        X, Y = np.meshgrid(x, y, indexing="ij")
        lab = cell.labels(np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)).reshape(nx, ny)
        inside.append(~np.isin(lab, [c3.EXT, c3.OUTV]))
    nb = lambda im, i: rng.poisson(im * 200.0 / max(np.percentile(im[inside[i]], 99.5), 1e-12) + 5.0).astype(np.float32)
    _M.update(blue=[nb(b, i) for i, b in enumerate(blue)], red=[nb(r, i) for i, r in enumerate(red)], inside=inside)
    tz = np.load(f"{F.CACHE}/templates.npz"); tpos, treg = tz["pos"], tz["reg"].astype(np.int64)
    nreg = max(int(treg.max()) + 1, 32); tpl = [None] * nreg
    for r in range(nreg):
        m = treg == r
        if m.sum() < 200: continue
        tpl[r] = [x / m.sum() for x in _section_stack(None, (tpos[m], np.ones(int(m.sum()), np.float32)))]
    # pre-pass regions as in fluor.load_points (0 cytosol, 1 nucleoplasm, NE): templates from the pre-pass itself where the
    # pack has none (nucleoplasm/cytosol/NE always have packed instances)
    _M["tpl"] = tpl; log("region templates:", sum(t is not None for t in tpl))
    rows = []
    with mp.get_context("fork").Pool(NPROC) as pool:
        for name, rr in pool.imap_unordered(_model_gene, genes):
            rows += rr
    C = pd.DataFrame(rows); os.makedirs(CA, exist_ok=True)
    C.to_csv(f"{CA}/tier2_model_{VER}_cells.tsv", sep="\t", index=False)
    log(f"model {VER}: {C.gene.nunique()} genes, {len(C)} section-cells; nucleus area per section:",
        C.groupby("section").nuc_area_um2.median().round(1).to_dict())
    np.savez_compressed(f"{CA}/tier2_model_{VER}_example.npz", blue=_M["blue"][3], red=_M["red"][3])


# ----------------------------------------------------------------- comparison
def compare():
    """per-gene model vs HPA metrics. Primary model image = 'expected' (infinite-copy image of the gene's region shares);
    'sampled' (the gene's own molecules, shot noise of low copy numbers) is reported alongside. Calibrated values use an
    affine imaging transfer HPA = a + b * model fitted once on the Jurkat fit half against v5 (expected mode) and frozen
    (tier2_transfer.json); calibrated agreement is only reported when b >= 0.1 (otherwise the transfer is flat and any
    model would 'agree')."""
    from scipy.stats import spearmanr, pearsonr
    H = pd.read_csv(f"{CA}/tier2_hpa_genes.tsv", sep="\t", index_col=0)
    Call = pd.read_csv(f"{CA}/tier2_model_{VER}_cells.tsv", sep="\t")
    hold, fit = read_split()
    MS = ("nc", "rim", "nuc_frac", "punct")
    agg = {}
    for mode in ("expected", "sampled"):
        Cm = Call[Call["mode"] == mode]
        Mg = agg_cells(Cm.assign(field=Cm.section)); Mg["copies"] = Cm.groupby("gene").copies.first(); agg[mode] = Mg
    J = H.join(agg["expected"], lsuffix="_hpa", rsuffix="_model", how="inner")
    J = J.join(agg["sampled"][list(MS)].add_suffix("_sampled"))
    J["split"] = ["holdout" if g in hold else "fit" if g in fit else "?" for g in J.index]
    lc = f"{OUT}/loc_compare_{VER}.tsv"
    if os.path.exists(lc):
        T = pd.read_csv(lc, sep="\t", index_col=0, low_memory=False, usecols=["gene", "model_regions", "hpa_jurkat", "mass_pg"])
        J = J.join(T[~T.index.duplicated()])
    cls = lambda x: np.where(x > 0.5, 1, np.where(x < -0.5, -1, 0))      # enriched / similar / depleted (log2 +-0.5)
    tf = f"{OUT}/tier2_transfer.json"
    if VER == "v5" or not os.path.exists(tf):
        fitj = J[J.split == "fit"]; T_ = {}
        for m in ("nc", "rim", "nuc_frac"):
            ok = fitj[f"{m}_model"].notna() & fitj[f"{m}_hpa"].notna()
            b_, a_ = np.polyfit(fitj.loc[ok, f"{m}_model"], fitj.loc[ok, f"{m}_hpa"], 1)
            T_[m] = dict(a=round(float(a_), 4), b=round(float(b_), 4), n=int(ok.sum()),
                         r=round(float(pearsonr(fitj.loc[ok, f"{m}_model"], fitj.loc[ok, f"{m}_hpa"])[0]), 3))
        json.dump(dict(fitted_on=f"tier2 {VER} (expected mode) vs HPA, Jurkat fit half (OLS HPA = a + b model); frozen", **T_),
                  open(tf, "w"), indent=1)
    TR = json.load(open(tf))
    for m in ("nc", "rim", "nuc_frac"):
        J[f"{m}_model_cal"] = TR[m]["a"] + TR[m]["b"] * J[f"{m}_model"]
    for m in ("nc", "rim"):
        J[f"{m}_diff"] = J[f"{m}_model"] - J[f"{m}_hpa"]
        J[f"{m}_agree2x"] = J[f"{m}_diff"].abs() <= 1.0
        J[f"{m}_class_agree"] = (cls(J[f"{m}_model"]) == cls(J[f"{m}_hpa"])) & J[f"{m}_model"].notna() & J[f"{m}_hpa"].notna()
        J[f"{m}_agree2x_cal"] = (J[f"{m}_model_cal"] - J[f"{m}_hpa"]).abs() <= 1.0
        J[f"{m}_class_agree_cal"] = (cls(J[f"{m}_model_cal"]) == cls(J[f"{m}_hpa"])) & J[f"{m}_model"].notna() & J[f"{m}_hpa"].notna()
    J.round(4).to_csv(f"{OUT}/tier2_{VER}.tsv", sep="\t")
    out = dict(version=VER, date=time.strftime("%Y-%m-%d"), pixel_nm=PX, transfer=TR,
               hpa_cells=int(H.n_cells.sum()), hpa_genes=int(len(H)), model_sections=int(Call.section.nunique()),
               definitions="nc: log2(mean nucleus interior / mean cytoplasm interior); rim: log2(peak 1-px shell mean from "
                           "0.24 um outside to 0.4 um inside the cell edge / mean cytoplasm interior); nuc_frac: share of cell "
                           "signal in the nucleus mask; punct: share of cell signal in top-hat spots. agree_2fold: |log2 model "
                           "- log2 HPA| <= 1; class: enriched (> +0.5) / similar / depleted (< -0.5) in log2; *_cal: after the "
                           "frozen affine imaging transfer. Primary = 'expected' model images; 'sampled' alongside.", sets={})
    excl = {"CD3E", "CD3D"} if VER not in ("v4", "v5", "v6") else set()   # v7+: excluded from hold-out scores (DECISION Pedro)
    out["excluded_from_holdout"] = sorted(excl)
    for name, sub in (("all", J), ("fit_half", J[J.split == "fit"]), ("holdout_half", J[(J.split == "holdout") & ~J.index.isin(excl)])):
        d = dict(n_genes=int(len(sub)))
        for mode, sfx in (("expected", "_model"), ("sampled", "_sampled")):
            dm = {}
            for m in MS:
                a, b = sub[f"{m}{sfx}"], sub[f"{m}_hpa"]; ok = a.notna() & b.notna()
                if ok.sum() < 4: continue
                dd = dict(n=int(ok.sum()), spearman=round(float(spearmanr(a[ok], b[ok])[0]), 3),
                          pearson=round(float(pearsonr(a[ok], b[ok])[0]), 3), mean_abs_diff=round(float((a[ok] - b[ok]).abs().mean()), 3),
                          median_model=round(float(a[ok].median()), 3), median_hpa=round(float(b[ok].median()), 3))
                if mode == "expected" and m in ("nc", "rim"):
                    dd["agree_2fold"] = round(float(sub.loc[ok, f"{m}_agree2x"].mean()), 3)
                    dd["class_agree"] = round(float(sub.loc[ok, f"{m}_class_agree"].mean()), 3)
                    if TR[m]["b"] >= 0.1:
                        dd["agree_2fold_cal"] = round(float(sub.loc[ok, f"{m}_agree2x_cal"].mean()), 3)
                        dd["class_agree_cal"] = round(float(sub.loc[ok, f"{m}_class_agree_cal"].mean()), 3)
                if mode == "expected" and m in ("nc", "rim", "nuc_frac") and TR[m]["b"] >= 0.1:
                    dd["mean_abs_diff_cal"] = round(float((sub.loc[ok, f"{m}_model_cal"] - b[ok]).abs().mean()), 3)
                dm[m] = dd
            d[mode] = dm
        out["sets"][name] = d
    json.dump(out, open(f"{OUT}/tier2_{VER}.json", "w"), indent=1)
    log(json.dumps({k: {m: {x: v["expected"][m].get(x) for x in ("spearman", "agree_2fold", "class_agree", "agree_2fold_cal")}
                        for m in v["expected"]} for k, v in out["sets"].items()}, indent=1))


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "split": split()
    if step == "collect": collect()
    if step == "hpa": hpa()
    if step == "model": model()
    if step == "compare": compare()
