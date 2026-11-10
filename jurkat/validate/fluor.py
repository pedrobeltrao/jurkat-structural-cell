"""Simulated fluorescence images of the packed Jurkat cell (validation backlog §8 item 6).

What a GFP fusion / immunostain of a given protein would look like in the model, with shape-independent metrics and
a qualitative comparison with HPA (incl. the JURKAT line where imaged) and OpenCell (HEK293T endogenous tags).

Method
  * Copies: every packed instance (wc/tiles/*.npz, 7,485 tiles, each read once by a fork pool of 8) plus the pre-pass
    (nucleosomes, H1, microtubules, actin filaments, NPCs, 80S). An instance of ingredient I contributes the summed
    stoichiometry of the channel's genes in I (inventory ingredients_v0.tsv genes/stoich; NPC = 7R5K composition, 80S
    = one copy per ribosomal protein, MT unit = one TUBA1B-TUBB4B heterodimer, F_ACTB = one actin).
  * Region per instance is exact: instances are stored in job order, region = repeat(job_region, job_placed).
  * Volume: weighted counts on a 40 nm grid. Optical axis = model z, i.e. perpendicular to the truncated flat face of
    cell 1 (EM z = 0 block face), so an xy section is like a cell resting on glass.
  * PSF (Gaussian approximations, applied in Fourier space per z-plane):
      confocal  FWHM 250 nm lateral / 600 nm axial (pinhole: plane weight exp(-dz^2/2 sz^2));
      airyscan  FWHM 140 nm / 400 nm;
      widefield FWHM 270 nm in focus, lateral sigma grows with defocus (sigma^2 = s0^2 + (1.2 dz)^2, NA 1.4 oil
                geometric cone), every plane contributes its full intensity (no optical sectioning).
  * Camera: 65 nm pixels (100x, 6.5 um sCMOS pixel), photon (Poisson) noise; peak in-cell signal scaled to 200 photons
    (confocal, airyscan) or 2,000 (widefield), background 5 photons, read noise 1.5 e-.
  * Images: one optical section through the nucleus centre (z of the nucleosome centroid), max and sum projections.
  * Metrics (fluor_metrics.tsv; on the noise-free confocal section unless named mol_*):
      mol_<comp>   fraction of molecules per compartment (model truth, whole cell)
      img_rim/img_cyto/img_nuc   fraction of section signal at the cell rim (-400..+300 nm from the PM), cytoplasm, nucleus
      pm_cyto, nuc_cyto          mean rim / mean cytoplasm interior, mean nucleus interior / mean cytoplasm interior
      edge_enrich, depth50_um    mean 0-0.3 um from the edge / mean 1-2 um; depth containing half the in-cell signal
      puncta_n, puncta_per100um2, puncta_diam_nm   spots on the noisy confocal section (top-hat, > 4 MAD, >= 3 px)
    Radial profiles from the cell edge (section) go to fluor_radial.tsv.
  * External: HPA subcellular_location.tsv (local), per-cell-line HPA ICC/IF annotations from the HPA XML of each panel
    gene (cached under data/reference/hpa_subcellular/xml/), OpenCell localisation grades (data/reference/opencell/).
    Disagreement = model molecule fraction in compartments the reference does not annotate (and total-variation
    distance to the reference distribution); circular cases (localisation taken from HPA) are flagged.

Usage (conda_envs/vcell/bin/python):
  python jurkat/validate/fluor.py collect    # read tiles once -> data/derived/validation/fluor/cache/points.npz (~20 min)
  python jurkat/validate/fluor.py external   # fetch HPA XML per panel gene + OpenCell csv (+ a few HPA example images)
  python jurkat/validate/fluor.py render     # images, fluor_metrics.tsv, fluor_radial.tsv, figures
  python jurkat/validate/fluor.py compare    # fluor_compare.tsv (model vs HPA/OpenCell), disagreement ranking
  python jurkat/validate/fluor.py all
Long runs: setsid nohup ... > logs/fluor.log 2>&1 &
Outputs: data/derived/validation/{fluor_metrics.tsv, fluor_radial.tsv, fluor_compare.tsv, fluor_panel.png, ...},
         data/derived/validation/fluor/<protein>_{section,proj}.png"""
import os, sys, re, json, glob, gzip, time
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
import numpy as np, pandas as pd

V = (__import__("os").environ.get("VCELL_ROOT", ".") + ""); D = f"{V}/data/derived"; W = f"{D}/wc"
OUT = f"{D}/validation"; FD = f"{OUT}/fluor"; CACHE = f"{FD}/cache"; REF = f"{V}/data/reference"
NPROC = 8
sys.path.insert(0, f"{V}/jurkat/yc/pipeline")
import jpack_test as _jp                                               # v5: region list incl. nucleolus and shells
RN = list(_jp.RNAMES)
COMPS = ["PM", "cytosol", "nucleus", "NE", "ER", "mito", "golgi", "endolysosomal"]
_R2C = {"cytosol": "cytosol", "nucleoplasm": "nucleus", "nucleolus": "nucleus", "matrix": "mito", "ER_lumen": "ER", "golgi_lumen": "golgi",
        "endo_lumen": "endolysosomal", "perinuclear": "NE", "PM": "PM", "NE": "NE", "ER": "ER", "MITO_OM": "mito", "MITO_IM": "mito",
        "GOLGI": "golgi", "ENDO": "endolysosomal", "sh_PM_cyto": "PM", "sh_PM_exo": "PM", "sh_PM_cortex": "PM", "sh_NE_lamina": "NE",
        "sh_NE_nuc": "NE", "sh_ER_cyto": "ER", "sh_ER_lumen": "ER", "sh_GOLGI_cyto": "golgi", "sh_GOLGI_lumen": "golgi",
        "sh_ENDO_cyto": "endolysosomal", "sh_ENDO_lumen": "endolysosomal", "sh_OM_cyto": "mito",
        "speckle": "nucleus", "sh_NPC": "NE", "sh_PM_periph": "PM", "sh_ENDO_periph": "endolysosomal", "sh_GOLGI_periph": "golgi"}  # v6
REG2COMP = np.array([COMPS.index(_R2C[r]) for r in RN])               # RN index -> COMPS index
H = 40.0                       # grid (nm)
PIX = 65.0                     # camera pixel (nm)
F2S = 1 / 2.3548
PSF = {"confocal": dict(sxy=250 * F2S, sz=600 * F2S, peak=200.0),
       "airyscan": dict(sxy=140 * F2S, sz=400 * F2S, peak=200.0),
       "widefield": dict(sxy=270 * F2S, slope=1.2, peak=2000.0)}
BG, READ = 5.0, 1.5

# name, group, matcher (gene exact | regex | ingredient id), label
PANEL = [
    ("CD3E", "T-cell PM", ("gene", "CD3E"), "CD3E (all)"), ("CD3E_free", "suspect", ("ing", "CD3E"), "free CD3E"),
    ("CD247", "T-cell PM", ("gene", "CD247"), "CD247 (CD3ζ)"), ("PTPRC", "T-cell PM", ("gene", "PTPRC"), "PTPRC (CD45)"),
    ("CD28", "T-cell PM", ("gene", "CD28"), "CD28"), ("TFRC", "T-cell PM", ("gene", "TFRC"), "TFRC"),
    ("CD2", "T-cell PM", ("gene", "CD2"), "CD2"), ("LAT", "T-cell PM", ("gene", "LAT"), "LAT"),
    ("LCK", "T-cell PM", ("gene", "LCK"), "LCK"), ("ZAP70", "T-cell PM", ("gene", "ZAP70"), "ZAP70"),
    ("TOMM20", "organelle", ("gene", "TOMM20"), "TOMM20"), ("CALR", "organelle", ("gene", "CALR"), "CALR"),
    ("GOLGA2", "organelle", ("gene", "GOLGA2"), "GOLGA2 (GM130)"), ("LAMP1", "organelle", ("gene", "LAMP1"), "LAMP1"),
    ("LMNB1", "organelle", ("gene", "LMNB1"), "LMNB1"), ("H2B", "organelle", ("re", r"^H2B"), "H2B (all)"),
    ("ACTB", "organelle", ("gene", "ACTB"), "ACTB"), ("TUBB", "organelle", ("re", r"^TUBB"), "β-tubulin (all)"),
    ("NUP98", "organelle", ("gene", "NUP98"), "NUP98"), ("RPS6", "organelle", ("gene", "RPS6"), "RPS6 (ribosome)"),
    ("CD7", "suspect", ("gene", "CD7"), "CD7"), ("CD99", "suspect", ("gene", "CD99"), "CD99"),
    ("SELPLG", "suspect", ("gene", "SELPLG"), "SELPLG (PSGL-1)"), ("HSPA5", "suspect", ("gene", "HSPA5"), "HSPA5 (BiP)"),
    ("PDIA3", "suspect", ("gene", "PDIA3"), "PDIA3"),
    # non-HPA localisation sources and/or changed in inventory v4
    ("PGK1", "v4/non-HPA", ("gene", "PGK1"), "PGK1"), ("PRDX1", "v4/non-HPA", ("gene", "PRDX1"), "PRDX1"),
    ("LDHA", "v4/non-HPA", ("gene", "LDHA"), "LDHA"), ("VDAC1", "v4/non-HPA", ("gene", "VDAC1"), "VDAC1"),
    ("GSTP1", "v4/non-HPA", ("gene", "GSTP1"), "GSTP1"), ("CLIC1", "v4/non-HPA", ("gene", "CLIC1"), "CLIC1"),
    ("CLIC4", "v4/non-HPA", ("gene", "CLIC4"), "CLIC4"), ("P4HB", "v4/non-HPA", ("gene", "P4HB"), "P4HB (PDI)"),
    ("PDIA6", "v4/non-HPA", ("gene", "PDIA6"), "PDIA6"), ("GOT2", "v4/non-HPA", ("gene", "GOT2"), "GOT2"),
    ("HSPD1", "v4/non-HPA", ("gene", "HSPD1"), "HSPD1"), ("ATP5F1A", "v4/non-HPA", ("gene", "ATP5F1A"), "ATP5F1A"),
    ("HSPE1", "v4/non-HPA", ("gene", "HSPE1"), "HSPE1"), ("PPIA", "v4/non-HPA", ("gene", "PPIA"), "PPIA"),
    ("MYL6", "v4/non-HPA", ("gene", "MYL6"), "MYL6"), ("RAC2", "v4/non-HPA", ("gene", "RAC2"), "RAC2"),
    ("DBI", "v4/non-HPA", ("gene", "DBI"), "DBI (ACBP)"), ("ESYT1", "v4/non-HPA", ("gene", "ESYT1"), "ESYT1"),
    ("RTN4", "v4/non-HPA", ("gene", "RTN4"), "RTN4"), ("UBA1", "v4/non-HPA", ("gene", "UBA1"), "UBA1"),
]
NAMES = [p[0] for p in PANEL]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------- channel weights per library
def ingredient_table():
    I = pd.read_csv(f"{D}/ingredients_v0.tsv", sep="\t")
    return {r.ing_id: (str(r.genes).split(";"), [float(s) for s in str(r.stoich).split(";")]) for r in I.itertuples()}


def chan_weights(lib_names):
    """(n_lib, n_chan) float32: copies of each channel's protein per instance of each library"""
    ING = ingredient_table(); Wt = np.zeros((len(lib_names), len(PANEL)), np.float32)
    for i, l in enumerate(lib_names):
        ing = l[3:] if l.startswith("MT_") else (l[2:] if l.startswith("F_") else l)
        if ing not in ING: continue
        gs, st = ING[ing]
        for c, (_, _, (kind, val), _) in enumerate(PANEL):
            if kind == "ing": Wt[i, c] = sum(st) if ing == val else 0.0
            elif kind == "gene": Wt[i, c] = sum(s for g, s in zip(gs, st) if g == val)
            else: Wt[i, c] = sum(s for g, s in zip(gs, st) if re.match(val, g))
    return Wt


# ----------------------------------------------------------------- step 1: collect panel instances from the tiles
_SEL = None


def _collect(files):
    P, L, R = [], [], []
    for f in files:
        z = np.load(f)
        t = z["types"]; m = _SEL[t]
        if not m.any(): continue
        reg = np.repeat(z["job_region"].astype(np.int8), z["job_placed"])
        P.append(z["pos"][m].astype(np.float32)); L.append(t[m].astype(np.int32)); R.append(reg[m])
    if not P: return np.zeros((0, 3), np.float32), np.zeros(0, np.int32), np.zeros(0, np.int8)
    return np.concatenate(P), np.concatenate(L), np.concatenate(R)


def collect():
    global _SEL
    import multiprocessing as mp
    os.makedirs(CACHE, exist_ok=True)
    libs = json.load(open(f"{W}/libs.json"))["libs"]; Wt = chan_weights(libs)
    _SEL = Wt.sum(1) > 0
    log(f"{_SEL.sum()} of {len(libs)} libraries carry a panel protein")
    fs = sorted(glob.glob(f"{W}/tiles/*.npz"), key=lambda s: int(os.path.basename(s)[:-4]))
    chunks = [fs[i::96] for i in range(96)]
    P, L, R = [], [], []
    with mp.get_context("fork").Pool(NPROC) as pool:
        for k, (p, l, r) in enumerate(pool.imap_unordered(_collect, chunks)):
            P.append(p); L.append(l); R.append(r)
            if k % 8 == 0: log(f"chunk {k + 1}/{len(chunks)}: {sum(len(x) for x in L):,} panel instances")
    P, L, R = np.concatenate(P), np.concatenate(L), np.concatenate(R)
    np.savez(f"{CACHE}/points.npz", pos=P, lib=L, reg=R, lib_names=np.array(libs), n_tiles=len(fs))
    log(f"saved {len(L):,} instances from {len(fs)} tiles")


# ----------------------------------------------------------------- loading all channel points (tiles + pre-pass)
def load_points():
    z = np.load(f"{CACHE}/points.npz")
    libs = list(z["lib_names"]); Wt = chan_weights(libs)
    pos, lib, reg = z["pos"], z["lib"], z["reg"]
    PL = json.load(open(f"{W}/prepass_libs.json")); Wp = chan_weights(PL)
    pt = np.load(f"{W}/prepass_types.npy", mmap_mode="r"); ppos = np.load(f"{W}/prepass_pos.npy", mmap_mode="r")
    # v6 accounting: pre-pass F-actin is cortical (8-150 nm under the PM) -> cortex shell region (PM compartment)
    preg = np.array([1 if (l == "NUCLEOSOME" or l.startswith("H1-")) else RN.index("NE") if l == "NPC" else
                     RN.index("sh_PM_cortex") if (l == "F_ACTB" and "sh_PM_cortex" in RN) else 0 for l in PL], np.int8)
    m = Wp.sum(1)[pt] > 0
    qpos = np.asarray(ppos[m], np.float32); qt = np.asarray(pt[m])
    out = {}
    for c, name in enumerate(NAMES):
        wa = Wt[lib, c]; ma = wa > 0; wb = Wp[qt, c]; mb = wb > 0
        out[name] = (np.concatenate([pos[ma], qpos[mb]]), np.concatenate([wa[ma], wb[mb]]),
                     np.concatenate([reg[ma], preg[qt[mb]]]))
    return out


# ----------------------------------------------------------------- geometry helpers
def cell_geom():
    sys.path.insert(0, f"{V}/jurkat/yc/pipeline")
    import cell3d as c3
    return c3, c3.Cell.load(f"{D}/jgeom_cell1.npz")


def grid_box():
    M = pd.read_parquet(f"{W}/tiles.parquet"); meta = json.load(open(f"{W}/tiles_meta.json"))
    lo0, T = np.array(meta["lo0"]), meta["TILE"]
    lo = lo0 + T * M[["ti", "tj", "tk"]].min().values - 800.0
    hi = lo0 + T * (M[["ti", "tj", "tk"]].max().values + 1) + 800.0
    n = np.ceil((hi - lo) / H).astype(int); n += n % 2
    return lo, n


def plane_labels(cell, c3, lo, n, z0, h):
    """label map of the plane z = z0 sampled at pixel centres of size h (nm) covering the grid box"""
    nx, ny = int(round(n[0] * H / h)), int(round(n[1] * H / h))
    x = lo[0] + (np.arange(nx) + 0.5) * h; y = lo[1] + (np.arange(ny) + 0.5) * h
    X, Y = np.meshgrid(x, y, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z0)], 1)
    return cell.labels(P).reshape(nx, ny)


# ----------------------------------------------------------------- PSF in Fourier space
def _freq(nx, ny):
    fx = np.fft.fftfreq(nx, d=H)[:, None]; fy = np.fft.rfftfreq(ny, d=H)[None, :]
    return (fx ** 2 + fy ** 2).astype(np.float32)


def _gxy(F2, s):
    return np.exp(-2 * np.pi ** 2 * s ** 2 * F2).astype(np.float32)


def render_channel(vol_fft, zc, iz0, F2, nx, ny):
    """vol_fft: rfft2 of every z-plane (nz, nx, ny//2+1); returns dict of 40 nm images"""
    dz = (np.arange(vol_fft.shape[0]) - iz0) * H
    img = {}
    for mode in ("confocal", "airyscan"):
        p = PSF[mode]; wz = np.exp(-dz ** 2 / (2 * p["sz"] ** 2)).astype(np.float32)
        acc = np.tensordot(wz, vol_fft, axes=(0, 0))
        img[mode] = np.fft.irfft2(acc * _gxy(F2, p["sxy"]), s=(nx, ny))
    p = PSF["widefield"]; acc = np.zeros(vol_fft.shape[1:], np.complex64)
    for k in np.nonzero(np.abs(vol_fft[:, 0, 0]) > 0)[0]:
        s = np.sqrt(p["sxy"] ** 2 + (p["slope"] * dz[k]) ** 2); acc += vol_fft[k] * _gxy(F2, s)
    img["widefield"] = np.fft.irfft2(acc, s=(nx, ny))
    gc = _gxy(F2, PSF["confocal"]["sxy"])
    img["sum"] = np.fft.irfft2(vol_fft.sum(0) * gc, s=(nx, ny))
    # max projection of the confocal 3D image: blur along z (separable), then laterally per plane
    from scipy.ndimage import gaussian_filter1d
    sz = PSF["confocal"]["sz"] / H
    zr = gaussian_filter1d(vol_fft.real, sz, axis=0, mode="constant") + 1j * gaussian_filter1d(vol_fft.imag, sz, axis=0, mode="constant")
    mx = np.full((nx, ny), -np.inf, np.float32)
    for k in range(zr.shape[0]):
        if zr[k, 0, 0].real <= 1e-6: continue
        mx = np.maximum(mx, np.fft.irfft2(zr[k] * gc, s=(nx, ny)))
    img["max"] = np.where(np.isfinite(mx), mx, 0)
    return {k: np.clip(v, 0, None).astype(np.float32) for k, v in img.items()}


def to_camera(im):
    from scipy.ndimage import zoom
    return np.clip(zoom(im, H / PIX, order=1), 0, None) * (PIX / H) ** 2


def add_noise(im, peak, ref, rng):
    s = peak / max(ref, 1e-12)
    return rng.poisson(im * s + BG).astype(np.float32) + rng.normal(0, READ, im.shape).astype(np.float32), s


# ----------------------------------------------------------------- metrics
def section_metrics(img, lab, c3, rng_noisy):
    from scipy.ndimage import distance_transform_edt, label as cclabel, white_tophat
    inside = ~np.isin(lab, [c3.EXT, c3.OUTV])
    nuc = np.isin(lab, [c3.NUCP, c3.NUCL, c3.PERI, c3.PORE, getattr(c3, "SPK", c3.NUCP)])
    depth = distance_transform_edt(inside) * PIX; outd = distance_transform_edt(~inside) * PIX
    sd = np.where(inside, depth, -outd)                                   # signed depth (nm), + inside
    nd = distance_transform_edt(~nuc) * PIX; ndin = distance_transform_edt(nuc) * PIX
    rim = (sd > -400) & (sd < 300)
    cyto_in = inside & ~nuc & (sd > 600) & (nd > 400)
    nuc_in = nuc & (ndin > 400)
    tot = img[(sd > -400)].sum() + 1e-30
    m = dict(img_rim=img[rim].sum() / tot, img_nuc=img[nuc & ~rim].sum() / tot)
    m["img_cyto"] = 1 - m["img_rim"] - m["img_nuc"]
    cy = img[cyto_in].mean() if cyto_in.any() else np.nan
    m["pm_cyto"] = img[rim & (sd > 0)].mean() / cy
    m["nuc_cyto"] = img[nuc_in].mean() / cy
    a = img[inside & (depth < 300)].mean(); b = img[inside & (depth > 1000) & (depth < 2000)].mean()
    m["edge_enrich"] = a / b
    dv = depth[inside]; iv = img[inside]; o = np.argsort(dv); cs = np.cumsum(iv[o]); cs /= cs[-1]
    m["depth50_um"] = float(dv[o][np.searchsorted(cs, 0.5)]) / 1000
    bins = np.arange(0, 6001, 200); prof = []
    for lo_, hi_ in zip(bins[:-1], bins[1:]):
        s = inside & (depth >= lo_) & (depth < hi_)
        prof.append(img[s].mean() if s.sum() > 20 else np.nan)
    prof = np.array(prof) / img[inside].mean()
    # puncta on the noisy section
    nz = rng_noisy; th = white_tophat(nz, size=int(round(1200 / PIX)))
    v = th[inside]; med = np.median(v); mad = np.median(np.abs(v - med)) * 1.4826 + 1e-9
    spots, k = cclabel((th > med + 4 * mad) & inside)
    if k:
        area = np.bincount(spots.ravel())[1:]; area = area[area >= 3]
    else:
        area = np.zeros(0)
    m["puncta_n"] = int(len(area)); m["puncta_per100um2"] = len(area) / (inside.sum() * PIX ** 2 / 1e6) * 100
    m["puncta_diam_nm"] = float(np.median(2 * np.sqrt(area * PIX ** 2 / np.pi))) if len(area) else np.nan
    m["cyto_cv"] = float(img[cyto_in].std() / cy) if cyto_in.any() else np.nan
    return m, prof, bins, sd


# ----------------------------------------------------------------- plotting
GREEN = None


def green_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("g", [(0, 0, 0), (0.0, 0.55, 0.1), (0.55, 1.0, 0.55), (1, 1, 1)])


def show(ax, im, lab=None, c3=None, title=None, bar_um=5.0, vmax=None):
    vmax = vmax if vmax is not None else np.percentile(im, 99.7)
    ax.imshow(im.T, origin="lower", cmap=green_cmap(), vmin=np.percentile(im, 1), vmax=vmax, interpolation="nearest")
    if lab is not None:
        ax.contour((~np.isin(lab, [c3.EXT, c3.OUTV])).T.astype(float), [0.5], colors=["#888888"], linewidths=0.4, alpha=0.6)
        ax.contour(np.isin(lab, [c3.NUCP, c3.NUCL, c3.PERI, c3.PORE, getattr(c3, "SPK", c3.NUCP)]).T.astype(float), [0.5], colors=["#6b8fd6"],
                   linewidths=0.4, alpha=0.6)
    L = bar_um * 1000 / PIX; ny = im.shape[1]
    ax.plot([im.shape[0] * 0.05, im.shape[0] * 0.05 + L], [ny * 0.06] * 2, color="white", lw=2)
    ax.text(im.shape[0] * 0.05 + L / 2, ny * 0.09, f"{bar_um:g} µm", color="white", ha="center", va="bottom", fontsize=6)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=7, color="black", pad=2)


# ----------------------------------------------------------------- step 3: render + metrics
_G = {}


def _render_one(name):
    pos, w, reg = _G["pts"][name]
    lo, n, iz0, F2, labp = _G["lo"], _G["n"], _G["iz0"], _G["F2"], _G["labp"]
    c3 = _G["c3"]
    ijk = np.floor((pos - lo) / H).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < n), 1)
    flat = (ijk[ok, 2] * n[0] + ijk[ok, 0]) * n[1] + ijk[ok, 1]
    vol = np.bincount(flat, weights=w[ok], minlength=n[0] * n[1] * n[2]).astype(np.float32).reshape(n[2], n[0], n[1])
    vf = np.fft.rfft2(vol).astype(np.complex64); del vol
    ims = render_channel(vf, None, iz0, F2, n[0], n[1]); del vf
    cam = {k: to_camera(v) for k, v in ims.items()}
    inside = ~np.isin(labp, [c3.EXT, c3.OUTV])
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 32)
    noisy, snr = {}, {}
    for k in ("confocal", "airyscan", "widefield", "max", "sum"):
        peak = PSF.get(k, PSF["confocal"])["peak"]
        ref = np.percentile(cam[k][inside], 99.5)
        noisy[k], s = add_noise(cam[k], peak, ref, rng)
        sig = np.median(cam[k][inside]) * s
        snr[k] = sig / np.sqrt(sig + BG + READ ** 2)
    met, prof, bins, sd = section_metrics(cam["confocal"], labp, c3, noisy["confocal"])
    tot = float(w.sum()); cf = np.bincount(REG2COMP[reg], weights=w, minlength=len(COMPS)) / max(tot, 1e-30)
    row = dict(protein=name, copies=round(tot), **{f"mol_{c}": round(float(x), 4) for c, x in zip(COMPS, cf)},
               **{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in met.items()},
               snr_median_confocal=round(float(snr["confocal"]), 2))
    np.savez_compressed(f"{CACHE}/img_{name}.npz", **{f"cam_{k}": v for k, v in cam.items()},
                        **{f"noisy_{k}": v for k, v in noisy.items()})
    return row, prof


def render():
    import multiprocessing as mp
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(FD, exist_ok=True)
    c3, cell = cell_geom()
    lo, n = grid_box(); log("grid", n, "voxels", int(np.prod(n)))
    pts = load_points(); log("points loaded:", {k: len(v[0]) for k, v in pts.items()})
    hz = pts["H2B"]; z0 = float(np.average(hz[0][:, 2], weights=hz[1])); c_nuc = np.average(hz[0], 0, weights=hz[1])
    iz0 = int(np.floor((z0 - lo[2]) / H)); z0 = lo[2] + (iz0 + 0.5) * H
    log(f"nucleus centroid {np.round(c_nuc)}; section z0 = {z0:.0f} nm")
    # flat-face check: cell cross-section area vs z on a 200 nm grid
    xs = np.arange(lo[0], lo[0] + n[0] * H, 200); ys = np.arange(lo[1], lo[1] + n[1] * H, 200)
    zs = np.arange(lo[2], lo[2] + n[2] * H, 200)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    lab3 = cell.labels(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)).reshape(X.shape)
    area = ((~np.isin(lab3, [c3.EXT, c3.OUTV])).sum((0, 1)) * 0.04)
    nz_ = np.nonzero(area)[0]
    face = dict(z_min=float(zs[nz_[0]]), z_max=float(zs[nz_[-1]]), area_first_um2=float(area[nz_[0]]),
                area_last_um2=float(area[nz_[-1]]), area_max_um2=float(area.max()))
    xa = (~np.isin(lab3, [c3.EXT, c3.OUTV])).sum((1, 2)); ya = (~np.isin(lab3, [c3.EXT, c3.OUTV])).sum((0, 2))
    face.update(x_first=float(xa[np.nonzero(xa)[0][0]] * 0.04), x_last=float(xa[np.nonzero(xa)[0][-1]] * 0.04),
                y_first=float(ya[np.nonzero(ya)[0][0]] * 0.04), y_last=float(ya[np.nonzero(ya)[0][-1]] * 0.04))
    log("cross-section areas at the extremes (um2):", face)
    labp = plane_labels(cell, c3, lo, n, z0, PIX)
    F2 = _freq(n[0], n[1])
    # camera image shape must match labp
    _G.update(pts=pts, lo=lo, n=n, iz0=iz0, F2=F2, labp=labp, c3=c3)
    from scipy.ndimage import zoom
    shp = zoom(np.zeros((n[0], n[1]), np.float32), H / PIX, order=1).shape
    if shp != labp.shape:
        labp = labp[:shp[0], :shp[1]] if labp.shape[0] >= shp[0] and labp.shape[1] >= shp[1] else np.pad(
            labp, ((0, max(0, shp[0] - labp.shape[0])), (0, max(0, shp[1] - labp.shape[1]))), constant_values=c3.EXT)[:shp[0], :shp[1]]
        _G["labp"] = labp
    rows, profs = {}, {}
    with mp.get_context("fork").Pool(NPROC) as pool:
        for name, (row, prof) in zip(NAMES, pool.imap(_render_one, NAMES)):
            rows[name] = row; profs[name] = prof; log("rendered", name, {k: row[k] for k in ("copies", "pm_cyto", "nuc_cyto", "puncta_n")})
    T = pd.DataFrame([rows[k] for k in NAMES])
    T.insert(1, "group", [p[1] for p in PANEL]); T.insert(2, "label", [p[3] for p in PANEL])
    T.to_csv(f"{OUT}/fluor_metrics.tsv", sep="\t", index=False)
    bins = np.arange(0, 6001, 200)
    R = pd.DataFrame([dict(protein=k, depth_um=(b + 100) / 1000, rel_intensity=round(float(v), 4))
                      for k in NAMES for b, v in zip(bins[:-1], profs[k])])
    R.to_csv(f"{OUT}/fluor_radial.tsv", sep="\t", index=False)
    json.dump(dict(z0_nm=z0, nucleus_centroid_nm=c_nuc.tolist(), grid_lo=lo.tolist(), grid_n=n.tolist(), grid_nm=H,
                   pixel_nm=PIX, psf=PSF, bg=BG, read=READ, face=face), open(f"{OUT}/fluor_setup.json", "w"), indent=1)
    figures(labp, c3)


def figures(labp=None, c3=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if labp is None:
        c3, cell = cell_geom(); S = json.load(open(f"{OUT}/fluor_setup.json"))
        lo, n = np.array(S["grid_lo"]), np.array(S["grid_n"])
        labp = plane_labels(cell, c3, lo, n, S["z0_nm"], PIX)
    T = pd.read_csv(f"{OUT}/fluor_metrics.tsv", sep="\t").set_index("protein")
    lab_of = {p[0]: p[3] for p in PANEL}
    imgs = {k: np.load(f"{CACHE}/img_{k}.npz") for k in NAMES}
    # crop to the cell bounding box + 1 um
    ins = np.nonzero(~np.isin(labp, [c3.EXT, c3.OUTV])); m = int(1000 / PIX)
    sl = (slice(max(ins[0].min() - m, 0), ins[0].max() + m), slice(max(ins[1].min() - m, 0), ins[1].max() + m))
    lab_c = labp[sl]
    for k in NAMES:
        z = imgs[k]
        for kind, arrs in (("section", [("noisy_confocal", "confocal section"), ("noisy_airyscan", "Airyscan section"),
                                         ("noisy_widefield", "widefield section")]),
                           ("proj", [("noisy_max", "confocal max projection"), ("noisy_sum", "sum projection")])):
            fig, axs = plt.subplots(1, len(arrs), figsize=(3.2 * len(arrs), 2.6), facecolor="white")
            for ax, (a, t) in zip(np.atleast_1d(axs), arrs):
                show(ax, z[a][sl], lab_c if kind == "section" else None, c3, t)
            fig.suptitle(f"{lab_of[k]}: {T.loc[k, 'copies']:,.0f} copies", fontsize=9)
            fig.tight_layout(); fig.savefig(f"{FD}/{k}_{kind}.png", dpi=150); plt.close(fig)
    # montage
    nc = 8; nr = int(np.ceil(len(NAMES) / nc))
    h_w = lab_c.shape[1] / lab_c.shape[0]
    fig, axs = plt.subplots(nr, nc, figsize=(nc * 2.0, nr * 2.0 * h_w + 0.6), facecolor="white")
    for ax, k in zip(axs.ravel(), NAMES):
        show(ax, imgs[k]["noisy_confocal"][sl], None, c3, f"{lab_of[k]}  PM/C {T.loc[k, 'pm_cyto']:.1f}  N/C {T.loc[k, 'nuc_cyto']:.1f}")
    for ax in axs.ravel()[len(NAMES):]: ax.axis("off")
    fig.suptitle("Simulated confocal sections through the nucleus centre (model Jurkat cell 1, pack v4; FWHM 250/600 nm, "
                 "65 nm pixels, Poisson noise, peak 200 photons)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fluor_panel.png", dpi=130); plt.close(fig)
    fig, axs = plt.subplots(nr, nc, figsize=(nc * 2.0, nr * 2.0 * h_w + 0.6), facecolor="white")
    for ax, k in zip(axs.ravel(), NAMES):
        show(ax, imgs[k]["noisy_max"][sl], None, c3, lab_of[k])
    for ax in axs.ravel()[len(NAMES):]: ax.axis("off")
    fig.suptitle("Simulated confocal max projections along z (optical axis perpendicular to the truncated face)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fluor_panel_proj.png", dpi=130); plt.close(fig)
    # PSF comparison
    sel = ["PTPRC", "TOMM20", "CALR", "LAMP1", "NUP98", "TUBB"]
    fig, axs = plt.subplots(len(sel), 3, figsize=(7.5, 2.2 * len(sel) * h_w + 0.5), facecolor="white")
    for i, k in enumerate(sel):
        for j, (a, t) in enumerate([("noisy_widefield", "widefield"), ("noisy_confocal", "confocal"), ("noisy_airyscan", "Airyscan")]):
            show(axs[i, j], imgs[k][a][sl], None, c3, f"{lab_of[k]}: {t}")
    fig.tight_layout(); fig.savefig(f"{OUT}/fluor_psf_compare.png", dpi=130); plt.close(fig)
    # radial profiles
    R = pd.read_csv(f"{OUT}/fluor_radial.tsv", sep="\t")
    groups = [("T-cell PM", None), ("organelle", None), ("suspect", None), ("v4/non-HPA", None)]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6), facecolor="white", sharey=True)
    for ax, (g, _) in zip(axs, groups):
        for k in [p[0] for p in PANEL if p[1] == g]:
            r = R[R.protein == k]; ax.plot(r.depth_um, r.rel_intensity, lw=1, label=lab_of[k])
        ax.set_title(g, fontsize=9); ax.set_xlabel("depth from cell edge (µm), section"); ax.set_yscale("log")
        ax.legend(fontsize=5, ncol=2); ax.axhline(1, color="grey", lw=0.5)
    axs[0].set_ylabel("intensity / mean in-cell intensity")
    fig.tight_layout(); fig.savefig(f"{OUT}/fluor_radial.png", dpi=130); plt.close(fig)
    log("figures written")


# ----------------------------------------------------------------- step 2: external references
HPA_MAP = {"nucleus": ["Nucleoplasm", "Nuclear bodies", "Nuclear speckles", "Nucleoli", "Nucleoli fibrillar center",
                       "Nucleoli rim", "Mitotic chromosome", "Kinetochore", "Perinuclear theca"],
           "NE": ["Nuclear membrane"],
           "cytosol": ["Cytosol", "Cytoplasmic bodies", "Rods & Rings", "Aggresome", "Actin filaments", "Microtubules",
                       "Microtubule ends", "Intermediate filaments", "Centrosome", "Centriolar satellite", "Basal body",
                       "Mitotic spindle", "Midbody", "Midbody ring", "Cytokinetic bridge", "Cleavage furrow"],
           "mito": ["Mitochondria"], "ER": ["Endoplasmic reticulum"], "golgi": ["Golgi apparatus"],
           "PM": ["Plasma membrane", "Cell Junctions", "Focal adhesion sites"],
           "endolysosomal": ["Vesicles", "Endosomes", "Lysosomes", "Peroxisomes", "Lipid droplets"]}
T2C = {t.lower(): c for c, ts in HPA_MAP.items() for t in ts}
OC_MAP = {"cytoplasmic": "cytosol", "nucleoplasm": "nucleus", "nucleolus_gc": "nucleus", "nucleolus_fc_dfc": "nucleus",
          "chromatin": "nucleus", "nuclear_punctae": "nucleus", "nucleus_se": "nucleus", "nuclear_membrane": "NE",
          "membrane": "PM", "cell_contact": "PM", "focal_adhesions": "PM", "er": "ER", "golgi": "golgi",
          "mitochondria": "mito", "vesicles": "endolysosomal", "centrosome": "cytosol", "cytoskeleton": "cytosol",
          "big_aggregates": "cytosol", "microtubules": "cytosol"}
PANEL_GENES = sorted({p[2][1] for p in PANEL if p[2][0] in ("gene", "ing")} | {"H2BC12", "TUBB", "TUBB4B"})


def external():
    import urllib.request, xml.etree.ElementTree as ET
    xd = f"{REF}/hpa_subcellular/xml"; os.makedirs(xd, exist_ok=True); od = f"{REF}/opencell"; os.makedirs(od, exist_ok=True)
    ocf = f"{od}/opencell-localization-annotations.csv"
    if not os.path.exists(ocf):
        urllib.request.urlretrieve("https://opencell.czbiohub.org/data/datasets/opencell-localization-annotations.csv", ocf)
    hpa = pd.read_csv(f"{REF}/hpa_subcellular/subcellular_location.tsv", sep="\t").drop_duplicates("Gene name").set_index("Gene name")
    rows = []
    for g in PANEL_GENES:
        if g not in hpa.index: log("not in HPA tsv:", g); continue
        ens = hpa.at[g, "Gene"]; f = f"{xd}/{g}_{ens}.xml.gz"
        if not os.path.exists(f):
            for _ in range(3):
                try:
                    data = urllib.request.urlopen(f"https://www.proteinatlas.org/{ens}.xml", timeout=120).read(); break
                except Exception as e:
                    log("retry", g, e); time.sleep(5)
            else:
                continue
            gzip.open(f, "wb").write(data); time.sleep(0.5)
        root = ET.fromstring(gzip.open(f).read())
        for ce in root.iter("cellExpression"):
            if ce.get("technology") != "ICC/IF": continue
            for sa in ce.iter("subAssay"):
                if sa.get("type") != "human": continue
                ver = sa.find("verification"); ver = ver.text if ver is not None else ""
                for d in sa.iter("data"):
                    cl = d.find("cellLine")
                    if cl is None: continue
                    locs = [l.text for l in d.findall("location")]
                    urls = [u.text for u in d.iter("imageUrl")]
                    rows.append(dict(gene=g, ensg=ens, cell_line=cl.text, reliability=ver, locations=";".join(locs),
                                     image_urls=";".join(urls)))
    C = pd.DataFrame(rows); C.to_csv(f"{REF}/hpa_subcellular/panel_celllines.tsv", sep="\t", index=False)
    lines = C.cell_line.value_counts(); log("HPA ICC/IF cell lines over the panel:", lines.to_dict())
    J = C[C.cell_line.str.upper().str.contains("JURKAT")]
    log(f"panel genes imaged in JURKAT: {sorted(J.gene.unique())}")
    # a handful of example images: JURKAT where available (first image per gene, max 12)
    ed = f"{FD}/hpa_examples"; os.makedirs(ed, exist_ok=True); nimg = 0
    for r in J.drop_duplicates("gene").itertuples():
        if nimg >= 12 or not r.image_urls: continue
        u = r.image_urls.split(";")[0]; fn = f"{ed}/{r.gene}_JURKAT_{os.path.basename(u)}"
        if not os.path.exists(fn):
            try: urllib.request.urlretrieve(u, fn); time.sleep(0.5)
            except Exception as e: log("image fail", u, e); continue
        nimg += 1
    log(f"{nimg} HPA example images in {ed}")


# ----------------------------------------------------------------- step 4: comparison with HPA / OpenCell
def _dist(terms, mapping, weights):
    q = {}
    for t, wt in zip(terms, weights):
        c = mapping.get(str(t).strip().lower())
        if c: q[c] = max(q.get(c, 0.0), wt)
    s = sum(q.values()); return {k: v / s for k, v in q.items()} if s else None


def compare():
    T = pd.read_csv(f"{OUT}/fluor_metrics.tsv", sep="\t").set_index("protein")
    hpa = pd.read_csv(f"{REF}/hpa_subcellular/subcellular_location.tsv", sep="\t").drop_duplicates("Gene name").set_index("Gene name")
    OC = pd.read_csv(f"{REF}/opencell/opencell-localization-annotations.csv").drop_duplicates("target_name").set_index("target_name")
    C = pd.read_csv(f"{REF}/hpa_subcellular/panel_celllines.tsv", sep="\t")
    inv = pd.read_csv(f"{D}/inventory_v0.tsv", sep="\t").set_index("gene")
    chg = pd.read_csv(f"{D}/inventory_v4_changes.tsv", sep="\t"); chg = chg[chg.field == "comp_weights"]
    chg_genes = set(g for s in chg.gene.astype(str) for g in s.split(";"))
    rep = {"H2B": "H2BC12", "TUBB": "TUBB", "CD3E_free": "CD3E"}
    rows = []
    for k in NAMES:
        g = rep.get(k, k); p = {c: T.loc[k, f"mol_{c}"] for c in COMPS}
        r = dict(protein=k, gene=g, model_top=max(p, key=p.get),
                 model_dist=";".join(f"{c}:{v:.2f}" for c, v in sorted(p.items(), key=lambda kv: -kv[1]) if v >= 0.02),
                 loc_source=inv.loc[g, "loc_source"] if g in inv.index else "", v4_changed=g in chg_genes or k == "CD3E_free")
        refs = {}
        cg = C[(C.gene == g) & (C.cell_line != "Sperm")]
        if len(cg):        # HPA ICC/IF per cell line (sperm excluded): weight = share of imaged lines reporting the term
            cnt = {}
            for locs in cg.drop_duplicates("cell_line").locations.fillna("").astype(str):
                for t in set(locs.split(";")): cnt[t] = cnt.get(t, 0) + 1
            refs["hpa"] = _dist(list(cnt), T2C, [v / cg.cell_line.nunique() for v in cnt.values()])
            r["hpa_lines_locs"] = " | ".join(f"{a}:{b}" for a, b in cg.drop_duplicates("cell_line")[["cell_line", "locations"]].values)
            r["hpa_rel"] = cg.reliability.iloc[0]
        if g in hpa.index:
            r["hpa_main"] = hpa.at[g, "Main location"]; r["hpa_add"] = hpa.at[g, "Additional location"]
            if "hpa" not in refs:
                main = str(hpa.at[g, "Main location"]).split(";"); add = str(hpa.at[g, "Additional location"]).split(";")
                refs["hpa"] = _dist(main + add, T2C, [1.0] * len(main) + [0.3] * len(add))
        cj = C[(C.gene == g) & C.cell_line.str.upper().str.contains("JURKAT")]
        if len(cj):
            locs = ";".join(cj.locations.fillna("").astype(str)).split(";")
            refs["hpa_jurkat"] = _dist(locs, T2C, [1.0] * len(locs)); r["hpa_jurkat"] = ";".join(sorted(set(locs)))
        r["hpa_lines"] = ";".join(sorted(set(C[C.gene == g].cell_line))) if (C.gene == g).any() else ""
        if g in OC.index:
            o = OC.loc[g]; terms, wts = [], []
            for col, wt in (("annotations_grade_3", 1.0), ("annotations_grade_2", 0.6), ("annotations_grade_1", 0.3)):
                if isinstance(o[col], str):
                    for t in o[col].split(";"): terms.append(t); wts.append(wt)
            refs["opencell"] = _dist(terms, OC_MAP, wts)
            r["opencell"] = " | ".join(f"g{3 - i}:{o[c]}" for i, c in enumerate(["annotations_grade_3", "annotations_grade_2",
                                                                                     "annotations_grade_1"]) if isinstance(o[c], str))
        for ref, q in refs.items():
            if not q: continue
            r[f"unsup_{ref}"] = round(sum(v for c, v in p.items() if q.get(c, 0) == 0), 3)    # model signal where ref sees none
            r[f"tvd_{ref}"] = round(0.5 * sum(abs(p.get(c, 0) - q.get(c, 0)) for c in COMPS), 3)
            r[f"missed_{ref}"] = ";".join(c for c, v in q.items() if v >= 0.3 and p.get(c, 0) < 0.05)   # ref compartment the model lacks
        src = str(r["loc_source"])
        r["circular_hpa"] = ("hpa" in src) and not r["v4_changed"]
        indep = [r.get(f"tvd_{x}") for x in ("opencell", "hpa_jurkat") if r.get(f"tvd_{x}") is not None]
        if not r["circular_hpa"] and r.get("tvd_hpa") is not None: indep.append(r["tvd_hpa"])
        r["disagree_indep"] = round(max(indep), 3) if indep else np.nan
        r["informative"] = (not r["circular_hpa"]) or ("hpa_jurkat" in refs) or ("opencell" in refs) or (r.get("tvd_hpa", 0) or 0) >= 0.3
        r["disagree_any"] = round(max([r.get(f"tvd_{x}") for x in ("hpa", "hpa_jurkat", "opencell") if r.get(f"tvd_{x}") is not None] or [np.nan]), 3)
        rows.append(r)
    R = pd.DataFrame(rows).sort_values(["disagree_any", "disagree_indep"], ascending=False)
    R.to_csv(f"{OUT}/fluor_compare.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(R[["protein", "model_dist", "hpa_main", "hpa_jurkat", "opencell", "loc_source", "circular_hpa", "informative",
             "tvd_hpa", "tvd_hpa_jurkat", "tvd_opencell", "disagree_any"]].to_string(index=False))


def hpa_figure():
    """model confocal section next to the HPA JURKAT image (green = target, blue = nucleus, red = microtubules)"""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ed = f"{FD}/hpa_examples"; fs = sorted(glob.glob(f"{ed}/*_JURKAT_*.jpg"))
    if not fs: return
    c3, cell = cell_geom(); S = json.load(open(f"{OUT}/fluor_setup.json"))
    labp = plane_labels(cell, c3, np.array(S["grid_lo"]), np.array(S["grid_n"]), S["z0_nm"], PIX)
    ins = np.nonzero(~np.isin(labp, [c3.EXT, c3.OUTV])); m = int(1000 / PIX)
    sl = (slice(max(ins[0].min() - m, 0), ins[0].max() + m), slice(max(ins[1].min() - m, 0), ins[1].max() + m))
    fig, axs = plt.subplots(2, len(fs), figsize=(3.3 * len(fs), 6.4), facecolor="white")
    for j, f in enumerate(fs):
        g = os.path.basename(f).split("_")[0]
        show(axs[0, j], np.load(f"{CACHE}/img_{g}.npz")["noisy_confocal"][sl], labp[sl], c3, f"model {g}: confocal section")
        im = plt.imread(f); c = im.shape[0] // 2; h = int(0.3 * im.shape[0])
        axs[1, j].imshow(im[c - h:c + h, c - h:c + h]); axs[1, j].set_xticks([]); axs[1, j].set_yticks([])
        L = 10 / 0.08; axs[1, j].plot([20, 20 + L], [2 * h - 25] * 2, color="white", lw=2)
        axs[1, j].set_title(f"HPA JURKAT {g} (green), central crop, 10 µm bar", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fluor_hpa_jurkat.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for s, f in (("collect", collect), ("external", external), ("render", render), ("compare", compare)):
        if step in (s, "all"): log("==", s); f()
    if step == "figures": figures()
    if step in ("compare", "all", "hpa_figure"): hpa_figure()
