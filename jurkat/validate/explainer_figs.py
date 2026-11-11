"""Explainer-page figures for the Jurkat whole-cell model v7 (chromatin v8 + v7 geometry). Read-only on all model files.
usage: NUMBA_NUM_THREADS=16 python explainer_figs.py OUTDIR [fig1 fig2 fig3 fig4 fig5]
Writes JPEGs (<= 1500 px wide) to OUTDIR and a numbers JSON (explainer_numbers.json) next to them.
Caches (fields on grids, depths) go to OUTDIR/../cache."""
import os, sys, json, time
import numpy as np, pandas as pd
import numba as nb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage as ndi
from PIL import Image

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
sys.path.insert(0, f"{V}/jurkat/yc/pipeline")
import cell3d as c3

D = f"{V}/data/derived"; C8 = f"{D}/chromatin/v8"
GEO = f"{D}/jgeom_cell1_v7.npz"; GEO_OLD = f"{D}/jgeom_cell1_v7_emgolgi.npz"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
CACHE = os.path.join(os.path.dirname(os.path.abspath(OUT)), "cache"); os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)
NUMF = os.path.join(OUT, "..", "explainer_numbers_" + "_".join(sys.argv[2:] or ["all"]) + ".json")
NUM = json.load(open(NUMF)) if os.path.exists(NUMF) else {}

COL = dict(nucleus="#2f6fb5", cytosol="#e0703a", mito="#2d8a3e", er="#d4719a", golgi="#5b44a8", endo="#c94545",
           pm="#3a3f46", ne="#d39b1c", nucleolus="#1d3f6e", pol2="#e8710a", ctcf="#1967d2", cohesin="#9334e6")
plt.rcParams.update({"font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 13, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})
HALO = [pe.withStroke(linewidth=3.5, foreground="white")]


def save(fig, name):
    p = os.path.join(OUT, name); tmp = p + ".png"
    fig.savefig(tmp, dpi=100, facecolor="white"); plt.close(fig)
    im = Image.open(tmp).convert("RGB")
    if im.width > 1500:
        im = im.resize((1500, round(im.height * 1500 / im.width)), Image.LANCZOS)
    im.save(p, quality=85, optimize=True); os.remove(tmp); print("wrote", p, im.size, flush=True)


def scalebar(ax, x, y, L, label, lw=4, fs=14, color="k"):
    ax.plot([x, x + L], [y, y], color=color, lw=lw, solid_capstyle="butt")
    ax.text(x + L / 2, y, label, ha="center", va="bottom", fontsize=fs, color=color, path_effects=HALO)


# ----------------------------------------------------------------------------------- geometry helpers
_G = {}
def geo(fn=GEO):
    if fn not in _G:
        _G[fn] = c3.Cell.load(fn)
    return _G[fn]


@nb.njit(parallel=True, cache=True)
def pts_field(P, G, k):
    out = np.empty(P.shape[0], np.float32)
    for i in nb.prange(P.shape[0]):
        out[i] = c3.field_sdf(P[i, 0], P[i, 1], P[i, 2], G, k)
    return out


@nb.njit(parallel=True, cache=True)
def grid_field(G, k, x0, y0, z0, h, nx, ny, nz):
    out = np.empty((nx, ny, nz), np.float32)
    for i in nb.prange(nx):
        for j in range(ny):
            for l in range(nz):
                out[i, j, l] = c3.field_sdf(x0 + i * h, y0 + j * h, z0 + l * h, G, k)
    return out


_DEEP = {}
def depth_below_inm(P):
    """distance below the INM midsurface (nm). The NE field is clamped about 100 nm inside, so: exact field value where
    the depth is < BAND, deeper = BAND + Euclidean distance (40 nm grid, linear interpolation) to the depth-BAND surface."""
    P = np.ascontiguousarray(P, np.float64)
    d = -pts_field(P, geo().G, c3.F_NE) - 0.5 * c3.PERI_W
    BAND = 80.
    if not _DEEP:
        F = fields_grid(40.); dg = -F["ne"] - 0.5 * c3.PERI_W
        _DEEP["edt"] = (ndi.distance_transform_edt(dg > BAND) * float(F["h"])).astype(np.float32)
        _DEEP["lo"] = F["lo"]; _DEEP["h"] = float(F["h"])
    deep = d >= BAND - 1
    if deep.any():
        g = ((P[deep] - _DEEP["lo"]) / _DEEP["h"]).T
        d[deep] = BAND + ndi.map_coordinates(_DEEP["edt"], g, order=1, mode="nearest")
    return d


def plane_labels(o, e1, e2, n1, n2, h, fn=GEO):
    """labels on a plane: point = o + i h e1 + j h e2 (i < n1, j < n2); returns [n2, n1] (row = e2)"""
    i, j = np.meshgrid(np.arange(n1), np.arange(n2))
    P = o[None] + (i.ravel()[:, None] * h) * e1[None] + (j.ravel()[:, None] * h) * e2[None]
    return geo(fn).labels(P).reshape(n2, n1)


def fields_grid(h, fn=GEO, tag="v7"):
    """cell-bbox grid of the main SDF fields (cached)"""
    cf = f"{CACHE}/fields_{tag}_{int(h)}.npz"
    if os.path.exists(cf):
        return dict(np.load(cf))
    G = geo(fn).G
    zf = G[c3.G_ZFACE]
    c = grid_field(G, c3.F_CELL, -12000., -12000., zf, 200., 121, 121, 80)
    idx = np.argwhere(c < 0); lo = np.array([-12000., -12000., zf]) + idx.min(0) * 200 - 200; hi = np.array([-12000., -12000., zf]) + idx.max(0) * 200 + 200
    lo[2] = max(lo[2], zf); n = np.ceil((hi - lo) / h).astype(int) + 1
    out = dict(lo=lo, h=np.float64(h))
    for nm, k in (("cell", c3.F_CELL), ("ne", c3.F_NE), ("mito", c3.F_MITO), ("er", c3.F_ER), ("golgi", c3.F_GOLGI),
                  ("endo", c3.F_ENDO), ("nucleolus", c3.F_NUCLEOLUS)):
        out[nm] = grid_field(G, k, lo[0], lo[1], lo[2], float(h), n[0], n[1], n[2])
        print("field", nm, out[nm].shape, flush=True)
    np.savez(cf, **out)
    return out


def centrosome():
    J = json.load(open(f"{D}/prepass_v7/cytoskeleton_cell1.json"))["log"]["centrioles"]
    z = np.load(f"{D}/prepass_v7/cytoskeleton_cell1.npz")
    return np.array(J["new_centre_nm"]), z["pos"][z["kind"] == 0].astype(np.float64), J


# ----------------------------------------------------------------------------------- chromatin helpers
_N = {}
def nuc():
    if not _N:
        z = np.load(f"{C8}/chromatin_v8_cell1.npz")
        for k in ("pos", "chain", "bp", "state", "chain_names", "chain_nuc_offset", "chain_n_nuc", "func_kind", "func_bp",
                  "func_chain", "func_nuc", "func_gene", "func_state", "func_pos"):
            _N[k] = z[k]
        print("nucleosomes loaded", _N["pos"].shape, flush=True)
    return _N


def chrom_colors():
    """46 chains: 23 hues (homologue a darker, b lighter); hue order spread so neighbours differ"""
    import colorsys
    names = [str(s) for s in nuc()["chain_names"]]
    chroms = [str(i) for i in range(1, 23)] + ["X", "Y"]
    hues = {}
    order = np.argsort((np.arange(24) * 7) % 24)            # step 7/24 around the wheel
    for r, c in enumerate(chroms):
        hues[c] = ((r * 7) % 24) / 24.0
    cols = []
    for nm in names:
        c, hom = nm[:-1], nm[-1]
        l = 0.40 if hom == "a" else 0.62
        s = 0.80 if hom == "a" else 0.70
        cols.append(colorsys.hls_to_rgb(hues[c], l, s))
    return names, np.array(cols)


def nucleosome_depths(step=20):
    cf = f"{CACHE}/nuc_depth_{step}.npz"
    if os.path.exists(cf):
        return dict(np.load(cf))
    N = nuc(); idx = np.arange(0, N["pos"].shape[0], step)
    d = depth_below_inm(N["pos"][idx])
    np.savez(cf, idx=idx, d=d); return dict(idx=idx, d=d)


def nucleus_mask(h=100.):
    F = fields_grid(100.)
    return F, F["ne"] < -0.5 * c3.PERI_W


# =================================================================================== fig 1
def fig1():
    N = nuc(); names, cols = chrom_colors()
    pos = N["pos"]; ch = N["chain"]
    F, M = nucleus_mask()
    lo, h = F["lo"], float(F["h"])
    idx = np.argwhere(M); cen = lo + (idx.min(0) + idx.max(0)) / 2 * h
    z0 = cen[2]
    # radial ordering (every 20th nucleosome)
    dd = nucleosome_depths(); di = dd["idx"]; dep = dd["d"]
    chs = ch[di]
    chrom_of = np.array([n[:-1] for n in names])
    rows = []
    for c in [str(i) for i in range(1, 23)] + ["X", "Y"]:
        m = np.isin(chs, np.where(chrom_of == c)[0])
        rows.append((c, float(np.median(dep[m]) / 1000), float((dep[m] < 500).mean())))
    rows.sort(key=lambda r: r[1])
    NUM["chrom_median_depth_um_sorted_periph_to_interior"] = [(r[0], round(r[1], 2)) for r in rows]
    NUM["chrom_frac_within_0p5um_of_INM"] = {r[0]: round(r[2], 3) for r in rows}
    NUM["nucleosomes_median_depth_um_all"] = round(float(np.median(dep) / 1000), 2)

    fig = plt.figure(figsize=(15, 7.4))
    ax = fig.add_axes([0.0, 0.02, 0.50, 0.96]); ax2 = fig.add_axes([0.50, 0.02, 0.50, 0.96])
    # (a) slab
    m = np.abs(pos[:, 2] - z0) < 50
    P = pos[m]; cc = ch[m]
    rng = np.random.default_rng(0); o = rng.permutation(P.shape[0])
    ax.scatter(P[o, 0] / 1000, P[o, 1] / 1000, s=0.6, c=cols[cc[o]], lw=0, rasterized=True)
    # nuclear outline (INM) on the slab plane
    n1, n2, hh = 700, 520, 20.
    ox = np.array([cen[0] - n1 * hh / 2, cen[1] - n2 * hh / 2, z0])
    i, j = np.meshgrid(np.arange(n1), np.arange(n2))
    PP = np.c_[ox[0] + i.ravel() * hh, ox[1] + j.ravel() * hh, np.full(i.size, z0)]
    ne = pts_field(PP, geo().G, c3.F_NE).reshape(n2, n1)
    X = (ox[0] + np.arange(n1) * hh) / 1000; Y = (ox[1] + np.arange(n2) * hh) / 1000
    ax.contour(X, Y, ne, levels=[0], colors=[COL["ne"]], linewidths=2.5)
    # direct labels at the densest point of each chain in the slab
    bins = 60
    ex = [X[0], X[-1], Y[0], Y[-1]]
    for k in np.unique(cc):
        mk = cc == k
        if mk.sum() < 1500:
            continue
        Hk, xe, ye = np.histogram2d(P[mk, 0] / 1000, P[mk, 1] / 1000, bins=bins, range=[ex[:2], ex[2:]])
        Hk = ndi.gaussian_filter(Hk, 1.5); a, b = np.unravel_index(Hk.argmax(), Hk.shape)
        ax.text((xe[a] + xe[a + 1]) / 2, (ye[b] + ye[b + 1]) / 2, names[k], fontsize=13, ha="center", va="center",
                color="k", fontweight="bold", path_effects=HALO)
    ax.set_aspect("equal"); ax.axis("off")
    jj, ii = np.where(ne < 0)
    ax.set_xlim(X[ii.min()] - 0.25, X[ii.max()] + 0.25); ax.set_ylim(Y[jj.min()] - 1.2, Y[jj.max()] + 1.0)
    scalebar(ax, X[ii.min()], Y[jj.min()] - 0.9, 2.0, "2 µm")
    ax.text(0.02, 0.98, "a", transform=ax.transAxes, fontsize=22, fontweight="bold", va="top")
    ax.text(0.10, 0.98, "100 nm section, all 46 chains", transform=ax.transAxes, fontsize=15, va="top")

    # (b) selected chromosomes, oblique view
    sel = ["18a", "18b", "19a", "19b", "Xa", "Ya"]
    selcol = {"18": "#1b9e77", "19": "#d95f02", "X": "#7570b3", "Y": "#e7298a"}
    el, az = np.deg2rad(28), np.deg2rad(-35)
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    R = Rx @ Rz                                    # rows: screen x, screen y, depth (towards viewer = +)
    R = np.array([R[0], R[2], -R[1]])
    # nucleus silhouette
    vox = lo + idx * h - cen
    Q = vox @ R.T
    gx = np.arange(Q[:, 0].min() - 300, Q[:, 0].max() + 300, 80.); gy = np.arange(Q[:, 1].min() - 300, Q[:, 1].max() + 300, 80.)
    Hs, _, _ = np.histogram2d(Q[:, 0], Q[:, 1], bins=[gx, gy]); Hs = ndi.binary_closing(Hs > 0, iterations=2)
    ax2.contourf((gx[:-1] + 40) / 1000, (gy[:-1] + 40) / 1000, Hs.T.astype(float), levels=[0.5, 1.5], colors=["#eef2f8"])
    ax2.contour((gx[:-1] + 40) / 1000, (gy[:-1] + 40) / 1000, Hs.T.astype(float), levels=[0.5], colors=[COL["ne"]], linewidths=2.5)
    # nucleolus projected (faint)
    Mn = F["nucleolus"] < 0
    vn = lo + np.argwhere(Mn) * h - cen; Qn = vn @ R.T
    Hn, _, _ = np.histogram2d(Qn[:, 0], Qn[:, 1], bins=[gx, gy])
    ax2.contourf((gx[:-1] + 40) / 1000, (gy[:-1] + 40) / 1000, (Hn.T > 0).astype(float), levels=[0.5, 1.5], colors=["#c9d3e3"])
    pts, cl, labs = [], [], []
    for s in sel:
        k = names.index(s); a0 = N["chain_nuc_offset"][k]; n = N["chain_n_nuc"][k]
        p = (pos[a0:a0 + n:8] - cen) @ R.T
        pts.append(p); base = np.array(to_rgb(selcol[s[:-1]]))
        if s.endswith("b"):
            base = 1 - 0.55 * (1 - base)
        cl.append(np.repeat(base[None], p.shape[0], 0)); labs.append((s, p[:, :2].mean(0)))
    pts = np.concatenate(pts); cl = np.concatenate(cl)
    dz = pts[:, 2]; t = (dz - dz.min()) / np.ptp(dz)
    shade = 0.55 + 0.45 * t                          # nearer = brighter
    cl = np.clip(cl * shade[:, None] + (1 - shade[:, None]) * 0.15, 0, 1)
    o = np.argsort(dz)
    ax2.scatter(pts[o, 0] / 1000, pts[o, 1] / 1000, s=1.6, c=cl[o], lw=0, rasterized=True)
    pretty = {"18": "chr18 (gene-poor)", "19": "chr19 (gene-rich)", "X": "chrX", "Y": "chrY"}
    for s, xy in labs:
        ax2.text(xy[0] / 1000, xy[1] / 1000, s, fontsize=15, fontweight="bold", ha="center", va="center", path_effects=HALO)
    hand = [Line2D([], [], marker="o", ls="", ms=11, mfc=selcol[c], mec="none", label=pretty[c]) for c in ("18", "19", "X", "Y")]
    ax2.legend(handles=hand, loc="upper right", bbox_to_anchor=(1.0, 0.90), frameon=False, fontsize=14)
    ax2.set_aspect("equal"); ax2.axis("off")
    xl = ax2.get_xlim(); yl = ax2.get_ylim()
    ax2.text(0.02, 0.98, "b", transform=ax2.transAxes, fontsize=22, fontweight="bold", va="top")
    ax2.text(0.10, 0.98, "whole territories, oblique side view\n(a = darker, b = lighter homologue)", transform=ax2.transAxes, fontsize=15, va="top")
    scalebar(ax2, xl[0] + 0.3, yl[0] + 0.3, 2.0, "2 µm")
    save(fig, "chrom_territories.jpg")


# =================================================================================== fig 2
def fig2():
    N = nuc(); names = [str(s) for s in N["chain_names"]]
    CH, CHR = "11a", "chr11"; ci = names.index(CH)
    LO, HI = 116_800_000, 119_800_000
    a0 = N["chain_nuc_offset"][ci]; n = N["chain_n_nuc"][ci]
    bp = N["bp"][a0:a0 + n].astype(np.int64); P = N["pos"][a0:a0 + n].astype(np.float64)
    m = (bp >= LO) & (bp < HI); bpr = bp[m]; Pr = P[m]
    fk, fb, fc = N["func_kind"], N["func_bp"].astype(np.int64), N["func_chain"]
    fm = (fc == ci) & (fb >= LO) & (fb < HI)
    fpos = N["func_pos"][fm].astype(np.float64); fkind = fk[fm]; fbp = fb[fm]; fgene = N["func_gene"][fm]
    B = pd.read_csv(f"{C8}/genome_bins_10kb.tsv.gz", sep="\t"); B = B[(B.chrom == CHR) & (B.end > LO) & (B.start < HI)]
    CT = pd.read_csv(f"{C8}/ctcf_sites.tsv.gz", sep="\t"); CT = CT[(CT.chrom == CHR) & (CT.start > LO) & (CT.end < HI)]
    GN = pd.read_csv(f"{D}/chromatin/genes.tsv.gz", sep="\t"); EX = pd.read_csv(f"{D}/chromatin/jurkat_expression.tsv.gz", sep="\t")
    GN = GN.merge(EX[["gene_id", "tpm_best"]], on="gene_id", how="left")
    GN = GN[(GN.chrom == CHR) & (GN.end > LO) & (GN.start < HI) & (GN.gene_type == "protein_coding")]
    pol_genes = set(fgene[fkind == 0])
    # numbers
    NUM["fig2_locus"] = dict(chain=CH, start=LO, end=HI, nucleosomes=int(m.sum()), pol2=int((fkind == 0).sum()),
                             ctcf=int((fkind == 1).sum()), cohesin=int(np.isin(fkind, [2, 3]).sum()),
                             pol2_genes=sorted(pol_genes), A_frac_bins=round(float((B.e1 > 0).mean()), 3),
                             end_to_end_nm=round(float(np.linalg.norm(Pr[-1] - Pr[0])), 0),
                             rg_nm=round(float(np.sqrt(((Pr - Pr.mean(0)) ** 2).sum(1).mean())), 0))
    cmap = plt.get_cmap("viridis"); norm = plt.Normalize(LO / 1e6, HI / 1e6)

    fig = plt.figure(figsize=(15, 8.6))
    # ---------------- tracks (left)
    tr = ["position", "A/B (E1)", "LAD", "RNA (TPM)", "CTCF", "cohesin", "genes"]
    hgt = [0.35, 1.0, 0.45, 1.0, 0.7, 0.5, 2.2]
    gs = fig.add_gridspec(len(tr), 1, left=0.105, right=0.47, top=0.95, bottom=0.08, height_ratios=hgt, hspace=0.18)
    axs = [fig.add_subplot(gs[k]) for k in range(len(tr))]
    xs = np.linspace(LO, HI, 400) / 1e6
    axs[0].imshow(xs[None], aspect="auto", cmap=cmap, norm=norm, extent=[LO / 1e6, HI / 1e6, 0, 1])
    mid = ((B.start + B.end) / 2 / 1e6).values
    axs[1].bar(mid, B.e1.clip(lower=0), width=0.01, color="#c0392b", lw=0)
    axs[1].bar(mid, B.e1.clip(upper=0), width=0.01, color="#2c6fbb", lw=0)
    axs[1].axhline(0, color="#888", lw=0.8); axs[1].text(1.01, 1.0, "A", color="#c0392b", fontsize=13, ha="left", va="top", transform=axs[1].transAxes)
    axs[1].text(1.01, 0.0, "B", color="#2c6fbb", fontsize=13, ha="left", va="bottom", transform=axs[1].transAxes)
    axs[2].fill_between(mid, 0, B.lad.values, step="mid", color="#555", lw=0); axs[2].set_ylim(0, 1.05)
    if B.lad.max() == 0:
        axs[2].text(0.5, 0.5, "no LAD in this window", transform=axs[2].transAxes, ha="center", va="center", fontsize=11.5, color="#777")
    g = GN.dropna(subset=["tpm_best"])
    axs[3].vlines(g.tss / 1e6, 0, np.log10(1 + g.tpm_best), color="#444", lw=1.6)
    gp = g[g.gene_name.isin(pol_genes)]
    axs[3].vlines(gp.tss / 1e6, 0, np.log10(1 + gp.tpm_best), color=COL["pol2"], lw=2.2)
    axs[3].set_yticks([0, 1, 2, 3]); axs[3].set_yticklabels(["1", "10", "100", "1k"], fontsize=11)
    axs[4].vlines(CT.start / 1e6, 0, 0.45, color="#bbb", lw=1)
    cb = fbp[fkind == 1] / 1e6
    axs[4].vlines(cb, 0.55, 1.0, color=COL["ctcf"], lw=1.6); axs[4].set_ylim(0, 1.05)
    axs[4].text(LO / 1e6 - 0.03, 0.22, "motif", fontsize=10, ha="right", va="center", color="#888")
    axs[4].text(LO / 1e6 - 0.03, 0.78, "bound", fontsize=10, ha="right", va="center", color=COL["ctcf"])
    axs[5].vlines(fbp[np.isin(fkind, [2, 3])] / 1e6, 0, 1, color=COL["cohesin"], lw=1.6); axs[5].set_ylim(0, 1.05)
    # genes: protein-coding, Pol II-loaded in colour
    gg = GN.sort_values("start"); rowend = []
    for _, r in gg.iterrows():
        s, e = max(r.start, LO) / 1e6, min(r.end, HI) / 1e6
        for k, re_ in enumerate(rowend):
            if s > re_ + 0.08:
                rowend[k] = e; row = k; break
        else:
            rowend.append(e); row = len(rowend) - 1
        on = r.gene_name in pol_genes
        cc = COL["pol2"] if on else "#b8b8b8"
        y = -row
        axs[6].plot([s, e], [y, y], color=cc, lw=4 if on else 2.5, solid_capstyle="butt")
        tss = r.tss / 1e6
        if LO / 1e6 <= tss <= HI / 1e6:
            axs[6].plot([tss], [y], marker=">" if r.strand == "+" else "<", color=cc, ms=6)
        if on and (r.gene_name in ("CD3D", "CD3G") or (r.tpm_best or 0) > 60):
            axs[6].text((s + e) / 2, y + 0.33, r.gene_name, fontsize=10.5, ha="center", va="bottom",
                        fontweight="bold" if r.gene_name.startswith("CD3") else "normal", path_effects=HALO)
    axs[6].set_ylim(-len(rowend) + 0.3, 1.2)
    for k, a in enumerate(axs):
        a.set_xlim(LO / 1e6, HI / 1e6)
        a.set_ylabel(tr[k], rotation=0, ha="right", va="center", fontsize=13.5)
        a.spines["left"].set_visible(False); a.set_yticks([] if k != 3 else a.get_yticks())
        if k < len(axs) - 1:
            a.set_xticks([]); a.spines["bottom"].set_visible(False)
    axs[0].axis("off"); axs[0].text(LO / 1e6 - 0.03, 0.5, "position", ha="right", va="center", fontsize=13.5)
    axs[-1].set_xlabel(f"{CHR} (homologue a), Mb"); axs[-1].spines["left"].set_visible(False)
    axs[6].legend(handles=[Line2D([], [], color=COL["pol2"], lw=4, label="Pol II loaded"),
                           Line2D([], [], color="#b8b8b8", lw=2.5, label="no Pol II")], loc="lower left",
                  bbox_to_anchor=(0, -0.62), ncol=2, frameon=False, fontsize=12)
    # ---------------- 3D path (right), PCA projection
    mu = Pr.mean(0); U, S, Vt = np.linalg.svd(Pr - mu, full_matrices=False)
    E = Vt[:2]
    q = (Pr - mu) @ E.T / 1000; qf = (fpos - mu) @ E.T / 1000
    ax = fig.add_axes([0.51, 0.06, 0.48, 0.90])
    seg = np.stack([q[:-1], q[1:]], 1)
    jump = np.linalg.norm(Pr[1:] - Pr[:-1], axis=1) < 60          # no lines across deletions / long links
    lc = LineCollection(seg[jump], colors=cmap(norm(bpr[:-1][jump] / 1e6)), lw=0.9, alpha=0.85)
    ax.add_collection(lc)
    ax.scatter(q[::3, 0], q[::3, 1], s=1.2, c=cmap(norm(bpr[::3] / 1e6)), lw=0, rasterized=True)
    k0 = fkind == 0; k1 = fkind == 1; k2 = np.isin(fkind, [2, 3])
    ax.scatter(qf[k2, 0], qf[k2, 1], s=70, marker="D", c=COL["cohesin"], edgecolors="white", lw=1.0, zorder=5, label="cohesin")
    ax.scatter(qf[k1, 0], qf[k1, 1], s=80, marker="^", c=COL["ctcf"], edgecolors="white", lw=1.0, zorder=6, label="CTCF")
    ax.scatter(qf[k0, 0], qf[k0, 1], s=95, marker="o", c=COL["pol2"], edgecolors="k", lw=1.0, zorder=7, label="Pol II")
    for gname in ("CD3E", "CD3D", "CD3G"):
        mm = (fgene == gname) & k0
        if mm.any():
            NUM.setdefault("fig2_cd3_pol2", {})[gname] = int(mm.sum())
    ax.set_aspect("equal"); ax.axis("off"); ax.autoscale()
    xl = ax.get_xlim(); yl = ax.get_ylim()
    pad = 0.08 * max(np.ptp(xl), np.ptp(yl))
    ax.set_xlim(xl[0] - pad, xl[1] + pad); ax.set_ylim(yl[0] - pad, yl[1] + pad)
    ax.set_aspect("equal", adjustable="datalim"); ax.apply_aspect(); xl = ax.get_xlim(); yl = ax.get_ylim()
    mcd = np.isin(fgene, ["CD3E", "CD3D", "CD3G"]) & k0
    if mcd.any():
        c_ = qf[mcd].mean(0)
        tx = c_[0]; ty = (q[:, 1].min() + yl[0]) / 2 + 0.06 * np.ptp(yl)
        ax.annotate("CD3D / CD3G\nPol II", xy=c_, xytext=(tx, ty), fontsize=14, fontweight="bold", ha="center",
                    arrowprops=dict(arrowstyle="-", color="k", lw=1.2), path_effects=HALO, zorder=9)
    ax.legend(loc="upper left", frameon=False, fontsize=14, markerscale=1.0, bbox_to_anchor=(0.0, 0.96), ncol=3)
    scalebar(ax, xl[1] - 0.12 * np.ptp(xl) - 0.5, yl[0] + 0.05 * np.ptp(yl), 0.5, "500 nm")
    ax.text(0.0, 1.02, "3D path of the 3 Mb (projected on its main plane)", transform=ax.transAxes, fontsize=14, va="top", color="#333")
    # inset: where it sits in the nucleus (xy projection)
    ia = fig.add_axes([0.52, 0.05, 0.22, 0.22])
    F, M = nucleus_mask(); lo, h = F["lo"], float(F["h"])
    proj = M.any(2).T
    xs_ = (lo[0] + np.arange(M.shape[0]) * h) / 1000; ys_ = (lo[1] + np.arange(M.shape[1]) * h) / 1000
    ia.contourf(xs_, ys_, proj.astype(float), levels=[0.5, 1.5], colors=["#eef2f8"])
    ia.contour(xs_, ys_, proj.astype(float), levels=[0.5], colors=[COL["ne"]], linewidths=1.8)
    Pc = P[::40]
    ia.scatter(Pc[:, 0] / 1000, Pc[:, 1] / 1000, s=0.3, c="#9aa4b1", lw=0, rasterized=True)
    ia.scatter(Pr[::5, 0] / 1000, Pr[::5, 1] / 1000, s=1.0, c=cmap(norm(bpr[::5] / 1e6)), lw=0, rasterized=True)
    ia.set_aspect("equal"); ia.axis("off")
    ia.text(0.5, -0.04, "chr11a in the nucleus (top view)", transform=ia.transAxes, ha="center", va="top", fontsize=11.5)
    save(fig, "chrom_genome_link.jpg")


# =================================================================================== fig 3
def fig3():
    N = nuc(); pos = N["pos"]
    F, M = nucleus_mask(); lo, h = F["lo"], float(F["h"])
    idx = np.argwhere(M); cen = lo + (idx.min(0) + idx.max(0)) / 2 * h; z0 = cen[2]
    T = 200.
    sl = np.abs(pos[:, 2] - z0) < T / 2
    Ps = pos[sl]
    fk = N["func_kind"]; fp = N["func_pos"]
    fs = np.abs(fp[:, 2] - z0) < T / 2
    R = np.load(f"{C8}/prepass/mrna_nascent_cell1.npz")["pts"]
    rs = np.abs(R[:, 2] - z0) < T / 2
    # plane labels (nucleolus, NE)
    hh = 20.; n1, n2 = 680, 520
    o = np.array([cen[0] - n1 * hh / 2, cen[1] - n2 * hh / 2, z0])
    L = plane_labels(o, np.array([1., 0, 0]), np.array([0., 1, 0]), n1, n2, hh)
    X = (o[0] + np.arange(n1) * hh) / 1000; Y = (o[1] + np.arange(n2) * hh) / 1000
    i_, j_ = np.meshgrid(np.arange(n1), np.arange(n2))
    PP = np.c_[o[0] + i_.ravel() * hh, o[1] + j_.ravel() * hh, np.full(i_.size, z0)]
    ne = pts_field(PP, geo().G, c3.F_NE).reshape(n2, n1)

    fig = plt.figure(figsize=(15, 7.6))
    ax = fig.add_axes([0.0, 0.02, 0.60, 0.96])
    Hn, xe, ye = np.histogram2d(Ps[:, 0] / 1000, Ps[:, 1] / 1000, bins=[np.r_[X, X[-1] + hh / 1000], np.r_[Y, Y[-1] + hh / 1000]])
    Hn = ndi.gaussian_filter(Hn.T, 1.0)
    vmax = np.percentile(Hn[Hn > 0], 99)
    ax.imshow(Hn, origin="lower", extent=[X[0], X[-1] + hh / 1000, Y[0], Y[-1] + hh / 1000], cmap="Greys", vmin=0, vmax=vmax * 1.3,
              interpolation="bilinear")
    ax.contourf(X, Y, (L == c3.NUCL).astype(float), levels=[0.5, 1.5], colors=[COL["nucleolus"]], alpha=0.18)
    ax.contour(X, Y, (L == c3.NUCL).astype(float), levels=[0.5], colors=[COL["nucleolus"]], linewidths=1.6, linestyles="--")
    ax.contour(X, Y, ne, levels=[0], colors=[COL["ne"]], linewidths=2.5)
    ax.scatter(R[rs, 0] / 1000, R[rs, 1] / 1000, s=0.25, c=COL["pol2"], alpha=0.5, lw=0, rasterized=True)
    ax.scatter(fp[fs & (fk == 0), 0] / 1000, fp[fs & (fk == 0), 1] / 1000, s=9, c=COL["pol2"], lw=0, rasterized=True)
    ax.scatter(fp[fs & (fk == 1), 0] / 1000, fp[fs & (fk == 1), 1] / 1000, s=9, c=COL["ctcf"], lw=0, rasterized=True)
    ax.scatter(fp[fs & np.isin(fk, [2, 3]), 0] / 1000, fp[fs & np.isin(fk, [2, 3]), 1] / 1000, s=9, c=COL["cohesin"], lw=0, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(X[0], X[-1]); ax.set_ylim(Y[0], Y[-1])
    scalebar(ax, X[0] + 0.4, Y[0] + 0.35, 2.0, "2 µm")
    hand = [Patch(fc="#777", label="nucleosomes (density)"), Line2D([], [], marker="o", ls="", c=COL["pol2"], ms=8, label="Pol II + nascent RNA"),
            Line2D([], [], marker="o", ls="", c=COL["ctcf"], ms=8, label="CTCF"), Line2D([], [], marker="o", ls="", c=COL["cohesin"], ms=8, label="cohesin"),
            Patch(fc=COL["nucleolus"], alpha=0.25, ec=COL["nucleolus"], ls="--", label="nucleolus"), Line2D([], [], c=COL["ne"], lw=2.5, label="nuclear envelope")]
    ax.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=13, ncol=3, columnspacing=1.0, handletextpad=0.4)
    # zoom inset on the periphery: lamina band
    nucp = (L == c3.NUCP)
    # choose a peripheral point: leftmost NE crossing on the centre row
    row = n2 // 2; cols_ = np.where(ne[row] < 0)[0]
    zx, zy = X[cols_[0]], Y[row]
    w = 0.6
    axz = fig.add_axes([0.40, 0.03, 0.20, 0.30])
    mz = (np.abs(Ps[:, 0] / 1000 - zx) < w) & (np.abs(Ps[:, 1] / 1000 - zy) < w)
    axz.scatter(Ps[mz, 0] / 1000, Ps[mz, 1] / 1000, s=1.0, c="#666", lw=0, rasterized=True)
    for k, cc in ((0, COL["pol2"]), (1, COL["ctcf"])):
        mk = fs & (fk == k) & (np.abs(fp[:, 0] / 1000 - zx) < w) & (np.abs(fp[:, 1] / 1000 - zy) < w)
        axz.scatter(fp[mk, 0] / 1000, fp[mk, 1] / 1000, s=22, c=cc, lw=0)
    mk = fs & np.isin(fk, [2, 3]) & (np.abs(fp[:, 0] / 1000 - zx) < w) & (np.abs(fp[:, 1] / 1000 - zy) < w)
    axz.scatter(fp[mk, 0] / 1000, fp[mk, 1] / 1000, s=22, c=COL["cohesin"], lw=0)
    # INM / ONM and the lamina band limit (INM - 16 nm - 10.9 nm)
    axz.contour(X, Y, ne, levels=[-c3.PERI_W / 2, c3.PERI_W / 2], colors=[COL["ne"]], linewidths=1.6)
    axz.contourf(X, Y, ne, levels=[-c3.PERI_W / 2 - 26.9, -c3.PERI_W / 2], colors=["#f3d48a"], alpha=0.8)
    axz.set_xlim(zx - w * 0.7, zx + w * 1.3); axz.set_ylim(zy - w, zy + w); axz.set_aspect("equal")
    axz.set_xticks([]); axz.set_yticks([])
    for s_ in axz.spines.values():
        s_.set_visible(True); s_.set_color("#333")
    scalebar(axz, zx + w * 0.55, zy - w * 0.9, 0.2, "200 nm", fs=12, lw=3)
    axz.text(0.03, 0.97, "lamina band", transform=axz.transAxes, fontsize=12, va="top", color="#8a6410", fontweight="bold", path_effects=HALO)
    ax.add_patch(plt.Rectangle((zx - w * 0.7, zy - w), w * 2, 2 * w, fill=False, ec="#333", lw=1.4))
    # ---------------- radial enrichment panel
    axr = fig.add_axes([0.68, 0.15, 0.30, 0.66])
    rng = np.random.default_rng(1)
    vox = lo + idx * h + rng.uniform(-h / 2, h / 2, idx.shape)
    vox = vox[rng.choice(vox.shape[0], min(1_000_000, vox.shape[0]), replace=False)]
    dv = depth_below_inm(vox); dv = dv[dv > 0]
    dd = nucleosome_depths(); dn = dd["d"]
    dp = {k: depth_below_inm(fp[(fk == k) if k < 2 else np.isin(fk, [2, 3])]) for k in (0, 1, 2)}
    dr = depth_below_inm(R[::10])
    edges = np.r_[0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000]
    vol = np.histogram(dv, edges)[0] / dv.size
    def enr(d):
        return (np.histogram(d, edges)[0] / d.size) / vol
    xc = (edges[:-1] + edges[1:]) / 2 / 1000
    series = [("nucleosomes", dn, "#555555", "-"), ("Pol II", dp[0], COL["pol2"], "-"), ("nascent RNA", dr, COL["pol2"], ":"),
              ("CTCF", dp[1], COL["ctcf"], "-"), ("cohesin", dp[2], COL["cohesin"], "-")]
    for nm, d, cc, ls in series:
        axr.plot(xc, enr(d), color=cc, lw=2.2, ls=ls, marker="o", ms=4, label=nm)
    axr.axhline(1, color="#aaa", lw=1)
    axr.set_xlabel("depth below nuclear envelope (µm)"); axr.set_ylabel("enrichment vs. volume")
    axr.legend(frameon=False, fontsize=12.5, loc="upper right")
    axr.set_xlim(0, 2.45)
    q = lambda d: float(np.median(d) / 1000)
    NUM["fig3_median_depth_um"] = dict(volume=round(q(dv), 2), nucleosomes=round(q(dn), 2), pol2=round(q(dp[0]), 2),
                                       ctcf=round(q(dp[1]), 2), cohesin=round(q(dp[2]), 2), nascent_rna=round(q(dr), 2))
    NUM["fig3_frac_within_500nm"] = {k: round(float((d < 500).mean()), 3) for k, d in
                                     (("volume", dv), ("nucleosomes", dn), ("pol2", dp[0]), ("ctcf", dp[1]), ("cohesin", dp[2]))}
    NUM["fig3_min_depth_nm"] = {k: round(float(d.min()), 1) for k, d in (("nucleosomes_sub", dn), ("pol2", dp[0]), ("ctcf", dp[1]), ("cohesin", dp[2]))}
    NUM["fig3_enrichment_bins_nm"] = edges.tolist()
    NUM["fig3_enrichment"] = {nm: np.round(enr(d), 2).tolist() for nm, d, _, _ in series}
    # A-compartment shares
    st = N["state"]
    pol = fk == 0; fn_ = N["func_nuc"]
    NUM["pol2_frac_on_A_nucleosome"] = round(float((st[fn_[pol]] & 1).astype(bool).mean()), 3)
    NUM["ctcf_frac_on_A_nucleosome"] = round(float((st[fn_[fk == 1]] & 1).astype(bool).mean()), 3)
    NUM["nucleosome_frac_A"] = round(float((st[::50] & 1).astype(bool).mean()), 3)
    di = dd["idx"]; sA = (st[di] & 1).astype(bool); sL = (st[di] & 2).astype(bool)
    NUM["median_depth_um_A_B_LAD"] = dict(A=round(q(dn[sA]), 2), B=round(q(dn[~sA]), 2), LAD=round(q(dn[sL]), 2), nonLAD=round(q(dn[~sL]), 2))
    NUM["counts_func"] = dict(pol2=int(pol.sum()), ctcf=int((fk == 1).sum()), cohesin=int(np.isin(fk, [2, 3]).sum()),
                              nascent_rna_beads=int(R.shape[0]), nucleosomes=int(pos.shape[0]))
    save(fig, "chrom_functional.jpg")


# =================================================================================== fig 4
LABCOL = {c3.CYTO: "#f9e2d6", c3.PERI: COL["ne"], c3.NUCP: "#97b7da", c3.NUCL: COL["nucleolus"], c3.IMS: COL["mito"], c3.MATRIX: COL["mito"],
          c3.ERL: COL["er"], c3.GOLGI: COL["golgi"], c3.GIC: COL["golgi"], c3.ENDO: COL["endo"], c3.PORE: COL["ne"],
          c3.SPK: "#97b7da", c3.EXT: "#ffffff", c3.OUTV: "#ffffff"}


def label_rgb(L):
    img = np.ones(L.shape + (3,))
    for k, c in LABCOL.items():
        img[L == k] = to_rgb(c)
    return img


def render(F, view, cut_nucleus=True, px=40., with_er=False):
    """z-buffer splat render of SDF surfaces (marching-cubes-free): surface voxels + SDF-gradient normals"""
    lo, h = F["lo"], float(F["h"])
    R = view
    light = np.array([-0.35, 0.55, 0.75]); light /= np.linalg.norm(light)
    ccell = lo + np.array(F["cell"].shape) / 2 * h
    layers = []
    mn = F["ne"] < 0; idn = np.argwhere(mn); ncen = lo + (idn.min(0) + idn.max(0)) / 2 * h
    for nm, col in (("cell", COL["pm"]), ("ne", COL["nucleus"]), ("nucleolus", "#1d3f6e"), ("mito", COL["mito"]),
                    ("golgi", COL["golgi"]), ("endo", COL["endo"])) + ((("er", COL["er"]),) if with_er else ()):
        S = F[nm]; ins = S < 0
        if nm == "nucleolus":
            ins &= F["ne"] < -c3.PERI_W
        surf = ins & ~ndi.binary_erosion(ins)
        ii = np.argwhere(surf)
        gx, gy, gz = np.gradient(S)
        nrm = np.c_[gx[surf], gy[surf], gz[surf]]; nrm /= np.linalg.norm(nrm, axis=1)[:, None] + 1e-9
        P = lo + ii * h
        Q = P @ R.T; Nq = nrm @ R.T
        if nm == "cell":
            keep = (Q[:, 2] < (ccell @ R.T)[2])                  # back half of the PM only
            Q, Nq = Q[keep], Nq[keep]
        if nm == "ne" and cut_nucleus:
            keep = Q[:, 2] < (ncen @ R.T)[2] + 600
            Q, Nq = Q[keep], Nq[keep]
        sh = 0.30 + 0.70 * np.abs(Nq @ light)
        base = np.array(to_rgb(col))
        if nm == "cell":
            base = 1 - 0.18 * (1 - base); sh = 0.75 + 0.25 * sh
        rgb = np.clip(base[None] * sh[:, None] + 0.12 * (sh[:, None] ** 8), 0, 1)
        layers.append((Q, rgb))
    Q = np.concatenate([l[0] for l in layers]); RGB = np.concatenate([l[1] for l in layers])
    u0, v0 = Q[:, 0].min() - 200, Q[:, 1].min() - 200
    nu = int((Q[:, 0].max() + 200 - u0) / px) + 1; nv = int((Q[:, 1].max() + 200 - v0) / px) + 1
    img = np.ones((nv, nu, 3)); zb = np.full((nv, nu), -np.inf)
    o = np.argsort(Q[:, 2])
    Q = Q[o]; RGB = RGB[o]
    iu = ((Q[:, 0] - u0) / px).astype(int); iv = ((Q[:, 1] - v0) / px).astype(int)
    for du in (0, 1):
        for dv in (0, 1):
            a = np.clip(iu + du, 0, nu - 1); b = np.clip(iv + dv, 0, nv - 1)
            lin = b * nu + a
            flat = img.reshape(-1, 3); zf = zb.reshape(-1)
            upd = Q[:, 2] >= zf[lin] - 1e-6
            # later (nearer) points overwrite earlier: numpy assigns the last occurrence
            flat[lin[upd]] = RGB[upd]; zf[lin[upd]] = Q[upd, 2]
    return img, (u0, v0, px), R


def view_matrix(el_deg, az_deg):
    el, az = np.deg2rad(el_deg), np.deg2rad(az_deg)
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    R = Rx @ Rz
    return np.array([R[0], R[2], -R[1]])     # screen x, screen y, toward viewer


def mix(c, f):
    """c blended towards white by fraction f"""
    c = np.array(to_rgb(c)); return tuple(c + (1 - c) * f)


def section_ax(a, Lp, ex, star_xy, title, nm):
    a.imshow(label_rgb(Lp), origin="lower", extent=ex, interpolation="nearest")
    inside = ((Lp != c3.EXT) & (Lp != c3.OUTV)).astype(float)
    a.contour(np.linspace(ex[0], ex[1], Lp.shape[1]), np.linspace(ex[2], ex[3], Lp.shape[0]), inside, levels=[0.5],
              colors=[COL["pm"]], linewidths=1.8)
    if star_xy is not None:
        a.plot(*star_xy, marker="*", ms=22, mfc="#ffd400", mec="k", mew=1.3, zorder=6)
    r = np.where(inside.any(1))[0]; c = np.where(inside.any(0))[0]
    y0, y1 = ex[2] + (r[0] - 4) * (ex[3] - ex[2]) / Lp.shape[0], ex[2] + (r[-1] + 4) * (ex[3] - ex[2]) / Lp.shape[0]
    x0, x1 = ex[0] + (c[0] - 4) * (ex[1] - ex[0]) / Lp.shape[1], ex[0] + (c[-1] + 4) * (ex[1] - ex[0]) / Lp.shape[1]
    a.set_xlim(x0, x1); a.set_ylim(y0, y1)
    a.set_aspect("equal"); a.axis("off")
    a.text(0.0, 1.0, nm, transform=a.transAxes, fontsize=22, fontweight="bold", va="bottom")
    a.text(0.07, 1.0, title, transform=a.transAxes, fontsize=14, va="bottom")
    return x0, y0


def fig4():
    F = fields_grid(40.)
    cen, cp, J = centrosome()
    R = view_matrix(32, -30)
    img, (u0, v0, px), _ = render(F, R)
    fig = plt.figure(figsize=(15, 8.6))
    ax = fig.add_axes([0.0, 0.0, 0.46, 1.0])
    ext = [u0 / 1000, (u0 + img.shape[1] * px) / 1000, v0 / 1000, (v0 + img.shape[0] * px) / 1000]
    ax.imshow(img, origin="lower", extent=ext, interpolation="bilinear")
    q = cen @ R.T
    ax.plot(q[0] / 1000, q[1] / 1000, marker="*", ms=26, mfc="#ffd400", mec="k", mew=1.4, zorder=5)
    ax.set_aspect("equal"); ax.axis("off")
    scalebar(ax, ext[0] + 0.5, ext[2] + 0.3, 5.0, "5 µm")
    ax.text(0.02, 0.99, "a", transform=ax.transAxes, fontsize=22, fontweight="bold", va="top")
    ax.text(0.08, 0.99, "3D surfaces; front of PM and nucleus cut away,\nER hidden (see b, c)", transform=ax.transAxes, fontsize=14, va="top")
    lo = F["lo"]; shp = np.array(F["cell"].shape) * float(F["h"])
    hh = 25.
    n1 = int(shp[0] / hh); n2 = int(shp[1] / hh); n3 = int(shp[2] / hh)
    mn = np.argwhere(F["ne"] < 0); ncz = lo[2] + (mn[:, 2].min() + mn[:, 2].max()) / 2 * float(F["h"])
    Lxy = plane_labels(np.array([lo[0], lo[1], ncz]), np.array([1., 0, 0]), np.array([0., 1, 0]), n1, n2, hh)
    Lxz = plane_labels(np.array([lo[0], cen[1], lo[2]]), np.array([1., 0, 0]), np.array([0., 0, 1]), n1, n3, hh)
    a = fig.add_axes([0.48, 0.47, 0.51, 0.48])
    ex = [lo[0] / 1000, (lo[0] + n1 * hh) / 1000, lo[1] / 1000, (lo[1] + n2 * hh) / 1000]
    x0, y0 = section_ax(a, Lxy, ex, None, f"xy section, nucleus mid-height (z = {ncz / 1000:.1f} µm)", "b")
    scalebar(a, x0 + 0.3, y0 + 0.1, 5.0, "5 µm", fs=13)
    a = fig.add_axes([0.48, 0.03, 0.30, 0.38])
    ex = [lo[0] / 1000, (lo[0] + n1 * hh) / 1000, lo[2] / 1000, (lo[2] + n3 * hh) / 1000]
    section_ax(a, Lxz, ex, (cen[0] / 1000, cen[2] / 1000), "xz section through the centrosome", "c")
    hand = [Line2D([], [], color=COL["pm"], lw=2.5, label="plasma membrane"), Patch(fc=LABCOL[c3.CYTO], label="cytosol"),
            Patch(fc=LABCOL[c3.NUCP], label="nucleus"), Patch(fc=COL["nucleolus"], label="nucleolus"), Patch(fc=COL["ne"], label="nuclear envelope"),
            Patch(fc=COL["mito"], label="mitochondria"), Patch(fc=COL["er"], label="ER"), Patch(fc=COL["golgi"], label="Golgi"),
            Patch(fc=COL["endo"], label="endolysosomal"), Line2D([], [], marker="*", ls="", ms=18, mfc="#ffd400", mec="k", label="centrosome")]
    fig.legend(handles=hand, loc="lower left", bbox_to_anchor=(0.785, 0.02), ncol=1, frameon=False, fontsize=13)
    NUM["cell_volumes_um3"] = {k: json.loads(str(np.load(GEO)["summary"]))[k]["vol_mask_um3"] for k in ("cell", "mito", "er", "golgi", "endo", "nucleolus")}
    save(fig, "cell_organelles.jpg")


# =================================================================================== fig 5
def fig5():
    cen, cp, J = centrosome()
    G = geo().G
    # Golgi blob voxels near the centrosome -> plane: through the centrosome, containing the nucleus->centrosome direction
    F = fields_grid(40.); lo, h = F["lo"], float(F["h"])
    gi = np.argwhere(F["golgi"] < 0); gp = lo + gi * h
    near = np.linalg.norm(gp - cen, axis=1) < 2500; gp = gp[near]
    gcen = gp.mean(0)
    mn = np.argwhere(F["ne"] < 0); nucp = lo + mn * h
    # nearest nucleus point to the centrosome
    dn = np.linalg.norm(nucp - cen, axis=1); nn = nucp[dn.argmin()]
    NUM["golgi_v7"] = dict(centrosome_nm=cen.round(0).tolist(), mtoc_candidate_nm=J["mtoc_candidate_nm"],
                           centrosome_to_mtoc_nm=J["dist_to_mtoc_nm"], centrosome_to_nucleus_nm=round(float(dn.min()), 0),
                           golgi_centroid_to_centrosome_nm=round(float(np.linalg.norm(gcen - cen)), 0),
                           golgi_voxels_40nm_within_2p5um=int(near.sum()), golgi_vol_um3_within_2p5um=round(near.sum() * h ** 3 / 1e9, 2))
    e1 = nn - cen; e1 /= np.linalg.norm(e1)
    Cg = np.cov((gp - gcen).T); w, vv = np.linalg.eigh(Cg)
    e2 = vv[:, -1] - (vv[:, -1] @ e1) * e1; e2 /= np.linalg.norm(e2)
    nrm = np.cross(e1, e2)
    W = 4400.; hh = 5.; n = int(W / hh)
    o = cen - e1 * (W * 0.28) - e2 * (W / 2)
    L = plane_labels(o, e1, e2, n, n, hh)
    fig = plt.figure(figsize=(15, 8.3))
    ax = fig.add_axes([0.0, 0.02, 0.60, 0.96])
    ex = [-W * 0.28 / 1000, W * 0.72 / 1000, -W / 2000, W / 2000]
    lab = {c3.CYTO: "#fbf3ee", c3.GOLGI: COL["golgi"], c3.GIC: "#d9d1f0", c3.NUCP: "#c7d8ee", c3.NUCL: "#8fb0d9", c3.SPK: "#a9c1e3",
           c3.PERI: COL["ne"], c3.PORE: COL["ne"], c3.ERL: COL["er"], c3.IMS: COL["mito"], c3.MATRIX: "#6fbf7c", c3.ENDO: COL["endo"],
           c3.EXT: "#ffffff", c3.OUTV: "#ffffff"}
    img = np.ones(L.shape + (3,))
    for k, c in lab.items():
        img[L == k] = to_rgb(c)
    ax.imshow(img, origin="lower", extent=ex, interpolation="nearest")
    # MTs: centerline points within 40 nm of the plane (cross-sections) and a +-300 nm slab (projected, faint)
    cl = np.load(f"{D}/cytoskel/stage3/mt_stage3_b_centerlines.npz"); mp = cl["points_model_nm"].astype(np.float64); off = cl["offsets"]
    rel = mp - cen; a1 = rel @ e1 / 1000; a2 = rel @ e2 / 1000; a3 = rel @ nrm
    inwin = (a1 > ex[0]) & (a1 < ex[1]) & (a2 > ex[2]) & (a2 < ex[3])
    mid = np.searchsorted(off, np.arange(mp.shape[0]), side="right") - 1
    segs = []
    for t in np.unique(mid[inwin & (np.abs(a3) < 300)]):
        s = slice(off[t], off[t + 1]); m_ = (np.abs(a3[s]) < 300) & inwin[s]
        xy = np.c_[a1[s], a2[s]]; br = np.where(~m_)[0]
        # split runs
        runs = np.split(np.arange(m_.size), np.where(np.diff(m_.astype(int)) != 0)[0] + 1)
        for r_ in runs:
            if m_[r_[0]] and r_.size > 2:
                segs.append(xy[r_])
    ax.add_collection(LineCollection(segs, colors="#2b2b2b", lw=1.6, alpha=0.8))
    # centrioles: dimers within 40 nm of the plane, plus faint projection
    rc = cp - cen; c1 = rc @ e1 / 1000; c2 = rc @ e2 / 1000; c3_ = rc @ nrm
    ax.scatter(c1, c2, s=1.5, c="#e6b800", lw=0, alpha=0.35, rasterized=True)
    ax.scatter(c1[np.abs(c3_) < 40], c2[np.abs(c3_) < 40], s=3, c="#8a6d00", lw=0, rasterized=True)
    ax.set_xlim(ex[0], ex[1]); ax.set_ylim(ex[2], ex[3]); ax.set_aspect("equal"); ax.axis("off")
    scalebar(ax, ex[0] + 0.15, ex[2] + 0.15, 1.0, "1 µm")
    # zoom inset on a stack
    gl = np.argwhere(L == c3.GOLGI)
    # pick the Golgi pixel with most lumen pixels in a 300 nm window
    Lg = ndi.uniform_filter((L == c3.GOLGI).astype(float), 60)
    b, a = np.unravel_index(Lg.argmax(), Lg.shape)
    zx = ex[0] + a * hh / 1000; zy = ex[2] + b * hh / 1000; w = 0.22
    axz = fig.add_axes([0.385, 0.56, 0.21, 0.40])
    axz.imshow(img, origin="lower", extent=ex, interpolation="nearest")
    axz.set_xlim(zx - w, zx + w); axz.set_ylim(zy - w, zy + w); axz.set_xticks([]); axz.set_yticks([])
    for s_ in axz.spines.values():
        s_.set_visible(True); s_.set_color("#333"); s_.set_linewidth(1.4)
    scalebar(axz, zx - w + 0.03, zy - w + 0.03, 0.1, "100 nm", fs=12, lw=3)
    ax.add_patch(plt.Rectangle((zx - w, zy - w), 2 * w, 2 * w, fill=False, ec="#333", lw=1.4))
    hand = [Patch(fc=COL["golgi"], label="cisternal lumen"), Patch(fc="#d9d1f0", label="intercisternal space"),
            Patch(fc="#fbf3ee", ec="#ccc", label="cytosol"), Patch(fc="#c7d8ee", label="nucleus"), Patch(fc=COL["ne"], label="nuclear envelope"),
            Patch(fc=COL["er"], label="ER"), Patch(fc=COL["mito"], label="mitochondria"), Patch(fc=COL["endo"], label="endolysosomal"),
            Line2D([], [], color="#2b2b2b", lw=1.6, label="microtubules (±300 nm)"), Line2D([], [], marker="o", ls="", ms=7, mfc="#8a6d00", mec="none", label="centrioles")]
    fig.legend(handles=hand, loc="lower left", bbox_to_anchor=(0.615, 0.01), frameon=False, fontsize=12.5, ncol=2, columnspacing=1.0)
    ax.annotate("centrioles", xy=(0, 0), xytext=(-0.55, 0.45), fontsize=13, fontweight="bold", path_effects=HALO,
                arrowprops=dict(arrowstyle="-", lw=1.2, color="k"))
    # ---------------- side panel: v6 blob vs v7 Golgi (side view, y projection)
    a2x = fig.add_axes([0.62, 0.36, 0.37, 0.60])
    Fo = fields_grid(80., GEO_OLD, "emgolgi"); Fn = fields_grid(80.)
    lo8, h8 = Fn["lo"], float(Fn["h"])
    xs = (lo8[0] + np.arange(Fn["cell"].shape[0]) * h8) / 1000; zs = (lo8[2] + np.arange(Fn["cell"].shape[2]) * h8) / 1000
    pj = lambda m: m.any(1).T.astype(float)
    cellp = pj(Fn["cell"] < 0); nucp_ = pj(Fn["ne"] < 0)
    a2x.contourf(xs, zs, cellp, levels=[0.5, 1.5], colors=["#fbf3ee"]); a2x.contour(xs, zs, cellp, levels=[0.5], colors=[COL["pm"]], linewidths=1.8)
    a2x.contourf(xs, zs, nucp_, levels=[0.5, 1.5], colors=["#c7d8ee"]); a2x.contour(xs, zs, nucp_, levels=[0.5], colors=[COL["ne"]], linewidths=1.8)
    go = pj(Fo["golgi"] < 0); gn = pj(Fn["golgi"] < 0)
    a2x.contourf(xs, zs, go, levels=[0.5, 1.5], colors=["#b9b9b9"]); a2x.contour(xs, zs, go, levels=[0.5], colors=["#666"], linewidths=1.2)
    a2x.contourf(xs, zs, gn, levels=[0.5, 1.5], colors=[COL["golgi"]])
    a2x.plot(cen[0] / 1000, cen[2] / 1000, marker="*", ms=22, mfc="#ffd400", mec="k", mew=1.2, zorder=6)
    ogc = lo8 + np.argwhere(Fo["golgi"] < 0).mean(0) * h8
    a2x.annotate("", xy=(cen[0] / 1000, cen[2] / 1000), xytext=(ogc[0] / 1000, ogc[2] / 1000),
                 arrowprops=dict(arrowstyle="->", lw=2, color="k", shrinkA=14, shrinkB=14))
    a2x.text(ogc[0] / 1000 + 1.0, ogc[2] / 1000, "v6 Golgi blob\n(projected)", ha="left", va="center", fontsize=13, path_effects=HALO)
    a2x.text(cen[0] / 1000 + 1.6, cen[2] / 1000, "v7 Golgi\naround the\ncentrosome", ha="left", va="center", fontsize=13,
             color=COL["golgi"], fontweight="bold", path_effects=HALO)
    a2x.set_aspect("equal"); a2x.axis("off")
    xl = a2x.get_xlim(); yl = a2x.get_ylim()
    scalebar(a2x, xl[0] + 0.3, yl[0] + 0.3, 5.0, "5 µm", fs=13)
    a2x.text(0.0, 1.0, "side view (y projection)", transform=a2x.transAxes, fontsize=14, va="bottom")
    NUM["golgi_v6_blob_centroid_to_centrosome_nm"] = round(float(np.linalg.norm(ogc - cen)), 0)
    # lumen / GIC share in the section, and cisternae count along a line
    gin = np.isin(L, [c3.GOLGI, c3.GIC])
    NUM["golgi_section_lumen_frac"] = round(float((L[gin] == c3.GOLGI).mean()), 3)
    save(fig, "golgi.jpg")


if __name__ == "__main__":
    want = sys.argv[2:] or ["fig1", "fig2", "fig3", "fig4", "fig5"]
    for f in want:
        t = time.time(); globals()[f](); print(f, "done", round(time.time() - t, 1), "s", flush=True)
        json.dump(NUM, open(NUMF, "w"), indent=1)
