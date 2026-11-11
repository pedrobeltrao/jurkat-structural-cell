"""Multiscale zoom strips for the explainer page of the Jurkat whole-cell model v7 (read-only on all model files).

usage: NUMBA_NUM_THREADS=16 python explainer_zoom.py OUTDIR [npc ribosome chromatin ruler]
Writes zoom_npc.jpg, zoom_ribosome.jpg, zoom_chromatin.jpg, scale_bar_ruler.jpg (<= 1600 px wide) and
zoom_numbers.json (box sizes, tile ids, counts, ids used in the captions) to OUTDIR/..

Data (all from the packed v7 model):
  data/derived/wc/tiles/<j>.npz           packed instances (types -> wc/libs.json, pos nm, quat; region = repeat(job_region, job_placed))
  data/derived/wc/prepass_*.npy           pre-pass instances bucketed per tile (types -> wc/prepass_libs.json)
  data/derived/wc/mrna_pts.npy            mRNA beads bucketed per tile
  jurkat/yc/pipeline/lib/<id>.npz         library: residue beads (CA / P, nm, library frame), sphere tree
  data/derived/jgeom_cell1_v7.npz         membranes (SDFs) and pores
  data/derived/chromatin/v8/chromatin_v8_cell1.npz   genome mapping (chain, bp) of nucleosomes and the functional layer
Pose convention (pack3d.quat_mat): world = M(q) @ x_lib + pos; checked against the NPC pre-pass (M e_z = pore normal).
Rendering: numba z-buffer sphere splatting with Lambert + specular shading, depth-difference ambient occlusion, depth cue
and outlines; 2x supersampled."""
import os, sys, json, time
import numpy as np, pandas as pd
import numba as nb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch, Circle
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
import matplotlib.patheffects as pe
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from PIL import Image

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
sys.path.insert(0, f"{V}/jurkat/yc/pipeline")
import cell3d as c3

D = f"{V}/data/derived"; WC = f"{D}/wc"; LIBD = f"{V}/jurkat/yc/pipeline/lib"; C8 = f"{D}/chromatin/v8"
PDBD = f"{V}/data/structures/pdb/assemblies"
GEO = f"{D}/jgeom_cell1_v7.npz"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(os.path.dirname(os.path.abspath(OUT)), "cache_zoom"); os.makedirs(CACHE, exist_ok=True)
NUMF = os.path.join(os.path.dirname(os.path.abspath(OUT)), "zoom_numbers.json")
NUM = json.load(open(NUMF)) if os.path.exists(NUMF) else {}
HALO = [pe.withStroke(linewidth=3, foreground="white")]

# ----------------------------------------------------------------------------------- colours (shared by all figures)
C = dict(ribo="#d9a21b", ribo_p="#efd27a", npc="#159f94", nuc="#b9c6d8", dna="#5c6d84", h1="#6c7d97",
         pol2="#e8710a", ctcf="#1967d2", cohesin="#8e3fd6", lamina="#93a86a", mt="#9cb8a4", actin="#b7a6c6",
         receptor="#e0736b", mrna="#c62828", nascent="#d9480f", mem="#c8c8c8",
         cyto="#f2d3b9", nucl="#c9dfcb", lumen="#efc3d2", matrix="#d9e6ad", tm="#cdbfe3")
REG_CYTO = {"cytosol", "mt_lattice", "mt_plus", "pcm", "vim_cage", "sh_PM_cyto", "sh_PM_cortex", "sh_ER_cyto", "sh_GOLGI_cyto",
            "sh_ENDO_cyto", "sh_OM_cyto", "sh_NPC", "sh_PM_periph", "sh_ENDO_periph", "sh_GOLGI_periph"}
REG_NUC = {"nucleoplasm", "nucleolus", "speckle", "sh_NE_lamina", "sh_NE_nuc"}
REG_LUM = {"ER_lumen", "golgi_lumen", "endo_lumen", "perinuclear", "sh_ER_lumen", "sh_GOLGI_lumen", "sh_ENDO_lumen", "sh_PM_exo"}
REG_MEM = {"PM", "NE", "ER", "MITO_OM", "MITO_IM", "GOLGI", "ENDO"}
SECT = {c3.CYTO: "#fbefe5", c3.PERI: "#bdbdbd", c3.NUCP: "#e3e9f1", c3.NUCL: "#b8c4d4", c3.IMS: "#d6e8c9", c3.MATRIX: "#d6e8c9",
        c3.ERL: "#f4d3df", c3.GOLGI: "#e2d6ef", c3.GIC: "#fbefe5", c3.ENDO: "#f2cccc", c3.PORE: "#e3e9f1",
        c3.SPK: "#d5dde9", c3.EXT: "#ffffff", c3.OUTV: "#ffffff"}
RGB = {k: np.array(to_rgb(v)) for k, v in C.items()}
MEM_MESO = np.array(to_rgb("#efefef"))


def save(fig, name, width=1600):
    p = os.path.join(OUT, name); tmp = p + ".png"
    fig.savefig(tmp, dpi=100, facecolor="white"); plt.close(fig)
    im = Image.open(tmp).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(p, quality=86, optimize=True); os.remove(tmp); print("wrote", p, im.size, flush=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------- model access
_S = {}
def S():
    if not _S:
        L = json.load(open(f"{WC}/libs.json"))
        _S["libs"] = np.array(L["libs"], dtype=object); _S["rnames"] = L["rnames"]
        _S["plibs"] = np.array(json.load(open(f"{WC}/prepass_libs.json")), dtype=object)
        M = pd.read_parquet(f"{WC}/tiles.parquet"); meta = json.load(open(f"{WC}/tiles_meta.json"))
        _S["lo0"] = np.array(meta["lo0"]); _S["T"] = float(meta["TILE"])
        _S["key"] = {tuple(t): i for i, t in enumerate(M[["ti", "tj", "tk"]].values)}
        for k in ("types", "pos", "quat", "off"):
            _S["p" + k] = np.load(f"{WC}/prepass_{k}.npy", mmap_mode="r")
        _S["mpts"] = np.load(f"{WC}/mrna_pts.npy", mmap_mode="r"); _S["moff"] = np.load(f"{WC}/mrna_off.npy")
        _S["cell"] = c3.Cell.load(GEO)
        _S["pores"] = np.load(GEO)["pores"].reshape(-1, 6)
    return _S


def tiles_for(lo, hi):
    s = S(); a = np.floor((lo - s["lo0"]) / s["T"]).astype(int); b = np.floor((hi - s["lo0"]) / s["T"]).astype(int)
    out = []
    for i in range(a[0], b[0] + 1):
        for j in range(a[1], b[1] + 1):
            for k in range(a[2], b[2] + 1):
                if (i, j, k) in s["key"]:
                    out.append(s["key"][(i, j, k)])
    return sorted(out)


def corners(c, F, half):
    sg = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    return c[None] + (sg * half[None]) @ F


def gather(c, F, half, margin=80.0, with_tiles=True):
    """instances whose centre lies in the oriented box (centre c, rows of F = local axes, half sizes) + margin.
    Returns dict: name (object), region (object), pos, quat, src ('t'/'p'), q (local coords), tiles, mrna beads (local)."""
    s = S(); P = corners(c, F, half + margin); lo, hi = P.min(0), P.max(0)
    tl = tiles_for(lo, hi)
    N, R, X, Q, SRC = [], [], [], [], []
    for j in tl:
        if with_tiles:
            z = np.load(f"{WC}/tiles/{j}.npz")
            t = z["types"]; reg = np.repeat(z["job_region"], z["job_placed"])
            q = (z["pos"] - c) @ F.T; m = np.all(np.abs(q) <= half + margin, 1)
            N.append(s["libs"][t[m]]); R.append(np.array(s["rnames"], dtype=object)[reg[m]]); X.append(z["pos"][m]); Q.append(z["quat"][m])
            SRC.append(np.full(m.sum(), "t", dtype=object))
        a, b = s["poff"][j], s["poff"][j + 1]
        if b > a:
            pp = np.asarray(s["ppos"][a:b], np.float64); q = (pp - c) @ F.T; m = np.all(np.abs(q) <= half + margin, 1)
            nm = s["plibs"][np.asarray(s["ptypes"][a:b])[m]]
            N.append(nm); R.append(np.full(m.sum(), "prepass", dtype=object)); X.append(pp[m]); Q.append(np.asarray(s["pquat"][a:b])[m])
            SRC.append(np.full(m.sum(), "p", dtype=object))
    mb = []
    for j in tl:
        a, b = s["moff"][j], s["moff"][j + 1]
        if b > a:
            pp = np.asarray(s["mpts"][a:b], np.float64); q = (pp - c) @ F.T; m = np.all(np.abs(q) <= half + 5, 1)
            mb.append(q[m])
    out = dict(name=np.concatenate(N), region=np.concatenate(R), pos=np.concatenate(X).astype(np.float64),
               quat=np.concatenate(Q).astype(np.float64), src=np.concatenate(SRC), tiles=tl,
               mrna=np.concatenate(mb) if mb else np.zeros((0, 3)))
    out["q"] = (out["pos"] - c) @ F.T
    return out


_L = {}
def lib(name):
    if name not in _L:
        z = np.load(f"{LIBD}/{name}.npz")
        d = dict(beads=z["beads"].astype(np.float64), kind=z["kind"].astype(np.int8),
                 st_c=(z["st_c30"] if "st_c30" in z.files else z["st_c"]).astype(np.float64),
                 st_r=(z["st_r30"] if "st_r30" in z.files else z["st_r"]).astype(np.float64))
        d["src"] = str(z["src"]) if "src" in z.files else "afdb"
        d["memb"] = str(z["membrane"]) if "membrane" in z.files else ""
        d["plddt_kept"] = float(z["plddt_frac_kept"]) if "plddt_frac_kept" in z.files else 1.0
        _L[name] = d
    return _L[name]


def qmats(q):
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    M = np.empty((len(q), 3, 3))
    M[:, 0, 0] = a*a+b*b-c*c-d*d; M[:, 0, 1] = 2*(b*c-a*d); M[:, 0, 2] = 2*(b*d+a*c)
    M[:, 1, 0] = 2*(b*c+a*d); M[:, 1, 1] = a*a-b*b+c*c-d*d; M[:, 1, 2] = 2*(c*d-a*b)
    M[:, 2, 0] = 2*(b*d-a*c); M[:, 2, 1] = 2*(c*d+a*b); M[:, 2, 2] = a*a-b*b-c*c+d*d
    return M


def expand(G, sel, c, F, what="beads"):
    """world -> local coordinates of beads (or sphere-tree spheres) of the selected instances.
    Returns xyz (n,3 local), kind (n), inst (n: index into G)."""
    idx = np.where(sel)[0]
    names = G["name"][idx]
    XYZ, K, I, RR = [], [], [], []
    for nm in np.unique(names):
        ii = idx[names == nm]; L = lib(nm)
        if what == "beads":
            b = L["beads"]; k = L["kind"]; r = np.zeros(len(b))
        else:
            b = L["st_c"]; k = np.zeros(len(b), np.int8); r = L["st_r"]
        M = qmats(G["quat"][ii])
        W = np.einsum("nij,bj->nbi", M, b) + G["pos"][ii][:, None, :]
        XYZ.append(((W - c) @ F.T).reshape(-1, 3)); K.append(np.tile(k, len(ii))); I.append(np.repeat(ii, len(b)))
        RR.append(np.tile(r, len(ii)))
    if not XYZ:
        return np.zeros((0, 3)), np.zeros(0, np.int8), np.zeros(0, int), np.zeros(0)
    return np.concatenate(XYZ), np.concatenate(K), np.concatenate(I), np.concatenate(RR)


RECEPT = ("KPNB1", "KPNA", "XPO", "TNPO", "IPO", "CSE1L", "NUTF2", "RAN", "NXF1", "NXT1")


def inst_color(name, region):
    """per-instance base colour (rgb)"""
    if name == "RIBO80S": return RGB["ribo"]
    if name == "NPC": return RGB["npc"]
    if name == "NUCLEOSOME": return RGB["nuc"]
    if name.startswith("H1-"): return RGB["h1"]
    if name == "CPX-2387": return RGB["pol2"]
    if name == "CTCF": return RGB["ctcf"]
    if name in ("CPX-5989", "CPX-5991"): return RGB["cohesin"]
    if name in ("LMNA", "LMNB1", "LMNB2"): return RGB["lamina"]
    if name.startswith("MT_"): return RGB["mt"]
    if name == "F_ACTB": return RGB["actin"]
    if region == "sh_NPC_fg" or (name.startswith(RECEPT) and region in ("sh_NPC_fg",)): return RGB["receptor"]
    if region in REG_MEM: return RGB["tm"]
    if region in REG_NUC: return RGB["nucl"]
    if region in REG_LUM: return RGB["lumen"]
    if region == "matrix": return RGB["matrix"]
    return RGB["cyto"]


LAB2REG = {c3.CYTO: "cytosol", c3.GIC: "cytosol", c3.PORE: "cytosol", c3.NUCP: "nucleoplasm", c3.NUCL: "nucleoplasm", c3.SPK: "nucleoplasm",
           c3.ERL: "ER_lumen", c3.GOLGI: "ER_lumen", c3.ENDO: "ER_lumen", c3.PERI: "ER_lumen", c3.MATRIX: "matrix", c3.IMS: "ER_lumen",
           c3.EXT: "cytosol", c3.OUTV: "cytosol"}


def bead_colors(G, inst, kind):
    """soluble proteins are coloured by the compartment their centre lies in (labels of the v7 geometry), membrane-region
    instances lilac, NPC-channel (sh_NPC_fg) instances coral, named complexes by name"""
    names = G["name"]
    lab = S()["cell"].labels(G["pos"]) if len(names) else np.zeros(0, np.int8)
    regs = np.array([r if (r in REG_MEM or r in ("sh_NPC_fg", "prepass")) else LAB2REG.get(int(l), "cytosol")
                     for r, l in zip(G["region"], lab)], dtype=object)
    base = np.zeros((len(G["name"]), 3))
    key = np.char.add(names.astype(str), "|" + np.array(regs, dtype=str)) if len(names) else np.zeros(0, str)
    uk, inv = np.unique(key, return_inverse=True)
    for i, k in enumerate(uk):
        n, r = k.split("|", 1); base[inv == i] = inst_color(n, r)
    col = base[inst].copy()
    nm = names[inst]
    col[(nm == "RIBO80S") & (kind == 0)] = RGB["ribo_p"]
    col[(nm == "NUCLEOSOME") & (kind == 1)] = RGB["dna"]
    return col


def membrane_voxels(c, F, half, h):
    """grid points of the oriented box lying inside a bilayer (|SDF| < leaflet half-thickness) -> local coords"""
    s = S()
    ax = [np.arange(-half[i], half[i] + 1e-6, h) for i in range(3)]
    gx, gy, gz = np.meshgrid(*ax, indexing="ij")
    Ql = np.c_[gx.ravel(), gy.ravel(), gz.ravel()]
    out = []
    for a in range(0, len(Ql), 4_000_000):
        q = Ql[a:a + 4_000_000]; Pw = c + q @ F
        g = s["cell"].mems(Pw)[:, :7]
        m = np.any(np.abs(g) < c3.MEM_HALF[None, :7] + 0.25 * h, 1)
        out.append(q[m])
    return np.concatenate(out)


# ----------------------------------------------------------------------------------- renderer
@nb.njit(cache=True)
def _splat(x, y, z, r, cid, x0, y1, px, zb, nrm, cidx):
    H, W = zb.shape
    for i in range(x.shape[0]):
        rr = r[i]
        cu = (x[i] - x0) / px; cv = (y1 - y[i]) / px; rp = rr / px
        u0 = max(int(cu - rp), 0); u1 = min(int(cu + rp) + 1, W - 1)
        v0 = max(int(cv - rp), 0); v1 = min(int(cv + rp) + 1, H - 1)
        r2 = rr * rr
        for v in range(v0, v1 + 1):
            dy = (v + 0.5 - cv) * px
            for u in range(u0, u1 + 1):
                dx = (u + 0.5 - cu) * px
                d2 = dx * dx + dy * dy
                if d2 < r2:
                    h = np.sqrt(r2 - d2); zz = z[i] + h
                    if zz > zb[v, u]:
                        zb[v, u] = zz
                        nrm[v, u, 0] = dx / rr; nrm[v, u, 1] = -dy / rr; nrm[v, u, 2] = h / rr
                        cidx[v, u] = cid[i]


def render(xyz, rad, col, extent, npx, fog=0.55, ao=0.5, outline=None, zrange=None, spec=0.18, amb=0.42):
    """xyz local (x right, y up, z toward viewer), rad (n), col (n,3); extent (x0, x1, y0, y1) in nm; npx = width px.
    Returns float rgb image (H, W, 3) and the depth buffer."""
    x0, x1, y0, y1 = extent; px = (x1 - x0) / npx; H = int(round((y1 - y0) / px))
    zb = np.full((H, npx), -1e30); nrm = np.zeros((H, npx, 3)); cidx = np.full((H, npx), -1, np.int64)
    if len(xyz):
        ucol, cid = np.unique(np.round(col, 4), axis=0, return_inverse=True)
        _splat(xyz[:, 0].astype(np.float64), xyz[:, 1].astype(np.float64), xyz[:, 2].astype(np.float64),
               np.asarray(rad, np.float64), cid.ravel().astype(np.int64), float(x0), float(y1), float(px), zb, nrm, cidx)
    else:
        ucol = np.zeros((1, 3))
    bg = zb < -1e29
    img = np.ones((H, npx, 3))
    if bg.all():
        return img, zb
    base = ucol[np.maximum(cidx, 0)]
    Lv = np.array([-0.45, 0.55, 0.70]); Lv /= np.linalg.norm(Lv); Hv = Lv + np.array([0, 0, 1.0]); Hv /= np.linalg.norm(Hv)
    diff = np.clip(nrm @ Lv, 0, 1); sp = np.clip(nrm @ Hv, 0, 1) ** 28
    sh = amb + (1 - amb) * diff
    rgb = base * sh[..., None] + spec * sp[..., None]
    z = zb.copy(); zf = z[~bg]
    zmax = np.percentile(zf, 99.5); zmin = np.percentile(zf, 0.5) if zrange is None else zmax - zrange
    z[bg] = zmin - 0.1 * (zmax - zmin)
    if ao > 0:
        sig = max(2.0, 0.012 * npx)
        bl = ndi.gaussian_filter(z, sig)
        occ = np.clip((bl - z) / (0.06 * (zmax - zmin) + 1e-9), 0, 1)
        rgb *= (1 - ao * occ)[..., None]
    f = np.clip((zmax - z) / (zmax - zmin + 1e-9), 0, 1) * fog
    rgb = rgb * (1 - f[..., None]) + f[..., None]
    if outline is not None:
        e = np.zeros_like(bg)
        for sx, sy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            zs = np.roll(np.roll(z, sx, 1), sy, 0)
            e |= (zs - z) > outline
        rgb[e & ~bg] *= 0.45
    rgb[bg] = 1.0
    return np.clip(rgb, 0, 1), zb


def downs(img, f=2):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((im.width // f, im.height // f), Image.LANCZOS)) / 255.0


# ----------------------------------------------------------------------------------- whole-cell section
def section(c, F, h=25.0, half=9500.0):
    n = int(2 * half / h)
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    loc = np.c_[(i.ravel() + 0.5) * h - half, half - (j.ravel() + 0.5) * h, np.zeros(n * n)]
    Lb = S()["cell"].labels(c + loc @ F).reshape(n, n)
    img = np.ones((n, n, 3))
    for k, v in SECT.items():
        img[Lb == k] = to_rgb(v)
    # membranes drawn as lines only for the PM, NE and mitochondria (ER / Golgi / endosomes are filled lumens)
    grp = np.zeros_like(Lb)
    grp[np.isin(Lb, [c3.EXT, c3.OUTV])] = 1
    grp[np.isin(Lb, [c3.NUCP, c3.NUCL, c3.SPK, c3.PERI, c3.PORE])] = 2
    grp[np.isin(Lb, [c3.IMS, c3.MATRIX])] = 3
    edge = np.zeros_like(Lb, bool)
    for sx, sy in ((1, 0), (0, 1)):
        gs = np.roll(np.roll(grp, sx, 1), sy, 0)
        edge |= (gs != grp)
    edge = ndi.binary_dilation(edge, iterations=1)
    img[edge] = to_rgb("#8f8f8f")
    inside = ~np.isin(Lb, [c3.EXT, c3.OUTV])
    rr = np.where(inside.any(1))[0]; cc = np.where(inside.any(0))[0]
    return img, (-half, half, -half, half), Lb, (cc[0] * h - half, cc[-1] * h - half, half - rr[-1] * h, half - rr[0] * h)


# ----------------------------------------------------------------------------------- figure layout
def nice_bar(width_nm, frac=0.28):
    t = width_nm * frac; e = 10 ** np.floor(np.log10(t))
    for m in (5, 2, 1):
        if m * e <= t:
            L = m * e; break
    lab = (f"{L/1000:g} µm" if L >= 1000 else (f"{L:g} nm" if L >= 1 else f"{L*10:g} Å"))
    return L, lab


def scalebar(ax, ext, color="k", pos="br"):
    x0, x1, y0, y1 = ext; L, lab = nice_bar(x1 - x0)
    pad = 0.06 * (x1 - x0)
    xs = x1 - pad - L if pos[1] == "r" else x0 + pad
    ys = y0 + 0.06 * (y1 - y0) if pos[0] == "b" else y1 - 0.13 * (y1 - y0)
    ax.plot([xs, xs + L], [ys, ys], color=color, lw=3.2, solid_capstyle="butt", zorder=10)
    ax.text(xs + L / 2, ys + 0.025 * (y1 - y0), lab, ha="center", va="bottom", fontsize=10.5, color=color,
            path_effects=HALO, zorder=11)


def strip(panels, titles, boxes, fname, legend=None, notes=None, height=3.72, big=2):
    """panels: list of (img, extent); boxes[i] = (x0, x1, y0, y1) region of panel i enlarged in panel i+1 (or None)"""
    n = len(panels)
    fig = plt.figure(figsize=(16, height))
    gap = 0.032; left = 0.008; w = (1 - 2 * left - (n - 1) * gap) / n
    hfr = w * 16 / height
    bottom = 0.15 if legend else 0.06
    axs = []
    for i, (img, ext) in enumerate(panels):
        np.savez_compressed(f"{CACHE}/{fname[:-4]}_p{i}.npz", img=np.asarray(img, np.float32), ext=np.array(ext, float))
        ax = fig.add_axes([left + i * (w + gap), bottom, w, hfr])
        ax.imshow(img, extent=ext, origin="upper", interpolation="lanczos")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#9a9a9a"); sp.set_linewidth(0.8)
        ax.set_title(titles[i], fontsize=12, pad=4, color="#333")
        pos = "br"
        if i < len(boxes) and boxes[i] is not None:
            bx = 0.5 * (boxes[i][0] + boxes[i][1]); by = 0.5 * (boxes[i][2] + boxes[i][3])
            fx = (bx - ext[0]) / (ext[1] - ext[0]); fy = (by - ext[2]) / (ext[3] - ext[2])
            if fx > 0.6 and fy < 0.4:
                pos = "bl"
        scalebar(ax, ext, pos=pos)
        if notes and notes[i]:
            for nt in notes[i]:
                if nt[0] == "arrow":
                    _, x, y, t, col, dx, dy = nt
                    ax.annotate(t, xy=(x, y), xytext=(x + dx, y + dy), textcoords="data", fontsize=9.5, color=col, fontweight="bold",
                                ha="center", va="center", path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                                arrowprops=dict(arrowstyle="-", color=col, lw=1.3, shrinkA=2, shrinkB=1), zorder=15)
                else:
                    x, y, t, kw = nt
                    ax.text(x, y, t, transform=ax.transAxes, path_effects=HALO, zorder=12, **kw)
        axs.append(ax)
    for i, b in enumerate(boxes):
        if b is None or i + 1 >= n:
            continue
        bx0, bx1, by0, by1 = b
        axs[i].add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec="#222", lw=1.3, zorder=9))
        for (xa, ya, xb, yb) in ((bx1, by1, 0, 1), (bx1, by0, 0, 0)):
            cp = ConnectionPatch(xyA=(xa, ya), coordsA="data", axesA=axs[i], xyB=(xb, yb), coordsB="axes fraction",
                                 axesB=axs[i + 1], color="#555", lw=0.9, ls=(0, (3, 2)), zorder=20)
            fig.add_artist(cp)
    if big is not None:
        img, ext = panels[big]
        f2 = plt.figure(figsize=(6.4, 6.4)); a2 = f2.add_axes([0, 0, 1, 1])
        a2.imshow(img, extent=ext, origin="upper", interpolation="lanczos"); a2.set_xlim(ext[0], ext[1]); a2.set_ylim(ext[2], ext[3])
        a2.axis("off"); scalebar(a2, ext)
        if notes and notes[big]:
            for nt in notes[big]:
                if nt[0] == "arrow":
                    _, x, y, t, col, dx, dy = nt
                    a2.annotate(t, xy=(x, y), xytext=(x + dx, y + dy), fontsize=12, color=col, fontweight="bold", ha="center", va="center",
                                path_effects=[pe.withStroke(linewidth=3, foreground="white")], arrowprops=dict(arrowstyle="-", color=col, lw=1.5))
                else:
                    x, y, t, kw = nt; kw = dict(kw); kw["fontsize"] = 12
                    a2.text(x, y, t, transform=a2.transAxes, path_effects=HALO, **kw)
        save(f2, fname[:-4] + "_context.jpg", width=640)
    if legend:
        hs = [Line2D([], [], marker="o", ls="", ms=9, mfc=c, mec="#555", mew=0.5, label=l) for l, c in legend]
        fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False, fontsize=10.5, handletextpad=0.25,
                   columnspacing=1.1, bbox_to_anchor=(0.5, 0.0))
    save(fig, fname)
    return axs


# ----------------------------------------------------------------------------------- generic panels
def panel_meso(c, F, L, T, keep, npx=640, mem_h=4.0, extra=None, mrna=True, mrna_col=None):
    """mesoscale slab (L x L x T): sphere trees of selected classes (keep(name) -> bool) + membranes + mRNA beads"""
    half = np.array([L / 2, L / 2, T / 2])
    G = gather(c, F, half, margin=60)
    sel = np.array([keep(n) for n in G["name"]], bool) if len(G["name"]) else np.zeros(0, bool)
    xyz, k, inst, r = expand(G, sel, c, F, "st")
    col = np.array([inst_color(n, rg) for n, rg in zip(G["name"], G["region"])]) if len(G["name"]) else np.zeros((0, 3))
    col = col[inst]
    m = np.all(np.abs(xyz) <= half + 0.0, 1)
    xyz, r, col = xyz[m], r[m], col[m]
    mv = membrane_voxels(c, F, half, mem_h)
    X = [xyz, mv]; Rr = [r, np.full(len(mv), 0.6 * mem_h)]; Cc = [col, np.tile(MEM_MESO, (len(mv), 1))]
    if mrna and len(G["mrna"]):
        mb = G["mrna"][np.all(np.abs(G["mrna"]) <= half, 1)]
        X.append(mb); Rr.append(np.full(len(mb), 2.0)); Cc.append(np.tile(RGB["mrna"] if mrna_col is None else mrna_col, (len(mb), 1)))
    if extra:
        for (xx, rr, cc) in extra:
            mm = np.all(np.abs(xx) <= half, 1); X.append(xx[mm]); Rr.append(rr[mm]); Cc.append(cc[mm])
    X = np.concatenate(X); Rr = np.concatenate(Rr); Cc = np.concatenate(Cc)
    img, _ = render(X, Rr, Cc, (-L / 2, L / 2, -L / 2, L / 2), 2 * npx, fog=0.5, ao=0.45, outline=12.0, zrange=T)
    counts = pd.Series(G["name"][sel]).value_counts().to_dict()
    return downs(img), (-L / 2, L / 2, -L / 2, L / 2), dict(tiles=G["tiles"], counts={k: int(v) for k, v in counts.items()},
                                                             membrane_voxels=int(len(mv)))


def panel_context(c, F, L, T, zclip_bg, hl=None, npx=640, mem_h=1.0, extra=None, nascent=None, zclip_hl=None, outline=2.5, ao=0.55, recolor=None):
    """full context box: every instance in the slab at residue-bead level (CA 0.55 nm / P 0.75 nm), membranes (1 nm grid),
    mRNA beads (1 nm). Background beads are clipped to z <= zclip_bg; instances flagged by hl(G) -> bool array are drawn whole."""
    half = np.array([L / 2, L / 2, T / 2])
    G = gather(c, F, half, margin=70)
    inbox = np.all(np.abs(G["q"]) <= half + 70, 1)
    xyz, kind, inst, _ = expand(G, inbox, c, F, "beads")
    col = bead_colors(G, inst, kind)
    if recolor is not None:
        col = recolor(G, inst, kind, col)
    r = np.where(kind == 1, 0.78, 0.58)
    H = hl(G) if hl is not None else np.zeros(len(G["name"]), bool)
    ish = H[inst]
    m_bg = (~ish) & np.all(np.abs(xyz[:, :2]) <= half[:2], 1) & (xyz[:, 2] <= zclip_bg) & (xyz[:, 2] >= -half[2])
    m_hl = ish & np.all(np.abs(xyz[:, :2]) <= half[:2], 1)
    if zclip_hl is not None:
        m_hl &= xyz[:, 2] <= zclip_hl
    keep = m_bg | m_hl
    mv = membrane_voxels(c, F, half, mem_h); mv = mv[mv[:, 2] <= zclip_bg]
    X = [xyz[keep], mv]; Rr = [r[keep], np.full(len(mv), 0.95 * mem_h)]; Cc = [col[keep], np.tile(RGB["mem"], (len(mv), 1))]
    mb = G["mrna"]; mb = mb[np.all(np.abs(mb[:, :2]) <= half[:2], 1) & (mb[:, 2] >= -half[2]) & (mb[:, 2] <= max(zclip_bg, half[2] if hl else zclip_bg))]
    X.append(mb); Rr.append(np.full(len(mb), 1.0)); Cc.append(np.tile(RGB["mrna"], (len(mb), 1)))
    if extra:
        for (xx, rr, cc) in extra:
            X.append(xx); Rr.append(rr); Cc.append(cc)
    X = np.concatenate(X); Rr = np.concatenate(Rr); Cc = np.concatenate(Cc)
    img, _ = render(X, Rr, Cc, (-L / 2, L / 2, -L / 2, L / 2), 2 * npx, fog=0.45, ao=ao, outline=outline, zrange=T)
    vis = np.unique(inst[keep])
    info = dict(tiles=G["tiles"], n_instances=int(len(vis)), n_species=int(len(np.unique(G["name"][vis]))),
                n_beads=int(keep.sum()), n_membrane_voxels=int(len(mv)), n_mrna_beads=int(len(mb)),
                counts_named={k: int(v) for k, v in pd.Series(G["name"][vis]).value_counts().items()
                              if k in ("RIBO80S", "NPC", "NUCLEOSOME", "CPX-2387", "CTCF", "CPX-5989", "CPX-5991", "LMNB1", "LMNB2", "LMNA",
                                       "MT_TUBA1B-TUBB4B", "F_ACTB") or str(k).startswith("H1-")},
                regions={k: int(v) for k, v in pd.Series(G["region"][vis]).value_counts().items()})
    return downs(img), (-L / 2, L / 2, -L / 2, L / 2), G, info


def verify(G, c, F, tag):
    """pose checks: membrane-embedded instances sit on their membrane; soluble instance centres in their compartment"""
    s = S(); res = {}
    mreg = {"PM": c3.M_PM, "NE": c3.M_NE, "ER": c3.M_ER, "MITO_OM": c3.M_OM, "MITO_IM": c3.M_IM, "GOLGI": c3.M_GOLGI, "ENDO": c3.M_ENDO}
    m = np.isin(G["region"].astype(str), list(mreg))
    if m.any():
        g = s["cell"].mems(G["pos"][m]); mi = np.array([mreg[r] for r in G["region"][m]])
        d = np.abs(g[np.arange(m.sum()), mi])
        res["tm_centre_to_membrane_nm_median"] = round(float(np.median(d)), 2); res["tm_n"] = int(m.sum())
        # membrane normal vs library +z
        P = G["pos"][m]; eps = 1.0; gr = np.zeros((len(P), 3))
        for a in range(3):
            dp = np.zeros(3); dp[a] = eps
            gr[:, a] = (s["cell"].mems(P + dp)[np.arange(len(P)), mi] - s["cell"].mems(P - dp)[np.arange(len(P)), mi]) / (2 * eps)
        gr /= np.linalg.norm(gr, axis=1)[:, None] + 1e-9
        z = qmats(G["quat"][m])[:, :, 2]
        res["tm_abs_cos_libz_normal_median"] = round(float(np.median(np.abs((z * gr).sum(1)))), 3)
    lab = s["cell"].labels(G["pos"])
    for rg, want in (("cytosol", [c3.CYTO, c3.GIC]), ("nucleoplasm", [c3.NUCP, c3.SPK]), ("ER_lumen", [c3.ERL])):
        mm = G["region"] == rg
        if mm.sum():
            res[f"{rg}_centre_label_ok"] = round(float(np.isin(lab[mm], want).mean()), 3)
    for nm, want in (("NUCLEOSOME", [c3.NUCP, c3.SPK, c3.NUCL]), ("RIBO80S", [c3.CYTO, c3.GIC])):
        mm = G["name"] == nm
        if mm.sum():
            res[f"{nm}_centre_label_ok"] = round(float(np.isin(lab[mm], want).mean()), 3)
    log("verify", tag, res)
    return res


# ----------------------------------------------------------------------------------- atomic structures
VDW = {"C": 0.17, "N": 0.155, "O": 0.152, "S": 0.18, "P": 0.18}
CPK = {"N": "#3050f8", "O": "#e8262b", "S": "#e6c229", "P": "#ff8a00"}


def read_atoms(pdb):
    import gemmi
    st = gemmi.read_structure(f"{PDBD}/{pdb}-assembly1.cif.gz"); st.remove_hydrogens(); st.remove_ligands_and_waters()
    st.setup_entities()
    doc = gemmi.cif.read(f"{PDBD}/{pdb}-assembly1.cif.gz"); b = doc.sole_block()
    ed = dict(zip(b.find_values("_entity.id"), b.find_values("_entity.pdbx_description")))
    sub2ent = {}
    for e in st.entities:
        for sc in e.subchains:
            sub2ent[sc] = e.name
    X, E, ENT, CH, RN, RS, AN, POLY = [], [], [], [], [], [], [], []
    for ch in st[0]:
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            na = bool(info and info.is_nucleic_acid())
            ent = sub2ent.get(res.subchain, "?")
            for a in res:
                X.append((a.pos.x / 10, a.pos.y / 10, a.pos.z / 10)); E.append(a.element.name); ENT.append(ent)
                CH.append(ch.name); RN.append(res.name); RS.append(res.seqid.num); AN.append(a.name); POLY.append(na)
    d = dict(xyz=np.array(X), el=np.array(E), ent=np.array(ENT), chain=np.array(CH), resn=np.array(RN), resi=np.array(RS),
             atom=np.array(AN), na=np.array(POLY))
    d["desc"] = {k: str(v).strip("'\"") for k, v in ed.items()}
    return d


def view_frame(A, tilt_deg=0.0, axis=None):
    """rows: screen x, y, toward viewer. axis = direction that should point up (y) before tilting"""
    X = A - A.mean(0)
    w, Vv = np.linalg.eigh(np.cov(X.T))
    up = Vv[:, 0] if axis is None else axis / np.linalg.norm(axis)
    e1 = Vv[:, 2] - (Vv[:, 2] @ up) * up; e1 /= np.linalg.norm(e1)
    n = np.cross(e1, up)
    F = np.array([e1, up, n])
    t = np.deg2rad(tilt_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return Rx @ F


def ball_stick(A, el, cols, F, c, L, npx=640, faded=None, rball=0.05, rbond=0.024, core=None):
    """ball-and-stick: atoms (rball) + bonds (< 0.19 nm) sampled as sphere chains"""
    q = (A - c) @ F.T
    tr = cKDTree(A); pairs = np.array(sorted(tr.query_pairs(0.19)))
    X = [q]; R = [np.full(len(q), rball)]; Cc = [cols]
    if len(pairs):
        t = np.linspace(0, 1, 9)[1:-1]
        for a, b in pairs:
            seg = q[a][None] + t[:, None] * (q[b] - q[a])[None]
            X.append(seg); R.append(np.full(len(t), rbond))
            Cc.append(np.where((t < 0.5)[:, None], cols[a][None], cols[b][None]))
    X = np.concatenate(X); R = np.concatenate(R); Cc = np.concatenate(Cc)
    if core is not None:                       # atoms outside the highlighted residues: thinner sticks, paler
        isc = np.r_[core, np.repeat(core[pairs[:, 0]] & core[pairs[:, 1]], 7) if len(pairs) else np.zeros(0, bool)]
        R = np.where(isc, R, R * 0.6); Cc = np.where(isc[:, None], Cc, 0.45 * Cc + 0.55)
    if faded is not None:
        X = np.concatenate([X, faded[0]]); R = np.concatenate([R, faded[1]]); Cc = np.concatenate([Cc, faded[2]])
    img, _ = render(X, R, Cc, (-L / 2, L / 2, -L / 2, L / 2), 2 * npx, fog=0.45, ao=0.25, outline=0.06, zrange=L * 0.9, spec=0.35, amb=0.5)
    return downs(img), (-L / 2, L / 2, -L / 2, L / 2)


def residues_near(A, mask, d):
    """mask of whole residues having an atom within d nm of the atoms in mask"""
    key = np.char.add(A["chain"].astype(str), np.char.add(":", A["resi"].astype(str)))
    if d <= 0:
        return np.isin(key, np.unique(key[mask]))
    c = A["xyz"][mask].mean(0); pre = np.linalg.norm(A["xyz"] - c, axis=1) < 3.0
    dd, _ = cKDTree(A["xyz"][mask]).query(A["xyz"][pre])
    hit = np.zeros(len(key), bool); hit[np.where(pre)[0][dd < d]] = True
    return np.isin(key, np.unique(key[hit])) & pre | mask


def atom_cols(el, carbon_rgb):
    cols = np.tile(np.asarray(carbon_rgb, float), (len(el), 1))
    for e, h in CPK.items():
        cols[el == e] = to_rgb(h)
    return cols


def spacefill(A, el, cols, F, c, L, npx=640, outline=0.25, fog=0.4):
    q = (A - c) @ F.T
    rad = np.full(len(el), 0.17)
    for e, r in VDW.items():
        rad[el == e] = r
    img, zb = render(q, rad, cols, (-L / 2, L / 2, -L / 2, L / 2), 2 * npx, fog=fog, ao=0.55, outline=outline, spec=0.15)
    return downs(img), (-L / 2, L / 2, -L / 2, L / 2)


# ======================================================================================= figure 1: nuclear pore
def pick_pore():
    P = S()["pores"]; best = None
    for k in range(len(P)):
        c, nrm = P[k, :3], P[k, 3:] / np.linalg.norm(P[k, 3:])
        t1 = np.cross(nrm, [1, 0, 0]); t1 = t1 if np.linalg.norm(t1) > 0.1 else np.cross(nrm, [0, 1, 0]); t1 /= np.linalg.norm(t1)
        t2 = np.cross(nrm, t1)
        d = P[:, :3] - c
        for th in np.deg2rad(np.arange(0, 180, 6)):
            e1 = np.cos(th) * t1 + np.sin(th) * t2; n = np.cross(e1, nrm)
            m = (np.abs(d @ n) < 45) & (np.abs(d @ e1) < 950) & (np.abs(d @ nrm) < 400)
            sc = int(m.sum())
            if best is None or sc > best[0]:
                best = (sc, k, np.array([e1, nrm, n]))
    return best


def fig_npc():
    s = S(); sc, k, F = pick_pore(); c = s["pores"][k, :3].copy()
    log("pore", k, "pores near plane", sc)
    NUM["npc"] = dict(pore_index=int(k), centre_nm=[round(float(v), 1) for v in c], frame_rows=np.round(F, 4).tolist(),
                      pores_in_2um_section=sc)
    # panel 1: cell section + pores near the plane
    img1, ext1, Lb, crop = section(c, F)
    d = (s["pores"][:, :3] - c) @ F.T; m = np.abs(d[:, 2]) < 150
    cx = 0.5 * (crop[0] + crop[1]); cy = 0.5 * (crop[2] + crop[3]); Lc = max(crop[1] - crop[0], crop[3] - crop[2]) + 600
    e1x = (cx - Lc / 2, cx + Lc / 2, cy - Lc / 2, cy + Lc / 2)
    p1 = panel_from_section(img1, ext1, e1x, dots=(d[m, :2], C["npc"], 3.2))
    # panel 2: 2 µm NE region (sphere trees of NPC, ribosomes, nucleosomes, lamina + membranes)
    L2, T2 = 2000.0, 90.0
    keep2 = lambda n: n in ("NPC", "RIBO80S", "NUCLEOSOME", "LMNB1", "LMNB2", "LMNA", "MT_TUBA1B-TUBB4B", "F_ACTB")
    img2, ext2, inf2 = panel_meso(c, F, L2, T2, keep2, mem_h=4.0)
    log("panel2", inf2["counts"])
    # panel 3: 300 nm context box, everything at residue level, section through the pore axis
    L3, T3 = 300.0, 80.0
    img3, ext3, G3, inf3 = panel_context(c, F, L3, T3, zclip_bg=0.0)
    inf3["verify"] = verify(G3, c, F, "npc")
    log("panel3", inf3)
    # panel 4: NPC 7R5K
    A = read_atoms("7r5k")
    ents = sorted(set(A["ent"]), key=lambda x: int(x) if x.isdigit() else 999)
    pal = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
    pal = np.r_[pal, plt.get_cmap("Set2")(np.arange(8))[:, :3]]
    pal = 0.75 * pal + 0.25                                    # muted
    ecol = {e: pal[i % len(pal)] for i, e in enumerate(ents)}
    cols4 = np.array([ecol[e] for e in A["ent"]])
    ctr = A["xyz"].mean(0)
    w, Vv = np.linalg.eigh(np.cov((A["xyz"][::50] - ctr).T)); ax_ring = Vv[:, 0]
    F4 = view_frame(A["xyz"][::50], tilt_deg=-35, axis=ax_ring)
    L4 = 135.0
    qq_ = (A["xyz"][::20] - ctr); rad_ = np.linalg.norm(qq_ - np.outer(qq_ @ ax_ring, ax_ring), axis=1)
    NUM["sizes_nm"] = NUM.get("sizes_nm", {}); NUM["sizes_nm"]["NPC_7R5K_diameter"] = round(float(2 * np.percentile(rad_, 99.9)), 1)
    NUM["sizes_nm"]["NPC_7R5K_height"] = round(float(np.ptp(qq_ @ ax_ring)), 1)
    img4, ext4 = spacefill(A["xyz"], A["el"], cols4, F4, ctr, L4, outline=0.6, fog=0.35)
    # panel 5: a few residues of a nucleoporin (an FG-free beta-propeller/alpha-solenoid contact) -> pick Nup93 if present
    desc = A["desc"]; tgt = None
    for e in ents:
        if "93" in desc.get(e, "") and "Nup" in desc.get(e, "") or desc.get(e, "").upper().startswith("NUCLEAR PORE COMPLEX PROTEIN NUP93"):
            tgt = e; break
    if tgt is None:
        tgt = ents[0]
    mm = (A["ent"] == tgt)
    ch0 = A["chain"][mm][0]; mc = mm & (A["chain"] == ch0)
    rs = np.unique(A["resi"][mc]); r0 = rs[len(rs) // 2]
    # residues r0..r0+3 of that chain plus atoms within 0.45 nm
    core = mc & (A["resi"] >= r0) & (A["resi"] <= r0 + 3)
    cc = A["xyz"][core].mean(0)
    near = residues_near(A, core, 0.45)
    tr = cKDTree(A["xyz"][near]); shell = near.copy()
    Fz = view_frame(A["xyz"][core]); L5 = 3.0
    colsC = atom_cols(A["el"], ecol[tgt])
    sel = near; L5 = 2.4
    img5, ext5 = ball_stick(A["xyz"][sel], A["el"][sel], colsC[sel], Fz, cc, L5, core=core[sel])
    rn = A["resn"][core]; ri = A["resi"][core]
    resl = [f"{a}{b}" for a, b in dict.fromkeys(zip(rn, ri))]
    # boxes
    q4 = (cc - ctr) @ F4.T
    bx = 6.0
    NUM["npc"].update(panel1_section_extent_nm=[round(float(v)) for v in e1x], panel2=dict(L=L2, T=T2, **inf2),
                      panel3=dict(L=L3, T=T3, **inf3), pdb="7R5K", pdb_atoms=int(len(A["el"])), pdb_entities=len(ents),
                      closeup=dict(entity=tgt, desc=desc.get(tgt, ""), chain=str(ch0), residues=resl, atoms=int(sel.sum())))
    nb3 = 70.0
    strip([p1, (img2, ext2), (img3, ext3), (img4, ext4), (img5, ext5)],
          ["cell (section)", "nuclear envelope", "NPC in context", "NPC (PDB 7R5K)", "nucleoporin atoms"],
          [(-L2 / 2, L2 / 2, -L2 / 2, L2 / 2), (-L3 / 2, L3 / 2, -L3 / 2, L3 / 2), (-nb3, nb3, -nb3, nb3),
           (q4[0] - bx, q4[0] + bx, q4[1] - bx, q4[1] + bx)],
          "zoom_npc.jpg",
          legend=[("NPC", C["npc"]), ("transport receptors", C["receptor"]), ("ribosome", C["ribo"]), ("nucleosome", C["nuc"]),
                  ("lamina", C["lamina"]), ("membrane", C["mem"]), ("cytosolic proteins", C["cyto"]), ("nuclear proteins", C["nucl"]),
                  ("membrane proteins", C["tm"]), ("mRNA", C["mrna"])],
          notes=[None, [(0.03, 0.93, "cytoplasm", dict(fontsize=9.5, color="#555")), (0.03, 0.03, "nucleus", dict(fontsize=9.5, color="#555"))],
                 None, None, [(0.03, 0.93, "Nup93 " + " ".join(resl[:4]), dict(fontsize=8.6, color="#333"))]])


def panel_from_section(img, ext, crop, dots=None):
    """crop the section image and burn in dots (local xy nm, colour, radius px at 2x)"""
    x0, x1, y0, y1 = ext; n = img.shape[0]; h = (x1 - x0) / n
    a0 = int((crop[0] - x0) / h); a1 = int((crop[1] - x0) / h); b0 = int((y1 - crop[3]) / h); b1 = int((y1 - crop[2]) / h)
    pad = lambda v, lo, hi: min(max(v, lo), hi)
    sub = np.ones((b1 - b0, a1 - a0, 3))
    sa0, sa1, sb0, sb1 = pad(a0, 0, n), pad(a1, 0, n), pad(b0, 0, n), pad(b1, 0, n)
    sub[sb0 - b0:sb1 - b0, sa0 - a0:sa1 - a0] = img[sb0:sb1, sa0:sa1]
    if dots is not None:
        xy, col, rp = dots
        H, W = sub.shape[:2]; yy, xx = np.mgrid[0:H, 0:W]
        cu = (xy[:, 0] - crop[0]) / h; cv = (crop[3] - xy[:, 1]) / h
        for u, v in zip(cu, cv):
            m = (xx - u) ** 2 + (yy - v) ** 2 <= rp ** 2
            sub[m] = to_rgb(col)
    return sub, crop


# ======================================================================================= figure 2: ribosomes
def pick_polysome():
    z = np.load(f"{D}/prepass_v7/polysomes_cell1.npz")
    ch = z["chain"]; pos = z["pos"].astype(np.float64); er = z["er"]
    o = np.argsort(ch, kind="stable"); chs = ch[o]
    u, st, cnt = np.unique(chs, return_index=True, return_counts=True)
    cand = []
    rng = np.random.default_rng(3)
    ok = np.where((cnt >= 9) & (cnt <= 16))[0]
    rng.shuffle(ok)
    for i in ok[:6000]:
        P = pos[o[st[i]:st[i] + cnt[i]]]
        X = P - P.mean(0); w, Vv = np.linalg.eigh(np.cov(X.T))
        ext = np.ptp(X @ Vv, 0)
        if ext[2] > 160 or ext[1] > 150 or ext[0] > 170:
            continue
        cand.append((np.sqrt(w[0]) / np.sqrt(w[2]), int(u[i]), P.mean(0), Vv, int(cnt[i]), bool(er[o[st[i]]])))
    cand.sort(key=lambda t: t[0])
    cand = cand[:400]
    cen = np.array([t[2] for t in cand])
    g = S()["cell"].mems(cen)
    best = None
    for t, gg in zip(cand, g):
        der = abs(gg[c3.M_ER])
        sc = t[0] + (0 if (60 < der < 250 and not t[5]) else 1.0)
        if best is None or sc < best[0]:
            best = (sc, t, der)
    _, (flat, cid, cen, Vv, n, isER), der = best
    m = ch == cid
    return cid, cen, Vv, n, isER, der, pos[m], flat


def fig_ribosome():
    s = S()
    cid, c, Vv, nrib, isER, der, Pch, flat = pick_polysome()
    F = np.array([Vv[:, 2], Vv[:, 1], Vv[:, 0]])
    if np.linalg.det(F) < 0:
        F[2] *= -1
    eps = 1.0; gr = np.array([(s["cell"].mems((c + eps * e)[None])[0, c3.M_ER] - s["cell"].mems((c - eps * e)[None])[0, c3.M_ER]) / 2
                              for e in np.eye(3)])
    if gr @ F[2] < 0:                                    # ER sheet behind the polysome (viewed from the cytosol)
        F[2] *= -1; F[0] *= -1
    log("polysome", cid, nrib, "ER-bound", isER, "dist ER", der, "flatness", flat)
    NUM["ribosome"] = dict(polysome_chain=int(cid), ribosomes_in_chain=int(nrib), er_bound=isER, centre_nm=[round(float(v), 1) for v in c],
                           dist_to_er_nm=round(float(der), 1))
    zp = np.load(f"{D}/prepass_v7/polysomes_cell1.npz")
    img1, ext1, Lb, crop = section(c, F)
    q = (zp["pos"][::4].astype(np.float64) - c) @ F.T; m = np.abs(q[:, 2]) < 60
    cx = 0.5 * (crop[0] + crop[1]); cy = 0.5 * (crop[2] + crop[3]); Lc = max(crop[1] - crop[0], crop[3] - crop[2]) + 600
    e1x = (cx - Lc / 2, cx + Lc / 2, cy - Lc / 2, cy + Lc / 2)
    p1 = panel_from_section(img1, ext1, e1x, dots=(q[m, :2], C["ribo"], 1.0))
    L2, T2 = 1600.0, 80.0
    keep2 = lambda n: n in ("RIBO80S", "NPC", "MT_TUBA1B-TUBB4B", "F_ACTB", "NUCLEOSOME")
    img2, ext2, inf2 = panel_meso(c, F, L2, T2, keep2, mem_h=4.0)
    log("panel2", inf2["counts"])
    L3, T3 = 200.0, 70.0
    chpos = Pch

    def hl(G):
        h = np.zeros(len(G["name"]), bool)
        r = G["name"] == "RIBO80S"
        if r.any():
            d, _ = cKDTree(chpos).query(G["pos"][r])
            h[np.where(r)[0][d < 0.5]] = True
        return h
    img3, ext3, G3, inf3 = panel_context(c, F, L3, T3, zclip_bg=-14.0, hl=hl)
    inf3["verify"] = verify(G3, c, F, "ribosome")
    inf3["chain_ribosomes_drawn"] = int(hl(G3).sum())
    log("panel3", inf3)
    A = read_atoms("6qzp")
    isrna = A["na"]
    cols = np.where(isrna[:, None], RGB["ribo"][None], RGB["ribo_p"][None])
    ctr = A["xyz"].mean(0)
    F4 = view_frame(A["xyz"][::20], tilt_deg=20)
    L4 = 34.0
    w_, V_ = np.linalg.eigh(np.cov((A["xyz"][::5] - ctr).T)); pr_ = (A["xyz"][::5] - ctr) @ V_
    NUM.setdefault("sizes_nm", {})["ribosome_6QZP_extent"] = [round(float(v), 1) for v in np.ptp(pr_, 0)[::-1]]
    img4, ext4 = spacefill(A["xyz"], A["el"], cols, F4, ctr, L4, outline=0.25)
    # close-up: 28S rRNA (longest RNA chain): 4 nucleotides at the peptidyl-transferase centre region -> use residues closest to
    # the subunit-interface centre if no annotation; we choose the 28S nucleotides nearest the centroid of the whole particle
    rna_ch = pd.Series(A["chain"][isrna]).value_counts().index[0]
    mc = isrna & (A["chain"] == rna_ch)
    P_atoms = A["atom"] == "P"
    cand = mc & P_atoms
    dd = np.linalg.norm(A["xyz"][cand] - ctr, axis=1)
    rmid = int(A["resi"][cand][np.argmin(dd)])
    core = mc & (A["resi"] >= rmid - 1) & (A["resi"] <= rmid + 2)
    cc = A["xyz"][core].mean(0)
    sel = residues_near(A, core, 0.4)
    colsC = atom_cols(A["el"], RGB["ribo"]); colsC[~isrna & (A["el"] == "C")] = RGB["ribo_p"]
    Fz = view_frame(A["xyz"][core]); L5 = 2.6
    img5, ext5 = ball_stick(A["xyz"][sel], A["el"][sel], colsC[sel], Fz, cc, L5, core=core[sel])
    resl = [f"{a}{b}" for a, b in dict.fromkeys(zip(A["resn"][core], A["resi"][core]))]
    q4 = (cc - ctr) @ F4.T; bx = 1.6
    # box around one chain ribosome in panel 3
    qr = (chpos - c) @ F.T; j = np.argmin(np.linalg.norm(qr[:, :2], axis=1)); b3 = 17.0
    NUM["ribosome"].update(panel1_section_extent_nm=[round(float(v)) for v in e1x], panel2=dict(L=L2, T=T2, **inf2),
                           panel3=dict(L=L3, T=T3, **inf3), pdb="6QZP", pdb_atoms=int(len(A["el"])),
                           closeup=dict(chain=str(rna_ch), desc=A["desc"].get(A["ent"][mc][0], ""), residues=resl, atoms=int(sel.sum())),
                           rna_atoms=int(isrna.sum()), protein_atoms=int((~isrna).sum()))
    strip([p1, (img2, ext2), (img3, ext3), (img4, ext4), (img5, ext5)],
          ["cell (section)", "cytoplasm", "polysome in context", "80S ribosome (PDB 6QZP)", "rRNA atoms"],
          [(-L2 / 2, L2 / 2, -L2 / 2, L2 / 2), (-L3 / 2, L3 / 2, -L3 / 2, L3 / 2),
           (qr[j, 0] - b3, qr[j, 0] + b3, qr[j, 1] - b3, qr[j, 1] + b3), (q4[0] - bx, q4[0] + bx, q4[1] - bx, q4[1] + bx)],
          "zoom_ribosome.jpg",
          legend=[("ribosome rRNA", C["ribo"]), ("ribosomal proteins", C["ribo_p"]), ("mRNA", C["mrna"]), ("membrane", C["mem"]),
                  ("microtubule", C["mt"]), ("actin", C["actin"]), ("cytosolic proteins", C["cyto"]), ("membrane proteins", C["tm"]),
                  ("ER-lumen proteins", C["lumen"]), ("nucleosome", C["nuc"])],
          notes=[None, None, None, None, [(0.03, 0.93, "28S rRNA " + " ".join(resl[:4]), dict(fontsize=8.6, color="#333"))]])


# ======================================================================================= figure 3: chromatin
PREF = ["ACTB", "GAPDH", "CD3E", "CD3D", "CD3G", "LCK", "PTPRC", "CD2", "EEF1A1", "B2M", "MYC", "NPM1", "HNRNPA1", "TUBB", "ZAP70",
        "CD28", "IL2RG", "LAT", "CD5", "CD7", "TMSB4X", "PFN1", "CFL1", "EEF2", "HSP90AB1", "ENO1", "PKM", "LDHA", "TPT1", "UBC"]


def fig_chromatin():
    N = np.load(f"{C8}/chromatin_v8_cell1.npz")
    fk = N["func_kind"]; fpos = N["func_pos"].astype(np.float64); fg = N["func_gene"]; fb = N["func_bp"]; fc = N["func_chain"]
    fst = N["func_state"]; names = [str(x) for x in N["chain_names"]]
    pol = np.where(fk == 0)[0]; tr = cKDTree(fpos)
    GN = pd.read_csv(f"{D}/chromatin/genes.tsv.gz", sep="\t"); EX = pd.read_csv(f"{D}/chromatin/jurkat_expression.tsv.gz", sep="\t")
    tpm = dict(zip(EX.gene_name, EX.tpm_best))
    best = None
    for i in pol[fst[pol] == 1]:
        g = str(fg[i])
        if g.startswith("ENSG") or g.startswith(("LINC", "MIR", "RNU", "SNOR")):
            continue
        nb_ = tr.query_ball_point(fpos[i], 70.0)
        kk = fk[nb_]
        same = int(((kk == 0) & (fg[nb_] == g)).sum()) - 1
        sc = (2 * (kk == 1).any() + 2 * np.isin(kk, [2, 3]).any() + (3 if g in PREF else 0) + min(np.log10(1 + (tpm.get(g) or 0)), 3) / 1.5
              + min(same, 3) - 0.5 * ((kk == 0).sum() > 6))
        if best is None or sc > best[0]:
            best = (sc, i)
    i0 = best[1]; gene = str(fg[i0]); chn = names[fc[i0]]; bp0 = int(fb[i0])
    gr = GN[GN.gene_name == gene].iloc[0]
    c = fpos[i0].copy()
    # frame: plane of the local nucleosome fibre (+-40 nucleosomes along the chain)
    ci = fc[i0]; a0 = int(N["chain_nuc_offset"][ci]); nn = int(N["chain_n_nuc"][ci])
    bpc = N["bp"][a0:a0 + nn].astype(np.int64); jn = np.searchsorted(bpc, bp0)
    Pf = N["pos"][a0 + max(jn - 40, 0):a0 + min(jn + 40, nn)].astype(np.float64)
    w, Vv = np.linalg.eigh(np.cov((Pf - Pf.mean(0)).T))
    F = np.array([Vv[:, 2], Vv[:, 1], Vv[:, 0]])
    if np.linalg.det(F) < 0:
        F[2] *= -1
    log("gene", gene, chn, bp0, "score", best[0], "state", int(fst[i0]))
    NUM["chromatin"] = dict(gene=gene, chain=chn, pol2_bp=bp0, gene_start=int(gr.start), gene_end=int(gr.end), strand=str(gr.strand),
                            tss=int(gr.tss), tpm=float(tpm.get(gene) or 0), pol2_state=("promoter-paused" if fst[i0] == 1 else "elongating")
                            , centre_nm=[round(float(v), 1) for v in c])
    # panel 1: nucleus section + nucleosome density
    HS = 9500.0
    img1, ext1, Lb, crop = section(c, F, h=20.0, half=HS)
    nucm = np.isin(Lb, [c3.NUCP, c3.NUCL, c3.SPK, c3.PERI])
    rr = np.where(nucm.any(1))[0]; cc_ = np.where(nucm.any(0))[0]; h1 = 20.0
    ncrop = (cc_[0] * h1 - HS - 600, cc_[-1] * h1 - HS + 600, HS - rr[-1] * h1 - 600, HS - rr[0] * h1 + 600)
    side = max(ncrop[1] - ncrop[0], ncrop[3] - ncrop[2]); mx = 0.5 * (ncrop[0] + ncrop[1]); my = 0.5 * (ncrop[2] + ncrop[3])
    ncrop = (mx - side / 2, mx + side / 2, my - side / 2, my + side / 2)
    # nucleosome density in a 100 nm slab
    P = N["pos"]; step = 3
    q = (P[::step].astype(np.float64) - c) @ F.T; m = np.abs(q[:, 2]) < 50; q = q[m]
    H2, xe, ye = np.histogram2d(q[:, 0], q[:, 1], bins=[np.arange(-HS, HS + h1, h1)] * 2)
    H2 = ndi.gaussian_filter(H2.T[::-1], 1.0); dens = np.clip(H2 / np.percentile(H2[H2 > 0], 97), 0, 1)
    nb_ = RGB["dna"]
    img1 = img1 * (1 - 0.75 * dens[..., None]) + 0.75 * dens[..., None] * nb_[None, None]
    p1 = panel_from_section(img1, ext1, ncrop)
    # panel 2: 1 µm (nucleosome sphere trees, functional layer, nascent RNA, membranes)
    zn = np.load(f"{C8}/prepass/mrna_nascent_cell1.npz")
    npts = zn["pts"].astype(np.float64); noff = zn["off"]
    qn = (npts - c) @ F.T
    L2, T2 = 1000.0, 60.0
    keep2 = lambda n: n in ("NUCLEOSOME", "CPX-2387", "CTCF", "CPX-5989", "CPX-5991", "NPC", "LMNB1", "LMNB2", "LMNA") or n.startswith("H1-")
    mn = np.all(np.abs(qn) <= [L2 / 2, L2 / 2, T2 / 2], 1)
    img2, ext2, inf2 = panel_meso(c, F, L2, T2, keep2, mem_h=4.0, mrna=False,
                                  extra=[(qn[mn], np.full(mn.sum(), 2.5), np.tile(RGB["nascent"], (mn.sum(), 1)))])
    log("panel2", inf2["counts"])
    # panel 3: 150 nm box, chromatin objects whole, soluble nuclear proteins clipped behind the mid-plane
    L3, T3 = 150.0, 36.0
    chrom_names = {"NUCLEOSOME", "CPX-2387", "CTCF", "CPX-5989", "CPX-5991"}

    def hl(G):
        return np.array([(n in chrom_names or n.startswith("H1-")) for n in G["name"]]) & (np.abs(G["q"][:, 2]) <= T3 / 2) & \
            np.all(np.abs(G["q"][:, :2]) <= L3 / 2 + 10, 1)
    # linker DNA: consecutive nucleosomes (v8 genome order) inside the box, DNA ends from the 1kx5 library beads
    half3 = np.array([L3, L3, T3]) / 2 + 30
    qa = (N["pos"].astype(np.float64) - c) @ F.T if False else None
    G3pre = gather(c, F, half3, margin=0, with_tiles=False)
    nm = G3pre["name"] == "NUCLEOSOME"
    Pn = G3pre["pos"][nm]; Qn = G3pre["quat"][nm]
    # map to v8 indices
    box_lo = Pn.min(0) - 1; box_hi = Pn.max(0) + 1
    allP = N["pos"]
    cand = np.where(np.all((allP >= box_lo) & (allP <= box_hi), 1))[0]
    dd, jj = cKDTree(allP[cand].astype(np.float64)).query(Pn)
    gidx = cand[jj]; okm = dd < 0.5
    Lnb = lib("NUCLEOSOME"); bP = Lnb["beads"][Lnb["kind"] == 1]
    endA = 0.5 * (bP[0] + bP[-1]); endB = 0.5 * (bP[146] + bP[147])
    Mn = qmats(Qn)
    EA = np.einsum("nij,j->ni", Mn, endA) + Pn; EB = np.einsum("nij,j->ni", Mn, endB) + Pn
    in_locus = okm & (N["chain"][gidx] == ci) & (np.abs(N["bp"][gidx].astype(np.int64) - bp0) < 25_000)
    locus_pos = Pn[in_locus]
    order = np.argsort(gidx); used = {}
    link_pts = []; link_loc = []; nlink = 0
    gi_s = gidx[order]
    for a, b in zip(order[:-1], order[1:]):
        if not (okm[a] and okm[b]) or gidx[b] != gidx[a] + 1:
            continue
        if N["chain"][gidx[a]] != N["chain"][gidx[b]]:
            continue
        opts = []
        for ea_k, ea in (("A", EA[a]), ("B", EB[a])):
            if used.get(a) == ea_k:
                continue
            for eb_k, eb in (("A", EA[b]), ("B", EB[b])):
                opts.append((np.linalg.norm(ea - eb), ea_k, eb_k, ea, eb))
        dmin, ka, kb, ea, eb = min(opts, key=lambda t: t[0])
        if dmin > 45 or abs((Pn[a] - c) @ F[2]) > T3 / 2 or abs((Pn[b] - c) @ F[2]) > T3 / 2:
            continue
        used[b] = kb; nlink += 1
        t = np.linspace(0, 1, max(int(dmin / 0.5), 2))
        link_pts.append(ea[None] + t[:, None] * (eb - ea)[None]); link_loc.append(np.full(len(t), bool(in_locus[a] and in_locus[b])))
    LP = (np.concatenate(link_pts) - c) @ F.T if link_pts else np.zeros((0, 3))
    LL = np.concatenate(link_loc) if link_loc else np.zeros(0, bool)
    m_l = np.all(np.abs(LP[:, :2]) <= L3 / 2, 1)
    LP = LP[m_l]; LL = LL[m_l]
    FADE = lambda x: 0.42 * x + 0.58
    lcol = np.where(LL[:, None], RGB["dna"][None], FADE(RGB["dna"])[None])

    def recolor(G, inst, kind, col):
        """the CFL1 locus fibre (+-25 kb on the same allele) in full colour, other chromatin faded"""
        chrom = np.array([(n == "NUCLEOSOME" or n.startswith("H1-")) for n in G["name"]])
        loc = np.zeros(len(G["name"]), bool)
        if len(locus_pos):
            d, _ = cKDTree(locus_pos).query(G["pos"]); loc = d < 0.5
            hn = np.array([n.startswith("H1-") for n in G["name"]])
            if hn.any():                                   # H1 on a locus nucleosome
                d2, _ = cKDTree(locus_pos).query(G["pos"][hn]); loc[np.where(hn)[0][d2 < 12]] = True
        f = chrom[inst] & ~loc[inst]
        col = col.copy(); col[f] = FADE(col[f])
        return col
    # nascent RNA beads (gene-tagged) in the box
    mn3 = np.all(np.abs(qn[:, :2]) <= L3 / 2, 1) & (np.abs(qn[:, 2]) <= T3 / 2 + 20)
    extra = [(LP, np.full(len(LP), 1.0), lcol),
             (qn[mn3], np.full(mn3.sum(), 1.0), np.tile(RGB["nascent"], (mn3.sum(), 1)))]
    img3, ext3, G3, inf3 = panel_context(c, F, L3, T3, zclip_bg=-8.0, hl=hl, extra=extra, outline=6.0, ao=0.3, recolor=recolor)
    inf3["verify"] = verify(G3, c, F, "chromatin")
    inf3["linker_segments"] = nlink; inf3["locus_nucleosomes"] = int(in_locus.sum()); inf3["nascent_rna_beads"] = int(mn3.sum())
    # functional-layer genes in the box
    q_f = (fpos - c) @ F.T; mf = np.all(np.abs(q_f) <= [L3 / 2, L3 / 2, T3 / 2], 1)
    inf3["func_in_box"] = [dict(kind=["PolII", "CTCF", "cohesin-STAG1", "cohesin-STAG2"][fk[i]], gene=str(fg[i]), chain=names[fc[i]],
                                bp=int(fb[i])) for i in np.where(mf)[0]]
    # nascent transcripts in box: genes
    tid = np.searchsorted(noff, np.where(mn3)[0], side="right") - 1
    inf3["nascent_genes"] = sorted(set(str(zn["gene"][t]) for t in tid))
    log("panel3", {k: v for k, v in inf3.items() if k != "func_in_box"}, inf3["func_in_box"])
    # panel 4: nucleosome 1KX5
    A = read_atoms("1kx5")
    desc = A["desc"]; hcol = {}
    for e, dsc in desc.items():
        u = dsc.upper()
        hcol[e] = (RGB["dna"] if "DNA" in u else to_rgb("#8fa9cc") if "H3" in u else to_rgb("#a8c49c") if "H4" in u else
                   to_rgb("#e6cf8f") if "H2A" in u else to_rgb("#dba8a8") if "H2B" in u else RGB["nuc"])
    cols4 = np.array([hcol.get(e, RGB["nuc"]) for e in A["ent"]])
    ctr = A["xyz"].mean(0)
    dnaP = A["xyz"][A["na"] & (A["atom"] == "P")]
    w, Vv = np.linalg.eigh(np.cov((dnaP - dnaP.mean(0)).T)); sup = Vv[:, 0]
    F4 = view_frame(A["xyz"], tilt_deg=-25, axis=np.cross(sup, Vv[:, 2]))
    F4 = np.array([Vv[:, 2], Vv[:, 1], Vv[:, 0]]) if True else F4      # face-on: superhelix axis toward the viewer
    if np.linalg.det(F4) < 0:
        F4[2] *= -1
    t = np.deg2rad(20); Ry = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]]); F4 = Ry @ F4
    L4 = 13.5
    pr_ = (A["xyz"] - ctr) @ Vv
    NUM.setdefault("sizes_nm", {})["nucleosome_1KX5_extent"] = [round(float(v), 1) for v in np.ptp(pr_, 0)[::-1]]
    img4, ext4 = spacefill(A["xyz"], A["el"], cols4, F4, ctr, L4, outline=0.12)
    # close-up: H3 arginine nearest a DNA phosphate (minor-groove insertion)
    h3ent = [e for e, d in desc.items() if "H3" in d.upper()]
    argm = np.isin(A["ent"], h3ent) & (A["resn"] == "ARG") & np.isin(A["atom"], ["NH1", "NH2", "NE"])
    Pm = A["na"] & (A["atom"] == "P")
    dd, jj = cKDTree(A["xyz"][Pm]).query(A["xyz"][argm])
    k = np.argmin(dd); ach = A["chain"][argm][k]; ari = A["resi"][argm][k]
    core = (A["chain"] == ach) & (A["resi"] == ari)
    cc = A["xyz"][core].mean(0)
    Pn_ = A["na"] & (np.linalg.norm(A["xyz"] - cc, axis=1) < 0.75)
    core = core | residues_near(A, Pn_, 0.0)
    sel = residues_near(A, core, 0.55)
    colsC = np.array([hcol.get(e, RGB["nuc"]) for e in A["ent"]])
    colsC = atom_cols(A["el"], np.zeros(3)) * 0 + colsC
    for e_, h_ in CPK.items():
        colsC[A["el"] == e_] = to_rgb(h_)
    Fz = view_frame(A["xyz"][core]); L5 = 2.6
    img5, ext5 = ball_stick(A["xyz"][sel], A["el"][sel], colsC[sel], Fz, cc, L5, core=core[sel])
    q4 = (cc - ctr) @ F4.T; bx = 1.5
    # box around one nucleosome near the centre in panel 3
    qq = G3["q"][G3["name"] == "NUCLEOSOME"]
    j = np.argmin(np.linalg.norm(qq[:, :2] - np.array([-25.0, 0.0]), axis=1) + 2 * np.maximum(qq[:, 2], 0) * 0)
    b3 = 9.0
    NUM["chromatin"].update(panel2=dict(L=L2, T=T2, **inf2), panel3=dict(L=L3, T=T3, **inf3), pdb="1KX5",
                            pdb_atoms=int(len(A["el"])), closeup=dict(chain=str(ach), residue=f"ARG{ari}", atoms=int(sel.sum()),
                                                                     min_NH_P_nm=round(float(dd[k]), 3)))
    chrom = chn[:-1]; hom = chn[-1]
    lab = f"{gene} promoter, chr{chrom}:{bp0:,} ({gr.strand}), allele {hom}"
    arrows = []
    names_k = ["Pol II", "CTCF", "cohesin", "cohesin"]; cols_k = [C["pol2"], C["ctcf"], C["cohesin"], C["cohesin"]]
    for i in np.where(mf)[0]:
        x, y = q_f[i, :2]
        if np.hypot(x, y) < 1:
            dx, dy = 28.0, 30.0
        else:
            u = np.array([x, y]) / np.hypot(x, y); dx, dy = 26 * u
            if abs(x + dx) > 62: dx = -dx
            if abs(y + dy) > 62: dy = -dy
        arrows.append(("arrow", x, y, names_k[fk[i]], cols_k[fk[i]], dx, dy))
    strip([p1, (img2, ext2), (img3, ext3), (img4, ext4), (img5, ext5)],
          ["nucleus (section)", "chromatin, 1 µm", f"{gene} promoter in context", "nucleosome (PDB 1KX5)", "histone H3–DNA atoms"],
          [(-L2 / 2, L2 / 2, -L2 / 2, L2 / 2), (-L3 / 2, L3 / 2, -L3 / 2, L3 / 2), (qq[j, 0] - b3, qq[j, 0] + b3, qq[j, 1] - b3, qq[j, 1] + b3),
           (q4[0] - bx, q4[0] + bx, q4[1] - bx, q4[1] + bx)],
          "zoom_chromatin.jpg",
          legend=[("nucleosome histones", C["nuc"]), ("DNA", C["dna"]), ("H1", C["h1"]), ("Pol II", C["pol2"]), ("nascent RNA", C["nascent"]),
                  ("CTCF", C["ctcf"]), ("cohesin", C["cohesin"]), ("nuclear proteins", C["nucl"]), ("membrane", C["mem"])],
          notes=[None, None, [(0.03, 0.93, lab, dict(fontsize=8.6, color="#222", fontweight="bold"))] + arrows, None,
                 [(0.03, 0.93, f"H3 Arg{ari} · DNA", dict(fontsize=8.6, color="#333"))]])
    # mark the Pol II in panel 3? (drawn in orange; nothing else)


# ======================================================================================= ruler
def fig_ruler():
    """log ruler 10 µm -> 0.1 nm; each icon (a thumbnail from the zoom strips) has a leader to its real size on the axis"""
    nz = NUM["npc"]; c = np.array(nz["centre_nm"]); F = np.array(nz["frame_rows"])
    _, _, Lb, crop = section(c, F, h=20.0, half=9500.0)
    h = 20.0; ins = ~np.isin(Lb, [c3.EXT, c3.OUTV]); nucm = np.isin(Lb, [c3.NUCP, c3.NUCL, c3.SPK, c3.PERI, c3.PORE])
    ext = lambda m: max(np.ptp(np.where(m.any(0))[0]), np.ptp(np.where(m.any(1))[0])) * h
    sz = NUM.get("sizes_nm", {})
    items = [("cell", ext(ins), "zoom_npc_p0"), ("nucleus", ext(nucm), "zoom_chromatin_p0"),
             ("nuclear pore", sz.get("NPC_7R5K_diameter", 120.0), "zoom_npc_p3"),
             ("ribosome", max(sz.get("ribosome_6QZP_extent", [28.0])), "zoom_ribosome_p3"),
             ("nucleosome", max(sz.get("nucleosome_1KX5_extent", [11.0])), "zoom_chromatin_p3"),
             ("carbon atom", 0.34, "zoom_npc_p4")]
    NUM["ruler"] = {k: round(float(v), 2) for k, v, _ in items}
    fig = plt.figure(figsize=(16, 4.2)); ax = fig.add_axes([0.03, 0.16, 0.94, 0.80])
    lo, hi = np.log10(0.1), np.log10(20000.0)
    ax.set_xlim(hi, lo); ax.set_ylim(0, 1); ax.axis("off")
    ax.plot([hi, lo], [0.12, 0.12], color="#333", lw=1.6)
    labs = {-1: "1 Å", 0: "1 nm", 1: "10 nm", 2: "100 nm", 3: "1 µm", 4: "10 µm"}
    for e in range(-1, 5):
        ax.plot([e, e], [0.12, 0.17], color="#333", lw=1.3)
        ax.text(e, 0.09, labs[e], ha="center", va="top", fontsize=11.5)
        for mm in range(2, 10):
            x = e + np.log10(mm)
            if lo <= x <= hi:
                ax.plot([x, x], [0.12, 0.145], color="#777", lw=0.8)
    W = fig.get_size_inches()[0] * 0.94; span = hi - lo
    iw = 0.115 * span                                          # icon width in axis units
    slots = np.linspace(hi - 0.09 * span, lo + 0.09 * span, len(items))
    for (nm, size, thumb), xs in zip(items, slots):
        x = np.log10(size)
        z = np.load(f"{CACHE}/{thumb}.npz"); img = z["img"]
        if nm == "carbon atom":                              # a single carbon at its van der Waals radius
            n = 200; yy, xx = np.mgrid[0:n, 0:n] / n * 2 - 1; r2 = xx ** 2 + yy ** 2
            nzz = np.sqrt(np.clip(1 - r2, 0, 1)); Lv = np.array([-0.45, 0.55, 0.7]); Lv /= np.linalg.norm(Lv)
            d = np.clip(-0.45 * xx - 0.55 * yy + 0.7 * nzz, 0, 1) * 0 + np.clip(xx * Lv[0] - yy * Lv[1] + nzz * Lv[2], 0, 1)
            col = np.array(to_rgb("#8c8c8c")); img = np.ones((n, n, 3))
            m = r2 < 0.8 ** 2
            sh = (0.45 + 0.55 * d)[..., None] * col + 0.25 * (d[..., None] ** 30)
            img[m] = np.clip(sh[m], 0, 1)
        ax.imshow(img, extent=(xs + iw / 2, xs - iw / 2, 0.36, 0.36 + 0.5), aspect="auto", zorder=3)
        ax.add_patch(Rectangle((xs - iw / 2, 0.36), iw, 0.5, fill=False, ec="#bbb", lw=0.8, zorder=4))
        ax.plot([xs, x], [0.36, 0.17], color="#666", lw=1.0, zorder=2)
        ax.plot([x], [0.12], marker="o", ms=6, color="#222", zorder=5)
        lab = f"{size / 1000:.0f} µm" if size >= 1000 else (f"{size:.0f} nm" if size >= 10 else (f"{size:.1f} nm" if size >= 1 else f"{size * 10:.1f} Å"))
        ax.text(xs, 0.89, f"{nm}  ·  {lab}", ha="center", va="bottom", fontsize=11)
    save(fig, "scale_bar_ruler.jpg")


if __name__ == "__main__":
    which = sys.argv[2:] or ["npc", "ribosome", "chromatin"]
    t0 = time.time()
    for w in which:
        {"npc": fig_npc, "ribosome": fig_ribosome, "chromatin": fig_chromatin, "ruler": fig_ruler}[w]()
        json.dump(NUM, open(NUMF, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        log("done", w, round(time.time() - t0))
