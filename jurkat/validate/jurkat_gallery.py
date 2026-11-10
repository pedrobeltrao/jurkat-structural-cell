"""Gallery: simulated confocal sections of the model next to HPA immunofluorescence images of JURKAT cells.

Reuses fluor.py (same PSF, camera, noise and the same section through the nucleus centre as fluor_setup.json) for genes
outside the fluorescence panel that HPA imaged in JURKAT (data/reference/hpa_subcellular/jurkat_images/, CC BY-SA 4.0).
Model images go to a separate cache so the panel outputs are untouched.

Usage (conda_envs/vcell/bin/python):
  python jurkat/validate/jurkat_gallery.py collect   # read tiles once (~20 min)
  python jurkat/validate/jurkat_gallery.py render    # model sections
  python jurkat/validate/jurkat_gallery.py figure    # data/derived/validation/jurkat_gallery.png (+ gallery_*.png pages)
Long runs: setsid nohup ... > logs/jurkat_gallery.log 2>&1 &"""
import os, sys, glob, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fluor as F

IMG = f"{F.REF}/hpa_subcellular/jurkat_images"
GENES = ["LCP1", "MYO1G", "FNBP1", "PTPRCAP", "CD28", "CD2", "CD3E", "CD3D",
         "RAD21", "ARID2", "IKZF1", "H1-10", "ACSF3", "ACADS", "SH2D1A", "RACK1",
         "ARHGAP15", "ARHGAP45", "ADD3", "STAM", "GIMAP7", "STK17B", "SELPLG", "ITK"]
F.PANEL = [(g, "jurkat", ("gene", g), g) for g in GENES] + [("H2B", "ref", ("re", r"^H2B"), "H2B (all)")]
F.NAMES = [p[0] for p in F.PANEL]
F.CACHE = f"{F.FD}/cache_gallery"
F.NPROC = 16
VER = os.environ.get("LOC_VERSION", "v5")
F.CACHE = f"{F.FD}/cache_gallery" + ("" if VER == "v5" else f"_{VER}")
OUT = f"{F.OUT}/jurkat_gallery.png" if VER == "v5" else f"{F.OUT}/jurkat_gallery_{VER}.png"


def render():
    os.makedirs(F.CACHE, exist_ok=True)
    c3, cell = F.cell_geom(); S = json.load(open(f"{F.OUT}/fluor_setup.json"))
    lo, n = np.array(S["grid_lo"]), np.array(S["grid_n"]); z0 = S["z0_nm"]; iz0 = int(np.floor((z0 - lo[2]) / F.H))
    pts = F.load_points(); F.log("points:", {k: len(v[0]) for k, v in pts.items()})
    labp = F.plane_labels(cell, c3, lo, n, z0, F.PIX)
    from scipy.ndimage import zoom
    shp = zoom(np.zeros((n[0], n[1]), np.float32), F.H / F.PIX, order=1).shape
    labp = np.pad(labp, ((0, max(0, shp[0] - labp.shape[0])), (0, max(0, shp[1] - labp.shape[1]))), constant_values=c3.EXT)[:shp[0], :shp[1]]
    F._G.update(pts=pts, lo=lo, n=n, iz0=iz0, F2=F._freq(n[0], n[1]), labp=labp, c3=c3)
    import multiprocessing as mp
    rows = []
    with mp.get_context("fork").Pool(F.NPROC) as pool:
        for row, _ in pool.imap(F._render_one, GENES):
            rows.append(row); F.log("rendered", row["protein"], row["copies"])
    pd.DataFrame(rows).to_csv(f"{F.CACHE}/metrics.tsv", sep="\t", index=False)


def _hpa_image(g):
    fs = sorted(glob.glob(f"{IMG}/{g}_*_blue_red_green.jpg"))
    return fs[0] if fs else None


def figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    c3, cell = F.cell_geom(); S = json.load(open(f"{F.OUT}/fluor_setup.json"))
    labp = F.plane_labels(cell, c3, np.array(S["grid_lo"]), np.array(S["grid_n"]), S["z0_nm"], F.PIX)
    ins = np.nonzero(~np.isin(labp, [c3.EXT, c3.OUTV])); m = int(1000 / F.PIX)
    sl = (slice(max(ins[0].min() - m, 0), ins[0].max() + m), slice(max(ins[1].min() - m, 0), ins[1].max() + m))
    lc = pd.read_csv(f"{F.OUT}/loc_compare_{VER}.tsv", sep="\t").drop_duplicates("gene").set_index("gene")
    REG = {"sh_PM_cortex": "cortex", "sh_PM_cyto": "PM shell", "nucleoplasm": "nucleus", "matrix": "mito matrix",
           "MITO_OM": "mito OM", "ENDO": "endo-lys.", "sh_ENDO_cyto": "endo-lys. shell", "GOLGI": "Golgi",
           "sh_GOLGI_cyto": "Golgi shell", "speckle": "speckles", "sh_NPC": "NPC-proximal"}
    def regions(s):
        out = []
        for t in str(s).split(";"):
            k, v = t.split(":"); v = float(v)
            if v >= 0.05: out.append(f"{REG.get(k, k)} {100 * v:.0f}%")
        return ", ".join(out[:3])
    genes = [g for g in GENES if _hpa_image(g) and os.path.exists(f"{F.CACHE}/img_{g}.npz")]
    per = 12; ncol = 4
    pages = [genes[i:i + per] for i in range(0, len(genes), per)]
    outs = []
    for pi, gs in enumerate(pages):
        nrow = int(np.ceil(len(gs) / ncol))
        fig, axs = plt.subplots(2 * nrow, ncol, figsize=(4.2 * ncol, 3.55 * 2 * nrow), facecolor="white")
        axs = np.atleast_2d(axs)
        for ax in axs.ravel(): ax.axis("off")
        for k, g in enumerate(gs):
            r, c = 2 * (k // ncol), k % ncol
            sc = lc.jurkat_score.get(g, np.nan)
            verdict = "agrees" if sc <= 0.2 else ("disagrees" if sc >= 0.75 else "partly agrees")
            col = {"agrees": "#1f7a56", "partly agrees": "#8a5a00", "disagrees": "#a3322a"}[verdict]
            a0 = axs[r, c]; a0.axis("on")
            F.show(a0, np.load(f"{F.CACHE}/img_{g}.npz")["noisy_confocal"][sl], labp[sl], c3)
            a0.set_title(f"{g}  ·  model: {regions(lc.model_regions.get(g, ''))}", fontsize=8, pad=3)
            a0.text(0.98, 0.97, verdict, transform=a0.transAxes, ha="right", va="top", fontsize=8, color="white",
                    bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none"))
            im = plt.imread(_hpa_image(g)); h0 = im.shape[0] // 2; h = int(0.27 * im.shape[0])
            a1 = axs[r + 1, c]; a1.axis("on"); a1.imshow(im[h0 - h:h0 + h, h0 - h:h0 + h])
            a1.set_xticks([]); a1.set_yticks([])
            L = 10 / 0.08; a1.plot([15, 15 + L], [2 * h - 20] * 2, color="white", lw=2)
            a1.text(15 + L / 2, 2 * h - 30, "10 µm", color="white", ha="center", fontsize=6)
            a1.set_title(f"HPA Jurkat: {str(lc.hpa_jurkat.get(g, '?')).replace(';', ', ')}", fontsize=8, pad=3)
        fig.suptitle("Model confocal section (green; grey = cell outline, blue = nucleus)  vs  HPA immunofluorescence in Jurkat "
                     "(green = protein, blue = nucleus, red = microtubules).  HPA images © Human Protein Atlas, CC BY-SA 4.0",
                     fontsize=9, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        o = OUT.replace(".png", f"_{pi + 1}.png"); fig.savefig(o, dpi=110); plt.close(fig); outs.append(o)
    print("\n".join(outs))


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "collect": F.collect()
    if step == "render": render()
    if step == "figure": figure()
