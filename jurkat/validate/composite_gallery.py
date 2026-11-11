"""Three-channel model vs HPA Jurkat comparison, in the HPA colour scheme:
green = protein, red = microtubules, blue = DNA.

Model channels, all on the same confocal section (fluor_setup.json):
  * protein: v6 pack images from cache_gallery_v6 (noisy confocal);
  * DNA: nucleosome pre-pass image from fluor/cache/img_H2B.npz;
  * microtubules: stage 3 variant b centerlines (EM-scaffolded network plus aster; v7 candidate), polymer only.
    Each point is weighted by axial PSF distance, then blurred laterally with the same Gaussian confocal PSF.
HPA: the *_blue_red_green.jpg composites (CC BY-SA 4.0), each cropped to the cell nearest the field centre.

Usage: conda_envs/vcell/bin/python jurkat/validate/composite_gallery.py GENE [GENE ...]
Output: data/derived/validation/composite_gallery.png
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fluor as F
from scipy import ndimage as ndi

IMG = f"{F.REF}/hpa_subcellular/jurkat_images"
GAL = f"{F.FD}/cache_gallery_v6"
HPA_PX = 0.08          # µm per HPA pixel (approximate; used for scale bars and crop size)


def mt_image(S, shape):
    z = np.load(f"{F.D}/cytoskel/stage3/mt_stage3_b_centerlines.npz", allow_pickle=True)
    p = z["points_model_nm"].astype(np.float64)
    lo, n = np.array(S["grid_lo"]), np.array(S["grid_n"]); psf = S["psf"]["confocal"]
    w = np.exp(-(p[:, 2] - S["z0_nm"]) ** 2 / (2 * psf["sz"] ** 2))
    ij = np.floor((p[:, :2] - lo[:2]) / F.H).astype(int)
    ok = (w > 1e-3) & np.all((ij >= 0) & (ij < n[:2]), 1)
    im = np.zeros((n[0], n[1]), np.float32)
    np.add.at(im, (ij[ok, 0], ij[ok, 1]), w[ok])
    im = ndi.gaussian_filter(im, psf["sxy"] / F.H)
    cam = F.to_camera(im)[:shape[0], :shape[1]]
    rng = np.random.default_rng(7)
    noisy, _ = F.add_noise(cam, psf["peak"], np.percentile(cam[cam > 0], 99.5), rng)
    return noisy


def norm(im, mask, lo=1, hi=99.7):
    a, b = np.percentile(im[mask], lo), np.percentile(im[mask], hi)
    return np.clip((im - a) / max(b - a, 1e-9), 0, 1)


def hpa_field(g, half_um=20.0):
    """field of view with a few cells: the window (over all JURKAT fields of the gene) with the most cell signal"""
    import matplotlib.pyplot as plt
    best = None; h = int(half_um / HPA_PX)
    for f in sorted(glob.glob(f"{IMG}/{g}_*_blue_red_green.jpg")):
        im = plt.imread(f); sig = im[..., 2].astype(np.float32) + im[..., 1]
        for i in range(h, im.shape[0] - h + 1, 64):
            for j in range(h, im.shape[1] - h + 1, 64):
                v = sig[i - h:i + h:8, j - h:j + h:8].mean()
                if best is None or v > best[0]: best = (v, im, i, j, os.path.basename(f))
    if best is None: return None, None
    _, im, i, j, fn = best
    return im[i - h:i + h, j - h:j + h], fn


def hpa_crop(g, half_um=11.0):
    fs = sorted(glob.glob(f"{IMG}/{g}_*_blue_red_green.jpg"))
    import matplotlib.pyplot as plt
    best = None
    for f in fs:
        im = plt.imread(f).astype(np.float32) / 255.0
        b = ndi.gaussian_filter(im[..., 2], 3)
        lab, nl = ndi.label(b > max(0.15, np.percentile(b, 70)))
        if nl == 0: continue
        sizes = ndi.sum(np.ones_like(b), lab, range(1, nl + 1))
        cms = ndi.center_of_mass(np.ones_like(b), lab, range(1, nl + 1))
        h = int(half_um / HPA_PX); c0 = np.array(im.shape[:2]) / 2
        for s, cm in zip(sizes, cms):
            cm = np.array(cm)
            if not (8000 < s < 60000) or np.any(cm < h) or np.any(cm > np.array(im.shape[:2]) - h): continue
            score = np.linalg.norm(cm - c0) - 0.002 * s
            if best is None or score < best[0]: best = (score, f, cm.astype(int), h, im)
    if best is None: return None, None
    _, f, cm, h, im = best
    return im[cm[0] - h:cm[0] + h, cm[1] - h:cm[1] + h], os.path.basename(f)


def main(genes):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    c3, cell = F.cell_geom(); S = json.load(open(f"{F.OUT}/fluor_setup.json"))
    labp = F.plane_labels(cell, c3, np.array(S["grid_lo"]), np.array(S["grid_n"]), S["z0_nm"], F.PIX)
    dna = np.load(f"{F.FD}/cache/img_H2B.npz")["noisy_confocal"]
    shp = dna.shape; labp = labp[:shp[0], :shp[1]]
    inside = ~np.isin(labp, [c3.EXT, c3.OUTV])
    mt = mt_image(S, shp)
    ins = np.nonzero(inside); m = int(1500 / F.PIX)
    sl = (slice(max(ins[0].min() - m, 0), ins[0].max() + m), slice(max(ins[1].min() - m, 0), ins[1].max() + m))
    R, B = norm(mt, inside), norm(dna, inside)
    fig, axs = plt.subplots(2, len(genes), figsize=(3.6 * len(genes), 7.4), facecolor="black")
    for j, g in enumerate(genes):
        G = norm(np.load(f"{GAL}/img_{g}.npz")["noisy_confocal"], inside)
        rgb = np.stack([R, G, B * 0.85], -1)[sl].transpose(1, 0, 2)[::-1]
        a = axs[0, j]; a.imshow(rgb, interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
        L = 5000 / F.PIX; a.plot([10, 10 + L], [rgb.shape[0] - 12] * 2, color="white", lw=2)
        a.set_title(f"{g}: model (v6 + EM microtubules)", color="white", fontsize=9)
        hc, fn = hpa_crop(g)
        a = axs[1, j]; a.set_xticks([]); a.set_yticks([])
        if hc is not None:
            a.imshow(hc); L = 5 / HPA_PX; a.plot([10, 10 + L], [hc.shape[0] - 12] * 2, color="white", lw=2)
        a.set_title(f"{g}: HPA Jurkat", color="white", fontsize=9)
    fig.text(0.5, 0.005, "green = protein, red = microtubules, blue = DNA; 5 µm bars. Model: simulated confocal section (v6 proteins, "
             "stage-3 EM-scaffolded microtubules, nucleosomes). HPA images © Human Protein Atlas, CC BY-SA 4.0",
             color="white", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = f"{F.OUT}/composite_gallery.png"; fig.savefig(out, dpi=120, facecolor="black"); print(out)




# ---------------------------------------------------------------- xz section (optical axis = model y)
# In cell 1 the microtubules run mostly along model z (mean |direction| x 0.46, y 0.30, z 0.70), so an xy section
# shows them as dots. The cell's orientation relative to a coverslip is unknown (the EM z axis is the block axis),
# so a section containing the long axis is an equally valid view and is closer to how MTs appear in the HPA images.
def plane_render(pos, w, y0, lo, n, psf, rng, ref=None):
    wy = np.exp(-(pos[:, 1] - y0) ** 2 / (2 * psf["sz"] ** 2)) * w
    ij = np.floor((pos[:, [0, 2]] - lo[[0, 2]]) / F.H).astype(int)
    ok = (wy > 1e-4) & np.all((ij >= 0) & (ij < n[[0, 2]]), 1)
    im = np.zeros((n[0], n[2]), np.float32); np.add.at(im, (ij[ok, 0], ij[ok, 1]), wy[ok])
    cam = F.to_camera(ndi.gaussian_filter(im, psf["sxy"] / F.H))
    noisy, _ = F.add_noise(cam, psf["peak"], np.percentile(cam[cam > 0], 99.5) if ref is None else ref, rng)
    return noisy


def main_xz(genes):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.environ.setdefault("LOC_VERSION", "v6")
    import jurkat_gallery as JG                                   # sets F.PANEL / F.CACHE to the v6 gallery cache
    S = json.load(open(f"{F.OUT}/fluor_setup.json")); psf = S["psf"]["confocal"]
    lo, n = np.array(S["grid_lo"]), np.array(S["grid_n"]); y0 = S["nucleus_centroid_nm"][1]
    pts = F.load_points(); rng = np.random.default_rng(11)
    c3, cell = F.cell_geom()
    nx, nz = int(round(n[0] * F.H / F.PIX)), int(round(n[2] * F.H / F.PIX))
    X, Z = np.meshgrid(lo[0] + (np.arange(nx) + .5) * F.PIX, lo[2] + (np.arange(nz) + .5) * F.PIX, indexing="ij")
    lab = cell.labels(np.stack([X.ravel(), np.full(X.size, y0), Z.ravel()], 1)).reshape(nx, nz)
    dna = plane_render(*pts["H2B"][:2], y0, lo, n, psf, rng)
    shp = dna.shape; lab = lab[:shp[0], :shp[1]]; inside = ~np.isin(lab, [c3.EXT, c3.OUTV])
    mz = np.load(f"{F.D}/cytoskel/stage3/mt_stage3_b_centerlines.npz")["points_model_nm"].astype(np.float64)
    mt = plane_render(mz, np.ones(len(mz)), y0, lo, n, psf, rng)[:shp[0], :shp[1]]
    ins = np.nonzero(inside); m = int(1500 / F.PIX)
    sl = (slice(max(ins[0].min() - m, 0), ins[0].max() + m), slice(max(ins[1].min() - m, 0), ins[1].max() + m))
    R, B = norm(mt, inside), norm(dna, inside)
    fig, axs = plt.subplots(2, len(genes), figsize=(3.6 * len(genes), 8.4), facecolor="black")
    for j, g in enumerate(genes):
        G = norm(plane_render(*pts[g][:2], y0, lo, n, psf, rng)[:shp[0], :shp[1]], inside)
        rgb = np.stack([R, G, B * 0.85], -1)[sl].transpose(1, 0, 2)[::-1]
        a = axs[0, j]; a.imshow(rgb, interpolation="nearest"); a.set_xticks([]); a.set_yticks([])
        L = 5000 / F.PIX; a.plot([10, 10 + L], [rgb.shape[0] - 12] * 2, color="white", lw=2)
        a.set_title(f"{g}: model", color="white", fontsize=10)
        hc, fn = hpa_field(g)
        a = axs[1, j]; a.set_xticks([]); a.set_yticks([])
        if hc is not None:
            a.imshow(hc); L = 10 / HPA_PX; a.plot([15, 15 + L], [hc.shape[0] - 20] * 2, color="white", lw=2)
        a.set_title(f"{g}: HPA Jurkat (10 µm bar)", color="white", fontsize=10)
    fig.text(0.5, 0.005, "green = protein, red = microtubules, blue = DNA. Model (5 µm bar): simulated confocal section through the nucleus "
             f"centre containing the cell's long axis ({os.environ['LOC_VERSION']} proteins, EM-scaffolded microtubules, nucleosomes); flat bottom edge = EM block face. "
             "HPA images © Human Protein Atlas, CC BY-SA 4.0", color="white", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = f"{F.OUT}/composite_gallery_xz.png" if os.environ["LOC_VERSION"] == "v6" else f"{F.OUT}/composite_gallery_xz_{os.environ['LOC_VERSION']}.png"; fig.savefig(out, dpi=120, facecolor="black"); print(out)


if __name__ == "__main__":
    (main_xz if os.environ.get("PLANE") == "xz" else main)(sys.argv[1:])
