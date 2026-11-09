import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.size"] = 9


def scalebar(ax, x0, y0, length_px, label, color="k", lw=3):
    ax.plot([x0, x0 + length_px], [y0, y0], color=color, lw=lw, solid_capstyle="butt")
    ax.text(x0 + length_px / 2, y0 - 0.025 * ax.get_ylim()[0], label, ha="center", va="bottom", color=color, fontsize=8)


def annotate_column(png, out, z0=-30, px_per_nm=6, labels=None):
    im = np.asarray(Image.open(png))
    H, W = im.shape[:2]
    fig = plt.figure(figsize=(W / 300, H / 300 + 0.55), dpi=300)
    ax = fig.add_axes([0, 0, 1, H / (H + 0.55 * 300)])
    ax.imshow(im, interpolation="lanczos"); ax.set_axis_off()
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    tax = fig.add_axes([0, H / (H + 0.55 * 300), 1, 1 - H / (H + 0.55 * 300)]); tax.set_xlim(0, W); tax.set_ylim(0, 1); tax.set_axis_off()
    for a, b, txt in labels:
        xa, xb = (a - z0) * px_per_nm, (b - z0) * px_per_nm
        tax.plot([xa + 4, xb - 4], [0.18, 0.18], color="0.35", lw=0.8)
        tax.text((xa + xb) / 2, 0.32, txt, ha="center", va="bottom", fontsize=6.5)
    scalebar(ax, W - 50 * px_per_nm - 40, H - 30, 50 * px_per_nm, "50 nm", color="0.1")
    fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".svg"); plt.close(fig)


def annotate_cell(png_cell, png_zoom, out, labels, zoom_box_px, cell_px_per_nm=1.0, zoom_px_per_nm=3.0):
    a = np.asarray(Image.open(png_cell)); b = np.asarray(Image.open(png_zoom))
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), dpi=300, gridspec_kw=dict(width_ratios=[1, b.shape[1] / b.shape[0]]))
    axs[0].imshow(a, interpolation="lanczos"); axs[1].imshow(b, interpolation="lanczos")
    for ax in axs: ax.set_axis_off()
    for x, y, txt in labels:
        axs[0].annotate(txt, (x, y), fontsize=7, ha="center", va="center", color="0.1",
                        bbox=dict(boxstyle="round,pad=0.2", fc=(1, 1, 1, 0.75), ec="none"))
    x0, y0, x1, y1 = zoom_box_px
    axs[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="0.15", lw=0.8))
    scalebar(axs[0], 120, a.shape[0] - 80, 1000 * cell_px_per_nm, "1 µm")
    scalebar(axs[1], b.shape[1] - 100 * zoom_px_per_nm - 60, b.shape[0] - 50, 100 * zoom_px_per_nm, "100 nm", color="0.1")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".svg"); plt.close(fig)
