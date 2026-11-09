"""Goodsell/Illustrate-style non-photorealistic renderer: orthographic z-buffer of spheres, flat colours,
depth fog, dark molecular outlines + light internal contours, compartment-tinted background."""
import numpy as np
import numba as nb
from PIL import Image


def hex2rgb(h):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255


@nb.njit(cache=True)
def raster(U, V, D, Rp, inst, H, W):
    zbuf = np.full((H, W), np.inf); ibuf = -np.ones((H, W), np.int64)
    for b in range(U.shape[0]):
        u, v, d, r = U[b], V[b], D[b], Rp[b]
        i0 = max(int(v - r), 0); i1 = min(int(v + r) + 1, H)
        j0 = max(int(u - r), 0); j1 = min(int(u + r) + 1, W)
        r2 = r * r
        for i in range(i0, i1):
            dv = i + 0.5 - v
            for j in range(j0, j1):
                du = j + 0.5 - u
                q = du * du + dv * dv
                if q <= r2:
                    dd = d - np.sqrt(r2 - q)
                    if dd < zbuf[i, j]:
                        zbuf[i, j] = dd; ibuf[i, j] = b
    return zbuf, ibuf


@nb.njit(cache=True)
def outlines(zbuf, mol, H, W, px_per_nm, nolo, nohi, dz_in):
    """returns 0 none, 1 internal contour, 2 molecular outline"""
    out = np.zeros((H, W), np.int8)
    for i in range(H):
        for j in range(W):
            z0 = zbuf[i, j]; m0 = mol[i, j]
            best = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ii = i + di; jj = j + dj
                    if ii < 0 or jj < 0 or ii >= H or jj >= W:
                        continue
                    z1 = zbuf[ii, jj]; m1 = mol[ii, jj]
                    if m0 < 0:
                        continue
                    if m1 != m0:
                        if z1 > z0 - 0.01 or m1 < 0:   # we are in front: draw the silhouette on our side
                            best = 2
                    elif (m0 < nolo or m0 >= nohi) and abs(z1 - z0) > dz_in:
                        if best < 1:
                            best = 1
            out[i, j] = best
    return out


def render(pos, rad, col, mol, bounds, view="y", px_per_nm=6.0, ycut=0.0, depth=40.0, ss=2,
           bg_fn=None, fog=(0.97, 0.96, 0.92), fog_strength=0.55, outline_rgb=(0.12, 0.10, 0.10),
           dz_in=1.6, line_w=1, no_inner=(0, 0), outline_alpha=1.0):
    """pos: (N,3) nm; image horizontal axis = z (bounds[0]), vertical = x (bounds[1]); camera looks along +y."""
    (z0, z1), (x0, x1) = bounds
    sel = (pos[:, 1] >= ycut) & (pos[:, 1] <= ycut + depth) & (pos[:, 2] > z0 - 5) & (pos[:, 2] < z1 + 5) & (pos[:, 0] > x0 - 5) & (pos[:, 0] < x1 + 5)
    p, r, c, m = pos[sel], rad[sel], col[sel], mol[sel]
    s = px_per_nm * ss
    W = int((z1 - z0) * s); H = int((x1 - x0) * s)
    U = (p[:, 2] - z0) * s; V = (p[:, 0] - x0) * s; D = p[:, 1].astype(np.float64)
    zbuf, ib = raster(U, V, D, r * s, np.arange(len(p)), H, W)
    hit = ib >= 0
    molbuf = np.where(hit, m[np.clip(ib, 0, None)], -1)
    img = np.zeros((H, W, 3))
    if bg_fn is not None:
        zz = z0 + (np.arange(W) + 0.5) / s; xx = x0 + (np.arange(H) + 0.5) / s
        img[:] = bg_fn(zz[None, :], xx[:, None])
    else:
        img[:] = fog
    cc = c[np.clip(ib, 0, None)]
    t = np.clip((zbuf - ycut) / depth, 0, 1)[..., None] * fog_strength
    shaded = cc * (1 - t) + np.asarray(fog) * t
    img[hit] = shaded[hit]
    ol = outlines(zbuf, molbuf, H, W, s, no_inner[0], no_inner[1], dz_in)
    if line_w > 1:
        from scipy.ndimage import maximum_filter
        ol2 = maximum_filter((ol == 2).astype(np.uint8), size=line_w)
        ol = np.where(ol2 > 0, 2, ol)
    ocol = np.asarray(outline_rgb)
    img[ol == 1] = img[ol == 1] * 0.55 + ocol * 0.45
    img[ol == 2] = img[ol == 2] * (1 - outline_alpha) + ocol * outline_alpha
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    if ss > 1:
        im = im.resize((W // ss, H // ss), Image.LANCZOS)
    return im
