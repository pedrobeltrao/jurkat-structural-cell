"""Read jrc_jurkat-1 label layers in one consistent frame (the _seg / raw-EM frame).

Published-data quirks (PLANNING_NOTES §1, Findings 5-6; verified 2026-11-08):
  * every *_pred layer is flipped in y relative to *_seg, masks, contacts and raw: y_seg = Y - 1 - y_pred
    (exact at s2/s3; nucleus_pred vs nucleus_seg Dice 0.19 -> 0.97 after the flip). read() un-flips them.
  * raw EM (em/fibsem-uint16) is offset from the labels by (0, +9, -2) s0 voxels (x, y, z): raw = label + RAW_OFFSET_S0.
Axes are x, y, z (n5 order). Local copies are used when present, otherwise S3 over HTTP.
"""
import os
import numpy as np
import tensorstore as ts

ROOT = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
LOCAL = f"{ROOT}/data/em/jrc_jurkat-1/jrc_jurkat-1.n5/labels/"
REMOTE = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_jurkat-1/jrc_jurkat-1.n5/"
RAW_OFFSET_S0 = np.array([0, 9, -2])


def open_layer(layer, scale):
    path = f"{LOCAL}{layer}/{scale}/"
    kv = {"driver": "file", "path": path} if os.path.exists(path + "attributes.json") else \
         {"driver": "http", "base_url": f"{REMOTE}labels/{layer}/{scale}/"}
    return ts.open({"driver": "n5", "kvstore": kv}).result()


def read(layer, scale, sl=None):
    """Read layer[sl] (tuple of slices in the _seg frame, x,y,z) with *_pred layers un-flipped in y."""
    a = open_layer(layer, scale)
    sl = tuple(sl) if sl is not None else (slice(None),) * 3
    sl = tuple(s.indices(n)[:2] for s, n in zip(sl, a.shape))
    (x0, x1), (y0, y1), (z0, z1) = sl
    if layer.endswith("_pred"):
        Y = a.shape[1]
        out = a[x0:x1, Y - y1:Y - y0, z0:z1].read().result()
        return out[:, ::-1, :]
    return a[x0:x1, y0:y1, z0:z1].read().result()
