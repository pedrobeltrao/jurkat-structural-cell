"""v7 virtual volume regions (jpack_test.VIRTUAL; JWC_REGIONS=v7): cytosol voxels selected by distance rules.

  mt_lattice  centre 14-26 nm from an MT axis (MT outer radius 12.5 nm; a MAP touching the lattice): lattice MAPs (R7)
  mt_plus     the same band, restricted to the plus-end zone: the last PLUS_L nm of every MT centreline of length
              >= PLUS_MIN_L (segments shorter than that are mostly trace fragments, not whole MTs); plus end by the
              stage-3 lattice convention (minus end = end nearer the NE; aster MTs start at the MTOC): +TIP proteins (R7)
  pcm         within R_PCM nm of the centriole-pair centre (cytoskeleton pre-pass json, log.centrioles.new_centre_nm):
              pericentriolar material / centriolar satellites (R6c)
  vim_cage    cytosol within VIM_D nm of the outer nuclear membrane surface (nearest NE-midsurface mesh vertex): perinuclear vimentin cage (R6d; filaments
              are not built, the vimentin pool is enriched where the cage is)
Context (MT axes, plus-end zones, PCM centre) is loaded once before the worker fork (jwc)."""
import json, numpy as np
from scipy.spatial import cKDTree
import cell3d as c3

BAND = (14.0, 26.0)
PLUS_L = 400.0
PLUS_MIN_L = 1000.0
R_PCM = 400.0
VIM_D = 300.0


class Axes:
    def __init__(self, A, B):
        self.A, self.B = A, B; self.M = 0.5 * (A + B)
        self.hl = float(0.5 * np.linalg.norm(B - A, axis=1).max()) if len(A) else 0.0
        self.tree = cKDTree(self.M) if len(A) else None

    def dist(self, P, thr):
        out = np.full(len(P), np.inf)
        if self.tree is None or not len(P): return out
        d0, _ = self.tree.query(P, k=1, distance_upper_bound=thr + self.hl)
        cand = np.nonzero(np.isfinite(d0))[0]
        if not len(cand): return out
        lst = self.tree.query_ball_point(P[cand], thr + self.hl)
        for i, js in zip(cand, lst):
            js = np.asarray(js); a = self.A[js]; d = self.B[js] - a
            t = np.clip(np.einsum("ij,ij->i", P[i] - a, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-9), 0, 1)
            out[i] = float(np.sqrt(((a + t[:, None] * d - P[i]) ** 2).sum(1)).min())
        return out


def _segs(polys):
    A = [p[:-1] for p in polys if len(p) > 1]; B = [p[1:] for p in polys if len(p) > 1]
    return (np.concatenate(A), np.concatenate(B)) if A else (np.zeros((0, 3)), np.zeros((0, 3)))


def load_context(centerlines, cyto_json, geom=None, ne_mesh=None):
    z = np.load(centerlines); P = z["points_model_nm"].astype(np.float64); off = z["offsets"]; fl = z["flipped_for_lattice"]
    polys = [P[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    plus = []
    for p, f in zip(polys, fl):
        q = p[::-1] if f else p                          # lattice order: minus -> plus
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        if L[-1] < PLUS_MIN_L: continue
        plus.append(q[L >= L[-1] - PLUS_L])
    J = json.load(open(cyto_json))
    cen = np.array(J["log"]["centrioles"]["new_centre_nm"], np.float64)
    ctx = dict(ax=Axes(*_segs(polys)), plus=Axes(*_segs(plus)), pcm=cen, n_plus=len(plus),
               plus_um=float(sum(np.linalg.norm(np.diff(q, axis=0), axis=1).sum() for q in plus) / 1e3))
    if geom is not None and ne_mesh is not None:            # NE midsurface vertices (56 nm mesh): distance beyond the
        import trimesh                                       # +-120 nm band of the stored NE field
        m = trimesh.load(ne_mesh, process=False)
        ctx["ne_tree"] = cKDTree(np.asarray(m.vertices, np.float64) - np.load(geom)["centre_nm"])
    return ctx


def masks(P, lab, S, ctx, names):
    """boolean masks (len(P)) for each virtual region name; P points, lab labels, S membrane SDFs (n, NMEM) or None
    (then ctx["memfun"](Q) evaluates them on the cytosol points only)"""
    cyt = lab == c3.CYTO; out = {}
    idx = np.nonzero(cyt)[0]; Q = P[idx]
    need_ax = any(n in names for n in ("mt_lattice", "mt_plus"))
    if need_ax:
        d = ctx["ax"].dist(Q, BAND[1]); band = (d >= BAND[0]) & (d <= BAND[1])
    for n in names:
        m = np.zeros(len(P), bool)
        if n == "mt_lattice":
            m[idx[band]] = True
        elif n == "mt_plus":
            if band.any():
                dp = ctx["plus"].dist(Q[band], BAND[1]); bi = idx[band]
                m[bi[(dp >= BAND[0]) & (dp <= BAND[1])]] = True
        elif n == "pcm":
            m[idx[np.linalg.norm(Q - ctx["pcm"], axis=1) <= R_PCM]] = True
        elif n == "vim_cage":
            # distance to the ONM surface ~ nearest NE-midsurface vertex - PERI_W/2 - MEM_HALF (vertex spacing ~56 nm:
            # overestimates by <= ~20 nm, irrelevant at VIM_D = 300 nm)
            d, _ = ctx["ne_tree"].query(Q, k=1, distance_upper_bound=VIM_D + 0.5 * c3.PERI_W + 50.0)
            ne = d - 0.5 * c3.PERI_W - c3.MEM_HALF[c3.M_NE]
            m[idx[ne <= VIM_D]] = True
        out[n] = m
    return out
