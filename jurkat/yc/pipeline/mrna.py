"""Full-length mRNAs (inventory v3, item C1): every mRNA is threaded through the cytosol at its full length before
proteins are packed, so the packer fills the space around them (as for chromatin).

Lengths: lognormal transcripts (mean MEAN_NT including UTRs and poly(A), sd LOG_SIGMA in log space), one bead per
nucleotide, STEP nm apart (single-stranded RNA, persistence ~1-2 nm: the walk keeps PERSIST of its direction).
Polysome mRNAs already contain their ribosome channels and the short linkers between ribosomes (polysome.py); the rest
of each transcript becomes a 3' tail leaving the entry of the first ribosome and a 5' tail leaving the exit of the
last one (TAIL5_FRACTION of the remainder on the 5' side: 5'UTR + the ORF upstream of the ribosomes). Free mRNAs
(transcripts without ribosomes) start at random cytosolic points. Threads avoid membranes, all pre-placed objects
and each other; a thread that gets stuck ends early.
"""
import numpy as np
import numba as nb
import pack3d as p3
import cell3d as c3

MEAN_NT = 1500.0
LOG_SIGMA = 0.5
MIN_NT = 300
STEP = 0.6            # nm per nucleotide bead
BEAD_R = 0.5
PERSIST = 1.0         # direction weight against unit gaussian noise (0.9 in the old in-tile threads)
TAIL5_FRACTION = (0.1, 0.4)
H = 10.0


@nb.njit(cache=True)
def thread_all(seed, glo, ghi, G, pores, pidx, PT, PQ, PC, OC, OR, starts, dirs, lens, R0, cap_s, cap_e,
               step, bead_r, persist, mask, maxd, nstraight):
    """One self-avoiding thread per row of lens. starts[i] with |dirs[i]| > 0: thread anchored there (polysome tails,
    FG domains; it first steps straight out along dirs[i] until clear, <= nstraight beads); dirs[i] == 0: random start
    in an allowed compartment. Beads must lie in compartments of the label bitmask `mask`, clear of membranes, and
    (maxd > 0) within maxd of the thread's start. Returns points and offsets."""
    np.random.seed(seed)
    dims, head, nxt, sid, cen, rad, cnt = p3.grid_new(glo, ghi, H, cap_s, cap_e)
    p3.g_insert(glo, H, dims, head, nxt, sid, cen, rad, cnt, OC, OR)
    n = lens.shape[0]
    pts = np.empty((lens.sum() + nstraight * n, 3), np.float32); off = np.zeros(n + 1, np.int64)
    out = np.zeros(c3.NMEM); bead = np.empty((1, 3)); prev = np.empty((1, 3)); br = np.full(1, bead_r)
    nb_ = 0; nst = 0
    for c in range(n):
        anchored = dirs[c, 0] * dirs[c, 0] + dirs[c, 1] * dirs[c, 1] + dirs[c, 2] * dirs[c, 2] > 0.0
        ok = False
        if anchored:
            # leave the anchor (ribosome channel, nucleoporin domain) straight out until clear of its envelope
            nst = 0
            for j in range(1, nstraight + 1):
                for k in range(3):
                    bead[0, k] = starts[c, k] + step * j * dirs[c, k]
                if not p3.g_overlap(glo, H, dims, head, nxt, sid, cen, rad, bead, br, 0.0):
                    ok = True; nst = j
                    break
            if ok:
                for j in range(1, nst):
                    for k in range(3):
                        pts[nb_, k] = starts[c, k] + step * j * dirs[c, k]
                        prev[0, k] = pts[nb_, k]
                    if j <= nst - 2:      # the bead next to the walk start is inserted with the walk's lag
                        p3.g_insert(glo, H, dims, head, nxt, sid, cen, rad, cnt, prev, br)
                    nb_ += 1
            dx, dy, dz = dirs[c, 0], dirs[c, 1], dirs[c, 2]
        else:
            for at in range(500):
                while True:
                    for k in range(3):
                        bead[0, k] = (np.random.random() * 2 - 1) * R0
                    if bead[0, 0] ** 2 + bead[0, 1] ** 2 + bead[0, 2] ** 2 < R0 * R0:
                        break
                if p3.g_overlap(glo, H, dims, head, nxt, sid, cen, rad, bead, br, 0.0):
                    continue
                m = c3.mem_sdfs(bead[0, 0], bead[0, 1], bead[0, 2], G, pores, pidx, PT, PQ, PC, out)
                if (mask >> c3.label_from(bead[0, 0], bead[0, 1], bead[0, 2], G, pores, out, m)) & 1 and c3.min_mem_dist(out, -1) > bead_r + c3.HALF:
                    ok = True
                    break
            dx = np.random.normal(); dy = np.random.normal(); dz = np.random.normal()
            dn = np.sqrt(dx * dx + dy * dy + dz * dz); dx /= dn; dy /= dn; dz /= dn
        if not ok:
            off[c + 1] = nb_
            continue
        for k in range(3):
            pts[nb_, k] = bead[0, k]
        nb_ += 1; nthis = 1 + (nst - 1 if anchored else 0); stuck = 0; pending = False
        while nthis < lens[c] and stuck < 300:
            ex = persist * dx + np.random.normal(); ey = persist * dy + np.random.normal(); ez = persist * dz + np.random.normal()
            en = np.sqrt(ex * ex + ey * ey + ez * ez); ex /= en; ey /= en; ez /= en
            bead[0, 0] = pts[nb_ - 1, 0] + step * ex
            bead[0, 1] = pts[nb_ - 1, 1] + step * ey
            bead[0, 2] = pts[nb_ - 1, 2] + step * ez
            if maxd > 0.0:
                ddx = bead[0, 0] - pts[off[c], 0]; ddy = bead[0, 1] - pts[off[c], 1]; ddz = bead[0, 2] - pts[off[c], 2]
                if ddx * ddx + ddy * ddy + ddz * ddz > maxd * maxd:
                    stuck += 1; continue
            if p3.g_overlap(glo, H, dims, head, nxt, sid, cen, rad, bead, br, 0.0):
                stuck += 1; continue
            m = c3.mem_sdfs(bead[0, 0], bead[0, 1], bead[0, 2], G, pores, pidx, PT, PQ, PC, out)
            if not ((mask >> c3.label_from(bead[0, 0], bead[0, 1], bead[0, 2], G, pores, out, m)) & 1) or c3.min_mem_dist(out, -1) < bead_r + c3.HALF:
                stuck += 1; continue
            if pending:   # insert the previous bead (lagged, so consecutive beads do not self-collide)
                if not p3.g_insert(glo, H, dims, head, nxt, sid, cen, rad, cnt, prev, br):
                    off[c + 1] = nb_
                    return pts[:nb_], off
            for k in range(3):
                prev[0, k] = pts[nb_ - 1, k]
            pending = True
            for k in range(3):
                pts[nb_, k] = bead[0, k]
            nb_ += 1; nthis += 1
            dx, dy, dz = ex, ey, ez; stuck = 0
        for back in range(min(2, nthis)):
            for k in range(3):
                prev[0, k] = pts[nb_ - 1 - back, k]
            p3.g_insert(glo, H, dims, head, nxt, sid, cen, rad, cnt, prev, br)
        off[c + 1] = nb_
    return pts[:nb_], off


def lengths(rng, n):
    mu = np.log(MEAN_NT) - LOG_SIGMA ** 2 / 2
    return np.maximum(MIN_NT, rng.lognormal(mu, LOG_SIGMA, n)).astype(np.int64)


def place(cell, poly, ch, n_free, obstacles, seed=29, log=print):
    """poly: polysome dict (polysome.place); ch: mRNA channel path in the ribosome frame (5'->3');
    n_free: free mRNAs; obstacles: (centres, radii) of everything pre-placed in the cytosol."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    tex = ch[0] - ch[3]; tex /= np.linalg.norm(tex)
    ten = ch[-1] - ch[-4]; ten /= np.linalg.norm(ten)
    chn = poly["poly_chain"]; Rm = Rotation.from_quat(np.c_[poly["poly_q"][:, 1:], poly["poly_q"][:, :1]].astype(np.float64)).as_matrix()
    off = poly["poly_mrna_off"]
    starts, dirs, lens, kind = [], [], [], []
    chains = np.unique(chn)
    L = lengths(rng, len(chains) + n_free)
    drawn = 0
    for ci, c in enumerate(chains):
        idx = np.where(chn == c)[0]; first, last = idx[0], idx[-1]
        P = poly["poly_mrna"][off[first]:off[last + 1]]
        have = int(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() / STEP)
        drawn += have
        rest = max(0, L[ci] - have)
        f5 = rng.uniform(*TAIL5_FRACTION)
        # 3' tail from the entry of the first ribosome, 5' tail from the exit of the last one (outward directions)
        starts.append(Rm[first] @ ch[-1] + poly["poly_pos"][first]); dirs.append(Rm[first] @ ten); lens.append(int(rest * (1 - f5))); kind.append(1)
        starts.append(Rm[last] @ ch[0] + poly["poly_pos"][last]); dirs.append(Rm[last] @ tex); lens.append(int(rest * f5)); kind.append(2)
    for i in range(n_free):
        starts.append(np.zeros(3)); dirs.append(np.zeros(3)); lens.append(int(L[len(chains) + i])); kind.append(0)
    starts = np.array(starts, np.float64); dirs = np.array(dirs, np.float64); lens = np.maximum(np.array(lens, np.int64), 1)
    OC, OR = obstacles
    G = cell.G
    glo = np.full(3, -G[0] - 10.0); ghi = -glo
    a = cell.args(glo, ghi, margin=50.0)
    cap_s = int(lens.sum()) + len(OR) + 1000
    pts, toff = thread_all(seed, glo, ghi, *a, OC.astype(np.float64), OR.astype(np.float64), starts, dirs, lens, G[2] - 30.0,
                           cap_s, int(cap_s * 3 + 8 * len(OR)), STEP, BEAD_R, PERSIST, 1 << c3.CYTO, 0.0, 20)
    got = np.diff(toff)
    kind = np.array(kind, np.int8)
    log(f"mRNA: {len(chains)} polysome mRNAs (+{got[kind > 0].sum() / 1e6:.1f}M nt of tails to {drawn / 1e6:.1f}M nt through "
        f"ribosomes) and {n_free} free mRNAs ({got[kind == 0].sum() / 1e6:.1f}M nt); threads reached "
        f"{got.sum() / lens.sum():.1%} of their target length; total {(got.sum() + drawn) / 1e6:.1f}M nt")
    return dict(mrna_pts=pts, mrna_off=toff, mrna_kind=kind, mrna_target=lens.astype(np.int32))
