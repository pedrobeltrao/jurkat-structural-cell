"""Missing-cap estimate below the open face of cell 1 (v5; PLANNING_NOTES §7g open boundary) -> summary["open_face"].

Cross-section area A(z) of the cell (labels != exterior / outside-volume) every 16 nm from the face up to 1.5 um, from the
final geometry (cell3d labels). Side-wall extrapolation: r(z)^2 = A(z)/pi is fitted near the face and extended below it to
r = 0. The near-face growth is steep (41 -> 71 um2 in 0.26 um; a sphere of the cell's size would grow ~4x slower), so the
bottom is flattened and a sphere-based cap overestimates the missing part. DECISION (v5): linear r^2 fit over the first
FIT_UM = 0.32 um (the steep regime, before the cross-section saturates at ~91 um2); the quadratic 1 um fit and a
sphere-matched cap are reported as alternatives. f_vol and f_pm (jwc.step_counts copy-number scaling) are rewritten.
usage: jface_cap.py   (rewrites data/derived/jgeom_cell1.npz summary in place)"""
import os, sys, json, numpy as np
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell3d as c3

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
FN = f"{V}/data/derived/jgeom_cell1.npz"
H = 16.0
FIT_UM = 0.32


def cap(z, A, fit_um, deg):
    m = z <= fit_um + 1e-9
    c = np.polyfit(z[m], A[m] / np.pi, deg)
    roots = np.roots(c); roots = roots[np.isreal(roots)].real; roots = roots[roots < 0]
    z0 = roots.max() if len(roots) else -1.0
    zs = np.linspace(z0, 0.0, 400); r2 = np.clip(np.polyval(c, zs), 0, None); r = np.sqrt(r2)
    vol = float(np.trapezoid(np.pi * r2, zs)); dr = np.gradient(r, zs)
    area = float(np.trapezoid(2 * np.pi * r * np.sqrt(1 + dr ** 2), zs)) + 0.0
    return dict(height_um=round(-float(z0), 3), vol_um3=round(vol, 2), area_um2=round(area, 2), fit_um=fit_um, deg=deg)


def main():
    z = dict(np.load(FN)); cell = c3.Cell(z["G"], z["pores"]); G = cell.G
    summ = json.loads(str(z["summary"])); of = summ["open_face"]; zf = G[c3.G_ZFACE]
    R = G[0]; g = np.arange(-R, R, H) + H / 2
    X, Y = np.meshgrid(g, g, indexing="ij")
    zs = zf + 0.5 + np.arange(0, 1500, H)
    A = []
    for zz in zs:
        lab = cell.labels(np.stack([X.ravel(), Y.ravel(), np.full(X.size, zz)], 1))
        A.append(float((~np.isin(lab, [c3.EXT, c3.OUTV])).sum() * H * H / 1e6))
    A = np.array(A); zu = (zs - zf) / 1e3
    main_ = cap(zu, A, FIT_UM, 1)
    alt = dict(quadratic_1um=cap(zu, A, 1.0, 2), linear_0p5um=cap(zu, A, 0.5, 1))
    Rm = (summ["cell"]["vol_sdf_um3"] * 3 / 4 / np.pi) ** (1 / 3)              # sphere of the cell's volume
    h = Rm - np.sqrt(max(Rm ** 2 - A[0] / np.pi, 0))                            # sphere cap with the footprint as its base
    alt["sphere_matched"] = dict(height_um=round(h, 3), vol_um3=round(np.pi * h * h * (3 * Rm - h) / 3, 2),
                                 area_um2=round(2 * np.pi * Rm * h, 2))
    vol, area = of["cell_vol_modelled_um3"], of["pm_area_modelled_um2"]
    of.update(cap_height_um=main_["height_um"], cap_vol_um3=main_["vol_um3"], cap_area_um2=main_["area_um2"],
              f_vol=round(vol / (vol + main_["vol_um3"]), 4), f_pm=round(area / (area + main_["area_um2"]), 4),
              footprint_um2=round(float(A[0]), 2), method=f"side-wall extrapolation, r^2 linear in z over the first {FIT_UM} um",
              alternatives=alt, A_profile_um2=[round(float(a), 1) for a in A[::2]], A_profile_step_nm=2 * H)
    of.pop("fit", None); of.pop("A_first_um", None)
    summ["open_face"] = of; z["summary"] = json.dumps(summ)
    print(json.dumps(of, indent=1, default=float))
    tmp = FN[:-4] + ".tmp.npz"; np.savez_compressed(tmp, **z); os.replace(tmp, FN)
    print("updated", FN)


if __name__ == "__main__":
    main()
