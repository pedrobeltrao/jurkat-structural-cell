"""Stage 2: cell geometry, 200 nm tiling, per-tile region measures and membrane meshes.
Notebook cells 24, 25 (+ the cell-28 re-index to canonical tile order).

  python s2_geometry.py            # real cell outline, nucleus and vacuoles from segmented cryo-FIB/SEM meshes (cellshape.py)
  python s2_geometry.py --sphere   # the analytic cell (sphere, ellipsoidal nucleus, resampled vacuole lobes)
"""
from _common import *
import inventory as inv, cell3d as c3, tiles3d as t3


def main():
    df, geo = inv.build()
    shapes = None
    if "--sphere" not in sys.argv:
        import cellshape
        shapes = cellshape.build(geo, np.random.default_rng(11), log=log)
    cell = c3.build(geo, seed=11, log=log, shapes=shapes)
    cell.to_json(f"{C.W3D}/geometry.json")
    log("prims", len(cell.PT), "pores", len(cell.pores))
    # Monte-Carlo volume check (4M points in the cell sphere)
    rng = np.random.default_rng(0); R0 = cell.G[0]
    P = rng.uniform(-R0, R0, (4_000_000, 3)); P = P[np.linalg.norm(P, axis=1) < R0]
    lab = c3.label_points(P, cell.G, cell.pores, np.arange(len(cell.PT)), cell.PT, cell.PQ, cell.PC)
    vol = pd.Series(np.bincount(lab, minlength=15) / len(P) * 4 / 3 * np.pi * (R0 / 1000) ** 3, index=c3.LABEL_NAMES)
    log("MC volumes (um3):", vol.round(3).to_dict())

    tiles = t3.tile_list(cell)
    t0 = time.time()
    gres = t3.run_geometry(f"{C.W3D}/geometry.json", tiles)
    log(f"{len(gres)} tiles in {time.time() - t0:.0f}s")
    # results arrive in imap order: re-index to the canonical tile list
    by = {r[0]: r for r in gres}
    names = [t3.tile_name(t) for t in tiles]
    measure = pd.DataFrame(np.array([by[t][1] for t in tiles]), columns=t3.REGION_NAMES, index=names)
    labvol = pd.DataFrame(np.array([by[t][2] for t in tiles]), columns=c3.LABEL_NAMES, index=names)
    measure.to_parquet(f"{C.W3D}/tile_measure.parquet"); labvol.to_parquet(f"{C.W3D}/tile_labvol.parquet")
    tot = measure.sum()
    log("volumes (um3):", (tot[t3.VOL_REGIONS] / 1e9).round(3).to_dict())
    log("areas (um2):", (tot[["mem_" + m for m in t3.MEM_REGIONS]] / 1e6).round(2).to_dict())


if __name__ == "__main__":
    main()
