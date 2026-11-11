"""Ribosome spatial statistics: model vs EM (validation backlog item 2).

EM: ribosome instances of cell 1 from ribo_seg (classes 1 cytosolic, 3 ER sheet, 4 ER tube; nuclear class 2 skipped),
model frame nm = (x*4, y*4, z*3.44) - centre_nm. Model: the polysome pre-pass (EM ribosomes kept or nudged + added
ribosomes grown as polysome chains). EM detection recall is ~0.45-0.55, so the model is thinned at random to the EM
count before comparing nearest-neighbour (NN) distances and the pair-correlation-like count of neighbours within r.
Output: data/derived/validation/ribosome_nn.json (+ printed table)."""
import os, json, numpy as np, pandas as pd
from scipy.spatial import cKDTree

V = (__import__("os").environ.get("VCELL_ROOT", ".") + ""); D = f"{V}/data/derived"; OUT = f"{D}/validation"
R_BINS = [20, 25, 30, 35, 40, 50, 60, 80, 100]


def stats(P, name):
    t = cKDTree(P); d, _ = t.query(P, k=2); nn = d[:, 1]
    cnt = {f"n_within_{r}": round(float(np.mean([len(x) - 1 for x in t.query_ball_point(P[:200000], r)])), 3) for r in (30, 50, 100)}
    h = {f"frac_nn_lt_{r}": round(float((nn < r).mean()), 4) for r in R_BINS}
    return dict(set=name, n=int(len(P)), nn_median=round(float(np.median(nn)), 2), nn_p10=round(float(np.percentile(nn, 10)), 2),
                nn_p90=round(float(np.percentile(nn, 90)), 2), **h, **cnt)


def main():
    os.makedirs(OUT, exist_ok=True)
    c = np.load(os.environ.get("VAL_GEOM", f"{D}/jgeom_cell1.npz"))["centre_nm"]
    E = pd.read_parquet(f"{D}/ribo_seg_instances.parquet"); E = E[(E.cell2 == 1) & (E.cls != 2)]
    PE = E[["x", "y", "z"]].values * np.array([4.0, 4.0, 3.44]) - c
    z = np.load(os.environ.get("VAL_POLY", f"{D}/prepass/polysomes_cell1.npz")); PM = z["pos"].astype(np.float64); org = z["origin"]
    rng = np.random.default_rng(0)
    thin = PM[rng.random(len(PM)) < len(PE) / len(PM)]
    rows = [stats(PE, "EM (cell 1, non-nuclear)"), stats(PM, "model (all)"), stats(thin, "model thinned to EM count"),
            stats(PM[org == org.max()], f"model origin={org.max()}"), stats(PM[org == org.min()], f"model origin={org.min()}")]
    # uniform random reference in the same volume as the thinned model (shuffle within the cell's ribosome hull bounding box is
    # not a good null; use the model positions with random 200 nm jitter as a 'no-clustering' reference)
    jit = thin + rng.normal(0, 120, thin.shape)
    rows.append(stats(jit, "thinned model, 120 nm jitter (no-clustering reference)"))
    T = pd.DataFrame(rows).set_index("set"); print(T.T.to_string())
    json.dump(dict(rows=rows, origin_codes=np.unique(org).tolist(), note="EM recall ~0.5: compare the EM row with 'thinned' row"),
              open(f"{OUT}/ribosome_nn.json", "w"), indent=1)


if __name__ == "__main__":
    main()
