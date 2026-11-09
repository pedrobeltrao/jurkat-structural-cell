"""Shared setup for the stage scripts (s1..s5). Ported from notebook/pipeline_driver_notebook.ipynb."""
import os, sys, json, time, warnings

# numpy 2.x on Apple Silicon (Accelerate) raises spurious FP warnings in matmul; results are unaffected
warnings.filterwarnings("ignore", message=r".*encountered in matmul")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
import numpy as np, pandas as pd
import config as C

INV = f"{C.RESULTS}/inventory"
ING_CSV = f"{INV}/ingredients.csv"           # packing inventory (s1 output)
OLIGO_JSON = f"{INV}/oligo_map.json"         # ingredient -> [library id, chains per instance]
F_ACTIN_JSON = f"{INV}/f_actin.json"         # ACT1 copies reserved for cables and patches (s1 -> s3)
os.makedirs(INV, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def load_inventory():
    ing = pd.read_csv(ING_CSV)
    oligo_map = {k: tuple(v) for k, v in json.load(open(OLIGO_JSON)).items()}
    return ing, oligo_map


def canonical_tiles():
    """Tile list in canonical order (tile_measure.parquet rows are written in this order by s2)."""
    import tiles3d as t3
    names = pd.read_parquet(f"{C.W3D}/tile_measure.parquet").index
    import re
    return [tuple(int(x) for x in re.findall(r"[+-]\d+", n)) for n in names]
