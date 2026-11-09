# Fetch the AFDB (NVIDIA) dimer models selected by jurkat/inventory/afdb_dimers.py, one file per model from the AFDB
# file server (needs a browser User-Agent; plain curl gets 403). Unpublished model IDs return 404 -> listed for fallback.
# Output: data/structures/afdb_complexes/models/{id}.cif.gz, fetch_status.tsv
import os, time, gzip, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
O = "data/structures/afdb_complexes/models"
ids = list(pd.read_csv("data/derived/afdb_homodimers_v0.tsv", sep="\t").model) + list(pd.read_csv("data/derived/afdb_heterodimers_v0.tsv", sep="\t").model)
H = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) jurkat-cell-model/0.1 (academic; ETH Zurich)"}
def get(i):
    fn = f"{O}/{i}.cif.gz"
    if os.path.exists(fn): return i, "cached"
    for t in range(3):
        try:
            r = requests.get(f"https://alphafold.ebi.ac.uk/files/{i}-model_v1.cif", headers=H, timeout=60)
            if r.status_code == 404: return i, "404"
            if r.status_code == 200:
                b = r.content
                open(fn, "wb").write(b if b[:2] == b"\x1f\x8b" else gzip.compress(b)); return i, "ok"
        except Exception:
            pass
        time.sleep(5)
    return i, "failed"
with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(get, ids))
S = pd.DataFrame(res, columns=["model", "status"]); S.to_csv(f"{O}/../fetch_status.tsv", sep="\t", index=False)
print(S.status.value_counts().to_dict())
