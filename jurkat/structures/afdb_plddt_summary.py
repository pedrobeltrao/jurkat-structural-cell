# One pass over the shared AFDB v6 human proteome tar: per-model CA pLDDT summary (for the disordered-mass report and
# for trimming low-confidence regions, as in yeast: pLDDT < 50 removed). Fragments (F2.., proteins > 2700 aa) are kept
# as separate rows. Output: data/derived/afdb_plddt_summary.tsv (+ per-residue pLDDT strings for trimming later).
import tarfile, gzip, io, numpy as np, pandas as pd
TAR = __import__("os").environ.get("AFDB_HUMAN_TAR", "UP000005640_9606_HUMAN_v6.tar")
rows = []
with tarfile.open(TAR) as t:
    for m in t:
        if not m.name.endswith(".pdb.gz"): continue
        acc, frag = m.name.split("-")[1], m.name.split("-")[2]
        b = []
        for line in gzip.open(t.extractfile(m), "rt"):
            if line.startswith("ATOM") and line[12:16] == " CA ": b.append(float(line[60:66]))
        b = np.array(b)
        rows.append(dict(uniprot=acc, frag=frag, n_res=len(b), mean_plddt=round(b.mean(), 1),
                         frac_lt50=round((b < 50).mean(), 4), frac_lt70=round((b < 70).mean(), 4),
                         plddt=",".join(str(int(x)) for x in b)))
        if len(rows) % 5000 == 0: print(len(rows), flush=True)
pd.DataFrame(rows).to_csv("data/derived/afdb_plddt_summary.tsv.gz", sep="\t", index=False)
print("done", len(rows))
