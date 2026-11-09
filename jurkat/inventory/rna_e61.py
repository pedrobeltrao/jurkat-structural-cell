# E6-1 RNA consensus (PLANNING_NOTES §4b): per-gene TPM from 5 E6-1-like bulk sources, renormalised to protein-coding
# genes, plus Hart-style zTPM per source and an expression call.
# Output: data/derived/rna_e61.tsv (index = HGNC symbol; per-source TPM, median TPM, n_sources_expr, call).
import pandas as pd, numpy as np, glob
from scipy.stats import gaussian_kde
T = "data/transcriptomics/"
h = pd.read_csv(T + "hpa/hpa_v_rna_celline_Jurkat.tsv", sep="\t")
ens2sym = dict(zip(h.Gene, h["Gene name"]))
pc = set(pd.read_csv("data/reference/uniprot_human_reviewed.tsv.gz", sep="\t", usecols=["Gene Names (primary)"]).iloc[:, 0].dropna())
rna = {}
rna["HPA_E6.1"] = h[h["Cell line"] == "Jurkat E6.1"].groupby("Gene name").TPM.max()
e = pd.read_csv(T + "expression_atlas_E-MTAB-2706/E-MTAB-2706_Jurkat_tpm.tsv", sep="\t")
rna["E-MTAB-2706"] = e.groupby("Gene Name")[e.columns[-1]].max()
def rsem(f):
    x = pd.read_csv(f, sep="\t"); x["sym"] = x.gene_id.str.split(".").str[0].map(ens2sym); return x.groupby("sym").TPM.sum()
rna["GSE276689"] = pd.concat([rsem(f) for f in sorted(glob.glob(T + "geo_GSE276689/GSM850415[01]*"))], axis=1).mean(axis=1)
def fc(f):
    x = pd.read_csv(f, sep="\t", comment="#"); c = x.columns[-1]; rpk = x[c] / x.Length
    return pd.Series((rpk / rpk.sum() * 1e6).values, index=x.Geneid.map(ens2sym)).groupby(level=0).sum()
rna["GSE145453"] = pd.concat([fc(f) for f in sorted(glob.glob(T + "geo_GSE145453/GSM431850[56]*"))], axis=1).mean(axis=1)
x = pd.read_csv(glob.glob(T + "geo_GSE301789/GSM9089802*deseq2.annotation*")[0], sep="\t")
symcol = [c for c in x.columns if c.lower() in ("gene_name", "genename", "symbol", "gene_symbol", "gene")] or \
         [c for c in x.columns if "name" in c.lower() or "symbol" in c.lower()][:1]
ct = [c for c in x.columns if c.endswith("_tpm") and "WT" in c and "_CT" in c and c[-5].isdigit()]
rna["GSE301789"] = x.groupby(symcol[0])[ct].sum().mean(axis=1)
R = pd.DataFrame(rna)
R = R.loc[R.index.isin(pc)].fillna(0)
R = R / R.sum() * 1e6                                  # protein-coding TPM
def hart(s):                                           # Hart 2013 zTPM: KDE mode, right-half sd, on log2 TPM>0
    v = np.log2(s[s > 0].values); g = np.linspace(-8, 16, 2000); k = gaussian_kde(v, 0.1)(g)
    mu = g[k.argmax()]; sd = np.sqrt(np.mean((v[v > mu] - mu) ** 2))
    return (np.log2(s.clip(lower=1e-6)) - mu) / sd
Z = R.apply(hart)
out = R.add_suffix("_tpm").join(Z.add_suffix("_z"))
out["tpm_median"] = R.median(axis=1); out["z_median"] = Z.median(axis=1)
out["n_expr"] = (Z >= -3).sum(axis=1); out["n_off"] = (Z < -2.5).sum(axis=1)
out["call"] = np.where(out.n_expr >= 2, "expressed", np.where(out.n_off == R.shape[1], "off", "low"))
out.index.name = "gene"; out.to_csv("data/derived/rna_e61.tsv", sep="\t")
print(R.shape, out.call.value_counts().to_dict()); print(R.corr(method="spearman").round(2))
