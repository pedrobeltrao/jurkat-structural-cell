# SEC-based assembly state per protein (Python stand-in for CCprofiler calibrateMW + annotateMassDistribution).
# Data: Jurkat SEC-SWATH, Fossati 2021 (PXD018033), 60 fractions, mild lysis (0.5% NP-40).
#  1. protein profiles = sum of top-3 proteotypic peptides per fraction (1% FDR)
#  2. MW calibration at peak level: for each clean Complex Portal calibrant (scratch_checks/sec_calibrants_clean.tsv is
#     re-derived here), the co-elution peak = argmax over fractions of the MIN across normalised subunit profiles
#     (all subunits present); robust (Theil-Sen) fit log10 MW ~ fraction
#  3. per protein: fraction of intensity eluting at apparent MW >= 2x monomer MW = assembled fraction (lower bound)
# Output: data/derived/sec_assembly.tsv; prints calibration and comparison with inventory frac_in_complexes.
import re, itertools, numpy as np, pandas as pd
from scipy.stats import theilslopes, spearmanr
from scipy.ndimage import uniform_filter1d
D = "data/proteomics/fossati2021_jurkat_SEC_PXD018033/"
fm = pd.read_csv(D + "2020_Methods_in_molecular_biology_information_file.txt", sep="\t", skiprows=4).dropna()
fm.columns = ["file", "frac"]; fm = fm[fm.frac.astype(str).str.isdigit()]; fm["frac"] = fm.frac.astype(int)
fa = pd.read_csv(D + "feature_alignment.csv", sep="\t", usecols=["decoy", "filename", "FullPeptideName", "Intensity", "m_score", "ProteinName"])
fa = fa[(fa.decoy == 0) & (fa.m_score < 0.01) & ~fa.ProteinName.str.contains("/|DECOY")]
fa["file"] = fa.filename.str.split("/").str[-1]; fa = fa.merge(fm, on="file")
pep = fa.groupby(["ProteinName", "FullPeptideName", "frac"]).Intensity.sum().reset_index()
tot = pep.groupby(["ProteinName", "FullPeptideName"]).Intensity.sum().rename("t").reset_index()
tot["r"] = tot.groupby("ProteinName").t.rank(ascending=False)
prof = pep.merge(tot[tot.r <= 3][["ProteinName", "FullPeptideName"]]).groupby(["ProteinName", "frac"]).Intensity.sum().unstack(fill_value=0)
prof = prof.reindex(columns=range(1, 61), fill_value=0)
prof = pd.DataFrame(uniform_filter1d(prof.values.astype(float), 3, axis=1), index=prof.index, columns=prof.columns)
det = set(prof.index); print("SEC proteins:", len(det))

up = pd.read_csv("data/reference/uniprot_human_reviewed.tsv.gz", sep="\t").set_index("Entry")
cp = pd.read_csv("data/complexes/complex_portal/9606.tsv", sep="\t")
rows = []
for _, r in cp.iterrows():
    parts = re.findall(r"([A-Z][A-Z0-9]{5,9})(?:-\w+)?\((\d+)\)", str(r["Expanded participant list"]))
    if not parts or any(int(s) == 0 for _, s in parts): continue
    sub = {}
    for a, s in parts: sub[a] = sub.get(a, 0) + int(s)
    if len(sub) < 2 or not all(a in det for a in sub) or not all(a in up.index for a in sub): continue
    if any(isinstance(up.Transmembrane.get(a), str) for a in sub): continue
    mw = sum(up.Mass[a] * s for a, s in sub.items()) / 1e3
    if not (40 <= mw <= 5000): continue
    P = prof.loc[list(sub)].values; P = P / P.max(axis=1, keepdims=True)
    co = P.min(axis=0)                                   # co-elution signal: all subunits present
    pk = int(np.argmax(co)) + 1
    corr = np.nanmean([np.corrcoef(P[i], P[j])[0, 1] for i, j in itertools.combinations(range(len(P)), 2)])
    rows.append(dict(cpx=r["#Complex ac"], name=r["Recommended name"], n=len(sub), mw_kda=mw, peak=pk, co_height=co.max(), mcorr=corr))
cal = pd.DataFrame(rows)
cal = cal[(cal.co_height >= 0.5) & (cal.mcorr >= 0.6)]
cal["key"] = cal.name.str.replace(r",.*variant.*|\(.*|variant.*", "", regex=True).str.strip()
cal = cal.sort_values("co_height", ascending=False).drop_duplicates("key")
x, y = cal.peak.values, np.log10(cal.mw_kda.values)
slope, icpt, lo, hi = theilslopes(y, x)
res = y - (icpt + slope * x); keep = np.abs(res) < 2.5 * np.median(np.abs(res)) / 0.6745
slope, icpt, lo, hi = theilslopes(y[keep], x[keep])
res = y[keep] - (icpt + slope * x[keep])
print(f"calibrants: {len(cal)} (kept {keep.sum()} after outlier rejection); Spearman rho {spearmanr(x, y)[0]:.2f} (all), "
      f"{spearmanr(x[keep], y[keep])[0]:.2f} (kept)")
print(f"log10 MW(kDa) = {icpt:.3f} + {slope:.4f} * fraction ; residual SD {res.std():.2f} log10 (x{10**res.std():.1f}); "
      f"MW range at fractions 1..60: {10**(icpt+slope):.0f} .. {10**(icpt+60*slope):.0f} kDa")
cal.assign(kept=keep).to_csv("data/derived/sec_calibrants_peak.tsv", sep="\t", index=False)

# Complex Portal calibrants co-elute poorly in mild lysis (fit above is too flat: rho ~ -0.3). Use instead curated
# proteins of well-established native mass (monomers, oligomeric enzymes, large machines); apex fraction of each.
# Excluded known exceptions: RPL5 (free 5S RNP pool), HNRNPA1 (RNPs), TUBB (polymer/aggregates).
NATIVE = {"PFN1": 15, "CFL1": 18.5, "PPIA": 18, "SOD1": 32, "PRDX1": 44, "TPI1": 53, "PGK1": 45, "PARK7": 40, "YWHAZ": 55,
          "ENO1": 94, "GAPDH": 144, "LDHA": 147, "PKM": 232, "ALDOA": 157, "HSP90AB1": 166, "HSPA8": 71, "EEF2": 95,
          "NPM1": 190, "PSMA1": 750, "PSMB5": 750, "CCT2": 960, "TCP1": 960, "FASN": 540, "ACLY": 480, "VCP": 540}
g2a = {up["Gene Names (primary)"].get(a): a for a in prof.index if a in up.index}
cx = np.array([int(np.argmax(prof.loc[g2a[g]].values)) + 1 for g in NATIVE if g in g2a])
cy = np.log10([NATIVE[g] for g in NATIVE if g in g2a])
slope, icpt, _, _ = theilslopes(cy, cx); res = cy - (icpt + slope * cx)
print(f"curated native-mass calibration: n={len(cx)}, Spearman {spearmanr(cx, cy)[0]:.2f}; log10 MW = {icpt:.3f} + {slope:.4f}*frac; "
      f"residual SD {res.std():.2f} log10 (x{10**res.std():.1f}); fraction 20 -> {10**(icpt+20*slope):.0f} kDa, 60 -> {10**(icpt+60*slope):.0f} kDa")
app_mw = 10 ** (icpt + slope * np.arange(1, 61))
out = []
for a in prof.index:
    if a not in up.index: continue
    m = up.Mass[a] / 1e3; p = prof.loc[a].values; t = p.sum()
    if t <= 0: continue
    out.append(dict(uniprot=a, gene=up["Gene Names (primary)"].get(a), mono_kda=m, apex_frac=int(np.argmax(p)) + 1,
                    apex_app_kda=app_mw[np.argmax(p)], assembled_frac=p[app_mw >= 2 * m].sum() / t,
                    tm=isinstance(up.Transmembrane.get(a), str)))
S = pd.DataFrame(out); S.to_csv("data/derived/sec_assembly.tsv", sep="\t", index=False)
print(f"proteins scored {len(S)}; median assembled fraction {S.assembled_frac.median():.2f}; "
      f">50% assembled: {(S.assembled_frac > 0.5).mean():.1%}")
inv = pd.read_csv("data/derived/inventory_v0_complexed.tsv", sep="\t")
mrg = S.merge(inv[["uniprot", "frac_in_complexes", "copies", "mass_pg"]], on="uniprot")
mrg = mrg[mrg.copies > 0]
r = spearmanr(mrg.assembled_frac, mrg.frac_in_complexes.fillna(0))
print(f"vs inventory complex rule (n={len(mrg)}): Spearman {r[0]:.2f}")
w = mrg.mass_pg
print(f"mass-weighted assembled fraction among SEC-detected proteins: SEC {np.average(mrg.assembled_frac, weights=w):.2f} | "
      f"complex rule {np.average(mrg.frac_in_complexes.fillna(0), weights=w):.2f}")
for b in [(0, 0.01), (0.01, 0.5), (0.5, 1.01)]:
    sel = mrg[(mrg.frac_in_complexes.fillna(0) >= b[0]) & (mrg.frac_in_complexes.fillna(0) < b[1])]
    print(f"  rule frac_in_complexes in [{b[0]},{b[1]}): n={len(sel)}, median SEC assembled {sel.assembled_frac.median():.2f}")
