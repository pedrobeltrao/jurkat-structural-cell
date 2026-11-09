"""Rebuild only the transmembrane assemblies (after the SIFTS mapping fix + AF TM grafting); updates lib_build_log.tsv."""
import pandas as pd, jlib_build as jb
I = pd.read_csv(f"{jb.V}/data/derived/ingredients_v0.tsv", sep="\t")
L = jb.assemblies(I[I.transmembrane & (I.kind != "protein")])
fn = f"{jb.V}/data/derived/lib_build_log.tsv"; old = pd.read_csv(fn, sep="\t")
pd.concat([old[~old.ing_id.isin(L.ing_id)], L]).to_csv(fn, sep="\t", index=False)
n = L.note.fillna("")
print("rebuilt", len(L), "| oriented", n.str.contains("oriented").sum(), "| orient failed", n.str.contains("orient_failed").sum(),
      "| grafted", n.str.contains(r"graft: [1-9]").sum(), "| graft attempted but none", n.str.contains("graft: none").sum())
print(L[n.str.contains("orient_failed")].note.str.extract(r"orient_failed:([^;]*)")[0].value_counts().to_string())
