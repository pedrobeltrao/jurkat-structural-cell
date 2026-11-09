"""Copy conservation of the packing inventory against the proteome (run after s1).
Complex copy numbers are the median over subunits of pool/stoichiometry, so some subunits are used beyond their
measured copies (mass created) while others stay behind as monomers. Reports the excess per gene and in total."""
from _common import *
import inventory as inv, ingredients as ingm

df, _ = inv.build()
ing, _ = load_inventory()
mass = dict(zip(df.gene, df.mass_da))
_, cons = ingm.build_ingredients(df)                       # curated complexes
for r in pd.read_csv(f"{INV}/complexes_added.csv").itertuples():
    for g, s in json.loads(r.subunits).items():
        cons[g] = cons.get(g, 0) + r.copies * s
t = pd.DataFrame(dict(gene=df.gene, proteome=df.copies_per_cell, mass=df.mass_da))
t["in_complexes"] = t.gene.map(cons).fillna(0.0)
t["excess"] = (t.in_complexes - t.proteome).clip(lower=0)
rp = t.gene.str.match(r"^RP[LS]\d|^RPP\d")
tot = (t.proteome * t.mass).sum()
print(f"subunit copies used beyond the proteome: {(t.excess * t.mass).sum() / tot:.2%} of proteome mass "
      f"(ribosomal proteins {(t.excess * t.mass)[rp].sum() / tot:.2%})")
print(t[~rp].sort_values("excess", ascending=False).head(12)[["gene", "proteome", "in_complexes", "excess"]].round(0).to_string(index=False))
print(f"packing inventory mass / proteome mass = {(ing.copies_per_cell * ing.mass_da).sum() / tot:.4f}")
