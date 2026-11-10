"""v5 vs v4 comparison tables (pack summary per region, densities, crowding, hold-out localisation agreement, fluorescence
panel) -> data/derived/v5/v5_vs_v4.md (printed too). Run after run_v5_post.sh."""
import os, json, pandas as pd

V = (__import__("os").environ.get("VCELL_ROOT", ".") + ""); D = f"{V}/data/derived"; VAL = f"{D}/validation"; W = f"{D}/wc"
PANEL = ["LCK", "GOLGA2", "LMNB1", "TOMM20", "PGK1", "CD3E", "LCP1", "UBA1"]


def jl(f):
    return json.load(open(f)) if os.path.exists(f) else None


def main():
    L = ["# Pack v5 vs v4\n"]
    s4, s5 = jl(f"{W}/summary_v4.json"), jl(f"{W}/summary.json")
    if s4 and s5:
        L.append(f"Total: v4 {s4['placed_pg']} of {s4['expected_pg']} pg ({s4['mass_frac']:.3f}), {s4['instances']:,} instances; "
                 f"v5 {s5['placed_pg']} of {s5['expected_pg']} pg ({s5['mass_frac']:.3f}), {s5['instances']:,} instances.\n")
        L.append("| Region | v4 expected pg | v4 placed frac | v5 expected pg | v5 placed frac |\n|---|---|---|---|---|")
        for r in list(dict.fromkeys(list(s4["regions"]) + list(s5["regions"]))):
            a, b = s4["regions"].get(r, {}), s5["regions"].get(r, {})
            L.append(f"| {r} | {a.get('expected_pg', '')} | {a.get('frac', '')} | {b.get('expected_pg', '')} | {b.get('frac', '')} |")
    d4, d5 = jl(f"{VAL}/densities_v4.json"), jl(f"{VAL}/densities.json")
    if d4 and d5:
        L.append("\n## Densities (mg/ml, packed + pre-pass)\n\n| Region | v4 | v5 |\n|---|---|---|")
        a = {r["region"]: r["mg_per_ml"] for r in d4["rows"]}; b = {r["region"]: r["mg_per_ml"] for r in d5["rows"]}
        for r in dict.fromkeys(list(a) + list(b)): L.append(f"| {r} | {a.get(r, '')} | {b.get(r, '')} |")
        L.append(f"| nucleus/cytosol ratio | {d4['nuc_cyto_ratio']} | {d5['nuc_cyto_ratio']} |")
    c4, c5 = jl(f"{VAL}/crowding_v4.json"), jl(f"{VAL}/crowding.json")
    if c4 and c5:
        L.append("\n## Crowding: predicted D/D0 for a GFP-sized probe (r 2.3 nm)\n\n| Region | v4 | v5 |\n|---|---|---|")
        def g(c):
            out = {}
            for r in c["summary"]:
                if "D_2.3" in r: out[r.get("region")] = round(float(r["D_2.3"]), 3)
            return out
        a, b = g(c4), g(c5)
        for r in dict.fromkeys(list(a) + list(b)): L.append(f"| {r} | {a.get(r, '')} | {b.get(r, '')} |")
    h4, h5 = jl(f"{VAL}/loc_holdout_baseline.json"), jl(f"{VAL}/loc_holdout_v5.json")
    if h4 and h5:
        L.append("\n## Hold-out localisation agreement (agree = 1 - score; unweighted / mass-weighted)\n")
        L.append("| Set | Genes v4 / v5 | v4 intended | v5 intended | v4 placed | v5 placed | v4 top-hit (mass-w.) | v5 top-hit (mass-w.) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k in h4["sets"]:
            a, b = h4["sets"][k], h5["sets"].get(k)
            if not b: continue
            f = lambda x, m, q="agree": f"{x[m][q]['unweighted']:.3f} / {x[m][q]['mass_weighted']:.3f}"
            L.append(f"| {k} | {a['n_genes']} / {b['n_genes']} | {f(a, 'intended')} | {f(b, 'intended')} | {f(a, 'placed')} | {f(b, 'placed')} | "
                     f"{a['intended']['tophit']['mass_weighted']:.3f} | {b['intended']['tophit']['mass_weighted']:.3f} |")
    m4f, m5f = f"{VAL}/fluor_metrics_v4.tsv", f"{VAL}/fluor_metrics.tsv"
    if os.path.exists(m4f) and os.path.exists(m5f):
        m4 = pd.read_csv(m4f, sep="\t").set_index("protein"); m5 = pd.read_csv(m5f, sep="\t").set_index("protein")
        cols = ["mol_PM", "mol_cytosol", "mol_nucleus", "mol_NE", "mol_mito", "mol_golgi", "pm_cyto", "nuc_cyto", "edge_enrich"]
        L.append("\n## Fluorescence panel (model truth fractions and confocal-section metrics), v4 -> v5\n")
        L.append("| Protein | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
        for p in [x for x in m5.index if any(x.startswith(q) for q in PANEL)] or PANEL:
            if p not in m4.index or p not in m5.index: continue
            cells = []
            for c in cols:
                a, b = m4.at[p, c], m5.at[p, c]
                cells.append(f"{a:.3g} → {b:.3g}" if isinstance(a, float) or isinstance(b, float) else f"{a} → {b}")
            L.append(f"| {p} | " + " | ".join(cells) + " |")
    out = "\n".join(L) + "\n"
    os.makedirs(f"{D}/v5", exist_ok=True); open(f"{D}/v5/v5_vs_v4.md", "w").write(out); print(out)


if __name__ == "__main__":
    main()
