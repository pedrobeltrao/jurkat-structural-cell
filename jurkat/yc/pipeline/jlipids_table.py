"""Build data/lipids_jurkat.json (v1): per-membrane, per-leaflet lipid compositions for the Jurkat model.

Sources
  - Pogozheva et al. 2022 J Chem Inf Model 62:1036, SI Tables S1, S4, S6, S8-S11 (all-atom CHARMM compositions with
    per-species area per lipid) and Table S20 (D_P-P and hydrophobic thickness); parsed to
    data/literature/lipids/pogozheva2022_mammalian_tables.json
  - Zech et al. 2009 EMBO J 28:466, supplementary figures (Jurkat plasma-membrane TfR fraction): within-class species
    distribution of PC/SM and PS (read off the bar charts, rounded) -> replaces the PC, PS and SM species mix of the
    Pogozheva PM while keeping its class totals and leaflet asymmetry (Zech gives no class-level mol% in the SI)
Leaflets: 'cyt' = cytosolic (N) side, 'exo' = exoplasmic (extracellular / lumen / IMS). Mito IM: 'cyt' = matrix (N side).
Membranes: PM <- PMm (+ Zech species); ER, NE <- ERm; GOLGI <- GOLm; ENDO <- ENDm (LYSm kept as alternative "LYSO");
MITO_OM <- MOM (outer leaflet = cytosol); MITO_IM <- MIM (outer leaflet = IMS, inner = matrix).
Lipid names are CHARMM-GUI names (all-atom); the Martini mapping is done at MD export."""
import json, os
import numpy as np

V = (__import__("os").environ.get("VCELL_ROOT", ".") + "")
P = json.load(open(f"{V}/data/literature/lipids/pogozheva2022_mammalian_tables.json"))
HEAD = {"PC": "PC", "PE": "PE", "PS": "PS", "PI": "PI", "PA": "PA", "SM": "SM", "Cholesterol": "CHOL", "CL": "CL",
        "GlcCer": "GlcCer", "BMP": "BMP"}
# Zech 2009 Jurkat PM (TfR) within-class species fractions mapped to CHARMM species
ZECH_PC = {"POPC": 0.41, "DPPC": 0.18, "DOPC": 0.11, "PLPC": 0.10, "SAPC": 0.10, "SDPC": 0.10}     # db1, db0, 36:2 (x2), db>=5
ZECH_PS = {"SOPS": 0.45, "SDPS": 0.20, "POPS": 0.13, "DOPS": 0.08, "SAPS": 0.14}
ZECH_SM = {"PSM": 0.80, "NSM": 0.20}                                                              # SM 34:1;2 dominant
APL_DEFAULT = {"PC": 0.62, "PS": 0.60, "SM": 0.55}                                              # nm2 for added species


def head(species):
    s = species.split("(")[0]
    return HEAD.get(s, s)


def leaflets(tab, outer_is):
    rows = tab["lipids"]
    out = {"cyt": {}, "exo": {}}; apl = {"cyt": {}, "exo": {}}
    for r in rows:
        for lf, n, a in (("outer", r["n_outer"], r["apl_outer"]), ("inner", r["n_inner"], r["apl_inner"])):
            side = outer_is if lf == "outer" else ("cyt" if outer_is == "exo" else "exo")
            if n > 0:
                out[side][r["name"]] = out[side].get(r["name"], 0) + n
                apl[side][r["name"]] = a / 100.0                                                   # A2 -> nm2
    heads = {r["name"]: head(r["species"]) for r in rows}
    return out, apl, heads


def norm(d):
    t = sum(d.values()); return {k: round(v / t, 4) for k, v in d.items()}


def rescale_class(comp, apl, heads, cls, new_species):
    tot = sum(v for k, v in comp.items() if heads.get(k) == cls)
    if tot == 0:
        return
    for k in [k for k in comp if heads.get(k) == cls]:
        comp.pop(k)
    for sp, f in new_species.items():
        comp[sp] = comp.get(sp, 0) + tot * f
        heads[sp] = cls
        apl.setdefault(sp, APL_DEFAULT[cls])


def main():
    mem_src = {"PM": ("PMm", "exo"), "ER": ("ERm", "exo"), "GOLGI": ("GOLm", "exo"), "ENDO": ("ENDm", "exo"),
               "LYSO": ("LYSm", "exo"), "MITO_OM": ("MOM", "cyt"), "MITO_IM": ("MIM", "exo")}
    out = {"version": "v1 (2026-11-08)", "leaflet_convention": "cyt = cytosolic/N side (matrix for MITO_IM); exo = extracellular/lumen/IMS",
           "names": "CHARMM-GUI lipid names; Martini mapping at MD export", "membranes": {}, "heads": {}}
    heads_all = {}
    for m, (src, outer_is) in mem_src.items():
        comp, apl, heads = leaflets(P[src], outer_is)
        if m == "PM":
            for lf in ("cyt", "exo"):
                for cls, sp in (("PC", ZECH_PC), ("PS", ZECH_PS), ("SM", ZECH_SM)):
                    rescale_class(comp[lf], apl[lf], heads, cls, sp)
        heads_all.update(heads)
        e = {"source": f"Pogozheva 2022 SI ({src})" + ("; PC/PS/SM species mix from Zech 2009 Jurkat PM (TfR)" if m == "PM" else ""),
             "dpp_nm": P[src]["dpp_nm"], "hydrophobic_nm": P[src]["hydrophobic_nm"], "leaflets": {}}
        for lf in ("cyt", "exo"):
            f = norm(comp[lf])
            mapl = float(sum(f[k] * apl[lf].get(k, 0.6) for k in f))
            cls = {}
            for k, v in f.items():
                cls[heads[k]] = round(cls.get(heads[k], 0) + v, 4)
            e["leaflets"][lf] = {"species": f, "classes": dict(sorted(cls.items(), key=lambda kv: -kv[1])), "mean_apl_nm2": round(mapl, 4),
                                 "apl_nm2": {k: round(apl[lf].get(k, 0.6), 4) for k in f}}
        out["membranes"][m] = e
    out["membranes"]["NE"] = {"same_as": "ER"}
    out["heads"] = heads_all
    fn = f"{os.path.dirname(os.path.abspath(__file__))}/data/lipids_jurkat.json"
    json.dump(out, open(fn, "w"), indent=1)
    for m, e in out["membranes"].items():
        if "leaflets" in e:
            print(f"{m:8s} DPP {e['dpp_nm']:.2f} nm | cyt {e['leaflets']['cyt']['classes']} | exo {e['leaflets']['exo']['classes']}")


if __name__ == "__main__":
    main()
