"""Build a composition inventory for a haploid S. cerevisiae cell (S288C, log phase, rich glucose).

Layer 1 of the Maritan et al. 2021-style recipe: copies per cell, compartment, size.
"""
import re, json
import numpy as np
import pandas as pd

from config import DATA as D
TOTAL_PROTEINS = 4.2e7  # Ho et al. 2018 unified total (molecules/cell)

# ---------------- geometry (literature-anchored; see report) ----------------
CELL_VOL = 42.0          # um^3, mean log-phase cell volume (Jorgensen 2002, as used by Ho 2018)
WALL_THICK = 0.11        # um, lower bound of 110-200 nm (Orlean 2012)
FRAC = dict(nucleus=0.08, vacuole=0.09, mitochondrion=0.02, lipid_droplet=0.005)
# nucleus 8%: two segmented cryo-FIB/SEM nuclei (8.0%, 9.7%; organelle_stats.json), between fluorescence (~7%,
#   Jorgensen 2007) and serial-section EM (10.5%, Yamaguchi 2011)
# vacuole 9% and lipid droplets 0.5%: medians of five segmented cells (vacuoles 6-13%, a lower bound; droplets
#   0.3-0.8%, depositor labels); SXT gave 7% and 1% (Uchida 2011)
# mitochondria raised 1% -> 2% (assumption: SXT under-segments thin tubules; proteome-implied density otherwise ~480 mg/mL)
NUCLEOLUS_OF_NUC = 0.20  # Uchida 2011
ENDOMEMBRANE_FRAC = 0.01 # assumption: Golgi+endosomes+peroxisomes (not segmented by SXT)
CER_COVER = 0.48         # fraction of PM covered by cortical ER (Wolf 2012)
N_NPC = 117              # Winey 1997 mean


def geometry():
    R = (3 * CELL_VOL / (4 * np.pi)) ** (1 / 3)
    Rp = R - WALL_THICK
    g = {"cell_radius_um": R, "protoplast_radius_um": Rp, "cell_volume_um3": CELL_VOL}
    v = {}
    v["cell_wall"] = CELL_VOL - 4 / 3 * np.pi * Rp ** 3
    nuc = FRAC["nucleus"] * CELL_VOL
    v["nucleolus"] = NUCLEOLUS_OF_NUC * nuc
    v["nucleoplasm"] = nuc - v["nucleolus"]
    v["vacuole"] = FRAC["vacuole"] * CELL_VOL
    v["mitochondrion"] = FRAC["mitochondrion"] * CELL_VOL
    v["lipid_droplet"] = FRAC["lipid_droplet"] * CELL_VOL
    v["golgi_endosome"] = ENDOMEMBRANE_FRAC * CELL_VOL * 0.8
    v["peroxisome"] = ENDOMEMBRANE_FRAC * CELL_VOL * 0.2
    pm_area = 4 * np.pi * Rp ** 2
    cer_area = CER_COVER * pm_area
    v["ER"] = cer_area * 0.024 + 4 * np.pi * ((nuc * 3 / 4 / np.pi) ** (1 / 3)) ** 2 * 0.03  # cER sheet 24 nm + NE lumen ~30 nm
    v["nuclear_envelope"] = 0.0
    v["plasma_membrane"] = 0.0
    v["cytosol"] = 4 / 3 * np.pi * Rp ** 3 - sum(v[k] for k in v if k not in ("cell_wall",))
    rn = (nuc * 3 / 4 / np.pi) ** (1 / 3)
    areas = {"plasma_membrane": pm_area, "nuclear_envelope": 4 * np.pi * rn ** 2 * 2,
             "ER": cer_area * 2, "vacuole": 4 * np.pi * (v["vacuole"] * 3 / 4 / np.pi) ** (2 / 3)}
    g.update(volumes_um3=v, areas_um2=areas, nucleus_radius_um=rn, n_npc=N_NPC,
             cer_cover=CER_COVER, wall_thickness_um=WALL_THICK)
    return g


# ---------------- localization parsing ----------------
RULES = [  # (regex on lowercase text, compartment) - first match per location token
    (r"cell wall|secreted|extracellular|periplasm|cell surface", "cell_wall"),
    (r"nucleolus", "nucleolus"),
    (r"nuclear pore|nucleus membrane|nucleus inner membrane|nucleus outer membrane|nuclear envelope|spindle pole", "nuclear_envelope"),
    # mitochondrial sub-compartments (inventory v3); the generic term is the fallback
    (r"mitochondri\w* inner membrane|inner mitochondrial membrane|mitochondrial respiratory chain", "mito_IM"),
    (r"mitochondri\w* outer membrane|outer mitochondrial membrane", "mito_OM"),
    (r"mitochondri\w* intermembrane space", "mito_IMS"),
    (r"mitochondri\w* matrix|mitochondrial nucleoid", "mito_matrix"),
    (r"mitochondri", "mitochondrion"),
    (r"lipid droplet|lipid particle", "lipid_droplet"),
    (r"peroxisom", "peroxisome"),
    (r"vacuol", "vacuole"),
    (r"golgi|endosome|vesicle|prevacuolar|multivesicular|clathrin|trans-golgi|autophag", "golgi_endosome"),
    (r"endoplasmic reticulum|microsome", "ER"),
    (r"cell membrane|plasma membrane|eisosome|cell cortex", "plasma_membrane"),
    (r"nucleus|nucleoplasm|chromosome|chromatin|telomere|kinetochore", "nucleoplasm"),
    (r"cytoplasm|cytosol|cytoskeleton|p-body|stress granule|bud|ribosome|polysome|actin|microtubule", "cytosol"),
]


# Large-scale organelle proteomics papers that over-assign abundant cytosolic proteins
# (Sickmann 2003, Reinders 2006 mitochondrial proteomes). Location tokens supported ONLY by these are dropped.
HT_ORGANELLE_PROTEOMICS = {"14576278", "16823961"}


HUH2003 = "14562095"  # Huh et al. 2003 genome-wide GFP localization (steady-state bulk localization)


def parse_uniprot_cc(s):
    """Return ordered list of compartments. If any token is supported by Huh et al. 2003 GFP data,
    only Huh-supported tokens are kept (bulk steady-state localization)."""
    if not isinstance(s, str):
        return []
    s = s.split("Note=")[0]
    s = s.replace("SUBCELLULAR LOCATION:", "")
    out, huh = [], []
    for m in re.finditer(r"([^.;{}]+?)\s*(\{[^}]*\})?\s*[.;]", s):
        t, ev = m.group(1).strip().lower(), (m.group(2) or "")
        if not t:
            continue
        pm = set(re.findall(r"PubMed:(\d+)", ev))
        if pm and pm <= HT_ORGANELLE_PROTEOMICS and "ECO:0000250" not in ev:
            continue
        for rx, c in RULES:
            if re.search(rx, t):
                out.append(c)
                if HUH2003 in pm:
                    huh.append(c)
                break
    allc = out
    out = list(dict.fromkeys(huh if huh else out))
    return refine_mito(out, allc)


MITO_SUB = ("mito_IM", "mito_OM", "mito_IMS", "mito_matrix")
MITO_SECONDARY = 0.1   # weight factor of a secondary mitochondrial location behind a cytosolic/nuclear primary one


def refine_mito(comps, extra):
    """replace the generic 'mitochondrion' by the sub-compartments named in extra (e.g. a Huh 2003 GFP call of
    'mitochondrion' while UniProt also says 'mitochondrion inner membrane')"""
    if "mitochondrion" not in comps:
        return comps
    sub = [c for c in dict.fromkeys(extra) if c in MITO_SUB]
    if not sub:
        return comps
    out = []
    for c in comps:
        out += sub if c == "mitochondrion" else [c]
    return list(dict.fromkeys(out))


SOLUBLE = ("cytosol", "nucleoplasm", "nucleolus")


def tm_go_fallback(s):
    """membrane-bearing compartments named in GO CC (with 'cell periphery' read as plasma membrane), in GO order"""
    if not isinstance(s, str):
        return []
    out = []
    for t in s.lower().split(";"):
        t = re.sub(r"\[go:\d+\]", "", t).strip()
        if re.search(r"cell periphery", t):
            out.append("plasma_membrane"); continue
        for rx, c in RULES:
            if re.search(rx, t):
                if c not in SOLUBLE and c != "cell_wall":
                    out.append(c)
                break
    return list(dict.fromkeys(out))


def parse_go_cc(s):
    if not isinstance(s, str):
        return []
    out = []
    for t in s.lower().split(";"):
        t = re.sub(r"\[go:\d+\]", "", t).strip()
        if re.search(r"complex$", t) and not re.search(r"pore|ribosom|proteasome", t):
            continue
        for rx, c in RULES:
            if re.search(rx, t):
                out.append(c)
                break
    out = list(dict.fromkeys(out))
    # GO CC (no evidence filter available here) inherits high-throughput organelle calls for abundant
    # cytosolic enzymes; if cytosol is annotated, keep only cyto/nuclear terms.
    if "cytosol" in out:
        out = [c for c in out if c in ("cytosol", "nucleoplasm", "nucleolus")]
    return out


def weights(comps, vol):
    """Rank weights (1, 1/2, 1/4, ...) by annotation order (first-listed = primary); the cytosol and
    nucleoplasm shares are pooled and re-split by volume (equal concentration across the nuclear pore)."""
    if not comps:
        return {"cytosol": 1.0}
    raw = {c: 0.5 ** i for i, c in enumerate(comps)}
    # dual-localised cytosolic/nuclear proteins with a secondary mitochondrial location (e.g. SOD1, ADK1 "Cytoplasm.
    # Mitochondrion intermembrane space", typically noted as a small percent): the mitochondrial pool is minor
    if comps[0] in ("cytosol", "nucleoplasm"):
        for c in comps[1:]:
            if c in MITO_SUB or c == "mitochondrion":
                raw[c] *= MITO_SECONDARY
    cn = [c for c in comps if c in ("cytosol", "nucleoplasm")]
    if cn:
        pooled = sum(raw[c] for c in cn); tot = sum(vol[c] for c in cn)
        for c in cn:
            raw[c] = pooled * vol[c] / tot
    z = sum(raw.values())
    return {c: v / z for c, v in raw.items()}


def subunit_oligomer(s):
    if not isinstance(s, str):
        return 1
    s = s.lower()
    m = {"homodimer": 2, "homotrimer": 3, "homotetramer": 4, "homopentamer": 5, "homohexamer": 6,
         "homoheptamer": 7, "homooctamer": 8, "homodecamer": 10, "homododecamer": 12}
    for k, n in m.items():
        if re.search(r"\b" + k + r"\b", s.split(".")[0] + "." + s[:200]):
            return n
    return 1


def build():
    g = geometry()
    vol = dict(g["volumes_um3"])
    pax = pd.read_csv(f"{D}/paxdb_integrated.txt", sep="\t", comment="#",
                      names=["gene", "string_id", "ppm"])
    pax["orf"] = pax.string_id.str.replace("4932.", "", regex=False)
    pax = pax.groupby("orf", as_index=False).ppm.sum()
    u = pd.read_csv(f"{D}/uniprot_yeast.tsv", sep="\t")
    u.columns = ["uniprot", "reviewed", "orf", "gene", "length", "mass_da", "cc", "go_cc", "tm", "subunit", "name"]
    u["orf"] = u.orf.astype(str).str.split(r"[ ;]").str[0]
    u = u.drop_duplicates("orf")
    df = u.merge(pax, on="orf", how="left")
    matched_ppm = df.ppm.sum()
    df["ppm"] = df.ppm.fillna(0.0)
    # rescale so that total = Ho et al. 2018 total proteome size
    df["copies_per_cell"] = df.ppm / df.ppm.sum() * TOTAL_PROTEINS
    df["n_tm"] = df.tm.fillna("").str.count("TRANSMEM")
    # integral (multi-pass) proteins without TRANSMEM ranges are outer-membrane beta-barrels (TOM40, SAM50, MDM10,
    # MDM34): flag them as membrane proteins; membrane_orient uses their beta-strands as membrane segments
    barrel = df.cc.fillna("").str.contains("Multi-pass membrane protein") & (df.n_tm == 0)
    df.loc[barrel, "n_tm"] = 1
    locs, src = [], []
    for cc, go, ntm in zip(df.cc, df.go_cc, df.n_tm):
        a = parse_uniprot_cc(cc)
        if "mitochondrion" in a:          # UniProt generic: take the sub-compartment from GO when it names one
            gc = parse_go_cc(go)
            if sum(c in MITO_SUB for c in gc) == 1:
                a = refine_mito(a, gc)
        s_ = "UniProt" if a else None
        if not a:
            a = parse_go_cc(go); s_ = "GO" if a else "default_cytosol"
        if ntm > 0 and all(c in SOLUBLE for c in a):
            # TM protein with only soluble annotations (UniProt often just says "Membrane"): membrane locations from GO
            gm = tm_go_fallback(go)
            if gm:
                a, s_ = gm, s_ + "+GO_membrane"
        locs.append(a); src.append(s_)
    df["compartments"] = ["|".join(x) if x else "cytosol" for x in locs]
    df["loc_source"] = src
    df["oligomer_n"] = df.subunit.map(subunit_oligomer)
    w = [weights(l, vol) for l in locs]
    df["comp_weights"] = [json.dumps({k: round(v, 4) for k, v in x.items()}) for x in w]
    df["gene"] = df.gene.fillna(df.orf)
    keep = ["orf", "gene", "uniprot", "name", "length", "mass_da", "copies_per_cell", "ppm",
            "compartments", "comp_weights", "loc_source", "n_tm", "oligomer_n", "subunit"]
    df = df[keep].sort_values("copies_per_cell", ascending=False).reset_index(drop=True)
    df.attrs["paxdb_ppm_matched"] = matched_ppm
    return df, g


def per_compartment(df, g):
    rows = {}
    for cw, n, m in zip(df.comp_weights, df.copies_per_cell, df.mass_da):
        for c, w in json.loads(cw).items():
            r = rows.setdefault(c, [0.0, 0.0])
            r[0] += n * w; r[1] += n * w * m
    out = pd.DataFrame([(c, a, b) for c, (a, b) in rows.items()], columns=["compartment", "protein_copies", "protein_mass_da"])
    out["volume_um3"] = out.compartment.map(g["volumes_um3"])
    out["area_um2"] = out.compartment.map(g["areas_um2"])
    out["protein_mg_per_ml"] = out.protein_mass_da / 6.022e23 / (out.volume_um3 * 1e-15)  # g/L == mg/mL
    return out.sort_values("protein_copies", ascending=False)
